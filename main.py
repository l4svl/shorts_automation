from Config import TELEGRAM_TOKEN, TIKTOK_CREDENTIALS, TWITCH_CLIENT_ID, TWITCH_CLIENT_SECRET, YOUTUBE_CLIENT_SECRETS, \
    YOUTUBE_API_KEY
import asyncio
import os
import json
import random
import sqlite3
import subprocess
from datetime import datetime, timezone, timedelta
import cv2
from easyocr import Reader
from aiogram import Bot, Dispatcher
from aiogram.types import Message, CallbackQuery, FSInputFile
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.filters import Command, StateFilter
import requests
import ffmpeg
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from uuid import uuid4
import logging
import yt_dlp
from playwright.async_api import async_playwright, Error
import time
import shutil
import pytz
import re
# Настройка логирования
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# YouTube фильтры
ALLOWED_YT_CATEGORIES = {"19", "20"}  # Travel & Events, Gaming, People & Blogs
ALLOWED_YT_CATEGORY_NAMES = {
    "19": "Travel & Events",
    "20": "Gaming",
    "22": "People & Blogs",
}
ALLOWED_YT_REGIONS = ["RU","UA"]

# Адаптер для datetime в SQLite
def adapt_datetime(dt):
    return dt.isoformat()
def _make_hashtag(text: str) -> str:
    # Простейшая нормализация для хештегов: пробелы -> _, убираем запрещённые символы
    cleaned = re.sub(r'[^0-9A-Za-zА-Яа-яёЁ_]+', '', text.replace(' ', '_'))
    return cleaned[:60] if cleaned else 'video'

def _build_tiktok_description(title: str, channel_title: str, channel_id: str, part_num: int = None) -> str:
    author_tag = _make_hashtag(channel_title)
    title_tag = _make_hashtag(title)
    source_url = f"https://www.youtube.com/channel/{channel_id}"
    # Хештеги по твоему ТЗ (сохраняю точное написание “#рекоммендации” как в запросе)
    hashtags = f"#ютуб #{author_tag} #нарезка #тренды #рекоммендации #{title_tag}"
    title_str = f"{title} Часть {part_num}" if part_num is not None else title
    return f"{title_str}\nSource: {source_url}\n{hashtags}"

def _fetch_sponsorblock_segments(video_id: str, categories_to_skip=None):
    if categories_to_skip is None:
        categories_to_skip = {"sponsor", "selfpromo"}  # фокус на рекламе
    url = f"https://sponsor.ajay.app/api/skipSegments?videoID={video_id}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        segments = []
        for item in data:
            cat = item.get("category")
            seg = item.get("segment", [])
            if cat in categories_to_skip and len(seg) == 2:
                start, end = float(seg[0]), float(seg[1])
                if end > start:
                    segments.append((start, end))
        segments.sort(key=lambda x: x[0])
        return segments
    except Exception as e:
        logger.warning(f"SponsorBlock не доступен или ошибка ответа: {e}")
        return []

def _calc_non_ad_ranges(total_duration: float, ad_segments: list[tuple[float, float]]) -> list[tuple[float, float]]:
    # Строим не-рекламные интервалы на основе рекламных
    if total_duration <= 0:
        return []
    merged = []
    for s,e in sorted(ad_segments, key=lambda x: x[0]):
        if not merged or s > merged[-1][1]:
            merged.append([max(0.0, s), min(total_duration, e)])
        else:
            merged[-1][1] = max(merged[-1][1], min(total_duration, e))
    non_ad = []
    cursor = 0.0
    for s,e in merged:
        if s > cursor:
            non_ad.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < total_duration:
        non_ad.append((cursor, total_duration))
    # Фильтруем слишком короткие отрезки (например < 2с)
    non_ad = [(s,e) for (s,e) in non_ad if e - s >= 2.0]
    return non_ad

def _ffmpeg_cut_out_segments(input_path: str, output_path: str, keep_ranges: list[tuple[float, float]]) -> bool:
    # Строим filter_complex: trim/atrim для каждого диапазона, потом concat
    try:
        if not keep_ranges:
            logger.warning("Нет не-рекламных диапазонов — пропускаем")
            return False
        parts_v = []
        parts_a = []
        filter_cmds = []
        for idx, (s,e) in enumerate(keep_ranges):
            filter_cmds.append(f"[0:v]trim=start={s}:end={e},setpts=PTS-STARTPTS[v{idx}]")
            filter_cmds.append(f"[0:a]atrim=start={s}:end={e},asetpts=PTS-STARTPTS[a{idx}]")
            parts_v.append(f"[v{idx}]")
            parts_a.append(f"[a{idx}]")
        v_in = "".join(parts_v)
        a_in = "".join(parts_a)
        concat_cmd = f"{v_in}{a_in}concat=n={len(keep_ranges)}:v=1:a=1[vout][aout]"
        filter_complex = ";".join(filter_cmds + [concat_cmd])
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-filter_complex", filter_complex,
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "22",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            output_path
        ]
        logger.info(f"FFmpeg (remove ads): {' '.join(cmd)}")
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            logger.error(f"FFmpeg remove ads error:\n{proc.stderr}")
            return False
        return True
    except Exception as e:
        logger.error(f"Ошибка сборки команды remove ads: {e}")
        return False

def _ffmpeg_split_to_chunks(input_path: str, out_dir: str, base: str, chunk_seconds: int = 180) -> list[str]:
    # Нарезаем на 3-мин куски; перекодируем, чтобы дальше гарантированно собрать вертикаль
    try:
        os.makedirs(out_dir, exist_ok=True)
        pattern = os.path.join(out_dir, f"{base}_part_%03d.mp4")
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-c:v", "libx264", "-preset", "medium", "-crf", "21",
            "-c:a", "aac", "-b:a", "192k",
            "-f", "segment", "-segment_time", str(chunk_seconds),
            "-reset_timestamps", "1",
            pattern
        ]
        logger.info(f"FFmpeg split: {' '.join(cmd)}")
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            logger.error(f"FFmpeg split error:\n{proc.stderr}")
            return []
        # Собираем список файлов
        chunks = []
        for name in sorted(os.listdir(out_dir)):
            if name.startswith(f"{base}_part_") and name.endswith(".mp4"):
                chunks.append(os.path.join(out_dir, name))
        return chunks
    except Exception as e:
        logger.error(f"Ошибка нарезки на куски: {e}")
        return []


sqlite3.register_adapter(datetime, adapt_datetime)
async def publish_all_tiktok_drafts():
    print("Начинаю публикацию всех черновиков на TikTok. Пожалуйста, подождите...")
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(user_agent=random.choice(USER_AGENTS))

            # Загрузка и валидация куки
            cookies_path = 'tiktok_cookies.json'
            if not os.path.exists(cookies_path):
                raise ValueError("Файл tiktok_cookies.json не найден")
            with open(cookies_path, 'r') as f:
                cookies = json.load(f)
                valid_cookies = [
                    {**cookie, 'sameSite': cookie.get('sameSite', 'Lax' if cookie.get('secure', False) else 'None')}
                    for cookie in cookies if all(key in cookie for key in ['name', 'value', 'domain'])
                ]
                if not valid_cookies:
                    raise ValueError("Нет валидных cookies в tiktok_cookies.json")
                await context.add_cookies(valid_cookies)

            page = await context.new_page()
            await page.goto("https://www.tiktok.com/tiktokstudio/content", timeout=60000)

            # Проверка авторизации
            if await page.query_selector("text=Войти"):
                raise RuntimeError("Сессия не авторизована. Обновите куки.")

            # Переход в черновики
            draft_button_xpath = '//*[@id="root"]/div/div/div[2]/div[1]/div/div/div[2]/div[1]/div[2]/span/div'
            await page.wait_for_selector(f'xpath={draft_button_xpath}', state="visible", timeout=60000)
            await page.click(f'xpath={draft_button_xpath}')

            # Ожидание страницы черновиков
            await page.wait_for_url("https://www.tiktok.com/tiktokstudio/content?tab=draft", timeout=60000)
            await page.wait_for_timeout(5000)

            # Поиск реальных контейнеров черновиков
            drafts_xpath = '//*[@id="root"]/div/div/div[2]/div[2]/div/div/div/div[2]/div[2]/div[*]/div'
            draft_elements = await page.query_selector_all(f'xpath={drafts_xpath}')
            total_drafts = 0
            for element in draft_elements:
                text = await element.inner_text()
                if text and any(char.isdigit() for char in text):  # Проверяем наличие цифр
                    total_drafts += 1
            if total_drafts == 0:
                print("Нет черновиков для публикации.")
                await browser.close()
                return

            print(f"Найдено {total_drafts} черновиков. Начинаю публикацию...")

            published_count = 0
            for i in range(total_drafts):
                try:
                    print(f"Публикация черновика {i+1}/{total_drafts}...")

                    # Клик на "Редактировать"
                    edit_button_xpath = '//*[@id="root"]/div/div/div[2]/div[2]/div/div/div/div[2]/div[2]/div[1]/div/div[3]/div/button[1]'
                    await page.wait_for_selector(f'xpath={edit_button_xpath}', state="visible", timeout=30000)
                    await page.click(f'xpath={edit_button_xpath}')
                    await page.wait_for_timeout(3000)

                    # Клик "Опубликовать"
                    publish_xpath = '//*[@id="root"]/div/div/div[2]/div[2]/div/div/div/div[5]/div/button[1]'
                    await page.wait_for_selector(f'xpath={publish_xpath}', state="visible", timeout=60000)
                    publish_button = await page.query_selector(f'xpath={publish_xpath}')
                    max_wait = 10
                    start_time = time.time()
                    while time.time() - start_time < max_wait:
                        if await publish_button.is_enabled():
                            break
                        await page.wait_for_timeout(1000)
                    if not await publish_button.is_enabled():
                        print(f"Кнопка 'Опубликовать' не активировалась для черновика {i+1}. Пропускаю.")
                        await page.goto("https://www.tiktok.com/tiktokstudio/content?tab=draft", timeout=60000)
                        await page.wait_for_timeout(5000)
                        continue

                    await publish_button.click()
                    await page.wait_for_timeout(5000)

                    published_count += 1
                    await page.goto("https://www.tiktok.com/tiktokstudio/content?tab=draft", timeout=60000)
                    await page.wait_for_timeout(5000)

                except Exception as inner_e:
                    print(f"Ошибка при публикации черновика {i+1}: {inner_e}")
                    await page.goto("https://www.tiktok.com/tiktokstudio/content?tab=draft", timeout=60000)
                    await page.wait_for_timeout(5000)

            # Обновление куки
            cookies = await context.cookies()
            with open(cookies_path, 'w') as f:
                json.dump(cookies, f)

            await browser.close()
            print(f"Успешно опубликовано {published_count} из {total_drafts} черновиков!")

    except Exception as e:
        if 'page' in locals():
            await browser.close()
        print(f"Ошибка при публикации: {str(e)}.")

# Инициализация бота
bot = Bot(token=TELEGRAM_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot=bot, storage=storage)

# Глобальные переменные для автогенерации
last_autogeneration_time = None
autogeneration_task = None
autogeneration_running = False


# Состояния
class VideoGeneration(StatesGroup):
    platform = State()
    clip_count = State()
    categories = State()
    language = State()
    autogeneration = State()


# Настройка базы данных
def init_db():
    try:
        conn = sqlite3.connect('clips.db')
        c = conn.cursor()
        # Проверяем, существует ли старая таблица
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='clips'")
        if c.fetchone():
            # Переименовываем старую таблицу
            c.execute("ALTER TABLE clips RENAME TO clips_old")
            # Создаём новую таблицу с составным ключом
            c.execute('''CREATE TABLE clips
                         (
                             clip_id TEXT,
                             channel_name TEXT,
                             used_for TEXT,
                             created_at TIMESTAMP,
                             PRIMARY KEY (clip_id, used_for)
                         )''')
            # Переносим данные из старой таблицы
            c.execute('''INSERT OR IGNORE INTO clips (clip_id, channel_name, used_for, created_at)
                         SELECT clip_id, channel_name, used_for, created_at FROM clips_old''')
            # Удаляем старую таблицу
            c.execute("DROP TABLE clips_old")
        else:
            # Если таблицы нет, создаём новую
            c.execute('''CREATE TABLE clips
                         (
                             clip_id TEXT,
                             channel_name TEXT,
                             used_for TEXT,
                             created_at TIMESTAMP,
                             PRIMARY KEY (clip_id, used_for)
                         )''')
        # Создание остальных таблиц (без изменений)
        c.execute('''CREATE TABLE IF NOT EXISTS videos
                     (
                         video_id TEXT PRIMARY KEY,
                         file_path TEXT,
                         created_at TIMESTAMP
                     )''')
        c.execute('''CREATE TABLE IF NOT EXISTS video_metadata
                     (
                         video_id TEXT PRIMARY KEY,
                         processed_clips TEXT,
                         categories_data TEXT,
                         selected_categories TEXT,
                         created_at TIMESTAMP
                     )''')
        c.execute('''CREATE TABLE IF NOT EXISTS stats
                     (
                         platform TEXT,
                         clip_count INTEGER,
                         created_at TIMESTAMP
                     )''')
        c.execute('''CREATE TABLE IF NOT EXISTS used_youtube_videos
                            (
                                id TEXT PRIMARY KEY,
                                title TEXT,
                                channel_id TEXT,
                                channel_title TEXT,
                                created_at TIMESTAMP
                            )''')
        c.execute('''CREATE TABLE IF NOT EXISTS autogeneration_state
                     (
                         id INTEGER PRIMARY KEY,
                         running BOOLEAN,
                         last_generation TIMESTAMP,
                         platform TEXT,
                         selected_categories TEXT,
                         language TEXT,
                         clip_count INTEGER,
                         chat_id INTEGER
                     )''')
        conn.commit()
        logger.info("Все таблицы успешно созданы или обновлены")
    except sqlite3.Error as e:
        logger.error(f"Ошибка базы данных при создании таблиц: {e}")
        raise
    finally:
        conn.close()

def save_autogeneration_state(running, last_generation, platform, selected_categories, language, clip_count, chat_id):
    try:
        conn = sqlite3.connect('clips.db')
        c = conn.cursor()
        c.execute('''INSERT OR REPLACE INTO autogeneration_state
                     (id, running, last_generation, platform, selected_categories, language, clip_count, chat_id)
                     VALUES (1, ?, ?, ?, ?, ?, ?, ?)''',
                  (running, last_generation, platform, json.dumps(selected_categories), language, clip_count, chat_id))
        conn.commit()
        logger.info(f"Состояние автогенерации сохранено: running={running}, last_generation={last_generation}, platform={platform}, categories={selected_categories}, language={language}, clip_count={clip_count}, chat_id={chat_id}")
    except sqlite3.Error as e:
        logger.error(f"Ошибка сохранения состояния автогенерации: {e}")
    finally:
        conn.close()

def load_autogeneration_state():
    try:
        conn = sqlite3.connect('clips.db')
        c = conn.cursor()
        c.execute('SELECT running, last_generation, platform, selected_categories, language, clip_count, chat_id FROM autogeneration_state WHERE id=1')
        result = c.fetchone()
        if result:
            return {
                'running': bool(result[0]),
                'last_generation': datetime.fromisoformat(result[1]) if result[1] else None,
                'platform': result[2],
                'selected_categories': json.loads(result[3]) if result[3] else [],
                'language': result[4],
                'clip_count': result[5],
                'chat_id': result[6]
            }
        return None
    except sqlite3.Error as e:
        logger.error(f"Ошибка загрузки состояния автогенерации: {e}")
        return None
    finally:
        conn.close()
# Инициализация EasyOCR
reader = Reader(['en', 'ru'], gpu=True)


# Google OAuth для YouTube
def get_youtube_credentials():
    creds = None
    token_path = 'token.json'
    scopes = ['https://www.googleapis.com/auth/youtube.upload']

    if os.path.exists(token_path):
        try:
            creds = Credentials.from_authorized_user_file(token_path, scopes)
        except Exception as e:
            logger.error(f"Ошибка загрузки токена из {token_path}: {e}")

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                logger.error(f"Ошибка обновления токена: {e}")
                creds = None
        if not creds:
            try:
                flow = InstalledAppFlow.from_client_secrets_file(YOUTUBE_CLIENT_SECRETS, scopes)
                creds = flow.run_local_server(port=0)
            except Exception as e:
                logger.error(f"Ошибка авторизации OAuth: {e}")
                return None

        try:
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
            logger.info(f"Токен сохранен в {token_path}")
        except Exception as e:
            logger.error(f"Ошибка сохранения токена в {token_path}: {e}")

    return creds

def safe_remove(file_path):
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info(f"Удален файл: {file_path}")
        except Exception as e:
            logger.warning(f"Не удалось удалить файл {file_path}: {e}")
# Twitch API
async def get_twitch_token():
    try:
        url = "https://id.twitch.tv/oauth2/token"
        params = {
            "client_id": TWITCH_CLIENT_ID,
            "client_secret": TWITCH_CLIENT_SECRET,
            "grant_type": "client_credentials"
        }
        response = requests.post(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json().get("access_token")
    except requests.RequestException as e:
        logger.error(f"Ошибка получения токена Twitch: {e}")
        return None


async def get_top_categories(limit=10):
    token = await get_twitch_token()
    if not token:
        return []
    try:
        url = "https://api.twitch.tv/helix/games/top"
        headers = {
            "Client-ID": TWITCH_CLIENT_ID,
            "Authorization": f"Bearer {token}"
        }
        params = {"first": limit}
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        dynamic_categories = [(game["name"], game["id"]) for game in response.json().get("data", [])]
        fixed_categories = [("Dota 2", "29595"), ("Counter-Strike", "32399")]
        categories = fixed_categories + [cat for cat in dynamic_categories if
                                         cat[1] not in [fixed[1] for fixed in fixed_categories]]
        return categories[:limit + len(fixed_categories)]
    except requests.RequestException as e:
        logger.error(f"Ошибка получения категорий Twitch: {e}")
        return []


async def get_twitch_clips(category, language, count, max_attempts=5, search_depth_days=30):
    token = await get_twitch_token()
    if not token:
        return []
    try:
        conn = sqlite3.connect('clips.db')
        c = conn.cursor()
        url = "https://api.twitch.tv/helix/clips"
        headers = {
            "Client-ID": TWITCH_CLIENT_ID,
            "Authorization": f"Bearer {token}"
        }
        now = datetime.now(timezone.utc)
        month_ago = now - timedelta(days=search_depth_days)
        started_at = month_ago.isoformat().split('.')[0] + "Z"
        ended_at = now.isoformat().split('.')[0] + "Z"
        filtered_clips = []
        seen_clip_ids = set()
        attempt = 0
        russian_channels = {
            'lasthorizon_', 'migal', 'strogoat', 'bazimeh', 'buster', 'anarabdullaev',
            'bratishkinoff', 'Zubarefff', 'Nix', 'StarLadder1', 'Dota2RuHub', 'Recrent', 'evelone2004',
            'justhatemeeecat0_0', 'dota2_paragon_ru', 'betboom_ru', 'ramzes', 'relog_pari_dota_ru',
            'just_ns', 'yanewhykawaii', 'Grusti_ne', 'lowly_ya', 'IvanZolo578', 'rostislav_999', 'ALOHADANCETV',
            'dyrachyo'
        }
        while len(filtered_clips) < count and attempt < max_attempts:
            cursor = None
            clips_per_request = min(100, count * 2)
            attempt += 1
            logger.info(
                f"Попытка {attempt} поиска клипов для категории {category}, язык: {language}, требуется: {count}, "
                f"глубина поиска: {search_depth_days} дней"
            )
            while len(filtered_clips) < count:
                params = {
                    "game_id": category,
                    "first": clips_per_request,
                    "started_at": started_at,
                    "ended_at": ended_at,
                }
                if language and language != "any":
                    params["language"] = language.upper() if language == "ru" else language
                    if language == "ru" and params["language"] not in ["RU", "other"]:
                        params["language"] = "other"
                if cursor:
                    params["after"] = cursor
                response = requests.get(url, headers=headers, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                clips = data.get("data", [])
                logger.info(f"Получено клипов для категории {category} на попытке {attempt}: {len(clips)}")
                for clip in clips:
                    clip_id = clip.get('id')
                    if not clip_id:
                        continue
                    # Проверяем, использован ли клип для текущей платформы
                    c.execute("SELECT clip_id FROM clips WHERE clip_id=? AND used_for=?", (clip_id, params.get('platform', 'youtube')))
                    if c.fetchone():
                        logger.info(f"Клип {clip_id} уже использован для платформы {params.get('platform', 'youtube')}, пропускаем")
                        continue
                    if clip_id in seen_clip_ids:
                        continue
                    if language and language != "any":
                        if language == "ru":
                            if (clip.get('broadcaster_name', '').lower() in russian_channels or
                                    clip.get('language', '').lower() == "ru"):
                                filtered_clips.append(clip)
                                seen_clip_ids.add(clip_id)
                                logger.info(f"Выбран клип {clip_id} для категории {category}, язык: {language}")
                        else:
                            if (clip.get('language', '').lower() == language.lower() or
                                    not clip.get('language')):
                                filtered_clips.append(clip)
                                seen_clip_ids.add(clip_id)
                                logger.info(f"Выбран клип {clip_id} для категории {category}, язык: {language}")
                    else:
                        filtered_clips.append(clip)
                        seen_clip_ids.add(clip_id)
                        logger.info(f"Выбран клип {clip_id} для категории {category}, язык: {language}")
                cursor = data.get("pagination", {}).get("cursor")
                if not cursor or not clips:
                    break
            logger.info(f"Итого найдено подходящих клипов после попытки {attempt}: {len(filtered_clips)}")
            if len(filtered_clips) < count and attempt < max_attempts:
                search_depth_days += 30
                month_ago = now - timedelta(days=search_depth_days)
                started_at = month_ago.isoformat().split('.')[0] + "Z"
        conn.close()
        return filtered_clips[:count]
    except requests.RequestException as e:
        logger.error(f"Ошибка получения клипов Twitch: {e}")
        conn.close()
        return []

async def download_clip_with_ytdlp(clip_url, clip_id):
    try:
        ydl_opts = {
            'outtmpl': os.path.join(os.getcwd(), f'temp_{clip_id}.mp4'),
            'quiet': True,
            'no_warnings': True,
            'format': 'bestvideo+bestaudio/best',
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([clip_url])
        clip_path = os.path.join(os.getcwd(), f"temp_{clip_id}.mp4")
        max_attempts = 5
        for _ in range(max_attempts):
            if os.path.exists(clip_path) and os.access(clip_path, os.R_OK | os.W_OK):
                logger.info(f"Файл {clip_path} успешно загружен и доступен")
                return clip_path
            await asyncio.sleep(1)
        logger.error(f"Файл {clip_path} не был создан или недоступен после {max_attempts} попыток")
        return None
    except Exception as e:
        logger.error(f"Ошибка скачивания клипа {clip_id} с помощью yt-dlp: {e}")
        return None


def get_clip_duration(clip_path):
    try:
        probe = ffmpeg.probe(clip_path)
        duration = float(probe['format']['duration'])
        logger.info(f"Длительность клипа {clip_path}: {duration} секунд")
        return duration
    except ffmpeg.Error as e:
        logger.error(f"Ошибка получения длительности клипа {clip_path}: {e}")
        return 0
    except (KeyError, ValueError) as e:
        logger.error(f"Ошибка обработки длительности клипа {clip_path}: {e}")
        return 0


def detect_and_blur_ads(clip_path, output_path):
    try:
        if not os.path.exists(clip_path):
            logger.error(f"Файл не найден: {clip_path}")
            return None
        max_attempts = 5
        for _ in range(max_attempts):
            if os.access(clip_path, os.R_OK | os.W_OK):
                break
            logger.warning(f"Файл {clip_path} недоступен, ожидание 1 секунду...")
            time.sleep(1)
        else:
            logger.error(f"Нет прав доступа к файлу после {max_attempts} попыток: {clip_path}")
            return None
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("ffmpeg не найден в системе. Убедитесь, что он установлен и добавлен в PATH.")
            return None
        duration = get_clip_duration(clip_path)
        if duration == 0:
            logger.error(f"Не удалось определить длительность клипа {clip_path}")
            return None
        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            logger.error(f"Не удалось открыть видео: {clip_path}")
            return None
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = max(1, frame_count // 10)
        ad_regions = []
        for i in range(0, frame_count, frame_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = reader.readtext(rgb_frame, detail=1)
            height, width = frame.shape[:2]
            banner_zones = [
                (0, 0, width * 0.3, height * 0.3),
                (width * 0.7, 0, width * 0.3, height * 0.3),
                (0, height * 0.7, width * 0.3, height * 0.3),
                (width * 0.7, height * 0.7, width * 0.3, height * 0.3)
            ]
            game_zone = (width * 0.25, height * 0.25, width * 0.5, height * 0.5)
            ad_keywords = [
                'casino', 'Bet', 'Pari', 'win', 'bonus', 'promo', 'deposit', 'free', 'spin',
                'jackpot', 'gamble', 'бонус', 'выигрыш', 'промо', 'депозит', 'спины', 'джекпот',
                'играть', 'play', 'join', 'регистрация', 'register', 'ставки', 'казино', 'пари',
                'RP', 'Winline', '1win', '1xbet', 'Funpay', 'Lolz', 'VIP', 'ИГРЫ', 'БАННЕР',
                'приглашению', '2421', 'ВАНЗОЛО', 'GTASR', 'RAVE', 'КЛИКАЙ'
            ]
            for (bbox, text, conf) in results:
                if conf < 0.5:
                    continue
                if text.strip() and any(keyword.lower() in text.lower() for keyword in ad_keywords):
                    x_min = min([point[0] for point in bbox])
                    y_min = min([point[1] for point in bbox])
                    x_max = max([point[0] for point in bbox])
                    y_max = max([point[1] for point in bbox])
                    x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min
                    center_x, center_y = x + w / 2, y + h / 2
                    if (game_zone[0] <= center_x <= game_zone[0] + game_zone[2] and
                            game_zone[1] <= center_y <= game_zone[1] + game_zone[3]):
                        logger.info(f"Пропущен текст в игровой зоне: {text}")
                        continue
                    is_banner = False
                    for zone_x, zone_y, zone_w, zone_h in banner_zones:
                        if (zone_x <= center_x <= zone_x + zone_w and
                                zone_y <= center_y <= zone_y + zone_h):
                            is_banner = True
                            break
                    if not is_banner:
                        logger.info(f"Текст {text} не в зоне баннера, пропускаем")
                        continue
                    padding = 30
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(width - x, w + 2 * padding)
                    h = min(height - y, h + 2 * padding)
                    ad_regions.append((x, y, w, h))
                    logger.info(f"Обнаружена реклама: (x={x}, y={y}, w={w}, h={h}), текст: {text}, уверенность: {conf}")
        cap.release()
        if not ad_regions:
            logger.info(f"Реклама не обнаружена в {clip_path}, копируем без изменений")
            return ['ffmpeg', '-i', clip_path, '-c', 'copy', output_path]
        cmd = ['ffmpeg', '-i', clip_path, '-y']
        filter_complex = []
        current_stream = "[0:v]"
        for idx, (x, y, w, h) in enumerate(ad_regions):
            filter_complex.append(
                f"{current_stream}split=2[vmain{idx}][vblur{idx}];"
                f"[vblur{idx}]crop={w}:{h}:{x}:{y},boxblur=20:enable='between(t,0,{duration})'[vblurred{idx}];"
                f"[vmain{idx}][vblurred{idx}]overlay={x}:{y}[vout{idx}]"
            )
            current_stream = f"[vout{idx}]"
        filter_complex_str = ";".join(filter_complex)
        cmd.extend(['-filter_complex', filter_complex_str])
        cmd.extend(['-map', current_stream, '-map', '0:a', '-c:a', 'copy', output_path])
        logger.info(f"Команда ffmpeg для размытия: {' '.join(cmd)}")
        return cmd
    except Exception as e:
        logger.error(f"Ошибка при обнаружении рекламы: {e}")
        return None


async def add_watermark(clip_path, channel_name, clip_title, output_path):
    try:
        escaped_channel_name = channel_name.replace(':', '\\:').replace("'", "\\'").replace('"', '\\"')
        escaped_clip_title = clip_title.replace(':', '\\:').replace("'", "\\'").replace('"', '\\"')
        stream = ffmpeg.input(clip_path)
        stream = ffmpeg.output(
            stream,
            output_path,
            vf=(
                f"drawtext=text='Source\\: {escaped_channel_name}':fontcolor=white@0.9:fontsize=36:box=1:boxcolor=black@0.8:shadowx=2:shadowy=2:x=10:y=h-60:fontfile='C\\:/Windows/Fonts/arialbd.ttf',"
                f"drawtext=text='{escaped_clip_title}':fontcolor=white@0.9:fontsize=32:box=1:boxcolor=black@0.8:shadowx=2:shadowy=2:x=(w-text_w)/2:y=10:fontfile='C\\:/Windows/Fonts/arialbd.ttf'"
            ),
            y="-y"
        )
        await asyncio.get_event_loop().run_in_executor(None, lambda: stream.run())
        max_attempts = 5
        for _ in range(max_attempts):
            if os.path.exists(output_path) and os.access(output_path, os.R_OK | os.W_OK):
                logger.info(f"Водяной знак успешно добавлен в {output_path}")
                return
            await asyncio.sleep(1)
        logger.error(f"Файл {output_path} не доступен после добавления водяного знака")
        raise Exception("Файл не доступен после обработки")
    except ffmpeg.Error as e:
        logger.error(f"Ошибка добавления водяного знака: {e}")
        raise


async def create_transitioned_video(clips, output_path, platform, preview=False, resolution="1280:720"):
    try:
        durations = [get_clip_duration(clip) for clip in clips]
        logger.info(f"Длительности отдельных клипов: {durations}")

        if not all(durations):
            raise ValueError("Не удалось определить длительность одного или нескольких клипов")

        total_duration = sum(durations)
        logger.info(f"Суммарная длительность до обработки: {total_duration} секунд")
        expected_duration = total_duration - (len(clips) - 1) * 0.5  # Учитываем укорачивание из-за переходов
        logger.info(f"Ожидаемая длительность с учетом переходов: {expected_duration} секунд")

        cmd = ['ffmpeg', '-y']
        for i, clip in enumerate(clips):
            cmd.extend(['-r', '60', '-i', clip])
            logger.info(f"Добавлен клип {i}: {clip} с длительностью {durations[i]}")

        filter_complex = []

        if platform == "tiktok":
            for i in range(len(clips)):
                filter_complex.append(
                    f"[{i}:v]fps=60,scale=1080:1920:force_original_aspect_ratio=increase,"
                    f"crop=1080:1920,boxblur=20:1[bg{i}];"
                    f"[{i}:v]fps=60,scale=1080:-2:force_original_aspect_ratio=decrease,"
                    f"setsar=1:1[fg{i}];"
                    f"[bg{i}][fg{i}]overlay=(W-w)/2:(H-h)/2[v{i}_norm]"
                )
                filter_complex.append(
                    f"[{i}:a]aformat=sample_fmts=fltp:channel_layouts=stereo:sample_rates=48000[a{i}_norm]"
                )
        else:
            for i in range(len(clips)):
                scale_str = resolution if preview else "1920:1080"
                fps_str = "60"
                filter_complex.append(
                    f"[{i}:v]fps={fps_str},scale={scale_str}:force_original_aspect_ratio=decrease,"
                    f"pad={scale_str}:(ow-iw)/2:(oh-ih)/2,format=yuv420p[v{i}_norm]"
                )
                filter_complex.append(
                    f"[{i}:a]aformat=sample_fmts=fltp:channel_layouts=stereo:sample_rates=48000[a{i}_norm]"
                )

        xfade_offset = 0
        xfade_duration = 0.5

        for i in range(len(clips) - 1):
            xfade_offset += durations[i] - xfade_duration
            v_in = f"v{i}_norm" if i == 0 else f"v{i}_xf"
            a_in = f"a{i}_norm" if i == 0 else f"a{i}_af"

            filter_complex.append(
                f"[{v_in}][v{i + 1}_norm]xfade=transition=slideleft:duration={xfade_duration}:offset={xfade_offset}[v{i + 1}_xf]"
            )
            filter_complex.append(
                f"[{a_in}][a{i + 1}_norm]acrossfade=d={xfade_duration}[a{i + 1}_af]"
            )

        final_video_label = f"v{len(clips) - 1}_xf" if len(clips) > 1 else "v0_norm"
        final_audio_label = f"a{len(clips) - 1}_af" if len(clips) > 1 else "a0_norm"

        filter_complex_str = ";".join(filter_complex)
        logger.info(f"filter_complex: {filter_complex_str}")

        cmd.extend(['-filter_complex', filter_complex_str])
        cmd.extend(['-map', f'[{final_video_label}]', '-map', f'[{final_audio_label}]'])

        if preview:
            cmd.extend(['-c:v', 'libx264', '-b:v', '1000k', '-c:a', 'aac', '-b:a', '64k', output_path])
        else:
            cmd.extend([
                '-c:v', 'libx264',
                '-b:v', '8000k',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                output_path
            ])

        logger.info(f"Команда ffmpeg: {' '.join(cmd)}")

        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if process.returncode != 0:
            logger.error(f"Ошибка выполнения ffmpeg:\n{process.stderr}")
            raise RuntimeError(f"ffmpeg error: {process.stderr}")
        else:
            logger.info(f"Видео успешно создано: {output_path}")

        final_duration = get_clip_duration(output_path)
        logger.info(f"Итоговая длительность видео (из файла): {final_duration} секунд")
        return final_duration

    except Exception as e:
        logger.exception("Ошибка в create_transitioned_video")
        raise

async def trim_video(input_path, output_path, duration):
    try:
        cmd = ['ffmpeg', '-y', '-i', input_path, '-t', str(duration), '-c:v', 'libx264', '-b:v', '1000k', '-c:a', 'aac',
               '-b:a', '64k', output_path]
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if process.returncode != 0:
            logger.error(f"Ошибка обрезки видео:\n{process.stderr}")
            raise RuntimeError(f"ffmpeg error: {process.stderr}")
        logger.info(f"Видео успешно обрезано до {duration} секунд: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Ошибка при обрезке видео: {e}")
        return False

async def upload_to_youtube(video_path, processed_clips, categories_data, selected_categories, is_short=False):
    try:
        creds = get_youtube_credentials()
        if not creds:
            raise Exception("Не удалось получить credentials для YouTube")

        youtube = build('youtube', 'v3', credentials=creds)

        # Generate author hashtags
        authors = [clip['broadcaster_name'] for clip in processed_clips if clip.get('broadcaster_name')]
        author_hashtags = " ".join([f"#{author.replace(' ', '_')}" for author in authors if author])

        # Generate hashtags for all selected categories
        category_hashtags = " ".join([f"#{categories_data.get(cat_id, '').replace(' ', '')}"
                                    for cat_id in selected_categories if cat_id in categories_data])

        # Combine into description
        description = (
            f"Топ клипы Twitch #twitch #Clips #shorts {category_hashtags} {author_hashtags}"
        ).strip()

        request_body = {
            'snippet': {
                'title': description[:100] if is_short else description[:100],
                'description': description,
                'categoryId': '22' if is_short else '20',
                'tags': ['twitch', 'clips', 'shorts'] if is_short else ['twitch', 'clips']
            },
            'status': {
                'privacyStatus': 'public',
                'selfDeclaredMadeForKids': False
            }
        }
        media = MediaFileUpload(video_path)

        async def upload_with_timeout():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: youtube.videos().insert(
                part='snippet,status',
                body=request_body,
                media_body=media
            ).execute())

        response = await asyncio.wait_for(upload_with_timeout(), timeout=60)
        return response['id']
    except asyncio.TimeoutError:
        logger.error("Тайм-аут при загрузке на YouTube")
        raise Exception("Тайм-аут при загрузке на YouTube")
    except HttpError as e:
        logger.error(f"Ошибка загрузки на YouTube: {e}")
        raise
    except Exception as e:
        logger.error(f"Неожиданная ошибка при загрузке на YouTube: {e}")
        raise

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/824.5",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/91.0 Firefox",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.5 Safari/537.6",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.1 (KHTML, like Gecko) Version/14.1.1 Mobile/15E148 Safari/604.1",
]


async def autogenerate_video(platform="tiktok", clip_count=5, language="any"):
    logger.info(f"Запуск автогенерации видео: platform={platform}, clip_count={clip_count}, language={language}")
    try:
        MIN_DURATION = 61
        categories = await get_top_categories()
        if not categories:
            logger.error("Не удалось получить категории для автогенерации")
            return None, None, None, None

        categories_data = {id: name for name, id in categories}
        selected_categories = random.sample([cat[1] for cat in categories], min(3, len(categories)))
        logger.info(f"Выбраны категории: {selected_categories}")

        temp_clips = []
        processed_clips = []
        conn = sqlite3.connect('clips.db')
        c = conn.cursor()

        all_clips = []
        for category in selected_categories:
            clips = await get_twitch_clips(category, language, clip_count * 10)
            all_clips.extend(clips)
            logger.info(f"Получено {len(clips)} клипов для категории {category}")

        for clip in all_clips:
            if len(temp_clips) >= clip_count:
                break
            clip_id = clip.get('id')
            if not clip_id:
                logger.warning(f"Пропущен клип без ID: {clip}")
                continue
            c.execute("SELECT clip_id FROM clips WHERE clip_id=? AND used_for=?", (clip_id, platform))
            if c.fetchone():
                logger.info(f"Клип {clip_id} уже использован для платформы {platform}")
                continue
            clip_url = clip['url']
            clip_path = await download_clip_with_ytdlp(clip_url, clip_id)
            if not clip_path or not os.path.exists(clip_path):
                logger.error(f"Не удалось скачать клип {clip_id}")
                continue
            blurred_path = os.path.join(os.getcwd(), f"blurred_{clip_id}.mp4")
            blur_cmd = detect_and_blur_ads(clip_path, blurred_path)
            if blur_cmd:
                process = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: subprocess.run(blur_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
                if process.returncode != 0:
                    logger.error(f"Ошибка размытия рекламы для клипа {clip_id}: {process.stderr}")
                    os.remove(clip_path)
                    continue
                os.remove(clip_path)
                clip_path = blurred_path
            watermarked_path = os.path.join(os.getcwd(), f"watermarked_{clip_id}.mp4")
            try:
                await add_watermark(clip_path, clip['broadcaster_name'], clip['title'], watermarked_path)
                temp_clips.append(watermarked_path)
                processed_clips.append(clip)
                logger.info(f"Клип {clip_id} успешно обработан")
            except ffmpeg.Error as e:
                logger.error(f"Ошибка водяного знака для клипа {clip_id}: {e}")
                os.remove(clip_path)
                continue
            try:
                c.execute("INSERT INTO clips VALUES (?, ?, ?, ?)",
                          (clip_id, clip['broadcaster_name'], platform, datetime.now(timezone.utc)))
            except sqlite3.IntegrityError as e:
                logger.warning(f"Клип {clip_id} уже использован для платформы {platform}, пропускаем: {e}")
                os.remove(watermarked_path)
                continue
            os.remove(clip_path)

        preview_output_path = os.path.join(os.getcwd(), f"preview_{uuid4()}.mp4")
        full_output_path = os.path.join(os.getcwd(), f"full_{uuid4()}.mp4")
        preview_duration = await create_transitioned_video(temp_clips, preview_output_path, platform, preview=True)
        full_duration = await create_transitioned_video(temp_clips, full_output_path, platform, preview=False)
        logger.info(f"Длительности после первой сборки: preview={preview_duration}, full={full_duration}")

        if full_duration >= MIN_DURATION:
            logger.info(f"Длительность {full_duration} секунд превышает минимальную {MIN_DURATION}")
        else:
            attempts = 0
            max_attempts = 5
            while full_duration < MIN_DURATION and attempts < max_attempts:
                attempts += 1
                logger.info(f"Попытка {attempts}: Длительность {full_duration} < {MIN_DURATION}, добавляем клипы")
                needed_duration = MIN_DURATION - full_duration
                avg_clip_duration = sum(get_clip_duration(clip) for clip in temp_clips) / len(temp_clips) if temp_clips else 10
                extra_clip_count = max(1, int(needed_duration / avg_clip_duration) + 1)
                logger.info(f"Требуется добавление {extra_clip_count} клипов")

                for category in selected_categories:
                    new_clips = await get_twitch_clips(category, language, extra_clip_count * 2, max_attempts=attempts, search_depth_days=30 * attempts)
                    for clip in new_clips:
                        clip_id = clip['id']
                        if any(c['id'] == clip_id for c in processed_clips):
                            continue
                        c.execute("SELECT clip_id FROM clips WHERE clip_id=? AND used_for=?", (clip_id, platform))
                        if c.fetchone():
                            continue
                        clip_url = clip['url']
                        clip_path = await download_clip_with_ytdlp(clip_url, clip_id)
                        if not clip_path or not os.path.exists(clip_path):
                            continue
                        blurred_path = os.path.join(os.getcwd(), f"blurred_{clip_id}.mp4")
                        blur_cmd = detect_and_blur_ads(clip_path, blurred_path)
                        if blur_cmd:
                            process = await asyncio.get_event_loop().run_in_executor(
                                None, lambda: subprocess.run(blur_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
                            if process.returncode != 0:
                                os.remove(clip_path)
                                continue
                            os.remove(clip_path)
                            clip_path = blurred_path
                        watermarked_path = os.path.join(os.getcwd(), f"watermarked_{clip_id}.mp4")
                        try:
                            await add_watermark(clip_path, clip['broadcaster_name'], clip['title'], watermarked_path)
                            temp_clips.append(watermarked_path)
                            processed_clips.append(clip)
                        except ffmpeg.Error:
                            os.remove(clip_path)
                            continue
                        try:
                            c.execute("INSERT INTO clips VALUES (?, ?, ?, ?)",
                                      (clip_id, clip['broadcaster_name'], platform, datetime.now(timezone.utc)))
                        except sqlite3.IntegrityError as e:
                            logger.warning(f"Клип {clip_id} уже использован для платформы {platform}, пропускаем: {e}")
                            os.remove(watermarked_path)
                            continue
                        os.remove(clip_path)

                        if os.path.exists(preview_output_path):
                            os.remove(preview_output_path)
                        if os.path.exists(full_output_path):
                            os.remove(full_output_path)
                        preview_output_path = os.path.join(os.getcwd(), f"preview_{uuid4()}.mp4")
                        full_output_path = os.path.join(os.getcwd(), f"full_{uuid4()}.mp4")
                        preview_duration = await create_transitioned_video(temp_clips, preview_output_path, platform, preview=True)
                        full_duration = await create_transitioned_video(temp_clips, full_output_path, platform, preview=False)
                        logger.info(f"Длительности после добавления клипа {clip_id}: preview={preview_duration}, full={full_duration}")

                        if full_duration >= MIN_DURATION:
                            logger.info(f"Достигнута минимальная длительность {full_duration} >= {MIN_DURATION}")
                            break
                    if full_duration >= MIN_DURATION:
                        break
                if attempts == max_attempts:
                    logger.warning(f"Не удалось достичь минимальной длительности {MIN_DURATION} после {max_attempts} попыток")
                    break

        conn.commit()
        conn.close()

        if full_duration < MIN_DURATION:
            logger.error(f"Не удалось достичь нужной длительности: {full_duration}")
            return None, None, None, None

        video_id = uuid4().hex[:16]
        save_video_path(video_id, full_output_path)
        save_video_metadata(video_id, processed_clips, categories_data, selected_categories)
        for file in os.listdir():
            if file.startswith(("blurred_", "watermarked_", "preview_", "temp_")) and file.endswith(".mp4"):
                try:
                    os.remove(file)
                    logger.info(f"Удален временный файл: {file}")
                except Exception as e:
                    logger.warning(f"Не удалось удалить файл {file}: {e}")
        logger.info(f"Автогенерация завершена успешно: video_id={video_id}")
        return full_output_path, video_id, processed_clips, categories_data, selected_categories

    except Exception as e:
        logger.error(f"Ошибка в autogenerate_video: {e}", exc_info=True)
        conn.close()
        return None, None, None, None


async def autogeneration_loop(chat_id, platform, selected_categories, language, clip_count):
    global autogeneration_running, last_autogeneration_time
    interval = timedelta(minutes=30)

    state_data = load_autogeneration_state()
    if state_data and state_data['last_generation']:
        last_autogeneration_time = state_data['last_generation']
    else:
        last_autogeneration_time = datetime.now(timezone.utc).astimezone(pytz.timezone('Europe/Kiev'))

    while autogeneration_running:
        try:
            logger.info(f"Начало цикла автогенерации: running={autogeneration_running}")
            current_time = datetime.now(timezone.utc).astimezone(pytz.timezone('Europe/Kiev'))
            logger.info(f"Текущее время (EEST): {current_time}, Последняя генерация: {last_autogeneration_time}")

            time_since_last = current_time - last_autogeneration_time
            if time_since_last >= interval or time_since_last.total_seconds() < 1:
                logger.info("Запуск новой автогенерации")
                full_output_path, video_id, processed_clips, categories_data, selected_categories_result = await autogenerate_video(
                    platform=platform, clip_count=clip_count, language=language
                )
                if full_output_path and os.path.exists(full_output_path) and os.access(full_output_path, os.R_OK):
                    await bot.send_message(chat_id, text="Автогенерация завершена, видео готово!")
                    max_attempts = 3
                    for attempt in range(max_attempts):
                        try:
                            if not os.path.exists(full_output_path):
                                logger.error(f"Файл {full_output_path} не существует перед загрузкой")
                                await bot.send_message(chat_id, "Файл видео не найден. Попробуем снова через 30 минут.")
                                break
                            if platform == "tiktok":
                                await upload_to_tiktok(full_output_path, processed_clips, categories_data,
                                                       selected_categories_result, save_to_draft=True)
                                await bot.send_message(chat_id, text="Видео сохранено как черновик в TikTok")
                                await upload_to_youtube(full_output_path, processed_clips, categories_data,
                                                        selected_categories_result, is_short=True)
                                await bot.send_message(chat_id, text="Видео загружено на YouTube Shorts")
                            else:
                                youtube_id = await upload_to_youtube(full_output_path, processed_clips, categories_data,
                                                                     selected_categories_result)
                                await bot.send_message(chat_id,
                                                       f"Видео загружено на YouTube: https://youtube.com/watch?v={youtube_id}")

                            # Удаление файла и метаданных только после успешной загрузки
                            safe_remove(full_output_path)
                            delete_video_path(video_id)
                            delete_video_metadata(video_id)
                            logger.info(f"Файл {full_output_path} и метаданные удалены после успешной загрузки")
                            break
                        except Exception as upload_error:
                            logger.error(f"Попытка {attempt + 1} загрузки провалилась: {upload_error}")
                            if attempt < max_attempts - 1:
                                logger.info("Ожидание перед повторной попыткой...")
                                await asyncio.sleep(5)
                            else:
                                await bot.send_message(chat_id,
                                                       f"Ошибка загрузки видео: {str(upload_error)}. Следующая попытка через 30 минут.")
                                break

                    last_autogeneration_time = current_time
                    logger.info(f"Обновлено время последней генерации: {last_autogeneration_time}")
                    save_autogeneration_state(
                        autogeneration_running,
                        last_autogeneration_time,
                        platform,
                        selected_categories,
                        language,
                        clip_count,
                        chat_id
                    )
                else:
                    logger.error("Не удалось сгенерировать видео")
                    await bot.send_message(chat_id, "Не удалось сгенерировать видео. Попробуем снова через 30 минут.")

            time_left = interval - (current_time - last_autogeneration_time)
            minutes, seconds = divmod(int(time_left.total_seconds()), 60)
            logger.info(f"До следующей генерации: {minutes} мин {seconds} сек")
            await bot.send_message(chat_id,
                                   f"Следующая автогенерация через {minutes} мин {seconds} сек в {last_autogeneration_time + interval}")
            await asyncio.sleep(time_left.total_seconds())

        except asyncio.CancelledError:
            logger.info("Задача автогенерации была отменена")
            autogeneration_running = False
            save_autogeneration_state(
                autogeneration_running,
                last_autogeneration_time,
                platform,
                selected_categories,
                language,
                clip_count,
                chat_id
            )
            await bot.send_message(chat_id, "Автогенерация остановлена.")
            break
        except Exception as e:
            logger.error(f"Ошибка в цикле автогенерации: {e}")
            await bot.send_message(chat_id,
                                   f"Ошибка автогенерации: {str(e)}. Планирую следующую попытку через 30 минут.")
            await asyncio.sleep(interval.total_seconds())


async def upload_to_tiktok(
    video_path,
    processed_clips,
    categories_data,
    selected_categories,
    save_to_draft=False,
    override_description=None
):
    logger.info(f"Начало загрузки видео в TikTok: {video_path}")
    renamed_video_path = None
    try:
        if not os.path.exists(video_path):
            logger.error(f"Файл {video_path} не существует")
            raise FileNotFoundError(f"Video file {video_path} not found")

        renamed_video_path = os.path.join(os.path.dirname(video_path), "twitch_clips_video.mp4")
        shutil.copy2(video_path, renamed_video_path)
        logger.info(f"Файл переименован в {renamed_video_path} перед загрузкой")

        async with async_playwright() as p:
            logger.info("Запуск Playwright")
            user_agent = random.choice(USER_AGENTS)
            logger.info(f"Используемый user-agent: {user_agent}")

            browser = await asyncio.wait_for(
                p.chromium.launch(headless=True),
                timeout=30
            )
            context = await browser.new_context(
                user_agent=user_agent,
                viewport={'width': 1280, 'height': 720},
                extra_http_headers={
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Referer': 'https://www.tiktok.com/',
                    'DNT': '1',
                }
            )

            cookies_path = 'tiktok_cookies.json'
            if os.path.exists(cookies_path):
                logger.info(f"Загрузка cookies из {cookies_path}")
                with open(cookies_path, 'r') as f:
                    cookies = json.load(f)
                valid_cookies = []
                for cookie in cookies:
                    if not all(key in cookie for key in ['name', 'value', 'domain']):
                        logger.warning(f"Пропуск некорректного cookie: {cookie}")
                        continue
                    same_site = cookie.get('sameSite')
                    if same_site is None or same_site == '':
                        cookie['sameSite'] = 'Lax' if cookie.get('secure', False) else 'None'
                    elif isinstance(same_site, str):
                        same_site_lower = same_site.lower()
                        if same_site_lower in ['unspecified', 'no_restriction', 'lax', 'strict', 'none']:
                            cookie['sameSite'] = {
                                'unspecified': 'Lax',
                                'no_restriction': 'None',
                                'lax': 'Lax',
                                'strict': 'Strict',
                                'none': 'None'
                            }.get(same_site_lower, 'Lax')
                        else:
                            cookie['sameSite'] = 'Lax' if cookie.get('secure', False) else 'None'
                    valid_cookies.append(cookie)
                if valid_cookies:
                    await context.add_cookies(valid_cookies)
                    logger.info(f"Установлено {len(valid_cookies)} cookies")
                else:
                    raise ValueError("Нет валидных cookies в tiktok_cookies.json")
            else:
                raise ValueError("Файл tiktok_cookies.json не найден")

            page = await context.new_page()
            logger.info("Переход на главную страницу TikTok")
            await page.goto('https://www.tiktok.com/tiktokstudio/upload', timeout=30000)

            max_attempts = 10
            attempt = 0
            while attempt < max_attempts:
                try:
                    logger.info("Проверка кнопки 'Select video'")
                    await page.wait_for_selector('button:has-text("Select video")', timeout=5000)
                    logger.info("Кнопка 'Select video' найдена")
                    break
                except Error:
                    logger.info("Сессия не загружена, перезагрузка страницы...")
                    await page.reload(timeout=30000)
                    await asyncio.sleep(random.uniform(5, 10))
                    attempt += 1
            if attempt >= max_attempts:
                raise Exception("Не удалось загрузить сессию")

            logger.info("Переход на страницу загрузки TikTok")
            await page.goto('https://www.tiktok.com/tiktokstudio/upload', timeout=30000)

            if "login" in page.url:
                raise Exception("Cookies невалидны. Удалите tiktok_cookies.json и авторизуйтесь заново.")

            await asyncio.sleep(random.uniform(2, 5))
            await page.mouse.move(random.randint(300, 400), random.randint(200, 300))
            await asyncio.sleep(random.uniform(1, 3))

            logger.info(f"Загрузка файла видео: {renamed_video_path}")
            await page.set_input_files('input[type="file"]', renamed_video_path, timeout=10000)

            # Готовим описание и безопасные дефолты для хэштегов
            used_override = bool(override_description and override_description.strip())
            author_hashtags = ""
            category_hashtags = ""

            if used_override:
                description = override_description.strip()
                logger.info(f"Подготовленное описание (override): {description}")
            else:
                authors = [clip['broadcaster_name'] for clip in processed_clips if clip.get('broadcaster_name')]
                author_hashtags = " ".join([f"#{author.replace(' ', '_')}" for author in authors if author])

                category_hashtags = " ".join([
                    f"#{categories_data.get(cat_id, '').replace(' ', '')}"
                    for cat_id in selected_categories if cat_id in categories_data
                ])

                description = (
                    f" #нарезка  #Clips #тренды  {category_hashtags} {author_hashtags}"
                ).strip()
                logger.info(f"Подготовленное описание: {description}")

            await asyncio.sleep(2)
            await page.mouse.move(400, 500)
            await asyncio.sleep(1)

            logger.info("Заполнение описания")
            description_selectors = [
                'div[contenteditable="true"]',
                'div[role="textbox"]',
                'textarea',
                'div[aria-label*="description"]',
            ]
            description_field = None
            for selector in description_selectors:
                try:
                    description_field = await page.wait_for_selector(selector, timeout=30000)
                    logger.info(f"Поле описания найдено с селектором: {selector}")
                    break
                except Error:
                    logger.warning(f"Селектор {selector} не найден")

            if not description_field:
                raise Exception("Не удалось найти поле описания")

            await description_field.click()
            await page.keyboard.down("Control")
            await page.keyboard.press("a")
            await page.keyboard.up("Control")
            await page.keyboard.press("Backspace")
            await asyncio.sleep(0.5)
            await description_field.fill(description)

            filled_text = await description_field.evaluate("element => element.textContent || element.value")
            logger.info(f"Заполненное описание: {filled_text}")

            # Проверяем хэштеги только если не override
            if not used_override:
                need_refill = False
                if author_hashtags and author_hashtags not in filled_text:
                    need_refill = True
                if category_hashtags and category_hashtags not in filled_text:
                    need_refill = True
                if need_refill:
                    logger.warning("Хэштеги отсутствуют, повторное заполнение")
                    await description_field.fill(description)

            await asyncio.sleep(random.uniform(2, 5))
            await page.mouse.move(random.randint(600, 800), random.randint(400, 600))
            await asyncio.sleep(random.uniform(1, 3))

            logger.info("Ожидание обработки TikTok")
            await asyncio.sleep(5)

            if save_to_draft:
                logger.info("Попытка сохранения в черновики")
                draft_button_xpath = '//*[@id="root"]/div/div/div[2]/div[2]/div/div/div/div[5]/div/button[2]/div[2]'
                try:
                    draft_button = await page.wait_for_selector(f'xpath={draft_button_xpath}', timeout=30000)
                    await draft_button.click(timeout=10000)
                    logger.info("Кнопка 'Save to Draft' нажата")
                except Error as e:
                    logger.error(f"Ошибка при нажатии 'Save to Draft': {e}")
                    raise Exception("Не удалось сохранить видео как черновик")
            else:
                logger.info("Ожидание активной кнопки 'Post'")
                publish_button_selectors = [
                    'xpath=/html/body/div[1]/div/div/div[2]/div[2]/div/div/div/div[4]/div/button[1]',
                    'button:has-text("Опубликовать"):not([disabled])',
                    'button:has-text("Post"):not([disabled])',
                    'button[aria-label="Post"]:not([disabled])',
                ]
                publish_button = None
                max_attempts = 5
                attempt = 0
                while attempt < max_attempts:
                    attempt += 1
                    for selector in publish_button_selectors:
                        try:
                            publish_button = await page.wait_for_selector(selector, timeout=30000)
                            is_disabled = await publish_button.evaluate("el => el.hasAttribute('disabled')")
                            if not is_disabled:
                                logger.info(f"Кнопка 'Post' найдена с селектором: {selector}")
                                break
                            else:
                                logger.info(f"Кнопка 'Post' найдена, но неактивна")
                                publish_button = None
                        except Error:
                            logger.warning(f"Селектор {selector} не найден на попытке {attempt}")
                    if publish_button:
                        break
                    logger.info(f"Кнопка 'Post' не активна, ожидание 10 секунд (попытка {attempt}/{max_attempts})")
                    await asyncio.sleep(10)

                if not publish_button:
                    raise Exception("Не удалось найти активную кнопку 'Post'")

                logger.info("Повторная проверка описания перед публикацией")
                description_field = None
                for selector in description_selectors:
                    try:
                        description_field = await page.wait_for_selector(selector, timeout=10000)
                        break
                    except Error:
                        pass

                if description_field:
                    filled_text = await description_field.evaluate("element => element.textContent || element.value")
                    bad_filename_in_text = "twitch_clips_video" in filled_text.lower()
                    missing_our_hashtags = (not used_override) and (author_hashtags and author_hashtags not in filled_text)
                    if bad_filename_in_text or missing_our_hashtags:
                        await description_field.click()
                        await page.keyboard.down("Control")
                        await page.keyboard.press("a")
                        await page.keyboard.up("Control")
                        await page.keyboard.press("Backspace")
                        await asyncio.sleep(0.5)
                        await description_field.fill(description)

                logger.info("Нажатие кнопки 'Post'")
                await publish_button.click(timeout=10000)
                await page.wait_for_timeout(random.randint(5000, 10000))

            cookies = await context.cookies()
            with open(cookies_path, 'w') as f:
                json.dump(cookies, f)

            await browser.close()
            logger.info(f"Видео успешно {'сохранено как черновик' if save_to_draft else 'загружено'} в TikTok")

            try:
                conn = sqlite3.connect('clips.db')
                c = conn.cursor()
                c.execute("INSERT INTO stats (platform, clip_count, created_at) VALUES (?, ?, ?)",
                          ("tiktok", len(processed_clips), datetime.now(timezone.utc)))
                conn.commit()
            except sqlite3.Error as e:
                logger.error(f"Ошибка записи статистики для TikTok: {e}")
            finally:
                conn.close()

    except Exception as e:
        logger.error(f"Ошибка загрузки в TikTok: {e}")
        raise
    finally:
        if renamed_video_path and os.path.exists(renamed_video_path):
            os.remove(renamed_video_path)
            logger.info(f"Удален временный файл: {renamed_video_path}")

def save_video_path(video_id, file_path):
    try:
        conn = sqlite3.connect('clips.db')
        c = conn.cursor()
        c.execute("INSERT INTO videos (video_id, file_path, created_at) VALUES (?, ?, ?)",
                  (video_id, file_path, datetime.now(timezone.utc)))
        conn.commit()
        logger.info(f"Путь видео сохранен в БД: video_id={video_id}, file_path={file_path}")
    except sqlite3.Error as e:
        logger.error(f"Ошибка сохранения пути видео: {e}")
        raise
    finally:
        conn.close()

def save_video_metadata(video_id, processed_clips, categories_data, selected_categories):
    try:
        conn = sqlite3.connect('clips.db')
        c = conn.cursor()
        c.execute(
            "INSERT INTO video_metadata (video_id, processed_clips, categories_data, selected_categories, created_at) VALUES (?, ?, ?, ?, ?)",
            (video_id, json.dumps(processed_clips), json.dumps(categories_data), json.dumps(selected_categories),
             datetime.now(timezone.utc)))
        conn.commit()
        logger.info(f"Метаданные видео сохранены: video_id={video_id}")
    except sqlite3.Error as e:
        logger.error(f"Ошибка сохранения метаданных: {e}")
        raise
    finally:
        conn.close()

def get_video_path(video_id):
    try:
        conn = sqlite3.connect('clips.db')
        c = conn.cursor()
        c.execute("SELECT file_path FROM videos WHERE video_id=?", (video_id,))
        result = c.fetchone()
        if result:
            logger.info(f"Путь видео найден: video_id={video_id}, file_path={result[0]}")
            return result[0]
        return None
    except sqlite3.Error as e:
        logger.error(f"Ошибка получения пути видео: {e}")
        return None
    finally:
        conn.close()

def get_video_metadata(video_id):
    try:
        conn = sqlite3.connect('clips.db')
        c = conn.cursor()
        c.execute("SELECT processed_clips, categories_data, selected_categories FROM video_metadata WHERE video_id=?",
                  (video_id,))
        result = c.fetchone()
        if result:
            return json.loads(result[0]), json.loads(result[1]), json.loads(result[2])
        return None, None, None
    except sqlite3.Error as e:
        logger.error(f"Ошибка получения метаданных: {e}")
        return None, None, None
    finally:
        conn.close()

def delete_video_path(video_id):
    try:
        conn = sqlite3.connect('clips.db')
        c = conn.cursor()
        c.execute("DELETE FROM videos WHERE video_id=?", (video_id,))
        conn.commit()
        logger.info(f"Путь видео удален: video_id={video_id}")
    except sqlite3.Error as e:
        logger.error(f"Ошибка удаления пути видео: {e}")
    finally:
        conn.close()

def delete_video_metadata(video_id):
    try:
        conn = sqlite3.connect('clips.db')
        c = conn.cursor()
        c.execute("DELETE FROM video_metadata WHERE video_id=?", (video_id,))
        conn.commit()
        logger.info(f"Метаданные видео удалены: video_id={video_id}")
    except sqlite3.Error as e:
        logger.error(f"Ошибка удаления метаданных: {e}")
    finally:
        conn.close()

def get_stats():
    try:
        conn = sqlite3.connect('clips.db')
        c = conn.cursor()
        c.execute("SELECT platform, clip_count, created_at FROM stats ORDER BY created_at DESC LIMIT 10")
        results = c.fetchall()
        if not results:
            return "Статистика отсутствует."
        stats_text = "Последние 10 записей статистики:\n"
        for platform, clip_count, created_at in results:
            stats_text += f"Платформа: {platform}, Клипов: {clip_count}, Дата: {created_at}\n"
        return stats_text
    except sqlite3.Error as e:
        logger.error(f"Ошибка получения статистики: {e}")
        return "Ошибка при получении статистики."
    finally:
        conn.close()

async def generate_youtube_trending_to_tiktok(region=None, chunk_seconds=180):
    try:
        logger.info(f"[YT Trending] Старт (входной region={region})")
        api_key = YOUTUBE_API_KEY  # импортирован из Config
        if not api_key:
            raise RuntimeError("YOUTUBE_API_KEY не задан")

        # 0) Выбор региона
        if not region or region == "random":
            region = random.choice(ALLOWED_YT_REGIONS)
        logger.info(f"[YT Trending] Выбран регион: {region}")

        # 1) Перебор разрешённых категорий в случайном порядке
        categories_order = list(ALLOWED_YT_CATEGORIES)
        random.shuffle(categories_order)

        conn = sqlite3.connect('clips.db')
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS used_youtube_videos
                     (id TEXT PRIMARY KEY, title TEXT, channel_id TEXT, channel_title TEXT, created_at TIMESTAMP)""")

        candidate = None
        picked_category = None

        for cat_id in categories_order:
            url = ("https://www.googleapis.com/youtube/v3/videos"
                   f"?part=snippet&chart=mostPopular&regionCode={region}"
                   f"&videoCategoryId={cat_id}&maxResults=25&key={api_key}")
            r = requests.get(url, timeout=15)
            try:
                r.raise_for_status()
            except Exception as e:
                logger.warning(f"[YT Trending] Ошибка запроса для cat={cat_id}: {e}")
                continue

            items = r.json().get("items", [])
            logger.info(f"[YT Trending] Получено {len(items)} видео для cat={cat_id} ({ALLOWED_YT_CATEGORY_NAMES.get(cat_id,'?')})")

            # 2) Ищем первое новое видео строго из разрешённых категорий
            for it in items:
                vid = it.get("id")
                snip = it.get("snippet", {})
                # Доп.проверка на случай смешанных выдач
                if snip.get("categoryId") not in ALLOWED_YT_CATEGORIES:
                    continue
                c.execute("SELECT 1 FROM used_youtube_videos WHERE id=?", (vid,))
                if c.fetchone():
                    continue
                candidate = {
                    "id": vid,
                    "title": snip.get("title", "YouTube Video"),
                    "channel_id": snip.get("channelId", ""),
                    "channel_title": snip.get("channelTitle", "Author"),
                }
                picked_category = cat_id
                break

            if candidate:
                break

        if not candidate:
            conn.close()
            logger.info("[YT Trending] Подходящих новых видео не нашлось в разрешённых категориях/регионах")
            return False

        video_id = candidate["id"]
        title = candidate["title"]
        channel_id = candidate["channel_id"]
        channel_title = candidate["channel_title"]
        logger.info(f"[YT Trending] Выбран: {video_id} | {title} | {channel_title} | cat={picked_category} ({ALLOWED_YT_CATEGORY_NAMES.get(picked_category,'?')}) | region={region}")

        # 3) Скачиваем исходник
        raw_path = os.path.abspath(f"{video_id}.mp4")
        ydl_opts = {
            "format": "bv*+ba/b",
            "outtmpl": raw_path,
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        if not os.path.exists(raw_path):
            raise RuntimeError("Не удалось скачать исходное видео")

        # 4) SponsorBlock: вырезаем рекламные сегменты
        total_dur = get_clip_duration(raw_path)
        ad_segments = _fetch_sponsorblock_segments(video_id, {"sponsor", "selfpromo"})
        keep_ranges = _calc_non_ad_ranges(total_dur, ad_segments) if ad_segments else [(0.0, total_dur)]
        cleaned_path = os.path.abspath(f"clean_{video_id}.mp4")
        ok = _ffmpeg_cut_out_segments(raw_path, cleaned_path, keep_ranges)
        if not ok:
            logger.warning("Не удалось собрать очищенное видео, используем оригинал")
            cleaned_path = raw_path

        # 5) Режем на 3-минутные куски
        out_dir = os.path.abspath(f"yt_chunks_{video_id}")
        chunks = _ffmpeg_split_to_chunks(cleaned_path, out_dir, base=video_id, chunk_seconds=chunk_seconds)
        if not chunks:
            raise RuntimeError("Не удалось нарезать видео на фрагменты")

        # 6) Каждый кусок -> вертикаль 9:16 и загрузка в TikTok черновиком (в прямом порядке, с частью в описании)
        total_parts = len(chunks)
        uploaded = 0
        for part_num, chunk in enumerate(chunks, start=1):  # Прямой порядок
            vertical_path = os.path.abspath(f"yt_vert_{video_id}_{part_num:03d}.mp4")
            dur = await create_transitioned_video([chunk], vertical_path, platform="tiktok", preview=False)
            if not dur or not os.path.exists(vertical_path):
                logger.warning(f"Не удалось собрать вертикаль для {chunk}, пропуск")
                safe_remove(chunk)
                continue

            # Описание с частью
            description = _build_tiktok_description(title=title, channel_title=channel_title, channel_id=channel_id, part_num=part_num)

            # processed_clips — для статистики
            processed_for_stats = [{"broadcaster_name": channel_title}]

            try:
                await upload_to_tiktok(
                    vertical_path,
                    processed_clips=processed_for_stats,
                    categories_data={},
                    selected_categories=[],
                    save_to_draft=True,
                    override_description=description
                )
                await publish_all_tiktok_drafts()
                uploaded += 1
            except Exception as e:
                logger.error(f"Ошибка загрузки фрагмента {part_num}: {e}")

            # Немедленная очистка локальных файлов фрагмента
            safe_remove(vertical_path)
            safe_remove(chunk)

        # 7) Помечаем видео как использованное
        try:
            c.execute(
                "INSERT OR IGNORE INTO used_youtube_videos (id, title, channel_id, channel_title, created_at) VALUES (?, ?, ?, ?, ?)",
                (video_id, title, channel_id, channel_title, datetime.now(timezone.utc))
            )
            conn.commit()
        finally:
            conn.close()

        # 8) Удаление временных файлов исходника/очистки
        safe_remove(raw_path)
        if os.path.abspath(cleaned_path) != os.path.abspath(raw_path):
            safe_remove(cleaned_path)
        try:
            if os.path.isdir(out_dir):
                shutil.rmtree(out_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Не удалось удалить директорию {out_dir}: {e}")

        logger.info(f"[YT Trending] Готово. Загружено фрагментов: {uploaded}")
        return uploaded > 0

    except Exception as e:
        logger.error(f"[YT Trending] Ошибка: {e}", exc_info=True)
        return False


@dp.message(Command(commands=["start"]))
async def start_command(message: Message):
    global autogeneration_running, last_autogeneration_time
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="YouTube", callback_data="platform_youtube")],
            [InlineKeyboardButton(text="TikTok", callback_data="platform_tiktok")],
            [InlineKeyboardButton(text="Автогенерация", callback_data="autogeneration_menu")],
            [InlineKeyboardButton(text="Статистика", callback_data="show_stats")],
            [InlineKeyboardButton(text="Опубликовать", callback_data="publish_tiktok")],
            [InlineKeyboardButton(text="Генерация с YouTube", callback_data="yt_trending")],
        ]
    )
    status = "включена" if autogeneration_running else "выключена"
    await message.answer(f"Выберите действие (автогенерация {status}):", reply_markup=keyboard)


@dp.callback_query(lambda c: c.data == "autogeneration_menu")
async def autogeneration_menu(callback_query: CallbackQuery):
    global autogeneration_running
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Включить", callback_data="autogeneration_start")],
            [InlineKeyboardButton(text="Выключить", callback_data="autogeneration_stop")],
            [InlineKeyboardButton(text="Узнать время", callback_data="autogeneration_time")],
            [InlineKeyboardButton(text="Назад", callback_data="back_to_start")]
        ]
    )
    await callback_query.message.edit_text("Управление автогенерацией:", reply_markup=keyboard)
    await callback_query.answer()

@dp.callback_query(lambda c: c.data == "yt_trending")
async def handle_yt_trending(callback_query: CallbackQuery):
    await callback_query.answer("Запускаю генерацию из трендов YouTube. Это займёт некоторое время.")
    try:
        ok = await generate_youtube_trending_to_tiktok(region="UA", chunk_seconds=180)
        if ok:
            await callback_query.message.answer("Готово: фрагменты сохранены как черновики в TikTok.")
        else:
            await callback_query.message.answer("Не удалось сгенерировать ролики из YouTube трендов.")
    except Exception as e:
        logger.error(f"Ошибка yt_trending: {e}")
        await callback_query.message.answer("Произошла ошибка при генерации из YouTube.")

@dp.callback_query(lambda c: c.data == "autogeneration_start")
async def autogeneration_start(callback_query: CallbackQuery, state: FSMContext):
    state_data = load_autogeneration_state()
    if state_data and state_data['running']:
        await callback_query.message.answer("Автогенерация уже включена.")
        await callback_query.answer()
        return
    await state.set_state(VideoGeneration.platform)
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="YouTube", callback_data="autogeneration_platform_youtube")],
            [InlineKeyboardButton(text="TikTok", callback_data="autogeneration_platform_tiktok")]
        ]
    )
    await callback_query.message.edit_text("Выберите платформу для автогенерации:", reply_markup=keyboard)
    await callback_query.answer()


@dp.callback_query(lambda c: c.data.startswith("autogeneration_platform_"))
async def process_autogeneration_platform(callback_query: CallbackQuery, state: FSMContext):
    platform = callback_query.data.split("_")[-1]
    await state.update_data(platform=platform, autogeneration=True)  # Добавляем флаг автогенерации
    await state.set_state(VideoGeneration.clip_count)
    logger.info(f"Установлена платформа: {platform}, состояние: {VideoGeneration.clip_count.state}")
    await callback_query.message.edit_text("Введите количество клипов для автогенерации (1-20):")
    await callback_query.answer()

@dp.callback_query(lambda c: c.data.startswith("autogeneration_category_") and c.data.split("_")[2].isdigit())
async def process_autogeneration_category(callback_query: CallbackQuery, state: FSMContext):
    logger.info(f"Обработка категории: {callback_query.data}")
    parts = callback_query.data.split("_")
    if len(parts) < 3:
        logger.error(f"Некорректный callback_data: {callback_query.data}")
        await callback_query.answer("Ошибка в обработке категории.", show_alert=True)
        return
    category_id = parts[2]
    data = await state.get_data()
    categories_data = data.get("categories_data", {})
    selected_categories = data.get("selected_categories", set())
    was_selected = category_id in selected_categories
    if was_selected and len(selected_categories) > 1:
        selected_categories.remove(category_id)
    elif not was_selected and len(selected_categories) < 3:
        selected_categories.add(category_id)
    await state.update_data(selected_categories=selected_categories)
    categories = await get_top_categories()
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(
                text=f"✓ {categories_data.get(cat[1], cat[1])}" if cat[1] in selected_categories else categories_data.get(cat[1], cat[1]),
                callback_data=f"autogeneration_category_{cat[1]}_{datetime.now(timezone.utc).timestamp()}"
            )] for cat in categories
        ] + [[InlineKeyboardButton(text="Закончить выбор", callback_data="autogeneration_category_done")]]
    )
    await callback_query.message.edit_reply_markup(reply_markup=keyboard)
    await callback_query.answer()



@dp.callback_query(lambda c: c.data.startswith("autogeneration_language_"))
async def process_autogeneration_language(callback_query: CallbackQuery, state: FSMContext):
    logger.info(f"Обработка языка: {callback_query.data}")
    language = callback_query.data.split("_")[-1]
    try:
        await state.update_data(language=language)
        data = await state.get_data()
        platform = data.get("platform")
        selected_categories = data.get("selected_categories")
        clip_count = data.get("clip_count", 5)
        global autogeneration_running, autogeneration_task, last_autogeneration_time
        if autogeneration_running:
            logger.warning("Автогенерация уже активна")
            await callback_query.message.edit_text("Автогенерация уже запущена.")
            await callback_query.answer()
            return
        autogeneration_running = True  # Устанавливаем здесь
        last_autogeneration_time = datetime.now(timezone.utc).astimezone(pytz.timezone('Europe/Kiev'))
        logger.info(f"Создание задачи автогенерации: platform={platform}, categories={selected_categories}, language={language}, clip_count={clip_count}")
        autogeneration_task = asyncio.create_task(
            autogeneration_loop(callback_query.message.chat.id, platform, selected_categories, language, clip_count))
        save_autogeneration_state(
            autogeneration_running,
            last_autogeneration_time,
            platform,
            selected_categories,
            language,
            clip_count,
            callback_query.message.chat.id
        )
        interval = timedelta(minutes=30)
        next_run = last_autogeneration_time + interval
        await callback_query.message.edit_text(
            f"Автогенерация включена для платформы {platform}, категорий {', '.join([data.get('categories_data', {}).get(cat, cat) for cat in selected_categories])}, языка {language}. "
            f"Следующая автогенерация запланирована на {next_run.strftime('%H:%M:%S %Y-%m-%d')}."
        )
        await callback_query.answer("Автогенерация запущена")
    except Exception as e:
        logger.error(f"Ошибка при запуске автогенерации: {e}", exc_info=True)
        autogeneration_running = False
        autogeneration_task = None
        save_autogeneration_state(
            False,
            last_autogeneration_time,
            platform,
            selected_categories,
            language,
            clip_count,
            callback_query.message.chat.id
        )
        await callback_query.answer("Ошибка при запуске автогенерации. Попробуйте снова.", show_alert=True)
        await state.clear()

@dp.callback_query(lambda c: c.data == "autogeneration_stop")
async def autogeneration_stop(callback_query: CallbackQuery):
    global autogeneration_task, autogeneration_running
    state_data = load_autogeneration_state()
    if state_data and state_data['running']:
        logger.info("Остановка автогенерации по команде пользователя")
        autogeneration_running = False
        if autogeneration_task:
            autogeneration_task.cancel()
            try:
                await autogeneration_task
            except asyncio.CancelledError:
                logger.info("Задача автогенерации успешно отменена")
            autogeneration_task = None
        save_autogeneration_state(
            False,
            state_data['last_generation'],
            state_data['platform'],
            state_data['selected_categories'],
            state_data['language'],
            state_data['clip_count'],
            state_data['chat_id']
        )
        await callback_query.message.edit_text("Автогенерация выключена.")
    else:
        await callback_query.message.edit_text("Автогенерация уже выключена.")
    await callback_query.answer()


@dp.callback_query(lambda c: c.data == "autogeneration_time")
async def autogeneration_time(callback_query: CallbackQuery):
    state_data = load_autogeneration_state()
    if not autogeneration_running or not state_data or not state_data['running'] or not state_data['last_generation']:
        await callback_query.message.edit_text("Автогенерация выключена.")
        await callback_query.answer()
        return
    current_time = datetime.now(timezone.utc).astimezone(pytz.timezone('Europe/Kiev'))
    interval = timedelta(minutes=30)
    next_run = state_data['last_generation'] + interval
    if current_time >= next_run:
        next_run = current_time + interval
    time_left = next_run - current_time
    if time_left.total_seconds() > 0:
        minutes, seconds = divmod(int(time_left.total_seconds()), 60)
        await callback_query.message.edit_text(f"До следующей генерации: {minutes} мин {seconds} сек")
    else:
        await callback_query.message.edit_text("Генерация скоро начнется.")
    await callback_query.answer()

@dp.callback_query(lambda c: c.data == "back_to_start")
async def back_to_start(callback_query: CallbackQuery):
    await start_command(callback_query.message)
    await callback_query.answer()


@dp.callback_query(lambda c: c.data == "show_stats")
async def show_stats(callback_query: CallbackQuery):
    stats = get_stats()
    await callback_query.message.edit_text(stats)
    await callback_query.answer()


@dp.callback_query(lambda c: c.data.startswith("platform_"))
async def process_platform(callback_query: CallbackQuery, state: FSMContext):
    platform = callback_query.data.split("_")[1]
    await state.update_data(platform=platform)
    await state.set_state(VideoGeneration.clip_count)
    await callback_query.message.edit_text("Введите количество клипов (1-20):")
    await callback_query.answer()




@dp.callback_query(lambda c: c.data.startswith("category_"))
async def process_category(callback_query: CallbackQuery, state: FSMContext):
    logger.info(f"Обработка категории: {callback_query.data}")
    data = await state.get_data()
    if data.get("autogeneration"):
        logger.error("process_category вызван в контексте автогенерации, перенаправляем")
        await process_autogeneration_category(callback_query, state)
        return

    parts = callback_query.data.split("_")
    if len(parts) < 3:
        logger.error(f"Некорректный callback_data: {callback_query.data}")
        await callback_query.answer("Ошибка в обработке категории.", show_alert=True)
        return
    category_id = parts[1]
    categories_data = data.get("categories_data", {})
    selected_categories = data.get("selected_categories", set())
    was_selected = category_id in selected_categories
    if was_selected and len(selected_categories) > 1:
        selected_categories.remove(category_id)
    elif not was_selected and len(selected_categories) < 3:
        selected_categories.add(category_id)
    await state.update_data(selected_categories=selected_categories)
    categories = await get_top_categories()
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(
                text=f"✓ {categories_data.get(cat[1], cat[1])}" if cat[1] in selected_categories else categories_data.get(cat[1], cat[1]),
                callback_data=f"category_{cat[1]}_{datetime.now(timezone.utc).timestamp()}"
            )] for cat in categories
        ] + [[InlineKeyboardButton(text="Закончить выбор", callback_data="categories_done")]]
    )
    await callback_query.message.edit_reply_markup(reply_markup=keyboard)
    await callback_query.answer()

@dp.message(StateFilter(VideoGeneration.clip_count))
async def process_clip_count(message: Message, state: FSMContext):
    try:
        count = int(message.text)
        if count <= 0 or count > 20:
            await message.answer("Введите число от 1 до 20")
            return
        await state.update_data(clip_count=count)
        data = await state.get_data()
        if data.get("autogeneration"):
            await state.set_state(VideoGeneration.categories)
            categories = await get_top_categories()
            if not categories:
                await message.answer("Ошибка загрузки категорий. Попробуйте позже.")
                await state.clear()
                return
            categories_data = {cat[1]: cat[0] for cat in categories}
            await state.update_data(categories_data=categories_data)
            keyboard = InlineKeyboardMarkup(
                inline_keyboard=[
                    [InlineKeyboardButton(text=categories_data.get(cat[1], cat[1]),
                                          callback_data=f"autogeneration_category_{cat[1]}_{datetime.now(timezone.utc).timestamp()}")]
                    for cat in categories
                ] + [[InlineKeyboardButton(text="Закончить выбор", callback_data="autogeneration_category_done")]]
            )
            await message.answer("Выберите категории для автогенерации (до 3):", reply_markup=keyboard)
        else:
            await state.set_state(VideoGeneration.categories)
            categories = await get_top_categories()
            if not categories:
                await message.answer("Ошибка загрузки категорий. Попробуйте позже.")
                await state.clear()
                return
            categories_data = {cat[1]: cat[0] for cat in categories}
            await state.update_data(categories_data=categories_data)
            keyboard = InlineKeyboardMarkup(
                inline_keyboard=[
                    [InlineKeyboardButton(text=categories_data.get(cat[1], cat[1]),
                                          callback_data=f"category_{cat[1]}_{datetime.now(timezone.utc).timestamp()}")]
                    for cat in categories
                ] + [[InlineKeyboardButton(text="Закончить выбор", callback_data="categories_done")]]
            )
            await message.answer("Выберите категории (до 3):", reply_markup=keyboard)
    except ValueError:
        await message.answer("Пожалуйста, введите число")
    except Exception as e:
        logger.error(f"Ошибка в process_clip_count: {e}", exc_info=True)
        await message.answer("Произошла ошибка. Начните заново с /start.")
        await state.clear()


@dp.callback_query(lambda c: c.data.startswith("upload_"))
async def process_upload(callback_query: CallbackQuery, state: FSMContext):
    logger.info("Вызов функции process_upload")
    _, platform, video_id = callback_query.data.split("_", 2)
    video_path = get_video_path(video_id)
    processed_clips, categories_data, selected_categories = get_video_metadata(video_id)
    logger.info(
        f"Путь к видео: {video_path}, платформа: {platform}, video_id: {video_id}, категории: {selected_categories}")

    try:
        # Проверка существования и доступности файла
        if not video_path or not os.path.exists(video_path):
            logger.error(f"Файл видео не найден: {video_path}")
            await callback_query.message.answer("Файл видео не найден. Попробуйте заново.")
            return
        if not os.access(video_path, os.R_OK):
            logger.error(f"Нет прав на чтение файла: {video_path}")
            await callback_query.message.answer("Нет прав на чтение файла видео. Попробуйте заново.")
            return
        if not processed_clips or not categories_data or not selected_categories:
            logger.error(f"Метаданные отсутствуют: video_id={video_id}")
            await callback_query.message.answer("Метаданные видео не найдены. Попробуйте заново.")
            return

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                if platform == "youtube":
                    logger.info(f"Попытка {attempt + 1}: Загрузка на YouTube")
                    youtube_id = await upload_to_youtube(video_path, processed_clips, categories_data,
                                                         selected_categories)
                    await callback_query.message.answer(
                        f"Загружено на YouTube: https://youtube.com/watch?v={youtube_id}")
                else:
                    logger.info(f"Попытка {attempt + 1}: Загрузка на TikTok")
                    await upload_to_tiktok(video_path, processed_clips, categories_data, selected_categories,
                                           save_to_draft=True)
                    await callback_query.message.answer("Видео сохранено как черновик в TikTok")
                    logger.info(f"Попытка {attempt + 1}: Загрузка на YouTube Shorts")
                    shorts_id = await upload_to_youtube(video_path, processed_clips, categories_data,
                                                        selected_categories, is_short=True)
                    await callback_query.message.answer(
                        f"Загружено на YouTube Shorts: https://youtube.com/shorts/{shorts_id}")

                # Запись статистики
                try:
                    conn = sqlite3.connect('clips.db')
                    c = conn.cursor()
                    c.execute("INSERT INTO stats (platform, clip_count, created_at) VALUES (?, ?, ?)",
                              (platform, len(processed_clips), datetime.now(timezone.utc)))
                    conn.commit()
                except sqlite3.Error as e:
                    logger.error(f"Ошибка записи статистики: {e}")
                finally:
                    conn.close()

                # Удаление файла и метаданных только после успешной загрузки
                safe_remove(video_path)
                delete_video_path(video_id)
                delete_video_metadata(video_id)
                logger.info(f"Файл {video_path} и метаданные удалены после успешной загрузки")
                break

            except Exception as e:
                logger.error(f"Попытка {attempt + 1} загрузки провалилась: {e}")
                if attempt < max_attempts - 1:
                    logger.info("Ожидание перед повторной попыткой...")
                    await asyncio.sleep(5)
                else:
                    await callback_query.message.answer(
                        f"Не удалось загрузить видео после {max_attempts} попыток: {str(e)}. Попробуйте снова.")
                    # НЕ удаляем файл в случае ошибки, чтобы сохранить возможность повторной попытки
                    return

    except Exception as e:
        logger.error(f"Общая ошибка загрузки: {e}")
        await callback_query.message.answer(f"Ошибка при загрузке видео: {str(e)}. Попробуйте снова.")
    finally:
        # Очистка оставшихся временных файлов
        for file in os.listdir():
            if file.startswith(("blurred_", "preview_", "watermarked_")) and file.endswith(".mp4"):
                safe_remove(file)


@dp.callback_query(lambda c: c.data == "redo")
async def process_redo(callback_query: CallbackQuery, state: FSMContext):
    logger.info("Вызов функции process_redo")
    data = await state.get_data()
    video_id = data.get("video_id")
    video_path = get_video_path(video_id)
    if video_path and os.path.exists(video_path):
        os.remove(video_path)
        delete_video_path(video_id)
        delete_video_metadata(video_id)
    await state.clear()
    await start_command(callback_query.message)
    await callback_query.answer()

@dp.callback_query(lambda c: c.data == "autogeneration_category_done")
async def process_autogeneration_categories_done(callback_query: CallbackQuery, state: FSMContext):
    logger.info("➡ Вызван process_autogeneration_categories_done")
    data = await state.get_data()
    selected_categories = data.get("selected_categories", set())

    if not selected_categories:
        await callback_query.message.answer("Выберите хотя бы одну категорию.")
        await callback_query.answer()
        return

    await state.update_data(selected_categories=list(selected_categories))
    await state.set_state(VideoGeneration.language)

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🇷🇺 Русский", callback_data="autogeneration_language_ru")],
            [InlineKeyboardButton(text="🇬🇧 Английский", callback_data="autogeneration_language_en")],
            [InlineKeyboardButton(text="🌍 Любой", callback_data="autogeneration_language_any")]
        ]
    )

    await callback_query.message.edit_text("Выберите язык для автогенерации:", reply_markup=keyboard)
    await callback_query.answer()



@dp.callback_query(lambda c: c.data == "categories_done")
async def process_categories_done(callback_query: CallbackQuery, state: FSMContext):
    logger.info("Вызов process_categories_done")
    data = await state.get_data()
    if data.get("autogeneration"):  # Если это автогенерация, перенаправляем
        logger.error("process_categories_done вызван в контексте автогенерации, перенаправляем в process_autogeneration_categories_done")
        await process_autogeneration_categories_done(callback_query, state)
        return

    selected_categories = data.get("selected_categories", set())
    if not selected_categories:
        categories = await get_top_categories()
        selected_categories = [cat[1] for cat in random.sample(categories, min(3, len(categories)))]
    await state.update_data(selected_categories=list(selected_categories))
    await state.set_state(VideoGeneration.language)
    logger.info(f"Установлено состояние: {VideoGeneration.language.state}")
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Русский", callback_data="language_ru")],
            [InlineKeyboardButton(text="Английский", callback_data="language_en")],
            [InlineKeyboardButton(text="Любой", callback_data="language_any")]
        ]
    )
    await callback_query.message.edit_text("Выберите язык:", reply_markup=keyboard)
    await callback_query.answer()




@dp.callback_query(lambda c: c.data.startswith("language_"))
async def process_language(callback_query: CallbackQuery, state: FSMContext):
    logger.info(f"Вызов process_language с callback_data: {callback_query.data}")
    data = await state.get_data()
    if data.get("autogeneration"):
        logger.info("Перенаправление в process_autogeneration_language из-за флага autogeneration")
        await process_autogeneration_language(callback_query, state)
        return

    await callback_query.answer("Создание видео начато, это может занять 1-3 минуты. Пожалуйста, подождите.")
    temp_files = []
    try:
        current_state = await state.get_state()
        logger.info(f"Текущее состояние FSM: {current_state}")
        if current_state != VideoGeneration.language.state:
            logger.error(f"Неверное состояние: ожидалось {VideoGeneration.language.state}, получено {current_state}")
            await callback_query.message.answer("Состояние сброшено. Начните заново с /start.")
            await state.clear()
            return

        language = callback_query.data.split("_")[1]
        logger.info(f"Выбран язык: {language}")
        await state.update_data(language=language)
        data = await state.get_data()
        platform = data.get("platform")
        clip_count = data.get("clip_count")
        selected_categories = data.get("selected_categories", [])
        categories_data = data.get("categories_data", {})
        logger.info(f"Данные состояния: platform={platform}, clip_count={clip_count}, categories={selected_categories}")

        if not all([platform, clip_count, selected_categories]):
            logger.error("Отсутствуют необходимые данные в состоянии")
            await callback_query.message.answer("Ошибка: недостающие данные. Начните заново с /start.")
            await state.clear()
            return

        temp_clips = []
        processed_clips = []
        conn = sqlite3.connect('clips.db')
        c = conn.cursor()
        processed_count = 0

        logger.info("Начало загрузки клипов")
        all_clips = []
        for category in selected_categories:
            all_clips.extend(await get_twitch_clips(category, language, clip_count * 10))
        logger.info(f"Найдено клипов: {len(all_clips)}")

        for clip in all_clips:
            if processed_count >= clip_count:
                logger.info("Достигнуто нужное количество клипов")
                break
            clip_id = clip.get('id')
            if not clip_id:
                logger.warning(f"Отсутствует clip_id в клипе: {clip}")
                continue
            c.execute("SELECT clip_id FROM clips WHERE clip_id=? AND used_for=?", (clip_id, platform))
            if c.fetchone():
                logger.info(f"Клип {clip_id} уже использован для платформы {platform}")
                continue
            clip_url = clip['url']
            clip_path = await download_clip_with_ytdlp(clip_url, clip_id)
            if not clip_path or not os.path.exists(clip_path):
                logger.error(f"Не удалось скачать клип {clip_id}")
                continue
            temp_files.append(clip_path)
            blurred_path = os.path.join(os.getcwd(), f"blurred_{clip_id}.mp4")
            blur_cmd = detect_and_blur_ads(clip_path, blurred_path)
            if blur_cmd:
                process = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: subprocess.run(blur_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
                if process.returncode != 0:
                    logger.error(f"Ошибка размытия рекламы для {clip_id}: {process.stderr}")
                    os.remove(clip_path)
                    continue
                os.remove(clip_path)
                temp_files.remove(clip_path)
                clip_path = blurred_path
                temp_files.append(clip_path)
            watermarked_path = os.path.join(os.getcwd(), f"watermarked_{clip_id}.mp4")
            try:
                await add_watermark(clip_path, clip['broadcaster_name'], clip['title'], watermarked_path)
                temp_clips.append(watermarked_path)
                temp_files.append(watermarked_path)
                processed_clips.append(clip)
                processed_count += 1
                logger.info(f"Обработан клип {clip_id}")
            except ffmpeg.Error as e:
                logger.error(f"Ошибка водяного знака для {clip_id}: {e}")
                os.remove(clip_path)
                temp_files.remove(clip_path)
                continue
            try:
                c.execute("INSERT INTO clips VALUES (?, ?, ?, ?)",
                          (clip_id, clip['broadcaster_name'], platform, datetime.now(timezone.utc)))
            except sqlite3.IntegrityError as e:
                logger.warning(f"Клип {clip_id} уже использован для платформы {platform}, пропускаем: {e}")
                os.remove(watermarked_path)
                temp_files.remove(watermarked_path)
                continue
            os.remove(clip_path)
            temp_files.remove(clip_path)

        if not temp_clips:
            logger.error("Не удалось обработать ни одного клипа")
            await callback_query.message.answer("Не удалось обработать клипы. Попробуйте другую категорию или язык.")
            await state.clear()
            conn.close()
            return

        preview_path = os.path.join(os.getcwd(), f"preview_{uuid4()}.mp4")
        full_path = os.path.join(os.getcwd(), f"full_{uuid4()}.mp4")
        temp_files.append(preview_path)
        temp_files.append(full_path)
        logger.info(f"Создание видео: preview={preview_path}, full={full_path}")
        preview_duration = await create_transitioned_video(temp_clips, preview_path, platform, preview=True)
        full_duration = await create_transitioned_video(temp_clips, full_path, platform, preview=False)
        logger.info(f"Длительности после первой сборки: preview={preview_duration}, full={full_duration}")

        MIN_DURATION = 61
        if full_duration >= MIN_DURATION:
            logger.info(f"Длительность {full_duration} секунд превышает минимальную {MIN_DURATION}, автодобавление не требуется")
        else:
            attempts = 0
            max_attempts = 5
            while full_duration < MIN_DURATION and attempts < max_attempts:
                attempts += 1
                logger.info(f"Длительность {full_duration} < {MIN_DURATION}, добавляем клипы (попытка {attempts})")

                needed_duration = MIN_DURATION - full_duration
                avg_clip_duration = sum(get_clip_duration(clip) for clip in temp_clips) / len(temp_clips) if temp_clips else 10
                extra_clip_count = max(1, int(needed_duration / avg_clip_duration) + 1)
                logger.info(f"Требуется добавление {extra_clip_count} клипов для достижения {MIN_DURATION} секунд")

                for category in selected_categories:
                    new_clips = await get_twitch_clips(category, language, extra_clip_count * 2, max_attempts=attempts, search_depth_days=30 * attempts)
                    for clip in new_clips:
                        clip_id = clip['id']
                        if any(c['id'] == clip_id for c in processed_clips):
                            continue
                        c.execute("SELECT clip_id FROM clips WHERE clip_id=? AND used_for=?", (clip_id, platform))
                        if c.fetchone():
                            continue
                        clip_url = clip['url']
                        clip_path = await download_clip_with_ytdlp(clip_url, clip_id)
                        if not clip_path or not os.path.exists(clip_path):
                            continue
                        temp_files.append(clip_path)
                        blurred_path = os.path.join(os.getcwd(), f"blurred_{clip_id}.mp4")
                        blur_cmd = detect_and_blur_ads(clip_path, blurred_path)
                        if blur_cmd:
                            process = await asyncio.get_event_loop().run_in_executor(
                                None, lambda: subprocess.run(blur_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
                            if process.returncode != 0:
                                os.remove(clip_path)
                                temp_files.remove(clip_path)
                                continue
                            os.remove(clip_path)
                            temp_files.remove(clip_path)
                            clip_path = blurred_path
                            temp_files.append(clip_path)
                        watermarked_path = os.path.join(os.getcwd(), f"watermarked_{clip_id}.mp4")
                        try:
                            await add_watermark(clip_path, clip['broadcaster_name'], clip['title'], watermarked_path)
                            temp_clips.append(watermarked_path)
                            temp_files.append(watermarked_path)
                            processed_clips.append(clip)
                        except ffmpeg.Error:
                            os.remove(clip_path)
                            temp_files.remove(clip_path)
                            continue
                        try:
                            c.execute("INSERT INTO clips VALUES (?, ?, ?, ?)",
                                      (clip_id, clip['broadcaster_name'], platform, datetime.now(timezone.utc)))
                        except sqlite3.IntegrityError as e:
                            logger.warning(f"Клип {clip_id} уже использован для платформы {platform}, пропускаем: {e}")
                            os.remove(watermarked_path)
                            temp_files.remove(watermarked_path)
                            continue
                        os.remove(clip_path)
                        temp_files.remove(clip_path)

                        if os.path.exists(preview_path):
                            os.remove(preview_path)
                            temp_files.remove(preview_path)
                        if os.path.exists(full_path):
                            os.remove(full_path)
                            temp_files.remove(full_path)
                        preview_path = os.path.join(os.getcwd(), f"preview_{uuid4()}.mp4")
                        full_path = os.path.join(os.getcwd(), f"full_{uuid4()}.mp4")
                        temp_files.append(preview_path)
                        temp_files.append(full_path)
                        preview_duration = await create_transitioned_video(temp_clips, preview_path, platform, preview=True)
                        full_duration = await create_transitioned_video(temp_clips, full_path, platform, preview=False)
                        logger.info(f"Длительности после добавления клипа {clip_id}: preview={preview_duration}, full={full_duration}")

                        if full_duration >= MIN_DURATION:
                            logger.info(f"Достигнута минимальная длительность {full_duration} >= {MIN_DURATION}")
                            break
                    if full_duration >= MIN_DURATION:
                        break
                if attempts == max_attempts:
                    logger.warning(f"Не удалось достичь минимальной длительности {MIN_DURATION} после {max_attempts} попыток")
                    break

        try:
            conn.commit()
            logger.info("Изменения в базе данных успешно сохранены")
        except sqlite3.ProgrammingError as e:
            logger.error(f"Ошибка при сохранении в базе данных: {e}")
        finally:
            conn.close()
            logger.info("Соединение с базой данных закрыто")

        if full_duration < MIN_DURATION:
            logger.error(f"Не удалось достичь минимальной длительности: {full_duration}")
            await callback_query.message.answer("Не удалось создать видео длиной более 61 секунды.")
            await state.clear()
            return

        video_id = uuid4().hex[:16]
        save_video_path(video_id, full_path)
        save_video_metadata(video_id, processed_clips, categories_data, selected_categories)
        logger.info(f"Сохранены данные видео: video_id={video_id}")

        file = FSInputFile(preview_path)
        await callback_query.message.answer_video(file, caption="Предпросмотр видео")
        await callback_query.message.answer("Выберите, куда загрузить видео:",
                                            reply_markup=InlineKeyboardMarkup(
                                                inline_keyboard=[
                                                    [InlineKeyboardButton(text="TikTok",
                                                                          callback_data=f"upload_tiktok_{video_id}")],
                                                    [InlineKeyboardButton(text="YouTube",
                                                                          callback_data=f"upload_youtube_{video_id}")],
                                                    [InlineKeyboardButton(text="Удалить и пересоздать",
                                                                          callback_data="redo")]
                                                ]))
        await state.update_data(video_id=video_id)

    except Exception as e:
        logger.error(f"Ошибка в process_language: {e}", exc_info=True)
        await callback_query.message.answer(f"Произошла ошибка: {str(e)}. Попробуйте снова.")
    finally:
        for file in temp_files:
            if file and os.path.exists(file):
                try:
                    os.remove(file)
                    logger.info(f"Удален временный файл: {file}")
                except Exception as e:
                    logger.warning(f"Не удалось удалить файл {file}: {e}")
        await state.clear()




import time

@dp.callback_query(lambda c: c.data == "publish_tiktok")
async def process_publish_tiktok(callback_query: CallbackQuery):
    await callback_query.answer("Начинаю публикацию всех черновиков на TikTok. Пожалуйста, подождите...")
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(user_agent=random.choice(USER_AGENTS))

            # Загрузка и валидация куки
            cookies_path = 'tiktok_cookies.json'
            if not os.path.exists(cookies_path):
                raise ValueError("Файл tiktok_cookies.json не найден")
            with open(cookies_path, 'r') as f:
                cookies = json.load(f)
                valid_cookies = [
                    {**cookie, 'sameSite': cookie.get('sameSite', 'Lax' if cookie.get('secure', False) else 'None')}
                    for cookie in cookies if all(key in cookie for key in ['name', 'value', 'domain'])
                ]
                if not valid_cookies:
                    raise ValueError("Нет валидных cookies в tiktok_cookies.json")
                await context.add_cookies(valid_cookies)

            page = await context.new_page()
            await page.goto("https://www.tiktok.com/tiktokstudio/content", timeout=60000)

            # Проверка авторизации
            if await page.query_selector("text=Войти"):
                raise RuntimeError("Сессия не авторизована. Обновите куки.")

            # Переход в черновики
            draft_button_xpath = '//*[@id="root"]/div/div/div[2]/div[1]/div/div/div[2]/div[1]/div[2]/span/div'
            await page.wait_for_selector(f'xpath={draft_button_xpath}', state="visible", timeout=60000)
            await page.click(f'xpath={draft_button_xpath}')

            # Ожидание страницы черновиков
            await page.wait_for_url("https://www.tiktok.com/tiktokstudio/content?tab=draft", timeout=60000)
            await page.wait_for_timeout(5000)

            # Поиск реальных контейнеров черновиков (проверка наличия текста)
            drafts_xpath = '//*[@id="root"]/div/div/div[2]/div[2]/div/div/div/div[2]/div[2]/div[*]/div'
            draft_elements = await page.query_selector_all(f'xpath={drafts_xpath}')
            total_drafts = 0
            for element in draft_elements:
                text = await element.inner_text()
                if text and any(char.isdigit() for char in text):  # Проверяем наличие цифр (время или дата)
                    total_drafts += 1
            if total_drafts == 0:
                await callback_query.message.answer("Нет черновиков для публикации.")
                await browser.close()
                return

            await callback_query.message.answer(f"Найдено {total_drafts} черновиков. Начинаю публикацию...")

            published_count = 0
            for i in range(total_drafts):
                try:
                    await callback_query.message.answer(f"Публикация черновика {i+1}/{total_drafts}...")

                    # Клик на "Редактировать"
                    edit_button_xpath = '//*[@id="root"]/div/div/div[2]/div[2]/div/div/div/div[2]/div[2]/div[1]/div/div[3]/div/button[1]'
                    await page.wait_for_selector(f'xpath={edit_button_xpath}', state="visible", timeout=30000)
                    await page.click(f'xpath={edit_button_xpath}')
                    await page.wait_for_timeout(3000)

                    # Клик "Опубликовать"
                    publish_xpath = '//*[@id="root"]/div/div/div[2]/div[2]/div/div/div/div[5]/div/button[1]'
                    await page.wait_for_selector(f'xpath={publish_xpath}', state="visible", timeout=60000)
                    publish_button = await page.query_selector(f'xpath={publish_xpath}')
                    max_wait = 10
                    start_time = time.time()
                    while time.time() - start_time < max_wait:
                        if await publish_button.is_enabled():
                            break
                        await page.wait_for_timeout(1000)
                    if not await publish_button.is_enabled():
                        logger.warning(f"Кнопка 'Опубликовать' не активировалась для черновика {i+1}. Пропускаю.")
                        await page.goto("https://www.tiktok.com/tiktokstudio/content?tab=draft", timeout=60000)
                        await page.wait_for_timeout(5000)
                        continue

                    await publish_button.click()
                    await page.wait_for_timeout(5000)

                    published_count += 1
                    await page.goto("https://www.tiktok.com/tiktokstudio/content?tab=draft", timeout=60000)
                    await page.wait_for_timeout(5000)

                except Exception as inner_e:
                    logger.error(f"Ошибка при публикации черновика {i+1}: {inner_e}")
                    await page.goto("https://www.tiktok.com/tiktokstudio/content?tab=draft", timeout=60000)
                    await page.wait_for_timeout(5000)

            # Обновление куки
            cookies = await context.cookies()
            with open(cookies_path, 'w') as f:
                json.dump(cookies, f)

            await browser.close()
            await callback_query.message.answer(f"Успешно опубликовано {published_count} из {total_drafts} черновиков!")

    except Exception as e:
        if 'page' in locals():
            await browser.close()
        await callback_query.message.answer(f"Ошибка при публикации: {str(e)}.")
        logger.error(f"[TikTok Publish] Ошибка: {e}")

async def main():
    global autogeneration_running, last_autogeneration_time, autogeneration_task
    init_db()
    state_data = load_autogeneration_state()
    if state_data and state_data['running']:
        logger.info("Восстановление состояния автогенерации при запуске")
        autogeneration_running = True
        last_autogeneration_time = state_data['last_generation']
        if all(key in state_data for key in ['platform', 'selected_categories', 'language', 'clip_count', 'chat_id']):
            logger.info(f"Запуск задачи автогенерации с параметрами: {state_data}")
            autogeneration_task = asyncio.create_task(
                autogeneration_loop(
                    state_data['chat_id'],
                    state_data['platform'],
                    state_data['selected_categories'],
                    state_data['language'],
                    state_data['clip_count']
                )
            )
            logger.info("Задача автогенерации успешно восстановлена")
        else:
            logger.warning("Невалидные данные состояния автогенерации, сбрасываем")
            autogeneration_running = False
            save_autogeneration_state(
                False,
                last_autogeneration_time,
                state_data.get('platform'),
                state_data.get('selected_categories', []),
                state_data.get('language'),
                state_data.get('clip_count', 5),
                state_data.get('chat_id', 0)
            )
    else:
        logger.info("Автогенерация не была активна при запуске")
    max_retries = 5
    retry_delay = 2
    for attempt in range(max_retries):
        try:
            await dp.start_polling(bot, polling_timeout=15)
            logger.info("Успешное подключение к Telegram API")
            break
        except Exception as e:
            logger.error(f"Ошибка подключения на попытке {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Ожидание {retry_delay} секунд...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error("Достигнуто максимальное количество попыток.")
                raise

if __name__ == "__main__":
    asyncio.run(main())