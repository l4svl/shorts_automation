# Shorts Automation Bot

Telegram-бот для **автоматизации создания YouTube Shorts / TikTok видео**: скачивание клипов, удаление рекламы, нарезка, водяные знаки и публикация. Полная автоматизация контента.

![Demo](demo.gif)

## Особенности
- Скачивание с YouTube/Twitch (yt_dlp)
- Удаление рекламы (SponsorBlock API + OpenCV)
- Обработка видео: нарезка, блюр, водяные знаки (FFmpeg)
- Публикация на TikTok/YouTube (Playwright + Google API)
- Telegram-интерфейс с FSM, кнопками и автогенерацией
- База данных (SQLite) для клипов и метаданных

## Технологии
- Python 3
- `aiogram` — Telegram-бот (FSM, InlineKeyboard)
- `yt_dlp` — скачивание видео
- `ffmpeg-python` — обработка видео
- `playwright` — браузер-автоматизация
- `googleapiclient` — YouTube API
- `opencv-python` (cv2) — блюр рекламы
- `easyocr` — OCR для текста
- `sqlite3`, `asyncio`, `requests`, `json`, `logging`

## Установка и запуск
git clone https://github.com/l4svl/shorts_automation.git
cd shorts_automation
pip install -r requirements.txt
Настрой Config.py: токены (TELEGRAM_TOKEN, YOUTUBE_API_KEY, TWITCH_CLIENT_ID, TWITCH_CLIENT_SECRET, YOUTUBE_CLIENT_SECRETS)
Запустить cookies.py и пройти авторизацию
main.py

