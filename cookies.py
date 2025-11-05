import asyncio
import json
import time
from pathlib import Path

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

COOKIES_OUT = Path("tiktok_cookies.json")
TARGET_URL = "https://www.tiktok.com/tiktokstudio/upload"

REQUIRED_COOKIE_NAMES = {"sessionid", "sid_tt", "uid_tt", "tt_csrf_token"}
COOKIE_DOMAIN_FILTER = ("tiktok.com",)
LOGIN_TIMEOUT = 10 * 60  # 10 минут


def serialize_cookies(cookies):
    out = []
    for c in cookies:
        if not any(d in c.get("domain", "") for d in COOKIE_DOMAIN_FILTER):
            continue
        out.append({
            "name": c["name"],
            "value": c["value"],
            "domain": c["domain"],
            "path": c.get("path", "/"),
            "expires": c.get("expires", -1),
            "httpOnly": c.get("httpOnly", False),
            "secure": c.get("secure", True),
            "sameSite": c.get("sameSite", "Lax"),
        })
    return out


async def maybe_click_cookie_consent(page):
    candidates = [
        ('role=button[name="Accept all"]', True),
        ('role=button[name="Принять все"]', True),
        ('text="Accept all"', False),
        ('text="Принять все"', False),
    ]
    for sel, is_role in candidates:
        try:
            if is_role:
                await page.get_by_role("button", name=sel.split('name="', 1)[1][:-2]).click(timeout=3000)
            else:
                await page.locator(sel).first.click(timeout=3000)
            print("[i] Нажал на баннер согласия с cookie.")
            break
        except Exception:
            pass


async def try_autosave_cookies(context, filepath="tiktok_cookies.json"):
    cookies = await context.cookies()
    names = {c["name"] for c in cookies if any(d in c["domain"] for d in COOKIE_DOMAIN_FILTER)}

    print(f"[i] Найдено куков: {', '.join(sorted(names)) or 'ничего'}")

    if REQUIRED_COOKIE_NAMES.issubset(names):
        print("[✓] Все ключевые куки найдены — сохраняю.")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serialize_cookies(cookies), f, indent=2, ensure_ascii=False)
    else:
        print("[!] Не все ключевые куки найдены — сохранение отменено.")


async def main():
    print("[i] Запускаю браузер... (headless=False)")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, args=[
            "--disable-blink-features=AutomationControlled",
        ])
        context = await browser.new_context(
            viewport={"width": 1366, "height": 850},
            locale="en-US",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()

        print(f"[i] Открываю {TARGET_URL}")
        await page.goto(TARGET_URL, wait_until="domcontentloaded")
        await maybe_click_cookie_consent(page)

        print("\n=== ДЕЙСТВИЯ ===")
        print("1) Войди в аккаунт TikTok в открытом окне.")
        print("2) Пройди все проверки (почта/телефон/2FA).")
        print("3) Просто поброди по страницам — если ключевые куки появятся, я их сразу сохраню.\n")

        # Проверяем куки периодически
        start = time.time()
        while time.time() - start < LOGIN_TIMEOUT:
            await try_autosave_cookies(context)
            await asyncio.sleep(2.0)

        print("[!] Таймаут авторизации. Закрываю браузер.")
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
