#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import html
import math
import asyncio
import requests
import feedparser
from io import BytesIO
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from difflib import SequenceMatcher
from urllib.parse import urlparse

# Маркеты/крипта/картинка
import yfinance as yf
from pycoingecko import CoinGeckoAPI
from PIL import Image, ImageDraw
import aiohttp
from openai import AsyncOpenAI
from telegram import Bot

# ================== CONFIG ==================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SEND_TO_TELEGRAM = os.getenv("SEND_TO_TELEGRAM", "0") == "1"  # True для отправки
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHANNEL_RU = os.getenv("TELEGRAM_CHANNEL_RU", "-1002597393191")

# Жёсткий лимит Telegram
MAX_CAPTION = 1024
# Требуемый диапазон для подписи
TARGET_MIN = 600
TARGET_MAX = 800

SIGNATURE = "С вами был ReserveOne ☕️"
LOCAL_TZ = ZoneInfo("Europe/Vienna")
FRESHNESS_HOURS_MORNING = int(os.getenv("FRESHNESS_HOURS_MORNING", "10"))
MARKET_SOURCE_MODE = os.getenv("MARKET_SOURCE_MODE", "last_close")  # last_close | intraday
FORCE_GOLD_SPOT = os.getenv("FORCE_GOLD_SPOT", "0") == "1"

# Изображение
IMAGES_DIR = "images"
os.makedirs(IMAGES_DIR, exist_ok=True)
MORNING_IMAGE_PATH = os.path.join(IMAGES_DIR, "morning_digest.png")
TARGET_IMAGE_HEIGHT = int(os.getenv("TARGET_IMAGE_HEIGHT", "750"))
DALLE_MODEL = os.getenv("DALLE_MODEL", "dall-e-3")
DALLE_SIZE = os.getenv("DALLE_SIZE", "1024x1024")

# Новости
NEWS_COUNT = 15
CRYPTO_RATIO = float(os.getenv("CRYPTO_RATIO", "0.4"))
MIN_FIN_NEWS = int(os.getenv("MIN_FIN_NEWS", "6"))
CRYPTO_KEYWORD_BONUS = int(os.getenv("CRYPTO_KEYWORD_BONUS", "50"))

CRYPTO_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://decrypt.co/feed",
    "https://theblock.co/rss",
]
FINANCE_FEEDS = [
    "https://www.bloomberg.com/feed/podcast/etf-report.xml",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "https://www.reuters.com/finance/rss",
    "https://www.marketwatch.com/rss/topstories",
    "https://www.kitco.com/rss/",
    "https://www.cnbc.com/id/15839135/device/rss/rss.html",
    "https://www.investing.com/rss/news_301.rss",
    "https://www.investing.com/rss/news_25.rss",
    "https://www.morningbrew.com/feed.xml",
]

# OpenAI клиент
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ================== UTILS ==================
def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

def _strip_tags(s: str) -> str:
    return re.sub(r"<.*?>", "", s or "").strip()

def _unescape_then_escape(s: str) -> str:
    s = html.unescape(s or "")
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return s

def _norm_title(t: str) -> str:
    t = _strip_tags(t or "")
    t = re.sub(r"[\[\]\(\){}""\"'«»•·\-–—:;,.!?]", " ", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t

def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

# ================== IMAGE GENERATION ==================
def resize_image_height(image_bytes: BytesIO, target_height: int = 750) -> BytesIO:
    """
    Изменяет высоту изображения до target_height, сохраняя ширину.
    1024×1024 → 1024×750
    """
    try:
        img = Image.open(image_bytes)
        w, h = img.size
        if h <= target_height:
            image_bytes.seek(0)
            return image_bytes
        # Сохраняем ширину, меняем только высоту
        resized = img.resize((w, target_height), Image.Resampling.LANCZOS)
        out = BytesIO()
        resized.save(out, format="PNG")
        out.seek(0)
        print(f"✅ Изображение изменено: {w}×{h} → {w}×{target_height}")
        return out
    except Exception as e:
        print(f"❌ Ошибка изменения высоты изображения: {e}")
        image_bytes.seek(0)
        return image_bytes

def build_image_prompt_from_market_data(market_data: dict, crypto_data: dict | None) -> str:
    """
    Создаёт промпт для DALL·E на основе рыночных данных.
    """
    components = []

    # Анализ фондовых рынков
    sp_change = market_data.get("sp500", (None, 0))[1]
    nasdaq_change = market_data.get("nasdaq", (None, 0))[1]
    avg_change = (sp_change + nasdaq_change) / 2

    if avg_change > 1:
        components.append("stock markets rising bullish")
    elif avg_change < -1:
        components.append("stock markets declining bearish")
    else:
        components.append("stock markets stable neutral")

    # Золото
    gold_change = market_data.get("gold", (None, 0))[1]
    if gold_change > 1:
        components.append("gold prices up")
    elif gold_change < -1:
        components.append("gold prices down")

    # Нефть
    oil_change = market_data.get("oil", (None, 0))[1]
    if abs(oil_change) > 2:
        components.append("oil volatility")

    # Крипто
    if crypto_data and "bitcoin" in crypto_data:
        btc_change = crypto_data["bitcoin"].get("usd_24h_change", 0)
        if btc_change > 3:
            components.append("cryptocurrency rally bitcoin")
        elif btc_change < -3:
            components.append("crypto market correction")

    prompt_base = " and ".join(components[:3]) if components else "global financial markets overview"
    return f"Morning financial digest: {prompt_base}"

async def ai_generate_image(prompt: str) -> BytesIO | None:
    """
    Генерирует изображение через DALL·E 3.
    Размер: 1024×1024 → ресайз до 1024×750
    """
    if not openai_client:
        print("⚠️ OpenAI клиент не инициализирован (отсутствует API ключ)")
        return None

    try:
        print(f"🎨 Генерируем изображение через DALL·E 3...")
        print(f"   Промпт: {prompt}")

        img_prompt = (
            f"Digital illustration for a morning finance/crypto digest: '{prompt}'. "
            "Professional but friendly style, modern flat design, soft pastel colors, "
            "abstract financial symbols, charts, coins, NO text or numbers on image."
        )

        resp = await openai_client.images.generate(
            model=DALLE_MODEL,
            prompt=img_prompt,
            n=1,
            size=DALLE_SIZE,
            quality="standard",
        )

        img_url = resp.data[0].url
        print(f"✅ Изображение сгенерировано, скачиваем...")

        # Скачиваем изображение
        async with aiohttp.ClientSession() as session:
            async with session.get(img_url) as r:
                r.raise_for_status()
                buf = BytesIO(await r.read())

        # Ресайз: 1024×1024 → 1024×750
        buf = resize_image_height(buf, TARGET_IMAGE_HEIGHT)

        # Сохраняем локально для логов
        try:
            with open(MORNING_IMAGE_PATH, "wb") as f:
                f.write(buf.getvalue())
            print(f"✅ Изображение сохранено: {MORNING_IMAGE_PATH}")
        except Exception as e:
            print(f"⚠️ Не удалось сохранить изображение локально: {e}")

        buf.seek(0)
        return buf

    except Exception as e:
        print(f"❌ Ошибка генерации изображения: {e}")
        return None

def _placeholder_image(size=(1024, 750)) -> BytesIO:
    """
    Создаёт placeholder изображение, если генерация не удалась.
    """
    img = Image.new("RGB", size, (242, 244, 248))
    d = ImageDraw.Draw(img)

    # Рамка
    d.rectangle([40, 150, size[0]-40, size[1]-150], outline=(90, 120, 200), width=8)

    # Иконка графика
    d.ellipse([100, 220, 200, 320], outline=(90, 120, 200), width=6)
    d.line([250, 270, 900, 270], fill=(90, 120, 200), width=6)
    d.line([250, 320, 900, 320], fill=(90, 120, 200), width=6)
    d.line([250, 370, 900, 370], fill=(90, 120, 200), width=6)

    # "График"
    points = [(300, 450), (400, 380), (500, 420), (600, 350), (700, 390), (800, 330)]
    for i in range(len(points) - 1):
        d.line([points[i], points[i + 1]], fill=(90, 120, 200), width=5)

    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

# ================== NEWS ==================
def is_quality_news(title: str, summary: str) -> bool:
    text = f"{title} {summary}".lower()
    if len(title) < 10 or len(summary) < 20:
        return False
    spam = ["click here","subscribe","newsletter","advertisement","sponsored","promotion","подпишитесь","реклама"]
    if any(k in text for k in spam):
        return False
    has_numbers = bool(re.search(r"\$\s?\d[\d.,]*|\d+%|\d+\.\d+", text))
    has_event = any(w in text for w in [
        "earnings","revenue","profit","loss","ipo","merger","acquisition",
        "rate","inflation","gdp","unemployment","fed","ecb","sec",
        "bitcoin","ethereum","crypto","stock","market","выручка","прибыль","ставка","инфляция","биткоин","акции"
    ])
    return has_numbers or has_event

def score_item(n: dict) -> int:
    score = 0
    text = f"{n.get('title','')} {n.get('summary','')}".lower()
    if re.search(r"\b(биткоин|bitcoin|btc|ethereum|eth|crypto|blockchain)\b", text):
        score += CRYPTO_KEYWORD_BONUS
    if re.search(r"\$\s?\d[\d.,]*|\d+%|\d+\.\d+", text):
        score += 40
    host = (urlparse(n.get("link") or "").hostname or "").lower()
    for src, val in {
        "bloomberg.com": 20,"reuters.com": 20,"cnbc.com": 15,"coindesk.com": 15,
        "cointelegraph.com": 12,"marketwatch.com": 10,"investing.com": 8,"morningbrew.com": 4,
    }.items():
        if src in host: score += val
    age = float(n.get("age_hours") or 999.0)
    score -= int(age ** 1.1)
    return score

def get_feed_news(feeds: list[str], max_news: int) -> list[dict]:
    entries: list[dict] = []
    now = _utcnow()
    for url in feeds:
        try:
            d = feedparser.parse(url)
            if not d.entries:
                continue
            for entry in d.entries:
                st = entry.get("published_parsed") or entry.get("updated_parsed")
                if not st:
                    continue
                published_dt = datetime(*st[:6], tzinfo=timezone.utc)
                age_hours = (now - published_dt).total_seconds() / 3600.0
                if age_hours > FRESHNESS_HOURS_MORNING:
                    continue
                title = _unescape_then_escape(_strip_tags(entry.get("title") or ""))
                summary = _unescape_then_escape(_strip_tags(entry.get("summary") or ""))
                link = (entry.get("link") or "").strip()
                if not (title or summary):
                    continue
                if not is_quality_news(title, summary):
                    continue
                item = {
                    "title": title,
                    "title_norm": _norm_title(title),
                    "summary": summary,
                    "link": link,
                    "source": url,
                    "published_dt": published_dt,
                    "age_hours": age_hours,
                }
                entries.append(item)
        except Exception as e:
            print(f"⚠️ Feed error ({url}): {e}")

    if not entries:
        return []

    seen, uniq = set(), []
    for e in entries:
        lk = e.get("link") or ""
        if lk and lk not in seen:
            seen.add(lk)
            uniq.append(e)

    filtered: list[dict] = []
    for e in uniq:
        if any(_similar(e["title_norm"], x["title_norm"]) > 0.92 for x in filtered):
            continue
        filtered.append(e)

    for n in filtered:
        n["score"] = score_item(n)
    filtered.sort(key=lambda x: (x["score"], -x["published_dt"].timestamp()), reverse=True)

    def is_crypto_item(it: dict) -> bool:
        t = (it.get("title","") + " " + it.get("summary","")).lower()
        return bool(re.search(r"\b(bitcoin|btc|ethereum|eth|crypto)\b", t))

    crypto_items = [n for n in filtered if is_crypto_item(n)]
    fin_items = [n for n in filtered if not is_crypto_item(n)]

    max_crypto = max(1, int(NEWS_COUNT * CRYPTO_RATIO))
    balanced = crypto_items[:max_crypto] + fin_items
    balanced.sort(key=lambda x: (x["score"], -x["published_dt"].timestamp()), reverse=True)

    return balanced[: max_news * 2]

def filter_by_importance(news_list: list[dict], take: int) -> list[dict]:
    def is_crypto_item(it: dict) -> bool:
        t = (it.get("title","") + " " + it.get("summary","")).lower()
        return bool(re.search(r"\b(bitcoin|btc|ethereum|eth|crypto)\b", t))
    crypto = [n for n in news_list if is_crypto_item(n)]
    fin = [n for n in news_list if not is_crypto_item(n)]
    out = fin[:MIN_FIN_NEWS] + crypto
    if len(out) < take:
        out += [n for n in fin[MIN_FIN_NEWS:]]
    return out[:take]

# ================== MARKET DATA ==================
def soft_validate_market_data(data_type, current_value, change_percent) -> bool:
    rules = {
        "sp500": {"min": 2000, "max": 8000, "change_max": 10},
        "nasdaq": {"min": 5000, "max": 25000, "change_max": 10},
        "dxy": {"min": 80, "max": 120, "change_max": 5},
        "gold": {"min": 1000, "max": 5000, "change_max": 8},
        "oil": {"min": 20, "max": 150, "change_max": 15},
        "treasury": {"min": 0, "max": 10, "change_max": 2},
    }
    r = rules.get(data_type)
    if not r:
        return True
    ok_range = r["min"] <= current_value <= r["max"]
    ok_move = abs(change_percent) <= r["change_max"]
    if not ok_range or not ok_move:
        print(f"⚠️ Подозрительное значение {data_type.upper()}: {current_value} ({change_percent:+.2f}%) — публикуем как есть.")
    return True

def _float_series_value(x):
    # фикс FutureWarning: float(Series) -> float(ser.iloc[0])
    try:
        return float(x.iloc[0])
    except AttributeError:
        return float(x)

def _safe_close_pair(df):
    if df is None or df.empty or "Close" not in df.columns:
        return None
    last = df["Close"].iloc[-1]
    try:
        last_f = _float_series_value(last)
    except Exception:
        return None
    if math.isnan(last_f):
        return None
    if len(df["Close"]) >= 2:
        prev = df["Close"].iloc[-2]
        try:
            prev_f = _float_series_value(prev)
        except Exception:
            prev_f = last_f
    else:
        prev_f = last_f
    return last_f, prev_f

def _pair_last_close(df):
    pair = _safe_close_pair(df)
    if not pair: return None
    cur, prev = pair
    chg = (cur - prev) / prev * 100 if prev else 0.0
    return cur, chg

def _pair_intraday(df):
    return _pair_last_close(df)

def _yf_download_first_ok(tickers: list[str], period="2d", interval="1d"):
    for t in tickers:
        try:
            df = yf.download(t, period=period, interval=interval, auto_adjust=False, progress=False)
            if df is not None and not df.empty and "Close" in df.columns:
                last = df["Close"].iloc[-1]
                try:
                    _ = _float_series_value(last)
                except Exception:
                    continue
                return df
        except Exception as e:
            print(f"⚠️ YF error for {t}: {e}")
    return None

async def get_market_data():
    try:
        print("📊 Получаем рыночные данные...")
        sp500 = yf.download("^GSPC", period="2d", interval="1d", auto_adjust=False, progress=False)
        nasdaq = yf.download("^IXIC", period="2d", interval="1d", auto_adjust=False, progress=False)
        dxy = yf.download("DX-Y.NYB", period="2d", interval="1d", auto_adjust=False, progress=False)
        oil = yf.download("BZ=F", period="2d", interval="1d", auto_adjust=False, progress=False)
        tnx = yf.download("^TNX", period="2d", interval="1d", auto_adjust=False, progress=False)

        if FORCE_GOLD_SPOT:
            gold_df = _yf_download_first_ok(["XAUUSD=X", "GC=F", "MGC=F"], period="2d", interval="1d")
        else:
            gold_df = _yf_download_first_ok(["GC=F", "XAUUSD=X", "MGC=F"], period="2d", interval="1d")

        market_data = {}
        _pair = _pair_last_close if MARKET_SOURCE_MODE == "last_close" else _pair_intraday

        def _put(name, df):
            pair = _pair(df)
            if pair:
                cur, chg = pair
                if soft_validate_market_data(name, cur, chg):
                    market_data[name] = (cur, chg)

        _put("sp500", sp500)
        _put("nasdaq", nasdaq)
        _put("dxy", dxy)
        _put("oil", oil)
        _put("treasury", tnx)
        if gold_df is not None:
            _put("gold", gold_df)
        else:
            print("⚠️ Не удалось получить данные по золоту (GC=F/XAUUSD=X/MGC=F). Публикуем без золота.")

        for k, (v, c) in market_data.items():
            unit = "$" if k in ("sp500", "nasdaq", "gold", "oil") else ""
            print(f"✅ {k.upper()}: {unit}{v:.2f} ({c:+.2f}%)")

        return market_data
    except Exception as e:
        print(f"❌ Error fetching market data: {e}")
        return None

async def get_crypto_data():
    try:
        print("💰 Получаем данные криптовалют...")
        cg = CoinGeckoAPI()
        data = cg.get_price(
            ids="bitcoin,ethereum,binancecoin,ripple,solana",
            vs_currencies="usd",
            include_24hr_change=True,
        )
        if data and "bitcoin" in data:
            print(f"✅ BTC: ${data['bitcoin']['usd']:,.0f} ({data['bitcoin']['usd_24h_change']:+.2f}%)")
        return data
    except Exception as e:
        print(f"❌ Error fetching crypto data: {e}")
        return None

# ================== ФОРМАТИРОВАНИЕ ==================
def fmt_pct(x: float) -> str:
    return f"{x:+.1f}%"

def fmt_dollar_int(x: float) -> str:
    return f"${x:,.0f}".replace(",", " ")

def generate_market_summary(market_data: dict) -> str:
    """Генерирует краткое описание состояния рынков на основе данных"""
    if not market_data:
        return "Нейтральный тон"

    sp_change = market_data.get("sp500", (None, 0))[1]
    nasdaq_change = market_data.get("nasdaq", (None, 0))[1]

    # Определяем общий тон рынков
    avg_change = (sp_change + nasdaq_change) / 2

    if avg_change > 1.5:
        tone = "Позитивный тон на американских площадках"
    elif avg_change > 0.5:
        tone = "Умеренный рост на фондовых рынках"
    elif avg_change > -0.5:
        tone = "Нейтральный тон, рынки в боковике"
    elif avg_change > -1.5:
        tone = "Осторожность инвесторов, небольшое снижение"
    else:
        tone = "Коррекция на фондовых рынках"

    return tone

def build_visual_digest(market_data: dict, crypto_data: dict | None) -> str:
    """Формирует визуальный дайджест с рыночными данными"""

    # 1️⃣ Глобальные рынки
    market_summary = generate_market_summary(market_data)
    line1 = f"1️⃣ Глобальные рынки:\n📈 {market_summary}"

    # 2️⃣ Итоги торгов
    sp, sp_c = market_data.get("sp500", (None, None))
    nd, nd_c = market_data.get("nasdaq", (None, None))
    oi, oi_c = market_data.get("oil", (None, None))

    parts2 = []
    if sp is not None and sp_c is not None:
        parts2.append(f"S&P500 {fmt_dollar_int(sp)}({fmt_pct(sp_c)})")
    if nd is not None and nd_c is not None:
        parts2.append(f"Nasdaq {fmt_dollar_int(nd)}({fmt_pct(nd_c)})")
    if oi is not None and oi_c is not None:
        parts2.append(f"Нефть {fmt_dollar_int(oi)}({fmt_pct(oi_c)})")

    line2 = "2️⃣ Итоги торгов:\n📈 " + "\n· ".join(parts2)

    # 3️⃣ Трежерис, DXY, золото, нефть
    tn, tn_c = market_data.get("treasury", (None, None))
    dx, dx_c = market_data.get("dxy", (None, None))
    au, au_c = market_data.get("gold", (None, None))

    parts3 = []
    if tn is not None and tn_c is not None:
        parts3.append(f"10Y {tn:.1f}%({fmt_pct(tn_c)})")
    if dx is not None and dx_c is not None:
        parts3.append(f"DXY {dx:.1f}({fmt_pct(dx_c)})")
    if oi is not None and oi_c is not None:
        parts3.append(f"Нефть {fmt_dollar_int(oi)}({fmt_pct(oi_c)})")
    if au is not None and au_c is not None:
        parts3.append(f"Золото {fmt_dollar_int(au)}({fmt_pct(au_c)})")

    line3 = "3️⃣ Трежерис, DXY, золото, нефть:\n🏦 " + "\n· ".join(parts3)

    # 💎 Криптовалюты ТОП-5
    crypto_lines = ["💎 Криптовалюты (ТОП-5)"]

    if crypto_data:
        def add_crypto(coin_id, label, two_dec=False):
            if coin_id in crypto_data:
                price = crypto_data[coin_id]["usd"]
                change = crypto_data[coin_id]["usd_24h_change"]
                if two_dec:
                    price_str = f"${price:.2f}"
                else:
                    price_str = f"${price:,.0f}".replace(",", " ")
                crypto_lines.append(f"{label} {price_str}({change:+.1f}%)")

        add_crypto("bitcoin", "BTC")
        add_crypto("ethereum", "ETH")
        add_crypto("binancecoin", "BNB")
        add_crypto("ripple", "XRP", two_dec=True)
        add_crypto("solana", "SOL")

    line_crypto = "\n".join(crypto_lines)

    # Собираем всё вместе
    blocks = [line1, "", line2, "", line3, "", line_crypto]
    return "\n".join(blocks).strip()

# ================== MAIN ==================
async def send_morning_digest():
    print("🚀 Запуск утренней сводки...")

    # 1) новости (для статистики, но не включаем в пост)
    print("📰 Получаем новости из RSS (только свежие)…")
    pool = get_feed_news(CRYPTO_FEEDS + FINANCE_FEEDS, NEWS_COUNT)
    if pool:
        news_list = filter_by_importance(pool, NEWS_COUNT)
        print(f"✅ Отобрано {len(news_list)} актуальных новостей (≤ {FRESHNESS_HOURS_MORNING}ч)")
    else:
        news_list = []
        print("⚠️ Нет свежих новостей в окне актуальности")

    # 2) рынки/крипта
    market_data = await get_market_data()
    crypto_data = await get_crypto_data()

    if not market_data:
        print("❌ Не удалось получить рыночные данные. Отмена.")
        return

    # 3) формируем пост
    print("📝 Формируем утреннюю сводку…")
    now_local = datetime.now(LOCAL_TZ)
    header = f"🌅 Утренняя сводка - {now_local:%d.%m.%Y}"

    body = build_visual_digest(market_data, crypto_data)

    full_post = f"{header}\n\n{body}\n\n{SIGNATURE}"

    # 4) Генерируем изображение через DALL·E 3
    image = None
    if openai_client:
        img_prompt = build_image_prompt_from_market_data(market_data, crypto_data)
        image = await ai_generate_image(img_prompt)

        # Если не удалось сгенерировать, используем placeholder
        if image is None:
            print("⚠️ Используем placeholder изображение")
            image = _placeholder_image()
    else:
        print("⚠️ OpenAI API недоступен, изображение не будет сгенерировано")

    # ====== ОТПРАВКА В TELEGRAM ======
    if SEND_TO_TELEGRAM and TELEGRAM_TOKEN:
        try:
            bot = Bot(token=TELEGRAM_TOKEN)
            print(f"📤 Отправляем в Telegram канал {TELEGRAM_CHANNEL_RU}...")

            if image is not None:
                # Отправка с изображением
                caption_for_photo = full_post[:MAX_CAPTION]  # лимит 1024
                await bot.send_photo(
                    chat_id=TELEGRAM_CHANNEL_RU,
                    photo=image,
                    caption=caption_for_photo,
                    parse_mode=None,  # текст без HTML разметки
                )
                print("✅ Пост с изображением отправлен в Telegram!")
            else:
                # Отправка без изображения
                await bot.send_message(
                    chat_id=TELEGRAM_CHANNEL_RU,
                    text=full_post,
                    parse_mode=None,
                )
                print("✅ Пост отправлен в Telegram (без изображения)!")

        except Exception as e:
            print(f"❌ Ошибка отправки в Telegram: {e}")
    else:
        print("ℹ️ Отправка в Telegram отключена (SEND_TO_TELEGRAM=0)")

    # ====== PREVIEW В КОНСОЛИ ======
    print(f"\n================= PREVIEW (console) =================")
    print(full_post)
    print(f"=====================================================")
    print(f"🧮 Длина поста: {len(full_post)} символов (лимит для фото {MAX_CAPTION})")

    # Статистика
    print("\n📊 СТАТИСТИКА:")
    print(f"• Новостей отслежено: {len(news_list)}")
    print(f"• Символов (итог): {len(full_post)}")
    print(f"• Рыночные данные: {'✅' if market_data else '❌'}")
    print(f"• Крипто-данные: {'✅' if crypto_data else '❌'}")
    print(f"• Изображение: {'✅ сгенерировано' if image else '❌ нет'}")
    print(f"• Окно свежести: {FRESHNESS_HOURS_MORNING} ч")
    print(f"• Источник рынков: {MARKET_SOURCE_MODE}")
    print(f"• Отправка в TG: {'✅ включена' if SEND_TO_TELEGRAM else '❌ выключена'}")

if __name__ == "__main__":
    asyncio.run(send_morning_digest())