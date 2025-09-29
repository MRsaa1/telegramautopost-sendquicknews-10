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
from telegram import Bot
import openai

# ================== CONFIG ==================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHANNEL_RU = os.getenv("TELEGRAM_CHANNEL_RU", "-1002597393191")

SEND_TO_TELEGRAM = True                 # ВКЛЮЧЕНО
MAX_CAPTION = 1024                      # лимит подписи к фото в Telegram
SIGNATURE = "С вами был ReserveOne ☕️"

LOCAL_TZ = ZoneInfo("Europe/Vienna")
FRESHNESS_HOURS_MORNING = int(os.getenv("FRESHNESS_HOURS_MORNING", "10"))
MARKET_SOURCE_MODE = os.getenv("MARKET_SOURCE_MODE", "last_close")  # last_close | intraday
FORCE_GOLD_SPOT = os.getenv("FORCE_GOLD_SPOT", "0") == "1"

# Изображение
IMAGES_DIR = "images"
os.makedirs(IMAGES_DIR, exist_ok=True)
STATIC_IMAGE_PATH = os.path.join(IMAGES_DIR, "morning_digest_static.png")
TARGET_IMAGE_HEIGHT = int(os.getenv("TARGET_IMAGE_HEIGHT", "750"))

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
    t = re.sub(r"[\[\]\(\){}“”\"'«»•·\-–—:;,.!?]", " ", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t

def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

# ================== IMAGE HELPERS ==================
def static_image_exists() -> bool:
    try:
        return os.path.exists(STATIC_IMAGE_PATH) and os.path.getsize(STATIC_IMAGE_PATH) > 1024
    except Exception:
        return False

def save_static_image(image_bytes: BytesIO) -> bool:
    try:
        temp_path = STATIC_IMAGE_PATH + ".tmp"
        with open(temp_path, "wb") as f:
            f.write(image_bytes.getvalue())
        if os.path.getsize(temp_path) > 1024:
            os.replace(temp_path, STATIC_IMAGE_PATH)
            print(f"✅ Статичное изображение сохранено: {STATIC_IMAGE_PATH}")
            return True
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False
    except Exception as e:
        print(f"❌ Ошибка сохранения изображения: {e}")
        return False

def load_static_image() -> BytesIO | None:
    try:
        if os.path.exists(STATIC_IMAGE_PATH) and os.path.getsize(STATIC_IMAGE_PATH) > 1024:
            with open(STATIC_IMAGE_PATH, "rb") as f:
                buf = BytesIO(f.read())
                buf.seek(0)
                return buf
    except Exception as e:
        print(f"❌ Ошибка загрузки изображения: {e}")
    return None

def resize_image_height(image_bytes: BytesIO, target_height: int = 750) -> BytesIO:
    try:
        img = Image.open(image_bytes)
        w, h = img.size
        if h <= target_height:
            image_bytes.seek(0)
            return image_bytes
        resized = img.resize((w, target_height), Image.Resampling.LANCZOS)
        out = BytesIO()
        resized.save(out, format="PNG")
        out.seek(0)
        return out
    except Exception as e:
        print(f"❌ Ошибка изменения высоты изображения: {e}")
        image_bytes.seek(0)
        return image_bytes

def _probe_image_size(buf: BytesIO) -> tuple[int, int]:
    try:
        pos = buf.tell()
        img = Image.open(buf)
        w, h = img.size
        buf.seek(pos)
        return w, h
    except Exception:
        buf.seek(0)
        return (0, 0)

def _placeholder_image(size=(1024, 1024)) -> BytesIO:
    """Фоллбэк-картинка, если OpenAI не дал URL."""
    img = Image.new("RGB", size, (242, 244, 248))
    d = ImageDraw.Draw(img)
    # простая композиция
    d.rectangle([60, 200, size[0]-60, size[1]-200], outline=(90, 120, 200), width=6)
    d.ellipse([140, 280, 280, 420], outline=(90, 120, 200), width=6)
    d.line([320, 340, 880, 340], fill=(90, 120, 200), width=6)
    d.line([320, 400, 880, 400], fill=(90, 120, 200), width=6)
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
        if src in host:
            score += val
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

    # дедуп по ссылке
    seen, uniq = set(), []
    for e in entries:
        lk = e.get("link") or ""
        if lk and lk not in seen:
            seen.add(lk)
            uniq.append(e)

    # анти-дубль по заголовкам
    filtered: list[dict] = []
    for e in uniq:
        if any(_similar(e["title_norm"], x["title_norm"]) > 0.92 for x in filtered):
            continue
        filtered.append(e)

    # скоринг + сортировка
    for n in filtered:
        n["score"] = score_item(n)
    filtered.sort(key=lambda x: (x["score"], -x["published_dt"].timestamp()), reverse=True)

    # баланс крипто/фин
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
    """Мягкая проверка: логируем подозрительное, но НИЧЕГО не блокируем."""
    rules = {
        "sp500": {"min": 2000, "max": 8000, "change_max": 10},
        "nasdaq": {"min": 5000, "max": 25000, "change_max": 10},
        "dxy": {"min": 80, "max": 120, "change_max": 5},
        # золото расширяем до 5000, чтобы не ругалось на ваши значения
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
    return True  # ВСЕГДА пускаем дальше

def _safe_close_pair(df):
    if df is None or df.empty or "Close" not in df.columns:
        return None
    last = df["Close"].iloc[-1]
    try:
        last_f = float(last)
    except Exception:
        return None
    if math.isnan(last_f):
        return None
    if len(df["Close"]) >= 2:
        prev = df["Close"].iloc[-2]
        try:
            prev_f = float(prev)
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
                    last_f = float(last)
                except Exception:
                    continue
                if not math.isnan(last_f):
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

# ================== IMAGE GEN ==================
async def get_morning_image() -> BytesIO | None:
    if not SEND_TO_TELEGRAM:
        print("🖼️ Генерация изображения пропущена (SEND_TO_TELEGRAM=0).")
        return None
    try:
        # 1) Кэш
        if static_image_exists():
            cached = load_static_image()
            if cached:
                w, h = _probe_image_size(cached)
                print(f"🖼️ Кэш-изображение: {w}x{h}px")
                if h > TARGET_IMAGE_HEIGHT:
                    print(f"↘️ Уменьшаем высоту кэша до {TARGET_IMAGE_HEIGHT}px…")
                    resized = resize_image_height(cached, target_height=TARGET_IMAGE_HEIGHT)
                    save_static_image(resized)
                    recached = load_static_image()
                    if recached:
                        w2, h2 = _probe_image_size(recached)
                        print(f"✅ Кэш обновлён: {w2}x{h2}px")
                        return recached
                    resized.seek(0); return resized
                cached.seek(0); return cached

        print("🎨 Генерируем новое статичное изображение…")
        img_buf = None
        if OPENAI_API_KEY:
            try:
                client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
                prompt = (
                    "Digital illustration, professional and calm, modern flat style, soft morning colors. "
                    "Minimalist finance theme: coffee cup, charts, tickers, newspaper. Clean lines, pastel palette. No text."
                )
                resp = await client.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    n=1,
                    size="1024x1024",
                )
                img_url = resp.data[0].url if resp and resp.data else None
                if img_url:
                    img_buf = BytesIO(requests.get(img_url, timeout=20).content)
            except Exception as e:
                print(f"⚠️ Ошибка OpenAI image: {e}")

        if img_buf is None:
            print("🖼️ Используем плейсхолдер-картинку (фоллбэк).")
            img_buf = _placeholder_image((1024, 1024))

        w0, h0 = _probe_image_size(img_buf)
        print(f"🖼️ Новое изображение до сжатия: {w0}x{h0}px")
        resized = resize_image_height(img_buf, target_height=TARGET_IMAGE_HEIGHT)
        w1, h1 = _probe_image_size(resized)
        print(f"🖼️ После сжатия: {w1}x{h1}px")

        if save_static_image(resized):
            recached = load_static_image()
            if recached:
                w2, h2 = _probe_image_size(recached)
                print(f"✅ Сохранено в кэш: {w2}x{h2}px")
                return recached
        resized.seek(0)
        return resized
    except Exception as e:
        print(f"❌ Ошибка генерации/сжатия изображения: {e}")
        return None

# ================== DIGEST BUILD HELPERS ==================
def sanitize_markdown(text: str) -> str:
    if not text:
        return ""
    out = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    out = re.sub(r"__(.*?)__", r"\1", out)
    out = re.sub(r"_([^_]+)_", r"\1", out)
    out = re.sub(r"`([^`]+)`", r"\1", out)
    out = re.sub(r"\s+\n", "\n", out)
    return out.strip()

SECTION_ORDER = ["1️⃣","2️⃣","3️⃣","4️⃣","5️⃣","6️⃣","7️⃣"]
SECTION_EMOJI = {"1️⃣":"📊","2️⃣":"📈","3️⃣":"🏦","4️⃣":"🧭","5️⃣":"🏢","6️⃣":"🚀","7️⃣":"🌍"}

def enforce_seven_compact_lines(draft: str) -> str:
    seen = set()
    picked = []
    for raw in (l.strip() for l in (draft or "").splitlines()):
        if not raw:
            continue
        m = re.match(r"^([1-7]️⃣)\s*(.+)$", raw)
        if not m:
            continue
        tag, rest = m.groups()
        if tag in seen:
            continue
        one = re.split(r"(?<=\.)\s", rest, maxsplit=1)[0].strip()
        if ":" not in one:
            name = {
                "1️⃣":"Глобальные рынки",
                "2️⃣":"Итоги торгов",
                "3️⃣":"Трежерис, DXY, золото, нефть",
                "4️⃣":"Монетарная политика",
                "5️⃣":"Корпоративные новости",
                "6️⃣":"Криптовалюты",
                "7️⃣":"Геополитика",
            }[tag]
            one = f"{name}: {one}"
        picked.append(f"{tag} {one}")
        seen.add(tag)
        if len(picked) == 7:
            break
    picked.sort(key=lambda s: SECTION_ORDER.index(s[:2]) if s[:2] in SECTION_ORDER else 99)
    return "\n\n".join(picked)

def decorate_digest_with_emojis(digest: str) -> str:
    if not digest:
        return ""
    lines = []
    for line in digest.splitlines():
        m = re.match(r"^([1-7]️⃣)\s*(.*)$", line.strip())
        if not m:
            continue
        tag, rest = m.groups()
        if ":" in rest:
            head, tail = rest.split(":", 1)
            tail = tail.strip()
            if not tail.startswith(SECTION_EMOJI.get(tag, "")):
                tail = f"{SECTION_EMOJI.get(tag, '')} {tail}".strip()
            lines.append(f"{tag} {head.strip()}: {tail}")
        else:
            lines.append(f"{tag} {SECTION_EMOJI.get(tag,'')} {rest}".strip())
    return "\n\n".join(lines).strip()

def rebuild_lines_with_market_data(digest: str, market_data: dict | None) -> str:
    if not market_data or not digest:
        return digest

    def _safe_val(name, fmt_val):
        if name not in market_data:
            return None
        v, c = market_data[name]
        return fmt_val(v) + f"({c:+.1f}%)"

    sp = _safe_val("sp500", lambda v: f"S&P500 ${v:.0f}")
    nd = _safe_val("nasdaq", lambda v: f"Nasdaq ${v:.0f}")
    au = _safe_val("gold",  lambda v: f"Золото ${v:.0f}")
    oi = _safe_val("oil",   lambda v: f"Нефть ${v:.0f}")
    dx = _safe_val("dxy",   lambda v: f"DXY {v:.1f}")
    tn = _safe_val("treasury", lambda v: f"10Y {v:.1f}%")

    line2_parts = [p for p in (sp, nd, oi, dx, tn) if p]  # в 2️⃣ теперь можно дать и DXY/10Y компактно
    line3_parts = [p for p in (tn, dx, au, oi) if p]

    new_lines = []
    for line in digest.splitlines():
        if line.startswith("2️⃣"):
            txt = " · ".join(line2_parts) if line2_parts else ""
            new_lines.append(f"2️⃣ Итоги торгов:\n📈 {txt}".strip())
        elif line.startswith("3️⃣"):
            txt = " · ".join(line3_parts) if line3_parts else ""
            new_lines.append(f"3️⃣ Трежерис, DXY, золото, нефть:\n🏦 {txt}".strip())
        else:
            new_lines.append(line)
    return "\n\n".join(new_lines).strip()

def build_global_mood_line(market_data: dict | None) -> str:
    if not market_data:
        return "1️⃣ Глобальные рынки:\n➖ Нейтральный фон, явного драйвера нет."
    up = 0.0; down = 0.0
    def _add(name, weight=1.0, invert=False):
        nonlocal up, down
        if name in market_data:
            _, chg = market_data[name]
            chg = -chg if invert else chg
            if chg >= 0.05: up += weight * chg
            elif chg <= -0.05: down += weight * abs(chg)
    _add("sp500", 1.0); _add("nasdaq", 1.0)
    _add("dxy", 0.7, invert=True); _add("treasury", 0.7, invert=True)
    score = up - down
    if score > 0.3:
        emoji, phrase = "📈", "Позитивный тон на ожиданиях спроса на риск."
    elif score < -0.3:
        emoji, phrase = "📉", "Негативный фон из-за сильного доллара и доходностей."
    else:
        emoji, phrase = "➖", "Нейтральный рынок без явного тренда."
    return f"1️⃣ Глобальные рынки:\n{emoji} {phrase}"

def normalize_sections_spacing(text: str) -> str:
    fixed = []
    for line in text.splitlines():
        line = re.sub(r"\s*:\s*", ": ", line, count=1)
        fixed.append(line.strip())
    return "\n\n".join([l for l in fixed if l])

def ceo_multiline_layout(digest: str) -> str:
    """Делаем для 4–7 формат «разнесённый в две строки» и чистим повторы."""
    if not digest:
        return ""
    title_by_tag = {
        "4️⃣": "Монетарная политика",
        "5️⃣": "Корпоративные новости",
        "6️⃣": "Криптовалюты",
        "7️⃣": "Геополитика",
    }
    default_emoji = {"4️⃣":"🧭","5️⃣":"🏢","6️⃣":"🚀","7️⃣":"🌍"}

    out = []
    for s in [l for l in digest.splitlines() if l.strip()]:
        if not s[:2] in ("4️⃣","5️⃣","6️⃣","7️⃣"):
            out.append(s); continue
        tag = s[:2]
        rest = s[2:].strip()
        head, body = (rest.split(":", 1) + [""])[:2]
        head = (head.strip() or title_by_tag[tag]).strip()
        body = body.strip()

        # Вытащим эмодзи, если уже стоит
        emoji = ""
        if body and body[0] in ("📊","📈","🏦","🧭","🏢","🚀","🌍","➖","📉","📈"):
            emoji, body = body[0], body[1:].lstrip()
        emoji = emoji or default_emoji[tag]

        # Удалим дубли типа «Монетарная политика — ...»
        base = title_by_tag[tag].lower()
        for sep in ("— ", "- ", ": ", " — ", " - ", " : "):
            patt = (base + sep).lower()
            if body.lower().startswith(patt):
                body = body[len(patt):].lstrip()
                break

        out.append(f"{tag} {head}:")
        out.append(f"{emoji} {body}" if body else f"{emoji}")
        out.append("")
    while out and out[-1] == "": out.pop()
    return "\n".join(out)

def enforce_len_budget(header: str, body: str, tail: str, max_len: int) -> str:
    """
    Укорачиваем длинные куски, но сохраняем блок «Криптовалюты (ТОП-5)».
    Если всё равно длинно — укорачиваем описания в 4–7.
    """
    parts = body.split("\n\n")

    def total_len(h, items, t):
        return len(h) + 2 + len("\n\n".join(items)) + 2 + len(t)

    # Мягкая усадка
    trimmed = []
    for s in parts:
        if s.startswith("💎 Криптовалюты (ТОП-5)"):
            trimmed.append(s); continue
        if len(s) > 180:
            cut = s[:175]
            last = max(cut.rfind(")"), cut.rfind("%"), cut.rfind(" "), cut.rfind("·"))
            s = (cut[:last].rstrip() if last > 120 else cut.rstrip()) + "…"
        elif len(s) > 160:
            s = s[:160].rstrip() + "…"
        trimmed.append(s)
    parts = trimmed

    # Если всё равно длинно — коротим 4–7
    while total_len(header, parts, tail) > max_len and parts:
        for i in range(len(parts)):
            if parts[i][:2] in ("4️⃣","5️⃣","6️⃣","7️⃣") and len(parts[i]) > 100:
                parts[i] = parts[i][:100].rstrip() + "…"
        if total_len(header, parts, tail) <= max_len:
            break
        # в крайнем случае — убираем самый длинный из 4–7 (но стараемся не трогать 1–3 и TOP-5)
        lens = [(i, len(x)) for i, x in enumerate(parts)
                if (x[:2] in ("4️⃣","5️⃣","6️⃣","7️⃣")) and not x.startswith("💎")]
        if lens:
            idx = max(lens, key=lambda z: z[1])[0]
            parts.pop(idx)
        else:
            break

    return f"{header}\n\n" + "\n\n".join(parts) + f"\n\n{tail}"

# ================== LLM (черновик 7 строк) ==================
async def ai_format_morning_digest_compact_final(news_list, market_data, crypto_data):
    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

    real_data = ""
    if market_data:
        if "sp500" in market_data:
            v, c = market_data["sp500"]; real_data += f"S&P500 ${v:.0f}({c:+.1f}%), "
        if "nasdaq" in market_data:
            v, c = market_data["nasdaq"]; real_data += f"Nasdaq ${v:.0f}({c:+.1f}%), "
        if "gold" in market_data:
            v, c = market_data["gold"]; real_data += f"Золото ${v:.0f}({c:+.1f}%), "
        if "oil" in market_data:
            v, c = market_data["oil"]; real_data += f"Нефть ${v:.0f}({c:+.1f}%), "
        if "dxy" in market_data:
            v, c = market_data["dxy"]; real_data += f"DXY {v:.1f}({c:+.1f}%), "
        if "treasury" in market_data:
            v, c = market_data["treasury"]; real_data += f"10Y {v:.1f}%({c:+.1f}%)"

    news_titles = "\n".join([f"- {n['title']}" for n in news_list[:8]])

    prompt = f"""
Сделай компактный «CEO morning brief» на русском (до 900 символов всего).
Строго 7 строк, каждая начинается с номера 1️⃣..7️⃣ и БЕЗ доп.строк:
1️⃣ Глобальные рынки — качественно (без цифр).
2️⃣ Итоги торгов — ТОЛЬКО из набора: {real_data}
3️⃣ Трежерис/DXY/золото/нефть — ТОЛЬКО из набора: {real_data}
4️⃣ Монетарная политика — одно короткое наблюдение.
5️⃣ Корпоративные новости — один факт (крупные эмитенты/IPO/M&A/гайд).
6️⃣ Криптовалюты — один факт (регуляторы/ликвидность/институции).
7️⃣ Геополитика — один факт с рыночной релевантностью.

Тон: Bloomberg/NYT/Reuters. Без markdown, без жирного, без эмодзи.
Одна строка на пункт. Без вторых предложений.

Факт-база (используй только это):
{news_titles}
"""

    # Если нет ключа — грубый fallback, чтобы не сорвать пост
    if not client:
        return "\n".join([
            "1️⃣ Глобальные рынки: Нейтральный тон, драйверов мало.",
            "2️⃣ Итоги торгов: индексы и нефть без резких движений.",
            "3️⃣ Трежерис, DXY, золото, нефть: без сильных сдвигов.",
            "4️⃣ Монетарная политика: ФРС/ЕЦБ сохраняют осторожность.",
            "5️⃣ Корпоративные новости: крупные эмитенты в фокусе.",
            "6️⃣ Криптовалюты: ликвидность стабильна, регуляторы активны.",
            "7️⃣ Геополитика: риски на радаре инвесторов."
        ])

    resp = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Ты — финансовый редактор для CEO. Строго, кратко, фактологично."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=420,
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()

# ================== MAIN ==================
async def send_morning_digest():
    print("🚀 Запуск утренней сводки...")

    # 1) новости
    print("📰 Получаем новости из RSS (только свежие)…")
    pool = get_feed_news(CRYPTO_FEEDS + FINANCE_FEEDS, NEWS_COUNT)
    if not pool:
        print("⚠️ Нет свежих новостей в окне актуальности")
        return
    news_list = filter_by_importance(pool, NEWS_COUNT)
    print(f"✅ Отобрано {len(news_list)} актуальных новостей (≤ {FRESHNESS_HOURS_MORNING}ч)")

    # 2) рынки/крипта
    market_data = await get_market_data()
    crypto_data = await get_crypto_data()

    # 3) черновик LLM
    print("🤖 Формируем компактную сводку…")
    digest_raw = await ai_format_morning_digest_compact_final(news_list, market_data, crypto_data)

    # 4) пост-обработка
    digest = sanitize_markdown(digest_raw)
    digest = enforce_seven_compact_lines(digest)
    digest = rebuild_lines_with_market_data(digest, market_data)
    digest = decorate_digest_with_emojis(digest)

    # 1️⃣ — качественный mood-line
    mood_line = build_global_mood_line(market_data)
    lines = digest.split("\n\n")
    if lines and lines[0].startswith("1️⃣"):
        lines[0] = mood_line
    digest = "\n\n".join(lines)
    digest = normalize_sections_spacing(digest)

    # двухстрочная верстка 4–7
    digest = ceo_multiline_layout(digest)

    # 5) шапка/подпись
    now_local = datetime.now(LOCAL_TZ)
    header = f"🌅 Утренняя сводка — {now_local:%d.%m.%Y}"
    tail = SIGNATURE

    # 6) крипто ТОП-5 — всегда добавляем
    body = digest
    if crypto_data:
        crypto_lines = ["💎 Криптовалюты (ТОП-5)"]
        if "bitcoin" in crypto_data:
            p = crypto_data["bitcoin"]["usd"]; c = crypto_data["bitcoin"]["usd_24h_change"]
            crypto_lines.append(f"BTC ${p:,.0f}({c:+.1f}%)")
        if "ethereum" in crypto_data:
            p = crypto_data["ethereum"]["usd"]; c = crypto_data["ethereum"]["usd_24h_change"]
            crypto_lines.append(f"ETH ${p:,.0f}({c:+.1f}%)")
        if "binancecoin" in crypto_data:
            p = crypto_data["binancecoin"]["usd"]; c = crypto_data["binancecoin"]["usd_24h_change"]
            crypto_lines.append(f"BNB ${p:.0f}({c:+.1f}%)")
        if "ripple" in crypto_data:
            p = crypto_data["ripple"]["usd"]; c = crypto_data["ripple"]["usd_24h_change"]
            crypto_lines.append(f"XRP ${p:.2f}({c:+.1f}%)")
        if "solana" in crypto_data:
            p = crypto_data["solana"]["usd"]; c = crypto_data["solana"]["usd_24h_change"]
            crypto_lines.append(f"SOL ${p:.0f}({c:+.1f}%)")
        body = body + "\n\n" + "\n".join(crypto_lines)

    # 7) лимит Telegram
    full_post = enforce_len_budget(header, body, tail, MAX_CAPTION)
    print(f"\n================= PREVIEW (console) =================\n{full_post}\n====================================================")
    print(f"🧮 Длина поста: {len(full_post)} символов")

    # 8) отправка (с изображением)
    if SEND_TO_TELEGRAM:
        print("📤 Отправляем в Telegram…")
        image = await get_morning_image()
        try:
            bot = Bot(token=TELEGRAM_TOKEN)
            if image:
                await bot.send_photo(
                    chat_id=TELEGRAM_CHANNEL_RU,
                    photo=image,
                    caption=full_post,
                    parse_mode=None,  # чистый текст
                )
            else:
                await bot.send_message(
                    chat_id=TELEGRAM_CHANNEL_RU,
                    text=full_post,
                    parse_mode=None,
                )
            print("✅ Пост отправлен в Telegram!")
        except Exception as e:
            print(f"❌ Ошибка отправки в Telegram: {e}")
    else:
        print("📤 Отправка в Telegram отключена (SEND_TO_TELEGRAM=False).")

    # 9) статистика
    print("\n📊 СТАТИСТИКА:")
    print(f"• Новостей: {len(news_list)}")
    print(f"• Символов: {len(full_post)}")
    print(f"• Рыночные данные: {'✅' if market_data else '❌'}")
    print(f"• Крипто-данные: {'✅' if crypto_data else '❌'}")
    print(f"• Окно свежести: {FRESHNESS_HOURS_MORNING} ч")
    print(f"• Источник рынков: {MARKET_SOURCE_MODE}")
    print(f"• Картинка: {'кэш/сгенерирована' if static_image_exists() else 'нет'}")

if __name__ == "__main__":
    asyncio.run(send_morning_digest())