#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import html
import asyncio
import feedparser
import openai
from telegram import Bot
import requests
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from io import BytesIO
from difflib import SequenceMatcher
from urllib.parse import urlparse

import yfinance as yf
from pycoingecko import CoinGeckoAPI
from PIL import Image  # Pillow для автосжатия

# ===== CONFIG =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHANNEL_RU = "-1002597393191"

NEWS_COUNT = 15
SIGNATURE = "С вами был ReserveOne ☕️"

# Таймзона и «утреннее окно свежести»
LOCAL_TZ = ZoneInfo("Europe/Vienna")
FRESHNESS_HOURS_MORNING = int(os.getenv("FRESHNESS_HOURS_MORNING", "18"))  # только последние 18ч

# Источник рыночных данных: "last_close" (вчера vs позавчера) или "intraday"
MARKET_SOURCE_MODE = os.getenv("MARKET_SOURCE_MODE", "last_close")  # last_close | intraday

# Изображение (кешируем и уменьшаем высоту)
IMAGES_DIR = "images"
os.makedirs(IMAGES_DIR, exist_ok=True)
STATIC_IMAGE_PATH = os.path.join(IMAGES_DIR, "morning_digest_static.png")
TARGET_IMAGE_HEIGHT = int(os.getenv("TARGET_IMAGE_HEIGHT", "750"))

# Баланс крипто/финансов (при желании — отрегулировать через env)
CRYPTO_RATIO = float(os.getenv("CRYPTO_RATIO", "0.4"))          # целевая доля крипто в пуле
MIN_FIN_NEWS = int(os.getenv("MIN_FIN_NEWS", "6"))              # минимум финновостей
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

# ===== IMAGE UTILS =====
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

# ===== NEWS FRESHNESS/SCORING =====
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
    score -= int(age ** 1.1)  # экспоненциальный штраф по возрасту
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

    # анти-дубль по заголовкам (семантика)
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
    # гарантируем минимум финновостей
    def is_crypto_item(it: dict) -> bool:
        t = (it.get("title","") + " " + it.get("summary","")).lower()
        return bool(re.search(r"\b(bitcoin|btc|ethereum|eth|crypto)\b", t))
    crypto = [n for n in news_list if is_crypto_item(n)]
    fin = [n for n in news_list if not is_crypto_item(n)]
    out = fin[:MIN_FIN_NEWS] + crypto
    if len(out) < take:
        out += [n for n in fin[MIN_FIN_NEWS:]]  # добираем финновостями
    return out[:take]

# ===== MARKET DATA =====
def validate_market_data(data_type, current_value, change_percent) -> bool:
    rules = {
        "sp500": {"min": 2000, "max": 8000, "change_max": 10},
        "nasdaq": {"min": 5000, "max": 25000, "change_max": 10},
        "dxy": {"min": 80, "max": 120, "change_max": 5},
        "gold": {"min": 1000, "max": 3000, "change_max": 8},
        "oil": {"min": 20, "max": 150, "change_max": 15},
        "treasury": {"min": 0, "max": 10, "change_max": 2},
    }
    if data_type not in rules:
        return True
    r = rules[data_type]
    if not (r["min"] <= current_value <= r["max"]):
        print(f"⚠️ Подозрительное значение {data_type}: {current_value}")
        return False
    if abs(change_percent) > r["change_max"]:
        print(f"⚠️ Подозрительное изменение {data_type}: {change_percent:.2f}%")
        return False
    return True

def _pair_last_close(df):
    # вчерашний close vs позавчерашний — идеально для утреннего поста
    if df.empty or len(df) < 2:
        return None
    cur = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2])
    chg = (cur - prev) / prev * 100
    return cur, chg

def _pair_intraday(df):
    # последний Close vs предыдущий Close (если нужен более «онлайновый» вид)
    if df.empty:
        return None
    if len(df) >= 2:
        cur = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2])
    else:
        cur = float(df["Close"].iloc[-1])
        prev = cur
    chg = (cur - prev) / prev * 100 if prev else 0.0
    return cur, chg

async def get_market_data():
    try:
        print("📊 Получаем рыночные данные...")
        sp500 = yf.download("^GSPC", period="2d", interval="1d", auto_adjust=False)
        nasdaq = yf.download("^IXIC", period="2d", interval="1d", auto_adjust=False)
        dxy = yf.download("DX-Y.NYB", period="2d", interval="1d", auto_adjust=False)
        gold = yf.download("GC=F", period="2d", interval="1d", auto_adjust=False)
        oil = yf.download("BZ=F", period="2d", interval="1d", auto_adjust=False)
        tnx = yf.download("^TNX", period="2d", interval="1d", auto_adjust=False)

        market_data = {}
        _pair = _pair_last_close if MARKET_SOURCE_MODE == "last_close" else _pair_intraday

        sp = _pair(sp500)
        if sp and validate_market_data("sp500", sp[0], sp[1]):
            market_data["sp500"] = sp

        ndq = _pair(nasdaq)
        if ndq and validate_market_data("nasdaq", ndq[0], ndq[1]):
            market_data["nasdaq"] = ndq

        dx = _pair(dxy)
        if dx and validate_market_data("dxy", dx[0], dx[1]):
            market_data["dxy"] = dx

        au = _pair(gold)
        if au and validate_market_data("gold", au[0], au[1]):
            market_data["gold"] = au

        br = _pair(oil)
        if br and validate_market_data("oil", br[0], br[1]):
            market_data["oil"] = br

        tn = _pair(tnx)
        if tn and validate_market_data("treasury", tn[0], tn[1]):
            market_data["treasury"] = tn

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
        if data:
            if "bitcoin" in data:
                print(f"✅ BTC: ${data['bitcoin']['usd']:,.0f} ({data['bitcoin']['usd_24h_change']:+.2f}%)")
            if "ethereum" in data:
                print(f"✅ ETH: ${data['ethereum']['usd']:,.0f} ({data['ethereum']['usd_24h_change']:+.2f}%)")
        return data
    except Exception as e:
        print(f"❌ Error fetching crypto data: {e}")
        return None

# ===== IMAGE GEN (1024x1024 → уменьшение высоты) =====
async def get_morning_image() -> BytesIO | None:
    if static_image_exists():
        print("✅ Используем сохранённое статичное изображение")
        cached = load_static_image()
        if cached:
            return cached
        print("⚠️ Кэш повреждён — генерирую новое…")

    print("🎨 Генерируем новое статичное изображение…")
    try:
        client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
        prompt = (
            "Digital illustration, fun but professional, modern flat style, soft colors. "
            "Minimalist morning finance theme: coffee cup, newspaper icons, coins, charts. "
            "Clean lines, soft pastel palette. No text."
        )
        resp = await client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024",
        )
        img_url = resp.data[0].url
        buf = BytesIO(requests.get(img_url, timeout=20).content)

        resized = resize_image_height(buf, target_height=TARGET_IMAGE_HEIGHT)

        if save_static_image(resized):
            cached = load_static_image()
            if cached:
                return cached
        resized.seek(0)
        return resized
    except Exception as e:
        print(f"❌ Ошибка генерации изображения: {e}")
        return None

# ===== DIGEST BUILD HELPERS (пост-обработка стиля) =====
def sanitize_markdown(text: str) -> str:
    """Убираем **жирный**, __подчёркивание__ и прочий markdown/лишние пробелы."""
    if not text:
        return ""
    out = re.sub(r"\*\*(.*?)\*\*", r"\1", text)      # **bold**
    out = re.sub(r"__(.*?)__", r"\1", out)           # __underline__
    out = re.sub(r"_([^_]+)_", r"\1", out)           # _italic_
    out = re.sub(r"`([^`]+)`", r"\1", out)           # `code`
    out = re.sub(r"\s+\n", "\n", out)
    return out.strip()

def decorate_digest_with_emojis(digest: str) -> str:
    """Эмодзи для пунктов 1–7, если их нет."""
    if not digest:
        return ""
    mapping = {
        "1️⃣": "📊",  # Глобальные рынки (но 1️⃣ мы перезапишем mood-лайном)
        "2️⃣": "📈",
        "3️⃣": "🏦",
        "4️⃣": "🧭",
        "5️⃣": "🏢",
        "6️⃣": "🚀",
        "7️⃣": "🌍",
    }
    lines = digest.splitlines()
    out = []
    for line in lines:
        m = re.match(r"^([1-7]️⃣)\s*(.*)$", line.strip())
        if m:
            num, rest = m.groups()
            if ":" in rest:
                head, tail = rest.split(":", 1)
                tail = tail.strip()
                if not tail.startswith(tuple(mapping.values())):
                    tail = f"{mapping.get(num, '')} {tail}".strip()
                out.append(f"{num} {head.strip()}: {tail}")
            else:
                out.append(f"{num} {mapping.get(num,'')} {rest}".strip())
        else:
            out.append(line)
    return "\n".join(out).strip()

def rebuild_lines_with_market_data(digest: str, market_data: dict | None) -> str:
    """Гарантируем точность в пунктах 2 и 3 — строим их из реальных данных."""
    if not market_data or not digest:
        return digest

    def val(name, fmt_val):
        if name not in market_data:
            return None
        v, c = market_data[name]
        return fmt_val(v) + f"({c:+.1f}%)"

    sp = val("sp500", lambda v: f"S&P500 ${v:.0f}")
    nd = val("nasdaq", lambda v: f"Nasdaq ${v:.0f}")
    au = val("gold",  lambda v: f"Золото ${v:.0f}")
    oi = val("oil",   lambda v: f"Нефть ${v:.0f}")
    dx = val("dxy",   lambda v: f"DXY {v:.1f}")
    tn = val("treasury", lambda v: f"10Y {v:.1f}%")

    line2_parts = [p for p in (sp, nd, oi) if p]
    line3_parts = [p for p in (tn, dx, au, oi) if p]

    lines = digest.splitlines()
    new_lines = []
    for line in lines:
        if line.startswith("2️⃣"):
            txt = " · ".join(line2_parts) if line2_parts else ""
            new_lines.append(f"2️⃣ Итоги торгов: {txt}".strip())
        elif line.startswith("3️⃣"):
            txt = " · ".join(line3_parts) if line3_parts else ""
            new_lines.append(f"3️⃣ Трежерис, DXY, золото, нефть: {txt}".strip())
        else:
            new_lines.append(line)
    return "\n".join(new_lines).strip()

def build_global_mood_line(market_data: dict | None) -> str:
    """
    Делает качественную строку '1️⃣ Глобальные рынки: <эмодзи> ...'
    Логика:
      - считаем вклад S&P500 и Nasdaq (плюс), DXY и 10Y (минус для риска);
      - подбираем эмодзи и короткую фразу.
    """
    if not market_data:
        return "1️⃣ Глобальные рынки: ➖ Нейтральный фон, явного драйвера нет."

    up = 0.0
    down = 0.0

    def _add(name, weight=1.0, invert=False):
        nonlocal up, down
        if name in market_data:
            _, chg = market_data[name]
            chg = -chg if invert else chg
            if chg >= 0.05:
                up += weight * chg
            elif chg <= -0.05:
                down += weight * abs(chg)

    # индексы — позитив, DXY/10Y — негатив для риска
    _add("sp500", 1.0)
    _add("nasdaq", 1.0)
    _add("dxy", 0.7, invert=True)
    _add("treasury", 0.7, invert=True)

    score = up - down
    if score > 0.3:
        emoji, phrase = "📈", "Позитивный тон на ожиданиях спроса на риск."
    elif score < -0.3:
        emoji, phrase = "📉", "Негативный фон из-за сильного доллара и доходностей."
    else:
        emoji, phrase = "➖", "Нейтральный рынок без явного тренда."
    return f"1️⃣ Глобальные рынки: {emoji} {phrase}"

# ===== LLM DIGEST (compact) =====
async def ai_format_morning_digest_compact_final(news_list, market_data, crypto_data):
    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

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

    news_titles = "\n".join([f"- {n['title']}" for n in news_list[:7]])

    prompt = f"""
Сформируй КРАТКУЮ утреннюю сводку на русском (МАКС 900 символов).
Структура: 7 пунктов, каждый с номером 1️⃣..7️⃣ и коротким названием (без **жирного**).

1️⃣ Глобальные рынки (≤140) — качественная оценка настроения (без цифр).
2️⃣ Итоги торгов (≤120) — используй ТОЛЬКО: {real_data}
3️⃣ Трежерис, DXY, золото, нефть (≤120) — используй ТОЛЬКО: {real_data}
4️⃣ Монетарная политика (≤120)
5️⃣ Корпоративные новости (≤120)
6️⃣ Криптовалюты (≤160)
7️⃣ Геополитика (≤120)

⚠️ КРАТКО. Только факты из списка новостей и {real_data}. НЕ используй **жирный**.

Новости:
{news_titles}
"""

    resp = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Ты — финансовый редактор. Очень кратко, фактологично, без markdown-форматирования."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=400,
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()

# ===== MAIN SEND =====
async def send_morning_digest():
    print("🚀 Запуск утренней сводки...")

    # 1) свежие новости за утреннее окно
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

    # 3) текст сводки (черновик от LLM)
    print("🤖 Формируем компактную сводку…")
    digest_raw = await ai_format_morning_digest_compact_final(news_list, market_data, crypto_data)

    # === ПОСТ-ОБРАБОТКА СТИЛЯ ===
    digest = sanitize_markdown(digest_raw)                          # убираем ** и пр.
    digest = rebuild_lines_with_market_data(digest, market_data)    # 2 и 3 строго из market_data
    digest = decorate_digest_with_emojis(digest)                    # эмодзи под каждый пункт

    # --- перезаписываем пункт 1️⃣ на «качественный» mood-лайн
    mood_line = build_global_mood_line(market_data)
    lines = digest.splitlines()
    new_lines = []
    replaced = False
    for line in lines:
        if (not replaced) and line.strip().startswith("1️⃣"):
            new_lines.append(mood_line)
            replaced = True
        else:
            new_lines.append(line)
    digest = "\n".join(new_lines).strip()

    # 4) шапка с датой в Europe/Vienna
    now_local = datetime.now(LOCAL_TZ)
    header = f"🌅 Утренняя сводка — {now_local:%d.%m.%Y}"

    # 5) крипто-блок — многострочный, как просил
    crypto_section = ""
    if crypto_data:
        lines = ["💎 Криптовалюты (ТОП-5)"]
        if "bitcoin" in crypto_data:
            p = crypto_data["bitcoin"]["usd"]; c = crypto_data["bitcoin"]["usd_24h_change"]
            lines.append(f"BTC ${p:,.0f}({c:+.1f}%)")
        if "ethereum" in crypto_data:
            p = crypto_data["ethereum"]["usd"]; c = crypto_data["ethereum"]["usd_24h_change"]
            lines.append(f"ETH ${p:,.0f}({c:+.1f}%)")
        if "binancecoin" in crypto_data:
            p = crypto_data["binancecoin"]["usd"]; c = crypto_data["binancecoin"]["usd_24h_change"]
            lines.append(f"BNB ${p:.0f}({c:+.1f}%)")
        if "ripple" in crypto_data:
            p = crypto_data["ripple"]["usd"]; c = crypto_data["ripple"]["usd_24h_change"]
            lines.append(f"XRP ${p:.2f}({c:+.1f}%)")
        if "solana" in crypto_data:
            p = crypto_data["solana"]["usd"]; c = crypto_data["solana"]["usd_24h_change"]
            lines.append(f"SOL ${p:.0f}({c:+.1f}%)")
        crypto_section = "\n\n" + "\n".join(lines)

    # 6) собрать пост и уложить в лимит подписи к фото (1024)
    full_post = f"{header}\n\n{digest}{crypto_section}\n\n{SIGNATURE}"
    if len(full_post) > 1024:
        # сначала уберём крипто-хвост
        full_post = f"{header}\n\n{digest}\n\n{SIGNATURE}"
        if len(full_post) > 1024:
            excess = len(full_post) - 1021
            digest_short = digest[:-excess].rstrip()
            full_post = f"{header}\n\n{digest_short}…\n\n{SIGNATURE}"

    print(f"🧮 Длина поста: {len(full_post)} символов")

    # 7) получить/сжать картинку
    print("🖼️ Готовим изображение…")
    image = await get_morning_image()

    # 8) отправка в Telegram
    print("📤 Отправляем в Telegram…")
    try:
        bot = Bot(token=TELEGRAM_TOKEN)
        if image:
            await bot.send_photo(
                chat_id=TELEGRAM_CHANNEL_RU,
                photo=image,
                caption=full_post,
                parse_mode=None,   # чистый текст
            )
            print("✅ Пост с изображением отправлен!")
        else:
            await bot.send_message(
                chat_id=TELEGRAM_CHANNEL_RU,
                text=full_post,
                parse_mode=None,
            )
            print("✅ Пост без изображения отправлен!")
    except Exception as e:
        print(f"❌ Ошибка отправки в Telegram: {e}")

    # 9) статистика
    print("\n📊 СТАТИСТИКА:")
    print(f"• Новостей: {len(news_list)}")
    print(f"• Символов: {len(full_post)}")
    print(f"• Рыночные данные: {'✅' if market_data else '❌'}")
    print(f"• Крипто-данные: {'✅' if crypto_data else '❌'}")
    print(f"• Окно свежести: {FRESHNESS_HOURS_MORNING} ч")
    print(f"• Источник рынков: {MARKET_SOURCE_MODE}")
    print(f"• Картинка: {'✅' if image else '❌'}")

if __name__ == "__main__":
    asyncio.run(send_morning_digest())
