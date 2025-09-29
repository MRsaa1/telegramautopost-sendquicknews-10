#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import html
import math
import asyncio
import feedparser
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from io import BytesIO
from difflib import SequenceMatcher
from urllib.parse import urlparse

# --- внешние источники
import yfinance as yf
from pycoingecko import CoinGeckoAPI

# ================== CONFIG ==================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # опционально: LLM для пунктов 4–7
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHANNEL_RU = os.getenv("TELEGRAM_CHANNEL_RU", "-1002597393191")

SEND_TO_TELEGRAM = False                 # 🔒 ЗАГЛУШКА: ничего не уходит в Telegram
MAX_CAPTION = 1024

NEWS_COUNT = 15
SIGNATURE = "С вами был ReserveOne ☕️"

LOCAL_TZ = ZoneInfo("Europe/Vienna")
FRESHNESS_HOURS_MORNING = int(os.getenv("FRESHNESS_HOURS_MORNING", "10"))  # последние 10ч
MARKET_SOURCE_MODE = os.getenv("MARKET_SOURCE_MODE", "last_close")         # last_close | intraday
FORCE_GOLD_SPOT = os.getenv("FORCE_GOLD_SPOT", "0") == "1"

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

SECTION_EMOJI = {"1️⃣":"📊","2️⃣":"📈","3️⃣":"🏦","4️⃣":"🧭","5️⃣":"🏢","6️⃣":"🚀","7️⃣":"🌍"}

# ================== HELPERS: text ==================
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

def sanitize_markdown(text: str) -> str:
    if not text: return ""
    out = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    out = re.sub(r"__(.*?)__", r"\1", out)
    out = re.sub(r"_([^_]+)_", r"\1", out)
    out = re.sub(r"`([^`]+)`", r"\1", out)
    out = re.sub(r"\s+\n", "\n", out)
    return out.strip()

# ================== NEWS (pool, scoring, balance) ==================
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
            if not d.entries: continue
            for entry in d.entries:
                st = entry.get("published_parsed") or entry.get("updated_parsed")
                if not st: continue
                published_dt = datetime(*st[:6], tzinfo=timezone.utc)
                age_hours = (now - published_dt).total_seconds() / 3600.0
                if age_hours > FRESHNESS_HOURS_MORNING: continue
                title = _unescape_then_escape(_strip_tags(entry.get("title") or ""))
                summary = _unescape_then_escape(_strip_tags(entry.get("summary") or ""))
                link = (entry.get("link") or "").strip()
                if not (title or summary): continue
                if not is_quality_news(title, summary): continue
                entries.append({
                    "title": title,
                    "title_norm": _norm_title(title),
                    "summary": summary,
                    "link": link,
                    "source": url,
                    "published_dt": published_dt,
                    "age_hours": age_hours,
                })
        except Exception as e:
            print(f"⚠️ Feed error ({url}): {e}")

    if not entries: return []

    # dedup by link
    seen, uniq = set(), []
    for e in entries:
        lk = e.get("link") or ""
        if lk and lk not in seen:
            seen.add(lk); uniq.append(e)

    # semantic dedup
    filtered: list[dict] = []
    for e in uniq:
        if any(_similar(e["title_norm"], x["title_norm"]) > 0.92 for x in filtered):
            continue
        filtered.append(e)

    # scoring + sort
    for n in filtered: n["score"] = score_item(n)
    filtered.sort(key=lambda x: (x["score"], -x["published_dt"].timestamp()), reverse=True)

    # crypto/fin balance pool
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
    if len(out) < take: out += [n for n in fin[MIN_FIN_NEWS:]]
    return out[:take]

# ================== MARKET DATA ==================
def validate_market_data(data_type, current_value, change_percent) -> bool:
    rules = {
        "sp500": {"min": 2000, "max": 8000, "change_max": 10},
        "nasdaq": {"min": 5000, "max": 25000, "change_max": 10},
        "dxy": {"min": 80, "max": 120, "change_max": 5},
        "gold": {"min": 1000, "max": 3000, "change_max": 8},
        "oil": {"min": 20, "max": 150, "change_max": 15},
        "treasury": {"min": 0, "max": 10, "change_max": 2},
    }
    if data_type not in rules: return True
    r = rules[data_type]
    if not (r["min"] <= current_value <= r["max"]):
        print(f"⚠️ Подозрительное значение {data_type}: {current_value}")
        return False
    if abs(change_percent) > r["change_max"]:
        print(f"⚠️ Подозрительное изменение {data_type}: {change_percent:.2f}%")
        return False
    return True

def _safe_close_pair(df):
    if df is None or df.empty or "Close" not in df.columns: return None
    last = df["Close"].iloc[-1]
    try: last_f = float(last)
    except Exception: return None
    if math.isnan(last_f): return None
    if len(df["Close"]) >= 2:
        prev = df["Close"].iloc[-2]
        try: prev_f = float(prev)
        except Exception: return None
        if math.isnan(prev_f): return None
    else:
        prev_f = last_f
    return last_f, prev_f

def _pair_last_close(df):
    pair = _safe_close_pair(df)
    if not pair: return None
    cur, prev = pair
    chg = (cur - prev) / prev * 100 if prev else 0.0
    return cur, chg

_pair_intraday = _pair_last_close  # для краткости (можно заменить на иное поведение)

def _yf_download_first_ok(tickers: list[str], period="2d", interval="1d"):
    for t in tickers:
        try:
            df = yf.download(t, period=period, interval=interval, auto_adjust=False, progress=False)
            if df is not None and not df.empty and "Close" in df.columns and len(df["Close"]) >= 1:
                last = df["Close"].iloc[-1]
                try: last_f = float(last)
                except Exception: continue
                if not math.isnan(last_f): return df
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
            if pair and validate_market_data(name, pair[0], pair[1]):
                market_data[name] = pair

        _put("sp500", sp500); _put("nasdaq", nasdaq); _put("dxy", dxy)
        _put("oil", oil); _put("treasury", tnx)
        if gold_df is not None: _put("gold", gold_df)

        for k, (v, c) in market_data.items():
            unit = "$" if k in ("sp500", "nasdaq", "gold", "oil") else ""
            print(f"✅ {k.upper()}: {unit}{v:.2f} ({c:+.2f}%)")
        return market_data
    except Exception as e:
        print(f"❌ Error fetching market data: {e}")
        return None

# ================== CRYPTO SNAPSHOT (optional block at end) ==================
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
        return data
    except Exception as e:
        print(f"❌ Error fetching crypto data: {e}")
        return None

# ================== LLM 4–7 (optional) ==================
async def ai_make_points_4_7(news_list, market_data):
    """
    Возвращает 4 строки (4️⃣..7️⃣) — короткие факты.
    Если OPENAI_API_KEY пуст — безопасный фолбэк (обрезанные заголовки).
    """
    titles = [n["title"] for n in news_list[:8]]
    if not OPENAI_API_KEY:
        out = []
        cats = ["4️⃣ Монетарная политика", "5️⃣ Корпоративные новости",
                "6️⃣ Криптовалюты", "7️⃣ Геополитика"]
        for i, t in enumerate(titles[:4]):
            clean = re.sub(r"\s+", " ", t).strip()
            out.append(f"{cats[i]}: {clean}")
        # если заголовков мало — добьём заглушками
        while len(out) < 4:
            cats_idx = len(out)
            out.append(f"{['4️⃣ Монетарная политика','5️⃣ Корпоративные новости','6️⃣ Криптовалюты','7️⃣ Геополитика'][cats_idx]}: Короткий факт без воды.")
        return out

    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        prompt = (
            "Собери 4 однострочных факта (строго без переносов) для пунктов:\n"
            "4️⃣ Монетарная политика — одно наблюдение.\n"
            "5️⃣ Корпоративные новости — один факт (крупные эмитенты/IPO/M&A/гайд).\n"
            "6️⃣ Криптовалюты — один факт (регуляторы/ликвидность/институции).\n"
            "7️⃣ Геополитика — один факт с рыночной релевантностью.\n"
            "Тон: Bloomberg/Reuters. КРАТКО, без эмодзи и markdown. Используй только эти заголовки:\n"
            + "\n".join(f"- {t}" for t in titles)
        )
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ты — финансовый редактор. Пиши строго, кратко, по-русски."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=260,
        )
        txt = (resp.choices[0].message.content or "").strip()
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        # возьмём первые 4 строки
        out = lines[:4]
        # safety: если ИИ вернул мало — дополним обрезанными заголовками
        while len(out) < 4:
            i = len(out)
            t = titles[i] if i < len(titles) else "Факт дня."
            out.append(t[:180])
        # префиксуем правильными ярлыками
        tags = ["4️⃣ Монетарная политика", "5️⃣ Корпоративные новости",
                "6️⃣ Криптовалюты", "7️⃣ Геополитика"]
        pattern = r'^\d+\)\s*'
        out = [f"{tags[i]}: {re.sub(pattern, '', out[i]).strip()}" for i in range(4)]
        return out
    except Exception as e:
        print(f"⚠️ LLM fallback (error: {e})")
        # fallback на заголовки
        out = []
        cats = ["4️⃣ Монетарная политика", "5️⃣ Корпоративные новости",
                "6️⃣ Криптовалюты", "7️⃣ Геополитика"]
        for i, t in enumerate(titles[:4]):
            out.append(f"{cats[i]}: {t[:180]}")
        while len(out) < 4:
            idx = len(out)
            out.append(f"{cats[idx]}: Короткий факт без воды.")
        return out

# ================== FORMATTING: sections 1..7 ==================
def build_global_mood_line(market_data: dict | None) -> str:
    if not market_data:
        return "1️⃣ Глобальные рынки: ➖ Нейтральный фон без явных драйверов."
    up = down = 0.0
    def _add(name, weight=1.0, invert=False):
        nonlocal up, down
        if name in market_data:
            _, chg = market_data[name]
            chg = -chg if invert else chg
            if chg >= 0.05: up += weight * chg
            elif chg <= -0.05: down += weight * abs(chg)
    _add("sp500", 1.0); _add("nasdaq", 1.0); _add("dxy", 0.7, True); _add("treasury", 0.7, True)
    score = up - down
    if score > 0.3:  emoji, phrase = "📈", "Позитивный тон на ожиданиях спроса на риск."
    elif score < -0.3: emoji, phrase = "📉", "Негативный фон из-за сильного доллара и доходностей."
    else:            emoji, phrase = "➖", "Нейтральный рынок без явного тренда."
    return f"1️⃣ Глобальные рынки: {emoji} {phrase}"

def fmt_2_and_3(market_data: dict | None) -> tuple[str, str]:
    """
    2️⃣ — «Итоги торгов»: S&P500, Nasdaq, Нефть, DXY, 10Y (в одну строчку после переноса/emoji).
    3️⃣ — «Трежерис, DXY, золото, нефть»: короткая связная фраза.
    """
    if not market_data:
        return ("2️⃣ Итоги торгов: 📈 Данных мало для сводки.",
                "3️⃣ Трежерис, DXY, золото, нефть: 🏦 Данных мало для сводки.")
    def val(name, fmt_val):
        if name not in market_data: return None
        v, c = market_data[name]; return fmt_val(v) + f"({c:+.1f}%)"
    sp = val("sp500", lambda v: f"S&P500 ${v:.0f}")
    nd = val("nasdaq", lambda v: f"Nasdaq ${v:.0f}")
    oi = val("oil",   lambda v: f"Нефть ${v:.0f}")
    dx = val("dxy",   lambda v: f"DXY {v:.1f}")
    tn = val("treasury", lambda v: f"10Y {v:.1f}%")
    au = val("gold", lambda v: f"Золото ${v:.0f}")

    # 2️⃣ — компактный перечень (как в эталоне)
    line2_list = [p for p in (sp, nd, oi, dx, tn) if p]
    line2 = "2️⃣ Итоги торгов:\n" + ("📈 " + " · ".join(line2_list) if line2_list else "📈 —")

    # 3️⃣ — короткая фраза, объясняющая фон
    bits = []
    if tn: bits.append("доходности 10-леток")
    if dx: bits.append("индекс доллара DXY")
    trend_parts = []
    if "treasury" in market_data:
        _, ch = market_data["treasury"]; trend_parts.append(("растут", "снижаются")[ch<0])
        t_tr = ("растут","снижаются")[ch<0]
    else:
        t_tr = None
    if "dxy" in market_data:
        _, ch = market_data["dxy"]; d_tr = ("укрепляется","слабеет")[ch<0]
    else:
        d_tr = None
    pieces = []
    if t_tr: pieces.append(f"доходности 10-леток {t_tr}")
    if d_tr: pieces.append(f"DXY {d_tr}")
    if "gold" in market_data:
        v,c = market_data["gold"]; pieces.append(f"золото ${v:.0f}({c:+.1f}%)")
    if "oil" in market_data:
        v,c = market_data["oil"]; pieces.append(f"нефть ${v:.0f}({c:+.1f}%)")
    sent = " · ".join(pieces) if pieces else "без выраженного тренда"
    line3 = f"3️⃣ Трежерис, DXY, золото, нефть:\n🏦 {sent}"
    return line2, line3

def layout_multiline_all(lines: list[str]) -> str:
    """
    Превращает однострочные пункты в двухстрочные:
    'N️⃣ Название: <текст>' → 'N️⃣ Название:\n<эмодзи> <текст>'
    Эмодзи берём из SECTION_EMOJI, если внутри текста нет своего.
    """
    out = []
    for s in lines:
        s = s.strip()
        if not s: continue
        if re.match(r"^[1-7]️⃣", s):
            tag = s[:2]
            rest = s[2:].strip()
            head, body = (rest.split(":", 1) + [""])[:2]
            head = head.strip()
            body = body.strip()
            # уберём дубли «Название — ...»
            base = head.lower()
            for sep in ("— ", "- ", ": ", " — ", " - ", " : "):
                patt = (base + sep)
                if body.lower().startswith(patt):
                    body = body[len(patt):].lstrip()
                    break
            # возьмём первое emoji из body, если есть
            emoji = SECTION_EMOJI.get(tag, "")
            if body and body[0] in ("📊","📈","🏦","🧭","🏢","🚀","🌍","➖","📉"):
                emoji, body = body[0], body[1:].lstrip()
            out.append(f"{tag} {head}:")
            out.append(f"{emoji} {body}".rstrip())
            out.append("")  # пустая строка между блоками
        else:
            out.append(s)
    # убрать последний лишний перенос
    while out and out[-1] == "": out.pop()
    return "\n".join(out)

# ================== MAIN FLOW ==================
async def send_morning_digest():
    print("🚀 Запуск утренней сводки...")

    # 1) Новости
    print("📰 Получаем новости из RSS (только свежие)…")
    pool = get_feed_news(CRYPTO_FEEDS + FINANCE_FEEDS, NEWS_COUNT)
    if not pool:
        print("⚠️ Нет свежих новостей в окне актуальности")
        return
    news_list = filter_by_importance(pool, NEWS_COUNT)
    print(f"✅ Отобрано {len(news_list)} актуальных новостей (≤ {FRESHNESS_HOURS_MORNING}ч)")

    # 2) Рынки/крипто
    market_data = await get_market_data()
    crypto_data = await get_crypto_data()

    # 3) Секции 1–3 (строго по данным)
    line1 = build_global_mood_line(market_data)     # однострочный
    line2, line3 = fmt_2_and_3(market_data)         # уже двухстрочные

    # 4) Секции 4–7 (LLM или fallback)
    pts_4_7 = await ai_make_points_4_7(news_list, market_data)  # однострочные

    # Собираем список строк (1 — одностр, 2/3 — 2 стр, 4–7 — одностр)
    raw_lines = [line1, line2, line3] + pts_4_7

    # 4) Косметика
    raw_lines = [sanitize_markdown(x) for x in raw_lines if x]

    # 5) Превращаем всё в ДВУХСТРОЧНЫЙ формат по требованиям
    body = layout_multiline_all(raw_lines)

    # 6) Заголовок и подпись
    now_local = datetime.now(LOCAL_TZ)
    header = f"🌅 Утренняя сводка — {now_local:%d.%m.%Y}"
    tail = SIGNATURE

    # 7) Крипто-«хвост» (ТОП-5) — по желанию, добавляем в конце
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

    full_post = f"{header}\n\n{body}{crypto_section}\n\n{tail}"

    # жёсткий лимит подписи (если потом включим TG)
    if len(full_post) > MAX_CAPTION:
        # мягкая усадка: урезать крипто-хвост, затем длинные строки
        tmp = f"{header}\n\n{body}\n\n{tail}"
        if len(tmp) > MAX_CAPTION:
            excess = len(tmp) - (MAX_CAPTION - 1)
            tmp = tmp[:-excess].rstrip() + "…"
        full_post = tmp

    # === ПРЕДПРОСМОТР В КОНСОЛИ ===
    print("\n" + "="*58)
    print(full_post)
    print("="*58)
    print(f"🧮 Длина поста: {len(full_post)} символов")
    print("📤 Отправка в Telegram отключена (SEND_TO_TELEGRAM=False).")

# ================== RUN ==================
if __name__ == "__main__":
    asyncio.run(send_morning_digest())