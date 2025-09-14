import os
import datetime
import asyncio
import feedparser
import openai
from telegram import Bot
import re
import requests
from io import BytesIO
import html
import yfinance as yf
from pycoingecko import CoinGeckoAPI

# ===== CONFIG =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHANNEL_RU = "-1002597393191"

NEWS_COUNT = 15
SIGNATURE = "С вами был ReserveOne ☕️"

# Создаем папку для изображений
IMAGES_DIR = "images"
if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

# Путь к статичному изображению
STATIC_IMAGE_PATH = os.path.join(IMAGES_DIR, "morning_digest_static.png")

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
    "https://www.morningbrew.com/feed.xml"
]

# ===== IMAGE FUNCTIONS =====
def static_image_exists():
    """Проверяет, существует ли статичное изображение и оно не повреждено"""
    if not os.path.exists(STATIC_IMAGE_PATH):
        return False

    # Проверяем размер файла (должен быть больше 1KB)
    try:
        file_size = os.path.getsize(STATIC_IMAGE_PATH)
        if file_size < 1024:  # Меньше 1KB - файл повреждён
            print(f"⚠️ Файл изображения повреждён (размер: {file_size} байт)")
            return False
        return True
    except Exception as e:
        print(f"❌ Ошибка проверки файла изображения: {e}")
        return False

def save_static_image(image_bytes):
    """Сохраняет статичное изображение с проверкой"""
    try:
        # Создаём папку если её нет
        os.makedirs(IMAGES_DIR, exist_ok=True)

        # Сохраняем во временный файл сначала
        temp_path = STATIC_IMAGE_PATH + ".tmp"
        with open(temp_path, 'wb') as f:
            f.write(image_bytes.getvalue())

        # Проверяем что файл сохранился корректно
        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 1024:
            # Перемещаем в финальное место
            os.rename(temp_path, STATIC_IMAGE_PATH)
            print(f"✅ Статичное изображение сохранено: {STATIC_IMAGE_PATH}")
            return True
        else:
            print("❌ Ошибка сохранения изображения")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False

    except Exception as e:
        print(f"❌ Ошибка сохранения изображения: {e}")
        return False

def load_static_image():
    """Загружает статичное изображение с проверкой"""
    try:
        if os.path.exists(STATIC_IMAGE_PATH):
            with open(STATIC_IMAGE_PATH, 'rb') as f:
                image_data = f.read()
                if len(image_data) > 1024:  # Проверяем размер
                    return BytesIO(image_data)
                else:
                    print("⚠️ Загруженное изображение слишком маленькое")
        return None
    except Exception as e:
        print(f"❌ Ошибка загрузки изображения: {e}")
        return None

# ===== DATA VALIDATION =====
def validate_market_data(data_type, current_value, change_percent):
    """Валидация рыночных данных на разумность"""
    validation_rules = {
        "sp500": {"min": 2000, "max": 8000, "change_max": 10},
        "nasdaq": {"min": 5000, "max": 25000, "change_max": 10},
        "dxy": {"min": 80, "max": 120, "change_max": 5},
        "gold": {"min": 1000, "max": 3000, "change_max": 8},
        "oil": {"min": 20, "max": 150, "change_max": 15},
        "treasury": {"min": 0, "max": 10, "change_max": 2}
    }

    if data_type not in validation_rules:
        return True

    rules = validation_rules[data_type]

    # Проверяем разумность значения
    if not (rules["min"] <= current_value <= rules["max"]):
        print(f"⚠️ Подозрительное значение {data_type}: {current_value}")
        return False

    # Проверяем разумность изменения
    if abs(change_percent) > rules["change_max"]:
        print(f"⚠️ Подозрительное изменение {data_type}: {change_percent:.2f}%")
        return False

    return True

# ===== HELPERS =====
def get_feed_news(feeds, max_news):
    entries = []
    for url in feeds:
        try:
            d = feedparser.parse(url)
            for entry in d.entries:
                entry.source_url = url
            entries.extend(d.entries)
        except Exception as e:
            print(f"Error parsing {url}: {e}")

    entries = sorted(
        entries,
        key=lambda e: e.get("published_parsed", datetime.datetime.now().timetuple()),
        reverse=True
    )

    seen, fresh_news = set(), []
    for e in entries:
        link = e.get("link", "").strip()
        if link and link not in seen:
            seen.add(link)
            title = re.sub(r"<.*?>", "", e.get("title", "").strip())
            summary = re.sub(r"<.*?>", "", e.get("summary", "").strip())

            # Улучшенный фильтр новостей
            if is_quality_news(title, summary):
                fresh_news.append({
                    "title": title,
                    "summary": summary,
                    "link": link,
                    "source": getattr(e, "source_url", "")
                })
        if len(fresh_news) >= max_news * 2:  # Берем больше для фильтрации
            break

    # Дополнительная фильтрация по важности
    fresh_news = filter_by_importance(fresh_news)
    return fresh_news[:max_news]

def is_quality_news(title, summary):
    """Улучшенный фильтр качества новостей по топ-7 категориям"""
    text = f"{title} {summary}".lower()

    # Исключаем мусорные новости
    spam_keywords = [
        "click here", "subscribe", "newsletter", "advertisement", "sponsored",
        "clickbait", "you won't believe", "shocking", "amazing", "incredible",
        "breaking news", "urgent", "exclusive", "limited time", "act now",
        "free", "bonus", "discount", "sale", "offer", "deal", "promotion",
        "click to read", "read more", "continue reading", "full story",
        "подпишитесь", "реклама", "акция", "скидка", "бесплатно", "эксклюзив",
        "шокирующие", "невероятные", "срочно", "ограниченное время",
        # Добавляем новые фильтры
        "ожидает", "может", "возможно", "призывает", "сообщает источник",
        "согласно источникам", "анонимные источники", "слухи"
    ]

    for keyword in spam_keywords:
        if keyword in text:
            return False

    # Проверяем на минимальную длину
    if len(title) < 10 or len(summary) < 20:
        return False

    # Усиливаем требования к конкретности
    has_specific_data = bool(re.search(r'\$\d+|\d+%|\d+\.\d+', text))
    has_concrete_event = any(word in text for word in [
        "earnings", "revenue", "profit", "loss", "ipo", "merger", "acquisition",
        "rate", "inflation", "gdp", "unemployment", "fed", "ecb", "sec",
        "bitcoin", "ethereum", "crypto", "stock", "market", "trade",
        "выручка", "прибыль", "ставка", "инфляция", "биткоин", "акции"
    ])

    return has_specific_data or has_concrete_event

def filter_by_importance(news_list):
    """Дополнительная фильтрация по важности с приоритетом категориям"""
    scored_news = []

    for news in news_list:
        score = 0
        text = f"{news['title']} {news['summary']}".lower()

        # Баллы за важные категории (приоритет)
        category_scores = {
            "crypto": 15,       # Криптовалюты - высший приоритет
            "monetary": 13,     # Центробанки и регуляторы
            "corporate": 12,    # Корпоративные новости
            "markets": 12,      # Макроэкономика
            "geopolitics": 8,   # Геополитика
            "innovation": 8,    # Инновации
            "alternative": 6    # Альтернативные активы
        }

        # Проверяем каждую категорию
        if any(word in text for word in ["s&p 500", "nasdaq", "dow", "cpi", "gdp", "inflation", "федеральная резервная система", "фрс"]):
            score += category_scores["markets"]
        if any(word in text for word in ["federal reserve", "fed", "ecb", "regulation", "sec", "федеральная резервная система", "регулирование"]):
            score += category_scores["monetary"]
        if any(word in text for word in ["apple", "microsoft", "nvidia", "tesla", "earnings", "revenue", "акции", "выручка"]):
            score += category_scores["corporate"]
        if any(word in text for word in ["bitcoin", "btc", "ethereum", "eth", "crypto", "биткоин", "эфириум"]):
            score += category_scores["crypto"]
        if any(word in text for word in ["china", "russia", "trade war", "sanctions", "китай", "санкции"]):
            score += category_scores["geopolitics"]
        if any(word in text for word in ["ai", "artificial intelligence", "chatgpt", "искусственный интеллект"]):
            score += category_scores["innovation"]
        if any(word in text for word in ["real estate", "venture capital", "недвижимость"]):
            score += category_scores["alternative"]

        # Бонус за цифры
        if re.search(r'\$\d+', text):
            score += 3
        if re.search(r'\d+%', text):
            score += 2

        # Бонус за надежные источники
        reliable_sources = ['bloomberg', 'reuters', 'cnbc', 'coindesk', 'cointelegraph', 'marketwatch']
        if any(source in news['source'] for source in reliable_sources):
            score += 3

        scored_news.append((score, news))

    # Сортируем по важности
    scored_news.sort(key=lambda x: x[0], reverse=True)
    return [news for score, news in scored_news]

async def get_market_data():
    """Получение актуальных рыночных данных - ИСПРАВЛЕННАЯ версия"""
    try:
        print("📊 Получаем рыночные данные...")

        # Фондовые индексы - получаем данные за 2 дня для корректного расчета изменений
        sp500_data = yf.download("^GSPC", period="2d", interval="1d", auto_adjust=False)
        nasdaq_data = yf.download("^IXIC", period="2d", interval="1d", auto_adjust=False)
        dax_data = yf.download("^GDAXI", period="2d", interval="1d", auto_adjust=False)

        # Валюты и сырье - ИСПРАВЛЕНО: используем правильный символ для DXY
        dxy_data = yf.download("DX-Y.NYB", period="2d", interval="1d", auto_adjust=False)
        gold_data = yf.download("GC=F", period="2d", interval="1d", auto_adjust=False)
        oil_data = yf.download("BZ=F", period="2d", interval="1d", auto_adjust=False)

        # Трежерис
        treasury_data = yf.download("^TNX", period="2d", interval="1d", auto_adjust=False)

        # Получаем текущие цены и изменения
        market_data = {}

        # S&P 500
        if not sp500_data.empty and len(sp500_data) >= 2:
            sp500_current = float(sp500_data["Close"].iloc[-1])
            sp500_prev = float(sp500_data["Close"].iloc[-2])
            sp500_change = ((sp500_current - sp500_prev) / sp500_prev * 100)

            if validate_market_data("sp500", sp500_current, sp500_change):
                market_data["sp500"] = (sp500_current, sp500_change)
                print(f"✅ S&P 500: ${sp500_current:.2f} ({sp500_change:+.2f}%)")
            else:
                print(f"❌ S&P 500: данные не прошли валидацию")

        # Nasdaq
        if not nasdaq_data.empty and len(nasdaq_data) >= 2:
            nasdaq_current = float(nasdaq_data["Close"].iloc[-1])
            nasdaq_prev = float(nasdaq_data["Close"].iloc[-2])
            nasdaq_change = ((nasdaq_current - nasdaq_prev) / nasdaq_prev * 100)

            if validate_market_data("nasdaq", nasdaq_current, nasdaq_change):
                market_data["nasdaq"] = (nasdaq_current, nasdaq_change)
                print(f"✅ Nasdaq: ${nasdaq_current:.2f} ({nasdaq_change:+.2f}%)")
            else:
                print(f"❌ Nasdaq: данные не прошли валидацию")

        # DXY - ИСПРАВЛЕНО
        if not dxy_data.empty and len(dxy_data) >= 2:
            dxy_current = float(dxy_data["Close"].iloc[-1])
            dxy_prev = float(dxy_data["Close"].iloc[-2])
            dxy_change = ((dxy_current - dxy_prev) / dxy_prev * 100)

            if validate_market_data("dxy", dxy_current, dxy_change):
                market_data["dxy"] = (dxy_current, dxy_change)
                print(f"✅ DXY: {dxy_current:.2f} ({dxy_change:+.2f}%)")
            else:
                print(f"❌ DXY: данные не прошли валидацию")

        # Золото
        if not gold_data.empty and len(gold_data) >= 2:
            gold_current = float(gold_data["Close"].iloc[-1])
            gold_prev = float(gold_data["Close"].iloc[-2])
            gold_change = ((gold_current - gold_prev) / gold_prev * 100)

            if validate_market_data("gold", gold_current, gold_change):
                market_data["gold"] = (gold_current, gold_change)
                print(f"✅ Золото: ${gold_current:.2f} ({gold_change:+.2f}%)")
            else:
                print(f"❌ Золото: данные не прошли валидацию")

        # Нефть
        if not oil_data.empty and len(oil_data) >= 2:
            oil_current = float(oil_data["Close"].iloc[-1])
            oil_prev = float(oil_data["Close"].iloc[-2])
            oil_change = ((oil_current - oil_prev) / oil_prev * 100)

            if validate_market_data("oil", oil_current, oil_change):
                market_data["oil"] = (oil_current, oil_change)
                print(f"✅ Нефть: ${oil_current:.2f} ({oil_change:+.2f}%)")
            else:
                print(f"❌ Нефть: данные не прошли валидацию")

        # 10Y Treasury
        if not treasury_data.empty and len(treasury_data) >= 2:
            treasury_current = float(treasury_data["Close"].iloc[-1])
            treasury_prev = float(treasury_data["Close"].iloc[-2])
            treasury_change = ((treasury_current - treasury_prev) / treasury_prev * 100)

            if validate_market_data("treasury", treasury_current, treasury_change):
                market_data["treasury"] = (treasury_current, treasury_change)
                print(f"✅ 10Y Treasury: {treasury_current:.2f}% ({treasury_change:+.2f}%)")
            else:
                print(f"❌ 10Y Treasury: данные не прошли валидацию")

        return market_data

    except Exception as e:
        print(f"❌ Error fetching market data: {e}")
        return None

async def get_crypto_data():
    """Получение данных криптовалют"""
    try:
        print("💰 Получаем данные криптовалют...")
        cg = CoinGeckoAPI()
        crypto_data = cg.get_price(
            ids="bitcoin,ethereum,binancecoin,ripple,solana", 
            vs_currencies="usd", 
            include_24hr_change=True
        )

        if crypto_data:
            if "bitcoin" in crypto_data:
                btc_price = crypto_data["bitcoin"]["usd"]
                btc_change = crypto_data["bitcoin"]["usd_24h_change"]
                print(f"✅ BTC: ${btc_price:,.0f} ({btc_change:+.2f}%)")

            if "ethereum" in crypto_data:
                eth_price = crypto_data["ethereum"]["usd"]
                eth_change = crypto_data["ethereum"]["usd_24h_change"]
                print(f"✅ ETH: ${eth_price:,.0f} ({eth_change:+.2f}%)")

        return crypto_data
    except Exception as e:
        print(f"❌ Error fetching crypto data: {e}")
        return None

async def get_morning_image():
    """Получение изображения для утренней сводки - улучшенная версия"""
    # Проверяем, есть ли уже сохраненное изображение
    if static_image_exists():
        print("✅ Используем сохраненное статичное изображение")
        cached_image = load_static_image()
        if cached_image:
            return cached_image
        else:
            print("⚠️ Кэшированное изображение повреждено, генерируем новое...")

    print("🎨 Генерируем новое статичное изображение (только один раз)...")

    try:
        client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
        prompt = (
            "Digital illustration, fun but professional, modern flat style, soft colors. "
            "A minimalist composition featuring: "
            "A flat-designed coffee cup with steam rising in geometric shapes, "
            "a newspaper with clean typography and flat icons representing finance and crypto, "
            "geometric charts and graphs in pastel colors, "
            "flat design elements like coins, charts, and market symbols. "
            "Color palette: soft pastels (dusty rose, mint green, lavender, warm beige). "
            "Clean lines, no gradients, modern flat design aesthetic. "
            "Professional yet friendly financial morning theme. "
            "No text or words visible."
        )

        resp = await client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024"
        )

        img_url = resp.data[0].url
        image_bytes = BytesIO(requests.get(img_url).content)

        # Сохраняем изображение навсегда
        if save_static_image(image_bytes):
            return image_bytes
        else:
            print("⚠️ Не удалось сохранить изображение, используем временное")
            return image_bytes

    except Exception as e:
        print(f"❌ Ошибка генерации изображения: {e}")
        return None

async def ai_format_morning_digest_compact_final(news_list, market_data, crypto_data):
    """Финальная компактная версия с жестким контролем лимита - исправленная"""
    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

    # Формируем строку с реальными данными
    real_data = ""
    if market_data:
        if "sp500" in market_data:
            sp500_val, sp500_chg = market_data["sp500"]
            real_data += f"S&P500 ${sp500_val:.0f}({sp500_chg:+.1f}%), "
        if "nasdaq" in market_data:
            nasdaq_val, nasdaq_chg = market_data["nasdaq"]
            real_data += f"Nasdaq ${nasdaq_val:.0f}({nasdaq_chg:+.1f}%), "
        if "gold" in market_data:
            gold_val, gold_chg = market_data["gold"]
            real_data += f"Золото ${gold_val:.0f}({gold_chg:+.1f}%), "
        if "oil" in market_data:
            oil_val, oil_chg = market_data["oil"]
            real_data += f"Нефть ${oil_val:.0f}({oil_chg:+.1f}%), "
        if "dxy" in market_data:
            dxy_val, dxy_chg = market_data["dxy"]
            real_data += f"DXY {dxy_val:.1f}({dxy_chg:+.1f}%), "
        if "treasury" in market_data:
            treasury_val, treasury_chg = market_data["treasury"]
            real_data += f"10Y {treasury_val:.1f}%({treasury_chg:+.1f}%)"

    prompt = f"""
Сформируй КРАТКУЮ утреннюю сводку на русском языке (МАКСИМУМ 900 символов).
Структура из 7 пунктов с эмодзи:

1️⃣ Глобальные рынки (максимум 140 символов)
2️⃣ Итоги торгов (максимум 120 символов) - используй: {real_data}
3️⃣ Трежерис, DXY, золото, нефть (максимум 120 символов) - используй: {real_data}
4️⃣ Монетарная политика (максимум 120 символов)
5️⃣ Корпоративные новости (максимум 120 символов)
6️⃣ Криптовалюты (максимум 160 символов)
7️⃣ Геополитика (максимум 120 символов)

⚠️ КРАТКО! Только ключевые факты. Используй ТОЛЬКО реальные данные выше.
⚠️ НЕ ПРИДУМЫВАЙ цифры! Только из списка новостей.

Новости:
{chr(10).join([f"- {news['title']}" for news in news_list[:6]])}
"""

    resp = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Ты — финансовый аналитик. Пиши КРАТКО, используй ТОЛЬКО предоставленные данные."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=400,
        temperature=0.1
    )

    return resp.choices[0].message.content.strip()

async def send_morning_digest():
    """Отправка утренней сводки в Telegram"""
    print("�� Запуск утренней сводки...")

    # Получаем новости
    print("📰 Получаем новости из RSS...")
    news_list = get_feed_news(CRYPTO_FEEDS + FINANCE_FEEDS, NEWS_COUNT)
    if not news_list:
        print("⚠️ Нет свежих новостей")
        return

    print(f"✅ Найдено {len(news_list)} новостей")

    # Получаем реальные данные
    market_data = await get_market_data()
    crypto_data = await get_crypto_data()

    # Формируем сводку с жестким контролем лимита
    print("🤖 Формируем компактную сводку...")
    digest = await ai_format_morning_digest_compact_final(news_list, market_data, crypto_data)

    # Добавляем крипто-данные (компактно)
    crypto_section = ""
    if crypto_data:
        crypto_section = "\n\n�� Криптовалюты (ТОП-5)\n"

        if "bitcoin" in crypto_data:
            btc_price = crypto_data["bitcoin"]["usd"]
            btc_change = crypto_data["bitcoin"]["usd_24h_change"]
            crypto_section += f"BTC ${btc_price:,.0f}({btc_change:+.1f}%)\n"

        if "ethereum" in crypto_data:
            eth_price = crypto_data["ethereum"]["usd"]
            eth_change = crypto_data["ethereum"]["usd_24h_change"]
            crypto_section += f"ETH ${eth_price:,.0f}({eth_change:+.1f}%)\n"

        if "binancecoin" in crypto_data:
            bnb_price = crypto_data["binancecoin"]["usd"]
            bnb_change = crypto_data["binancecoin"]["usd_24h_change"]
            crypto_section += f"BNB ${bnb_price:.0f}({bnb_change:+.1f}%)\n"

        if "ripple" in crypto_data:
            xrp_price = crypto_data["ripple"]["usd"]
            xrp_change = crypto_data["ripple"]["usd_24h_change"]
            crypto_section += f"XRP ${xrp_price:.2f}({xrp_change:+.1f}%)\n"

        if "solana" in crypto_data:
            sol_price = crypto_data["solana"]["usd"]
            sol_change = crypto_data["solana"]["usd_24h_change"]
            crypto_section += f"SOL ${sol_price:.0f}({sol_change:+.1f}%)\n"
    else:
        crypto_section = "\n\n�� Криптовалюты (ТОП-5)\nДанные недоступны"

    # Формируем полный пост
    full_post = f"🌅 Утренняя сводка — {datetime.datetime.now().strftime('%d.%m.%Y')}\n\n{digest}{crypto_section}\n\n{SIGNATURE}"

    # Проверяем длину
    post_length = len(full_post)
    print(f"�� Длина поста: {post_length} символов")

    # Если превышает лимит, убираем крипто-секцию
    if post_length > 1024:
        print("⚠️ Пост превышает лимит, убираем крипто-секцию...")
        full_post = f"🌅 Утренняя сводка — {datetime.datetime.now().strftime('%d.%m.%Y')}\n\n{digest}\n\n{SIGNATURE}"
        post_length = len(full_post)
        print(f"📏 Длина после сокращения: {post_length} символов")

        # Если все еще превышает, обрезаем
        if post_length > 1024:
            print("⚠️ Все еще превышает, обрезаем...")
            # Считаем сколько символов нужно убрать
            excess = post_length - 1021  # 1021 + "..." = 1024
            digest_shortened = digest[:-excess]
            full_post = f"🌅 Утренняя сводка — {datetime.datetime.now().strftime('%d.%m.%Y')}\n\n{digest_shortened}...\n\n{SIGNATURE}"
            post_length = len(full_post)
            print(f"📏 Длина после обрезки: {post_length} символов")

    # Получаем статичное изображение
    print("🖼️ Получаем изображение...")
    try:
        image = await get_morning_image()
        print("✅ Изображение готово")
    except Exception as e:
        print(f"❌ Ошибка работы с изображением: {e}")
        image = None

    # Отправляем в Telegram
    print("📤 Отправляем в Telegram...")
    try:
        bot = Bot(token=TELEGRAM_TOKEN)

        if image:
            await bot.send_photo(
                chat_id=TELEGRAM_CHANNEL_RU,
                photo=image,
                caption=full_post,
                parse_mode=None
            )
            print("✅ Пост с изображением отправлен!")
        else:
            await bot.send_message(
                chat_id=TELEGRAM_CHANNEL_RU,
                text=full_post,
                parse_mode=None
            )
            print("✅ Пост без изображения отправлен!")

    except Exception as e:
        print(f"❌ Ошибка отправки в Telegram: {e}")

    # Статистика
    print(f"\n📊 СТАТИСТИКА:")
    print(f"• Новостей: {len(news_list)}")
    print(f"• Символов: {len(full_post)}")
    print(f"• Слов: {len(full_post.split())}")
    print(f"• Рыночные данные: {'✅' if market_data else '❌'}")
    print(f"• Крипто-данные: {'✅' if crypto_data else '❌'}")
    print(f"• Запас символов: {1024 - len(full_post)}")
    print(f"• Изображение: {'✅ (статичное)' if static_image_exists() else '❌'}")

    # Показываем источники новостей
    print(f"\n�� ИСТОЧНИКИ НОВОСТЕЙ:")
    sources = {}
    for news in news_list:
        domain = news['source'].split('/')[2] if '//' in news['source'] else news['source']
        sources[domain] = sources.get(domain, 0) + 1

    for source, count in sources.items():
        print(f"  • {source}: {count} новостей")

# Запускаем отправку в Telegram
if __name__ == "__main__":
    asyncio.run(send_morning_digest())