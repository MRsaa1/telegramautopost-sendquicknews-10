# ===== 1) Настройки лимита под подпись =====
CAPTION_LIMIT = 1020  # безопасная цель для caption (≤1024)

# ===== 2) Компактер чисел и текста =====
def _compact_money(val: str) -> str:
    # $109,428 -> $109k ; $4,008 -> $4.0k ; $972 -> $972 ; $2.78 -> $2.78
    # Работает только для $-формата; не трогаем проценты.
    m = re.match(r"\$(\d{1,3}(?:,\d{3})+|\d+\.\d+|\d+)", val)
    if not m:
        return val
    num = m.group(1).replace(",", "")
    try:
        f = float(num)
    except:
        return val
    if f >= 1000:
        # 1000+ -> k с 0/1 десятичным знаком (4.0k; 109k)
        k = f/1000.0
        s = f"{k:.1f}k" if k < 10 else f"{k:.0f}k"
        return "$" + s
    # <1000 — как есть, но уберём лишние нули 202.0 -> 202
    s = f"{f:.2f}".rstrip("0").rstrip(".")
    return "$" + s

def _tighten_spaces(text: str) -> str:
    # убираем двойные пустые строки и лишние пробелы
    t = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)          # хвостовые пробелы
    t = re.sub(r"\n{3,}", "\n\n", t)                              # ≤ один пустой между блоками
    t = re.sub(r" +([)%])", r"\1", t)                             # пробел перед ) или %
    t = re.sub(r"\$ +", r"$", t)                                  # $ 109 -> $109
    t = re.sub(r"· ", "·", t)                                     # точки-разделители без лишнего пробела
    return t.strip()

def _shorten_labels(text: str) -> str:
    reps = {
        "Итоги торгов:": "Итоги:",
        "Трежерис, DXY, золото, нефть:": "DXY/10Y/золото/нефть:",
        "Монетарная политика:": "Монетарная политика:",
        "Корпоративные новости:": "Корпоративные новости:",
        "Криптовалюты:": "Криптовалюты:",
        "Глобальные рынки:": "Глобальные рынки:",
        "Геополитика:": "Геополитика:",
        "Утренняя сводка —": "Утренняя сводка —",
    }
    for a, b in reps.items():
        text = text.replace(a, b)
    return text

def _compact_dollars_everywhere(text: str) -> str:
    # Пробегаем все $числа и компактируем
    def repl(m):
        return _compact_money(m.group(0))
    return re.sub(r"\$\d[\d,\.]*", repl, text)

def _trim_decimals_in_percents(text: str) -> str:
    # +0.60% -> +0.6% ; +0.0% -> +0.0% (оставим один знак, но нулевые оставим)
    return re.sub(r"([+-])(\d+)\.(\d+)%", lambda m: f"{m.group(1)}{m.group(2)}.{m.group(3)[0]}%", text)

def _one_line_sections_2_3(text: str) -> str:
    # гарантируем, что 2️⃣ и 3️⃣ — одна строка без переносов
    lines = []
    for block in text.split("\n"):
        lines.append(block.rstrip())
    t = "\n".join(lines)
    # убрать случайные \n после заголовков 2️⃣/3️⃣
    t = re.sub(r"^(2️⃣ [^\n]+):\n[📈🏦] ", r"\1: ", t, flags=re.MULTILINE)
    return t

def _crunch_two_line_sections(text: str) -> str:
    # Для 4–7: "4️⃣ Заголовок:\n🧭 Текст" -> "4️⃣ Заголовок: 🧭 Текст"
    t = re.sub(r"^([4-7]️⃣ [^\n]+):\n([📊📈🏦🧭🏢🚀🌍➖📉📈]) ?", r"\1: \2 ", text, flags=re.MULTILINE)
    return t

def enforce_len_budget_exact(header: str, body: str, tail: str, max_len: int) -> str:
    """
    Сохраняем все секции (1–7), блок ТОП-5 и подпись.
    Последовательно ужимаем до max_len.
    """
    text = f"{header}\n\n{body}\n\n{tail}"
    if len(text) <= max_len:
        return text

    # Пасс 1 — базовая чистка форматирования
    text = _tighten_spaces(text)
    text = _shorten_labels(text)
    text = _one_line_sections_2_3(text)
    text = _crunch_two_line_sections(text)
    if len(text) <= max_len:
        return text

    # Пасс 2 — числа и проценты
    text = _compact_dollars_everywhere(text)
    text = _trim_decimals_in_percents(text)
    text = _tighten_spaces(text)
    if len(text) <= max_len:
        return text

    # Пасс 3 — убрать пустые строки между секциями (оставить один \n)
    text = re.sub(r"\n{2,}", "\n", text).strip()
    if len(text) <= max_len:
        return text

    # Пасс 4 — микротрим длинных предложений внутри 4–7 (без удаления секций!)
    def trim_long_lines(t: str, target: int) -> str:
        lines = t.split("\n")
        for i, ln in enumerate(lines):
            if len(t) <= target:
                break
            if ln[:2] in ("4️⃣","5️⃣","6️⃣","7️⃣") and len(ln) > 140:
                lines[i] = ln[:140].rstrip(" ,.;:—-") + "…"
                t = "\n".join(lines)
        return "\n".join(lines)
    text = trim_long_lines(text, max_len)
    if len(text) <= max_len:
        return text

    # Пасс 5 — укоротить заголовок (если нужно): убрать эмодзи "🌅 "
    if text.startswith("🌅 "):
        text = text[2:].lstrip()
    if len(text) <= max_len:
        return text

    # Пасс 6 — крайний: слегка подрезать «Глобальные рынки» до одной фразы
    text = re.sub(r"^1️⃣ [^\n]+: [^\n]+", lambda m: (m.group(0)[:120] + "…") if len(text) > max_len else m.group(0),
                  text, count=1, flags=re.MULTILINE)
    # На этом стоп: ничего не удаляем, только подрезаем.
    return text
