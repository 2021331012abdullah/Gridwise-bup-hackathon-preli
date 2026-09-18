"""Deterministic time-window extraction for English, Bengali and Banglish.

Windows are start-inclusive / end-exclusive throughout, per section 5.1 of the
Problem Statement: "1 PM to 3 PM" means hours [13, 14].

The parser resolves a bare clock number against a day-part marker, because in
Bengali the marker carries the AM/PM information that English puts in the
suffix.  A trailing number with no marker of its own inherits the marker of the
number it is paired with, which is what "দুপুর ১টা থেকে ৩টা" (1 PM to 3 PM)
requires.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Tuple

BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

# Day-part markers → canonical bucket.
DAYPART = {
    # dawn / morning → AM
    "ভোর": "am", "bhor": "am", "সকাল": "am", "shokal": "am", "sokal": "am",
    "shokale": "am", "morning": "am",
    # midday / early afternoon → 12..15
    "দুপুর": "noon", "dupur": "noon", "dupure": "noon", "বেলা": "noon",
    "bela": "noon", "noon": "noon", "midday": "noon",
    # afternoon → 15..17
    "বিকাল": "pm", "বিকেল": "pm", "bikal": "pm", "bikel": "pm", "bikale": "pm",
    "afternoon": "pm",
    # evening → 18..19
    "সন্ধ্যা": "pm", "সন্ধ্যায়": "pm", "সন্ধে": "pm", "shondha": "pm",
    "sondha": "pm", "shondhya": "pm", "shondhay": "pm", "sondhya": "pm",
    "evening": "pm",
    # night → 20..23 for 8..11, early hours for 1..4
    "রাত": "night", "রাতে": "night", "raat": "night", "rat": "night",
    "raate": "night", "night": "night",
}

DAYPART = {unicodedata.normalize("NFC", k): v for k, v in DAYPART.items()}
_MARKER_RE = "|".join(sorted((re.escape(k) for k in DAYPART), key=len, reverse=True))

MIDNIGHT_WORDS = ["মধ্যরাত", "মধ্যরাতে", "রাইত বারোটা", "moddhoraat", "moddhorat",
                  "modhyorat", "midnight"]
ALLDAY_WORDS = [
    "round-the-clock", "round the clock", "entire planning day", "all 24 hours",
    "24-hour horizon", "24 hour horizon", "entire 24-hour", "24-hour period",
    "24 hour period", "throughout the day", "at any point during the day",
    "at any point during the 24", "any hour of the day",
    "all day", "saradin", "sara din", "shara din", "সারাদিন",
    "সারা দিন", "সারাক্ষণ", "২৪ ঘণ্টা", "24 ঘণ্টা", "24 ghonta", "puro din",
    "সারাদিনের", "সর্বক্ষণ", "চব্বিশ ঘণ্টা",
]

# Phrases that mean "every hour" only when no explicit window is stated.
# "must not exceed 155 kWh in any hour between 6 PM and 9 PM" is a per-hour cap
# inside a window, not an all-day directive, so these are a last resort only.
WEAK_ALLDAY_WORDS = [
    "in any hour", "at all times", "every hour", "at any time", "at any point",
    "completely prohibited today", "prohibited today", "banned today",
    "throughout today", "today", "আজ", "আজকে", "aaj", "aj ",
]

# Day-part words that can stand as a bound with no number attached.
BARE_BOUND = {
    "dawn": 6, "daybreak": 6, "sunrise": 6, "ভোর": 6, "bhor": 6, "bhore": 6,
    "বিয়ান": 6, "noon": 12, "midday": 12, "দুপুর": 12, "solar noon": 12,
}

MIDNIGHT_WORDS = [unicodedata.normalize("NFC", w) for w in MIDNIGHT_WORDS]
ALLDAY_WORDS = [unicodedata.normalize("NFC", w) for w in ALLDAY_WORDS]
WEAK_ALLDAY_WORDS = [unicodedata.normalize("NFC", w) for w in WEAK_ALLDAY_WORDS]
BARE_BOUND = {unicodedata.normalize("NFC", k): v for k, v in BARE_BOUND.items()}

WORD_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "a couple of": 2, "couple of": 2, "ek": 1, "dui": 2, "tin": 3, "char": 4,
    "এক": 1, "দুই": 2, "তিন": 3, "চার": 4, "পাঁচ": 5,
}

# Bengali clock numbers written as words, including the Sanskritised and
# dialectal forms that appear in formal notices and regional speech.
BN_CLOCK_WORDS = {
    "একটা": 1, "একটার": 1, "এক-ঘটিকা": 1, "দুইটা": 2, "দুটা": 2, "দুটো": 2,
    "দ্বি-ঘটিকা": 2, "দ্বিঘটিকা": 2, "দুইটার": 2, "তিনটা": 3, "তিনটে": 3,
    "তিনটার": 3, "ত্রি-ঘটিকা": 3, "চারটা": 4, "চারটে": 4, "চতুর-ঘটিকা": 4,
    "পাঁচটা": 5, "পাঁচটে": 5, "পঞ্চ-ঘটিকা": 5, "পঞ্চঘটিকা": 5, "পাঁচটার": 5,
    "ছয়টা": 6, "ছটা": 6, "ষষ্ঠ-ঘটিকা": 6, "সাতটা": 7, "সাতটার": 7,
    "আটটা": 8, "আটটার": 8, "নয়টা": 9, "নটা": 9, "দশটা": 10, "দশটার": 10,
    "এগারোটা": 11, "এগারটা": 11, "বারোটা": 12, "বারটা": 12, "বারোটার": 12,
}

# Regional spellings of the day-part markers used in dialectal notes.
DIALECT_MARKER = {
    "রাইত": "রাত", "বিয়ান": "সকাল", "বিয়ালে": "বিকাল", "অপরাহ্ন": "দুপুর",
    "পূর্বাহ্ন": "সকাল", "সইন্ধ্যা": "সন্ধ্যা",
}

BN_CLOCK_WORDS = {unicodedata.normalize("NFC", k): v for k, v in BN_CLOCK_WORDS.items()}
DIALECT_MARKER = {unicodedata.normalize("NFC", k): unicodedata.normalize("NFC", v)
                  for k, v in DIALECT_MARKER.items()}

# Connectors that mean "from"/"until" across the three languages.
FROM_WORDS = ["থেকে", "থাইকা", "তন", "theke", "thaika", "ton", "from"]
TO_WORDS = ["পর্যন্ত", "পয্যন্ত", "তক", "porjonto", "porjonto", "tok", "until", "to"]


def normalise_digits(text: str) -> str:
    return text.translate(BN_DIGITS)


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def canonicalise(text: str) -> str:
    """Rewrite dialectal and spelled-out forms into the canonical shapes the
    patterns below expect.  Runs before any matching so that every downstream
    rule sees one vocabulary instead of a dozen regional variants.

    Bengali text arrives in both NFC and NFD: য় is a single code point in one
    and য + nukta in the other, so the same word will not compare equal across
    sources unless it is normalised first.
    """
    out = normalise_digits(_nfc(text))

    # Regional day-part spellings → standard markers.
    for src, dst in DIALECT_MARKER.items():
        out = out.replace(src, dst)

    # Bengali clock words → "<n>টা".
    for word, val in sorted(BN_CLOCK_WORDS.items(), key=lambda kv: -len(kv[0])):
        out = re.sub(re.escape(word), f"{val}টা", out)

    # Dialectal connectors → standard ones.
    out = re.sub(r"থাইকা|\bতন\b", "থেকে", out)
    out = re.sub(r"পয্যন্ত|\bতক\b", "পর্যন্ত", out)
    out = re.sub(r"\bthaika\b", "theke", out, flags=re.IGNORECASE)

    # 4-digit military time: "0100 hours" → "01:00".
    out = re.sub(r"\b([01]\d|2[0-3])([0-5]\d)\s*hours?\b", r"\1:\2", out)

    return out


def resolve_hour(value: int, marker: Optional[str], is_end: bool = False) -> Optional[int]:
    """Map a clock number plus an optional day-part marker onto 0..24.

    24 is returned only for an end bound meaning midnight, and the caller turns
    that into a wrap or a stop at hour 23.
    """
    if value is None:
        return None
    bucket = DAYPART.get((marker or "").strip().lower()) if marker else None

    if bucket is None:
        if 0 <= value <= 24:
            return value
        return None

    if bucket == "am":
        # সকাল/ভোর never mean midnight: "shokal 12ta" is noon, not 00:00.
        # English "12 AM" is resolved by _parse_english_clock, not here.
        return 12 if value == 12 else value
    if bucket == "noon":
        # দুপুর ১২টা = 12, দুপুর ১টা = 13, দুপুর ৩টা = 15
        if value == 12:
            return 12
        if 1 <= value <= 5:
            return value + 12
        return value
    if bucket == "pm":
        if value == 12:
            return 12
        if value < 12:
            return value + 12
        return value
    if bucket == "night":
        # রাত ৮টা = 20 … রাত ১১টা = 23; রাত ১২টা = midnight; রাত ২টা = 02
        if value == 12:
            return 24 if is_end else 0
        if 5 <= value < 12:
            return value + 12
        if value < 5:
            return value
        return value
    return value


def _span(start: int, end: int) -> List[int]:
    """Expand a start-inclusive / end-exclusive span, handling wrap past midnight."""
    start %= 24
    if end == 24:
        return list(range(start, 24)) if start < 24 else []
    end %= 24
    if start == end:
        return [start]              # degenerate: the single hour named
    if start < end:
        return list(range(start, end))
    return sorted(list(range(start, 24)) + list(range(0, end)))


def _parse_english_clock(tok: str) -> Optional[int]:
    tok = tok.strip().lower()
    if tok in ("noon", "midday"):
        return 12
    if tok == "midnight":
        return 0
    m = re.match(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?$", tok)
    if not m:
        return None
    val = int(m.group(1))
    period = (m.group(3) or "").replace(".", "")
    if period == "pm" and val != 12:
        val += 12
    elif period == "am" and val == 12:
        val = 0
    return val if 0 <= val <= 24 else None


def extract_window(text: str) -> List[int]:
    """Return the hours a note refers to, ascending, or [] if none is found."""
    raw = canonicalise(text)
    low = raw.lower()

    # ── Whole day ────────────────────────────────────────────────────────
    if any(w in low for w in ALLDAY_WORDS):
        return list(range(24))

    # ── Bengali / Banglish "N hours starting at X" ───────────────────────
    m = re.search(
        rf"(?:({_MARKER_RE})\s*)?(\d{{1,2}})\s*(?:টা|ta)?\s*(?:থেকে|theke)\s*"
        rf"(?:শুরু\s*করে|shuru\s*kore)?\s*(?:পরবর্তী|আগামী|agami|porborti)?\s*"
        rf"(\d{{1,2}})\s*(?:ঘণ্টার|ঘণ্টা|ghontar|ghonta|hours?)",
        low,
    )
    if m:
        marker, start_s, dur_s = m.group(1), m.group(2), m.group(3)
        start = resolve_hour(int(start_s), marker)
        if start is not None:
            dur = int(dur_s)
            return sorted({(start + i) % 24 for i in range(max(1, min(24, dur)))})

    # ── English "for N hours starting/commencing at X" ────────────────────
    m = re.search(
        r"for\s+(\d+|one|two|three|four|five|six|a couple of|couple of)\s+hours?\s+"
        r"(?:starting|commencing|beginning)\s+(?:at|from)\s+"
        r"(\d{1,2}(?::\d{2})?\s*(?:am|pm)?|noon|midnight)",
        low,
    )
    if not m:
        m = re.search(
            r"(?:starting|commencing|beginning)\s+(?:at|from)\s+"
            r"(\d{1,2}(?::\d{2})?\s*(?:am|pm)?|noon|midnight)\s*,?\s*"
            r"(?:for\s+)?(?:the\s+next\s+)?(\d+|one|two|three|four|five|six|"
            r"a couple of|couple of)\s+hours?",
            low,
        )
        if m:
            start_tok, dur_tok = m.group(1), m.group(2)
        else:
            start_tok = dur_tok = None
    else:
        dur_tok, start_tok = m.group(1), m.group(2)
    if start_tok:
        start = _parse_english_clock(start_tok)
        dur = WORD_NUM.get(dur_tok, None) if not dur_tok.isdigit() else int(dur_tok)
        if start is not None and dur:
            return sorted({(start + i) % 24 for i in range(max(1, min(24, dur)))})

    # ── Reversed phrasing: "until 6 AM starting from 1 AM" ───────────────
    m = re.search(
        r"until\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?|noon|midnight)\s*,?\s*"
        r"(?:starting|beginning|commencing)\s+(?:from|at)\s+"
        r"(\d{1,2}(?::\d{2})?\s*(?:am|pm)?|noon|midnight)",
        low,
    )
    if m:
        end, start = _parse_english_clock(m.group(1)), _parse_english_clock(m.group(2))
        if start is not None and end is not None:
            return _span(start, end if end != 0 else 24)

    # ── Midnight → X ─────────────────────────────────────────────────────
    mid = "|".join(re.escape(w) for w in MIDNIGHT_WORDS)
    m = re.search(
        rf"(?:{mid})\s*(?:\([^)]*\))?\s*(?:থেকে|theke|to|until|till|through)\s*"
        rf"(?:({_MARKER_RE})\s*)?(\d{{1,2}})\s*(?:টা|ta|:00|am|pm)?",
        low,
    )
    if m:
        end = resolve_hour(int(m.group(2)), m.group(1), is_end=True)
        if end is not None:
            return _span(0, end if end != 0 else 24)

    # ── Sub-hour span inside one hour: "৯টা থেকে ৯:৩০", "9 to 9:30 AM" ────
    m = re.search(
        rf"(?:({_MARKER_RE})\s*)?(\d{{1,2}})\s*(?:টা|ta)?\s*(?::00)?\s*"
        rf"(?:থেকে|theke|to|until|-)\s*(\d{{1,2}}):(\d{{2}})",
        low,
    )
    if m and int(m.group(4)) != 0 and int(m.group(2)) == int(m.group(3)):
        h = resolve_hour(int(m.group(2)), m.group(1))
        if h is not None:
            return [h % 24]

    # ── Bengali / Banglish range, possibly several joined by "and again" ──
    bn_pat = (
        rf"(?:({_MARKER_RE})\s*)?(\d{{1,2}})\s*(?:টা|ta)\s*(?:থেকে|theke|to|until|-)\s*"
        rf"(?:({_MARKER_RE})\s*)?(\d{{1,2}})\s*(?:টা|ta)?"
    )
    if any(w in low for w in ["টা", "ta ", "theke", "থেকে", "porjonto", "পর্যন্ত"]):
        matches = list(re.finditer(bn_pat, low))
        multi = any(w in low for w in ["ebong", "এবং", "abar", "আবার", "and again", "o abar"])
        spans: List[int] = []
        for mm in (matches if multi else matches[:1]):
            m1, h1, m2, h2 = mm.group(1), int(mm.group(2)), mm.group(3), int(mm.group(4))
            start = resolve_hour(h1, m1 or m2)
            end = resolve_hour(h2, m2 or m1, is_end=True)
            if start is None or end is None:
                continue
            # No marker on either side and the end looks earlier than the start:
            # "11ta theke 1ta" means 11 AM to 1 PM, not an overnight wrap.
            if not m1 and not m2 and end < start and end + 12 > start:
                end += 12
            spans.extend(_span(start, end))
        if spans:
            return sorted(set(spans))

    # ── Bengali / Banglish "N টা থেকে <bare bound>" (dawn, noon, midnight) ─
    bare = "|".join(re.escape(w) for w in sorted(BARE_BOUND, key=len, reverse=True))
    m = re.search(
        rf"(?:({_MARKER_RE})\s*)?(\d{{1,2}})\s*(?:টা|ta)\s*(?:থেকে|theke|to|until)\s*"
        rf"(?:{bare})\b",
        low,
    )
    if m:
        start = resolve_hour(int(m.group(2)), m.group(1))
        tail = low[m.start():]
        end = next((v for k, v in BARE_BOUND.items() if k in tail), None)
        if start is not None and end is not None:
            return _span(start, end)

    m = re.search(
        rf"(?:({_MARKER_RE})\s*)?(\d{{1,2}})\s*(?:টা|ta)\s*(?:থেকে|theke|to|until)\s*"
        rf"(?:{mid})",
        low,
    )
    if m:
        start = resolve_hour(int(m.group(2)), m.group(1))
        if start is not None:
            return _span(start, 24)

    # ── English range: "X to/until/through/and Y", "between X and Y" ──────
    # "theke"/"porjonto" are included because code-switched notes pair English
    # clock tokens with Banglish connectors ("1 PM theke 3 PM").
    clock = r"\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)?|noon|midnight|midday"
    m = re.search(
        rf"(?:from|between|during)?\s*\b({clock})\s*"
        rf"(?:to|until|till|through|and|theke|–|—|-)\s*({clock})\b",
        low,
    )
    if m:
        a, b = _parse_english_clock(m.group(1)), _parse_english_clock(m.group(2))
        if a is not None and b is not None:
            # "10 to 3 PM": an unsuffixed start borrows the end's suffix.
            if not re.search(r"(am|pm)", m.group(1)) and re.search(r"pm", m.group(2)):
                if a < 12 and a != 12:
                    a += 12
            if "midnight" in m.group(2) and b == 0:
                b = 24
            if a != b or "midnight" in m.group(2):
                return _span(a, b)
            return [a % 24]

    # ── Single hour: "at 3 PM", "during the 15:00 hour" ──────────────────
    m = re.search(
        rf"(?:\bat\b|\bduring\s+(?:the\s+)?)\s*(?:({_MARKER_RE})\s*)?"
        rf"(\d{{1,2}})\s*(?::00)?\s*(am|pm)?\s*(?:টা|ta)?\s*(?:hour)?",
        low,
    )
    if m:
        marker = m.group(1)
        val = int(m.group(2))
        suffix = m.group(3)
        if suffix:
            h = _parse_english_clock(f"{val} {suffix}")
        else:
            h = resolve_hour(val, marker)
        if h is not None and 0 <= h < 24:
            return [h]

    # ── Bengali pair with no explicit "from", relying on পর্যন্ত ──────────
    # "অপরাহ্ন ২টায় … অপরাহ্ন ৫টা পর্যন্ত" states both bounds but joins them
    # with a participle rather than থেকে.
    if "পর্যন্ত" in low:
        toks = list(re.finditer(rf"(?:({_MARKER_RE})\s*)?(\d{{1,2}})\s*টা", low))
        if len(toks) >= 2:
            a_m, a_h = toks[0].group(1), int(toks[0].group(2))
            b_m, b_h = toks[1].group(1), int(toks[1].group(2))
            start = resolve_hour(a_h, a_m or b_m)
            end = resolve_hour(b_h, b_m or a_m, is_end=True)
            if start is not None and end is not None and start != end:
                return _span(start, end)

    # ── Last resort: "every hour"-style phrasing with no window at all ────
    if any(w in low for w in WEAK_ALLDAY_WORDS):
        return list(range(24))

    return []
