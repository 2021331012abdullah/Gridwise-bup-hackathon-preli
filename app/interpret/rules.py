"""L2 (deterministic channel) — regex/keyword interpretation.

This channel exists for three reasons:
  * it cross-checks the LLM, which is where factor inversion and hour
    off-by-ones are caught;
  * it supplies numbers and hours deterministically, so arithmetic never
    depends on the model;
  * it is the only channel that survives total network loss.

It must never be the sole interpretation path in normal operation — phrase
matching alone does not satisfy the LLM requirement in section 02 of the
Problem Statement — but it must be good enough to carry a request when the
provider is down.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional

from app.interpret.timeparse import canonicalise, extract_window

# ─── Vocabulary ──────────────────────────────────────────────────────────

def _n(words):
    return [unicodedata.normalize("NFC", w) for w in words]


# Banglish attaches possessive/locative suffixes straight onto English nouns
# ("Grider upor", "batteryte", "solarer"), which defeats word-boundary matching.
_BANGLISH_SUFFIX = re.compile(
    r"\b(grid|battery|solar|panel|inverter|charge|discharge|reserve|transformer|"
    r"substation|feeder|import)(er|e|te|ke|ta|r|tay|ey)\b",
    re.IGNORECASE,
)


def strip_banglish_suffix(text: str) -> str:
    return _BANGLISH_SUFFIX.sub(r"\1", text)


SOLAR_KW = _n([
    "solar", "pv", "photovoltaic", "panel", "array", "rooftop", "inverter",
    "string wash", "sunlight", "sun ", "irradiance", "soiling", "generation",
    "সোলার", "সৌর", "প্যানেল", "প্যানেলসমূহ", "রোদ", "সূর্য",
])
BATTERY_KW = _n([
    "battery", "batteries", "bess", "storage", "soc", "state of charge",
    "charge", "charging", "charger", "discharge", "discharging", "reserve",
    "cell", "accumulator", "stored", "stored energy", "ব্যাটারি", "ব্যাটারী", "চার্জ", "ডিসচার্জ",
    "রিজার্ভ", "সঞ্চয়", "ব্যাটারিতে", "ব্যাটারির",
])
GRID_KW = _n([
    "grid", "utility", "feeder", "transformer", "substation", "intake",
    "import", "draw", "busbar", "incomer", "mains", "গ্রিড", "আমদানি",
    "সাবস্টেশন", "ট্রান্সফরমার", "ফিডার", "বিদ্যুৎ টানা", "কারেন্ট",
])

# Words that turn a capability statement into a prohibition.
NEG_KW = _n([
    "do not", "don't", "must not", "cannot", "can't", "may not", "shall not",
    "no ", "not ", "never", "prohibit", "prohibited", "forbidden", "banned",
    "disable", "disabled", "unavailable", "isolated", "offline", "off ",
    "blocked", "prevent", "cease", "halt", "stop", "pause", "paused",
    "suspend", "suspended", "inhibit", "barred", "barring", "restrict",
    "shut down", "shutdown", "locked out", "out of service", "avoid",
    "disconnect", "disconnected", "refrain", "withhold",
    "বন্ধ", "নিষেধ", "নিষিদ্ধ", "যাবে না", "যাবেনা", "করা যাবে না", "পারবে না",
    "রাখুন", "রাখতে হবে", "বিরত", "স্থগিত", "নায়",
    "bondho", "jabe na", "jabena", "kora jabe na", "korben na", "nished",
    "bondho rakhun", "bondho thakbe", "prohibited", "bandho",
])

# Notes that describe campus life rather than the energy system.
DISTRACTOR_KW = _n([
    "cafeteria", "canteen", "menu", "library", "book return", "book-return",
    "registration", "deadline", "seminar", "club", "sports", "parking",
    "shuttle", "bus schedule", "exam", "convocation", "graduation",
    "lost and found", "tree plantation", "plantation", "painting", "repaint",
    "air condition", "air-condition", "hvac", "setpoint", "set point",
    "lighting", "perimeter", "security", "patrol", "cctv", "fire drill",
    "firmware patch", "was deployed", "was successfully", "servicing",
    "serviced", "visit", "lift", "elevator", "water heater", "fan ",
    "meter install", "smart energy meter", "swimming", "gym", "hostel",
    "ক্যাফেটেরিয়া", "ক্যান্টিন", "লাইব্রেরি", "নিবন্ধন", "সেমিনার", "ক্লাব",
    "খেলাধুলা", "পার্কিং", "বৃক্ষরোপণ", "রং করার", "শীতাতপ", "নিরাপত্তা",
    "মেনু", "সার্ভিসিং", "লিফট", "দাম বাড়ানো", "মূল্য",
])

# Time references that point outside today's 24-hour horizon.
FUTURE_KW = _n([
    "tomorrow", "next week", "next month", "next semester", "next quarter",
    "next year", "next monday", "next tuesday", "next wednesday",
    "next thursday", "next friday", "next saturday", "next sunday",
    "yesterday", "last week", "last night", "upcoming", "later this month",
    "আগামীকাল", "আগামী সপ্তাহ", "আগামী মাস", "পরের সপ্তাহ", "পরের মাস",
    "গতকাল", "পরশু", "আগামী", "agamikal", "porer shoptaho",
    "sesh hoyeche", "hoye geche",
])

# Bare Banglish "kal" (tomorrow) must be matched with word boundaries: as a
# substring it also fires inside "shokal" (morning) and "bikal" (afternoon),
# which would mark half the Banglish directives as future-scoped no_ops.
FUTURE_RE = re.compile(
    r"\b(?:kal|kaal|agami\s*kal|next\s+(?:week|month|semester|quarter|year))\b",
    re.IGNORECASE,
)

FRACTIONS = {
    "one-half": 0.5, "one half": 0.5, "half": 0.5, "অর্ধেক": 0.5,
    "one-third": 1 / 3, "one third": 1 / 3, "a third": 1 / 3,
    "two-thirds": 2 / 3, "two thirds": 2 / 3,
    "one-quarter": 0.25, "one quarter": 0.25, "a quarter": 0.25,
    "one-fourth": 0.25, "one fourth": 0.25, "a fourth": 0.25,
    "quarter": 0.25, "চার ভাগের এক ভাগ": 0.25, "এক-চতুর্থাংশ": 0.25,
    "এক চতুর্থাংশ": 0.25, "সিকি": 0.25,
    "three-quarters": 0.75, "three quarters": 0.75,
    "one-fifth": 0.2, "one fifth": 0.2, "a fifth": 0.2, "এক-পঞ্চমাংশ": 0.2,
    "two-fifths": 0.4, "three-fifths": 0.6, "four-fifths": 0.8,
}
FRACTIONS = {unicodedata.normalize("NFC", k): v for k, v in FRACTIONS.items()}

NUM_WORDS = {
    "zero": 0, "five": 5, "ten": 10, "fifteen": 15, "twenty": 20,
    "twenty-five": 25, "twenty five": 25, "thirty": 30, "thirty-five": 35,
    "forty": 40, "forty-five": 45, "fifty": 50, "fifty-five": 55,
    "sixty": 60, "sixty-five": 65, "seventy": 70, "seventy-five": 75,
    "eighty": 80, "eighty-five": 85, "ninety": 90, "ninety-five": 95,
    "hundred": 100, "one hundred": 100,
}

# Phrasings meaning "this much was removed" (the factor must be inverted).
REDUCTION_CUES = _n([
    "reduction", "reduced", "reduce", "reduces", "reduced by", "reduce by", "reduces by", "decrease by",
    "decreased by", "drop by", "dropped by", "drops by", "cut by", "down by",
    "curtail", "curtailed by", "curtail by", "depress", "depressed by",
    "de-rating", "derating", "loss of", "drop of", "dip", "fall of",
    "lower by", "shortfall", "কমে", "কম হবে", "হ্রাস", "কমিয়ে", "হ্রাস পাবে",
    "kome", "komano", "kom hobe", "kome jabe",
])
# Phrasings meaning "this much remains" (the factor is taken directly).
REMAINING_CUES = _n([
    "to about", "drop to", "dropped to", "drops to", "down to", "at about",
    "operate at", "run at", "yield", "produce", "produces", "remain at",
    "remaining", "of forecast", "of the forecast", "of normal", "of nominal",
    "of capacity", "of forecasted", "of rated", "merely", "only",
    "এ নেমে", "তে নেমে", "এ থাকবে", "এ আসবে", "এ পরিচালিত",
    "e neme", "e thakbe", "e ashbe", "produce korbe",
])

ZERO_SOLAR_CUES = _n([
    "completely off", "completely knocked out", "knocked out", "blackout",
    "completely fail", "complete failure", "fails completely", "total failure",
    "total eclipse", "zero output", "output zero", "no generation",
    "completely shut", "fully shut", "shut down entirely", "disconnect",
    "disconnected", "সম্পূর্ণ বন্ধ", "পুরোপুরি বন্ধ", "উৎপাদন বন্ধ", "শূন্য",
    "বিচ্ছিন্ন", "ekdom bondho", "completely bondho", "zero",
])
FULL_SOLAR_CUES = _n([
    "100% nominal", "full strength", "full capacity", "100 percent",
    "nominal output", "স্বাভাবিক পূর্ণ ক্ষমতা", "পূর্ণ ক্ষমতা", "পূর্ণ শক্তি",
])


def _compile_kw(words) -> "re.Pattern":
    """Build one matcher for a keyword list.

    ASCII keywords are anchored on word boundaries; matching them as bare
    substrings produces false positives that are hard to see and expensive to
    debug ("Chancellor" contains "cell", "Sustainability" contains "sustain",
    "shokal" contains "kal").  Bengali has no \b word boundary in the regex
    sense, so those keywords stay as plain substrings.
    """
    parts = []
    for w in words:
        w = unicodedata.normalize("NFC", w)
        esc = re.escape(w.strip())
        if not esc:
            continue
        if re.fullmatch(r"[\x00-\x7f\s\-']+", w.strip()):
            lead = r"\b" if w.strip()[0].isalnum() else ""
            trail = r"\b" if w.strip()[-1].isalnum() else ""
            parts.append(f"{lead}{esc}{trail}")
        else:
            parts.append(esc)
    return re.compile("|".join(parts), re.IGNORECASE) if parts else re.compile(r"(?!x)x")


_KW_CACHE: Dict[int, Any] = {}


def _has(text: str, words) -> bool:
    key = id(words)
    pat = _KW_CACHE.get(key)
    if pat is None:
        pat = _compile_kw(words)
        _KW_CACHE[key] = pat
    return bool(pat.search(text))


# ─── Numeric extraction ──────────────────────────────────────────────────

def _spelled_percent(text: str) -> Optional[float]:
    """Resolve "eighty percent" / "twenty percent" into a number."""
    for word, val in sorted(NUM_WORDS.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(word)}\s*(?:percent|per cent|%)", text):
            return float(val)
    return None


def extract_solar_factor(text: str) -> Optional[float]:
    """Return the fraction of solar that REMAINS, or None if not stated.

    The 'to' versus 'by' distinction is the highest-frequency semantic failure
    in this problem, so it is resolved here in code rather than left to the
    model: "reduced BY 80%" is factor 0.20, "reduced TO 80%" is factor 0.80.
    """
    low = strip_banglish_suffix(canonicalise(text).lower())

    if _has(low, FULL_SOLAR_CUES):
        return 1.0
    if _has(low, ZERO_SOLAR_CUES):
        return 0.0

    reduction = _has(low, REDUCTION_CUES)
    remaining = _has(low, REMAINING_CUES)

    # Word fractions: "one-fifth of normal output", "three quarters".
    for phrase, val in sorted(FRACTIONS.items(), key=lambda kv: -len(kv[0])):
        if phrase in low:
            # "reduce by three quarters" removes 0.75, leaving 0.25.
            if reduction and not remaining:
                return round(max(0.0, min(1.0, 1.0 - val)), 4)
            return val

    pct = _spelled_percent(low)
    if pct is None:
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent|per cent|শতাংশ)", low)
        if m:
            pct = float(m.group(1))
    if pct is None:
        return None

    # An explicit "to X%" always wins over a generic reduction cue.
    if re.search(rf"(?:to|at)\s+(?:about\s+|roughly\s+|approximately\s+|around\s+)?{pct:g}\s*(?:%|percent)", low):
        return round(max(0.0, min(1.0, pct / 100.0)), 4)
    if reduction:
        return round(max(0.0, min(1.0, 1.0 - pct / 100.0)), 4)
    return round(max(0.0, min(1.0, pct / 100.0)), 4)


def _all_numbers(text: str) -> List[float]:
    return [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)", text)]


def extract_reserve_kwh(text: str, capacity: float) -> Optional[float]:
    """Return the required battery floor in kWh, or None if not stated."""
    low = strip_banglish_suffix(canonicalise(text).lower())

    # "75% of battery rating (capacity 200 kWh)" — the percentage wins, because
    # the kWh figure in the note is the capacity, not the reserve.
    m_pct_of = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:%|percent|শতাংশ)\s*(?:of|এর|er)?\s*"
        r"(?:the\s+)?(?:total\s+|full\s+|rated\s+)?"
        r"(?:battery\s+)?(?:capacity|rating|ক্ষমতা|ধারণক্ষমতা)",
        low,
    )
    if m_pct_of:
        return round(float(m_pct_of.group(1)) / 100.0 * capacity, 4)

    if re.search(r"(?:full|total|entire)\s+(?:battery\s+)?capacity|পূর্ণ ধারণক্ষমতা", low):
        # A fraction qualifier in front of "total capacity" wins:
        # "three-quarters of total battery capacity" is 0.75 x capacity, not
        # the capacity itself.
        for phrase, val in sorted(FRACTIONS.items(), key=lambda kv: -len(kv[0])):
            if re.search(rf"{re.escape(phrase)}\s*(?:of|এর)?\s*(?:the\s+)?(?:total|full|entire)", low):
                return round(val * capacity, 4)
        m_cap = re.search(r"(\d+(?:\.\d+)?)\s*(?:kwh|kilowatt)", low)
        return round(float(m_cap.group(1)) if m_cap else capacity, 4)

    m = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:kwh|kw-h|kilowatt[- ]?hours?|kilowatt|"
        r"কিলোওয়াট-ঘণ্টা|কিলোওয়াট|ইউনিট)",
        low,
    )
    if m:
        return round(float(m.group(1)), 4)

    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent|শতাংশ)", low)
    if m:
        return round(float(m.group(1)) / 100.0 * capacity, 4)

    for phrase, val in FRACTIONS.items():
        if phrase in low:
            return round(val * capacity, 4)

    # "at least 120" with no unit, where 120 is clearly not a clock hour.
    m = re.search(r"(?:at least|minimum|no less than|কমপক্ষে|অন্তত|ন্যূনতম)\s*(\d+(?:\.\d+)?)", low)
    if m and float(m.group(1)) > 24:
        return round(float(m.group(1)), 4)
    return None


def extract_grid_cap_kwh(text: str) -> Optional[float]:
    """Return the hourly grid import ceiling in kWh, or None if not stated."""
    low = strip_banglish_suffix(canonicalise(text).lower())

    # Explicit total cut-off.
    if re.search(
        r"zero\s+grid|grid\s+(?:draw|import|intake)\s+zero|no\s+grid\s+(?:power|electricity|import)|"
        r"absolute\s+zero|complete\s+grid\s+cutoff|একদম বন্ধ|সম্পূর্ণ বন্ধ",
        low,
    ):
        return 0.0
    if re.search(r"(?<!\d)0(?:\.0+)?\s*(?:kwh|unit|units|ইউনিট)", low):
        return 0.0

    # Megawatt figures appear in substation notes: 0.09 MW = 90 kWh.
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:mw|megawatt)", low)
    if m:
        return round(float(m.group(1)) * 1000.0, 4)

    m = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:kwh|kw-h|kilowatt[- ]?hours?|units?|ইউনিট|কিলোওয়াট)",
        low,
    )
    if m:
        return round(float(m.group(1)), 4)

    m = re.search(
        r"(?:cap(?:ped)?\s+(?:at|to)|limit(?:ed)?\s+(?:at|to)|max(?:imum)?(?:\s+of)?|"
        r"exceed|surpass|above|beyond|সর্বোচ্চ|বেশি)\s*(\d+(?:\.\d+)?)",
        low,
    )
    if m:
        return round(float(m.group(1)), 4)
    return None


# ─── Classification ──────────────────────────────────────────────────────

def _is_distractor(low: str) -> bool:
    """True when the note is campus admin rather than an energy directive."""
    energy = _has(low, SOLAR_KW) or _has(low, BATTERY_KW) or _has(low, GRID_KW)
    if _has(low, DISTRACTOR_KW) and not energy:
        return True
    # Future-scoped notes never constrain today's horizon, even when they do
    # mention energy ("tariff hike next semester", "firmware patch yesterday").
    future = _has(low, FUTURE_KW) or bool(FUTURE_RE.search(low))
    if future and not re.search(r"\b(?:today|now|aj|আজ|আজকে)\b", low):
        return True
    # Demand/load is not adjustable in GridWise: only solar, battery and grid
    # import are.  A cap on campus demand is therefore not a max_grid_window.
    if re.search(r"(?:campus|total|facility|building|overall)\s+(?:demand|load|consumption)", low) \
       and not _has(low, GRID_KW):
        return True
    return False


def classify(text: str) -> tuple[str, float]:
    """Return (directive_type, confidence) for one note."""
    low = strip_banglish_suffix(canonicalise(text).lower())

    if _is_distractor(low):
        return "no_op", 0.75

    has_solar = _has(low, SOLAR_KW)
    has_batt = _has(low, BATTERY_KW)
    has_grid = _has(low, GRID_KW)
    negated = _has(low, NEG_KW)

    # Reserve before charge/discharge: "keep at least 120 kWh in the battery"
    # mentions the battery but is a floor, not a window.
    reserve_cue = re.search(
        r"at least|no less than|minimum|minimal|reserve|maintain|keep|sustain|"
        r"lock in|hold|store|not\s+(?:drop|dip|fall)\s+below|above|"
        r"কমপক্ষে|অন্তত|ন্যূনতম|রিজার্ভ|নিচে নামতে|বজায়|রাখতে হবে|"
        r"rakhte hobe|kompokkhe|at least",
        low,
    )
    if has_batt and reserve_cue and not re.search(
        r"charg|discharg|চার্জ|ডিসচার্জ", low
    ):
        return "minimum_battery_reserve", 0.80
    if has_batt and reserve_cue and re.search(
        r"(?:reserve|soc|state of charge|stored|রিজার্ভ|energy)", low
    ) and not negated:
        return "minimum_battery_reserve", 0.75

    # Grid cap.
    grid_cue = re.search(
        r"exceed|surpass|cap\b|capped|limit|max|ceiling|allowance|above|beyond|"
        r"below|under\s|within|stay at|not\s+go\s+over|"
        r"সর্বোচ্চ|সীমাবদ্ধ|বেশি|limit koren|cap rakhte",
        low,
    )
    grid_zero_cue = re.search(
        r"zero\s+grid|grid\s+.{0,20}zero|no\s+grid\s+(?:electricity|power|import|energy|current)|"
        r"grid\s+(?:electricity|power)?\s*.{0,12}(?:may not|must not|cannot|is to be|should)?\s*"
        r"(?:be\s+)?draw|একদম বন্ধ|সম্পূর্ণ বন্ধ|complete grid cutoff|ekdom bondho",
        low,
    )
    if has_grid and (grid_cue or grid_zero_cue):
        return "max_grid_window", 0.80

    # Discharge before charge: the substring "charge" lives inside "discharge".
    dis_re = r"discharg|ডিসচার্জ|খরচ|power\s+draw|current\s+out|কারেন্ট বাইর|বিদ্যুৎ খরচ"
    chg_re = r"(?<!dis)charg|চার্জ"
    has_dis = re.search(dis_re, low)
    has_chg = re.search(chg_re, low) and not re.search(r"ডিসচার্জ", low)

    if has_dis and has_chg and negated:
        # Both verbs present: the prohibition attaches to whichever verb it sits
        # closest to.  "charge kora jabe na, just discharge e thakbe" bans
        # charging only; "barring any charging" likewise.
        neg_pos = [low.find(w) for w in NEG_KW if w in low]
        neg_pos = [p for p in neg_pos if p >= 0]
        if neg_pos:
            def nearest(pattern: str) -> int:
                spots = [m.start() for m in re.finditer(pattern, low)]
                if not spots:
                    return 10 ** 6
                return min(abs(s - n) for s in spots for n in neg_pos)

            if nearest(chg_re) < nearest(dis_re):
                return "no_charge_window", 0.70
            return "no_discharge_window", 0.70

    if has_dis and negated:
        return "no_discharge_window", 0.80
    if has_chg and negated:
        return "no_charge_window", 0.80

    # Solar last: many notes mention solar only as the reason for another rule.
    if has_solar:
        # A solar note with no level and no window is a status remark, not a
        # directive ("ensure the inverters remain locked after maintenance").
        if extract_solar_factor(text) is None and not _has(low, REDUCTION_CUES):
            return "no_op", 0.50
        return "solar_reduction", 0.75

    if has_batt and negated:
        # Battery restricted but the direction is unstated; discharging is the
        # more common restriction and the safer assumption for a plan.
        return "no_discharge_window", 0.45

    return "no_op", 0.40


def rule_interpret(note: str, capacity: float) -> Dict[str, Any]:
    """Full deterministic interpretation of one note."""
    dtype, conf = classify(note)
    hours = extract_window(note)

    if dtype == "no_op":
        return {
            "directive_type": "no_op",
            "applies": False,
            "structured_adjustment": None,
            "confidence": conf,
            "explanation": "The note does not affect today's 24-hour energy schedule.",
        }

    adj: Dict[str, Any] = {"hours": hours}

    if dtype == "solar_reduction":
        factor = extract_solar_factor(note)
        if factor is None:
            # No stated level.  Leave it unset so the guardrail can source the
            # number elsewhere; inventing 1.0 here would mean "no reduction"
            # and would let the plan consume solar that does not exist.
            adj["factor"] = None
            conf *= 0.4
        else:
            adj["factor"] = factor
    elif dtype == "minimum_battery_reserve":
        reserve = extract_reserve_kwh(note, capacity)
        adj["minimum_energy_kwh"] = reserve
        if reserve is None:
            conf *= 0.4
    elif dtype == "max_grid_window":
        cap_val = extract_grid_cap_kwh(note)
        adj["max_grid_kwh"] = cap_val
        if cap_val is None:
            conf *= 0.4

    if not hours:
        conf *= 0.3

    return {
        "directive_type": dtype,
        "applies": True,
        "structured_adjustment": adj,
        "confidence": round(conf, 4),
        "explanation": f"Deterministic rule channel classified this note as {dtype}.",
    }
