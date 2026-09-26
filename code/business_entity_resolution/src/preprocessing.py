"""Rule-based text normalization for business names and addresses.

No external lookups: every rule below is a static mapping or regex. Mappings are
dictionaries (token -> canonical form), so unseen variants simply pass through
unchanged instead of being dropped.

Two levels of output:
  * normalize_business_name / normalize_address -> readable, lossless-ish canonical
    text (lowercase, ASCII-folded, abbreviations expanded). Used by later stages.
  * blocking_key -> a more aggressive but *symmetric* transform used only for sparse
    blocking (repeated-letter collapse), so typos like "Barnnesville" / "stret" match.

`transliterate=False` keeps non-Latin scripts intact (for the multilingual dense model)
while still applying the Latin-side rules.
"""

from __future__ import annotations

import re
import unicodedata

from unidecode import unidecode

# ---------------------------------------------------------------------------
# Mapping tables
# ---------------------------------------------------------------------------

# Legal-form / business-word variants -> canonical token. Applied to name tokens.
LEGAL_SUFFIX_MAP = {
    # corporation / incorporated
    "corp": "corporation", "corpn": "corporation", "corpo": "corporation",
    "inc": "incorporated", "incorp": "incorporated", "incorported": "incorporated",
    # limited / private
    "ltd": "limited", "lmtd": "limited", "limted": "limited",
    "pvt": "private", "pvte": "private", "prvt": "private", "pte": "private", "prv": "private",
    # company
    "co": "company", "comp": "company", "cmpny": "company", "cie": "company", "compagnie": "company",
    # common business-word abbreviations
    "intl": "international", "int'l": "international",
    "assoc": "association", "assn": "association", "asso": "association",
    "mfg": "manufacturing", "svc": "services", "svcs": "services",
    "bros": "brothers", "grp": "group", "mgmt": "management",
    "ctr": "center", "cntr": "center", "centre": "center",
    "dept": "department", "univ": "university", "hosp": "hospital",
    "ets": "etablissements", "etabl": "etablissements",
}

# Legal words written in Indic scripts, as they come out of unidecode + repeated-letter
# collapse (e.g. प्राइवेट -> "praivet", লিমিটেড -> "limited", "प्रा. लि." -> "pra li").
# Applied only to tokens that were non-Latin originally, so Latin "Li" stays "li".
TRANSLIT_TOKEN_MAP = {
    "praivet": "private", "praibhet": "private", "piraivet": "private", "praivr": "private",
    "pra": "private", "limited": "limited", "limitet": "limited", "limird": "limited",
    "li": "limited", "elelpi": "llp", "kmpni": "company", "kampni": "company",
}

# Multi-word legal phrases -> canonical token(s). Applied to the name string.
LEGAL_PHRASE_MAP = {
    "limited liability company": "llc",
    "limited liability partnership": "llp",
    "limited partnership": "lp",
    "professional limited liability company": "pllc",
    "doing business as": "dba",
    "trading as": "ta",
}

# Address abbreviations -> canonical token (applied to all countries).
ADDRESS_ABBREV_MAP = {
    "rd": "road", "st": "street", "str": "street", "strt": "street",
    "ave": "avenue", "av": "avenue", "avn": "avenue", "aven": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "boul": "boulevard", "bvd": "boulevard",
    "dr": "drive", "drv": "drive", "ln": "lane", "ct": "court", "crt": "court",
    "cir": "circle", "pl": "place", "plz": "plaza", "pkwy": "parkway", "pky": "parkway",
    "hwy": "highway", "fwy": "freeway", "expy": "expressway", "sq": "square",
    "ter": "terrace", "terr": "terrace", "trl": "trail", "tr": "trail", "cres": "crescent",
    "xing": "crossing", "mt": "mount", "ft": "fort", "pt": "point",
    "apt": "apartment", "apts": "apartments", "ste": "suite", "fl": "floor", "flr": "floor",
    "bldg": "building", "blk": "block", "rm": "room", "dept": "department",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "opp": "opposite", "nr": "near", "nagar": "nagar", "mkt": "market",
    "no": "no", "num": "no", "number": "no", "hno": "house no", "dno": "door no",
    "po": "po", "pob": "po box",
}

# Country-specific overrides on top of ADDRESS_ABBREV_MAP. Keys are casefolded
# country labels; any country not listed (including unseen ones) uses the defaults.
COUNTRY_ADDRESS_OVERRIDES = {
    "france": {
        "st": "saint", "ste": "sainte",
        "r": "rue", "ch": "chemin", "che": "chemin", "imp": "impasse", "rte": "route",
        "fg": "faubourg", "fbg": "faubourg", "qu": "quai", "qua": "quai", "crs": "cours",
        "pass": "passage", "res": "residence", "lot": "lotissement",
        "zi": "zone industrielle", "za": "zone artisanale",
    },
}

# Whole address components that carry no information ("<NULL>", "N/A", ...).
ADDRESS_PLACEHOLDERS = {"null", "none", "nan", "n/a", "na", "nil", "-", "--", "unknown", "not available"}

ORDINAL_WORDS = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
    "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
    "eleventh": "11", "twelfth": "12",
}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi",
    "wyoming": "wy", "district of columbia": "dc", "puerto rico": "pr",
}

INDIA_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "tamilnadu": "tn",
    "telangana": "tg", "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk",
    "west bengal": "wb", "delhi": "dl", "nct of delhi": "dl", "jammu and kashmir": "jk",
    "chandigarh": "ch", "puducherry": "py", "pondicherry": "py", "ladakh": "la",
}

# Native-script state names observed in Indian addresses -> state code.
NATIVE_REGION_MAP = {
    "महाराष्ट्र": "mh", "दिल्ली": "dl", "उत्तर प्रदेश": "up", "ಕರ್ನಾಟಕ": "ka",
    "தமிழ்நாடு": "tn", "পশ্চিমবঙ্গ": "wb", "ગુજરાત": "gj", "తెలంగాణ": "tg",
    "हरियाणा": "hr", "राजस्थान": "rj", "കേരളം": "kl", "बिहार": "br",
    "मध्य प्रदेश": "mp", "ఆంధ్రప్రదేశ్": "ap", "ਪੰਜਾਬ": "pb", "ଓଡ଼ିଶା": "od",
}

# Region-name -> code maps per country label. Unlisted countries get no region map.
COUNTRY_REGION_MAPS = {"us": US_STATES, "india": INDIA_STATES}

# ---------------------------------------------------------------------------
# Compiled regexes
# ---------------------------------------------------------------------------

_C1_CONTROLS = re.compile(r"[\u0080-\u009fÂ]")  # mojibake debris (e.g. "Â\x80\x93")
_APOSTROPHES = re.compile(r"['’`´]")
_DOTTED_ACRONYM = re.compile(r"\b(?:[a-z]\.){2,}[a-z]?\.?")
_SLASH_ABBREV = {  # before punctuation stripping
    re.compile(r"\bd\s*/\s*b\s*/\s*a\b"): " dba ",
    re.compile(r"\bt\s*/\s*a\b"): " ta ",
    re.compile(r"\bc\s*/\s*o\b"): " careof ",
    re.compile(r"\bs\s*/\s*o\b"): " sonof ",
}
_NUMERO = re.compile(r"\bn\s*[°º]|\bn[o°º]\s*\.")  # "N°", "No." -> "no"
# \w misses combining marks, so keep Indic vowel signs / Latin diacritics explicitly.
_NON_ALNUM = re.compile(r"[^\w\u0300-\u036f\u0900-\u0dff\u1cd0-\u1cff\ua8e0-\ua8ff]+|_")
_ORDINAL_SUFFIX = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")
_LEADING_ZEROS = re.compile(r"\b0+(\d)")
_REPEATED_LETTERS = re.compile(r"([a-z])\1+")
_NON_LATIN = re.compile(r"[^\x00-ɏ]")  # anything outside Basic Latin + Latin Ext


def _phrase_regex(mapping: dict) -> re.Pattern:
    keys = sorted(mapping, key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b")


_LEGAL_PHRASE_RE = _phrase_regex(LEGAL_PHRASE_MAP)
_REGION_RES = {c: (_phrase_regex(m), m) for c, m in COUNTRY_REGION_MAPS.items()}
_NATIVE_REGION = {unicodedata.normalize("NFKC", k): v for k, v in NATIVE_REGION_MAP.items()}
_NATIVE_REGION_RE = re.compile("|".join(re.escape(k) for k in sorted(_NATIVE_REGION, key=len, reverse=True)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_country(country) -> str:
    """Open-set country label: casefolded, whitespace-trimmed. Empty -> ''."""
    if country is None or country != country:  # None / NaN
        return ""
    return str(country).strip().casefold()


def _fold_latin_accents(text: str) -> str:
    """Strip combining marks only when they sit on a Latin base character, so Indic
    vowel signs (also combining marks) survive when transliteration is off."""
    out = []
    for ch in unicodedata.normalize("NFD", text):
        if unicodedata.combining(ch) and out and out[-1] < "ɐ":
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def _to_script(token: str, transliterate: bool) -> str:
    """Transliterate a non-Latin token to ASCII (and collapse the doubled vowels /
    consonants that unidecode emits for Indic scripts)."""
    if not transliterate or not _NON_LATIN.search(token):
        return token
    t = _REPEATED_LETTERS.sub(r"\1", unidecode(token).lower())
    core = t.strip(".,()")
    return TRANSLIT_TOKEN_MAP.get(core, t) if core else t


def _base_clean(text, transliterate: bool) -> str:
    if text is None or text != text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = _C1_CONTROLS.sub(" ", text)
    text = _fold_latin_accents(text).lower()
    text = _APOSTROPHES.sub("", text)
    text = _NUMERO.sub(" no ", text)
    text = text.replace("&", " and ").replace("@", " at ")
    for pat, rep in _SLASH_ABBREV.items():
        text = pat.sub(rep, text)
    text = _DOTTED_ACRONYM.sub(lambda m: m.group(0).replace(".", ""), text)
    if transliterate:
        text = " ".join(_to_script(t, True) for t in text.split())
        text = unidecode(text)  # any residual symbols (°, ™, ...)
    return text


def _tokens(text: str) -> list[str]:
    return [t for t in _NON_ALNUM.sub(" ", text).split() if t]


def _dedupe_consecutive(tokens: list[str]) -> list[str]:
    """Drop immediately repeated words ("health health"), a common injected noise."""
    out = []
    for t in tokens:
        if not out or out[-1] != t or not t.isalpha():  # never drop numbers
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_business_name(name, transliterate: bool = True) -> str:
    text = _base_clean(name, transliterate)
    toks = [LEGAL_SUFFIX_MAP.get(t, t) for t in _tokens(text)]
    text = _LEGAL_PHRASE_RE.sub(lambda m: LEGAL_PHRASE_MAP[m.group(1)], " ".join(toks))
    return " ".join(_dedupe_consecutive(text.split()))


def normalize_address(address, country=None, transliterate: bool = True) -> str:
    c = normalize_country(country)
    if address is None or address != address:
        return ""
    text = unicodedata.normalize("NFKC", str(address))
    text = ",".join(p for p in text.split(",")
                    if p.strip().strip("<>[]()").casefold() not in ADDRESS_PLACEHOLDERS)
    text = _NATIVE_REGION_RE.sub(lambda m: " " + _NATIVE_REGION[m.group(0)] + " ", text)
    text = _base_clean(text, transliterate)
    abbrev = ADDRESS_ABBREV_MAP
    if c in COUNTRY_ADDRESS_OVERRIDES:
        abbrev = {**ADDRESS_ABBREV_MAP, **COUNTRY_ADDRESS_OVERRIDES[c]}
    toks = []
    for t in _tokens(text):
        t = _LEADING_ZEROS.sub(r"\1", t)
        t = _ORDINAL_SUFFIX.sub(r"\1", t)
        t = ORDINAL_WORDS.get(t, t)
        toks.append(abbrev.get(t, t))
    text = " ".join(toks)
    if c in _REGION_RES:
        pat, mapping = _REGION_RES[c]
        text = pat.sub(lambda m: mapping[m.group(1)], text)
    return " ".join(_dedupe_consecutive(text.split()))


def light_name(name) -> str:
    """Cross-encoder input: removes encoding noise only (Unicode/mojibake, case, Latin
    accents, native scripts -> ASCII, punctuation). Legal forms and abbreviations stay
    as written ("pvt ltd", "corp", "inc"), so the model learns their equivalence itself."""
    return " ".join(_tokens(_base_clean(name, transliterate=True)))


def light_address(address) -> str:
    """Cross-encoder input: placeholder components dropped, each comma component cleaned
    like light_name; no abbreviation, ordinal, region or leading-zero rewriting."""
    if address is None or address != address:
        return ""
    comps = (" ".join(_tokens(_base_clean(p, transliterate=True)))
             for p in unicodedata.normalize("NFKC", str(address)).split(",")
             if p.strip().strip("<>[]()").casefold() not in ADDRESS_PLACEHOLDERS)
    return ", ".join(c for c in comps if c)


def blocking_key(normalized_text: str) -> str:
    """Aggressive-but-symmetric transform for sparse blocking only: collapse repeated
    letters so insertion/deletion typos of doubled letters become exact n-grams."""
    return _REPEATED_LETTERS.sub(r"\1", normalized_text)


def sparse_text(name: str, address: str, country=None) -> str:
    """Text fed to the char n-gram TF-IDF (ASCII, normalized, blocking-keyed)."""
    n = normalize_business_name(name)
    a = normalize_address(address, country)
    return blocking_key(f"{n} {a}".strip())


def dense_text(name: str, address: str, country=None) -> str:
    """Text fed to BGE-M3: same rules but native scripts kept (the model is multilingual)."""
    n = normalize_business_name(name, transliterate=False)
    a = normalize_address(address, country, transliterate=False)
    return f"{n}, {a}" if a else n


def normalize_records(names, addresses, countries, need_dense: bool = True
                      ) -> tuple[list, list, list, list, list]:
    """Batch helper (used from worker processes): returns
    (norm_name, norm_address, sparse_text, dense_text, sparse_name_text) lists."""
    nn, na, st, dt, sn = [], [], [], [], []
    for name, addr, ctry in zip(names, addresses, countries):
        n = normalize_business_name(name)
        a = normalize_address(addr, ctry)
        nn.append(n)
        na.append(a)
        st.append(blocking_key(f"{n} {a}".strip()))
        if need_dense:
            n2 = normalize_business_name(name, transliterate=False)
            a2 = normalize_address(addr, ctry, transliterate=False)
            dt.append(f"{n2}, {a2}" if a2 else n2)
        sn.append(blocking_key(n))
    return nn, na, st, dt, sn
