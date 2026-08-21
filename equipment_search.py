"""
Natural-language equipment search over already-scraped new/CPO inventory.

Prototype for a real ask: dealer.com's own inventory filters only expose
coarse facets (trim, color, price, drivetrain) -- there's no way to search
for "has a center console, not a folding bench" the way a shopper actually
thinks about a truck. But that level of detail already lives in the data:
imaging/sticker.py parses the real Ford Monroney window sticker into a
structured equipment grid (interior/exterior/functional/safety) plus an
optional-equipment list, with real literal strings like "Center Console",
"Split Fold Rear Seat", "60/40 Easyfold Rear Bench" -- confirmed against
the actual fleet on disk, not assumed.

Two pieces, deliberately separated by the CONCEPT_SYNONYMS/compile_query
boundary:

  1. The matcher (match_vehicle/search) -- deterministic, zero marginal
     cost, works over however the query got compiled.
  2. The query compiler (compile_query) -- turns "a center console and no
     folding bench" into structured include/exclude concepts. THIS
     PROTOTYPE compiles with a small curated synonym dictionary
     (CONCEPT_SYNONYMS) plus simple negation/connective parsing, not an
     LLM call -- no Anthropic API key is configured in this repo
     (confirmed: no ANTHROPIC_API_KEY in .env, no anthropic package
     installed), so this is the zero-setup-cost version. Swapping
     compile_query() for a real small-model call later is a contained
     change: same SearchFilter contract, wider vocabulary coverage than
     any hand-curated dictionary could reach, at a fraction-of-a-cent per
     query. The dictionary here is intentionally narrow (seating/console/
     bench, the case that motivated this) rather than a guess at
     exhaustive automotive vocabulary.

Searches vehicles with a real parsed window sticker (window-sticker.json)
AND/OR a vision-extracted seat-config claim (vision-equipment.json, see
imaging/seat_vision.py) -- a vehicle with neither is excluded, not
searched with an empty blob. The sticker alone was a real, confirmed gap:
Ford's own template doesn't always print every trim's front-seat
configuration explicitly (a real F-350 Lariat's sticker lists "Htd/
Ventilated Frt Seats" but never says bucket vs bench/console at all for
that trim), and used vehicles mostly have no sticker at all. Vision
extraction closes both: it answers directly from a real interior photo
(confirmed against sticker-verified ground truth before being wired in --
2/2 correct at high confidence, and the one genuinely ambiguous photo was
correctly flagged "can't tell" rather than guessed), and it doesn't need a
sticker to exist in the first place, so it's the only source of equipment
search coverage for used vehicles that never had one.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Query compilation (the swappable part)
# --------------------------------------------------------------------------

# concept -> literal substrings actually seen (or clearly implied) in real
# Ford Monroney sticker text. Case-insensitive substring match. Grounded in
# a real vocabulary survey of the fleet on disk (see module docstring) --
# extend this as real stickers turn up terms it doesn't cover yet.
CONCEPT_SYNONYMS: dict[str, list[str]] = {
    "console": ["console"],
    "bucket seat": ["bucket"],
    "bench seat": ["bench", "split fold", "split-fold", "flip-up", "flip up",
                   "40/20/40", "60/40", "50/50"],
    "captain's chairs": ["captain"],
    "third row": ["3rd row", "third row", "3rd-row"],
    "heated seats": ["heated seat", "htd seat", "htd/ventilated", "heated, "],
    "ventilated seats": ["ventilated", "cooled seat"],
    "leather": ["leather"],
    "moonroof": ["moonroof", "sunroof"],
    "navigation": ["navigation", "nav system"],
    "tow package": ["tow", "trailer tow", "5th wheel", "gooseneck"],
    "running boards": ["running board", "side step"],
    "bedliner": ["bedliner", "bed liner"],
    "tonneau cover": ["tonneau"],
    "4wd": ["4x4", "four wheel drive", "4wd"],
    "diesel": ["power stroke", "diesel", "duramax", "cummins"],
    "remote start": ["remote start"],
    "power seat": ["power driv", "power pass", "pwr driv", "pwr pass", "power seat"],
}

_NEGATION_RE = re.compile(r"\b(no|not|without)\s+(.+?)(?=(?:,| and |$))", re.IGNORECASE)
_TRIM_WORDS = {
    "xl", "xlt", "lariat", "king ranch", "platinum", "limited", "raptor", "badlands",
    "big bend", "outer banks", "sasquatch", "tremor", "sport", "premium", "select",
    "se", "sel", "sport utility", "base", "custom", "luxe", "n line", "st line",
    "sle", "denali", "at4", "z71", "trail boss", "high country",
}


@dataclass
class SearchFilter:
    model_terms: list[str] = field(default_factory=list)  # AND, matched against title/model/trim
    include: list[list[str]] = field(default_factory=list)  # AND across concepts, OR within
    exclude: list[list[str]] = field(default_factory=list)  # none of these concepts may match


_GENERIC_CONCEPT_WORDS = {"seat", "seats", "row"}


def _concept_terms(phrase: str) -> list[str]:
    """The literal substrings for whichever CONCEPT_SYNONYMS key `phrase`
    names, or [phrase] itself as a literal fallback if it doesn't match a
    known concept -- so an unrecognized-but-specific term (e.g. a literal
    trim package name) still works as a plain substring search rather than
    silently doing nothing.

    Matched by WORD overlap with the concept name, not full-phrase
    containment -- "folding bench" needs to hit the "bench seat" concept,
    and "folding bench" is not a substring of "bench seat" (nor the other
    way around); they just share the word "bench". Overlap is checked
    against the concept's MEANINGFUL words only (generic connectors like
    "seat"/"seats"/"row" don't count on their own) -- confirmed real bug
    without this: "leather seats" matched the "heated seats" concept
    purely because both phrases contain "seats", which silently searched
    for heated seats instead of leather."""
    phrase = phrase.strip().lower()
    phrase_words = set(re.findall(r"[a-z0-9/]+", phrase))
    for concept, terms in CONCEPT_SYNONYMS.items():
        concept_words = set(concept.split())
        meaningful = concept_words - _GENERIC_CONCEPT_WORDS or concept_words
        if meaningful & phrase_words:
            return terms
    return [phrase] if phrase else []


def compile_query(nl_query: str) -> SearchFilter:
    """Rule-based NL -> SearchFilter. See module docstring for why this
    isn't an LLM call in this prototype."""
    text = nl_query.strip()
    filt = SearchFilter()

    # Model/trim words pulled out first so they don't get treated as
    # equipment phrases (e.g. "XLT" isn't a concept, it's a trim filter).
    words_lower = text.lower()
    for trim in _TRIM_WORDS:
        if re.search(rf"\b{re.escape(trim)}\b", words_lower):
            filt.model_terms.append(trim)
    model_re = re.compile(r"\b([a-z]?-?\d{2,3}(?:sd)?|[a-z]{2,}(?:-\d{3})?)\b", re.IGNORECASE)
    # Common body-style/model tokens worth keeping as-is if present (F-150,
    # F-250, Bronco, Explorer, ...) -- anything alphanumeric that isn't a
    # stopword and isn't already captured as a trim word.
    stopwords = {"a", "an", "and", "the", "with", "no", "not", "without", "has", "have"}
    for m in re.finditer(r"\bF-?\d{3}(?:SD)?\b", text, re.IGNORECASE):
        filt.model_terms.append(m.group(0).lower())

    # Negated phrases -> exclude concepts. Strip them out of the text so
    # the positive pass below doesn't also treat them as include concepts.
    remainder = text
    for m in _NEGATION_RE.finditer(text):
        phrase = m.group(2).strip()
        terms = _concept_terms(phrase)
        if terms:
            filt.exclude.append(terms)
        remainder = remainder.replace(m.group(0), "")

    # Whatever's left, split on connectives, minus trim/model words and
    # stopwords already accounted for -> include concepts.
    for chunk in re.split(r",| and ", remainder, flags=re.IGNORECASE):
        chunk = chunk.strip().lower()
        for trim in filt.model_terms:
            chunk = chunk.replace(trim, "")
        words = [w for w in chunk.split() if w not in stopwords]
        phrase = " ".join(words).strip()
        if not phrase or re.fullmatch(r"f-?\d{3}(sd)?", phrase, re.IGNORECASE):
            continue
        terms = _concept_terms(phrase)
        if terms:
            filt.include.append(terms)

    return filt


# --------------------------------------------------------------------------
# Fleet loading + matching
# --------------------------------------------------------------------------

# "Leather Wrapped Str Wheel" / "...Wrap Str Whl..." lines describe the
# steering wheel's trim, not the seats -- confirmed real false positive
# without this filter: 4 real vehicles with synthetic (ActiveX/vinyl) seat
# material matched a "leather seats" query purely because their sticker
# also lists a leather-wrapped wheel, which no shopper is actually asking
# about. Genuine leather SEATING is disclosed via the sticker's own
# overview.interior_trim field (see below), which this doesn't touch.
_WHEEL_TRIM_RE = re.compile(r"wrap.{0,10}(str\.?|steering).{0,10}wh(ee)?l", re.IGNORECASE)


# A vision-extracted claim only counts if the model itself was confident
# -- "unclear" is never included (that's the model saying it doesn't
# know), and "low" confidence isn't trusted as a positive search signal
# either, only "high"/"medium".
_VISION_CONFIG_TERMS = {"bucket_console": "console", "bench": "bench"}
_TRUSTED_VISION_CONFIDENCE = {"high", "medium"}


def load_searchable_fleet(root: Path) -> list[dict]:
    """Every vehicle on disk with EITHER a real parsed window sticker OR a
    vision-extracted seat-config claim (imaging/seat_vision.py) -- a
    vehicle with neither has nothing this module can search against, so
    it's excluded, not included with an empty blob that would silently
    fail every query. Each record carries a flattened, lowercased blob of
    every known equipment/optional-equipment/vision-claim string for cheap
    substring matching, plus the fields a result listing wants to show."""
    root = Path(root)
    out = []
    for details_path in sorted(root.glob("*/*/details.json")):
        folder = details_path.parent
        sticker_path = folder / "window-sticker.json"
        vision_path = folder / "vision-equipment.json"
        if not sticker_path.is_file() and not vision_path.is_file():
            continue
        try:
            details = json.loads(details_path.read_text())
        except (OSError, ValueError):
            continue

        all_terms = []
        if sticker_path.is_file():
            try:
                sticker = json.loads(sticker_path.read_text())
            except (OSError, ValueError):
                sticker = {}
            equipment = sticker.get("equipment", {})
            overview = sticker.get("overview", {})
            # Real gap found while building this: seat MATERIAL (leather vs
            # cloth) isn't in the equipment grid at all on this fleet's real
            # stickers -- it's disclosed as the sticker's overview.interior_trim
            # field instead (confirmed: a King Ranch Expedition's only "leather"
            # hits in the equipment grid are a leather-WRAPPED STEERING WHEEL
            # and console trim; its actual leather SEATING only shows up as
            # interior_trim="Mesa Del Rio Leather"). Equipment-grid text alone
            # would silently miss every genuine leather-seat vehicle on a
            # "leather seats" query.
            all_terms += ([t for cat in equipment.values() for t in cat] + sticker.get("optional_equipment", [])
                          + [overview.get("interior_trim") or "", overview.get("exterior_color") or ""])
            all_terms = [t for t in all_terms if t and not _WHEEL_TRIM_RE.search(t)]

        if vision_path.is_file():
            try:
                vision = json.loads(vision_path.read_text())
            except (OSError, ValueError):
                vision = {}
            term = _VISION_CONFIG_TERMS.get(vision.get("config"))
            if term and vision.get("confidence") in _TRUSTED_VISION_CONFIDENCE:
                all_terms.append(term)

        v = details.get("vehicle", details)
        out.append({
            "folder": str(folder),
            "title": v.get("title"),
            "year": v.get("year"), "make": v.get("make"), "model": v.get("model"), "trim": v.get("trim"),
            "condition": v.get("condition"),
            "price": v.get("display_price"),
            "url": v.get("url"),
            "equipment_text": " | ".join(all_terms).lower(),
            "equipment_list": all_terms,
        })
    return out


def match_vehicle(record: dict, filt: SearchFilter) -> bool:
    title_blob = " ".join(str(record.get(k) or "") for k in ("title", "model", "trim")).lower()
    for term in filt.model_terms:
        if term not in title_blob:
            return False
    for concept_terms in filt.include:
        if not any(t in record["equipment_text"] for t in concept_terms):
            return False
    for concept_terms in filt.exclude:
        if any(t in record["equipment_text"] for t in concept_terms):
            return False
    return True


def search(root: Path, filt: SearchFilter) -> list[dict]:
    return [r for r in load_searchable_fleet(root) if match_vehicle(r, filt)]
