"""
Parses a Ford Monroney window-sticker PDF into structured per-vehicle data:
model/VIN/colors/drivetrain, the 4-column standard-equipment grid (exterior/
interior/functional/safety), optional equipment, pricing, and factory
warranties.

Only works when a real sticker exists -- used vehicles and pre-publish new
vehicles commonly have none (dtfb.py already detects and skips the
"please check back later" placeholder PDF before this module ever runs).
Callers must have a fallback to the site's own JSON feature blob for that
case; see dtfb.py's window-sticker download path.

Extraction is via `pdftotext -bbox` (word-level coordinates), not `-layout`
text or xml.etree: the coordinates are what let us tell the 4 equipment
columns apart, and a real sticker's barcode/QR noise text can contain a
literal unescaped '<' that breaks XML well-formedness (confirmed on a
Bronco Sport sticker, bbox line 623) -- a plain regex over the bbox output
sidesteps that entirely.

The section layout (column x-positions, header labels, "WARRANTY" sharing
the 4th equipment column's space, SOLD TO block) is Ford's fixed Monroney
template, not vehicle-specific data -- safe to anchor on.
"""
from __future__ import annotations

import html
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

_WORD_RE = re.compile(
    r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>',
    re.DOTALL,
)
_VIN_RE = re.compile(r'\b([A-HJ-NPR-Z0-9]{17})\b')
_CONTINUATION_ENDINGS = (",", ":", "-", "/", "&")


@dataclass
class Word:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str


def _is_garbage(text: str) -> bool:
    """Sticker barcodes/QR blocks come through pdftotext as "text" in a
    symbol font -- either Private Use Area glyphs or raw control bytes
    (one observed word was a bare 0x9F bullet-point glyph). Filtering by
    codepoint range here means every downstream consumer gets clean text."""
    if not text.strip():
        return True
    for ch in text:
        cp = ord(ch)
        if 0xE000 <= cp <= 0xF8FF:  # Private Use Area (barcode/symbol fonts)
            return True
        if cp < 0x20 or 0x7F <= cp <= 0x9F:  # C0/C1 control characters
            return True
    return False


def extract_words(pdf_path: Path) -> list[Word]:
    result = subprocess.run(
        ["pdftotext", "-bbox", str(pdf_path), "-"],
        capture_output=True, text=True, timeout=20, check=True,
    )
    words = []
    for m in _WORD_RE.finditer(result.stdout):
        x0, y0, x1, y1, text = m.groups()
        text = html.unescape(text)
        if _is_garbage(text):
            continue
        words.append(Word(float(x0), float(y0), float(x1), float(y1), text))
    words.sort(key=lambda w: (w.y0, w.x0))
    return words


def group_rows(words: list[Word], y_tol: float = 1.5) -> list[list[Word]]:
    """Cluster words (already sorted by (y0, x0)) into visual rows by y0
    proximity. A fixed tolerance is fine -- this is one document template
    with a known, fixed font size."""
    rows: list[list[Word]] = []
    for w in words:
        if rows and abs(w.y0 - rows[-1][0].y0) <= y_tol:
            rows[-1].append(w)
        else:
            rows.append([w])
    for row in rows:
        row.sort(key=lambda w: w.x0)
    return rows


def _row_text(row: list[Word]) -> str:
    return " ".join(w.text for w in row)


def _titlecase(text: str) -> str:
    """Best-effort readability pass over the sticker's abbreviated all-caps
    text (e.g. "ACTIVE GRILLE SHUTTERS" -> "Active Grille Shutters"). Not an
    attempt at official Ford marketing copy -- acronyms like "AM/FM" or
    "LED" will title-case imperfectly; the content is what matters here."""
    return text.title()


def _merge_continuations(lines: list[str]) -> list[str]:
    """A sticker equipment item occasionally wraps onto a second row with no
    larger line-gap to signal it (confirmed: "STEERING:TILT/TELESCOPE," +
    "CRUISE & AUDIO CONTROLS" on the Bronco Sport sticker, and a Maverick's
    "8YR/100,000 HYBRID UNIQUE" + "-COMPONENTS IF EQUIPPED", both pairs
    9.2pt rows like every other, single-row, standalone item). Punctuation
    is the only signal available: either the first line trails off
    incomplete, or the second line opens with a continuation mark instead
    of starting a real new item."""
    merged: list[str] = []
    for line in lines:
        starts_continuation = line.startswith(("-", "/", "&", ":", ","))
        if merged and (merged[-1].rstrip().endswith(_CONTINUATION_ENDINGS) or starts_continuation):
            merged[-1] = f"{merged[-1]} {line}"
        else:
            merged.append(line)
    return merged


def _find_row(rows: list[list[Word]], predicate) -> list[Word] | None:
    for row in rows:
        if predicate(row):
            return row
    return None


# --------------------------------------------------------------------------
# Vehicle overview (model/VIN/colors/engine/transmission)
# --------------------------------------------------------------------------

def _parse_overview(words: list[Word], rows: list[list[Word]]) -> dict:
    """Rows are grouped globally by y-proximity, but several logically
    distinct blocks (model/trim info, exterior/interior color info, the fuel
    economy sidebar) sit at nearly the same y as each other and would bleed
    together in a single row if we grouped by y alone. Each block below
    re-groups from an x-restricted word subset instead, so a shared y never
    merges unrelated columns."""
    overview: dict = {}

    full_text = " ".join(w.text for w in words)
    vin_match = _VIN_RE.search(full_text)
    if vin_match:
        overview["vin"] = vin_match.group(1)

    header = _find_row(rows, lambda r: {"VEHICLE", "DESCRIPTION"} <= {w.text.upper() for w in r})
    grid_header = _find_row(rows, lambda r: {"STANDARD", "EQUIPMENT"} <= {w.text.upper() for w in r})
    if header is None:
        return overview
    header_y = header[0].y0
    grid_y = grid_header[0].y0 if grid_header else header_y + 90

    # Model name + trim/drivetrain/passenger/engine/transmission all sit in
    # the left-hand block (x < 420, before the EXTERIOR/INTERIOR color
    # labels start around x=432) between the header and the equipment grid.
    left_words = [w for w in words if header_y < w.y0 < grid_y and w.x0 < 420]
    block_rows = group_rows(left_words)
    if block_rows:
        overview["model_line"] = _row_text(block_rows[0]).title()

    for row in block_rows[1:]:
        text = _row_text(row)
        upper = text.upper()
        if re.match(r"^\d{4}\b", text):
            overview["trim_drivetrain"] = text.title()
        elif "PASSENGER" in upper:
            overview["seating_capacity"] = text.title()
        elif "ENGINE" in upper or re.search(r"\bECOBOOST\b", upper):
            overview["engine"] = text.title()
        elif "TRANSMISSION" in upper or re.search(r"\bAUTO\b|\bMANUAL\b", upper):
            overview["transmission"] = text.title()

    color_words = [w for w in words if header_y < w.y0 < grid_y and 420 < w.x0 < 620]
    color_rows = group_rows(color_words)
    for label, key in (("EXTERIOR", "exterior_color"), ("INTERIOR", "interior_trim")):
        label_row = _find_row(color_rows, lambda r, label=label: len(r) == 1 and r[0].text.upper() == label)
        if label_row is None:
            continue
        label_y = label_row[0].y0
        value_row = _find_row(color_rows, lambda r, label_y=label_y: label_y < r[0].y0 < label_y + 15)
        if value_row is not None:
            overview[key] = _row_text(value_row).title()

    if "model_line" in overview and "trim_drivetrain" in overview:
        overview["model"] = f"{overview['trim_drivetrain'].split()[0]} " \
                             f"{overview['model_line']} {' '.join(overview['trim_drivetrain'].split()[1:])}"

    return overview


# --------------------------------------------------------------------------
# Standard equipment grid (exterior / interior / functional / safety)
# --------------------------------------------------------------------------

_EQUIPMENT_COLUMNS = ("EXTERIOR", "INTERIOR", "FUNCTIONAL", "SAFETY/SECURITY")
_COLUMN_KEYS = {
    "EXTERIOR": "exterior",
    "INTERIOR": "interior",
    "FUNCTIONAL": "functional_tech",
    "SAFETY/SECURITY": "safety_security",
}


def _parse_equipment_grid(rows: list[list[Word]]) -> tuple[dict, list[str]]:
    header = _find_row(
        rows,
        lambda r: {"EXTERIOR", "INTERIOR", "FUNCTIONAL"} <= {w.text.upper() for w in r}
        and any("SAFETY" in w.text.upper() for w in r),
    )
    if header is None:
        return {}, []

    col_starts = sorted((w.x0, w.text.upper()) for w in header if w.text.upper() in _EQUIPMENT_COLUMNS)
    # The last column has no next header to bound it against -- reuse the
    # previous column's width rather than an arbitrary large number, so it
    # doesn't swallow unrelated content (e.g. the fuel-economy sidebar)
    # sitting to the right of the grid.
    last_gap = col_starts[-1][0] - col_starts[-2][0] if len(col_starts) > 1 else 160.0
    bounds = []
    for i, (x0, label) in enumerate(col_starts):
        x1 = col_starts[i + 1][0] if i + 1 < len(col_starts) else x0 + last_gap
        bounds.append((label, x0 - 2, x1 - 2))

    end_row = _find_row(rows, lambda r: {"INCLUDED", "ON", "THIS", "VEHICLE"} <= {w.text.upper() for w in r})
    y_end = end_row[0].y0 if end_row else 1e9
    header_y = header[0].y0

    grid: dict = {}
    warranty_lines: list[str] = []
    for label, x0, x1 in bounds:
        col_rows = [r for r in rows if header_y < r[0].y0 < y_end
                    and any(x0 <= w.x0 < x1 for w in r)]

        warranty_row = _find_row(col_rows, lambda r: any(w.text.upper() == "WARRANTY" for w in r))
        if warranty_row is not None:
            wy = warranty_row[0].y0
            raw = [_row_text([w for w in r if x0 <= w.x0 < x1]) for r in col_rows if r[0].y0 > wy]
            warranty_lines = _merge_continuations([l for l in raw if l.strip()])
            col_rows = [r for r in col_rows if r[0].y0 < wy]

        lines = [_row_text([w for w in r if x0 <= w.x0 < x1]) for r in col_rows]
        lines = [l for l in lines if l.strip()]
        lines = _merge_continuations(lines)
        grid[_COLUMN_KEYS[label]] = [_titlecase(l) for l in lines]

    return grid, warranty_lines


_WARRANTY_RE = re.compile(r"(\d+)YR/([\d,]+)\s+(.*)")


def _format_warranty(raw: str) -> str:
    m = _WARRANTY_RE.match(raw)
    if not m:
        return raw.title()
    years, miles, label = m.groups()
    label = label.replace("BUMPER / BUMPER", "Bumper-to-Bumper").strip()
    if label and not label[0].isupper() or label.isupper():
        label = label.title()
    label = re.sub(r" -(\S)", r" - \1", label)  # "Unique -Components" -> "Unique - Components"
    # (only a hyphen with a preceding space -- compound words like
    # "Bumper-to-Bumper" have no space before their hyphens and are untouched)
    return f"{years}-Year / {miles}-Mile {label}"


# --------------------------------------------------------------------------
# Optional equipment / pricing / dealer
# --------------------------------------------------------------------------

def _find_label(row: list[Word], label_words: list[str]) -> int | None:
    """Index just past `label_words` in `row` (matched as a case-
    insensitive subsequence anywhere in the row), or None if absent."""
    n = len(label_words)
    texts = [w.text.upper() for w in row]
    for i in range(len(texts) - n + 1):
        if texts[i:i + n] == label_words:
            return i + n
    return None


def _parse_pricing(rows: list[list[Word]]) -> dict:
    """INCLUDED ON THIS VEHICLE (left: equipment group / optional-equipment
    lines) and PRICE INFORMATION (right: base price, destination &
    delivery) are two side-by-side columns that commonly land on the SAME
    visual row as each other -- confirmed real case: "RAPTOR SERIES" (an
    optional-equipment line, left column) shares a row with "BASE PRICE
    $76,775.00" (right column) on a real F-150 Raptor sticker. A label
    match anchored at the row's first N words -- what this used to do --
    misses every price whose row also carries left-column text ahead of
    it. Matching the label as a subsequence anywhere in the row, then
    taking the first money token that follows the match (not the last
    money token in the whole row, which could belong to a different label
    further right in the same merged row), finds it regardless of what
    else is sharing that y."""
    pricing: dict = {}
    money_re = re.compile(r"\$?[\d,]+\.\d{2}")

    def money_after(label_words: tuple[str, ...]) -> str | None:
        label_upper = list(label_words)
        for row in rows:
            start = _find_label(row, label_upper)
            if start is None:
                continue
            for w in row[start:]:
                if money_re.fullmatch(w.text.strip("$")):
                    return w.text if w.text.startswith("$") else f"${w.text}"
        return None

    base = money_after(("BASE", "PRICE"))
    if base:
        pricing["base_price"] = base
    dest = money_after(("DESTINATION", "&", "DELIVERY"))
    if dest:
        pricing["destination_and_delivery"] = dest
    total = money_after(("TOTAL", "VEHICLE", "&", "OPTIONS/OTHER"))
    if total:
        pricing["total_vehicle_and_options"] = total
    msrp = money_after(("TOTAL", "MSRP"))
    if msrp:
        pricing["total_msrp"] = msrp

    return pricing


def _parse_optional_equipment(words: list[Word], rows: list[list[Word]]) -> list[str]:
    header = _find_row(rows, lambda r: "OPTIONAL" in [w.text.upper() for w in r]
                        and any("EQUIPMENT" in w.text.upper() for w in r))
    # PRICE INFORMATION is a sibling section to the right at nearly the same
    # y, not something below OPTIONAL EQUIPMENT -- not a usable end marker.
    # SOLD TO is the next real section header below, in the same left band.
    end_row = _find_row(rows, lambda r: r[0].text.upper() == "SOLD" and len(r) > 1 and r[1].text.upper() == "TO")
    if header is None:
        return []
    header_y = header[0].y0
    y_end = end_row[0].y0 if end_row else header_y + 70
    # Optional-equipment lines run right up to (but not into) the PRICE
    # INFORMATION column beside them -- use its left edge as the right
    # bound rather than a guessed pixel width, since a "NO CHARGE" suffix
    # on a line can sit well past where the label text itself ends.
    price_header = _find_row(rows, lambda r: r[0].text.upper() == "PRICE"
                              and len(r) > 1 and r[1].text.upper() == "INFORMATION")
    x_end = price_header[0].x0 - 5 if price_header else 450
    # Filter the WORDS by that band before grouping into rows -- a row that
    # merely *starts* in-band still pulls in same-y sidebar text from far to
    # the right (the same bleed-through bug the equipment grid and overview
    # block both hit).
    left_words = [w for w in words if header_y < w.y0 < y_end and w.x0 < x_end]
    items = []
    for r in group_rows(left_words):
        text = _row_text(r).strip().lstrip(".")
        if text and "NO CHARGE" not in text.upper():
            items.append(_titlecase(text))
    return items


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def find_panel_split_x(pdf_path: Path, center_frac: float = 0.5, search_frac: float = 0.15,
                        min_gap: float = 8.0) -> float | None:
    """Some Ford Monroney stickers print as two side-by-side panels on one
    landscape page (equipment/pricing on the left, fuel economy/safety
    ratings on the right -- confirmed on the Bronco Sport sticker, a real
    ~11pt text-free gutter around x=678 on a 1224pt-wide page) rather than
    one continuous flow. As a single wide image that's a crammed, hard-to-
    read-on-mobile shot that also survives upload recompression worse than
    two normal-proportioned panels would.

    Detects this generically -- no fixed split coordinate -- by finding the
    widest text-free vertical gap within `search_frac` of the page's
    horizontal center, verified against every word's actual (x0, x1) span
    so nothing gets sliced through mid-word. Returns the gap's midpoint in
    PDF points, or None if there's no clear gutter (a genuinely single-flow
    layout) -- callers should leave the page unsplit in that case, not
    force a cut through real content."""
    words = extract_words(pdf_path)
    if not words:
        return None
    page_width = max(w.x1 for w in words)
    center = page_width * center_frac
    band = page_width * search_frac
    lo, hi = center - band, center + band

    # The page's very top margin carries barcode/serial metadata that can
    # render as plain-looking (non-garbage-filtered) concatenated text
    # spanning a wide x-range -- confirmed on a Maverick sticker
    # ("RAMPBUMPCAMPBOOKEXFLROTABATT" at y0=12, well above any real
    # content) blocking gutter detection entirely. Anchor the search to
    # start at the "VEHICLE DESCRIPTION" header row, the same real-content
    # start line _parse_overview() uses, rather than a guessed y cutoff.
    rows = group_rows(words)
    header = _find_row(rows, lambda r: {"VEHICLE", "DESCRIPTION"} <= {w.text.upper() for w in r})
    min_y = header[0].y0 if header else 0.0
    words = [w for w in words if w.y0 >= min_y]
    if not words:
        return None

    intervals = sorted((w.x0, w.x1) for w in words if w.x1 > lo and w.x0 < hi)
    if not intervals:
        return None
    merged = [list(intervals[0])]
    for x0, x1 in intervals[1:]:
        if x0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], x1)
        else:
            merged.append([x0, x1])

    gaps = []
    prev_end = lo
    for x0, x1 in merged:
        gaps.append((x0 - prev_end, prev_end, x0))
        prev_end = x1
    gaps.append((hi - prev_end, prev_end, hi))
    gaps.sort(reverse=True)

    best_gap, gap_start, gap_end = gaps[0]
    return (gap_start + gap_end) / 2 if best_gap >= min_gap else None


def parse_sticker(pdf_path: Path) -> dict:
    """Parse a Ford Monroney window-sticker PDF into structured JSON. Raises
    if pdftotext fails or the PDF has no readable text layer -- callers
    should already have filtered out the "check back later" placeholder
    case before calling this (see dtfb.py::is_placeholder_sticker)."""
    words = extract_words(pdf_path)
    rows = group_rows(words)

    overview = _parse_overview(words, rows)
    equipment, warranty_raw = _parse_equipment_grid(rows)
    pricing = _parse_pricing(rows)
    optional_equipment = _parse_optional_equipment(words, rows)

    return {
        "overview": overview,
        "equipment": equipment,
        "optional_equipment": optional_equipment,
        "warranties": [_format_warranty(w) for w in warranty_raw if w.strip()],
        "pricing": pricing,
    }
