"""
bps_helpers.py

Shared utilities for BPS parser scripts (11, 12, 13).
Not a standalone script — imported by the parsers.

Key responsibilities:
  - Read text/CSV/Excel BPS files
  - Detect multi-row header structure
  - Map columns by name (preferred) or position (fallback)
  - Standardize output rows
  - Accumulate and write parse failures
"""

import csv
import re
import sys
from pathlib import Path

# Force UTF-8 output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Standard output columns (all parsers must produce these) ───────────────
STANDARD_COLS = [
    "year",
    "geography_type",
    "geography_id",
    "geography_name",
    "state_fips",
    "county_fips",
    "full_county_fips",
    "cbsa_code",
    "single_family_units",
    "total_units",
    "single_family_share",
    "single_family_value",
    "multifamily_units",
    "multifamily_value",
    "total_value",
    "source_file",
]

# ── Failure log schema ─────────────────────────────────────────────────────
FAILURE_COLS = ["geo_level", "source_file", "year", "reason"]

# ── Regex helpers ──────────────────────────────────────────────────────────
_SURVEY_DATE_RE  = re.compile(r"^\d{4}(\d{2})?$")  # YYYY or YYYYMM
_CATEGORY_KEYWORDS = {"1-unit", "2-unit", "2-units", "3-4", "5+"}
_YEAR_RE        = re.compile(r"(20\d{2}|19\d{2})")


def infer_year_from_path(path: Path) -> str:
    m = _YEAR_RE.search(path.name)
    return m.group(1) if m else ""


# ── File reading ───────────────────────────────────────────────────────────
def read_text_lines(path: Path) -> tuple[list[str], str]:
    """Read all lines from a text file. Returns (lines, encoding)."""
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with open(path, encoding=enc, errors="replace") as f:
                return f.readlines(), enc
        except Exception:
            continue
    return [], "failed"


def read_excel_rows(path: Path) -> tuple[list[list], str]:
    """
    Read all rows from first sheet of an xlsx file.
    Returns (rows_as_lists_of_strings, sheet_name).
    Requires openpyxl.
    """
    try:
        import openpyxl
    except ImportError:
        raise RuntimeError("openpyxl not installed; cannot read .xlsx files")

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = wb.worksheets[0]
    rows = []
    for row in sheet.iter_rows(values_only=True):
        rows.append(["" if v is None else str(v) for v in row])
    wb.close()
    return rows, sheet.title


# ── CSV line parsing ───────────────────────────────────────────────────────
def parse_csv_line(line: str, delimiter: str = ",") -> list[str]:
    """Parse a single CSV line, strip whitespace from each field."""
    try:
        row = next(csv.reader([line], delimiter=delimiter))
        return [c.strip() for c in row]
    except Exception:
        return [c.strip() for c in line.split(delimiter)]


def rows_from_text(lines: list[str], delimiter: str = ",") -> list[list[str]]:
    """Convert raw text lines to list-of-list-of-strings."""
    return [parse_csv_line(ln, delimiter) for ln in lines if ln.strip()]


# ── Header detection ───────────────────────────────────────────────────────
_HEADER_KEYWORDS = {"survey date", "fips", "postal code", "name", "cbsa", "csa"}
_SUBHEADER_KEYWORDS = {"bldgs", "units", "value"}


def find_header_span(rows: list[list[str]]) -> tuple[int, int]:
    """
    Find (header_start, data_start) row indices.

    Strategy: locate the first data row (col 0 is a survey date YYYY or YYYYMM),
    then walk backward to the earliest non-blank row before it.

    This handles two observed BPS header layouts:
      Variant A (CBSA): Row 0=metadata, Row 1=category+identifier combined,
                        Row 2=sub-columns (Bldgs/Units/Value), Row 3=data
      State/County:     Row 0=category labels (1-unit, 2-units...),
                        Row 1=identifier+sub-columns (Date, FIPS, Bldgs...),
                        Row 2=blank, Row 3=data

    Returns (-1, -1) if not found.
    """
    # Step 1: find data_start — first row where col 0 is a survey date
    data_start = -1
    for i, row in enumerate(rows):
        col0 = row[0].strip() if row else ""
        if _SURVEY_DATE_RE.match(col0):
            data_start = i
            break

    if data_start <= 0:
        return (-1, data_start)

    # Step 2: header_start — earliest non-blank row before data_start
    header_start = -1
    for i in range(data_start):
        if any(c.strip() for c in rows[i]):
            header_start = i
            break

    return header_start, data_start


# ── Multi-row header combination ───────────────────────────────────────────
def combine_headers(header_rows: list[list[str]]) -> list[str]:
    """
    Merge 1-3 BPS header rows into single column labels.

    Strategy:
      1. Take the LAST row that contains main column names (has "date"/"fips")
         as the "category" row.
      2. Take the next row as the "sub-column" row (has "bldgs"/"units").
      3. Fill forward empty cells in the category row so each group's label
         covers all its sub-columns.
      4. Combine: "<category> <subcolumn>".strip()

    LAYOUT ASSUMPTION: The category row uses empty cells to span multiple
    sub-columns (i.e., "1-unit" appears once at the start of a 3-column group,
    not repeated for each sub-column). This is the standard BPS CSV layout
    observed in 2004-2025 files.
    """
    if not header_rows:
        return []

    # Find the category row (has "1-unit", "2-units", etc.) and sub-column row.
    # Priority: category labels > identifier labels, so that the fill-forward
    # propagates permit-type names (1-unit...) into the combined headers.
    cat_row  = None  # row with data category labels (1-unit, 2-units, ...)
    sub_row  = None  # row with sub-column labels (Bldgs, Units, Value)

    for row in header_rows:
        joined = " ".join(row).lower()
        if any(kw in joined for kw in _CATEGORY_KEYWORDS):
            cat_row = row   # this row has permit-type labels
        elif any(kw in joined for kw in _SUBHEADER_KEYWORDS):
            sub_row = row   # this row has Bldgs/Units/Value

    # Fallback: if no dedicated category row (e.g. old single-combined-row files),
    # use the identifier row as main.
    main_row = cat_row
    if main_row is None:
        for row in header_rows:
            joined = " ".join(row).lower()
            if any(kw in joined for kw in _HEADER_KEYWORDS) and "date" in joined:
                main_row = row
                break

    if main_row is None:
        return header_rows[-1] if header_rows else []

    # If sub_row not yet found, check remaining rows
    if sub_row is None:
        for row in header_rows:
            if row is main_row:
                continue
            joined = " ".join(row).lower()
            if any(kw in joined for kw in _SUBHEADER_KEYWORDS):
                sub_row = row
                break

    max_cols = max(len(main_row), len(sub_row) if sub_row else 0)

    def pad(row, n):
        return list(row) + [""] * (n - len(row))

    main_padded = pad(main_row, max_cols)
    sub_padded  = pad(sub_row, max_cols) if sub_row else [""] * max_cols

    # Fill forward empty cells in main_padded
    # ASSUMPTION: empty cell means "same category as previous non-empty cell"
    filled_main = []
    last = ""
    for val in main_padded:
        if val.strip():
            last = val.strip()
            filled_main.append(last)
        else:
            filled_main.append(last)

    # Combine
    combined = []
    for cat, sub in zip(filled_main, sub_padded):
        sub = sub.strip()
        cat = cat.strip()
        if sub and sub.lower() not in cat.lower():
            combined.append(f"{cat} {sub}".strip())
        else:
            combined.append(cat)

    return combined


# ── Column index mapping ───────────────────────────────────────────────────
def find_col_index(headers: list[str], *keywords: str) -> int:
    """
    Find the first column index whose label contains ALL the given keywords
    (case-insensitive). Returns -1 if not found.
    """
    kws = [k.lower() for k in keywords]
    for i, h in enumerate(headers):
        hl = h.lower()
        if all(k in hl for k in kws):
            return i
    return -1


def build_col_map(combined_headers: list[str], geo_type: str) -> dict:
    """
    Build a dict mapping logical field names to column indices.
    Uses keyword detection first; falls back to hardcoded positions.

    LAYOUT ASSUMPTION (positional fallback, Variant A - 2004-2025):
      State  files: id cols at 0-3, data from col 4
      County files: id cols at 0-3, data from col 4
      CBSA   files: id cols at 0-3, data from col 4

      Within data columns (offset from col 4):
        +0 1-unit Bldgs
        +1 1-unit Units   <- single_family_units
        +2 1-unit Value
        +3 2-units Bldgs
        +4 2-units Units
        +5 2-units Value
        +6 3-4 units Bldgs
        +7 3-4 units Units
        +8 3-4 units Value
        +9 5+ units Bldgs
        +10 5+ units Units
        +11 5+ units Value
        +12 Total Bldgs
        +13 Total Units   <- total_units
        +14 Total Value
    """
    POSITIONAL = {
        "state":  {"survey_date": 0, "geo_id1": 1, "geo_id2": 2, "name": 3,
                   "sf_units": 5, "total_units": 17},
        "county": {"survey_date": 0, "geo_id1": 1, "geo_id2": 2, "name": 3,
                   "sf_units": 5, "total_units": 17},
        "cbsa":   {"survey_date": 0, "geo_id1": 1, "geo_id2": 2, "name": 3,
                   "sf_units": 5, "total_units": 17},
    }

    pos = POSITIONAL.get(geo_type, POSITIONAL["state"])
    col_map = dict(pos)  # start with positional defaults
    col_map["method"] = "positional"

    if not combined_headers:
        return col_map

    # Try keyword-based detection (overrides positional)
    sf_idx = find_col_index(combined_headers, "1-unit", "units")
    if sf_idx == -1:
        sf_idx = find_col_index(combined_headers, "1 unit", "units")
    if sf_idx == -1:
        sf_idx = find_col_index(combined_headers, "single", "units")

    tot_idx = find_col_index(combined_headers, "total", "units")

    # Survey date
    date_idx = find_col_index(combined_headers, "survey date")
    if date_idx == -1:
        date_idx = find_col_index(combined_headers, "date")

    # Name
    name_idx = find_col_index(combined_headers, "name")

    # Geo id columns
    if geo_type == "state":
        geo1_idx = find_col_index(combined_headers, "fips", "state")
        geo2_idx = find_col_index(combined_headers, "postal")
        if geo2_idx == -1:
            geo2_idx = find_col_index(combined_headers, "abbreviation")
    elif geo_type == "county":
        geo1_idx = find_col_index(combined_headers, "fips", "state")
        geo2_idx = find_col_index(combined_headers, "fips", "county")
    else:  # cbsa
        geo1_idx = find_col_index(combined_headers, "csa")
        geo2_idx = find_col_index(combined_headers, "cbsa")
        if geo2_idx == -1:
            geo2_idx = find_col_index(combined_headers, "metro")

    overrides = {
        k: v for k, v in [
            ("survey_date", date_idx),
            ("geo_id1",     geo1_idx),
            ("geo_id2",     geo2_idx),
            ("name",        name_idx),
            ("sf_units",    sf_idx),
            ("total_units", tot_idx),
        ] if v != -1
    }

    if overrides:
        col_map.update(overrides)
        col_map["method"] = (
            "keyword" if sf_idx != -1 and tot_idx != -1 else "partial-keyword"
        )

    # If sf was found by keyword but total was not, the file has no pre-computed
    # Total column (observed in all state and county files). Compute total_units
    # by summing the four category unit columns: 1-unit, 2-units, 3-4 units, 5+.
    # Each category is 3 columns wide (Bldgs, Units, Value), so unit cols are
    # sf_idx, sf_idx+3, sf_idx+6, sf_idx+9.
    if sf_idx != -1 and tot_idx == -1:
        col_map["total_units"]    = "compute"
        col_map["extra_unit_cols"] = [sf_idx + 3, sf_idx + 6, sf_idx + 9]

    # Valuation and multifamily columns (same offsets for all geo types):
    # layout: ..., 1-unit[Bldgs, Units(sf), Value], 2-units[Bldgs, Units, Value],
    #         3-4 units[Bldgs, Units, Value], 5+ units[Bldgs, Units, Value], ...
    if sf_idx != -1:
        col_map["sf_value_col"]  = sf_idx + 1              # 1-unit Value ($1,000s)
        col_map["mf_unit_cols"]  = [sf_idx + 3, sf_idx + 6, sf_idx + 9]   # 2-, 3-4, 5+ Units
        col_map["mf_value_cols"] = [sf_idx + 4, sf_idx + 7, sf_idx + 10]  # 2-, 3-4, 5+ Value

    return col_map


# ── Safe field extraction ──────────────────────────────────────────────────
def safe_get(row: list[str], idx: int, default: str = "") -> str:
    try:
        return row[idx].strip()
    except (IndexError, AttributeError):
        return default


def safe_int(val: str) -> int | None:
    try:
        return int(float(val.replace(",", "")))
    except (ValueError, AttributeError):
        return None


# ── Row builder ────────────────────────────────────────────────────────────
def build_output_row(
    data_row:     list[str],
    col_map:      dict,
    geo_type:     str,
    year:         str,
    source_file:  str,
) -> dict | None:
    """
    Extract one standardized output record from a parsed data row.
    Returns None if the row appears to be a total/aggregate line to skip,
    or if critical fields are missing.
    """
    geo_id1  = safe_get(data_row, col_map["geo_id1"])
    geo_id2  = safe_get(data_row, col_map["geo_id2"])
    name     = safe_get(data_row, col_map["name"])
    sf_raw   = safe_get(data_row, col_map["sf_units"])

    sf_units = safe_int(sf_raw)

    if col_map.get("total_units") == "compute":
        # No pre-computed Total column: sum all four category unit columns.
        extra = [
            safe_int(safe_get(data_row, i))
            for i in col_map.get("extra_unit_cols", [])
        ]
        if sf_units is not None:
            tot_units = sf_units + sum(v for v in extra if v is not None)
        else:
            tot_units = None
    else:
        tot_raw   = safe_get(data_row, col_map["total_units"])
        tot_units = safe_int(tot_raw)

    # ── Valuation and multifamily ──────────────────────────────────────────
    sf_value = (
        safe_int(safe_get(data_row, col_map["sf_value_col"]))
        if "sf_value_col" in col_map else None
    )

    mf_unit_raw = [
        safe_int(safe_get(data_row, i))
        for i in col_map.get("mf_unit_cols", [])
    ]
    multifamily_units = (
        sum(v for v in mf_unit_raw if v is not None)
        if any(v is not None for v in mf_unit_raw) else None
    )

    mf_val_raw = [
        safe_int(safe_get(data_row, i))
        for i in col_map.get("mf_value_cols", [])
    ]
    multifamily_value = (
        sum(v for v in mf_val_raw if v is not None)
        if any(v is not None for v in mf_val_raw) else None
    )

    total_value = (
        (sf_value or 0) + (multifamily_value or 0)
        if sf_value is not None or multifamily_value is not None else None
    )

    # Skip rows where both unit counts are missing
    if sf_units is None and tot_units is None:
        return None

    # Compute share
    if sf_units is not None and tot_units and tot_units > 0:
        share = round(sf_units / tot_units, 4)
    else:
        share = None

    # Build geo fields
    state_fips = county_fips = full_county_fips = cbsa_code = ""

    if geo_type == "state":
        state_fips  = geo_id1.zfill(2) if geo_id1.isdigit() else geo_id1
        geography_id = state_fips
    elif geo_type == "county":
        state_fips  = geo_id1.zfill(2) if geo_id1.isdigit() else geo_id1
        county_fips = geo_id2.zfill(3) if geo_id2.isdigit() else geo_id2
        full_county_fips = state_fips + county_fips if state_fips and county_fips else ""
        geography_id = full_county_fips
    else:  # cbsa
        # LAYOUT ASSUMPTION: geo_id1 = CSA code (may be blank), geo_id2 = CBSA code
        cbsa_code    = geo_id2.strip()
        geography_id = cbsa_code

    if not geography_id:
        return None

    return {
        "year":                year,
        "geography_type":      geo_type,
        "geography_id":        geography_id,
        "geography_name":      name,
        "state_fips":          state_fips,
        "county_fips":         county_fips,
        "full_county_fips":    full_county_fips,
        "cbsa_code":           cbsa_code,
        "single_family_units": sf_units,
        "total_units":         tot_units,
        "single_family_share": share,
        "single_family_value": sf_value,
        "multifamily_units":   multifamily_units,
        "multifamily_value":   multifamily_value,
        "total_value":         total_value,
        "source_file":         source_file,
    }


# ── Main file parser ───────────────────────────────────────────────────────
def parse_bps_file(
    path: Path,
    geo_type: str,
    year: str,
) -> tuple[list[dict], str | None]:
    """
    Parse one BPS annual file (text or xlsx).

    Returns (records, error_message).
    If error_message is not None, parsing failed entirely.
    Records may still be empty even on success (e.g. no data rows found).
    """
    ext = path.suffix.lower()

    # ── Read rows ──────────────────────────────────────────────────────────
    if ext in {".xls"}:
        return [], "xlrd not installed; cannot read .xls files — install xlrd or convert to .xlsx"

    if ext in {".xlsx", ".xlsm"}:
        try:
            raw_rows, _ = read_excel_rows(path)
        except Exception as e:
            return [], f"Excel read error: {e}"
    elif ext in {".txt", ".csv", ".dat", ".asc", ".prn", ".tsv"}:
        lines, enc = read_text_lines(path)
        if not lines:
            return [], "file is empty or unreadable"
        delimiter = "\t" if ext == ".tsv" else ","
        raw_rows = rows_from_text(lines, delimiter)
    else:
        return [], f"unrecognized extension: {ext}"

    if not raw_rows:
        return [], "no rows after reading"

    # ── Locate header and data ─────────────────────────────────────────────
    header_start, data_start = find_header_span(raw_rows)

    if header_start == -1:
        return [], "cannot find header row (no 'survey date' keyword)"
    if data_start == -1:
        return [], "cannot find data rows (no YYYYMM pattern in col 0)"

    # ── Build combined column headers ──────────────────────────────────────
    header_rows    = raw_rows[header_start:data_start]
    combined_hdrs  = combine_headers(header_rows)
    col_map        = build_col_map(combined_hdrs, geo_type)

    # ── Parse data rows ────────────────────────────────────────────────────
    source_name = path.name
    records = []
    for row in raw_rows[data_start:]:
        col0 = row[0].strip() if row else ""
        if not _SURVEY_DATE_RE.match(col0):
            continue  # skip subtotals / blank / footer lines
        rec = build_output_row(row, col_map, geo_type, year, source_name)
        if rec:
            records.append(rec)

    return records, None


# ── Failure logging ────────────────────────────────────────────────────────
def save_failures(failures: list[dict], out_path: Path) -> None:
    """
    Write/merge failures to the shared bps_parse_failures.csv.
    Existing rows for other geo_levels are preserved.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if out_path.exists():
        try:
            with open(out_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing.append(row)
        except Exception:
            pass

    # Replace rows for the current geo levels being written
    if failures:
        current_geos = {r["geo_level"] for r in failures}
        existing = [r for r in existing if r.get("geo_level") not in current_geos]

    combined = existing + failures
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FAILURE_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(combined)


# ── Output writer ──────────────────────────────────────────────────────────
def write_output(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=STANDARD_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
