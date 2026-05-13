"""
12_parse_bps_county.py

Parse Census BPS annual county-level files (2000-2025) into a standardized table.
Run from project root:  C:\\Users\\mizue\\miniconda3\\python.exe scripts/12_parse_bps_county.py

Input:   data/raw/bps/county/co{year}a.txt
Output:  data/processed/bps/bps_county_annual_rawparsed.csv
         outputs/reports/bps_parse_failures.csv

Expected Census file layout (Variant A, observed 2004-2025):
  Row 0: metadata row -- skipped
  Row 1: main headers (Survey Date, FIPS State Numeric Code, FIPS County Numeric Code,
                        Name, 1-unit, [blank x2], 2-units, ...)
  Row 2: sub-headers  (Bldgs, Units, Value ($1,000), repeated)
  Row 3+: data rows, col 0 = YYYYMM

  LAYOUT ASSUMPTION: county_fips in col 2 is the 3-digit county code only
  (not the full 5-digit FIPS). The full 5-digit code is built as
  state_fips.zfill(2) + county_fips.zfill(3).

  NOTE: Some county files contain rows where county_fips = "000", representing
  the state total. These ARE included in the output (geography_type stays "county")
  so downstream code can filter them if needed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bps_helpers as bh

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR    = PROJECT_ROOT / "data" / "raw" / "bps" / "county"
OUTPUT_CSV   = PROJECT_ROOT / "data" / "processed" / "bps" / "bps_county_annual_rawparsed.csv"
FAILURES_CSV = PROJECT_ROOT / "outputs" / "reports" / "bps_parse_failures.csv"

GEO_TYPE = "county"

# ── Validation ─────────────────────────────────────────────────────────────
# County-level max: a very large county (e.g. Maricopa, AZ) can issue ~50k
# single-family permits in a peak year. Flag anything above 200k as suspicious.
MAX_PLAUSIBLE_SF_UNITS = 200_000


def validate_records(records: list[dict], source: str) -> list[str]:
    warnings = []
    for r in records:
        sf  = r.get("single_family_units")
        tot = r.get("total_units")
        fips = r.get("full_county_fips", "")
        if sf is not None and sf < 0:
            warnings.append(f"{source}: negative sf_units for {fips}")
        if sf is not None and sf > MAX_PLAUSIBLE_SF_UNITS:
            warnings.append(f"{source}: implausibly large sf_units ({sf}) for {fips}")
        if sf is not None and tot is not None and sf > tot:
            warnings.append(f"{source}: sf_units ({sf}) > total_units ({tot}) for {fips}")
        if fips and len(fips) != 5:
            warnings.append(f"{source}: unexpected full_county_fips length: '{fips}'")
    return warnings


def main():
    if not INPUT_DIR.exists() or not any(INPUT_DIR.iterdir()):
        print(f"[!] No files found in {INPUT_DIR}")
        print("    Run scripts/00_download_bps.py first.")
        return

    all_records = []
    failures    = []

    files = sorted(INPUT_DIR.rglob("*"))
    files = [f for f in files if f.is_file()]
    print(f"Found {len(files)} file(s) in {INPUT_DIR}\n")

    for fpath in files:
        year = bh.infer_year_from_path(fpath)
        print(f"  Parsing: {fpath.name}  (year={year or '?'})", end=" ... ")

        records, err = bh.parse_bps_file(fpath, GEO_TYPE, year)

        if err:
            print(f"FAILED: {err}")
            failures.append({
                "geo_level":   GEO_TYPE,
                "source_file": fpath.name,
                "year":        year,
                "reason":      err,
            })
            continue

        if not records:
            msg = "parsed OK but 0 records extracted"
            print(f"WARN: {msg}")
            failures.append({
                "geo_level":   GEO_TYPE,
                "source_file": fpath.name,
                "year":        year,
                "reason":      msg,
            })
            continue

        warnings = validate_records(records, fpath.name)
        for w in warnings:
            print(f"\n    [WARN] {w}", end="")

        print(f"OK  ({len(records)} county rows)")
        all_records.extend(records)

    # ── Write outputs ──────────────────────────────────────────────────────
    bh.write_output(all_records, OUTPUT_CSV)
    bh.save_failures(failures, FAILURES_CSV)

    print(f"\nTotal records : {len(all_records)}")
    print(f"Parse failures: {len(failures)}")
    print(f"Output        : {OUTPUT_CSV}")
    print(f"Failures log  : {FAILURES_CSV}")

    if failures:
        print("\nFailed files:")
        for f in failures:
            print(f"  {f['source_file']} ({f['year']}): {f['reason']}")


if __name__ == "__main__":
    main()
