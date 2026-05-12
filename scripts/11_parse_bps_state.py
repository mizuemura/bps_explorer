"""
11_parse_bps_state.py

Parse Census BPS annual state-level files (2000-2025) into a standardized table.
Run from project root:  C:\\Users\\mizue\\miniconda3\\python.exe scripts/11_parse_bps_state.py

Input:   data/raw/bps/state/st{year}a.txt
Output:  data/processed/bps/bps_state_annual_rawparsed.csv
         outputs/reports/bps_parse_failures.csv

Expected Census file layout (Variant A, observed 2004-2025):
  Row 0: metadata (year, "Total", blank columns) -- skipped
  Row 1: main column headers (Survey Date, FIPS State Numeric Code, Postal Code, Name,
                               1-unit, [blank x2], 2-units, [blank x2], ...)
  Row 2: sub-column headers  (Bldgs, Units, Value ($1,000), repeated per category)
  Row 3+: data rows with col 0 = YYYYMM survey date

If a file cannot be parsed using that layout, it is logged to the failures report
and skipped -- not silently dropped.
"""

import sys
from pathlib import Path

# Allow importing bps_helpers from the same scripts/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bps_helpers as bh

# Force UTF-8 output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR    = PROJECT_ROOT / "data" / "raw" / "bps" / "state"
OUTPUT_CSV   = PROJECT_ROOT / "data" / "processed" / "bps" / "bps_state_annual_rawparsed.csv"
FAILURES_CSV = PROJECT_ROOT / "outputs" / "reports" / "bps_parse_failures.csv"

GEO_TYPE = "state"

# ── Validation thresholds ──────────────────────────────────────────────────
# Annual national single-family permits have ranged from ~300k (2009) to ~1.4M
# (2005). Per-state max is well below 500k. Flag outliers for review.
MAX_PLAUSIBLE_SF_UNITS = 500_000


def validate_records(records: list[dict], source: str) -> list[str]:
    warnings = []
    for r in records:
        sf = r.get("single_family_units")
        tot = r.get("total_units")
        if sf is not None and sf < 0:
            warnings.append(f"{source}: negative single_family_units for {r['geography_id']}")
        if sf is not None and sf > MAX_PLAUSIBLE_SF_UNITS:
            warnings.append(f"{source}: implausibly large sf_units ({sf}) for {r['geography_id']}")
        if sf is not None and tot is not None and sf > tot:
            warnings.append(f"{source}: sf_units ({sf}) > total_units ({tot}) for {r['geography_id']}")
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
        rel  = fpath.relative_to(PROJECT_ROOT)
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

        print(f"OK  ({len(records)} records, method={records[0].get('source_file','?')})")
        # Re-print col_map method if accessible -- just show count cleanly
        print(f"         -> {len(records)} state rows")
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
