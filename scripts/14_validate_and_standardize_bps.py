"""
14_validate_and_standardize_bps.py

Validate, clean, and standardize raw-parsed BPS files. Adds derived time-series
fields. Produces final analysis-ready CSVs and three validation reports.

Run from project root:
    C:\\Users\\mizue\\miniconda3\\python.exe scripts/14_validate_and_standardize_bps.py

Inputs:
    data/processed/bps/bps_state_annual_rawparsed.csv
    data/processed/bps/bps_county_annual_rawparsed.csv
    data/processed/bps/bps_cbsa_annual_rawparsed.csv

Outputs:
    data/processed/bps/bps_state_annual_2000_2025.csv
    data/processed/bps/bps_county_annual_2000_2025.csv
    data/processed/bps/bps_cbsa_annual_2004_2024.csv  (CBSA only; pre-2004 uses MSA codes)
    outputs/reports/bps_validation_summary.csv
    outputs/reports/bps_missing_years.csv
    outputs/reports/bps_duplicate_records.csv
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROC_DIR     = PROJECT_ROOT / "data" / "processed" / "bps"
REPORT_DIR   = PROJECT_ROOT / "outputs" / "reports"

INPUTS = {
    "state":  PROC_DIR / "bps_state_annual_rawparsed.csv",
    "county": PROC_DIR / "bps_county_annual_rawparsed.csv",
    "cbsa":   PROC_DIR / "bps_cbsa_annual_rawparsed.csv",
}
OUTPUTS = {
    "state":  PROC_DIR / "bps_state_annual_2000_2025.csv",
    "county": PROC_DIR / "bps_county_annual_2000_2025.csv",
    "cbsa":   PROC_DIR / "bps_cbsa_annual_2004_2024.csv",
}

# Primary key column for each geography level
GEO_ID_COL = {
    "state":  "state_fips",
    "county": "full_county_fips",
    "cbsa":   "cbsa_code",
}

PERMIT_COLS = [
    "single_family_units", "total_units",
    "single_family_value", "multifamily_units", "multifamily_value", "total_value",
]

# CBSA uses 2004-2024 only (pre-2004 Metro files carry MSA codes, not CBSA codes)
GEO_YEAR_RANGE = {
    "state":  range(2000, 2026),
    "county": range(2000, 2026),
    "cbsa":   range(2004, 2025),
}

# Columns to include in final output (in order)
OUTPUT_COLS = [
    "year", "geography_type", "geography_id", "geography_name",
    "state_fips", "county_fips", "full_county_fips", "cbsa_code",
    "single_family_units", "total_units", "single_family_share",
    "single_family_value", "multifamily_units", "multifamily_value", "total_value",
    "yoy_change_units", "yoy_percent_change", "rolling_3yr_avg",
    "cumulative_sfh_2000_2025", "cumulative_sfh_2010_2025",
    "cumulative_sfh_2020_2025", "source_file",
]


# ── Step 1-2: Numeric cleaning ─────────────────────────────────────────────
def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip commas and whitespace from permit count fields; cast to float
    (float rather than int to accommodate NaN without nullable Int64).
    """
    for col in PERMIT_COLS:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
                .replace({"nan": np.nan, "": np.nan, "None": np.nan})
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── Step 3: Geography name standardization ─────────────────────────────────
def clean_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace. Do not alter casing — Census names
    use mixed case intentionally (e.g. 'McAllen', 'DeKalb')."""
    if "geography_name" in df.columns:
        df["geography_name"] = df["geography_name"].astype(str).str.strip()
        df.loc[df["geography_name"].isin(["nan", "None", ""]), "geography_name"] = np.nan
    return df


# ── Step 4: Year as integer ────────────────────────────────────────────────
def clean_year(df: pd.DataFrame) -> pd.DataFrame:
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    # Drop rows where year is completely unparseable
    n_bad = df["year"].isna().sum()
    if n_bad:
        print(f"    [WARN] Dropping {n_bad} rows with unparseable year")
        df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    return df


# ── Step 5: FIPS zero-padding ──────────────────────────────────────────────
def zpad_col(series: pd.Series, width: int) -> pd.Series:
    """Zero-pad a string column to `width` digits. Preserves blanks and NaN."""
    def _pad(val):
        if pd.isna(val) or str(val).strip() == "":
            return val
        s = str(val).strip()
        if s.isdigit():
            return s.zfill(width)
        return s  # non-numeric codes (e.g. CBSAs with alpha chars): leave as-is
    return series.apply(_pad)


def clean_fips(df: pd.DataFrame, geo_type: str) -> pd.DataFrame:
    if "state_fips" in df.columns:
        df["state_fips"] = zpad_col(df["state_fips"], 2)
    if "county_fips" in df.columns:
        df["county_fips"] = zpad_col(df["county_fips"], 3)

    # Recompute full_county_fips from the padded components
    if geo_type == "county" and "state_fips" in df.columns and "county_fips" in df.columns:
        df["full_county_fips"] = df.apply(
            lambda r: (
                str(r["state_fips"]) + str(r["county_fips"])
                if pd.notna(r["state_fips"]) and pd.notna(r["county_fips"])
                   and str(r["state_fips"]).strip() and str(r["county_fips"]).strip()
                else np.nan
            ),
            axis=1,
        )
    elif "full_county_fips" in df.columns:
        df["full_county_fips"] = zpad_col(df["full_county_fips"], 5)

    if "cbsa_code" in df.columns:
        df["cbsa_code"] = zpad_col(df["cbsa_code"], 5)

    return df


# ── Step 6: Duplicate detection ────────────────────────────────────────────
def check_duplicates(
    df: pd.DataFrame, geo_id_col: str, geo_type: str
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Find and remove duplicate (geography_id, year) records.
    Keeps the first occurrence. Returns (deduped_df, list_of_dupe_dicts).
    """
    dupe_mask = df.duplicated(subset=[geo_id_col, "year"], keep=False)
    dupes = df[dupe_mask].copy()
    dupes["geo_level"] = geo_type
    dupe_records = dupes.to_dict("records")

    # Keep first occurrence
    df_clean = df.drop_duplicates(subset=[geo_id_col, "year"], keep="first")
    return df_clean, dupe_records


# ── Step 7: Missing year detection ─────────────────────────────────────────
def check_missing_years(
    df: pd.DataFrame, geo_id_col: str, geo_type: str, year_range: range
) -> list[dict]:
    """
    For each geography, find years in the expected range with no record.
    Returns list of dicts for the missing years report.
    """
    expected = set(year_range)
    missing = []
    for geo_id, grp in df.groupby(geo_id_col, sort=False):
        present = set(grp["year"].dropna().astype(int))
        absent  = sorted(expected - present)
        name    = grp["geography_name"].dropna().iloc[0] if not grp["geography_name"].dropna().empty else ""
        for yr in absent:
            missing.append({
                "geo_level":      geo_type,
                "geography_id":   geo_id,
                "geography_name": name,
                "missing_year":   yr,
            })
    return missing


# ── Steps 8-9: Data quality checks ────────────────────────────────────────
def data_quality_stats(df: pd.DataFrame, geo_id_col: str) -> dict:
    """
    Return a dict of quality counts. Does not modify df.
    """
    sf = df["single_family_units"] if "single_family_units" in df.columns else pd.Series(dtype=float)
    tot = df["total_units"] if "total_units" in df.columns else pd.Series(dtype=float)

    return {
        "null_geography_id":  int(df[geo_id_col].isna().sum() + (df[geo_id_col] == "").sum()),
        "null_sf_units":      int(sf.isna().sum()),
        "negative_sf_units":  int((sf < 0).sum()),
        "sf_gt_total_units":  int((sf > tot).sum()),
        "null_total_units":   int(tot.isna().sum()),
    }


# ── Step 10a: YoY change ──────────────────────────────────────────────────
def add_yoy(df: pd.DataFrame, geo_id_col: str) -> pd.DataFrame:
    """
    Add yoy_change_units and yoy_percent_change. Only computed for rows
    where the PREVIOUS year's record for the same geography is consecutive
    (year - prev_year == 1). Non-consecutive gaps produce NaN.
    """
    df = df.sort_values([geo_id_col, "year"])

    grp        = df.groupby(geo_id_col)
    prev_year  = grp["year"].shift(1)
    prev_sf    = grp["single_family_units"].shift(1)
    consecutive = (df["year"] - prev_year) == 1

    df["yoy_change_units"]   = np.where(consecutive, df["single_family_units"] - prev_sf, np.nan)
    df["yoy_percent_change"] = np.where(
        consecutive & prev_sf.notna() & (prev_sf != 0),
        ((df["single_family_units"] - prev_sf) / prev_sf * 100).round(2),
        np.nan,
    )
    return df


# ── Step 10b: Rolling 3-year average ──────────────────────────────────────
def add_rolling_3yr(df: pd.DataFrame, geo_id_col: str, year_range: range) -> pd.DataFrame:
    """
    3-year rolling average of single_family_units per geography.

    METHOD: For each geography, reindex to the expected year range (filling
    gaps with NaN), apply rolling(3, min_periods=3), then map results back
    to the original rows. A window spanning a data gap propagates NaN rather
    than silently averaging across missing years.
    """
    if df.empty:
        df["rolling_3yr_avg"] = np.nan
        return df

    # Pivot: rows=year (full range), columns=geo_id, values=sf_units
    pivot = df.pivot_table(
        index="year", columns=geo_id_col,
        values="single_family_units", aggfunc="first"
    ).reindex(year_range)

    roll = pivot.rolling(window=3, min_periods=3).mean().round(1)

    # Stack back to long form and merge
    roll_long = (
        roll.stack(future_stack=True)
        .rename("rolling_3yr_avg")
        .reset_index()
    )
    roll_long.columns = ["year", geo_id_col, "rolling_3yr_avg"]
    roll_long["year"] = roll_long["year"].astype(int)

    df = df.merge(roll_long, on=["year", geo_id_col], how="left")
    return df


# ── Step 10c: Cumulative sums ──────────────────────────────────────────────
def add_cumulative(df: pd.DataFrame, geo_id_col: str) -> pd.DataFrame:
    """
    Add cumulative single-family units from three start years.
    For year Y, cumulative_sfh_XXXX_2025 is the sum of sf_units
    from XXXX through Y (inclusive) for that geography.
    Years before the start year receive NaN.
    """
    df = df.sort_values([geo_id_col, "year"])

    for start_yr, col in [
        (2000, "cumulative_sfh_2000_2025"),
        (2010, "cumulative_sfh_2010_2025"),
        (2020, "cumulative_sfh_2020_2025"),
    ]:
        df[col] = np.nan
        mask = df["year"] >= start_yr
        if mask.any():
            df.loc[mask, col] = (
                df.loc[mask]
                .groupby(geo_id_col)["single_family_units"]
                .cumsum()
            )
    return df


# ── Step 10d: Re-derive single_family_share cleanly ───────────────────────
def add_sf_share(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute share from cleaned numerics; avoids division-by-zero."""
    sf  = df["single_family_units"]
    tot = df["total_units"]
    df["single_family_share"] = np.where(
        tot.notna() & (tot > 0) & sf.notna(),
        (sf / tot).round(4),
        np.nan,
    )
    return df


# ── Main per-level processing function ────────────────────────────────────
def process(
    geo_type: str,
    all_dupes:   list,
    all_missing: list,
    val_summary: list,
) -> None:
    geo_id_col = GEO_ID_COL[geo_type]
    in_path    = INPUTS[geo_type]
    out_path   = OUTPUTS[geo_type]
    year_range = GEO_YEAR_RANGE[geo_type]
    yr_min, yr_max = min(year_range), max(year_range)

    print(f"\n{'='*60}")
    print(f"  {geo_type.upper()}")
    print(f"  Input : {in_path}")

    if not in_path.exists():
        print(f"  [SKIP] File not found. Run parsers 11-13 first.")
        val_summary.append({"geo_level": geo_type, "status": "input_missing"})
        return

    # ── Read ───────────────────────────────────────────────────────────────
    df = pd.read_csv(in_path, dtype=str)
    n_in = len(df)
    print(f"  Rows in: {n_in:,}")

    # Ensure geo_id column exists
    if geo_id_col not in df.columns:
        print(f"  [ERROR] Expected geo_id column '{geo_id_col}' not found in file.")
        print(f"          Available columns: {df.columns.tolist()}")
        val_summary.append({"geo_level": geo_type, "status": "missing_geo_id_col"})
        return

    # ── Steps 1-5: Clean ──────────────────────────────────────────────────
    df = clean_numeric(df)
    df = clean_names(df)
    df = clean_year(df)
    df = clean_fips(df, geo_type)

    # Rebuild geography_id column from the canonical geo_id_col
    df["geography_id"] = df[geo_id_col]

    # ── Filter out aggregate rows ─────────────────────────────────────────
    # State files include US total, 4 census regions (R1-R4), and 9 divisions
    # (D1-D9) as extra rows. Keep only valid 2-digit numeric state FIPS (01-56+72+78).
    if geo_type == "state":
        valid_fips = df["state_fips"].str.match(r"^\d{2}$", na=False)
        n_agg = (~valid_fips).sum()
        if n_agg:
            print(f"  Dropping {n_agg} non-state aggregate rows (US/region/division totals)")
            df = df[valid_fips].copy()
        # Pre-2004 BPS files used non-standard FIPS for territories:
        # 43 = Puerto Rico (correct: 72), 52 = Virgin Islands (correct: 78).
        # Remap to canonical codes then deduplicate.
        remap = {"43": "72", "52": "78"}
        if df["state_fips"].isin(remap.keys()).any():
            print("  Remapping legacy territory FIPS: 43->72 (PR), 52->78 (VI)")
            df["state_fips"] = df["state_fips"].replace(remap)
            df["geography_id"] = df["state_fips"]

    # County files include state-level subtotal rows (county_fips == "000").
    if geo_type == "county" and "county_fips" in df.columns:
        state_totals = df["county_fips"] == "000"
        n_st = state_totals.sum()
        if n_st:
            print(f"  Dropping {n_st} county-file state-subtotal rows (county_fips=000)")
            df = df[~state_totals].copy()

    # ── Step 6: Duplicates ────────────────────────────────────────────────
    df, dupes = check_duplicates(df, geo_id_col, geo_type)
    all_dupes.extend(dupes)
    n_dupes = len(dupes)
    if n_dupes:
        print(f"  [WARN] {n_dupes} duplicate (geo_id, year) pairs found and removed (kept first)")

    # ── Step 7: Missing years ─────────────────────────────────────────────
    missing = check_missing_years(df, geo_id_col, geo_type, year_range)
    all_missing.extend(missing)
    n_missing = len(missing)
    if n_missing:
        print(f"  [WARN] {n_missing} (geography, year) combinations missing from {yr_min}-{yr_max}")

    # ── Steps 8-9: Quality stats ──────────────────────────────────────────
    qc = data_quality_stats(df, geo_id_col)
    for k, v in qc.items():
        if v > 0:
            print(f"  [WARN] {k}: {v}")

    # ── Step 10: Derived fields ───────────────────────────────────────────
    print("  Adding derived fields...", end=" ")
    df = add_sf_share(df)
    df = add_yoy(df, geo_id_col)
    df = add_rolling_3yr(df, geo_id_col, year_range)
    df = add_cumulative(df, geo_id_col)
    print("done")

    # ── Filter to geo-specific year range and sort ────────────────────────
    df = df[df["year"].between(yr_min, yr_max)].copy()
    df = df.sort_values([geo_id_col, "year"]).reset_index(drop=True)

    # ── Write output (only keep defined output columns that exist) ─────────
    final_cols = [c for c in OUTPUT_COLS if c in df.columns]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[final_cols].to_csv(out_path, index=False)

    n_out = len(df)
    n_geog = df[geo_id_col].nunique()
    n_years   = df["year"].nunique()
    n_expected = len(year_range)
    print(f"  Rows out       : {n_out:,}")
    print(f"  Unique geogs   : {n_geog:,}")
    print(f"  Years covered  : {df['year'].min()}-{df['year'].max()} ({n_years} of {n_expected})")
    print(f"  Output         : {out_path}")

    # ── Validation summary row ─────────────────────────────────────────────
    val_summary.append({
        "geo_level":               geo_type,
        "status":                  "ok",
        "rows_in":                 n_in,
        "duplicate_pairs_removed": n_dupes // 2 if n_dupes else 0,
        "rows_out":                n_out,
        "unique_geographies":      n_geog,
        "years_covered":           n_years,
        "missing_year_instances":  n_missing,
        **{f"qc_{k}": v for k, v in qc.items()},
    })


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    all_dupes   = []
    all_missing = []
    val_summary = []

    for geo_type in ("state", "county", "cbsa"):
        process(geo_type, all_dupes, all_missing, val_summary)

    # ── Write reports ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Writing reports...")

    # Validation summary
    summary_path = REPORT_DIR / "bps_validation_summary.csv"
    pd.DataFrame(val_summary).to_csv(summary_path, index=False)
    print(f"  Validation summary : {summary_path}")

    # Missing years
    missing_path = REPORT_DIR / "bps_missing_years.csv"
    pd.DataFrame(all_missing, columns=["geo_level", "geography_id", "geography_name", "missing_year"]).to_csv(
        missing_path, index=False
    )
    print(f"  Missing years      : {missing_path}  ({len(all_missing):,} rows)")

    # Duplicate records
    dupe_path = REPORT_DIR / "bps_duplicate_records.csv"
    pd.DataFrame(all_dupes).to_csv(dupe_path, index=False)
    print(f"  Duplicate records  : {dupe_path}  ({len(all_dupes):,} rows)")

    print("\nAll done.")


if __name__ == "__main__":
    main()
