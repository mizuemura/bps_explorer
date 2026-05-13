"""
15_make_bps_cbsa_bubble_map.py

Map 2: Interactive CBSA bubble map of single-family housing units authorized
by Census Building Permits Survey (2000-2025).

Run from project root:
    C:\\Users\\mizue\\miniconda3\\python.exe scripts/15_make_bps_cbsa_bubble_map.py

Centroid source (auto-downloaded):
    Census CBSA Gazetteer file — tab-delimited, provides INTPTLAT/INTPTLONG
    for each CBSA. No shapefile library (geopandas/pyogrio) required.
    Source: https://www.census.gov/geographies/reference-files/time-series/geo/gazetteer-files.html
    Cached to: data/raw/boundaries/cbsa/

Inputs:
    data/processed/bps/bps_cbsa_annual_2004_2024.csv
    data/raw/boundaries/cbsa/{GAZETTEER_YEAR}_Gaz_cbsa_national.txt  (downloaded)

Outputs:
    outputs/figures/bps_cbsa_bubble_map.html           (animated, all years)
    outputs/figures/bps_cbsa_bubble_map_latest_year.html
    outputs/figures/bps_cbsa_bubble_map_latest_year.png (requires kaleido)

Static PNG:
    kaleido is NOT currently installed. To enable PNG export:
        C:\\Users\\mizue\\miniconda3\\Scripts\\pip.exe install kaleido
"""

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    print("ERROR: plotly not installed.")
    print("  C:\\Users\\mizue\\miniconda3\\Scripts\\pip.exe install plotly")
    sys.exit(1)

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT      = Path(__file__).resolve().parent.parent
BPS_CBSA_CSV      = PROJECT_ROOT / "data" / "processed" / "bps" / "bps_cbsa_annual_2004_2024.csv"
CBSA_BOUNDARY_DIR = PROJECT_ROOT / "data" / "raw" / "boundaries" / "cbsa"
FIGURES_DIR       = PROJECT_ROOT / "outputs" / "figures"

# ── Census Gazetteer config ────────────────────────────────────────────────
# The Gazetteer provides population-weighted internal-point lat/lon for each CBSA.
# Using 2024 definitions; pre-2004 MSA codes in BPS will not match (logged).
# LAYOUT ASSUMPTION: tab-delimited, UTF-8 or latin-1, one header row.
# Expected columns: GEOID (5-char), NAME, LSAD, INTPTLAT, INTPTLONG
# If the download fails or columns don't match, the script prints manual steps.
GAZETTEER_YEAR = 2024
GAZETTEER_URL  = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
    f"{GAZETTEER_YEAR}_Gazetteer/"
    f"{GAZETTEER_YEAR}_Gaz_cbsa_national.zip"
)
GAZETTEER_TXT = CBSA_BOUNDARY_DIR / f"{GAZETTEER_YEAR}_Gaz_cbsa_national.txt"

# ── Map settings (edit these to change map appearance) ────────────────────
# COLOR_COL choices:
#   "single_family_share"  — always available (incl. year 2000); yellow→red
#   "yoy_percent_change"   — shows boom/bust clearly; red/green diverging;
#                            NaN for year 2000 (no prior year) shows as gray
COLOR_COL = "single_family_share"
SIZE_COL  = "single_family_units"
SIZE_MAX  = 45        # maximum bubble diameter in pixels
PLAY_MS   = 700       # milliseconds per animation frame


# ── 1. Gazetteer download ──────────────────────────────────────────────────
def download_gazetteer() -> Path | None:
    """
    Download and cache the Census CBSA Gazetteer .txt file.
    Returns the path to the cached file, or None on failure.
    """
    CBSA_BOUNDARY_DIR.mkdir(parents=True, exist_ok=True)

    if GAZETTEER_TXT.exists() and GAZETTEER_TXT.stat().st_size > 0:
        print(f"  Gazetteer already cached: {GAZETTEER_TXT.name}")
        return GAZETTEER_TXT

    print(f"  Downloading Census CBSA Gazetteer ({GAZETTEER_YEAR}) ...")
    try:
        r = requests.get(GAZETTEER_URL, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"  [FAIL] Download error: {e}")
        _print_manual_steps()
        return None

    try:
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            txt_members = [m for m in z.namelist() if m.lower().endswith(".txt")]
            if not txt_members:
                print(f"  [FAIL] No .txt file in zip. Contents: {z.namelist()}")
                return None
            raw_bytes = z.read(txt_members[0])

        with open(GAZETTEER_TXT, "wb") as f:
            f.write(raw_bytes)

        print(f"  Saved: {GAZETTEER_TXT}  ({GAZETTEER_TXT.stat().st_size / 1024:.0f} KB)")
        return GAZETTEER_TXT

    except Exception as e:
        print(f"  [FAIL] Could not extract zip: {e}")
        return None


def _print_manual_steps() -> None:
    print("\n  Manual download steps:")
    print(f"    1. Open: {GAZETTEER_URL}")
    print(f"    2. Extract the .txt file from the zip.")
    print(f"    3. Place it at: {GAZETTEER_TXT}")
    print(f"    4. Re-run this script.\n")


# ── 2. Parse Gazetteer ────────────────────────────────────────────────────
def load_centroids(gz_path: Path) -> pd.DataFrame:
    """
    Parse the Census Gazetteer tab-delimited file into a DataFrame.
    Returns columns: cbsa_code, cbsa_name_gaz, lat, lon [, lsad]

    LAYOUT ASSUMPTION (stable across 2013-2024 Gazetteer releases):
      - Tab-delimited, one header row
      - GEOID       : 5-digit CBSA code (string, zero-padded)
      - NAME        : full CBSA/MSA name
      - INTPTLAT    : latitude of internal point (float, signed)
      - INTPTLONG   : longitude of internal point (float, signed, negative for US)
      - LSAD        : M1 = Metropolitan Statistical Area, M2 = Micropolitan

    If column names differ (older format), the pick_col() function tries
    several known synonyms before raising a descriptive error.
    """
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            raw = pd.read_csv(gz_path, sep="\t", dtype=str, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f"Could not parse {gz_path.name} with any known encoding.")

    raw.columns = [c.strip() for c in raw.columns]
    lower_map   = {c.lower().replace(" ", ""): c for c in raw.columns}

    def pick_col(*candidates):
        for c in candidates:
            if c.replace(" ", "").lower() in lower_map:
                return lower_map[c.replace(" ", "").lower()]
        return None

    geoid_col = pick_col("geoid", "cbsafp", "cbsa")
    name_col  = pick_col("name", "cbsaname", "areaname")
    lat_col   = pick_col("intptlat", "intpt_lat", "latitude", "lat")
    lon_col   = pick_col("intptlong", "intpt_long", "longitude", "lon", "long")
    lsad_col  = pick_col("lsad")

    missing = [label for label, col in
               [("GEOID", geoid_col), ("INTPTLAT", lat_col), ("INTPTLONG", lon_col)]
               if col is None]
    if missing:
        raise RuntimeError(
            f"Gazetteer file is missing expected columns: {missing}\n"
            f"Columns found: {raw.columns.tolist()}\n"
            f"Check the file at {gz_path} and compare to:\n"
            "  https://www.census.gov/geographies/reference-files/time-series/geo/gazetteer-files.html"
        )

    out = pd.DataFrame({
        "cbsa_code":     raw[geoid_col].str.strip().str.zfill(5),
        "cbsa_name_gaz": raw[name_col].str.strip() if name_col else pd.Series("", index=raw.index),
        "lat":           pd.to_numeric(raw[lat_col].str.strip(), errors="coerce"),
        "lon":           pd.to_numeric(raw[lon_col].str.strip(), errors="coerce"),
    })
    if lsad_col:
        out["lsad"] = raw[lsad_col].str.strip()

    n_bad = out[["lat", "lon"]].isna().any(axis=1).sum()
    if n_bad:
        print(f"  [WARN] Dropped {n_bad} Gazetteer rows with missing coordinates.")
    out = out.dropna(subset=["lat", "lon"])
    print(f"  Loaded {len(out):,} CBSA centroids.")
    return out


# ── 3. Load BPS CBSA data ─────────────────────────────────────────────────
def load_bps() -> pd.DataFrame:
    if not BPS_CBSA_CSV.exists():
        raise FileNotFoundError(
            f"Not found: {BPS_CBSA_CSV}\n"
            "Run scripts 00 -> 14 first."
        )

    fips_dtypes = {"cbsa_code": str, "state_fips": str,
                   "county_fips": str, "full_county_fips": str}
    df = pd.read_csv(BPS_CBSA_CSV, dtype=fips_dtypes)
    df["cbsa_code"] = df["cbsa_code"].str.strip().str.zfill(5)
    df["year"]      = df["year"].astype(int)

    for col in ["single_family_units", "total_units", "single_family_share",
                "yoy_percent_change", "rolling_3yr_avg",
                "cumulative_sfh_2000_2025", "cumulative_sfh_2010_2025",
                "cumulative_sfh_2020_2025"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"  Loaded {len(df):,} CBSA-year records "
          f"({df.cbsa_code.nunique()} CBSAs, "
          f"{df.year.min()}-{df.year.max()})")
    return df


# ── 4. Merge BPS with centroids ───────────────────────────────────────────
def merge(bps: pd.DataFrame, centroids: pd.DataFrame) -> pd.DataFrame:
    merged = bps.merge(
        centroids[["cbsa_code", "lat", "lon", "cbsa_name_gaz"]],
        on="cbsa_code", how="left",
    )
    unmatched_codes = merged.loc[merged["lat"].isna(), "cbsa_code"].unique()
    if len(unmatched_codes):
        unmatched_rows = merged["lat"].isna().sum()
        print(f"  [WARN] {len(unmatched_codes)} CBSA codes ({unmatched_rows:,} rows) "
              f"have no Gazetteer match and will be excluded.")
        print(f"         Likely cause: pre-2004 MSA codes in BPS do not match "
              f"{GAZETTEER_YEAR} CBSA definitions.")
        top = sorted(unmatched_codes[:15])
        print(f"         Sample unmatched: {top}" + (" ..." if len(unmatched_codes) > 15 else ""))

    out = merged.dropna(subset=["lat", "lon"]).copy()
    print(f"  Matched {out.cbsa_code.nunique():,} of {bps.cbsa_code.nunique():,} "
          f"CBSAs ({len(out):,} records).")
    return out


# ── 5. Color helpers ──────────────────────────────────────────────────────
def _color_cfg(col: str, df: pd.DataFrame) -> dict:
    """Return color_continuous_scale and range_color appropriate for col."""
    vals = df[col].dropna()
    if col == "yoy_percent_change":
        # Symmetric range clipped at the 3rd/97th percentile to avoid outlier dominance
        cap = float(min(max(np.nanpercentile(vals, 97), 20), 100))
        return dict(
            color_continuous_scale="RdYlGn",
            range_color=[-cap, cap],
        )
    else:  # single_family_share
        lo = float(np.nanpercentile(vals, 2))
        hi = float(np.nanpercentile(vals, 98))
        return dict(
            color_continuous_scale=[
                [0.0,  "#ffffb2"],
                [0.25, "#fecc5c"],
                [0.5,  "#fd8d3c"],
                [0.75, "#f03b20"],
                [1.0,  "#bd0026"],
            ],
            range_color=[max(lo, 0.0), min(hi, 1.0)],
        )


def _color_label(col: str) -> str:
    return {
        "single_family_share": "SF Share",
        "yoy_percent_change":  "YoY % Chg",
    }.get(col, col)


# ── 6. Hover template ────────────────────────────────────────────────────
def _hover_data(df: pd.DataFrame) -> dict:
    """
    Build hover_data dict for px.scatter_geo.
    Each key is a column name; value is a format string or bool.
    Columns absent from df are excluded to avoid Plotly errors.
    """
    candidates = {
        "year":                 True,
        "single_family_units":  ":,.0f",
        "total_units":          ":,.0f",
        "single_family_share":  ":.3f",
        "yoy_percent_change":   ":.1f",
        "rolling_3yr_avg":      ":,.0f",
        # suppress coordinate columns from tooltip
        "lat":                  False,
        "lon":                  False,
        "cbsa_name_gaz":        False,
    }
    return {k: v for k, v in candidates.items() if k in df.columns}


# ── 7. Figure builders ───────────────────────────────────────────────────
def _base_geo_layout(fig: go.Figure, color_label: str) -> go.Figure:
    """Apply shared geo + layout settings to any scatter_geo figure."""
    fig.update_geos(
        scope="usa",
        projection_type="albers usa",
        showland=True,    landcolor="rgb(242, 242, 238)",
        showlakes=True,   lakecolor="rgb(200, 222, 240)",
        showrivers=True,  rivercolor="rgb(200, 222, 240)",
        showcoastlines=True, coastlinecolor="rgb(160,160,160)",
        showsubunits=True,   subunitcolor="rgb(200,200,200)",
        showframe=False,
    )
    fig.update_layout(
        height=660,
        margin=dict(l=0, r=0, t=90, b=10),
        font=dict(family="Arial, sans-serif", size=12),
        coloraxis_colorbar=dict(
            title=dict(text=color_label, side="right"),
            thickness=14, len=0.55, x=1.0,
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def build_animated_map(df: pd.DataFrame) -> go.Figure:
    """
    Build an animated scatter_geo figure with a Play button and year slider.
    Each frame = one year; bubble size = single_family_units;
    bubble color = COLOR_COL (configured at top of script).
    """
    df_plot = df.sort_values(["year", "cbsa_code"]).copy()
    clabel  = _color_label(COLOR_COL)

    fig = px.scatter_geo(
        df_plot,
        lat="lat",
        lon="lon",
        size=SIZE_COL,
        color=COLOR_COL,
        hover_name="geography_name",
        hover_data=_hover_data(df_plot),
        animation_frame="year",
        size_max=SIZE_MAX,
        projection="natural earth",   # overridden by update_geos below
        title=(
            "U.S. Single-Family Housing Authorizations by Metro Area<br>"
            f"<sup>Bubble = SF units authorized | Color = {clabel} | "
            "Source: Census Building Permits Survey</sup>"
        ),
        labels={
            SIZE_COL:              "SF Units",
            "single_family_share": "SF Share",
            "yoy_percent_change":  "YoY %",
            "rolling_3yr_avg":     "3-yr Avg",
            COLOR_COL:             clabel,
        },
        **_color_cfg(COLOR_COL, df_plot),
    )

    fig = _base_geo_layout(fig, clabel)

    # Play / Pause buttons
    fig.update_layout(
        updatemenus=[dict(
            type="buttons", direction="left",
            showactive=False, x=0.18, y=0.02, xanchor="right",
            pad={"r": 10, "t": 10},
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, {
                         "frame":      {"duration": PLAY_MS, "redraw": True},
                         "fromcurrent": True,
                         "transition": {"duration": 200},
                     }]),
                dict(label="Pause", method="animate",
                     args=[[None], {
                         "frame":      {"duration": 0, "redraw": False},
                         "mode":       "immediate",
                         "transition": {"duration": 0},
                     }]),
            ],
        )],
    )

    # Slider cosmetics (wrapped in try/except: slider may be None before data loads)
    try:
        if fig.layout.sliders:
            slider_patch = dict(
                currentvalue={"prefix": "Year: ", "font": {"size": 14, "color": "#333"}},
                pad={"t": 50, "b": 10},
                len=0.82, x=0.18,
            )
            existing = fig.layout.sliders[0].to_plotly_json()
            existing.update(slider_patch)
            fig.update_layout(sliders=[existing])
    except Exception:
        pass  # not critical

    return fig


def build_latest_map(df: pd.DataFrame) -> tuple[go.Figure, int]:
    """
    Build a static scatter_geo for the most recent year with data.
    Returns (figure, year_int).
    """
    latest_yr  = int(df["year"].max())
    df_latest  = df[df["year"] == latest_yr].copy()
    clabel     = _color_label(COLOR_COL)

    # For latest-year map, use yoy_percent_change as color if available
    # (more informative for a single-year snapshot)
    use_col = "yoy_percent_change" if (
        "yoy_percent_change" in df_latest.columns
        and df_latest["yoy_percent_change"].notna().sum() > 5
        and COLOR_COL == "single_family_share"
    ) else COLOR_COL
    use_label = _color_label(use_col)

    fig = px.scatter_geo(
        df_latest,
        lat="lat",
        lon="lon",
        size=SIZE_COL,
        color=use_col,
        hover_name="geography_name",
        hover_data=_hover_data(df_latest),
        size_max=SIZE_MAX,
        projection="natural earth",
        title=(
            f"U.S. Single-Family Housing Authorizations by Metro Area ({latest_yr})<br>"
            f"<sup>Bubble = SF units authorized | Color = {use_label} | "
            "Source: Census BPS</sup>"
        ),
        labels={
            SIZE_COL:              "SF Units",
            "single_family_share": "SF Share",
            "yoy_percent_change":  "YoY %",
            "rolling_3yr_avg":     "3-yr Avg",
            use_col:               use_label,
        },
        **_color_cfg(use_col, df_latest),
    )

    fig = _base_geo_layout(fig, use_label)
    return fig, latest_yr


# ── 8. Save outputs ───────────────────────────────────────────────────────
def save_html(fig: go.Figure, path: Path, include_js: str = "cdn") -> None:
    """
    Write figure to HTML.
    include_js="cdn"  -> small file, requires internet to render
    include_js=True   -> fully self-contained (larger file, works offline)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs=include_js, full_html=True)
    kb = path.stat().st_size / 1024
    print(f"  HTML saved : {path.name}  ({kb:,.0f} KB)")


def save_png(fig: go.Figure, path: Path,
             width: int = 1600, height: int = 900) -> None:
    """
    Attempt PNG export via kaleido. Prints install instructions if unavailable.
    The animated figure exports its FIRST frame only.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(str(path), width=width, height=height, scale=2)
        kb = path.stat().st_size / 1024
        print(f"  PNG saved  : {path.name}  ({kb:,.0f} KB)")
    except Exception as e:
        print(f"  PNG skipped: {e}")
        print("  To enable PNG export, install kaleido:")
        print("    C:\\Users\\mizue\\miniconda3\\Scripts\\pip.exe install kaleido")


# ── Main ──────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("15_make_bps_cbsa_bubble_map.py")
    print("=" * 60)

    # 1. BPS data
    print("\n[1/4] Loading BPS CBSA data...")
    bps = load_bps()

    # 2. Centroids
    print("\n[2/4] Getting CBSA centroids (Census Gazetteer)...")
    gz_path = download_gazetteer()
    if gz_path is None:
        print("[!] Cannot build map without centroids. Exiting.")
        return
    centroids = load_centroids(gz_path)

    # 3. Merge
    print("\n[3/4] Merging...")
    df = merge(bps, centroids)
    if df.empty:
        print("[!] No matched records after merge. Cannot build map.")
        print("    Check that cbsa_code values in the BPS data are 5-digit codes "
              f"matching the {GAZETTEER_YEAR} Gazetteer.")
        return

    # 4. Build and save
    print("\n[4/4] Building figures...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Animated map (all years)
    print("  Building animated map (2000-2025)...")
    fig_anim = build_animated_map(df)
    anim_path = FIGURES_DIR / "bps_cbsa_bubble_map.html"
    save_html(fig_anim, anim_path)

    # Latest-year map
    fig_latest, latest_yr = build_latest_map(df)
    latest_html = FIGURES_DIR / "bps_cbsa_bubble_map_latest_year.html"
    latest_png  = FIGURES_DIR / "bps_cbsa_bubble_map_latest_year.png"
    save_html(fig_latest, latest_html)
    save_png(fig_latest, latest_png)

    print(f"\nOutputs:")
    print(f"  Animated map  : {anim_path}")
    print(f"  Latest ({latest_yr}) : {latest_html}")
    print(f"  PNG (if saved): {latest_png}")
    print(f"\nNote: HTML files use Plotly CDN — open in a browser with internet access.")
    print(f"      For offline HTML, change include_js='cdn' -> True in save_html() calls.")


if __name__ == "__main__":
    main()
