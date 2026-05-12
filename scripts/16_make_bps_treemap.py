"""
16_make_bps_treemap.py

Plot 1: Interactive treemaps of top geographies by single-family housing permits.

Run from project root:
    C:\\Users\\mizue\\miniconda3\\python.exe scripts/16_make_bps_treemap.py

Inputs:
    data/processed/bps/bps_state_annual_2000_2025.csv
    data/processed/bps/bps_county_annual_2000_2025.csv
    data/processed/bps/bps_cbsa_annual_2004_2024.csv

Outputs (6 HTML files):
    outputs/figures/bps_treemap_state_annual.html
    outputs/figures/bps_treemap_county_annual.html
    outputs/figures/bps_treemap_cbsa_annual.html
    outputs/figures/bps_treemap_state_cumulative_2020_2025.html
    outputs/figures/bps_treemap_county_cumulative_2020_2025.html
    outputs/figures/bps_treemap_cbsa_cumulative_2020_2025.html

Annual maps:
    - One go.Treemap trace per year (26 total, 2000-2025)
    - Slider toggles visibility via method='update'
    - Top 30 geographies per year (by single_family_units)
    - Tile size  = single_family_units
    - Tile color = single_family_share (0-1, consistent global scale per geo level)
    - Hover: name, SF units, total units, SF share, YoY%, 3-yr rolling avg

Cumulative maps:
    - Single trace for cumulative SF units 2020-2025 (top 30 by total cumulative)
    - Tile size  = cumulative_sfh_2020_2025
    - Tile color = avg SF share across 2020-2025 years with data
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
except ImportError:
    print("ERROR: plotly not installed.")
    print("  C:\\Users\\mizue\\miniconda3\\Scripts\\pip.exe install plotly")
    sys.exit(1)

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED    = PROJECT_ROOT / "data" / "processed" / "bps"
FIGURES_DIR  = PROJECT_ROOT / "outputs" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Settings ───────────────────────────────────────────────────────────────
TOP_N       = 30
YEARS       = list(range(2000, 2026))
CUMUL_START = 2020

GEO_CONFIGS = {
    "state": {
        "csv":    PROCESSED / "bps_state_annual_2000_2025.csv",
        "id_col": "geography_id",      # state FIPS
        "label":  "geography_name",
        "fips_dtype": {"geography_id": str, "state_fips": str},
        "title":  "State",
    },
    "county": {
        "csv":    PROCESSED / "bps_county_annual_2000_2025.csv",
        "id_col": "full_county_fips",
        "label":  "geography_name",
        "fips_dtype": {"full_county_fips": str, "state_fips": str, "county_fips": str},
        "title":  "County",
    },
    "cbsa": {
        "csv":    PROCESSED / "bps_cbsa_annual_2004_2024.csv",
        "id_col": "cbsa_code",
        "label":  "geography_name",
        "fips_dtype": {"cbsa_code": str},
        "title":  "CBSA / Metro Area",
    },
}

# ── Helpers ────────────────────────────────────────────────────────────────

def fmt(val, spec=",") -> str:
    """Format numeric value; return 'N/A' for NaN/None."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if spec == "%":
        return f"{val:+.1f}%"
    if spec == "share":
        return f"{val:.1%}"
    if spec == ",":
        return f"{int(val):,}"
    return str(val)


def load_geo(geo_key: str) -> pd.DataFrame:
    cfg = GEO_CONFIGS[geo_key]
    csv_path = cfg["csv"]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing {csv_path.name} -- run scripts/14_validate_and_standardize_bps.py first."
        )
    df = pd.read_csv(csv_path, dtype=cfg["fips_dtype"])
    df["year"] = df["year"].astype(int)
    # Ensure numeric columns
    for col in ["single_family_units", "total_units", "single_family_share",
                "yoy_percent_change", "rolling_3yr_avg",
                "cumulative_sfh_2000_2025", "cumulative_sfh_2010_2025",
                "cumulative_sfh_2020_2025"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def top_n_for_year(df: pd.DataFrame, year: int, id_col: str, n: int) -> pd.DataFrame:
    sub = df[df["year"] == year].copy()
    sub = sub.dropna(subset=["single_family_units"])
    sub = sub.nlargest(n, "single_family_units")
    return sub


def global_share_range(df: pd.DataFrame) -> tuple[float, float]:
    """Compute color scale min/max across the full dataset (ignore NaN)."""
    valid = df["single_family_share"].dropna()
    if valid.empty:
        return 0.0, 1.0
    return float(valid.min()), float(valid.max())


def make_trace(sub: pd.DataFrame, id_col: str, label_col: str,
               size_col: str, color_col: str,
               cmin: float, cmax: float,
               geo_title: str, visible: bool) -> go.Treemap:
    """Build one go.Treemap trace from a subset DataFrame."""
    ids     = sub[id_col].astype(str).tolist()
    labels  = sub[label_col].tolist()
    values  = sub[size_col].fillna(0).astype(int).tolist()
    colors  = sub[color_col].tolist()

    # Pre-format customdata columns to avoid NaN rendering issues in hovertemplate
    cd_sf    = [fmt(v) for v in sub["single_family_units"]]
    cd_tot   = [fmt(v) for v in sub["total_units"]]
    cd_share = [fmt(v, "share") for v in sub["single_family_share"]]
    cd_yoy   = [fmt(v, "%") for v in sub.get("yoy_percent_change", [np.nan] * len(sub))]
    cd_roll  = [fmt(v) for v in sub.get("rolling_3yr_avg", [np.nan] * len(sub))]

    customdata = list(zip(cd_sf, cd_tot, cd_share, cd_yoy, cd_roll))

    hovertemplate = (
        "<b>%{label}</b><br>"
        "SF units: %{customdata[0]}<br>"
        "Total units: %{customdata[1]}<br>"
        "SF share: %{customdata[2]}<br>"
        "YoY change: %{customdata[3]}<br>"
        "3-yr avg: %{customdata[4]}"
        "<extra></extra>"
    )

    return go.Treemap(
        ids=ids,
        labels=labels,
        parents=[""] * len(ids),
        values=values,
        customdata=customdata,
        hovertemplate=hovertemplate,
        texttemplate="<b>%{label}</b><br>%{value:,.0f}",
        marker=dict(
            colors=colors,
            colorscale="YlOrRd",
            cmin=cmin,
            cmax=cmax,
            showscale=True,
            colorbar=dict(
                title=dict(text="SF Share", side="right"),
                tickformat=".0%",
                thickness=14,
                len=0.7,
            ),
        ),
        visible=visible,
        name=str(geo_title),
    )


# ── Annual treemap ─────────────────────────────────────────────────────────

def build_annual_treemap(geo_key: str) -> go.Figure:
    cfg = GEO_CONFIGS[geo_key]
    id_col    = cfg["id_col"]
    label_col = cfg["label"]
    geo_title = cfg["title"]

    print(f"  Loading {cfg['csv'].name} ...")
    df = load_geo(geo_key)

    # Filter to rows that have actual data
    df = df.dropna(subset=["single_family_units"])

    # For county: exclude state-total rows (county_fips == "000")
    if geo_key == "county" and "county_fips" in df.columns:
        df = df[df["county_fips"] != "000"]

    cmin, cmax = global_share_range(df)

    traces = []
    steps  = []

    print(f"  Building {len(YEARS)} annual traces (top {TOP_N} per year) ...")
    for i, year in enumerate(YEARS):
        sub = top_n_for_year(df, year, id_col, TOP_N)
        if sub.empty:
            # Still add a placeholder trace so slider indices stay aligned
            trace = go.Treemap(
                ids=["no_data"], labels=[f"No data for {year}"],
                parents=[""], values=[1],
                visible=(i == 0), name=str(year),
            )
        else:
            trace = make_trace(sub, id_col, label_col,
                               "single_family_units", "single_family_share",
                               cmin, cmax, year, visible=(i == 0))
        traces.append(trace)

        visibility = [False] * len(YEARS)
        visibility[i] = True
        steps.append(dict(
            method="update",
            label=str(year),
            args=[
                {"visible": visibility},
                {"title": {"text": f"Top {TOP_N} {geo_title}s by SF Housing Permits -- {year}"}},
            ],
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=f"Top {TOP_N} {geo_title}s by SF Housing Permits -- {YEARS[0]}",
            font=dict(size=18),
            x=0.5,
            xanchor="center",
        ),
        margin=dict(t=110, l=10, r=10, b=80),
        sliders=[dict(
            active=0,
            currentvalue=dict(prefix="Year: ", font=dict(size=14)),
            pad=dict(t=50, b=10),
            steps=steps,
        )],
        annotations=[dict(
            text=(
                "Tile size = single-family units authorized | "
                "Color = SF share of all units | "
                f"Source: Census BPS 2000-2025"
            ),
            x=0.5, y=-0.06, xref="paper", yref="paper",
            showarrow=False, font=dict(size=11), xanchor="center",
        )],
    )
    return fig


# ── Cumulative treemap ─────────────────────────────────────────────────────

def build_cumulative_treemap(geo_key: str) -> go.Figure:
    cfg = GEO_CONFIGS[geo_key]
    id_col    = cfg["id_col"]
    label_col = cfg["label"]
    geo_title = cfg["title"]
    cumul_col = "cumulative_sfh_2020_2025"

    print(f"  Loading {cfg['csv'].name} for cumulative map ...")
    df = load_geo(geo_key)

    # For county: exclude state-total rows
    if geo_key == "county" and "county_fips" in df.columns:
        df = df[df["county_fips"] != "000"]

    # Keep only years in cumulative window
    window = df[df["year"] >= CUMUL_START].copy()

    if cumul_col not in window.columns or window[cumul_col].isna().all():
        raise ValueError(
            f"Column '{cumul_col}' missing or all-NaN in {cfg['csv'].name}. "
            "Run scripts/14_validate_and_standardize_bps.py first."
        )

    # One row per geography: take the last year's cumulative value (highest)
    latest = (
        window.sort_values("year")
        .groupby(id_col, as_index=False)
        .last()
    )

    # Compute average SF share across the window years
    avg_share = (
        window.groupby(id_col)["single_family_share"]
        .mean()
        .rename("avg_sf_share")
        .reset_index()
    )
    latest = latest.merge(avg_share, on=id_col, how="left")

    latest = latest.dropna(subset=[cumul_col])
    latest = latest.nlargest(TOP_N, cumul_col)

    if latest.empty:
        raise ValueError(f"No data for cumulative treemap ({geo_key}).")

    valid_share = latest["avg_sf_share"].dropna()
    cmin = float(valid_share.min()) if not valid_share.empty else 0.0
    cmax = float(valid_share.max()) if not valid_share.empty else 1.0

    ids    = latest[id_col].astype(str).tolist()
    labels = latest[label_col].tolist()
    values = latest[cumul_col].fillna(0).astype(int).tolist()
    colors = latest["avg_sf_share"].tolist()

    cd_cumul = [fmt(v) for v in latest[cumul_col]]
    cd_share = [fmt(v, "share") for v in latest["avg_sf_share"]]
    # Include latest-year SF units for context
    cd_sf_latest = [fmt(v) for v in latest["single_family_units"]]

    customdata = list(zip(cd_cumul, cd_share, cd_sf_latest))

    hovertemplate = (
        "<b>%{label}</b><br>"
        f"SF units {CUMUL_START}-2025: %{{customdata[0]}}<br>"
        f"Avg SF share {CUMUL_START}-2025: %{{customdata[1]}}<br>"
        "SF units (latest year): %{customdata[2]}"
        "<extra></extra>"
    )

    trace = go.Treemap(
        ids=ids,
        labels=labels,
        parents=[""] * len(ids),
        values=values,
        customdata=customdata,
        hovertemplate=hovertemplate,
        texttemplate="<b>%{label}</b><br>%{value:,.0f}",
        marker=dict(
            colors=colors,
            colorscale="YlOrRd",
            cmin=cmin,
            cmax=cmax,
            showscale=True,
            colorbar=dict(
                title=dict(text="Avg SF Share", side="right"),
                tickformat=".0%",
                thickness=14,
                len=0.7,
            ),
        ),
    )

    fig = go.Figure(data=[trace])
    fig.update_layout(
        title=dict(
            text=(
                f"Top {TOP_N} {geo_title}s by Cumulative SF Housing Permits "
                f"({CUMUL_START}-2025)"
            ),
            font=dict(size=18),
            x=0.5,
            xanchor="center",
        ),
        margin=dict(t=90, l=10, r=10, b=70),
        annotations=[dict(
            text=(
                "Tile size = cumulative SF units authorized | "
                "Color = avg SF share of all units | "
                "Source: Census BPS"
            ),
            x=0.5, y=-0.06, xref="paper", yref="paper",
            showarrow=False, font=dict(size=11), xanchor="center",
        )],
    )
    return fig


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    outputs = []

    for geo_key in ["state", "county", "cbsa"]:
        geo_title = GEO_CONFIGS[geo_key]["title"]

        # -- Annual --
        out_annual = FIGURES_DIR / f"bps_treemap_{geo_key}_annual.html"
        print(f"\n[{geo_title}] Annual treemap ...")
        try:
            fig = build_annual_treemap(geo_key)
            fig.write_html(
                str(out_annual),
                include_plotlyjs="cdn",
                full_html=True,
            )
            size_kb = out_annual.stat().st_size // 1024
            print(f"  Saved: {out_annual.name}  ({size_kb} KB)")
            outputs.append(out_annual)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
        except Exception as e:
            print(f"  ERROR: {e}")

        # -- Cumulative --
        out_cumul = FIGURES_DIR / f"bps_treemap_{geo_key}_cumulative_{CUMUL_START}_2025.html"
        print(f"\n[{geo_title}] Cumulative treemap ({CUMUL_START}-2025) ...")
        try:
            fig = build_cumulative_treemap(geo_key)
            fig.write_html(
                str(out_cumul),
                include_plotlyjs="cdn",
                full_html=True,
            )
            size_kb = out_cumul.stat().st_size // 1024
            print(f"  Saved: {out_cumul.name}  ({size_kb} KB)")
            outputs.append(out_cumul)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone. {len(outputs)} file(s) written to {FIGURES_DIR}")
    for p in outputs:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
