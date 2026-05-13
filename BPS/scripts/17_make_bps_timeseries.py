"""
17_make_bps_timeseries.py

Plot 1: Interactive time-series plots of single-family housing units authorized
from 2000-2025, at national, state, and county levels.

Run from project root:
    C:\\Users\\mizue\\miniconda3\\python.exe scripts/17_make_bps_timeseries.py

Inputs:
    data/processed/bps/bps_state_annual_2000_2025.csv
    data/processed/bps/bps_county_annual_2000_2025.csv
    data/processed/bps/bps_cbsa_annual_2004_2024.csv  (loaded but not plotted here)

Outputs (HTML):
    outputs/figures/bps_national_timeseries.html
    outputs/figures/bps_state_dropdown_timeseries.html
    outputs/figures/bps_top4_states_timeseries.html
    outputs/figures/bps_top_counties_in_top4_states_{state_slug}.html  (x4)

Outputs (CSV):
    outputs/reports/bps_top4_states.csv
    outputs/reports/bps_top_counties_by_state.csv

Design notes:
  - National: bar chart for annual SF units + dashed line for 3-yr rolling avg
    (rolling computed from the aggregated national series, not from state rows)
  - State dropdown: one pair of traces per state (actual + rolling avg);
    updatemenus dropdown toggles visibility so only the selected state shows.
    The rolling_3yr_avg column from script 14 is used directly.
  - Top 4 states: identified by cumulative_sfh_2000_2025 (falls back to sum
    of single_family_units if the cumulative column is missing).
  - County charts: top 10 counties per state by same cumulative logic.
    County-level state-total rows (county_fips == "000") are excluded.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.express as px
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
REPORTS_DIR  = PROJECT_ROOT / "outputs" / "reports"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

STATE_CSV  = PROCESSED / "bps_state_annual_2000_2025.csv"
COUNTY_CSV = PROCESSED / "bps_county_annual_2000_2025.csv"

TOP_N_STATES   = 4
TOP_N_COUNTIES = 10
YEARS          = list(range(2000, 2026))

# Plotly qualitative palette (distinct colours for multi-line charts)
COLORS = px.colors.qualitative.Plotly

# ── Shared layout defaults ─────────────────────────────────────────────────
LAYOUT_DEFAULTS = dict(
    font=dict(family="Arial, sans-serif", size=13),
    plot_bgcolor="white",
    paper_bgcolor="white",
    hovermode="x unified",
    legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="#ccc", borderwidth=1),
    xaxis=dict(
        showgrid=True, gridcolor="#e8e8e8",
        tickmode="linear", dtick=2,
    ),
)

RECESSION_SHAPES = [
    dict(type="rect", xref="x", yref="paper",
         x0=2007, x1=2009, y0=0, y1=1,
         fillcolor="rgba(180,180,180,0.18)", line_width=0,
         layer="below"),
    dict(type="rect", xref="x", yref="paper",
         x0=2020, x1=2020.5, y0=0, y1=1,
         fillcolor="rgba(180,180,180,0.18)", line_width=0,
         layer="below"),
]

SOURCE_NOTE = "Source: U.S. Census Bureau Building Permits Survey, 2000-2025"


# ── Data loaders ───────────────────────────────────────────────────────────

def load_state() -> pd.DataFrame:
    if not STATE_CSV.exists():
        raise FileNotFoundError(
            f"Missing {STATE_CSV.name} -- run scripts/14_validate_and_standardize_bps.py first."
        )
    df = pd.read_csv(STATE_CSV, dtype={"geography_id": str, "state_fips": str})
    df["year"] = df["year"].astype(int)
    for col in ["single_family_units", "total_units", "single_family_share",
                "yoy_percent_change", "rolling_3yr_avg",
                "cumulative_sfh_2000_2025", "cumulative_sfh_2010_2025",
                "cumulative_sfh_2020_2025"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_county() -> pd.DataFrame:
    if not COUNTY_CSV.exists():
        raise FileNotFoundError(
            f"Missing {COUNTY_CSV.name} -- run scripts/14_validate_and_standardize_bps.py first."
        )
    df = pd.read_csv(
        COUNTY_CSV,
        dtype={"full_county_fips": str, "state_fips": str, "county_fips": str,
               "geography_id": str},
    )
    df["year"] = df["year"].astype(int)
    # Drop state-total rows (county_fips == "000")
    if "county_fips" in df.columns:
        df = df[df["county_fips"] != "000"].copy()
    for col in ["single_family_units", "total_units", "single_family_share",
                "cumulative_sfh_2000_2025", "cumulative_sfh_2010_2025",
                "cumulative_sfh_2020_2025"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def cumulative_col_or_sum(df: pd.DataFrame, id_col: str,
                           name_col: str, n: int) -> pd.DataFrame:
    """Return top-n rows ranked by cumulative SF units (2000-2025).

    Uses cumulative_sfh_2000_2025 if present; falls back to summing
    single_family_units across all years.
    """
    if "cumulative_sfh_2000_2025" in df.columns:
        top = (
            df.groupby([id_col, name_col], as_index=False)["cumulative_sfh_2000_2025"]
            .max()
            .rename(columns={"cumulative_sfh_2000_2025": "cumulative_sf_units"})
            .nlargest(n, "cumulative_sf_units")
        )
    else:
        top = (
            df.groupby([id_col, name_col], as_index=False)["single_family_units"]
            .sum()
            .rename(columns={"single_family_units": "cumulative_sf_units"})
            .nlargest(n, "cumulative_sf_units")
        )
    return top.reset_index(drop=True)


def slugify(name: str) -> str:
    """Convert a name to a lowercase filename-safe slug."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def fmt_units(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{int(v):,}"


# ── 1. National time series ────────────────────────────────────────────────

def build_national_timeseries(df_state: pd.DataFrame) -> go.Figure:
    nat = (
        df_state.groupby("year", as_index=False)["single_family_units"]
        .sum()
        .sort_values("year")
    )
    nat["rolling_3yr"] = (
        nat["single_family_units"].rolling(3, min_periods=3).mean().round(0)
    )

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=nat["year"],
        y=nat["single_family_units"],
        name="Annual SF Units",
        marker_color="#4e8fd9",
        opacity=0.75,
        hovertemplate="Year: %{x}<br>SF Units: %{y:,.0f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=nat["year"],
        y=nat["rolling_3yr"],
        name="3-Year Rolling Avg",
        mode="lines",
        line=dict(color="#c0392b", width=2.5, dash="solid"),
        hovertemplate="3-yr avg: %{y:,.0f}<extra></extra>",
    ))

    peak_idx = nat["single_family_units"].idxmax()
    peak_row = nat.loc[peak_idx]
    fig.add_annotation(
        x=peak_row["year"], y=peak_row["single_family_units"],
        text=f"Peak: {fmt_units(peak_row['single_family_units'])}",
        showarrow=True, arrowhead=2, arrowcolor="#555",
        ax=30, ay=-40, font=dict(size=11, color="#555"),
    )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(
            text="U.S. Single-Family Housing Permits, 2000-2025",
            font=dict(size=20), x=0.5, xanchor="center",
        ),
        xaxis_title="Year",
        yaxis_title="Single-Family Units Authorized",
        yaxis=dict(tickformat=",", showgrid=True, gridcolor="#e8e8e8", zeroline=False),
        shapes=RECESSION_SHAPES,
        barmode="overlay",
        annotations=[
            dict(
                text=SOURCE_NOTE,
                x=0.5, y=-0.12, xref="paper", yref="paper",
                showarrow=False, font=dict(size=11, color="#777"), xanchor="center",
            ),
            dict(
                text="Shaded bands: 2007-2009 recession, 2020 COVID shock",
                x=0.5, y=-0.17, xref="paper", yref="paper",
                showarrow=False, font=dict(size=10, color="#999"), xanchor="center",
            ),
        ],
        margin=dict(t=90, l=80, r=30, b=100),
    )
    return fig


# ── 2. State dropdown time series ─────────────────────────────────────────

def build_state_dropdown(df_state: pd.DataFrame) -> go.Figure:
    states = (
        df_state[["geography_id", "geography_name"]]
        .drop_duplicates()
        .sort_values("geography_name")
    )

    traces  = []
    buttons = []

    n_states = len(states)
    traces_per_state = 2  # actual line + rolling avg line

    for i, (_, srow) in enumerate(states.iterrows()):
        sid   = srow["geography_id"]
        sname = srow["geography_name"]
        sub   = df_state[df_state["geography_id"] == sid].sort_values("year")

        color = COLORS[i % len(COLORS)]
        first = (i == 0)

        traces.append(go.Scatter(
            x=sub["year"],
            y=sub["single_family_units"],
            name="SF Units",
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=5),
            visible=first,
            hovertemplate=f"<b>{sname}</b><br>Year: %{{x}}<br>SF Units: %{{y:,.0f}}<extra></extra>",
        ))
        traces.append(go.Scatter(
            x=sub["year"],
            y=sub["rolling_3yr_avg"] if "rolling_3yr_avg" in sub.columns else [np.nan] * len(sub),
            name="3-Yr Rolling Avg",
            mode="lines",
            line=dict(color=color, width=2, dash="dash"),
            visible=first,
            hovertemplate="3-yr avg: %{y:,.0f}<extra></extra>",
        ))

        vis = [False] * (n_states * traces_per_state)
        vis[i * traces_per_state]     = True
        vis[i * traces_per_state + 1] = True

        buttons.append(dict(
            method="update",
            label=sname,
            args=[
                {"visible": vis},
                {"title": {"text": f"SF Housing Permits: {sname}, 2000-2025"}},
            ],
        ))

    first_name = states.iloc[0]["geography_name"]
    fig = go.Figure(data=traces)
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(
            text=f"SF Housing Permits: {first_name}, 2000-2025",
            font=dict(size=18), x=0.5, xanchor="center",
        ),
        xaxis_title="Year",
        yaxis_title="Single-Family Units Authorized",
        yaxis=dict(tickformat=",", showgrid=True, gridcolor="#e8e8e8", zeroline=False),
        shapes=RECESSION_SHAPES,
        updatemenus=[dict(
            type="dropdown",
            buttons=buttons,
            active=0,
            direction="down",
            showactive=True,
            x=0.01, xanchor="left",
            y=1.12, yanchor="top",
            bgcolor="white",
            bordercolor="#ccc",
            font=dict(size=12),
        )],
        annotations=[
            dict(
                text="Select state:",
                x=0.01, y=1.18, xref="paper", yref="paper",
                showarrow=False, font=dict(size=12), xanchor="left",
            ),
            dict(
                text=SOURCE_NOTE,
                x=0.5, y=-0.12, xref="paper", yref="paper",
                showarrow=False, font=dict(size=11, color="#777"), xanchor="center",
            ),
        ],
        margin=dict(t=120, l=80, r=30, b=90),
    )
    return fig


# ── 3. Top 4 states multi-line ─────────────────────────────────────────────

def build_top4_timeseries(df_state: pd.DataFrame,
                           top4: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    for i, row in top4.iterrows():
        sub = df_state[df_state["geography_id"] == row["geography_id"]].sort_values("year")
        color = COLORS[i % len(COLORS)]
        sname = row["geography_name"]
        cumul = fmt_units(row["cumulative_sf_units"])

        fig.add_trace(go.Scatter(
            x=sub["year"],
            y=sub["single_family_units"],
            name=f"{sname} ({cumul} total)",
            mode="lines+markers",
            line=dict(color=color, width=2.5),
            marker=dict(size=5),
            hovertemplate=f"<b>{sname}</b><br>Year: %{{x}}<br>SF Units: %{{y:,.0f}}<extra></extra>",
        ))
        if "rolling_3yr_avg" in sub.columns:
            fig.add_trace(go.Scatter(
                x=sub["year"],
                y=sub["rolling_3yr_avg"],
                name=f"{sname} 3-yr avg",
                mode="lines",
                line=dict(color=color, width=1.5, dash="dot"),
                showlegend=False,
                hovertemplate="3-yr avg: %{y:,.0f}<extra></extra>",
            ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(
            text=f"Top {TOP_N_STATES} States by SF Housing Permits, 2000-2025",
            font=dict(size=18), x=0.5, xanchor="center",
        ),
        xaxis_title="Year",
        yaxis_title="Single-Family Units Authorized",
        yaxis=dict(tickformat=",", showgrid=True, gridcolor="#e8e8e8", zeroline=False),
        shapes=RECESSION_SHAPES,
        annotations=[dict(
            text=SOURCE_NOTE,
            x=0.5, y=-0.12, xref="paper", yref="paper",
            showarrow=False, font=dict(size=11, color="#777"), xanchor="center",
        )],
        margin=dict(t=90, l=80, r=30, b=90),
    )
    return fig


# ── 4. County time series for one state ───────────────────────────────────

def build_county_timeseries(df_county: pd.DataFrame,
                             state_fips: str,
                             state_name: str) -> go.Figure:
    sub = df_county[df_county["state_fips"] == state_fips].copy()

    if sub.empty:
        raise ValueError(f"No county data found for state_fips={state_fips} ({state_name}).")

    top_counties = cumulative_col_or_sum(sub, "full_county_fips", "geography_name", TOP_N_COUNTIES)

    fig = go.Figure()

    for i, crow in top_counties.iterrows():
        cfips  = crow["full_county_fips"]
        cname  = crow["geography_name"]
        cumul  = fmt_units(crow["cumulative_sf_units"])
        csub   = sub[sub["full_county_fips"] == cfips].sort_values("year")
        color  = COLORS[i % len(COLORS)]

        fig.add_trace(go.Scatter(
            x=csub["year"],
            y=csub["single_family_units"],
            name=f"{cname} ({cumul} total)",
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=4),
            hovertemplate=f"<b>{cname}</b><br>Year: %{{x}}<br>SF Units: %{{y:,.0f}}<extra></extra>",
        ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(
            text=f"Top {TOP_N_COUNTIES} Counties in {state_name} -- SF Housing Permits, 2000-2025",
            font=dict(size=17), x=0.5, xanchor="center",
        ),
        xaxis_title="Year",
        yaxis_title="Single-Family Units Authorized",
        yaxis=dict(tickformat=",", showgrid=True, gridcolor="#e8e8e8", zeroline=False),
        shapes=RECESSION_SHAPES,
        annotations=[dict(
            text=SOURCE_NOTE,
            x=0.5, y=-0.12, xref="paper", yref="paper",
            showarrow=False, font=dict(size=11, color="#777"), xanchor="center",
        )],
        margin=dict(t=90, l=80, r=30, b=90),
    )
    return fig


# ── Summary CSV helpers ────────────────────────────────────────────────────

def save_top4_states_csv(top4: pd.DataFrame) -> None:
    out = REPORTS_DIR / "bps_top4_states.csv"
    top4.to_csv(out, index=False)
    print(f"  Saved: {out.name}")


def save_top_counties_csv(df_county: pd.DataFrame, top4: pd.DataFrame) -> None:
    rows = []
    for _, srow in top4.iterrows():
        sub  = df_county[df_county["state_fips"] == srow["geography_id"]].copy()
        tops = cumulative_col_or_sum(sub, "full_county_fips", "geography_name", TOP_N_COUNTIES)
        tops.insert(0, "state_fips", srow["geography_id"])
        tops.insert(1, "state_name", srow["geography_name"])
        rows.append(tops)
    out_df = pd.concat(rows, ignore_index=True)
    out = REPORTS_DIR / "bps_top_counties_by_state.csv"
    out_df.to_csv(out, index=False)
    print(f"  Saved: {out.name}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("Loading data ...")
    try:
        df_state = load_state()
        print(f"  State rows : {len(df_state):,}  ({df_state['year'].min()}-{df_state['year'].max()})")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        df_state = None

    try:
        df_county = load_county()
        print(f"  County rows: {len(df_county):,}")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        df_county = None

    outputs = []

    # ── 1. National time series ──────────────────────────────────────────
    if df_state is not None:
        print("\n[1] National time series ...")
        out = FIGURES_DIR / "bps_national_timeseries.html"
        try:
            fig = build_national_timeseries(df_state)
            fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
            print(f"  Saved: {out.name}  ({out.stat().st_size // 1024} KB)")
            outputs.append(out)
        except Exception as e:
            print(f"  ERROR: {e}")

    # ── 2. State dropdown ────────────────────────────────────────────────
    if df_state is not None:
        print("\n[2] State dropdown time series ...")
        out = FIGURES_DIR / "bps_state_dropdown_timeseries.html"
        try:
            fig = build_state_dropdown(df_state)
            fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
            print(f"  Saved: {out.name}  ({out.stat().st_size // 1024} KB)")
            outputs.append(out)
        except Exception as e:
            print(f"  ERROR: {e}")

    # ── 3. Top 4 states ──────────────────────────────────────────────────
    top4 = None
    if df_state is not None:
        print(f"\n[3] Top {TOP_N_STATES} states multi-line ...")
        try:
            top4 = cumulative_col_or_sum(
                df_state, "geography_id", "geography_name", TOP_N_STATES
            )
            print(f"  Top {TOP_N_STATES} states:")
            for _, row in top4.iterrows():
                print(f"    {row['geography_name']:30s}  {fmt_units(row['cumulative_sf_units'])} units")

            save_top4_states_csv(top4)

            out = FIGURES_DIR / "bps_top4_states_timeseries.html"
            fig = build_top4_timeseries(df_state, top4)
            fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
            print(f"  Saved: {out.name}  ({out.stat().st_size // 1024} KB)")
            outputs.append(out)
        except Exception as e:
            print(f"  ERROR: {e}")

    # ── 4. County plots for top 4 states ────────────────────────────────
    if df_county is not None and top4 is not None:
        print(f"\n[4] Top {TOP_N_COUNTIES} counties per top-{TOP_N_STATES} state ...")
        try:
            save_top_counties_csv(df_county, top4)
        except Exception as e:
            print(f"  ERROR saving county CSV: {e}")

        for _, srow in top4.iterrows():
            sfips  = srow["geography_id"]
            sname  = srow["geography_name"]
            slug   = slugify(sname)
            out    = FIGURES_DIR / f"bps_top_counties_in_top4_states_{slug}.html"
            print(f"  Building county chart for {sname} (FIPS {sfips}) ...")
            try:
                fig = build_county_timeseries(df_county, sfips, sname)
                fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
                print(f"    Saved: {out.name}  ({out.stat().st_size // 1024} KB)")
                outputs.append(out)
            except Exception as e:
                print(f"    ERROR: {e}")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\nDone. {len(outputs)} HTML file(s) written to {FIGURES_DIR}")
    for p in outputs:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
