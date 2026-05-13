"""
Streamlit app — US Census BPS Explorer
ARCH 159/259 Spring 2026 towards Affordable Net-Zero Housing in the US

Run from project root:
    streamlit run app/streamlit_app.py

Pages:
  1. CBSA Bubble Map
  2. Treemap
  3. National / State Time Series
  4. County Time Series
  5. Data Quality Report
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED    = PROJECT_ROOT / "data" / "processed" / "bps"
REPORTS      = PROJECT_ROOT / "outputs" / "reports"
GAZ_FILE     = PROJECT_ROOT / "data" / "raw" / "boundaries" / "cbsa" / "2024_Gaz_cbsa_national.txt"

STATE_CSV  = PROCESSED / "bps_state_annual_2000_2025.csv"
COUNTY_CSV = PROCESSED / "bps_county_annual_2000_2025.csv"
CBSA_CSV   = PROCESSED / "bps_cbsa_annual_2004_2024.csv"

YEARS = list(range(2000, 2026))

METRICS: dict[str, str] = {
    "single_family_units":  "Single-Family Units",
    "single_family_value":  "SF Valuation ($1,000s)",
    "single_family_share":  "SF Share of All Permits",
    "multifamily_units":    "Multifamily Units",
    "multifamily_value":    "Multifamily Valuation ($1,000s)",
    "total_units":          "Total Units",
    "yoy_percent_change":   "Year-over-Year % Change",
    "rolling_3yr_avg":      "3-Year Rolling Average",
    "total_value":          "Total Valuation ($1,000s)",
}

DIVERGING_METRICS = {"yoy_percent_change"}

RECESSION_SHAPES = [
    dict(type="rect", xref="x", yref="paper", x0=2007, x1=2009,
         y0=0, y1=1, fillcolor="rgba(180,180,180,0.18)", line_width=0, layer="below"),
    dict(type="rect", xref="x", yref="paper", x0=2019.9, x1=2020.5,
         y0=0, y1=1, fillcolor="rgba(180,180,180,0.18)", line_width=0, layer="below"),
]

SOURCE_NOTE = "Source: U.S. Census Bureau Building Permits Survey"

FIPS_TO_ABBR: dict[str, str] = {
    "01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT",
    "10":"DE","11":"DC","12":"FL","13":"GA","15":"HI","16":"ID","17":"IL",
    "18":"IN","19":"IA","20":"KS","21":"KY","22":"LA","23":"ME","24":"MD",
    "25":"MA","26":"MI","27":"MN","28":"MS","29":"MO","30":"MT","31":"NE",
    "32":"NV","33":"NH","34":"NJ","35":"NM","36":"NY","37":"NC","38":"ND",
    "39":"OH","40":"OK","41":"OR","42":"PA","44":"RI","45":"SC","46":"SD",
    "47":"TN","48":"TX","49":"UT","50":"VT","51":"VA","53":"WA","54":"WV",
    "55":"WI","56":"WY","72":"PR","78":"VI",
}

# ── US Census Bureau region mapping (state FIPS → region) ─────────────────
_FIPS_TO_REGION: dict[str, str] = {
    **dict.fromkeys(["09","23","25","33","44","50","34","36","42"], "Northeast"),
    **dict.fromkeys(["17","18","26","39","55","19","20","27","29","31","38","46"], "Midwest"),
    **dict.fromkeys(
        ["10","12","13","24","37","45","51","54","11","01","21","28","47","05","22","40","48"],
        "South",
    ),
    **dict.fromkeys(["04","08","16","30","32","35","49","56","02","06","15","41","53"], "West"),
}
REGION_PALETTE = {
    "Northeast": "#3498db",
    "Midwest":   "#f39c12ff",
    "South":     "#27ae60",
    "West":      "#e74c3c",
    "Other":     "#95a5a6",
}

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="US Census BPS Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data loaders ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_state() -> pd.DataFrame | None:
    if not STATE_CSV.exists():
        return None
    df = pd.read_csv(STATE_CSV, dtype={"geography_id": str, "state_fips": str})
    df["year"] = df["year"].astype(int)
    _coerce_numerics(df)
    return df


@st.cache_data(show_spinner=False)
def load_county() -> pd.DataFrame | None:
    if not COUNTY_CSV.exists():
        return None
    df = pd.read_csv(
        COUNTY_CSV,
        dtype={"full_county_fips": str, "state_fips": str,
               "county_fips": str, "geography_id": str},
    )
    df["year"] = df["year"].astype(int)
    if "county_fips" in df.columns:
        df = df[df["county_fips"] != "000"].copy()
    _coerce_numerics(df)
    return df


@st.cache_data(show_spinner=False)
def load_cbsa() -> pd.DataFrame | None:
    if not CBSA_CSV.exists():
        return None
    df = pd.read_csv(CBSA_CSV, dtype={"cbsa_code": str, "geography_id": str})
    df["year"] = df["year"].astype(int)
    _coerce_numerics(df)
    return df


@st.cache_data(show_spinner=False)
def load_centroids() -> pd.DataFrame | None:
    if not GAZ_FILE.exists():
        return None
    df = pd.read_csv(GAZ_FILE, sep="\t", dtype={"GEOID": str}, encoding="latin-1")
    df.columns = df.columns.str.strip()
    df["GEOID"] = df["GEOID"].str.strip().str.zfill(5)
    df["lat"]   = pd.to_numeric(df["INTPTLAT"],  errors="coerce")
    df["lon"]   = pd.to_numeric(df["INTPTLONG"], errors="coerce")
    df["name"]  = df["NAME"].str.strip()
    return df[["GEOID", "lat", "lon", "name"]].dropna()


@st.cache_data(show_spinner=False, ttl=86400)
def load_county_geojson() -> dict | None:
    try:
        url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
        with urlopen(url, timeout=15) as r:
            return json.load(r)
    except Exception:
        return None


def _coerce_numerics(df: pd.DataFrame) -> None:
    for col in [
        "single_family_units", "total_units", "single_family_share",
        "yoy_percent_change", "rolling_3yr_avg",
        "single_family_value", "multifamily_units", "multifamily_value", "total_value",
        "cumulative_sfh_2000_2025", "cumulative_sfh_2010_2025", "cumulative_sfh_2020_2025",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _missing_csv_warning(name: str) -> None:
    st.warning(
        f"**{name}** not found in `data/processed/bps/`. "
        "Run the preprocessing scripts — see the Data Quality Report page for the pipeline order."
    )


def _colorscale(metric: str) -> str:
    return "RdBu" if metric in DIVERGING_METRICS else "YlOrRd"


def _metric_fmt(metric: str) -> str:
    if metric == "single_family_share":
        return ".1%"
    if metric == "yoy_percent_change":
        return "+.1f"
    return ",.0f"


def _fmt(v, spec: str = "units") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if spec == "%":
        return f"{v:+.1f}%"
    if spec == "share":
        return f"{v:.1%}"
    return f"{int(v):,}"


def _region_color(state_fips: str) -> str:
    region = _FIPS_TO_REGION.get(str(state_fips).zfill(2), "Other")
    return REGION_PALETTE[region]


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("US Census BPS Explorer")
    st.caption("ARCH 159/259 Spring 2026 towards Affordable Net-Zero Housing in the US")
    st.divider()
    st.markdown(
        "The purpose of the **Building Permits Survey (BPS)** is to provide national, state, "
        "and local statistics on new privately-owned residential construction. The United States "
        "Code, Title 13, authorizes this survey, provides for voluntary responses, and provides "
        "an exception to confidentiality for public records. Data are available monthly, "
        "year-to-date, and annually at the national, state, CBSA (formerly MSA), county and "
        "place levels. The design and structure of this tool were informed by the U.S. Census "
        "Bureau's Building Permits Survey data visualizations, and the tool was developed as a "
        "prototype with coding assistance from Claude."
    )
    st.divider()

    page = st.radio(
        "Page",
        ["CBSA Bubble Map", "Treemap", "National / State Time Series",
         "County Time Series", "Data Quality Report"],
        label_visibility="collapsed",
    )


# ── Page 1: CBSA Bubble Map ────────────────────────────────────────────────

def page_bubble_map() -> None:
    st.header("CBSA Bubble Map")
    st.caption(
        "Each bubble is a Core-Based Statistical Area (metro or micro area). "
        "**Bubble size** = single-family units authorized. "
        "**Bubble color** = selected metric. "
        "Drag the year slider to compare across years."
    )

    df_cbsa   = load_cbsa()
    centroids = load_centroids()

    if df_cbsa is None:
        _missing_csv_warning("bps_cbsa_annual_2004_2024.csv")
        return
    if centroids is None:
        st.warning(
            "Centroid file not found at `data/raw/boundaries/cbsa/`. "
            "Run `scripts/15_make_bps_cbsa_bubble_map.py` once to download it."
        )
        return

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        year = st.select_slider("Year", options=list(range(2004, 2025)), value=2024)
    with c2:
        metric = st.selectbox("Color metric", list(METRICS.keys()),
                              format_func=lambda k: METRICS[k])
    with c3:
        top_n = st.number_input("Show top N CBSAs", min_value=10, max_value=500,
                                value=200, step=10)

    sub = df_cbsa[df_cbsa["year"] == year].copy()
    sub = sub.merge(centroids, left_on="cbsa_code", right_on="GEOID", how="inner")
    sub = sub.dropna(subset=["single_family_units", "lat", "lon"])
    sub = sub.nlargest(top_n, "single_family_units")

    if sub.empty:
        st.info("No data for the selected year / geography combination.")
        return

    color_col  = metric if metric in sub.columns else "single_family_share"
    hover_name = "geography_name" if "geography_name" in sub.columns else "name"

    fig = px.scatter_geo(
        sub,
        lat="lat", lon="lon",
        size="single_family_units",
        color=color_col,
        color_continuous_scale="Sunset",
        hover_name=hover_name,
        hover_data={
            "single_family_units": ":,.0f",
            "total_units":         ":,.0f",
            "single_family_share": ":.1%",
            "yoy_percent_change":  ":+.1f",
            "lat": False, "lon": False,
        },
        size_max=40,
        scope="usa",
        title=f"{METRICS[metric]} — {year}  (top {top_n} CBSAs by SF units)",
    )
    fig.update_layout(margin=dict(t=50, l=0, r=0, b=0), height=540)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Top 10 CBSAs this year"):
        top10 = sub.nlargest(10, "single_family_units")[
            ["geography_name", "single_family_units", "total_units",
             "single_family_share", "yoy_percent_change"]
        ].rename(columns={**METRICS, "geography_name": "CBSA"})
        st.dataframe(top10, use_container_width=True, hide_index=True)

    st.caption(SOURCE_NOTE)


# ── Page 2: Treemap ────────────────────────────────────────────────────────

def page_treemap() -> None:
    st.header("Treemap")

    # ── Row 1: year mode + year selector + geography + top N ──────────────
    r1c1, r1c2, r1c3, r1c4 = st.columns([2, 3, 2, 1])
    with r1c1:
        year_mode = st.radio(
            "Year mode",
            ["Single year", "Year range", "All years (2000-2025)"],
            key="tm_ymode",
        )
    with r1c2:
        if year_mode == "Single year":
            year = st.select_slider("Year", options=YEARS, value=2024, key="tm_year")
            yr_from = yr_to = year
        elif year_mode == "Year range":
            yr_from = st.select_slider("From year", options=YEARS, value=2015, key="tm_yfrom")
            yr_to   = st.select_slider("To year",   options=YEARS, value=2024, key="tm_yto")
            if yr_to < yr_from:
                yr_to = yr_from
            st.caption(f"Summing {yr_from}–{yr_to}. Color metric shows period average.")
        else:
            yr_from, yr_to = min(YEARS), max(YEARS)
            st.caption("Cumulative totals 2000–2025. Color metric shows period average.")
    with r1c3:
        geo_type = st.selectbox("Geography", ["State", "CBSA", "County"])
    with r1c4:
        top_n = st.number_input("Top N", min_value=5, max_value=50, value=30, step=5)

    # ── Row 2: size metric + color metric + color mode ─────────────────────
    r2c1, r2c2, r2c3 = st.columns([2, 2, 2])
    with r2c1:
        size_metric = st.selectbox(
            "Tile size",
            ["single_family_units", "total_units", "multifamily_units"],
            format_func=lambda k: METRICS[k],
            key="tm_size",
        )
    with r2c2:
        color_metric = st.selectbox(
            "Tile color",
            list(METRICS.keys()),
            format_func=lambda k: METRICS[k],
            key="tm_color",
            index=2,  # default: SF share
        )
    with r2c3:
        region_disabled = (geo_type == "CBSA")
        color_mode = st.radio(
            "Color mode",
            ["By metric", "By US Census Region"],
            key="tm_cmode",
            horizontal=True,
            disabled=region_disabled,
        )
        if region_disabled:
            st.caption("Region coloring unavailable for CBSA (spans multiple states).")

    # ── Reset drill-down when controls change ──────────────────────────────
    _drill_ctx = (geo_type, year_mode, yr_from, yr_to)
    if st.session_state.get("tm_drill_ctx") != _drill_ctx:
        st.session_state.tm_drill_ctx = _drill_ctx
        st.session_state.tm_sel_fips  = None
        st.session_state.tm_sel_name  = None

    # ── Load data ──────────────────────────────────────────────────────────
    loaders  = {"State": load_state, "CBSA": load_cbsa, "County": load_county}
    id_cols  = {"State": "state_fips", "CBSA": "cbsa_code", "County": "full_county_fips"}
    df       = loaders[geo_type]()
    id_col   = id_cols[geo_type]

    if df is None:
        _missing_csv_warning(f"bps_{geo_type.lower()}_annual_*.csv")
        return

    # ── Aggregate ──────────────────────────────────────────────────────────
    name_col = "geography_name"

    if year_mode == "Single year":
        sub = df[df["year"] == year].copy()
        if geo_type == "County" and "county_fips" in sub.columns:
            sub = sub[sub["county_fips"] != "000"]
        sub = sub.dropna(subset=[size_metric]).nlargest(top_n, size_metric)

        ids            = sub[id_col].astype(str).tolist()
        if geo_type == "County" and "state_fips" in sub.columns:
            labels = [
                f"{name}, {FIPS_TO_ABBR.get(str(sfips).zfill(2), str(sfips))}"
                for name, sfips in zip(sub[name_col], sub["state_fips"])
            ]
        else:
            labels = sub[name_col].tolist()
        size_vals      = sub[size_metric].fillna(0).astype(int).tolist()
        color_vals     = sub[color_metric].tolist() if color_metric in sub.columns else [0.0] * len(sub)
        state_fips_col = sub["state_fips"].tolist() if "state_fips" in sub.columns else [""] * len(sub)
        cd = list(zip(
            [_fmt(v) for v in sub["single_family_units"]],
            [_fmt(v) for v in sub["total_units"]],
            [_fmt(v, "share") for v in sub["single_family_share"]],
            [_fmt(v, "%") for v in sub.get("yoy_percent_change",
                                            pd.Series([np.nan] * len(sub)))],
        ))
        hover = (
            "<b>%{label}</b><br>"
            "SF units: %{customdata[0]}<br>"
            "Total units: %{customdata[1]}<br>"
            "SF share: %{customdata[2]}<br>"
            "YoY: %{customdata[3]}"
            "<extra></extra>"
        )
        title_suffix = str(year)

    else:
        window   = df[(df["year"] >= yr_from) & (df["year"] <= yr_to)]
        grp_cols = [id_col, name_col]
        if "state_fips" in window.columns and id_col != "state_fips":
            grp_cols.append("state_fips")

        agg_kwargs: dict = {"sf_sum": (size_metric, "sum")}
        if color_metric in window.columns:
            agg_kwargs["color_avg"] = (color_metric, "mean")
        agg = window.groupby(grp_cols, as_index=False).agg(**agg_kwargs)
        agg = agg.nlargest(top_n, "sf_sum")

        ids       = agg[id_col].astype(str).tolist()
        if geo_type == "County" and "state_fips" in agg.columns:
            labels = [
                f"{name}, {FIPS_TO_ABBR.get(str(sfips).zfill(2), str(sfips))}"
                for name, sfips in zip(agg[name_col], agg["state_fips"])
            ]
        else:
            labels = agg[name_col].tolist()
        size_vals      = agg["sf_sum"].fillna(0).astype(int).tolist()
        color_vals     = agg["color_avg"].tolist() if "color_avg" in agg.columns else [0.0] * len(agg)
        state_fips_col = (
            agg["state_fips"].tolist() if "state_fips" in agg.columns
            else (agg[id_col].tolist() if id_col == "state_fips" else [""] * len(agg))
        )
        cd = list(zip(
            [_fmt(v) for v in agg["sf_sum"]],
            ["N/A"] * len(agg),
            ["N/A"] * len(agg),
            ["N/A"] * len(agg),
        ))
        hover = (
            "<b>%{label}</b><br>"
            f"SF units ({yr_from}–{yr_to} sum): %{{customdata[0]}}"
            "<extra></extra>"
        )
        title_suffix = f"{yr_from}–{yr_to} cumulative"

    if not ids:
        st.info("No data for the selected combination.")
        return

    # ── Color ──────────────────────────────────────────────────────────────
    if color_mode == "By US Census Region" and not region_disabled:
        marker_colors = [_region_color(f) for f in state_fips_col]
        base_marker = dict(colors=marker_colors, showscale=False)
        legend_html = "  ".join(
            f'<span style="color:{c}; font-size:18px;">■</span>&nbsp;{r}'
            for r, c in REGION_PALETTE.items() if r != "Other"
        )
        st.markdown(legend_html, unsafe_allow_html=True)
    else:
        valid_c = [v for v in color_vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        cmin = float(min(valid_c)) if valid_c else 0.0
        cmax = float(max(valid_c)) if valid_c else 1.0
        base_marker = dict(
            colors=color_vals,
            colorscale=_colorscale(color_metric),
            cmin=cmin, cmax=cmax,
            showscale=True,
            colorbar=dict(
                title=dict(text=METRICS[color_metric], side="right"),
                tickformat=_metric_fmt(color_metric).replace("+", ""),
                thickness=14, len=0.8,
            ),
        )

    # ── Build treemap trace ────────────────────────────────────────────────
    use_drill_down = (geo_type == "State" and year_mode == "Single year")

    def _color_label() -> str:
        return METRICS[color_metric] if color_mode != "By US Census Region" else "US Census Region"

    # ── County drill-down view ─────────────────────────────────────────────
    if use_drill_down and st.session_state.get("tm_sel_fips"):
        sel_fips = st.session_state.tm_sel_fips
        sel_name = st.session_state.tm_sel_name

        if st.button("← All States"):
            st.session_state.tm_sel_fips = None
            st.session_state.tm_sel_name = None
            st.rerun()

        df_county = load_county()
        if df_county is None:
            _missing_csv_warning("bps_county_annual_2000_2025.csv")
            return

        c_sub = df_county[
            (df_county["year"] == year) &
            (df_county["state_fips"].astype(str) == str(sel_fips))
        ].dropna(subset=[size_metric]).copy()

        if c_sub.empty:
            st.info(f"No county data for {sel_name} in {year}.")
            return

        abbr     = FIPS_TO_ABBR.get(str(sel_fips).zfill(2), str(sel_fips))
        c_labels = [f"{n}, {abbr}" for n in c_sub["geography_name"]]
        c_ids    = c_sub["full_county_fips"].astype(str).tolist()
        c_vals   = c_sub[size_metric].fillna(0).astype(int).tolist()
        c_color_v = (
            c_sub[color_metric].tolist() if color_metric in c_sub.columns
            else [0.0] * len(c_sub)
        )

        if color_mode == "By US Census Region":
            c_marker = dict(colors=[_region_color(sel_fips)] * len(c_sub), showscale=False)
        else:
            valid_cv = [v for v in c_color_v if v is not None and not (isinstance(v, float) and np.isnan(v))]
            c_marker = dict(
                colors=c_color_v,
                colorscale=_colorscale(color_metric),
                cmin=float(min(valid_cv)) if valid_cv else 0.0,
                cmax=float(max(valid_cv)) if valid_cv else 1.0,
                showscale=True,
                colorbar=dict(
                    title=dict(text=METRICS[color_metric], side="right"),
                    tickformat=_metric_fmt(color_metric).replace("+", ""),
                    thickness=14, len=0.8,
                ),
            )

        c_cd = list(zip(
            [_fmt(v) for v in c_sub["single_family_units"]],
            [_fmt(v) for v in c_sub["total_units"]],
            [_fmt(v, "share") for v in c_sub["single_family_share"]],
            ["N/A"] * len(c_sub),
        ))
        c_hover = (
            "<b>%{label}</b><br>"
            "SF units: %{customdata[0]}<br>"
            "Total units: %{customdata[1]}<br>"
            "SF share: %{customdata[2]}"
            "<extra></extra>"
        )

        county_trace = go.Treemap(
            ids=c_ids,
            labels=c_labels,
            parents=[""] * len(c_ids),
            values=c_vals,
            customdata=c_cd,
            hovertemplate=c_hover,
            texttemplate="<b>%{label}</b><br>%{value:,.0f}",
            textfont=dict(size=16),
            marker=c_marker,
        )
        st.caption(f"**Tile size** = {METRICS[size_metric]} | **Tile color** = {_color_label()}")
        county_fig = go.Figure(data=[county_trace])
        county_fig.update_layout(
            title=f"{sel_name} Counties — {year}",
            margin=dict(t=50, l=5, r=5, b=5),
            height=600,
        )
        st.plotly_chart(county_fig, use_container_width=True)
        st.caption(SOURCE_NOTE)
        return

    # ── Flat treemap (state initial view, CBSA, county) ───────────────────
    trace = go.Treemap(
        ids=ids,
        labels=labels,
        parents=[""] * len(ids),
        values=size_vals,
        customdata=cd,
        hovertemplate=hover,
        texttemplate="<b>%{label}</b><br>%{value:,.0f}",
        textfont=dict(size=16),
        marker=base_marker,
    )
    drill_hint = " | Click a state to view its counties." if use_drill_down else ""
    st.caption(f"**Tile size** = {METRICS[size_metric]} | **Tile color** = {_color_label()}{drill_hint}")

    fig = go.Figure(data=[trace])
    fig.update_layout(
        title=f"Top {top_n} {geo_type}s — {title_suffix}",
        margin=dict(t=50, l=5, r=5, b=5),
        height=600,
    )

    if use_drill_down:
        event = st.plotly_chart(fig, use_container_width=True, on_select="rerun",
                                selection_mode="points")
        if event and event.selection and event.selection.points:
            pt = event.selection.points[0]
            # Match selected tile back to our ids/labels lists
            clicked_label = pt.get("label") or pt.get("text") or pt.get("id", "")
            clicked_id    = pt.get("id", "")
            matched_fips  = None
            matched_name  = None
            # Try matching by id first, then by label
            if clicked_id in ids:
                idx = ids.index(clicked_id)
                matched_fips = ids[idx]
                matched_name = labels[idx]
            elif clicked_label in labels:
                idx = labels.index(clicked_label)
                matched_fips = ids[idx]
                matched_name = clicked_label
            if matched_fips is not None:
                st.session_state.tm_sel_fips = matched_fips
                st.session_state.tm_sel_name = matched_name
                st.rerun()
    else:
        st.plotly_chart(fig, use_container_width=True)

    st.caption(SOURCE_NOTE)


# ── Page 3: National / State Time Series ──────────────────────────────────

def page_national_state() -> None:
    st.header("National / State Time Series")

    df_state = load_state()
    if df_state is None:
        _missing_csv_warning("bps_state_annual_2000_2025.csv")
        return

    view   = st.radio("View", ["National total", "Individual state"], horizontal=True)
    metric = st.selectbox(
        "Metric (Y axis)", list(METRICS.keys()),
        format_func=lambda k: METRICS[k], key="ts_metric",
    )

    if view == "National total":
        _national_chart(df_state, metric)
    else:
        _state_chart(df_state, metric)

    st.caption(SOURCE_NOTE + " | Shaded bands: 2007–09 recession, 2020 COVID shock.")


def _national_chart(df_state: pd.DataFrame, metric: str) -> None:
    # Sum additive columns across all states per year
    additive = ["single_family_units", "total_units", "single_family_value",
                "multifamily_units", "multifamily_value", "total_value"]
    agg_dict = {c: (c, "sum") for c in additive if c in df_state.columns}
    nat = df_state.groupby("year", as_index=False).agg(**agg_dict).sort_values("year")

    # Recompute derived metrics from national totals
    if "single_family_units" in nat.columns and "total_units" in nat.columns:
        nat["single_family_share"] = nat["single_family_units"] / nat["total_units"]
    if "single_family_units" in nat.columns:
        nat["rolling_3yr_avg"]     = nat["single_family_units"].rolling(3, min_periods=3).mean()
        nat["yoy_percent_change"]  = nat["single_family_units"].pct_change() * 100

    y_col  = metric if metric in nat.columns else "single_family_units"
    y_data = nat[y_col]
    y_fmt  = _metric_fmt(metric)
    y_label = METRICS[metric]

    # Rolling avg of the selected metric (for the overlay line)
    rolling = y_data.rolling(3, min_periods=3).mean()

    is_pct = metric in {"single_family_share", "yoy_percent_change"}
    bar_fmt = f"%{{y:{y_fmt}}}" + ("%" if "percent" in metric else "")

    st.caption(
        f"{y_label} aggregated across all 50 states + DC. "
        "Bar = annual value; red line = 3-year rolling average."
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=nat["year"], y=y_data,
        name=y_label, marker_color="#4e8fd9", opacity=0.7,
        hovertemplate=f"Year: %{{x}}<br>{y_label}: {bar_fmt}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=nat["year"], y=rolling,
        name="3-Year Rolling Avg", mode="lines",
        line=dict(color="#c0392b", width=2.5),
        hovertemplate=f"3-yr avg: {bar_fmt}<extra></extra>",
    ))

    valid = y_data.dropna()
    if not valid.empty:
        peak_idx = valid.abs().idxmax()
        peak_x   = nat.loc[peak_idx, "year"]
        peak_y   = valid[peak_idx]
        peak_txt = (
            f"Peak: {peak_y:.1%}" if metric == "single_family_share"
            else f"Peak: {peak_y:+.1f}%" if metric == "yoy_percent_change"
            else f"Peak: {int(peak_y):,}"
        )
        fig.add_annotation(
            x=peak_x, y=peak_y,
            text=peak_txt,
            showarrow=True, arrowhead=2, arrowcolor="#555",
            ax=30, ay=-40, font=dict(size=11, color="#555"),
        )

    ytick_fmt = y_fmt.replace("+", "")
    fig.update_layout(
        title=f"U.S. {y_label}, 2000–2025",
        xaxis=dict(title="Year", tickmode="linear", dtick=2,
                   showgrid=True, gridcolor="#e8e8e8"),
        yaxis=dict(title=y_label, tickformat=ytick_fmt,
                   showgrid=True, gridcolor="#e8e8e8", zeroline=False),
        shapes=RECESSION_SHAPES,
        hovermode="x unified",
        legend=dict(bgcolor="rgba(255,255,255,0.8)"),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=60, l=70, r=20, b=60), height=460,
    )
    st.plotly_chart(fig, use_container_width=True)


def _state_chart(df_state: pd.DataFrame, metric: str) -> None:
    state_names = (
        df_state[["geography_name", "geography_id"]]
        .drop_duplicates().sort_values("geography_name")
    )
    name_to_id = dict(zip(state_names["geography_name"], state_names["geography_id"]))

    c1, c2 = st.columns([3, 1])
    with c1:
        defaults = ["Texas", "Florida"] if "Texas" in name_to_id else list(name_to_id.keys())[:2]
        sel_names = st.multiselect(
            "State(s) — up to 6", options=list(name_to_id.keys()),
            default=defaults, max_selections=6,
        )
    with c2:
        show_rolling = st.checkbox("Show 3-yr rolling avg", value=True)

    if not sel_names:
        st.info("Select at least one state.")
        return

    colors = px.colors.qualitative.Plotly
    fig    = go.Figure()

    for i, sname in enumerate(sel_names):
        sid  = name_to_id[sname]
        sub  = df_state[df_state["geography_id"] == sid].sort_values("year")
        col  = colors[i % len(colors)]
        y_data = sub[metric] if metric in sub.columns else sub["single_family_units"]

        fig.add_trace(go.Scatter(
            x=sub["year"], y=y_data,
            name=sname, mode="lines+markers",
            line=dict(color=col, width=2), marker=dict(size=5),
            hovertemplate=(
                f"<b>{sname}</b><br>Year: %{{x}}<br>"
                f"{METRICS[metric]}: %{{y:{_metric_fmt(metric)}}}<extra></extra>"
            ),
        ))
        if show_rolling and "rolling_3yr_avg" in sub.columns and metric != "rolling_3yr_avg":
            fig.add_trace(go.Scatter(
                x=sub["year"], y=sub["rolling_3yr_avg"],
                name=f"{sname} 3-yr avg", mode="lines",
                line=dict(color=col, width=1.5, dash="dot"),
                showlegend=False,
                hovertemplate="3-yr avg: %{y:,.0f}<extra></extra>",
            ))

    fig.update_layout(
        title=f"{METRICS[metric]} by State, 2000–2025",
        xaxis=dict(title="Year", tickmode="linear", dtick=2,
                   showgrid=True, gridcolor="#e8e8e8"),
        yaxis=dict(title=METRICS[metric],
                   tickformat=_metric_fmt(metric).replace("+", ""),
                   showgrid=True, gridcolor="#e8e8e8", zeroline=False),
        shapes=RECESSION_SHAPES,
        hovermode="x unified",
        legend=dict(bgcolor="rgba(255,255,255,0.8)"),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=60, l=80, r=20, b=60), height=460,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Page 4: County Time Series ─────────────────────────────────────────────

def page_county() -> None:
    st.header("County Time Series")

    df_county = load_county()
    df_state  = load_state()

    if df_county is None:
        _missing_csv_warning("bps_county_annual_2000_2025.csv")
        return

    if df_state is not None:
        state_lookup = (
            df_state[["state_fips", "geography_name"]]
            .drop_duplicates().sort_values("geography_name")
        )
    else:
        state_lookup = (
            df_county[["state_fips"]].drop_duplicates()
            .assign(geography_name=lambda d: "FIPS " + d["state_fips"])
            .sort_values("geography_name")
        )

    fips_to_name = dict(zip(state_lookup["state_fips"], state_lookup["geography_name"]))
    name_to_fips = {v: k for k, v in fips_to_name.items()}

    ctrl1, ctrl2, ctrl3 = st.columns([3, 2, 1])
    with ctrl1:
        sel_state = st.selectbox(
            "State", sorted(name_to_fips.keys()),
            index=sorted(name_to_fips.keys()).index("Texas")
                  if "Texas" in name_to_fips else 0,
        )
    with ctrl2:
        metric = st.selectbox(
            "Metric (Y axis)", list(METRICS.keys()),
            format_func=lambda k: METRICS[k], key="co_metric",
        )
    with ctrl3:
        top_n = st.number_input("Top N counties", min_value=3, max_value=20, value=10, step=1)

    sfips = name_to_fips[sel_state]
    sub   = df_county[df_county["state_fips"] == sfips].copy()

    if sub.empty:
        st.info(f"No county data found for {sel_state} (FIPS {sfips}).")
        return

    # Rank counties by cumulative SF units 2000-2025
    if "cumulative_sfh_2000_2025" in sub.columns:
        rank = (
            sub.groupby(["full_county_fips", "geography_name"], as_index=False)
            ["cumulative_sfh_2000_2025"].max()
            .nlargest(top_n, "cumulative_sfh_2000_2025")
        )
    else:
        rank = (
            sub.groupby(["full_county_fips", "geography_name"], as_index=False)
            ["single_family_units"].sum()
            .nlargest(top_n, "single_family_units")
        )

    colors = px.colors.qualitative.Plotly

    # ── Layout: time series (left 3/5) + choropleth (right 2/5) ───────────
    ts_col, map_col = st.columns([3, 2])

    with ts_col:
        fig = go.Figure()
        for i, (_, crow) in enumerate(rank.iterrows()):
            cfips  = crow["full_county_fips"]
            cname  = crow["geography_name"]
            csub   = sub[sub["full_county_fips"] == cfips].sort_values("year")
            col    = colors[i % len(colors)]
            y_data = csub[metric] if metric in csub.columns else csub["single_family_units"]

            fig.add_trace(go.Scatter(
                x=csub["year"], y=y_data,
                name=cname, mode="lines+markers",
                line=dict(color=col, width=2), marker=dict(size=4),
                hovertemplate=(
                    f"<b>{cname}</b><br>Year: %{{x}}<br>"
                    f"{METRICS[metric]}: %{{y:{_metric_fmt(metric)}}}"
                    "<extra></extra>"
                ),
            ))

        fig.update_layout(
            title=f"Top {top_n} Counties in {sel_state} — {METRICS[metric]}",
            xaxis=dict(title="Year", tickmode="linear", dtick=2,
                       showgrid=True, gridcolor="#e8e8e8"),
            yaxis=dict(title=METRICS[metric],
                       tickformat=_metric_fmt(metric).replace("+", ""),
                       showgrid=True, gridcolor="#e8e8e8", zeroline=False),
            shapes=RECESSION_SHAPES,
            hovermode="closest",    # show only the nearest county on hover
            legend=dict(bgcolor="rgba(255,255,255,0.8)", font=dict(size=11)),
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=60, l=80, r=20, b=60), height=500,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(SOURCE_NOTE + " | Ranked by cumulative SF units 2000–2025.")

    with map_col:
        st.subheader("County Choropleth Map")
        avail_years = sorted(sub["year"].dropna().unique().astype(int).tolist())
        map_year = st.select_slider(
            "Map year", options=avail_years,
            value=avail_years[-1] if avail_years else 2024,
            key="co_myr",
        )

        with st.spinner("Loading county boundaries…"):
            county_geojson = load_county_geojson()

        if county_geojson is None:
            st.warning(
                "County boundary GeoJSON could not be loaded. "
                "An internet connection is required the first time."
            )
        else:
            map_col_name = metric if metric in sub.columns else "single_family_units"
            map_sub = sub[sub["year"] == map_year].dropna(subset=[map_col_name]).copy()

            if map_sub.empty:
                st.info(f"No county data for {sel_state} in {map_year}.")
            else:
                valid_c = map_sub[map_col_name].dropna()
                cmin = float(valid_c.min()) if not valid_c.empty else 0.0
                cmax = float(valid_c.max()) if not valid_c.empty else 1.0

                fig_map = px.choropleth(
                    map_sub,
                    geojson=county_geojson,
                    locations="full_county_fips",
                    color=map_col_name,
                    color_continuous_scale=_colorscale(metric),
                    range_color=(cmin, cmax),
                    scope="usa",
                    hover_name="geography_name",
                    hover_data={
                        map_col_name: f":{_metric_fmt(metric)}",
                        "full_county_fips": False,
                    },
                    labels={map_col_name: METRICS[metric]},
                    title=f"{METRICS[metric]}, {map_year}",
                )
                fig_map.update_geos(fitbounds="locations", visible=False)
                fig_map.update_layout(
                    margin=dict(t=50, l=0, r=0, b=0),
                    height=460,
                    coloraxis_colorbar=dict(
                        title=METRICS[metric],
                        tickformat=_metric_fmt(metric).replace("+", ""),
                        thickness=12, len=0.7,
                    ),
                )
                st.plotly_chart(fig_map, use_container_width=True)


# ── Page 5: Data Quality Report ────────────────────────────────────────────

def page_data_quality() -> None:
    st.header("Data Quality Report")

    with st.expander("Pipeline order (run from project root)", expanded=False):
        st.code(
            "00_download_bps.py\n"
            "11_parse_bps_state.py\n"
            "12_parse_bps_county.py\n"
            "13_parse_bps_cbsa.py\n"
            "14_validate_and_standardize_bps.py",
            language=None,
        )

    st.caption(
        "Summary statistics and validation flags generated by "
        "`scripts/14_validate_and_standardize_bps.py`. "
        "Run that script to refresh these tables."
    )

    val_summary = REPORTS / "bps_validation_summary.csv"
    if val_summary.exists():
        st.subheader("Validation summary")
        st.dataframe(pd.read_csv(val_summary), use_container_width=True, hide_index=True)
    else:
        st.info("`bps_validation_summary.csv` not found — run script 14.")

    missing_csv = REPORTS / "bps_missing_years.csv"
    if missing_csv.exists():
        st.subheader("Missing years")
        df_miss = pd.read_csv(missing_csv)
        st.caption(f"{len(df_miss):,} geography-year gaps detected.")
        geo_filter = st.selectbox(
            "Filter by geo level",
            ["All"] + sorted(df_miss["geo_level"].unique().tolist()),
            key="dq_geo",
        )
        if geo_filter != "All":
            df_miss = df_miss[df_miss["geo_level"] == geo_filter]
        st.dataframe(df_miss, use_container_width=True, hide_index=True)
    else:
        st.info("`bps_missing_years.csv` not found — run script 14.")

    dup_csv = REPORTS / "bps_duplicate_records.csv"
    if dup_csv.exists():
        st.subheader("Duplicate records")
        df_dup = pd.read_csv(dup_csv)
        if df_dup.empty:
            st.success("No duplicate records detected.")
        else:
            st.warning(f"{len(df_dup):,} duplicate rows found.")
            st.dataframe(df_dup, use_container_width=True, hide_index=True)

    fail_csv = REPORTS / "bps_parse_failures.csv"
    if fail_csv.exists():
        st.subheader("Parse failures")
        df_fail = pd.read_csv(fail_csv)
        if df_fail.empty:
            st.success("No parse failures.")
        else:
            st.warning(f"{len(df_fail):,} files failed to parse.")
            st.dataframe(df_fail, use_container_width=True, hide_index=True)

    st.subheader("Coverage check (from processed CSVs)")
    coverage_rows = []
    for label, loader, csv_path in [
        ("State",  load_state,  STATE_CSV),
        ("County", load_county, COUNTY_CSV),
        ("CBSA",   load_cbsa,   CBSA_CSV),
    ]:
        if not csv_path.exists():
            coverage_rows.append({"Level": label, "Status": "Missing", "Rows": "--",
                                   "Geographies": "--", "Years": "--", "SF units NaN%": "--"})
            continue
        dfx = loader()
        if dfx is None:
            coverage_rows.append({"Level": label, "Status": "Empty", "Rows": 0,
                                   "Geographies": 0, "Years": 0, "SF units NaN%": "--"})
            continue
        id_col = {"State": "geography_id", "County": "full_county_fips",
                  "CBSA": "cbsa_code"}.get(label, "geography_id")
        nan_pct = dfx["single_family_units"].isna().sum() / len(dfx) * 100 if len(dfx) else 0
        coverage_rows.append({
            "Level":         label,
            "Status":        "OK",
            "Rows":          f"{len(dfx):,}",
            "Geographies":   f"{dfx[id_col].nunique():,}" if id_col in dfx.columns else "--",
            "Years":         f"{dfx['year'].min()}–{dfx['year'].max()}",
            "SF units NaN%": f"{nan_pct:.1f}%",
        })
    st.dataframe(pd.DataFrame(coverage_rows), use_container_width=True, hide_index=True)


# ── Router ─────────────────────────────────────────────────────────────────

PAGES = {
    "CBSA Bubble Map":               page_bubble_map,
    "Treemap":                       page_treemap,
    "National / State Time Series":  page_national_state,
    "County Time Series":            page_county,
    "Data Quality Report":           page_data_quality,
}

PAGES[page]()
