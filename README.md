# Affordable Net-Zero Single-Family Housing in the U.S.

Research data pipeline and interactive explorer for U.S. single-family housing
permit activity (Census Building Permits Survey, 2000–2025).

---

## Project structure

```
NPDP_Parcel_2022/
├── app/
│   └── streamlit_app.py          # Interactive explorer
├── data/
│   ├── raw/
│   │   ├── bps/
│   │   │   ├── state/            # st{year}a.txt  (downloaded)
│   │   │   ├── county/           # co{year}a.txt
│   │   │   └── cbsa/             # ma{year}a.txt
│   │   └── boundaries/
│   │       └── cbsa/             # Gazetteer centroid file (downloaded)
│   └── processed/
│       └── bps/
│           ├── bps_state_annual_2000_2025.csv
│           ├── bps_county_annual_2000_2025.csv
│           └── bps_cbsa_annual_2000_2025.csv
├── outputs/
│   ├── figures/                  # HTML maps and charts
│   └── reports/                  # Validation and summary CSVs
├── scripts/
│   ├── 00_download_bps.py
│   ├── 10_inspect_bps_files.py
│   ├── 11_parse_bps_state.py
│   ├── 12_parse_bps_county.py
│   ├── 13_parse_bps_cbsa.py
│   ├── 14_validate_and_standardize_bps.py
│   ├── 15_make_bps_cbsa_bubble_map.py
│   ├── 16_make_bps_treemap.py
│   ├── 17_make_bps_timeseries.py
│   └── bps_helpers.py
├── requirements.txt
└── README.md
```

---

## 1. Install requirements

Use the full path to the Miniconda pip if Python is not on your PATH:

```powershell
C:\Users\mizue\miniconda3\Scripts\pip.exe install -r requirements.txt
```

To enable PNG/SVG export from Plotly (optional):

```powershell
C:\Users\mizue\miniconda3\Scripts\pip.exe install kaleido
```

---

## 2. Run the preprocessing pipeline

Run all scripts from the **project root** (`D:\NPDP_Parcel_2022\`).
Use the full Python path if Python is not on your PATH.

### Step 1 — Download raw BPS files (78 files, ~30 MB)

```powershell
C:\Users\mizue\miniconda3\python.exe scripts/00_download_bps.py
```

Downloads annual state, county, and CBSA permit files (2000–2025) from the
Census FTP into `data/raw/bps/`. Skips files that already exist.

### Step 2 — (Optional) Inspect raw files

```powershell
C:\Users\mizue\miniconda3\python.exe scripts/10_inspect_bps_files.py
```

Writes a file inventory and sample rows to `outputs/reports/`.

### Step 3 — Parse raw files

Run the three parsers in any order:

```powershell
C:\Users\mizue\miniconda3\python.exe scripts/11_parse_bps_state.py
C:\Users\mizue\miniconda3\python.exe scripts/12_parse_bps_county.py
C:\Users\mizue\miniconda3\python.exe scripts/13_parse_bps_cbsa.py
```

Each parser writes a `*_rawparsed.csv` to `data/processed/bps/` and appends
any failures to `outputs/reports/bps_parse_failures.csv`.

### Step 4 — Validate and standardize

```powershell
C:\Users\mizue\miniconda3\python.exe scripts/14_validate_and_standardize_bps.py
```

Produces the three analysis-ready CSVs in `data/processed/bps/`:

| File | Rows (approx.) |
|---|---|
| `bps_state_annual_2000_2025.csv` | 1,350 |
| `bps_county_annual_2000_2025.csv` | 80,000+ |
| `bps_cbsa_annual_2000_2025.csv` | 24,000+ |

Also writes validation reports to `outputs/reports/`.

### Step 5 — (Optional) Generate standalone HTML outputs

```powershell
C:\Users\mizue\miniconda3\python.exe scripts/15_make_bps_cbsa_bubble_map.py
C:\Users\mizue\miniconda3\python.exe scripts/16_make_bps_treemap.py
C:\Users\mizue\miniconda3\python.exe scripts/17_make_bps_timeseries.py
```

These scripts download the Census CBSA Gazetteer centroid file on first run
and write self-contained HTML files to `outputs/figures/`.

---

## 3. Run the Streamlit app

After completing at least through Step 4, launch the app from the project root:

```powershell
C:\Users\mizue\miniconda3\Scripts\streamlit.exe run app/streamlit_app.py
```

The app opens at `http://localhost:8501` in your browser.

### App pages

| Page | Description |
|---|---|
| **CBSA Bubble Map** | Scatter map of metro areas; year slider, metric selector |
| **Treemap** | Top geographies by SF units; year slider, geo-level selector |
| **National / State Time Series** | U.S. total or per-state line charts with recession bands |
| **County Time Series** | Top counties within a selected state |
| **Data Quality Report** | Validation flags, missing years, parse failures, coverage stats |

### Controls available in the app

| Control | Applies to |
|---|---|
| **Year** slider | Bubble Map, Treemap |
| **Geography type** (State / County / CBSA) | Treemap |
| **State** selector | State Time Series, County Time Series |
| **Metric** | All chart pages |

Available metrics:

| Key | Description |
|---|---|
| `single_family_units` | Annual SF units authorized |
| `single_family_share` | SF units as share of all permitted units |
| `yoy_percent_change` | Year-over-year % change (NaN for first year per geography) |
| `rolling_3yr_avg` | 3-year centered rolling average (NaN if any gap in 3-year window) |

---

## Data sources

- **Census Building Permits Survey (BPS):** annual permit counts by structure type,
  at state, county, and CBSA levels.
  <https://www.census.gov/construction/bps/>
- **Census CBSA Gazetteer:** lat/lon centroids for CBSA bubble map.
  <https://www.census.gov/geographies/reference-files/time-series/geo/gazetteer-files.html>

---

## Notes

- Raw BPS files are not loaded in the Streamlit app. The app reads only the
  processed CSVs from `data/processed/bps/`.
- FIPS codes (state, county) are stored as zero-padded strings (e.g. `"06"`, `"06037"`).
  Always read them with `dtype=str` in pandas.
- County rows where `county_fips == "000"` are state totals; they are excluded
  from county-level charts.
- CBSA codes for pre-2004 files may be MSA codes rather than CBSA codes.
  These are flagged in `outputs/reports/bps_parse_failures.csv`.
- PNG export from Plotly requires the `kaleido` package (not installed by default).
