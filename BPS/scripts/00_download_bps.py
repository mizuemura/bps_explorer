"""
00_download_bps.py

Download Census Building Permits Survey (BPS) annual files for 2000-2025.
Run from project root:  C:\\Users\\mizue\\miniconda3\\python.exe scripts/00_download_bps.py

Output folders:
    data/raw/bps/state/     <- st{year}a.txt
    data/raw/bps/county/    <- co{year}a.txt
    data/raw/bps/cbsa/      <- ma{year}a.txt  (2000-2023)
                               cbsa{year}a.txt (2024+)

Census FTP base: https://www2.census.gov/econ/bps/
Files are always the annual total ("a" suffix).

NOTE: Census renamed and reorganised the CBSA/Metro directory in 2024:
  2000-2023  ->  Metro (ending 2023)/ma{year}a.txt
  2024+      ->  CBSA (beginning Jan 2024)/cbsa{year}a.txt
2025 annual should be available by spring 2026; 404 = not yet released.
"""

import sys
import time
from pathlib import Path

import requests

# Force UTF-8 output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Download targets ───────────────────────────────────────────────────────
# Each entry: (geo_level, url_template, dest_dir, year_range)
#
# CBSA note: Census split the Metro folder in 2024.
#   2000-2023  Metro (ending 2023)/       filename: ma{year}a.txt
#   2024+      CBSA (beginning Jan 2024)/ filename: cbsa{year}a.txt
_CBSA_DIR = PROJECT_ROOT / "data" / "raw" / "bps" / "cbsa"

SOURCES = [
    (
        "state",
        "https://www2.census.gov/econ/bps/State/st{year}a.txt",
        PROJECT_ROOT / "data" / "raw" / "bps" / "state",
        range(2000, 2026),
    ),
    (
        "county",
        "https://www2.census.gov/econ/bps/County/co{year}a.txt",
        PROJECT_ROOT / "data" / "raw" / "bps" / "county",
        range(2000, 2026),
    ),
    (
        "cbsa",
        "https://www2.census.gov/econ/bps/Metro%20(ending%202023)/ma{year}a.txt",
        _CBSA_DIR,
        range(2004, 2024),   # 2004-2023 inclusive (pre-2004 uses MSA codes, not CBSA)
    ),
    (
        "cbsa",
        "https://www2.census.gov/econ/bps/CBSA%20(beginning%20Jan%202024)/cbsa{year}a.txt",
        _CBSA_DIR,
        range(2024, 2026),   # 2024-2025 inclusive
    ),
]

TIMEOUT   = 30   # seconds per request
RETRY_MAX = 3
RETRY_DELAY = 5  # seconds between retries


# ── Download helper ────────────────────────────────────────────────────────
def download_file(url: str, dest: Path, session: requests.Session) -> str:
    """
    Download url → dest.
    Returns one of: 'skipped' | 'ok' | '404' | 'error:<msg>'
    """
    if dest.exists() and dest.stat().st_size > 0:
        return "skipped"

    for attempt in range(1, RETRY_MAX + 1):
        try:
            resp = session.get(url, timeout=TIMEOUT, stream=True)
            if resp.status_code == 404:
                return "404"
            resp.raise_for_status()

            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            return "ok"

        except requests.exceptions.Timeout:
            if attempt < RETRY_MAX:
                time.sleep(RETRY_DELAY)
                continue
            return "error:timeout"
        except requests.exceptions.RequestException as e:
            if attempt < RETRY_MAX:
                time.sleep(RETRY_DELAY)
                continue
            return f"error:{e}"

    return "error:max_retries"


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    results = {"ok": 0, "skipped": 0, "404": 0, "error": 0}

    session = requests.Session()
    session.headers.update({"User-Agent": "research-pipeline/1.0"})

    total = sum(len(list(yr)) for _, _, _, yr in SOURCES)
    done  = 0

    for geo_level, url_tmpl, dest_dir, year_range in SOURCES:
        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{geo_level.upper()}]  -> {dest_dir}")

        for year in year_range:
            url  = url_tmpl.format(year=year)
            dest = dest_dir / Path(url).name
            done += 1

            status = download_file(url, dest, session)

            if status == "ok":
                tag = f"OK    {dest.stat().st_size / 1024:>8.1f} KB"
            elif status == "skipped":
                tag = "skip  (already exists)"
            elif status == "404":
                tag = "404   (not released yet)"
            else:
                tag = f"FAIL  {status}"

            print(f"  [{done:>3}/{total}] {year}  {tag}")

            if status == "ok":
                results["ok"] += 1
            elif status == "skipped":
                results["skipped"] += 1
            elif status == "404":
                results["404"] += 1
            else:
                results["error"] += 1

            # Polite delay between requests
            if status == "ok":
                time.sleep(0.5)

    print(f"\nDone.  Downloaded={results['ok']}  "
          f"Skipped={results['skipped']}  "
          f"404={results['404']}  "
          f"Errors={results['error']}")
    print("\nNext step: run scripts/10_inspect_bps_files.py to verify formats.")


if __name__ == "__main__":
    main()
