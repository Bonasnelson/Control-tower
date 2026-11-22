"""Download the raw public datasets into data/raw/.

    python scripts/download_data.py            # Philadelphia + Chicago
    python scripts/download_data.py --city philadelphia

Philadelphia: City Payments FY2017 (OpenDataPhilly / Carto), 238,894 rows.
Chicago: City of Chicago "Payments" (Socrata s4vu-giwb), rows with a department.
Elliptic needs a Kaggle login and is downloaded by hand (see README).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
PHILA_COLS = ("fy,fm,check_date,document_no,dept,department_title,char_,character_title,sub_obj,"
              "sub_obj_title,vendor_name,doc_ref_no_prefix,doc_ref_no_prefix_definition,"
              "contract_number,contract_description,transaction_amount")
SOURCES = {
    "philadelphia": (
        "philadelphia_city_payments_fy2017.csv",
        "https://phl.carto.com/api/v2/sql",
        {"q": f"SELECT {PHILA_COLS} FROM city_payments_fy2017", "format": "csv"},
    ),
    "chicago": (
        "chicago_payments.csv",
        "https://data.cityofchicago.org/resource/s4vu-giwb.csv",
        {"$where": "department_name IS NOT NULL", "$limit": "1000000"},
    ),
}


def download(city: str, force: bool = False) -> Path:
    name, url, params = SOURCES[city]
    out = ROOT / "data" / "raw" / name
    if out.exists() and not force:
        print(f"[skip] {out} exists")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[get ] {city} -> {out}")
    with requests.get(url, params=params, stream=True, timeout=600) as r:
        r.raise_for_status()
        tmp = out.with_suffix(".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        tmp.replace(out)
    with open(out, encoding="utf-8") as f:
        rows = sum(1 for _ in f) - 1
    print(f"[done] {rows:,} rows")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", choices=list(SOURCES), nargs="*", default=list(SOURCES))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for c in a.city:
        download(c, a.force)
