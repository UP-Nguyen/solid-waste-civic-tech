from __future__ import annotations

"""Generate a real demographics CSV for the NYC sanitation dashboard.

This script pulls ACS 5-year profile data for New York PUMAs, parses the NYC
community district names embedded in the API's NAME field, and writes a CSV in
this schema:

community_board,population,median_income,poverty_rate,pct_black,pct_hispanic,pct_white,pct_asian

Notes
-----
- NYC PUMAs generally approximate Community Districts, but they are not a
  perfect one-to-one match everywhere. Some ACS PUMAs represent combined
  community districts (for example, "Districts 1 & 2").
- For combined districts, this script duplicates the same ACS demographic row
  to each referenced community board and adds a note column so we can keep the 
  caveat visible
- Percent variables are returned by the Census API as percentages from 0 to 100;
  this script converts them to proportions from 0 to 1 to match the dashboard's
  expected convention.
"""

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

ACS_YEAR = 2023
ACS_URL = f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5/profile"

# Variables chosen to match the dashboard schema.
ACS_VARIABLES = {
    "population": "DP05_0001E",       # total population
    "median_income": "DP03_0062E",   # median household income
    "poverty_rate": "DP03_0128PE",   # all people below poverty level (%)
    "pct_hispanic": "DP05_0076PE",   # Hispanic or Latino (of any race) (%)
    "pct_white": "DP05_0082PE",      # Not Hispanic or Latino: White alone (%)
    "pct_black": "DP05_0083PE",      # Not Hispanic or Latino: Black alone (%)
    "pct_asian": "DP05_0085PE",      # Not Hispanic or Latino: Asian alone (%)
}

BOROUGH_BY_PUMA_PREFIX = {
    "038": "MANHATTAN",
    "042": "BRONX",
    "043": "BROOKLYN",
    "044": "QUEENS",
    "045": "STATEN ISLAND",
}


def fetch_acs_profile() -> pd.DataFrame:
    params = {
        "get": ",".join(["NAME", *ACS_VARIABLES.values()]),
        "for": "public use microdata area:*",
        "in": "state:36",
    }
    resp = requests.get(ACS_URL, params=params, timeout=60)
    resp.raise_for_status()
    rows = resp.json()
    df = pd.DataFrame(rows[1:], columns=rows[0])

    # Keep only NYC PUMAs by code prefix.
    df["puma"] = df["public use microdata area"].astype(str).str.zfill(5)
    df = df[df["puma"].str[:3].isin(BOROUGH_BY_PUMA_PREFIX)].copy()
    return df


def extract_cd_numbers(name: str) -> list[int]:
    # Examples handled:
    # "NYC-Bronx Community District 12--Wakefield ..."
    # "NYC-Bronx Community Districts 1 & 2--Melrose ..."
    # "NYC-Manhattan Community Districts 4 & 5--Chelsea ..."
    m = re.search(r"Community Districts?\s+([^\-]+)--", name)
    if not m:
        return []
    number_text = m.group(1)
    nums = [int(x) for x in re.findall(r"\d+", number_text)]
    return nums


def borough_from_name_or_puma(name: str, puma: str) -> str:
    name_upper = name.upper()
    for boro in ["MANHATTAN", "BRONX", "BROOKLYN", "QUEENS", "STATEN ISLAND"]:
        if boro in name_upper:
            return boro
    return BOROUGH_BY_PUMA_PREFIX.get(puma[:3], "UNKNOWN")


def expand_to_community_boards(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for _, row in df.iterrows():
        puma = row["puma"]
        name = row["NAME"]
        boro = borough_from_name_or_puma(name, puma)
        cd_numbers = extract_cd_numbers(name)
        if not cd_numbers:
            continue
        is_combined = len(cd_numbers) > 1

        for cd in cd_numbers:
            rec = {
                "community_board": f"{cd:02d} {boro}",
                "source_name": name,
                "source_puma": puma,
                "approximation_note": (
                    "Combined ACS PUMA row duplicated across multiple community districts"
                    if is_combined
                    else "Single-district ACS PUMA"
                ),
            }
            for out_col, acs_col in ACS_VARIABLES.items():
                rec[out_col] = row.get(acs_col)
            records.append(rec)

    out = pd.DataFrame(records)

    # Convert numeric fields.
    for col in ACS_VARIABLES:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Convert percentages to proportions expected by your merge/demo code.
    pct_cols = ["poverty_rate", "pct_hispanic", "pct_white", "pct_black", "pct_asian"]
    for col in pct_cols:
        out[col] = out[col] / 100.0

    # Keep one row per community board. If duplicate rows exist for the same
    # community_board because of messy source labeling, keep the first.
    out = out.sort_values(["community_board", "source_puma"]).drop_duplicates(
        subset=["community_board"], keep="first"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="data/processed/demographics_placeholder.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw = fetch_acs_profile()
    final = expand_to_community_boards(raw)

    # Save the dashboard-facing columns first, keep provenance columns too.
    ordered_cols = [
        "community_board",
        "population",
        "median_income",
        "poverty_rate",
        "pct_black",
        "pct_hispanic",
        "pct_white",
        "pct_asian",
        "source_puma",
        "source_name",
        "approximation_note",
    ]
    final = final[ordered_cols]
    final.to_csv(out_path, index=False)

    print(f"Saved {len(final)} rows to {out_path}")
    print(final.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
