from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_FILE = PROCESSED_DIR / "neighborhood_month.parquet"
MERGED_FILE = PROCESSED_DIR / "neighborhood_month_with_demo.parquet"
COMMUNITY_DISTRICT_GEOJSON = EXTERNAL_DIR / "community_districts.geojson"

BOROUGH_CODES = {
    "MANHATTAN": 1,
    "MN": 1,
    "BRONX": 2,
    "BX": 2,
    "BROOKLYN": 3,
    "BK": 3,
    "K": 3,
    "QUEENS": 4,
    "QN": 4,
    "Q": 4,
    "STATEN ISLAND": 5,
    "SI": 5,
    "R": 5,
}


def load_main_dataset(prefer_merged: bool = True) -> pd.DataFrame:
    """Load the best available processed dataset."""
    path = MERGED_FILE if prefer_merged and MERGED_FILE.exists() else FEATURE_FILE
    if not path.exists():
        raise FileNotFoundError(
            "No processed dataset found. Run src/extract_311.py and src/build_features.py first."
        )
    return pd.read_parquet(path)



def parse_BoroCD(value: Any) -> int | None:
    """Parse community board strings such as '01 MANHATTAN' into numeric BoroCD values.
    NYC community district geojson files often use an integer key like 101, 202, 303, etc.
    This helper converts common text formats into that code.
    """
    if pd.isna(value):
        return None

    text = str(value).strip().upper()
    if not text:
        return None

    # Case 1: already numeric-looking, like 101 or 202
    if text.isdigit() and len(text) == 3:
        return int(text)

    # Case 2: patterns like "01 MANHATTAN" or "MANHATTAN 01"
    borough_match = None
    for borough_label, borough_code in BOROUGH_CODES.items():
        if borough_label in text:
            borough_match = borough_code
            break

    district_numbers = re.findall(r"\d+", text)
    if borough_match is None or not district_numbers:
        return None

    district = int(district_numbers[0])
    return borough_match * 100 + district



def add_BoroCD(df: pd.DataFrame, column: str = "community_board") -> pd.DataFrame:
    out = df.copy()
    out["BoroCD"] = out[column].apply(parse_BoroCD)
    return out



def load_community_district_geojson() -> dict[str, Any]:
    if not COMMUNITY_DISTRICT_GEOJSON.exists():
        raise FileNotFoundError(
            "Missing community district geojson. Save it to data/external/community_districts.geojson."
        )
    with open(COMMUNITY_DISTRICT_GEOJSON, "r", encoding="utf-8") as f:
        return json.load(f)



def summarize_for_choropleth(df: pd.DataFrame) -> pd.DataFrame:
    required = {"month", "community_board", "complaints"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for choropleth: {sorted(missing)}")

    metric_col = "complaints_per_1000" if "complaints_per_1000" in df.columns else "complaints"

    grouped = (
        df.groupby(["month", "borough", "community_board"], as_index=False)
        .agg(
            complaints=("complaints", "sum"),
            metric=(metric_col, "mean" if metric_col == "complaints_per_1000" else "sum"),
        )
        .rename(columns={"metric": metric_col})
    )
    grouped = add_BoroCD(grouped)
    grouped = grouped[grouped["BoroCD"].notna()].copy()
    grouped["BoroCD"] = grouped["BoroCD"].astype(int)
    return grouped



def summarize_for_scatter(df: pd.DataFrame) -> pd.DataFrame:
    required = {"month", "community_board", "complaints", "median_income"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Animated scatterplot needs demographic columns. Missing: "
            f"{sorted(missing)}. Run src/merge_demo.py after creating a demographic file."
        )
    if "complaints_per_1000" not in df.columns:
        raise ValueError("Animated scatterplot needs complaints_per_1000. Add population in merge_demo.py.")

    agg_dict: dict[str, tuple[str, str]] = {
        "complaints": ("complaints", "sum"),
        "complaints_per_1000": ("complaints_per_1000", "mean"),
        "median_income": ("median_income", "first"),
    }
    if "population" in df.columns:
        agg_dict["population"] = ("population", "first")
    if "pct_repeat_descriptor" in df.columns:
        agg_dict["pct_repeat_descriptor"] = ("pct_repeat_descriptor", "mean")

    grouped = df.groupby(["month", "borough", "community_board"], as_index=False).agg(**agg_dict)
    return grouped.sort_values(["month", "complaints_per_1000"], ascending=[True, False])



def summarize_for_ranking(df: pd.DataFrame) -> pd.DataFrame:
    required = {"month", "community_board", "repeat_descriptor_complaints", "complaints"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for ranking chart: {sorted(missing)}")

    agg_dict: dict[str, tuple[str, str]] = {
        "repeat_descriptor_complaints": ("repeat_descriptor_complaints", "sum"),
        "complaints": ("complaints", "sum"),
    }
    if "population" in df.columns:
        agg_dict["population"] = ("population", "first")

    grouped = df.groupby(["month", "borough", "community_board"], as_index=False).agg(**agg_dict)
    grouped["repeat_burden_share"] = grouped["repeat_descriptor_complaints"] / grouped["complaints"].clip(lower=1)
    if "population" in grouped.columns:
        grouped["repeat_burden_per_1000"] = (
            grouped["repeat_descriptor_complaints"] / grouped["population"].clip(lower=1)
        ) * 1000
    else:
        grouped["repeat_burden_per_1000"] = grouped["repeat_descriptor_complaints"]

    grouped = grouped.sort_values(["month", "repeat_burden_per_1000"], ascending=[True, False])
    grouped["rank_within_month"] = grouped.groupby("month")["repeat_burden_per_1000"].rank(
        method="first", ascending=False
    )
    return grouped
