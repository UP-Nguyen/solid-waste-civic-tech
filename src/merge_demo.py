from __future__ import annotations

import re
import pandas as pd

from config import FEATURE_FILE, PROCESSED_DIR

REQUIRED_COLS = {"community_board", "population", "median_income"}
OPTIONAL_COLS = {
    "poverty_rate",
    "limited_english_pct",
    "broadband_pct",
    "pct_black",
    "pct_hispanic",
    "pct_white",
    "pct_asian",
    "source_puma",
    "source_name",
    "approximation_note",
}


def normalize_cb(value: object) -> str | None:
    if pd.isna(value):
        return None

    s = str(value).upper().strip()

    # collapse repeated whitespace
    s = re.sub(r"\s+", " ", s)

    # normalize borough wording variants
    s = s.replace("MN", "MANHATTAN")
    s = s.replace("BK", "BROOKLYN")
    s = s.replace("BX", "BRONX")
    s = s.replace("QN", "QUEENS")
    s = s.replace("SI", "STATEN ISLAND")

    # remove extra punctuation / parentheses if present
    s = s.replace("(", " ").replace(")", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()

    # standardize leading board number if present
    m = re.match(r"^(\d{1,2})\s+(.*)$", s)
    if m:
        board_num = f"{int(m.group(1)):02d}"
        borough = m.group(2).strip()
        return f"{board_num} {borough}"

    return s


def main() -> None:
    features = pd.read_parquet(FEATURE_FILE)
    demo_path = PROCESSED_DIR / "demographics_placeholder.csv"

    if not demo_path.exists():
        print("No demographic file found yet.")
        print(f"Put your demographic file here: {demo_path}")
        print("Required columns: community_board, population, median_income")
        return

    demo = pd.read_csv(demo_path)

    missing = REQUIRED_COLS - set(demo.columns)
    if missing:
        raise ValueError(
            "Demographic file is missing required columns: "
            f"{sorted(missing)}. Found columns: {sorted(demo.columns)}"
        )

    keep_cols = [c for c in demo.columns if c in REQUIRED_COLS or c in OPTIONAL_COLS]
    demo = demo[keep_cols].copy()

    # normalize join keys on both sides
    features["community_board"] = features["community_board"].map(normalize_cb)
    demo["community_board"] = demo["community_board"].map(normalize_cb)

    # debug: show unmatched boards
    feature_cb = set(features["community_board"].dropna().unique())
    demo_cb = set(demo["community_board"].dropna().unique())

    missing_in_demo = sorted(feature_cb - demo_cb)
    if missing_in_demo:
        print("Community boards in features but not demographics:")
        print(missing_in_demo)

    merged = features.merge(demo, on="community_board", how="left")

    numeric_cols = [
        "population",
        "median_income",
        "poverty_rate",
        "limited_english_pct",
        "broadband_pct",
        "pct_black",
        "pct_hispanic",
        "pct_white",
        "pct_asian",
    ]

    for col in numeric_cols:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged["complaints_per_1000"] = (
        merged["complaints"] / merged["population"].clip(lower=1)
    ) * 1000

    merged["repeat_descriptor_per_1000"] = (
        merged["repeat_descriptor_complaints"] / merged["population"].clip(lower=1)
    ) * 1000

    out_path = PROCESSED_DIR / "neighborhood_month_with_demo.parquet"
    merged.to_parquet(out_path, index=False)

    print(f"Saved merged dataset to {out_path}")
    print("Rows with missing population by borough:")
    if "borough" in merged.columns:
        print(merged[merged["population"].isna()].groupby("borough").size())
    print(merged.head())


if __name__ == "__main__":
    main()