"""Merge demographic data onto the monthly feature table."""

from __future__ import annotations

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
    print("Columns:")
    print(merged.columns.tolist())
    print(merged.head())


if __name__ == "__main__":
    main()