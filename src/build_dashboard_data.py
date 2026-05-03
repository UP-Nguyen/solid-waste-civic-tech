from __future__ import annotations

import pandas as pd

from config import FEATURE_FILE, PROCESSED_DIR


def main() -> None:
    src = PROCESSED_DIR / "neighborhood_month_with_demo.parquet"
    path = src if src.exists() else FEATURE_FILE
    if not path.exists():
        raise FileNotFoundError("Run src/extract_311.py and src/build_features.py first.")

    df = pd.read_parquet(path)
    for col in ["month", "borough", "community_board", "complaint_type"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    if "avg_response_hours" not in df.columns and "median_response_hours" in df.columns:
        df["avg_response_hours"] = df["median_response_hours"]
    if "open_cases" not in df.columns:
        df["open_cases"] = 0
    if "repeat_descriptor_complaints" not in df.columns:
        df["repeat_descriptor_complaints"] = 0
    if "pct_repeat_descriptor" not in df.columns:
        df["pct_repeat_descriptor"] = (
            df["repeat_descriptor_complaints"] / df["complaints"].clip(lower=1)
        )

    agg_dict: dict[str, tuple[str, str]] = {
        "complaints": ("complaints", "sum"),
        "open_cases": ("open_cases", "sum"),
        "repeat_descriptor_complaints": ("repeat_descriptor_complaints", "sum"),
        "avg_response_hours": ("avg_response_hours", "mean"),
        "pct_repeat_descriptor": ("pct_repeat_descriptor", "mean"),
    }
    if "complaints_per_1000" in df.columns:
        agg_dict["complaints_per_1000"] = ("complaints_per_1000", "mean")
    if "repeat_descriptor_per_1000" in df.columns:
        agg_dict["repeat_descriptor_per_1000"] = ("repeat_descriptor_per_1000", "mean")
    if "population" in df.columns:
        agg_dict["population"] = ("population", "first")
    if "median_income" in df.columns:
        agg_dict["median_income"] = ("median_income", "first")

    agg = df.groupby(["month", "borough", "community_board", "complaint_type"], as_index=False).agg(**agg_dict)
    out = PROCESSED_DIR / "neighborhood_month_agg.parquet"
    agg.to_parquet(out, index=False)
    print(f"Saved dashboard dataset to {out}")
    print(agg.head())


if __name__ == "__main__":
    main()
