from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "data" / "processed" / "neighborhood_month_with_demo.parquet"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "bayes_model_input.parquet"


def main() -> None:
    print(f"Reading from: {INPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE).copy()

    group_cols = ["month", "borough", "community_board"]

    agg_dict = {
        "complaints": ("complaints", "sum"),
        "population": ("population", "mean"),
        "median_income": ("median_income", "mean"),
        "poverty_rate": ("poverty_rate", "mean"),
    }

    for col in ["pct_black", "pct_hispanic", "pct_white", "pct_asian"]:
        if col in df.columns:
            agg_dict[col] = (col, "mean")

    model_df = df.groupby(group_cols, as_index=False).agg(**agg_dict)

    numeric_cols = [c for c in model_df.columns if c not in group_cols]
    for col in numeric_cols:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

    model_df["month_dt"] = pd.to_datetime(model_df["month"], format="%Y-%m", errors="coerce")
    model_df = model_df.sort_values(["community_board", "month_dt"]).reset_index(drop=True)

    model_df["lag_complaints"] = model_df.groupby("community_board")["complaints"].shift(1)
    model_df["month_num"] = model_df["month_dt"].dt.month
    model_df["log_population"] = np.log(model_df["population"].clip(lower=1))
    model_df["median_income_k"] = model_df["median_income"] / 1000.0
    model_df["lag_complaints_log1p"] = np.log1p(model_df["lag_complaints"])

    model_df = model_df.dropna(
        subset=["complaints", "population", "median_income_k", "lag_complaints_log1p", "month_num"]
    ).copy()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    model_df.to_parquet(OUTPUT_FILE, index=False)

    print(f"Saved modeling dataset to {OUTPUT_FILE}")
    print(f"Rows: {len(model_df)}")
    print(model_df.head())


if __name__ == "__main__":
    main()