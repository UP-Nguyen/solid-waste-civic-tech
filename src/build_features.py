"""Consolidate cleaned 311 data into a dashboard-ready monthly feature table."""

import pandas as pd

from config import RAW_311_FILE, FEATURE_FILE

#Flag repeated complaint descriptors within the same geography-month.
def add_repeat_flags(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    key_cols = ["month", "borough", "community_board", "complaint_type", "descriptor"]
    counts = (
        df.groupby(key_cols, dropna=False)
        .size()
        .rename("descriptor_repeat_count")
        .reset_index()
    )
    df = df.merge(counts, on=key_cols, how="left")
    df["is_repeat_descriptor"] = df["descriptor_repeat_count"].fillna(0).ge(2)
    return df


def build_monthly_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_repeat_flags(df)

    grouped = (
        df.groupby(["month", "borough", "community_board", "complaint_type"], dropna=False)
        .agg(
            complaints=("unique_key", "count"),
            median_response_hours=("response_hours", "median"),
            avg_response_hours=("response_hours", "mean"),
            open_cases=("status", lambda s: s.astype(str).str.contains("Open", case=False, na=False).sum()),
            repeat_descriptor_complaints=("is_repeat_descriptor", "sum"),
            unique_descriptors=("descriptor", pd.Series.nunique),
        )
        .reset_index()
    )

    grouped["pct_repeat_descriptor"] = (
        grouped["repeat_descriptor_complaints"] / grouped["complaints"]
    )
    return grouped.sort_values(["month", "borough", "community_board", "complaint_type"]).reset_index(drop=True)


def main() -> None:
    df = pd.read_parquet(RAW_311_FILE)
    features = build_monthly_features(df)
    features.to_parquet(FEATURE_FILE, index=False)
    print(f"Saved feature table to {FEATURE_FILE}")
    print(features.head())


if __name__ == "__main__":
    main()
