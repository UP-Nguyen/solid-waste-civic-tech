from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from viz_utils import add_BoroCD, load_community_district_geojson, load_main_dataset  # noqa: E402


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> pd.DataFrame:
    df = load_main_dataset(prefer_merged=True).copy()
    df["month"] = df["month"].astype(str)
    df["borough"] = df["borough"].astype(str)
    df["community_board"] = df["community_board"].astype(str)
    df["complaint_type"] = df["complaint_type"].astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_geojson() -> dict:
    return load_community_district_geojson()


def month_sort_key(value: str) -> pd.Timestamp:
    return pd.to_datetime(value, format="%Y-%m", errors="coerce")


def build_map_frame(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "complaints": ("complaints", "sum"),
        "open_cases": ("open_cases", "sum"),
        "repeat_descriptor_complaints": ("repeat_descriptor_complaints", "sum"),
        "median_response_hours": ("median_response_hours", "median"),
        "avg_response_hours": ("avg_response_hours", "mean"),
        "pct_repeat_descriptor": ("pct_repeat_descriptor", "mean"),
    }
    if "complaints_per_1000" in df.columns:
        aggregations["complaints_per_1000"] = ("complaints_per_1000", "mean")
    if "repeat_descriptor_per_1000" in df.columns:
        aggregations["repeat_descriptor_per_1000"] = ("repeat_descriptor_per_1000", "mean")

    grouped = df.groupby(["month", "borough", "community_board"], as_index=False).agg(**aggregations)
    grouped = add_BoroCD(grouped)
    grouped = grouped[grouped["BoroCD"].notna()].copy()
    grouped["BoroCD"] = grouped["BoroCD"].astype(int)
    grouped["repeat_share_pct"] = grouped["pct_repeat_descriptor"].fillna(0) * 100
    return grouped.sort_values(["month", metric], ascending=[True, False])


def build_overview(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["month", "borough", "community_board"], as_index=False)
        .agg(
            complaints=("complaints", "sum"),
            open_cases=("open_cases", "sum"),
            repeat_descriptor_complaints=("repeat_descriptor_complaints", "sum"),
            avg_response_hours=("avg_response_hours", "mean"),
            pct_repeat_descriptor=("pct_repeat_descriptor", "mean"),
        )
        .sort_values("month")
    )


def build_type_trend(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["month", "complaint_type"], as_index=False)
        .agg(
            complaints=("complaints", "sum"),
            repeat_descriptor_complaints=("repeat_descriptor_complaints", "sum"),
        )
        .sort_values(["month", "complaint_type"])
    )


def build_district_series(df: pd.DataFrame, district: str) -> pd.DataFrame:
    district_df = df[df["community_board"] == district].copy()
    return (
        district_df.groupby(["month", "complaint_type"], as_index=False)
        .agg(
            complaints=("complaints", "sum"),
            open_cases=("open_cases", "sum"),
            repeat_descriptor_complaints=("repeat_descriptor_complaints", "sum"),
            avg_response_hours=("avg_response_hours", "mean"),
        )
        .sort_values(["month", "complaint_type"])
    )


def format_metric_label(metric: str) -> str:
    labels = {
        "complaints": "Complaints",
        "open_cases": "Open cases",
        "repeat_descriptor_complaints": "Repeat complaints",
        "median_response_hours": "Median response hours",
        "avg_response_hours": "Average response hours",
        "pct_repeat_descriptor": "Repeat share",
        "complaints_per_1000": "Complaints per 1,000",
        "repeat_descriptor_per_1000": "Repeat complaints per 1,000",
    }
    return labels.get(metric, metric.replace("_", " ").title())

