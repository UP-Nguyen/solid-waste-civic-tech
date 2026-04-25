from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from viz_utils import add_boro_cd, load_community_district_geojson, month_sort_key  # noqa: E402

PROCESSED_DIR = BASE_DIR / "data" / "processed"
MERGED_FILE = PROCESSED_DIR / "neighborhood_month_with_demo.parquet"
AGG_FILE = PROCESSED_DIR / "neighborhood_month_agg.parquet"
FEATURE_FILE = PROCESSED_DIR / "neighborhood_month.parquet"

METRIC_LABELS = {
    "complaints": "Complaints",
    "open_cases": "Open cases",
    "repeat_descriptor_complaints": "Repeat complaints",
    "median_response_hours": "Median response hours",
    "avg_response_hours": "Average response hours",
    "pct_repeat_descriptor": "Repeat share",
    "complaints_per_1000": "Complaints per 1,000",
    "repeat_descriptor_per_1000": "Repeat complaints per 1,000",
}


def format_metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


@st.cache_data(show_spinner="Loading data...")
def load_dashboard_data() -> pd.DataFrame:
    for path in [MERGED_FILE, AGG_FILE, FEATURE_FILE]:
        if path.exists():
            df = pd.read_parquet(path)
            for col in ["month", "borough", "community_board", "complaint_type"]:
                if col in df.columns:
                    df[col] = df[col].astype(str)
            return df
    raise FileNotFoundError(
        "No processed dataset found. Run src/extract_311.py, src/build_features.py, and optional merge/build scripts first."
    )


@st.cache_data(show_spinner="Preparing dashboard frames...")
def prepare_dashboard_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    grouped = df.copy()
    if "avg_response_hours" not in grouped.columns and "median_response_hours" in grouped.columns:
        grouped["avg_response_hours"] = grouped["median_response_hours"]
    if "open_cases" not in grouped.columns:
        grouped["open_cases"] = 0
    if "repeat_descriptor_complaints" not in grouped.columns:
        grouped["repeat_descriptor_complaints"] = 0
    if "pct_repeat_descriptor" not in grouped.columns:
        grouped["pct_repeat_descriptor"] = (
            grouped["repeat_descriptor_complaints"] / grouped["complaints"].clip(lower=1)
        )

    agg_dict: dict[str, tuple[str, str]] = {
        "complaints": ("complaints", "sum"),
        "open_cases": ("open_cases", "sum"),
        "repeat_descriptor_complaints": ("repeat_descriptor_complaints", "sum"),
        "avg_response_hours": ("avg_response_hours", "mean"),
        "pct_repeat_descriptor": ("pct_repeat_descriptor", "mean"),
    }
    if "complaints_per_1000" in grouped.columns:
        agg_dict["complaints_per_1000"] = ("complaints_per_1000", "mean")
    if "repeat_descriptor_per_1000" in grouped.columns:
        agg_dict["repeat_descriptor_per_1000"] = ("repeat_descriptor_per_1000", "mean")

    finest = grouped.groupby(["month", "borough", "community_board", "complaint_type"], as_index=False).agg(**agg_dict)
    by_district = finest.groupby(["month", "borough", "community_board"], as_index=False).agg(**agg_dict)
    by_district = add_boro_cd(by_district)
    by_district = by_district[by_district["boro_cd"].notna()].copy()
    by_district["boro_cd"] = by_district["boro_cd"].astype(int)
    by_district["repeat_share_pct"] = by_district["pct_repeat_descriptor"].fillna(0) * 100

    overview = by_district.sort_values("month")
    type_trend = finest.groupby(["month", "complaint_type"], as_index=False).agg(
        complaints=("complaints", "sum"),
        repeat_descriptor_complaints=("repeat_descriptor_complaints", "sum"),
    )
    return {"finest": finest, "overview": overview, "type_trend": type_trend}


def filter_data(df: pd.DataFrame, sel_types: list[str], sel_boroughs: list[str], start: str, end: str) -> pd.DataFrame:
    out = df.copy()
    if sel_types:
        out = out[out["complaint_type"].isin(sel_types)]
    if sel_boroughs:
        out = out[out["borough"].isin(sel_boroughs)]
    return out[(out["month"] >= start) & (out["month"] <= end)]


@st.cache_data(show_spinner="Rendering map...")
def build_single_month_map(_geojson: dict, _map_df: pd.DataFrame, metric: str, month: str, global_min: float, global_max: float) -> str:
    month_data = _map_df[_map_df["month"] == month].copy()
    if month_data.empty:
        m = folium.Map(location=[40.7128, -74.0060], zoom_start=10, tiles="CartoDB dark_matter")
        return m._repr_html_()

    month_data = month_data.groupby("boro_cd", as_index=False).agg({
        metric: "sum",
        "community_board": "first",
        "borough": "first",
        "complaints": "sum",
        "open_cases": "sum",
        "avg_response_hours": "mean",
        "repeat_share_pct": "mean",
    })

    bins = list(np.linspace(global_min, global_max, 7)) if global_max > global_min else [global_min, global_max + 1]
    lookup = month_data.set_index("boro_cd").to_dict(orient="index")
    gj = copy.deepcopy(_geojson)
    for feat in gj["features"]:
        props = feat.get("properties", {})
        bcd = props.get("boro_cd", props.get("BoroCD"))
        row = lookup.get(bcd, {})
        props.update({
            "community_board": row.get("community_board", "N/A"),
            "borough": row.get("borough", "N/A"),
            "metric_value": round(float(row.get(metric, 0) or 0), 1),
            "complaints": int(row.get("complaints", 0)),
            "open_cases": int(row.get("open_cases", 0)),
            "avg_response_hours": round(float(row.get("avg_response_hours", 0) or 0), 1),
            "repeat_share_pct": round(float(row.get("repeat_share_pct", 0) or 0), 1),
        })
        feat["properties"] = props

    m = folium.Map(location=[40.7128, -74.0060], zoom_start=10, tiles="CartoDB dark_matter")
    folium.Choropleth(
        geo_data=gj,
        data=month_data,
        columns=["boro_cd", metric],
        key_on="feature.properties.BoroCD",
        fill_color="YlOrRd",
        fill_opacity=0.75,
        line_opacity=0.2,
        legend_name=f"{format_metric_label(metric)} ({month})",
        nan_fill_color="transparent",
        bins=bins,
    ).add_to(m)
    folium.GeoJson(
        gj,
        style_function=lambda x: {"fillOpacity": 0, "color": "#ffffff", "weight": 0.3},
        tooltip=folium.GeoJsonTooltip(
            fields=["community_board", "borough", "metric_value", "complaints", "open_cases", "avg_response_hours", "repeat_share_pct"],
            aliases=["District:", "Borough:", f"{format_metric_label(metric)}:", "Complaints:", "Open cases:", "Avg hrs:", "Repeat %:"],
        ),
    ).add_to(m)
    return m._repr_html_()


def main() -> None:
    st.set_page_config(page_title="NYC Sanitation Burden Explorer", layout="wide")
    st.title("NYC Sanitation Burden Explorer")
    st.caption("Explorer for sanitation-related 311 burden across NYC community districts.")

    try:
        df = load_dashboard_data()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    frames = prepare_dashboard_frames(df)
    finest = frames["finest"]
    overview = frames["overview"]
    type_trend = frames["type_trend"]

    has_demo = "complaints_per_1000" in overview.columns
    months = sorted(finest["month"].dropna().unique(), key=month_sort_key)
    complaint_opts = sorted(finest["complaint_type"].dropna().unique())
    borough_opts = sorted(finest["borough"].dropna().unique())

    st.sidebar.header("Filters")
    sel_types = st.sidebar.multiselect("Complaint types", complaint_opts, default=complaint_opts)
    sel_boroughs = st.sidebar.multiselect("Boroughs", borough_opts, default=borough_opts)
    sel_months = st.sidebar.select_slider("Month range", options=months, value=(months[0], months[-1]))
    sel_focused = st.sidebar.selectbox("Focused month (map)", options=months, index=len(months) - 1)
    metric_opts = ["complaints", "repeat_descriptor_complaints", "open_cases", "avg_response_hours"]
    if has_demo:
        metric_opts.insert(1, "complaints_per_1000")
        if "repeat_descriptor_per_1000" in overview.columns:
            metric_opts.insert(3, "repeat_descriptor_per_1000")
    metric = st.sidebar.radio("Map metric", metric_opts, index=0, format_func=format_metric_label)

    filtered_range = filter_data(finest, sel_types, sel_boroughs, *sel_months)
    if filtered_range.empty:
        st.warning("No data for the selected range.")
        st.stop()

    filtered_overview = prepare_dashboard_frames(filtered_range)["overview"]
    filtered_type = prepare_dashboard_frames(filtered_range)["type_trend"]
    global_min = float(filtered_overview[metric].min()) if not filtered_overview.empty else 0
    global_max = float(filtered_overview[metric].max()) if not filtered_overview.empty else 1

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Focused month", sel_focused)
    c2.metric("Range", f"{sel_months[0]} → {sel_months[1]}")
    total_complaints = int(filtered_range["complaints"].sum())
    total_repeat = int(filtered_range["repeat_descriptor_complaints"].sum())
    c3.metric("Complaints (range)", f"{total_complaints:,}")
    c4.metric("Repeat share", f"{(total_repeat / max(total_complaints, 1)) * 100:.1f}%")

    @st.fragment
    def render_focused_fragment() -> None:
        map_col, rank_col = st.columns([1.8, 1.0])
        with map_col:
            st.subheader(f"Burden Map — {sel_focused}")
            try:
                geojson = load_community_district_geojson()
                map_html = build_single_month_map(geojson, filtered_overview, metric, sel_focused, global_min, global_max)
                components.html(map_html, height=540, scrolling=False)
            except FileNotFoundError as e:
                st.info(str(e))

        with rank_col:
            st.subheader("Highest-Burden Districts")
            month_slice = filtered_overview[filtered_overview["month"] == sel_focused].copy()
            if month_slice.empty:
                month_slice = filtered_overview[filtered_overview["month"] == filtered_overview["month"].max()].copy()
            ranking = month_slice.sort_values(metric, ascending=False).head(12).copy()
            ranking["label"] = ranking["community_board"] + " (" + ranking["borough"] + ")"
            fig_rank = px.bar(
                ranking.sort_values(metric, ascending=True),
                x=metric,
                y="label",
                orientation="h",
                color="borough",
                labels={"label": "Community district", metric: format_metric_label(metric)},
                title=f"Top districts — {sel_focused}",
            )
            fig_rank.update_layout(height=640, margin=dict(l=0, r=0, t=48, b=0), showlegend=False)
            st.plotly_chart(fig_rank, use_container_width=True)

    render_focused_fragment()

    t1, t2 = st.columns(2)
    with t1:
        st.subheader("Citywide Trend")
        city = filtered_overview.groupby("month", as_index=False).agg(
            complaints=("complaints", "sum"),
            open_cases=("open_cases", "sum"),
            repeat_descriptor_complaints=("repeat_descriptor_complaints", "sum"),
        )
        fig_city = px.line(
            city,
            x="month",
            y=["complaints", "repeat_descriptor_complaints", "open_cases"],
            markers=True,
            title=f"Monthly burden ({sel_months[0]} to {sel_months[1]})",
            labels={"value": "Count", "variable": "Metric"},
        )
        fig_city.update_layout(height=360, margin=dict(l=0, r=0, t=48, b=0))
        st.plotly_chart(fig_city, use_container_width=True)

    with t2:
        st.subheader("Complaint Mix Over Time")
        fig_mix = px.area(
            filtered_type,
            x="month",
            y="complaints",
            color="complaint_type",
            title=f"Complaint types ({sel_months[0]} to {sel_months[1]})",
        )
        fig_mix.update_layout(height=360, margin=dict(l=0, r=0, t=48, b=0))
        st.plotly_chart(fig_mix, use_container_width=True)

    st.subheader("District Drilldown")
    dist_opts = sorted(filtered_overview["community_board"].dropna().unique())
    sel_dist = st.selectbox("Community district", dist_opts, index=0)

    dist_series = (
        filtered_range[filtered_range["community_board"] == sel_dist]
        .groupby(["month", "complaint_type"], as_index=False)
        .agg(
            complaints=("complaints", "sum"),
            open_cases=("open_cases", "sum"),
            repeat_descriptor_complaints=("repeat_descriptor_complaints", "sum"),
            avg_response_hours=("avg_response_hours", "mean"),
        )
        .sort_values(["month", "complaint_type"])
    )
    dist_total = (
        filtered_overview[filtered_overview["community_board"] == sel_dist]
        .groupby("month", as_index=False)
        .agg(
            complaints=("complaints", "sum"),
            open_cases=("open_cases", "sum"),
            repeat_descriptor_complaints=("repeat_descriptor_complaints", "sum"),
            avg_response_hours=("avg_response_hours", "mean"),
            pct_repeat_descriptor=("pct_repeat_descriptor", "mean"),
        )
        .sort_values("month")
    )

    d1, d2 = st.columns([1.4, 1.0])
    with d1:
        fig_dist = px.line(
            dist_total,
            x="month",
            y=["complaints", "repeat_descriptor_complaints", "open_cases"],
            markers=True,
            title=f"{sel_dist}: trend ({sel_months[0]} to {sel_months[1]})",
            labels={"value": "Count", "variable": "Metric"},
        )
        fig_dist.update_layout(height=380, margin=dict(l=0, r=0, t=48, b=0))
        st.plotly_chart(fig_dist, use_container_width=True)

    with d2:
        latest = dist_total[dist_total["month"] == sel_focused]
        if latest.empty and not dist_total.empty:
            latest = dist_total[dist_total["month"] == dist_total["month"].max()]
        if not latest.empty:
            row = latest.iloc[0]
            st.metric("Focused month", str(row["month"]))
            st.metric("Complaints", f"{int(row['complaints']):,}")
            st.metric("Repeat complaints", f"{int(row['repeat_descriptor_complaints']):,}")
            st.metric("Avg response hours", f"{row['avg_response_hours']:.1f}")
            st.metric("Repeat share", f"{row['pct_repeat_descriptor'] * 100:.1f}%")

    breakdown = (
        dist_series.groupby("complaint_type", as_index=False)
        .agg(
            complaints=("complaints", "sum"),
            repeat_descriptor_complaints=("repeat_descriptor_complaints", "sum"),
            avg_response_hours=("avg_response_hours", "mean"),
        )
        .sort_values("complaints", ascending=False)
    )
    fig_breakdown = px.bar(
        breakdown,
        x="complaint_type",
        y="complaints",
        color="repeat_descriptor_complaints",
        title=f"{sel_dist}: complaint breakdown ({sel_months[0]} to {sel_months[1]})",
        labels={"complaint_type": "Type", "complaints": "Complaints", "repeat_descriptor_complaints": "Repeat"},
    )
    fig_breakdown.update_layout(height=360, margin=dict(l=0, r=0, t=48, b=0))
    st.plotly_chart(fig_breakdown, use_container_width=True)

    with st.expander("Show underlying data table"):
        st.dataframe(dist_series.sort_values(["month", "complaint_type"]).reset_index(drop=True), use_container_width=True)


if __name__ == "__main__":
    main()
