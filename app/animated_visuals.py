from __future__ import annotations

from pathlib import Path
import sys

import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from viz_utils import (  # noqa: E402
    COMMUNITY_DISTRICT_GEOJSON,
    load_community_district_geojson,
    load_main_dataset,
    summarize_for_choropleth,
    summarize_for_ranking,
    summarize_for_scatter,
)

st.set_page_config(page_title="NYC Sanitation Animated Visuals", layout="wide")
st.title("NYC Sanitation Animated Visuals")
st.caption(
    "Three animation-ready views for the v1 solid waste civic tech prototype. "
    "These charts are designed for month-by-month change over time."
)

try:
    df = load_main_dataset(prefer_merged=True)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

st.sidebar.header("Filters")
complaint_options = sorted(df["complaint_type"].dropna().astype(str).unique()) if "complaint_type" in df.columns else []
selected_types = st.sidebar.multiselect(
    "Complaint types",
    complaint_options,
    default=complaint_options,
)
borough_options = sorted(df["borough"].dropna().astype(str).unique()) if "borough" in df.columns else []
selected_boroughs = st.sidebar.multiselect(
    "Boroughs",
    borough_options,
    default=borough_options,
)

filtered = df.copy()
if selected_types and "complaint_type" in filtered.columns:
    filtered = filtered[filtered["complaint_type"].astype(str).isin(selected_types)]
if selected_boroughs and "borough" in filtered.columns:
    filtered = filtered[filtered["borough"].astype(str).isin(selected_boroughs)]

if filtered.empty:
    st.warning("No data remains after filtering.")
    st.stop()

# 1. Animated choropleth
st.subheader("1) Animated choropleth of complaints per 1,000 residents")
try:
    geojson = load_community_district_geojson()
    choro_df = summarize_for_choropleth(filtered)
    metric_col = "complaints_per_1000" if "complaints_per_1000" in choro_df.columns else "complaints"
    hover_data = {
        "community_board": True,
        "borough": True,
        "complaints": True,
        metric_col: ':.2f' if metric_col == 'complaints_per_1000' else True,
        "boro_cd": False,
    }
    fig_choro = px.choropleth_mapbox(
        choro_df,
        geojson=geojson,
        locations="boro_cd",
        featureidkey="properties.boro_cd",
        color=metric_col,
        animation_frame="month",
        hover_name="community_board",
        hover_data=hover_data,
        mapbox_style="carto-positron",
        center={"lat": 40.7128, "lon": -74.0060},
        zoom=8.8,
        opacity=0.75,
        color_continuous_scale="YlOrRd",
        title="Community district sanitation burden over time",
    )
    fig_choro.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=650)
    st.plotly_chart(fig_choro, use_container_width=True)
except FileNotFoundError:
    st.info(
        "To render the choropleth, add a community district geojson file at "
        f"`{COMMUNITY_DISTRICT_GEOJSON}` with a `properties.boro_cd` field."
    )
except Exception as e:
    st.error(f"Could not build choropleth: {e}")

# 2. Animated scatterplot
st.subheader("2) Animated scatterplot of complaints per capita vs median income")
try:
    scatter_df = summarize_for_scatter(filtered)
    size_col = "population" if "population" in scatter_df.columns else None
    color_col = "borough" if "borough" in scatter_df.columns else None
    hover_data = {
        "community_board": True,
        "complaints": True,
        "complaints_per_1000": ':.2f',
    }
    if "pct_repeat_descriptor" in scatter_df.columns:
        hover_data["pct_repeat_descriptor"] = ':.2%'

    fig_scatter = px.scatter(
        scatter_df,
        x="median_income",
        y="complaints_per_1000",
        animation_frame="month",
        animation_group="community_board",
        size=size_col,
        color=color_col,
        hover_name="community_board",
        hover_data=hover_data,
        title="Complaints per 1,000 residents vs. median income",
        labels={
            "median_income": "Median household income",
            "complaints_per_1000": "Complaints per 1,000 residents",
        },
        height=650,
    )
    fig_scatter.update_traces(marker=dict(sizemode="area", sizemin=6))
    fig_scatter.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_scatter, use_container_width=True)
except Exception as e:
    st.info(str(e))

# 3. Animated ranking chart
st.subheader("3) Animated ranking chart of top community districts by repeat complaint burden")
try:
    ranking_df = summarize_for_ranking(filtered)
    top_n = st.slider("Top N districts per month", min_value=5, max_value=20, value=10)
    metric_col = "repeat_burden_per_1000"
    plot_df = ranking_df[ranking_df["rank_within_month"] <= top_n].copy()
    plot_df["community_board_label"] = (
        plot_df["community_board"].astype(str) + " (" + plot_df["borough"].astype(str) + ")"
    )
    fig_rank = px.bar(
        plot_df,
        x=metric_col,
        y="community_board_label",
        color="borough",
        animation_frame="month",
        orientation="h",
        hover_name="community_board",
        hover_data={
            "repeat_descriptor_complaints": True,
            "complaints": True,
            "repeat_burden_share": ':.2%',
            metric_col: ':.2f',
            "community_board_label": False,
        },
        title="Top districts by repeat complaint burden over time",
        labels={
            metric_col: "Repeat complaint burden per 1,000" if "population" in plot_df.columns else "Repeat complaint burden",
            "community_board_label": "Community district",
        },
        height=650,
    )
    fig_rank.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_rank, use_container_width=True)
except Exception as e:
    st.error(f"Could not build ranking chart: {e}")
