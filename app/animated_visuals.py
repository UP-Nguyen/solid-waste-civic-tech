from __future__ import annotations

import streamlit as st

from sanitation_explorer.data import (
    build_map_frame,
    build_overview,
    build_type_trend,
    format_metric_label,
    load_dashboard_data,
    load_geojson,
    month_sort_key,
)
from sanitation_explorer.maps import render_map
from sanitation_explorer.sections import (
    render_drilldown,
    render_ranking,
    render_summary_metrics,
    render_trends,
)


st.set_page_config(page_title="NYC Sanitation Burden Explorer", layout="wide")
st.title("NYC Sanitation Burden Explorer")
st.caption(
    "Akbar's interactive tool for exploring sanitation-related 311 burden across NYC community districts."
)

try:
    df = load_dashboard_data()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

has_demo = "complaints_per_1000" in df.columns
months = sorted(df["month"].dropna().unique(), key=month_sort_key)
complaint_options = sorted(df["complaint_type"].dropna().unique())
borough_options = sorted(df["borough"].dropna().unique())

st.sidebar.header("Filters")
selected_types = st.sidebar.multiselect("Complaint types", complaint_options, default=complaint_options)
selected_boroughs = st.sidebar.multiselect("Boroughs", borough_options, default=borough_options)
selected_months = st.sidebar.select_slider("Month range", options=months, value=(months[0], months[-1]))

metric_options = ["complaints", "repeat_descriptor_complaints", "open_cases", "avg_response_hours"]
if has_demo:
    metric_options.insert(1, "complaints_per_1000")
    metric_options.insert(3, "repeat_descriptor_per_1000")
metric = st.sidebar.radio("Map metric", metric_options, index=0, format_func=format_metric_label)

map_mode = st.sidebar.radio("Map style", ["2D", "3D"], index=0)
show_animation = st.sidebar.checkbox("Show animated map", value=True)
extrusion_scale = st.sidebar.slider("3D height scale", min_value=0.5, max_value=3.0, value=1.6, step=0.1)

filtered = df.copy()
if selected_types:
    filtered = filtered[filtered["complaint_type"].isin(selected_types)]
if selected_boroughs:
    filtered = filtered[filtered["borough"].isin(selected_boroughs)]

start_month, end_month = selected_months
filtered = filtered[(filtered["month"] >= start_month) & (filtered["month"] <= end_month)]

if filtered.empty:
    st.warning("No data remains after the current filters.")
    st.stop()

overview = build_overview(filtered)
type_trend = build_type_trend(filtered)
map_df = build_map_frame(filtered, metric)
available_map_months = sorted(map_df["month"].unique(), key=month_sort_key)
selected_map_month = st.sidebar.selectbox(
    "Focused month",
    available_map_months,
    index=len(available_map_months) - 1,
)
month_slice = map_df[map_df["month"] == selected_map_month].copy()

render_summary_metrics(filtered, available_map_months, overview)

map_col, rank_col = st.columns([1.8, 1.0])
with map_col:
    st.subheader("Burden Map")
    try:
        geojson = load_geojson()
        render_map(
            geojson=geojson,
            map_df=map_df,
            month_slice=month_slice,
            metric=metric,
            map_mode=map_mode,
            show_animation=show_animation,
            extrusion_scale=extrusion_scale,
            selected_map_month=selected_map_month,
        )
    except FileNotFoundError as exc:
        st.info(str(exc))

with rank_col:
    render_ranking(month_slice, metric, selected_map_month)

render_trends(overview, type_trend)
render_drilldown(filtered, overview)
