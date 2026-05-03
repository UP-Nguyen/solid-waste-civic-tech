from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from sanitation_explorer.data import build_district_series, format_metric_label


def render_summary_metrics(filtered: pd.DataFrame, available_map_months: list[str], overview: pd.DataFrame) -> None:
    metric_cols = st.columns(4)
    metric_cols[0].metric("Months in view", f"{len(available_map_months)}")
    metric_cols[1].metric("District-month rows", f"{len(overview):,}")
    metric_cols[2].metric("Complaints", f"{int(filtered['complaints'].sum()):,}")
    metric_cols[3].metric(
        "Repeat share",
        f"{(filtered['repeat_descriptor_complaints'].sum() / max(filtered['complaints'].sum(), 1)) * 100:.1f}%",
    )


def render_ranking(month_slice: pd.DataFrame, metric: str, selected_map_month: str) -> None:
    st.subheader("Highest-Burden Districts")
    ranking = month_slice.sort_values(metric, ascending=False).head(12).copy()
    ranking["label"] = ranking["community_board"] + " (" + ranking["borough"] + ")"
    fig_rank = px.bar(
        ranking.sort_values(metric, ascending=True),
        x=metric,
        y="label",
        orientation="h",
        color="borough",
        labels={"label": "Community district", metric: format_metric_label(metric)},
        title=f"Top districts in {selected_map_month}",
    )
    fig_rank.update_layout(height=640, margin=dict(l=0, r=0, t=48, b=0), showlegend=False)
    st.plotly_chart(fig_rank, width="stretch")


def render_trends(overview: pd.DataFrame, type_trend: pd.DataFrame) -> None:
    trend_col, persistence_col = st.columns(2)

    with trend_col:
        st.subheader("Citywide Trend")
        city_trend = (
            overview.groupby("month", as_index=False)
            .agg(
                complaints=("complaints", "sum"),
                open_cases=("open_cases", "sum"),
                repeat_descriptor_complaints=("repeat_descriptor_complaints", "sum"),
                avg_response_hours=("avg_response_hours", "mean"),
            )
            .sort_values("month")
        )
        fig_city = px.line(
            city_trend,
            x="month",
            y=["complaints", "repeat_descriptor_complaints", "open_cases"],
            markers=True,
            title="Monthly sanitation burden across filtered districts",
            labels={"value": "Count", "variable": "Metric"},
        )
        fig_city.update_layout(height=360, margin=dict(l=0, r=0, t=48, b=0))
        st.plotly_chart(fig_city, width="stretch")

    with persistence_col:
        st.subheader("Complaint Mix Over Time")
        fig_mix = px.area(
            type_trend,
            x="month",
            y="complaints",
            color="complaint_type",
            title="How complaint types shift month to month",
        )
        fig_mix.update_layout(height=360, margin=dict(l=0, r=0, t=48, b=0))
        st.plotly_chart(fig_mix, width="stretch")


def render_drilldown(filtered: pd.DataFrame, overview: pd.DataFrame) -> None:
    st.subheader("District Drilldown")
    district_options = sorted(
        overview["community_board"].dropna().unique(),
        key=lambda value: (value.split(" ", 1)[-1], value),
    )
    selected_district = st.selectbox("Community district", district_options, index=0)

    district_series = build_district_series(filtered, selected_district)
    district_total = (
        overview[overview["community_board"] == selected_district]
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

    drill_col1, drill_col2 = st.columns([1.4, 1.0])
    with drill_col1:
        fig_district = px.line(
            district_total,
            x="month",
            y=["complaints", "repeat_descriptor_complaints", "open_cases"],
            markers=True,
            title=f"{selected_district}: trend over time",
            labels={"value": "Count", "variable": "Metric"},
        )
        fig_district.update_layout(height=380, margin=dict(l=0, r=0, t=48, b=0))
        st.plotly_chart(fig_district, width="stretch")

    with drill_col2:
        latest_district = district_total[district_total["month"] == district_total["month"].max()]
        if not latest_district.empty:
            row = latest_district.iloc[0]
            st.metric("Latest month", str(row["month"]))
            st.metric("Complaints", f"{int(row['complaints']):,}")
            st.metric("Repeat complaints", f"{int(row['repeat_descriptor_complaints']):,}")
            st.metric("Average response hours", f"{row['avg_response_hours']:.1f}")
            st.metric("Repeat share", f"{row['pct_repeat_descriptor'] * 100:.1f}%")

    type_breakdown = (
        district_series.groupby("complaint_type", as_index=False)
        .agg(
            complaints=("complaints", "sum"),
            repeat_descriptor_complaints=("repeat_descriptor_complaints", "sum"),
            avg_response_hours=("avg_response_hours", "mean"),
        )
        .sort_values("complaints", ascending=False)
    )
    fig_breakdown = px.bar(
        type_breakdown,
        x="complaint_type",
        y="complaints",
        color="repeat_descriptor_complaints",
        title=f"{selected_district}: complaint type breakdown",
        labels={
            "complaint_type": "Complaint type",
            "complaints": "Complaints",
            "repeat_descriptor_complaints": "Repeat complaints",
        },
    )
    fig_breakdown.update_layout(height=360, margin=dict(l=0, r=0, t=48, b=0))
    st.plotly_chart(fig_breakdown, width="stretch")

    with st.expander("Show underlying district table"):
        district_table = district_series.sort_values(["month", "complaint_type"]).reset_index(drop=True)
        st.dataframe(district_table, width="stretch")
