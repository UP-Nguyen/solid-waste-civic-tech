
from __future__ import annotations

import copy
import sys
from pathlib import Path

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
BAYES_PRED_FILE = PROCESSED_DIR / "bayes_predictions.parquet"

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

DEMOGRAPHIC_LABELS = {
    "median_income": "Median household income",
    "poverty_rate": "Poverty rate",
    "pct_black": "% Black",
    "pct_hispanic": "% Hispanic/Latino",
    "pct_white": "% White",
    "pct_asian": "% Asian",
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
    raise FileNotFoundError("No processed dataset found.")


@st.cache_data(show_spinner="Loading Bayesian predictions...")
def load_bayes_predictions() -> pd.DataFrame:
    if not BAYES_PRED_FILE.exists():
        raise FileNotFoundError("Bayesian predictions not found. Run src/predict_bayesian_model.py first.")
    df = pd.read_parquet(BAYES_PRED_FILE)
    for col in ["month", "borough", "community_board"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


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
        grouped["pct_repeat_descriptor"] = grouped["repeat_descriptor_complaints"] / grouped["complaints"].clip(lower=1)

    agg_dict = {
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

    demo_cols = [c for c in DEMOGRAPHIC_LABELS if c in grouped.columns]
    if demo_cols:
        demo_df = grouped.groupby(["month", "borough", "community_board"], as_index=False)[demo_cols].mean()
        by_district = by_district.merge(demo_df, on=["month", "borough", "community_board"], how="left")
    if "population" in grouped.columns:
        pop_df = grouped.groupby(["month", "borough", "community_board"], as_index=False)["population"].mean()
        by_district = by_district.merge(pop_df, on=["month", "borough", "community_board"], how="left")

    by_district = add_boro_cd(by_district)
    by_district = by_district[by_district["boro_cd"].notna()].copy()
    by_district["boro_cd"] = by_district["boro_cd"].astype(int)
    by_district["repeat_share_pct"] = by_district["pct_repeat_descriptor"].fillna(0) * 100

    type_trend = finest.groupby(["month", "complaint_type"], as_index=False).agg(
        complaints=("complaints", "sum"),
        repeat_descriptor_complaints=("repeat_descriptor_complaints", "sum"),
    )
    return {"finest": finest, "overview": by_district.sort_values("month"), "type_trend": type_trend}


def filter_data(df: pd.DataFrame, sel_types: list[str], sel_boroughs: list[str], start: str, end: str) -> pd.DataFrame:
    out = df.copy()
    if sel_types:
        out = out[out["complaint_type"].isin(sel_types)]
    if sel_boroughs:
        out = out[out["borough"].isin(sel_boroughs)]
    return out[(out["month"] >= start) & (out["month"] <= end)]


@st.cache_data(show_spinner="Rendering burden map...")
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


def build_bayes_map_figure(bayes_df: pd.DataFrame, geojson: dict, focused_month: str, map_metric: str):
    month_slice = bayes_df[bayes_df["month"] == focused_month].copy()
    if month_slice.empty:
        return None
    label_map = {
        "predicted_complaints": "Predicted burden (posterior median)",
        "prediction_interval_width": "Prediction uncertainty (p90 - p10)",
    }
    fig = px.choropleth_mapbox(
        month_slice,
        geojson=geojson,
        locations="boro_cd",
        featureidkey="properties.BoroCD",
        color=map_metric,
        color_continuous_scale="Magma",
        hover_name="community_board",
        hover_data={
            "complaints": True,
            "predicted_complaints": ":.1f",
            "predicted_complaints_p10": ":.1f",
            "predicted_complaints_p90": ":.1f",
            "prediction_interval_width": ":.1f",
            "boro_cd": False,
        },
        mapbox_style="carto-darkmatter",
        zoom=9.3,
        center={"lat": 40.7128, "lon": -74.0060},
        opacity=0.75,
        title=f"{label_map.get(map_metric, map_metric)} — {focused_month}",
    )
    fig.update_coloraxes(
        colorbar=dict(
            thickness=14,
            len=0.72,
            y=0.5,
        )
    )
    fig.update_layout(
    height=620,
    margin=dict(l=0, r=0, t=48, b=0),
)
    return fig


def available_demo_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in DEMOGRAPHIC_LABELS if c in df.columns]


def build_demo_overview(filtered_overview: pd.DataFrame, demo_col: str) -> pd.DataFrame:
    if demo_col not in filtered_overview.columns:
        return pd.DataFrame()
    cols = ["month", "borough", "community_board", "boro_cd", "complaints", "complaints_per_1000", demo_col]
    if "population" in filtered_overview.columns:
        cols.append("population")
    demo_df = filtered_overview[[c for c in cols if c in filtered_overview.columns]].copy()
    return demo_df.dropna(subset=[demo_col])


def build_demo_map_figure(demo_df: pd.DataFrame, geojson: dict, focused_month: str, demo_col: str):
    month_slice = demo_df[demo_df["month"] == focused_month].copy()
    if month_slice.empty:
        return None
    fig = px.choropleth_mapbox(
        month_slice,
        geojson=geojson,
        locations="boro_cd",
        featureidkey="properties.BoroCD",
        color=demo_col,
        color_continuous_scale="Viridis",
        hover_name="community_board",
        hover_data={demo_col: True, "complaints": True, "complaints_per_1000": ":.2f", "boro_cd": False},
        mapbox_style="carto-darkmatter",
        zoom=9.3,
        center={"lat": 40.7128, "lon": -74.0060},
        opacity=0.75,
        title=f"{DEMOGRAPHIC_LABELS.get(demo_col, demo_col)} — {focused_month}",
    )
    fig.update_layout(
    height=620,
    margin=dict(l=0, r=0, t=48, b=0),
)
    return fig


def build_demo_scatter(demo_df: pd.DataFrame, demo_col: str):
    if "complaints_per_1000" not in demo_df.columns:
        return None
    size_col = "population" if "population" in demo_df.columns else None
    fig = px.scatter(
        demo_df.sort_values("month"),
        x=demo_col, y="complaints_per_1000", animation_frame="month", color="borough", size=size_col,
        hover_name="community_board",
        title=f"Complaints per 1,000 vs {DEMOGRAPHIC_LABELS.get(demo_col, demo_col)}",
        labels={demo_col: DEMOGRAPHIC_LABELS.get(demo_col, demo_col), "complaints_per_1000": "Complaints per 1,000 residents"},
    )
    fig.update_layout(
    height=620,
    margin=dict(l=0, r=0, t=48, b=0),
)
    return fig


def make_overlap_table(filtered_overview: pd.DataFrame, vulnerability_col: str, focused_month: str) -> pd.DataFrame:
    month_df = filtered_overview[filtered_overview["month"] == focused_month].copy()
    cols = ["borough", "community_board", "complaints_per_1000", vulnerability_col]
    extra = [c for c in ["median_income", "poverty_rate", "pct_black", "pct_hispanic", "pct_white", "pct_asian"] if c in month_df.columns]
    month_df = month_df[list(dict.fromkeys(cols + extra))].dropna(subset=["complaints_per_1000", vulnerability_col]).copy()
    if month_df.empty:
        return month_df
    burden_cut = month_df["complaints_per_1000"].quantile(0.75)
    if vulnerability_col == "median_income":
        vuln_cut = month_df["median_income"].quantile(0.25)
        month_df["overlap_flag"] = np.where(
            (month_df["complaints_per_1000"] >= burden_cut) & (month_df["median_income"] <= vuln_cut),
            "High burden + low income", "Other"
        )
    else:
        vuln_cut = month_df[vulnerability_col].quantile(0.75)
        month_df["overlap_flag"] = np.where(
            (month_df["complaints_per_1000"] >= burden_cut) & (month_df[vulnerability_col] >= vuln_cut),
            "High burden + high vulnerability", "Other"
        )
    return month_df.sort_values(["overlap_flag", "complaints_per_1000"], ascending=[False, False])


def main() -> None:
    st.set_page_config(page_title="NYC Sanitation Burden Explorer", layout="wide")
    st.title("NYC Sanitation Burden Explorer")
    st.caption("Explorer for sanitation-related 311 burden across NYC community districts, with demographic context and Bayesian prediction.")

    try:
        df = load_dashboard_data()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    try:
        bayes_df = load_bayes_predictions()
    except FileNotFoundError:
        bayes_df = None

    frames = prepare_dashboard_frames(df)
    finest = frames["finest"]
    overview = frames["overview"]
    type_trend = frames["type_trend"]

    has_demo = "complaints_per_1000" in overview.columns
    demo_cols = available_demo_cols(overview)
    months = sorted(finest["month"].dropna().unique(), key=month_sort_key)
    complaint_opts = sorted(finest["complaint_type"].dropna().unique())
    borough_opts = sorted(finest["borough"].dropna().unique())

    st.sidebar.header("Filters")
    sel_types = st.sidebar.multiselect("Complaint types", complaint_opts, default=complaint_opts)
    sel_boroughs = st.sidebar.multiselect("Boroughs", borough_opts, default=borough_opts)
    sel_months = st.sidebar.select_slider("Month range", options=months, value=(months[0], months[-1]))
    sel_focused = st.sidebar.selectbox("Focused month (maps)", options=months, index=len(months) - 1)

    metric_opts = ["complaints", "repeat_descriptor_complaints", "open_cases", "avg_response_hours"]
    if has_demo:
        metric_opts.insert(1, "complaints_per_1000")
        if "repeat_descriptor_per_1000" in overview.columns:
            metric_opts.insert(3, "repeat_descriptor_per_1000")
    metric = st.sidebar.radio("Observed burden map metric", metric_opts, index=0, format_func=format_metric_label)

    filtered_source = filter_data(df, sel_types, sel_boroughs, *sel_months)
    if filtered_source.empty:
        st.warning("No data for the selected range.")
        st.stop()

    filtered_frames = prepare_dashboard_frames(filtered_source)
    filtered_range = filtered_frames["finest"]
    filtered_overview = filtered_frames["overview"]
    filtered_type = filtered_frames["type_trend"]

    bayes_filtered = None
    if bayes_df is not None:
        bayes_filtered = bayes_df.copy()
        if sel_boroughs:
            bayes_filtered = bayes_filtered[bayes_filtered["borough"].isin(sel_boroughs)]
        bayes_filtered = bayes_filtered[(bayes_filtered["month"] >= sel_months[0]) & (bayes_filtered["month"] <= sel_months[1])]
        if not bayes_filtered.empty:
            bayes_filtered = add_boro_cd(bayes_filtered)
            bayes_filtered = bayes_filtered[bayes_filtered["boro_cd"].notna()].copy()
            bayes_filtered["boro_cd"] = bayes_filtered["boro_cd"].astype(int)

    global_min = float(filtered_overview[metric].min()) if not filtered_overview.empty else 0
    global_max = float(filtered_overview[metric].max()) if not filtered_overview.empty else 1

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Focused month", sel_focused)
    c2.metric("Range", f"{sel_months[0]} → {sel_months[1]}")
    total_complaints = int(filtered_range["complaints"].sum())
    total_repeat = int(filtered_range["repeat_descriptor_complaints"].sum())
    c3.metric("Complaints (range)", f"{total_complaints:,}")
    c4.metric("Repeat share", f"{(total_repeat / max(total_complaints, 1)) * 100:.1f}%")

    st.header("Spatial Overview")
    st.caption("The observed burden map comes first, followed by the Bayesian model-based view for the same focused month.")

    st.subheader("Observed Burden Map")
    try:
        geojson = load_community_district_geojson()
        map_html = build_single_month_map(geojson, filtered_overview, metric, sel_focused, global_min, global_max)
        components.html(map_html, height=700, scrolling=False)
    except FileNotFoundError as e:
        st.info(str(e))

    st.subheader("Bayesian Map")
    if bayes_filtered is None or bayes_filtered.empty:
        st.info("Run src/predict_bayesian_model.py to enable Bayesian maps.")
    else:
        bayes_map_metric = st.radio(
            "Bayesian map view",
            ["predicted_complaints", "prediction_interval_width"],
            horizontal=True,
            format_func=lambda x: "Predicted burden" if x == "predicted_complaints" else "Prediction uncertainty",
            key="bayes_map_stacked_full",
        )
        st.caption("Model-based view for the same focused month. Toggle between predicted burden and prediction uncertainty.")
        try:
            geojson = load_community_district_geojson()
            fig_bayes_map = build_bayes_map_figure(bayes_filtered, geojson, sel_focused, bayes_map_metric)
            if fig_bayes_map is not None:
                st.plotly_chart(fig_bayes_map, use_container_width=True)
        except FileNotFoundError as e:
            st.info(str(e))

    rank1, rank2 = st.columns(2)
    with rank1:
        st.subheader("Highest Observed Burden")
        month_slice = filtered_overview[filtered_overview["month"] == sel_focused].copy()
        if month_slice.empty:
            month_slice = filtered_overview[filtered_overview["month"] == filtered_overview["month"].max()].copy()
        ranking = month_slice.sort_values(metric, ascending=False).head(12).copy()
        ranking["label"] = ranking["community_board"] + " (" + ranking["borough"] + ")"
        fig_rank = px.bar(
            ranking.sort_values(metric, ascending=True),
            x=metric, y="label", orientation="h", color="borough",
            labels={"label": "Community district", metric: format_metric_label(metric)},
            title=f"Observed burden — {sel_focused}",
        )
        fig_rank.update_layout(height=480, margin=dict(l=0, r=0, t=48, b=0), showlegend=False)
        st.plotly_chart(fig_rank, use_container_width=True)

    with rank2:
        st.subheader("Highest Predicted Burden")
        if bayes_filtered is None or bayes_filtered.empty:
            st.info("No Bayesian predictions available for the selected filters.")
        else:
            latest_pred = bayes_filtered[bayes_filtered["month"] == sel_focused].copy()
            if latest_pred.empty:
                latest_pred = bayes_filtered[bayes_filtered["month"] == bayes_filtered["month"].max()].copy()
            latest_pred = latest_pred.sort_values("predicted_complaints", ascending=False).head(12)
            fig_top_pred = px.bar(
                latest_pred.sort_values("predicted_complaints", ascending=True),
                x="predicted_complaints", y="community_board", orientation="h", color="borough",
                title=f"Predicted burden — {sel_focused}",
                labels={"predicted_complaints": "Predicted complaints (posterior median)", "community_board": "Community district"},
            )
            fig_top_pred.update_layout(height=480, margin=dict(l=0, r=0, t=48, b=0), showlegend=False)
            st.plotly_chart(fig_top_pred, use_container_width=True)

    st.header("Bayesian Modeling")

    if bayes_filtered is None or bayes_filtered.empty:
        st.info("Run src/predict_bayesian_model.py to enable Bayesian predictions.")
    else:
        bayes_tabs = st.tabs(["Predicted Burden and Uncertainty", "Model Fit by District"])
        with bayes_tabs[0]:
            latest_pred = bayes_filtered[bayes_filtered["month"] == sel_focused].copy()
            if latest_pred.empty:
                latest_pred = bayes_filtered[bayes_filtered["month"] == bayes_filtered["month"].max()].copy()
            b1, b2 = st.columns(2)
            with b1:
                fig_unc = px.scatter(
                    latest_pred,
                    x="predicted_complaints", y="prediction_interval_width", color="borough",
                    hover_name="community_board", size="complaints",
                    title=f"Predicted burden and uncertainty — {sel_focused}",
                    labels={"predicted_complaints": "Predicted complaints (posterior median)",
                            "prediction_interval_width": "Prediction interval width (p90 - p10)",
                            "complaints": "Observed complaints"},
                )
                fig_unc.update_layout(height=500, margin=dict(l=0, r=0, t=48, b=0))
                st.plotly_chart(fig_unc, use_container_width=True)
            with b2:
                st.markdown("#### Highest Predicted Complaint Burden")
                st.dataframe(
                    latest_pred.sort_values("predicted_complaints", ascending=False)[[
                        "month", "borough", "community_board", "complaints",
                        "predicted_complaints", "predicted_complaints_p10",
                        "predicted_complaints_p90", "prediction_interval_width"
                    ]].head(15),
                    use_container_width=True,
                )

        with bayes_tabs[1]:
            pred_dist_opts = sorted(bayes_filtered["community_board"].dropna().unique())
            pred_sel_dist = st.selectbox("Community district for model fit view", pred_dist_opts, key="bayes_dist_select")
            dist_pred = bayes_filtered[bayes_filtered["community_board"] == pred_sel_dist].copy().sort_values("month")
            m1, m2 = st.columns(2)
            with m1:
                st.markdown("#### How actual complaints compare with the model's expected trend")
                fig_actual_pred = px.line(
                    dist_pred, x="month", y=["complaints", "predicted_complaints"], markers=True,
                    title=f"{pred_sel_dist}: model fit over time",
                    labels={"value": "Complaints", "variable": "Series"},
                )
                fig_actual_pred.update_layout(height=420, margin=dict(l=0, r=0, t=48, b=0))
                st.plotly_chart(fig_actual_pred, use_container_width=True)
                st.caption("The lighter line shows reported complaints. The darker line shows the model's expected complaint level based on past patterns, seasonality, and neighborhood context.")
            with m2:
                st.markdown("#### Prediction interval")
                fig_band = px.line(
                    dist_pred, x="month", y=["predicted_complaints_p10", "predicted_complaints_p90"],
                    title=f"{pred_sel_dist}: expected range",
                    labels={"value": "Predicted complaints", "variable": "Interval bound"},
                )
                fig_band.update_layout(height=420, margin=dict(l=0, r=0, t=48, b=0))
                st.plotly_chart(fig_band, use_container_width=True)

    st.header("Time Trends")
    t1, t2 = st.columns(2)
    with t1:
        st.subheader("Citywide Trend")
        city = filtered_overview.groupby("month", as_index=False).agg(
            complaints=("complaints", "sum"),
            open_cases=("open_cases", "sum"),
            repeat_descriptor_complaints=("repeat_descriptor_complaints", "sum"),
        )
        fig_city = px.line(city, x="month", y=["complaints", "repeat_descriptor_complaints", "open_cases"], markers=True,
                           title=f"Monthly burden ({sel_months[0]} to {sel_months[1]})",
                           labels={"value": "Count", "variable": "Metric"})
        fig_city.update_layout(height=360, margin=dict(l=0, r=0, t=48, b=0))
        st.plotly_chart(fig_city, use_container_width=True)

    with t2:
        st.subheader("Complaint Mix Over Time")
        fig_mix = px.area(filtered_type, x="month", y="complaints", color="complaint_type",
                          title=f"Complaint types ({sel_months[0]} to {sel_months[1]})")
        fig_mix.update_layout(height=360, margin=dict(l=0, r=0, t=48, b=0))
        st.plotly_chart(fig_mix, use_container_width=True)


if __name__ == "__main__":
    main()
