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


def available_demo_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in DEMOGRAPHIC_LABELS if c in df.columns]


def build_demo_overview(filtered_overview: pd.DataFrame, demo_col: str) -> pd.DataFrame:
    if demo_col not in filtered_overview.columns:
        return pd.DataFrame()
    cols = ["month", "borough", "community_board", "boro_cd", "complaints", "complaints_per_1000", demo_col]
    if "population" in filtered_overview.columns:
        cols.append("population")
    cols = [c for c in cols if c in filtered_overview.columns]
    demo_df = filtered_overview[cols].copy()
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
    fig.update_layout(height=540, margin=dict(l=0, r=0, t=48, b=0))
    return fig


def build_demo_scatter(demo_df: pd.DataFrame, demo_col: str):
    if "complaints_per_1000" not in demo_df.columns:
        return None
    size_col = "population" if "population" in demo_df.columns else None
    fig = px.scatter(
        demo_df.sort_values("month"),
        x=demo_col,
        y="complaints_per_1000",
        animation_frame="month",
        color="borough",
        size=size_col,
        hover_name="community_board",
        title=f"Complaints per 1,000 vs {DEMOGRAPHIC_LABELS.get(demo_col, demo_col)}",
        labels={demo_col: DEMOGRAPHIC_LABELS.get(demo_col, demo_col), "complaints_per_1000": "Complaints per 1,000 residents"},
    )
    fig.update_layout(height=540, margin=dict(l=0, r=0, t=48, b=0))
    return fig


def make_overlap_table(filtered_overview: pd.DataFrame, vulnerability_col: str, focused_month: str) -> pd.DataFrame:
    month_df = filtered_overview[filtered_overview["month"] == focused_month].copy()
    cols = ["borough", "community_board", "complaints_per_1000", vulnerability_col]
    extra = [c for c in ["median_income", "poverty_rate", "pct_black", "pct_hispanic", "pct_white", "pct_asian"] if c in month_df.columns]
    use_cols = list(dict.fromkeys(cols + extra))
    month_df = month_df[use_cols].dropna(subset=["complaints_per_1000", vulnerability_col]).copy()
    if month_df.empty:
        return month_df

    burden_cut = month_df["complaints_per_1000"].quantile(0.75)
    if vulnerability_col == "median_income":
        vuln_cut = month_df["median_income"].quantile(0.25)
        month_df["overlap_flag"] = np.where(
            (month_df["complaints_per_1000"] >= burden_cut) & (month_df["median_income"] <= vuln_cut),
            "High burden + low income",
            "Other",
        )
    else:
        vuln_cut = month_df[vulnerability_col].quantile(0.75)
        month_df["overlap_flag"] = np.where(
            (month_df["complaints_per_1000"] >= burden_cut) & (month_df[vulnerability_col] >= vuln_cut),
            "High burden + high vulnerability",
            "Other",
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
    sel_focused = st.sidebar.selectbox("Focused month (map)", options=months, index=len(months) - 1)

    metric_opts = ["complaints", "repeat_descriptor_complaints", "open_cases", "avg_response_hours"]
    if has_demo:
        metric_opts.insert(1, "complaints_per_1000")
        if "repeat_descriptor_per_1000" in overview.columns:
            metric_opts.insert(3, "repeat_descriptor_per_1000")
    metric = st.sidebar.radio("Burden map metric", metric_opts, index=0, format_func=format_metric_label)

    filtered_source = filter_data(df, sel_types, sel_boroughs, *sel_months)
    if filtered_source.empty:
        st.warning("No data for the selected range.")
        st.stop()

    filtered_frames = prepare_dashboard_frames(filtered_source)
    filtered_range = filtered_frames["finest"]
    filtered_overview = filtered_frames["overview"]
    filtered_type = filtered_frames["type_trend"]

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
                x=metric, y="label", orientation="h", color="borough",
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

    st.subheader("District Drilldown")
    dist_opts = sorted(filtered_overview["community_board"].dropna().unique())
    sel_dist = st.selectbox("Community district", dist_opts, index=0)

    dist_series = (
        filtered_range[filtered_range["community_board"] == sel_dist]
        .groupby(["month", "complaint_type"], as_index=False)
        .agg(complaints=("complaints", "sum"),
             open_cases=("open_cases", "sum"),
             repeat_descriptor_complaints=("repeat_descriptor_complaints", "sum"),
             avg_response_hours=("avg_response_hours", "mean"))
        .sort_values(["month", "complaint_type"])
    )
    dist_total = (
        filtered_overview[filtered_overview["community_board"] == sel_dist]
        .groupby("month", as_index=False)
        .agg(complaints=("complaints", "sum"),
             open_cases=("open_cases", "sum"),
             repeat_descriptor_complaints=("repeat_descriptor_complaints", "sum"),
             avg_response_hours=("avg_response_hours", "mean"),
             pct_repeat_descriptor=("pct_repeat_descriptor", "mean"))
        .sort_values("month")
    )

    d1, d2 = st.columns([1.4, 1.0])
    with d1:
        fig_dist = px.line(dist_total, x="month", y=["complaints", "repeat_descriptor_complaints", "open_cases"], markers=True,
                           title=f"{sel_dist}: trend ({sel_months[0]} to {sel_months[1]})",
                           labels={"value": "Count", "variable": "Metric"})
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
        .agg(complaints=("complaints", "sum"),
             repeat_descriptor_complaints=("repeat_descriptor_complaints", "sum"),
             avg_response_hours=("avg_response_hours", "mean"))
        .sort_values("complaints", ascending=False)
    )
    fig_breakdown = px.bar(breakdown, x="complaint_type", y="complaints", color="repeat_descriptor_complaints",
                           title=f"{sel_dist}: complaint breakdown ({sel_months[0]} to {sel_months[1]})",
                           labels={"complaint_type": "Type", "complaints": "Complaints", "repeat_descriptor_complaints": "Repeat"})
    fig_breakdown.update_layout(height=360, margin=dict(l=0, r=0, t=48, b=0))
    st.plotly_chart(fig_breakdown, use_container_width=True)

    with st.expander("Show underlying data table"):
        st.dataframe(dist_series.sort_values(["month", "complaint_type"]).reset_index(drop=True), use_container_width=True)

    st.header("Equity Context")
    if not has_demo or not demo_cols:
        st.info("Merged demographic columns were not found. Run src/merge_demo.py and reload the app.")
    else:
        selected_demo = st.selectbox("Demographic variable", demo_cols,
                                     index=demo_cols.index("median_income") if "median_income" in demo_cols else 0,
                                     format_func=lambda x: DEMOGRAPHIC_LABELS.get(x, x))
        eq1, eq2, eq3 = st.tabs(["Map + Scatter", "Poverty / Race", "Overlap"])

        with eq1:
            st.subheader("Demographic map and burden comparison")
            demo_df = build_demo_overview(filtered_overview, selected_demo)
            c_left, c_right = st.columns(2)
            with c_left:
                if demo_df.empty:
                    st.info(f"No data available for {DEMOGRAPHIC_LABELS.get(selected_demo, selected_demo)}.")
                else:
                    try:
                        geojson = load_community_district_geojson()
                        map_fig = build_demo_map_figure(demo_df, geojson, sel_focused, selected_demo)
                        if map_fig is not None:
                            st.plotly_chart(map_fig, use_container_width=True)
                    except FileNotFoundError as e:
                        st.info(str(e))
            with c_right:
                scatter_fig = build_demo_scatter(demo_df, selected_demo)
                if scatter_fig is not None:
                    st.plotly_chart(scatter_fig, use_container_width=True)
            st.caption("This view is descriptive. It shows spatial overlap and correlation, not causation.")

        with eq2:
            st.subheader("Poverty and race-share context")
            demo_choices_v2 = [c for c in demo_cols if c != "median_income"]
            if not demo_choices_v2:
                st.info("No poverty or race-share variables found.")
            else:
                selected_demo_v2 = st.selectbox("Choose a poverty or race-share variable", demo_choices_v2,
                                                format_func=lambda x: DEMOGRAPHIC_LABELS.get(x, x), key="v2_demo")
                demo_df_v2 = build_demo_overview(filtered_overview, selected_demo_v2)
                c_left2, c_right2 = st.columns(2)
                with c_left2:
                    try:
                        geojson = load_community_district_geojson()
                        map_fig2 = build_demo_map_figure(demo_df_v2, geojson, sel_focused, selected_demo_v2)
                        if map_fig2 is not None:
                            st.plotly_chart(map_fig2, use_container_width=True)
                    except FileNotFoundError as e:
                        st.info(str(e))
                with c_right2:
                    scatter_fig2 = build_demo_scatter(demo_df_v2, selected_demo_v2)
                    if scatter_fig2 is not None:
                        st.plotly_chart(scatter_fig2, use_container_width=True)
                st.caption("These patterns may help identify where sanitation burden and demographic context warrant closer review.")

        with eq3:
            st.subheader("High burden + high vulnerability")
            overlap_candidates = []
            if "median_income" in demo_cols:
                overlap_candidates.append("median_income")
            if "poverty_rate" in demo_cols:
                overlap_candidates.append("poverty_rate")
            for c in ["pct_black", "pct_hispanic", "pct_white", "pct_asian"]:
                if c in demo_cols:
                    overlap_candidates.append(c)

            if not overlap_candidates:
                st.info("No suitable vulnerability fields found.")
            else:
                overlap_var = st.selectbox("Choose vulnerability context variable", overlap_candidates,
                                           format_func=lambda x: DEMOGRAPHIC_LABELS.get(x, x))
                overlap_df = make_overlap_table(filtered_overview, overlap_var, sel_focused)
                flagged = overlap_df[overlap_df["overlap_flag"] != "Other"].copy()
                o1, o2 = st.columns([1.4, 1.0])
                with o1:
                    st.markdown("### Highlighted districts")
                    st.dataframe(flagged, use_container_width=True)
                with o2:
                    if not flagged.empty:
                        fig_overlap = px.bar(flagged, x="community_board", y="complaints_per_1000", color="overlap_flag",
                                             title=f"Flagged districts — {sel_focused}")
                        fig_overlap.update_layout(height=500, margin=dict(l=0, r=0, t=48, b=0))
                        st.plotly_chart(fig_overlap, use_container_width=True)
                st.caption("Flagged districts are defined using simple percentile thresholds. This is a screening tool, not a causal or definitive equity finding.")

    st.header("Bayesian Prediction")
    if bayes_df is None:
        st.info("Run src/predict_bayesian_model.py to enable Bayesian predictions.")
    else:
        bayes_filtered = bayes_df.copy()
        if sel_boroughs:
            bayes_filtered = bayes_filtered[bayes_filtered["borough"].isin(sel_boroughs)]
        bayes_filtered = bayes_filtered[(bayes_filtered["month"] >= sel_months[0]) & (bayes_filtered["month"] <= sel_months[1])]

        if bayes_filtered.empty:
            st.info("No Bayesian predictions available for the selected filters.")
        else:
            bayes_tabs = st.tabs(["Top Predicted Districts", "Prediction vs Uncertainty", "Actual vs Predicted"])

            with bayes_tabs[0]:
                latest_pred = bayes_filtered[bayes_filtered["month"] == sel_focused].copy()
                if latest_pred.empty:
                    latest_pred = bayes_filtered[bayes_filtered["month"] == bayes_filtered["month"].max()].copy()
                latest_pred = latest_pred.sort_values("predicted_complaints", ascending=False).head(15)
                fig_top_pred = px.bar(
                    latest_pred.sort_values("predicted_complaints", ascending=True),
                    x="predicted_complaints", y="community_board", orientation="h", color="borough",
                    title=f"Top predicted districts — {sel_focused}",
                    labels={"predicted_complaints": "Predicted complaints (posterior median)", "community_board": "Community district"},
                )
                fig_top_pred.update_layout(height=550, margin=dict(l=0, r=0, t=48, b=0))
                st.plotly_chart(fig_top_pred, use_container_width=True)
                st.dataframe(
                    latest_pred[["month", "borough", "community_board", "complaints", "predicted_complaints",
                                 "predicted_complaints_p10", "predicted_complaints_p90", "prediction_interval_width"]],
                    use_container_width=True,
                )

            with bayes_tabs[1]:
                latest_pred = bayes_filtered[bayes_filtered["month"] == sel_focused].copy()
                if latest_pred.empty:
                    latest_pred = bayes_filtered[bayes_filtered["month"] == bayes_filtered["month"].max()].copy()
                fig_unc = px.scatter(
                    latest_pred,
                    x="predicted_complaints", y="prediction_interval_width", color="borough",
                    hover_name="community_board", size="complaints",
                    title=f"Predicted burden vs uncertainty — {sel_focused}",
                    labels={"predicted_complaints": "Predicted complaints (posterior median)",
                            "prediction_interval_width": "Prediction interval width (p90 - p10)",
                            "complaints": "Observed complaints"},
                )
                fig_unc.update_layout(height=550, margin=dict(l=0, r=0, t=48, b=0))
                st.plotly_chart(fig_unc, use_container_width=True)

            with bayes_tabs[2]:
                pred_dist_opts = sorted(bayes_filtered["community_board"].dropna().unique())
                pred_sel_dist = st.selectbox("Community district for Bayesian comparison", pred_dist_opts, key="bayes_dist_select")
                dist_pred = bayes_filtered[bayes_filtered["community_board"] == pred_sel_dist].copy().sort_values("month")
                fig_actual_pred = px.line(dist_pred, x="month", y=["complaints", "predicted_complaints"], markers=True,
                                          title=f"{pred_sel_dist}: actual vs predicted complaints",
                                          labels={"value": "Complaints", "variable": "Series"})
                fig_actual_pred.update_layout(height=450, margin=dict(l=0, r=0, t=48, b=0))
                st.plotly_chart(fig_actual_pred, use_container_width=True)
                fig_band = px.line(dist_pred, x="month", y=["predicted_complaints_p10", "predicted_complaints_p90"],
                                   markers=False, title=f"{pred_sel_dist}: prediction interval",
                                   labels={"value": "Predicted complaints", "variable": "Interval bound"})
                fig_band.update_layout(height=350, margin=dict(l=0, r=0, t=48, b=0))
                st.plotly_chart(fig_band, use_container_width=True)
                st.dataframe(
                    dist_pred[["month", "complaints", "predicted_complaints", "predicted_complaints_p10",
                               "predicted_complaints_p90", "prediction_interval_width"]],
                    use_container_width=True,
                )

        st.caption("Bayesian predictions are exploratory and uncertainty-aware. They are based on the fitted hierarchical count model and should be interpreted as screening estimates, not exact forecasts.")


if __name__ == "__main__":
    main()
