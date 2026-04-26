import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
FEATURE_FILE = BASE_DIR / "data" / "processed" / "neighborhood_month.parquet"
MERGED_FILE = BASE_DIR / "data" / "processed" / "neighborhood_month_with_demo.parquet"
GEOJSON_FILE = BASE_DIR / "data" / "external" / "community_districts.geojson"

st.set_page_config(page_title="NYC Sanitation Burden Explorer", layout="wide")
st.title("NYC Sanitation Burden Explorer")
st.caption("Dashboard for sanitation-related 311 burden with demographic context.")

DEMOGRAPHIC_LABELS = {
    "median_income": "Median household income",
    "poverty_rate": "Poverty rate",
    "pct_black": "% Black",
    "pct_hispanic": "% Hispanic/Latino",
    "pct_white": "% White",
    "pct_asian": "% Asian",
}

# Helpers
def load_geojson():
    if not GEOJSON_FILE.exists():
        return None
    with open(GEOJSON_FILE, "r") as f:
        gj = json.load(f)

    # Normalize BoroCD / boro_cd
    for feat in gj["features"]:
        props = feat.get("properties", {})
        if "BoroCD" in props and "boro_cd" not in props:
            props["boro_cd"] = props["BoroCD"]
        elif "boro_cd" in props and "BoroCD" not in props:
            props["BoroCD"] = props["boro_cd"]

    return gj


def add_borocd(df):
    borough_map = {
        "MANHATTAN": 1,
        "BRONX": 2,
        "BROOKLYN": 3,
        "QUEENS": 4,
        "STATEN ISLAND": 5,
    }

    out = df.copy()

    def parse_cd(val):
        if pd.isna(val):
            return np.nan, None
        s = str(val).strip()
        parts = s.split(" ", 1)
        if len(parts) < 2:
            return np.nan, None
        cd_part = parts[0]
        borough_part = parts[1].upper()
        try:
            cd_num = int(cd_part)
        except ValueError:
            return np.nan, None
        borough_num = borough_map.get(borough_part)
        if borough_num is None:
            return np.nan, borough_part
        return int(f"{borough_num}{cd_num:02d}"), borough_part

    parsed = out["community_board"].apply(parse_cd)
    out["BoroCD"] = parsed.apply(lambda x: x[0])
    out["borough_from_cb"] = parsed.apply(lambda x: x[1])
    return out


def get_available_demo_cols(df):
    return [c for c in DEMOGRAPHIC_LABELS if c in df.columns]


def make_demographic_map(df, geojson, month, demographic_col):
    if geojson is None:
        st.warning("GeoJSON file not found. Cannot render map.")
        return None

    map_df = (
        df[df["month"].astype(str) == month]
        .groupby(["community_board"], as_index=False)
        .agg(
            demographic_value=(demographic_col, "mean"),
            complaints=("complaints", "sum"),
            complaints_per_1000=("complaints_per_1000", "mean"),
            population=("population", "mean"),
        )
    )

    map_df = add_borocd(map_df)
    map_df = map_df.dropna(subset=["BoroCD"]).copy()
    map_df["BoroCD"] = map_df["BoroCD"].astype(int)

    fig = px.choropleth_mapbox(
        map_df,
        geojson=geojson,
        locations="BoroCD",
        featureidkey="properties.BoroCD",
        color="demographic_value",
        color_continuous_scale="Viridis",
        hover_name="community_board",
        hover_data={
            "demographic_value": True,
            "complaints": True,
            "complaints_per_1000": ":.2f",
            "population": True,
            "BoroCD": False,
        },
        mapbox_style="carto-darkmatter",
        zoom=9.3,
        center={"lat": 40.7128, "lon": -74.0060},
        opacity=0.7,
        title=f"{DEMOGRAPHIC_LABELS.get(demographic_col, demographic_col)} — {month}",
    )
    fig.update_layout(height=600, margin={"r":0, "t":50, "l":0, "b":0})
    return fig


def make_scatter(df, demo_col):
    scatter_df = (
        df.groupby(["month", "borough", "community_board"], as_index=False)
        .agg(
            complaints_per_1000=("complaints_per_1000", "mean"),
            population=("population", "mean"),
            demographic_value=(demo_col, "mean"),
        )
        .dropna(subset=["complaints_per_1000", "demographic_value"])
        .sort_values("month")
    )

    fig = px.scatter(
        scatter_df,
        x="demographic_value",
        y="complaints_per_1000",
        animation_frame="month",
        color="borough",
        size="population",
        hover_name="community_board",
        title=f"Complaints per 1,000 vs {DEMOGRAPHIC_LABELS.get(demo_col, demo_col)}",
        labels={
            "demographic_value": DEMOGRAPHIC_LABELS.get(demo_col, demo_col),
            "complaints_per_1000": "Complaints per 1,000 residents",
            "borough": "Borough",
        },
    )
    fig.update_layout(height=600)
    return fig


def make_overlap_table(df, vulnerability_col, focused_month):
    month_df = df[df["month"].astype(str) == focused_month].copy()

    month_df = (
        month_df.groupby(["borough", "community_board"], as_index=False)
        .agg(
            complaints_per_1000=("complaints_per_1000", "mean"),
            median_income=("median_income", "mean") if "median_income" in month_df.columns else ("complaints", "size"),
            poverty_rate=("poverty_rate", "mean") if "poverty_rate" in month_df.columns else ("complaints", "size"),
            pct_black=("pct_black", "mean") if "pct_black" in month_df.columns else ("complaints", "size"),
            pct_hispanic=("pct_hispanic", "mean") if "pct_hispanic" in month_df.columns else ("complaints", "size"),
            pct_white=("pct_white", "mean") if "pct_white" in month_df.columns else ("complaints", "size"),
            pct_asian=("pct_asian", "mean") if "pct_asian" in month_df.columns else ("complaints", "size"),
        )
    )

    if vulnerability_col == "median_income":
        burden_cut = month_df["complaints_per_1000"].quantile(0.75)
        vuln_cut = month_df["median_income"].quantile(0.25)
        month_df["overlap_flag"] = np.where(
            (month_df["complaints_per_1000"] >= burden_cut)
            & (month_df["median_income"] <= vuln_cut),
            "High burden + low income",
            "Other",
        )
        month_df["vulnerability_value"] = month_df["median_income"]

    else:
        burden_cut = month_df["complaints_per_1000"].quantile(0.75)
        vuln_cut = month_df[vulnerability_col].quantile(0.75)
        month_df["overlap_flag"] = np.where(
            (month_df["complaints_per_1000"] >= burden_cut)
            & (month_df[vulnerability_col] >= vuln_cut),
            "High burden + high vulnerability",
            "Other",
        )
        month_df["vulnerability_value"] = month_df[vulnerability_col]

    return month_df.sort_values(
        ["overlap_flag", "complaints_per_1000"],
        ascending=[False, False]
    )


# Load data
if MERGED_FILE.exists():
    df = pd.read_parquet(MERGED_FILE)
    st.sidebar.success("Loaded merged dataset with demographic context.")
elif FEATURE_FILE.exists():
    df = pd.read_parquet(FEATURE_FILE)
    st.sidebar.info("Loaded feature dataset without demographic context.")
else:
    st.error("No processed dataset found yet. Run the pipeline first.")
    st.stop()

geojson = load_geojson()

# Sidebar filters
complaint_options = sorted(df["complaint_type"].dropna().astype(str).unique())
selected_types = st.sidebar.multiselect(
    "Complaint types",
    complaint_options,
    default=complaint_options[: min(3, len(complaint_options))],
)

month_options = sorted(df["month"].dropna().astype(str).unique())
selected_months = st.sidebar.multiselect(
    "Months",
    month_options,
    default=month_options,
)

borough_options = sorted(df["borough"].dropna().astype(str).unique())
selected_boroughs = st.sidebar.multiselect(
    "Boroughs",
    borough_options,
    default=borough_options,
)

filtered = df.copy()
if selected_types:
    filtered = filtered[filtered["complaint_type"].astype(str).isin(selected_types)]
if selected_months:
    filtered = filtered[filtered["month"].astype(str).isin(selected_months)]
if selected_boroughs:
    filtered = filtered[filtered["borough"].astype(str).isin(selected_boroughs)]

if filtered.empty:
    st.warning("No rows match the current filters.")
    st.stop()

available_demo_cols = get_available_demo_cols(filtered)

# Existing summary section
summary = (
    filtered.groupby(["borough", "community_board"], as_index=False)["complaints"]
    .sum()
    .sort_values("complaints", ascending=False)
)

col1, col2 = st.columns(2)
with col1:
    fig_bar = px.bar(
        summary.head(15),
        x="community_board",
        y="complaints",
        color="borough",
        title="Top 15 Community Boards by Complaints",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    trend = (
        filtered.groupby(["month", "complaint_type"], as_index=False)["complaints"]
        .sum()
        .sort_values("month")
    )
    fig_line = px.line(
        trend,
        x="month",
        y="complaints",
        color="complaint_type",
        markers=True,
        title="Monthly Complaint Trend",
    )
    st.plotly_chart(fig_line, use_container_width=True)

st.subheader("Neighborhood summary")
st.dataframe(summary, use_container_width=True)

if "complaints_per_1000" in filtered.columns:
    st.subheader("Complaints per 1,000 residents")
    per_capita = (
        filtered.groupby(["borough", "community_board"], as_index=False)["complaints_per_1000"]
        .mean()
        .sort_values("complaints_per_1000", ascending=False)
    )
    st.dataframe(per_capita, use_container_width=True)

# New Equity Context section
# maybe workshop the name?
st.header("Equity Context")

if not available_demo_cols:
    st.info("No demographic columns are available yet.")
    st.stop()

focused_month = st.selectbox(
    "Focused month for maps / overlap analysis",
    month_options,
    index=len(month_options) - 1
)

selected_demo = st.selectbox(
    "Demographic variable",
    available_demo_cols,
    index=available_demo_cols.index("median_income") if "median_income" in available_demo_cols else 0,
    format_func=lambda x: DEMOGRAPHIC_LABELS.get(x, x),
)

tab1, tab2, tab3 = st.tabs(["v1: Map + Scatter", "v2: Poverty / Race", "v3: Overlap"])

with tab1:
    st.subheader("v1 — Median income context")
    map_col, scatter_col = st.columns(2)

    with map_col:
        map_fig = make_demographic_map(filtered, geojson, focused_month, selected_demo)
        if map_fig is not None:
            st.plotly_chart(map_fig, use_container_width=True)

    with scatter_col:
        scatter_fig = make_scatter(filtered, selected_demo)
        st.plotly_chart(scatter_fig, use_container_width=True)

    st.caption("This view is descriptive. It shows spatial overlap and correlation, not causation.")

with tab2:
    st.subheader("v2 — Poverty and race-share context")

    demo_choices_v2 = [c for c in available_demo_cols if c != "median_income"]
    if not demo_choices_v2:
        st.info("No poverty / race-share variables found yet.")
    else:
        selected_demo_v2 = st.selectbox(
            "Choose a poverty or race-share variable",
            demo_choices_v2,
            format_func=lambda x: DEMOGRAPHIC_LABELS.get(x, x),
            key="v2demo",
        )

        map_col2, scatter_col2 = st.columns(2)

        with map_col2:
            map_fig2 = make_demographic_map(filtered, geojson, focused_month, selected_demo_v2)
            if map_fig2 is not None:
                st.plotly_chart(map_fig2, use_container_width=True)

        with scatter_col2:
            scatter_fig2 = make_scatter(filtered, selected_demo_v2)
            st.plotly_chart(scatter_fig2, use_container_width=True)

        st.caption("These patterns may help identify where sanitation burden and demographic context warrant closer review.")

with tab3:
    st.subheader("v3 — High burden + high vulnerability")

    overlap_candidates = []
    if "median_income" in available_demo_cols:
        overlap_candidates.append("median_income")
    if "poverty_rate" in available_demo_cols:
        overlap_candidates.append("poverty_rate")
    for c in ["pct_black", "pct_hispanic", "pct_white", "pct_asian"]:
        if c in available_demo_cols:
            overlap_candidates.append(c)

    if not overlap_candidates:
        st.info("No suitable vulnerability fields found.")
    else:
        overlap_var = st.selectbox(
            "Choose vulnerability context variable",
            overlap_candidates,
            format_func=lambda x: DEMOGRAPHIC_LABELS.get(x, x),
        )

        overlap_df = make_overlap_table(filtered, overlap_var, focused_month)

        flagged = overlap_df[overlap_df["overlap_flag"] != "Other"].copy()

        c1, c2 = st.columns([1.4, 1.0])

        with c1:
            st.markdown("### Highlighted districts")
            st.dataframe(flagged, use_container_width=True)

        with c2:
            fig_overlap = px.bar(
                flagged,
                x="community_board",
                y="complaints_per_1000",
                color="overlap_flag",
                title=f"Flagged districts — {focused_month}",
            )
            fig_overlap.update_layout(height=500)
            st.plotly_chart(fig_overlap, use_container_width=True)

        st.caption(
            "Flagged districts are defined using simple percentile thresholds. "
            "This is a screening tool, not a causal or definitive equity finding."
        )