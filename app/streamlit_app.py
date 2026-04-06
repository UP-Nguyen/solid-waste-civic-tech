import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
FEATURE_FILE = BASE_DIR / "data" / "processed" / "neighborhood_month.parquet"
MERGED_FILE = BASE_DIR / "data" / "processed" / "neighborhood_month_with_demo.parquet"

st.set_page_config(page_title="NYC Sanitation Burden Explorer", layout="wide")
st.title("NYC Sanitation Burden Explorer")
st.caption("First-pass dashboard for sanitation-related 311 complaints by community board.")

if MERGED_FILE.exists():
    df = pd.read_parquet(MERGED_FILE)
    st.sidebar.success("Loaded merged dataset with demographic context.")
elif FEATURE_FILE.exists():
    df = pd.read_parquet(FEATURE_FILE)
    st.sidebar.info("Loaded feature dataset without demographic context.")
else:
    st.error("No processed dataset found yet. Run the pipeline first.")
    st.stop()

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
