from __future__ import annotations

import copy

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

from sanitation_explorer.data import format_metric_label


def metric_series_to_color(values: pd.Series) -> list[list[int]]:
    if values.empty:
        return []
    min_value = float(values.min())
    max_value = float(values.max())
    span = max(max_value - min_value, 1e-9)
    colors: list[list[int]] = []
    for value in values.fillna(min_value):
        normalized = (float(value) - min_value) / span
        colors.append(
            [
                int(250 - normalized * 18),
                int(196 - normalized * 70),
                int(148 - normalized * 68),
                220,
            ]
        )
    return colors


def collapse_duplicate_districts(month_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if month_df.empty or month_df["BoroCD"].is_unique:
        return month_df.copy()

    aggregations: dict[str, str] = {
        "month": "first",
        "borough": "first",
        "community_board": "first",
        "complaints": "sum",
        "open_cases": "sum",
        "repeat_descriptor_complaints": "sum",
        "avg_response_hours": "mean",
        "repeat_share_pct": "mean",
    }
    if "complaints_per_1000" in month_df.columns:
        aggregations["complaints_per_1000"] = "mean"
    if "repeat_descriptor_per_1000" in month_df.columns:
        aggregations["repeat_descriptor_per_1000"] = "mean"
    if metric not in aggregations and metric in month_df.columns:
        aggregations[metric] = "mean"

    return month_df.groupby("BoroCD", as_index=False).agg(aggregations)


def build_3d_geojson(base_geojson: dict, month_df: pd.DataFrame, metric: str) -> dict:
    month_df = collapse_duplicate_districts(month_df, metric)
    metric_lookup = month_df.set_index("BoroCD").to_dict(orient="index")
    enriched = copy.deepcopy(base_geojson)
    if month_df.empty:
        return {"type": "FeatureCollection", "features": []}

    colors = metric_series_to_color(month_df[metric])
    color_lookup = {int(month_df.iloc[idx]["BoroCD"]): colors[idx] for idx in range(len(month_df))}
    max_metric = max(float(month_df[metric].max()), 1.0)

    features = []
    for feature in enriched["features"]:
        properties = feature["properties"]
        boro_cd = properties.get("BoroCD")
        row = metric_lookup.get(boro_cd)
        if row is None:
            continue

        metric_value = float(row.get(metric, 0) or 0)
        elevation = max(150, int((metric_value / max_metric) * 12000))
        properties.update(
            {
                "community_board": row["community_board"],
                "borough": row["borough"],
                "complaints": int(row["complaints"]),
                "open_cases": int(row["open_cases"]),
                "repeat_descriptor_complaints": int(row["repeat_descriptor_complaints"]),
                "avg_response_hours": round(float(row["avg_response_hours"]), 1),
                "repeat_share_pct": round(float(row["repeat_share_pct"]), 1),
                metric: round(metric_value, 2),
                "fill_color": color_lookup[int(boro_cd)],
                "elevation": elevation,
            }
        )
        if "complaints_per_1000" in row:
            properties["complaints_per_1000"] = round(float(row["complaints_per_1000"]), 2)
        if "repeat_descriptor_per_1000" in row:
            properties["repeat_descriptor_per_1000"] = round(float(row["repeat_descriptor_per_1000"]), 2)
        features.append(feature)

    return {"type": "FeatureCollection", "features": features}


def render_map(
    *,
    geojson: dict,
    map_df: pd.DataFrame,
    month_slice: pd.DataFrame,
    metric: str,
    map_mode: str,
    show_animation: bool,
    extrusion_scale: float,
    selected_map_month: str,
) -> None:
    hover_data = {
        "borough": True,
        "complaints": True,
        "open_cases": True,
        "repeat_descriptor_complaints": True,
        "avg_response_hours": ":.1f",
        "repeat_share_pct": ":.1f",
        "BoroCD": False,
    }
    if "complaints_per_1000" in month_slice.columns:
        hover_data["complaints_per_1000"] = ":.2f"
    if "repeat_descriptor_per_1000" in month_slice.columns:
        hover_data["repeat_descriptor_per_1000"] = ":.2f"

    if map_mode == "3D":
        three_d_geojson = build_3d_geojson(geojson, month_slice, metric)
        tooltip_lines = [
            "<b>{community_board}</b> ({borough})",
            f"{format_metric_label(metric)}: {{{metric}}}",
            "Complaints: {complaints}",
            "Repeat complaints: {repeat_descriptor_complaints}",
            "Open cases: {open_cases}",
            "Avg response hours: {avg_response_hours}",
            "Repeat share: {repeat_share_pct}%",
        ]
        if "complaints_per_1000" in month_slice.columns:
            tooltip_lines.insert(2, "Complaints per 1,000: {complaints_per_1000}")
        deck = pdk.Deck(
            map_style="mapbox://styles/mapbox/dark-v11",
            initial_view_state=pdk.ViewState(
                latitude=40.7128,
                longitude=-74.0060,
                zoom=9.1,
                pitch=58,
                bearing=-18,
            ),
            layers=[
                pdk.Layer(
                    "GeoJsonLayer",
                    three_d_geojson,
                    pickable=False,
                    stroked=False,
                    filled=True,
                    extruded=False,
                    opacity=0.08,
                    get_fill_color=[255, 214, 170, 18],
                ),
                pdk.Layer(
                    "GeoJsonLayer",
                    three_d_geojson,
                    pickable=True,
                    stroked=True,
                    filled=True,
                    extruded=True,
                    wireframe=False,
                    get_fill_color="properties.fill_color",
                    get_line_color=[255, 224, 196, 90],
                    get_elevation="properties.elevation",
                    elevation_scale=extrusion_scale,
                    line_width_min_pixels=1,
                    material={
                        "ambient": 0.35,
                        "diffuse": 0.6,
                        "shininess": 28,
                        "specularColor": [255, 220, 180],
                    },
                ),
            ],
            tooltip={
                "html": "<br/>".join(tooltip_lines),
                "style": {
                    "backgroundColor": "#0f1720",
                    "color": "white",
                    "border": "1px solid rgba(255, 214, 170, 0.35)",
                },
            },
        )
        st.pydeck_chart(deck, width="stretch", height=640)
        st.caption(
            f"3D district extrusion for {selected_map_month}. Height and color both encode {format_metric_label(metric).lower()}."
        )
        return

    if show_animation:
        fig_map = px.choropleth_map(
            map_df,
            geojson=geojson,
            locations="BoroCD",
            featureidkey="properties.BoroCD",
            color=metric,
            animation_frame="month",
            hover_name="community_board",
            hover_data=hover_data,
            center={"lat": 40.7128, "lon": -74.0060},
            zoom=8.8,
            opacity=0.72,
            color_continuous_scale="YlOrRd",
            map_style="carto-positron",
            title=f"{format_metric_label(metric)} by community district",
        )
    else:
        fig_map = px.choropleth_map(
            month_slice,
            geojson=geojson,
            locations="BoroCD",
            featureidkey="properties.BoroCD",
            color=metric,
            hover_name="community_board",
            hover_data=hover_data,
            center={"lat": 40.7128, "lon": -74.0060},
            zoom=8.8,
            opacity=0.72,
            color_continuous_scale="YlOrRd",
            map_style="carto-positron",
            title=f"{format_metric_label(metric)} in {selected_map_month}",
        )

    fig_map.update_layout(margin=dict(l=0, r=0, t=48, b=0), height=640)
    st.plotly_chart(fig_map, width="stretch")
