import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.cluster import KMeans
from pulp import *
import plotly.express as px

st.set_page_config(page_title="NYC 311 Optimization Dashboard", layout="wide")

st.title("🚛 NYC 311 Sanitation Optimization Dashboard")

# -------------------------------
# 1. LOAD DATA
# -------------------------------

@st.cache_data
def load_data():

    url = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"

    params = {
        "$where": "complaint_type in ('Missed Collection','Illegal Dumping') "
                  "AND latitude IS NOT NULL AND longitude IS NOT NULL",
        "$limit": 20000
    }

    response = requests.get(url, params=params)

    # safety check (IMPORTANT)
    if response.status_code != 200:
        st.error(f"API Error: {response.status_code}")
        return pd.DataFrame()

    data = response.json()

    # SECOND safety check (THIS FIXES YOUR ERROR)
    if not isinstance(data, list):
        st.error("Unexpected API response format")
        return pd.DataFrame()

    df = pd.DataFrame.from_records(data)

    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

    return df[['latitude','longitude','borough']].dropna()

df = load_data()

st.subheader("📍 Raw Data Preview")
st.write(df.head())


# -------------------------------
# 2. CLUSTERING
# -------------------------------

k = st.sidebar.slider("Number of Clusters", 3, 12, 8)

kmeans = KMeans(n_clusters=k, random_state=42)
df['cluster'] = kmeans.fit_predict(df[['latitude','longitude']])


# -------------------------------
# 3. DEMAND PER CLUSTER
# -------------------------------

cluster_demand = df.groupby('cluster').size().reset_index()
cluster_demand.columns = ['cluster','demand']


# -------------------------------
# 4. OPTIMIZATION MODEL
# -------------------------------

capacity = st.sidebar.slider("Truck Capacity", 10, 100, 30)
total_trucks = st.sidebar.slider("Total Trucks Available", 5, 50, 20)

model = LpProblem("Truck_Allocation", LpMinimize)

clusters = cluster_demand['cluster'].tolist()

trucks = LpVariable.dicts("trucks", clusters, lowBound=0, cat='Integer')
unmet = LpVariable.dicts("unmet", clusters, lowBound=0)

model += lpSum([unmet[i] for i in clusters])

for i in clusters:
    demand = cluster_demand[cluster_demand['cluster']==i]['demand'].values[0]
    model += unmet[i] >= demand - capacity * trucks[i]

model += lpSum([trucks[i] for i in clusters]) == total_trucks

for i in clusters:
    model += trucks[i] >= 1
    model += trucks[i] <= 6

model.solve()


# -------------------------------
# 5. RESULTS TABLE
# -------------------------------

results = []

for i in clusters:
    demand = cluster_demand[cluster_demand['cluster']==i]['demand'].values[0]

    results.append({
        "Cluster": i,
        "Demand": demand,
        "Trucks": trucks[i].value(),
        "Unmet": unmet[i].value()
    })

result_df = pd.DataFrame(results)

st.subheader("🚛 Optimization Results")
st.dataframe(result_df)


# -------------------------------
# 6. MAP VISUALIZATION
# -------------------------------

st.subheader("🗺️ Spatial Cluster View")

fig_map = px.scatter_mapbox(
    df,
    lat="latitude",
    lon="longitude",
    color="cluster",
    zoom=9,
    height=600,
    mapbox_style="carto-positron"
)

st.plotly_chart(fig_map, use_container_width=True)


# -------------------------------
# 7. DEMAND BAR CHART
# -------------------------------

st.subheader("📊 Demand per Cluster")

fig_bar = px.bar(
    cluster_demand,
    x="cluster",
    y="demand",
    color="demand"
)

st.plotly_chart(fig_bar, use_container_width=True)


# -------------------------------
# 8. TRUCK ALLOCATION PLOT
# -------------------------------

st.subheader("🚛 Truck Allocation")

fig_trucks = px.bar(
    result_df,
    x="Cluster",
    y="Trucks",
    color="Trucks"
)

st.plotly_chart(fig_trucks, use_container_width=True)