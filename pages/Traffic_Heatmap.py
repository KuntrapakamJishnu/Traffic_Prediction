import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from location_catalog import amaravati_heatmap_locations

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Traffic Heatmap Dashboard",
    layout="wide"
)

st.title("Traffic Heatmap Dashboard")

# ==================================================
# FIXED LOCATIONS
# ==================================================
LOCATIONS = amaravati_heatmap_locations()

# ==================================================
# INITIALIZE SESSION STATE (CRITICAL)
# ==================================================
if "traffic_df" not in st.session_state:
    st.session_state.traffic_df = pd.DataFrame([
        {
            "location": loc["location"],
            "lat": loc["lat"],
            "lon": loc["lon"],
            "traffic_flow": np.random.randint(150, 450)
        }
        for loc in LOCATIONS
    ])

if "traffic_map" not in st.session_state:
    df = st.session_state.traffic_df

    center_lat = df["lat"].mean()
    center_lon = df["lon"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles="OpenStreetMap",
        control_scale=True
    )

    heat_data = [
        [row.lat, row.lon, row.traffic_flow]
        for row in df.itertuples()
    ]

    HeatMap(
        heat_data,
        radius=35,
        blur=20,
        max_zoom=13
    ).add_to(m)

    for row in df.itertuples():
        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=6,
            popup=f"""
            <b>{row.location}</b><br>
            Traffic Flow: {row.traffic_flow}
            """,
            color="blue",
            fill=True,
            fill_opacity=0.7
        ).add_to(m)

    st.session_state.traffic_map = m

# ==================================================
# RENDER MAP (NO RERUN, NO BLINK)
# ==================================================
st.subheader(" Traffic Density Heatmap")

st_folium(
    st.session_state.traffic_map,
    height=520,
    width=1200,
    returned_objects=[],   #  MOST IMPORTANT LINE
)

# ==================================================
# DATA TABLE
# ==================================================
st.markdown("---")
st.subheader(" Traffic Data")

st.dataframe(
    st.session_state.traffic_df.sort_values("traffic_flow", ascending=False),
    use_container_width=True
)
