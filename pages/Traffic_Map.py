import streamlit as st
import pandas as pd
try:
    import folium
    from streamlit_folium import st_folium
    from folium.plugins import MarkerCluster
    import branca.colormap as bcm
    _HAS_FOLIUM = True
except Exception:
    folium = None
    st_folium = None
    MarkerCluster = None
    bcm = None
    _HAS_FOLIUM = False

# Import psycopg2 lazily and handle if it's not installed
try:
    import psycopg2
except Exception:
    psycopg2 = None

# ==============================
# POSTGRESQL CONFIG
# ==============================
DB_CONFIG = {
    "dbname": "traffic_db",
    "user": "postgres",
    "password": "postgres123",  
    "host": "localhost",
    "port": 5433
}


import json
import os
from pathlib import Path

# Legacy sample coords (kept for backwards compatibility)
LOCATION_COORDS = {
    1: (12.9716, 77.5946),   # Bangalore
    2: (13.0827, 80.2707),   # Chennai
    3: (17.3850, 78.4867),   # Hyderabad
    4: (19.0760, 72.8777),   # Mumbai
    5: (28.6139, 77.2090),   # Delhi
}

# Small mapping of major Dhanbad zones to lat/lon and friendly names
LOCATION_POI = {
    0: {"name": "Dhanbad CBD", "lat": 23.7925, "lon": 86.4350},
    1: {"name": "Baliapur", "lat": 23.7350, "lon": 86.3850},
    2: {"name": "Govindpur", "lat": 23.7450, "lon": 86.4500},
    3: {"name": "IIT-ISM", "lat": 23.8045, "lon": 86.4340},
}

# Try to load SUMO edge -> coords mapping created by scripts/generate_location_coords.py
LOCATION_COORDS_STR = {}
coords_path = Path("cube_layout") / "location_coords.json"
if coords_path.exists():
    try:
        with open(coords_path, 'r', encoding='utf-8') as fh:
            raw = json.load(fh)
            # values are [lat, lon]
            for k, v in raw.items():
                try:
                    LOCATION_COORDS_STR[k] = (float(v[0]), float(v[1]))
                except Exception:
                    continue
    except Exception:
        LOCATION_COORDS_STR = {}

# ==============================
# LOAD LOCATION DATA
# ==============================
def load_location_data():
    # Prefer aggregated CSV (real location names) when available
    try:
        agg_path = Path("aggregated_traffic_all_new_ok.csv")
        if agg_path.exists():
            df_agg = pd.read_csv(agg_path)
            # compute average avg_flow per location_name
            grp = df_agg.groupby("location_name")["avg_flow"].mean().reset_index().rename(columns={"location_name": "location", "avg_flow": "avg_flow"})

            # pick a sample row per location to extract zone/time_window/max_flow
            sample_rows = df_agg.sort_values("avg_flow", ascending=False).groupby("location_name").first().reset_index()
            sample_rows = sample_rows.rename(columns={"location_name": "location"})

            df = grp.merge(sample_rows[["location", "zone", "time_window", "max_flow"]], on="location", how="left")

            # assign approximate coordinates centered on Dhanbad (IIT-ISM area) so map shows relevant region
            base_lat = 23.795
            base_lon = 86.435
            n = len(df)
            cols = max(1, int(n**0.5))
            delta_lat = 0.002
            delta_lon = 0.002
            lat_list = []
            lon_list = []
            for i in range(n):
                r = i // cols
                c = i % cols
                lat_list.append(base_lat + (r * delta_lat))
                lon_list.append(base_lon + (c * delta_lon))

            df["lat"] = lat_list
            df["lon"] = lon_list
            return df.rename(columns={"location": "location", "avg_flow": "avg_flow"})

        # Next prefer cube CSV + coords mapping if present
        coords_path = Path("cube_layout") / "location_coords.json"
        csv_path = Path("cube_layout") / "cube_city_traffic.csv"
        if csv_path.exists():
            import json
            coords = {}
            if coords_path.exists():
                with open(coords_path, 'r', encoding='utf-8') as fh:
                    coords = json.load(fh)

            df_rt = pd.read_csv(csv_path)
            # compute latest flow per location
            df_latest = df_rt.sort_values(["location", "timestep"]).groupby("location").tail(1)
            df_latest = df_latest.groupby("location")["flow"].mean().reset_index().rename(columns={"flow": "avg_flow"})
            return df_latest

        # CSV not present: try DB
        if psycopg2 is not None:
            try:
                conn = psycopg2.connect(**DB_CONFIG)
                try:
                    # resilient query: try COALESCE first, then fallback to location_id only
                    query_try = """
                        SELECT COALESCE(location::text, location_id::text) AS location,
                               AVG(predicted_flow) AS avg_flow
                        FROM traffic_predictions
                        GROUP BY COALESCE(location::text, location_id::text);
                    """
                    query_fallback = """
                        SELECT location_id::text AS location,
                               AVG(predicted_flow) AS avg_flow
                        FROM traffic_predictions
                        GROUP BY location_id::text;
                    """
                    try:
                        df = pd.read_sql(query_try, conn)
                    except Exception:
                        df = pd.read_sql(query_fallback, conn)
                except Exception:
                    df = pd.DataFrame()
                finally:
                    try:
                        conn.close()
                    except Exception:
                        pass

                if not df.empty:
                    # try to map numeric locations to friendly names and POIs
                    try:
                        df["location_id_num"] = pd.to_numeric(df["location"], errors="coerce")
                        df["location_name"] = df["location_id_num"].map(lambda x: LOCATION_POI.get(int(x), {}).get("name") if not pd.isna(x) and int(x) in LOCATION_POI else None)
                        df["location_name"] = df["location_name"].fillna(df["location"].astype(str))
                        # if lat/lon available in LOCATION_POI, add them
                        lat_list = []
                        lon_list = []
                        for v in df["location"]:
                            lat = None
                            lon = None
                            try:
                                lid = int(v)
                                if lid in LOCATION_POI:
                                    lat = LOCATION_POI[lid]["lat"]
                                    lon = LOCATION_POI[lid]["lon"]
                            except Exception:
                                pass
                            lat_list.append(lat)
                            lon_list.append(lon)
                        df["lat"] = lat_list
                        df["lon"] = lon_list
                    except Exception:
                        pass
                    return df
            except Exception:
                # DB connection failed; fall through to CSV fallback
                pass

        st.warning("No DB and no local CSV available; no location data to display.")
        return pd.DataFrame()
    except Exception:
        st.warning("Failed to load location data; check CSV/DB and paths.")
        return pd.DataFrame()

# ==============================
# STREAMLIT UI
# ==============================
st.title("🗺️ Traffic Congestion Map")

df = load_location_data()

if df.empty:
    st.warning("No traffic data available yet.")
else:
    # Expect textual 'location' column
    if "location" not in df.columns:
        st.error("Data does not contain 'location' column.")
        st.stop()

    # Center map: prefer data centroid when lat/lon present
    if not _HAS_FOLIUM:
        st.warning("Optional packages `folium` and `streamlit_folium` are not installed. Install with `pip install folium streamlit-folium` to view the map.")
        st.info("No map available (missing packages).")
    else:
        if {"lat", "lon"}.issubset(df.columns):
            try:
                center = [float(df["lat"].mean()), float(df["lon"].mean())]
            except Exception:
                center = [23.795, 86.435]
        else:
            # fallback to Dhanbad center
            center = [23.795, 86.435]
        # base map with light tiles
        m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")
        marker_cluster = MarkerCluster(name="Locations").add_to(m)

    # determine severity thresholds from data (quantiles) for meaningful coloring
    try:
        q90 = df["avg_flow"].quantile(0.90)
        q70 = df["avg_flow"].quantile(0.70)
    except Exception:
        q90 = df["avg_flow"].max()
        q70 = df["avg_flow"].median()

    # determine value range for color scaling
    try:
        values = pd.to_numeric(df.get("avg_flow") if "avg_flow" in df.columns else df.get("predicted_flow"), errors="coerce").dropna()
        vmin = float(values.min()) if not values.empty else 0.0
        vmax = float(values.max()) if not values.empty else 1.0
    except Exception:
        vmin, vmax = 0.0, 1.0

    # build linear colormap if available
    if bcm is not None:
        colormap = bcm.LinearColormap(["green", "orange", "red"], vmin=vmin, vmax=vmax)
        colormap.caption = "Avg Flow"
        colormap.add_to(m)

    for _, row in df.iterrows():
        # skip rows with missing location or avg_flow
        location = row.get("location") if "location" in row.index else row.get("location_name")
        avg_flow = row.get("avg_flow") if "avg_flow" in row.index else row.get("predicted_flow")
        if pd.isna(location) or pd.isna(avg_flow):
            continue

        # resolve coordinates: prefer string-key mapping from SUMO edges, else try integer mapping
        coords = None
        # try string mapping (e.g., 'n00_n01')
        try:
            if str(location) in LOCATION_COORDS_STR:
                coords = LOCATION_COORDS_STR[str(location)]
        except Exception:
            coords = None

        # fallback to integer mapping (legacy)
        if coords is None:
            try:
                lid = int(location)
                coords = LOCATION_COORDS.get(lid)
            except Exception:
                coords = None

        # If still no coords, try lat/lon columns from the DataFrame (aggregated CSV fallback)
        if coords is None:
            # try mapping by location_name to POI lat/lon
            try:
                # if the row contains location_name, check LOCATION_POI by name
                name = row.get("location_name") if "location_name" in row.index else None
                if name:
                    for lid, info in LOCATION_POI.items():
                        if info.get("name") == name:
                            coords = (info.get("lat"), info.get("lon"))
                            break
            except Exception:
                pass

        if coords is None:
            if "lat" in df.columns and "lon" in df.columns:
                try:
                    lat = float(row.get("lat"))
                    lon = float(row.get("lon"))
                except Exception:
                    continue
            else:
                continue
        else:
            lat, lon = coords

        # Traffic severity by data quantiles (relative severity)
        if avg_flow >= q90:
            color = "red"
            level = "High"
        elif avg_flow >= q70:
            color = "orange"
            level = "Moderate"
        else:
            color = "green"
            level = "Normal"

        if _HAS_FOLIUM:
            # richer popup: include zone/time_window/max_flow when present
            popup_html = f"<b>{location}</b><br/>Avg Flow: {float(avg_flow):.1f}<br/>Traffic Level: {level}"
            if "zone" in row.index and pd.notna(row.get("zone")):
                popup_html += f"<br/>Zone: {row.get('zone')}"
            if "time_window" in row.index and pd.notna(row.get("time_window")):
                popup_html += f"<br/>Time Window: {row.get('time_window')}"
            if "max_flow" in row.index and pd.notna(row.get("max_flow")):
                popup_html += f"<br/>Max Flow: {row.get('max_flow')}"

            # radius scaling
            try:
                val = float(avg_flow)
                radius = 6 + (20.0 * (val - vmin) / (vmax - vmin + 1e-6))
            except Exception:
                radius = 8

            # color from colormap when available
            try:
                if bcm is not None:
                    color = colormap(float(avg_flow))
                else:
                    color = color
            except Exception:
                pass

            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                popup=folium.Popup(popup_html, max_width=300),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8
            ).add_to(marker_cluster)

    # render map once after adding all markers
    if _HAS_FOLIUM:
        folium.LayerControl().add_to(m)
        st_folium(m, width=1000, height=600)
