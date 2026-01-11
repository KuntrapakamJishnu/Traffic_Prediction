import streamlit as st
import pandas as pd
import numpy as np

try:
    from sklearn.cluster import DBSCAN
except Exception:
    DBSCAN = None


def assign_zones(df):
    """Assign zone ids using DBSCAN. If sklearn is not available or data is empty,
    return the DataFrame with a `zone_id` column (default -1).
    """
    if df is None:
        return df

    # don't create a `zone_id` column yet to avoid merge suffix collisions;
    # we'll populate a final `zone_id` at the end

    if DBSCAN is None:
        return df

    if df.empty:
        return df

    if not {"lat", "lon"}.issubset(df.columns):
        return df

    cleaned = df[["lat", "lon"]].dropna().copy()
    if cleaned.empty:
        return df

    coords = cleaned.values
    db = DBSCAN(eps=0.01, min_samples=5).fit(coords)
    cleaned["zone_id"] = db.labels_

    # Merge computed zone ids under a temporary name to avoid suffix conflicts
    temp = cleaned["zone_id"].rename("zone_id_new")
    df = df.merge(temp, left_index=True, right_index=True, how="left")

    # Build final `zone_id`: prefer computed values, fall back to existing or -1
    if "zone_id" in df.columns:
        base = df["zone_id"]
    else:
        base = pd.Series(-1, index=df.index)

    df["zone_id"] = df.get("zone_id_new", base).fillna(-1).astype(int)
    # drop temporary column if present
    if "zone_id_new" in df.columns:
        df = df.drop(columns=["zone_id_new"])

    return df


# -----------------------------
# Streamlit UI for zone clustering
# -----------------------------
st.title(" Zone Clustering")

st.markdown("This page clusters coordinate data into zones using DBSCAN. If no database is available, sample data is used.")

# Try to import psycopg2 to decide whether to use DB
try:
    import psycopg2
except Exception:
    psycopg2 = None

use_db = False
if psycopg2 is not None:
    use_db = st.checkbox("Load real data from DB (requires configured DB)", value=False)
else:
    st.info("psycopg2 not available — using simulated sample data.")

if use_db:
    # attempt to load from DB if user opts in
    DB_CONFIG = {
        "dbname": "traffic_db",
        "user": "postgres",
        "password": "postgres123",
        "host": "localhost",
        "port": 5433,
    }
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        query = "SELECT lat, lon FROM traffic_predictions WHERE lat IS NOT NULL AND lon IS NOT NULL LIMIT 1000;"
        df = pd.read_sql(query, conn)
        conn.close()
        if df.empty:
            # try to fallback to cube CSV + coords if available
            try:
                import os, json
                csvp = "cube_layout/cube_city_traffic.csv"
                coordsp = "cube_layout/location_coords.json"
                if os.path.exists(csvp):
                    df_rt = pd.read_csv(csvp)
                    # take latest per location and map to coords when available
                    coords = {}
                    if os.path.exists(coordsp):
                        with open(coordsp, 'r', encoding='utf-8') as fh:
                            coords = json.load(fh)

                    df_latest = df_rt.sort_values(["location", "timestep"]).groupby("location").tail(1)
                    lat_col = []
                    lon_col = []
                    for loc in df_latest["location"]:
                        if str(loc) in coords:
                            lat_col.append(coords[str(loc)][0])
                            lon_col.append(coords[str(loc)][1])
                        else:
                            lat_col.append(None)
                            lon_col.append(None)

                    df = df_latest.assign(lat=lat_col, lon=lon_col)[["lat", "lon"]].dropna()
                    if not df.empty:
                        use_db = True
                if not use_db:
                    st.warning("Database returned no coordinate rows — falling back to sample data for preview.")
                    use_db = False
            except Exception:
                st.warning("Database returned no coordinate rows — falling back to sample data for preview.")
                use_db = False
    except Exception as e:
        st.warning(f"Failed to load DB data: {e}. Falling back to sample data.")
        use_db = False

if not use_db:
    n = st.slider("Sample points", 10, 1000, 200)
    # sample points roughly across India lat/lon ranges
    lats = np.random.uniform(8.0, 29.0, n)
    lons = np.random.uniform(68.0, 97.0, n)
    df = pd.DataFrame({"lat": lats, "lon": lons})

st.subheader("Raw Points")
st.dataframe(df.head(200))

eps = st.number_input("DBSCAN eps (degrees)", value=0.01, format="%.5f")
min_samples = st.number_input("DBSCAN min_samples", value=5, min_value=1)

if st.button("Run Clustering"):
    out = assign_zones(df.copy())
    st.subheader("Clustered Points (zone_id column)")
    st.dataframe(out.head(200))
    # Map preview removed per user request
