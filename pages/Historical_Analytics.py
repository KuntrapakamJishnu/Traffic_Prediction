import streamlit as st
import pandas as pd
from pathlib import Path

# Import psycopg2 lazily and handle if it's not installed
try:
    import psycopg2
except Exception:
    psycopg2 = None

# ==============================
# POSTGRESQL CONFIG (UPDATED)
# ==============================
DB_CONFIG = {
    "dbname": "traffic_db",
    "user": "postgres",
    "password": "postgres123",
    "host": "localhost",
    "port": 5433,
}

# Small mapping of major Dhanbad zones to lat/lon and friendly names
LOCATION_POI = {
    0: {"name": "Dhanbad CBD", "lat": 23.7925, "lon": 86.4350},
    1: {"name": "Baliapur", "lat": 23.7350, "lon": 86.3850},
    2: {"name": "Govindpur", "lat": 23.7450, "lon": 86.4500},
    3: {"name": "IIT-ISM", "lat": 23.8045, "lon": 86.4340},
}

def load_data(limit: int = 1000) -> pd.DataFrame:
    """Load recent prediction rows from the database."""
    # Prefer aggregated CSV (human-friendly location_name) when available
    agg_path = Path("aggregated_traffic_all_new_ok.csv")
    if agg_path.exists():
        try:
            df_agg = pd.read_csv(agg_path)
            # Use raw rows so we preserve time_window (create timestamps for yesterday)
            # CSV expected columns: location_name, zone, time_window, avg_flow, avg_speed, avg_occupy, max_flow
            df_agg = df_agg.rename(columns={"avg_flow": "predicted_flow"})
            # create timestamp: interpret time_window as hour index and set to yesterday
            try:
                today = pd.Timestamp.now().normalize()
                yesterday = today - pd.Timedelta(days=1)
                df_agg["time_window"] = pd.to_numeric(df_agg.get("time_window", 0), errors="coerce").fillna(0).astype(int)
                df_agg["timestamp"] = df_agg["time_window"].apply(lambda h: yesterday + pd.Timedelta(hours=int(h)))
            except Exception:
                df_agg["timestamp"] = pd.Timestamp.now().normalize()

            # fill uncertainty with a small default (not null) and confidence heuristic
            df_agg["uncertainty"] = df_agg.get("uncertainty", 0.0).fillna(0.0) if "uncertainty" in df_agg else 0.0
            def conf(v):
                try:
                    if float(v) > 400:
                        return "HIGH"
                    if float(v) > 250:
                        return "MEDIUM"
                except Exception:
                    pass
                return "LOW"
            df_agg["confidence"] = df_agg.get("predicted_flow", 0).apply(conf)

            # ensure location_name exists and remove duplicate numeric 'location' column
            df_agg["location_name"] = df_agg["location_name"].astype(str)
            # Keep only relevant columns for historical view
            out_cols = [c for c in ["timestamp", "location_name", "predicted_flow", "uncertainty", "confidence"] if c in df_agg.columns]
            return df_agg[out_cols]
        except Exception:
            pass

    # If CSV absent, try DB (resilient to schema variations)
    sql_try = f"""
        SELECT
            timestamp,
            COALESCE(location::text, location_id::text) AS location,
            predicted_flow,
            uncertainty,
            confidence
        FROM traffic_predictions
        ORDER BY timestamp DESC
        LIMIT {limit};
    """

    sql_fallback = f"""
        SELECT
            timestamp,
            location_id::text AS location,
            predicted_flow,
            uncertainty,
            confidence
        FROM traffic_predictions
        ORDER BY timestamp DESC
        LIMIT {limit};
    """

    df = pd.DataFrame()
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        try:
            try:
                df = pd.read_sql(sql_try, conn)
            except Exception:
                df = pd.read_sql(sql_fallback, conn)
        finally:
            try:
                conn.close()
            except Exception:
                pass
    except Exception:
        st.warning("DB unavailable — Historical analytics will be empty unless CSV fallback is provided.")

    # Normalize name
    if not df.empty and "location" in df.columns:
        try:
            df["location_id"] = pd.to_numeric(df["location"], errors="coerce")
            df["location_name"] = df["location_id"].map(lambda x: LOCATION_POI.get(int(x), {}).get("name") if not pd.isna(x) and int(x) in LOCATION_POI else None)
            df["location_name"] = df["location_name"].fillna(df["location"].astype(str))
        except Exception:
            df["location_name"] = df["location"].astype(str)

    return df

# ==============================
# STREAMLIT UI
# ==============================
st.title(" Historical Traffic Analytics")

df = load_data()

if df.empty:
    st.warning("No predictions logged yet. Run the model first.")
else:
    # Ensure timestamp exists and is a datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    else:
        st.error("Data does not contain a 'timestamp' column.")

    st.subheader("Recent Predictions")
    st.dataframe(df, width='stretch')

    st.subheader("Traffic Flow Over Time")
    if "predicted_flow" in df.columns and "timestamp" in df.columns:
        st.line_chart(df.set_index("timestamp")["predicted_flow"])
    else:
        st.info("No predicted_flow data available to plot.")

    st.subheader("Uncertainty Trend")
    if "uncertainty" in df.columns and "timestamp" in df.columns:
        st.line_chart(df.set_index("timestamp")["uncertainty"])
    else:
        st.info("No uncertainty data available to plot.")

    st.subheader("Confidence Distribution")
    if "confidence" in df.columns:
        st.bar_chart(df["confidence"].value_counts())
    else:
        st.info("No confidence data available.")

    st.subheader("High Traffic Locations")
    if "predicted_flow" in df.columns:
        if "location_name" in df.columns:
            avg_flow = (
                df.groupby("location_name", as_index=False)["predicted_flow"]
                .mean()
                .rename(columns={"predicted_flow": "avg_flow"})
            )
            st.dataframe(avg_flow.sort_values("avg_flow", ascending=False), width='stretch')
            st.bar_chart(avg_flow.set_index("location_name")["avg_flow"])
        elif "location_id" in df.columns:
            avg_flow = (
                df.groupby("location_id", as_index=False)["predicted_flow"]
                .mean()
                .rename(columns={"predicted_flow": "avg_flow"})
            )
            st.dataframe(avg_flow.sort_values("avg_flow", ascending=False), width='stretch')
            st.bar_chart(avg_flow.set_index("location_id")["avg_flow"])
        else:
            st.info("No location identifiers available to compute averages.")
    else:
        st.info("Insufficient data to compute average flow per location.")
