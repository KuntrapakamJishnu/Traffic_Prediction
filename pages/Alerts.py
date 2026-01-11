import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np

# Import psycopg2 lazily
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

# ==============================
# LOCATION MASTER
# ==============================
LOCATION_POI = {
    0: {"name": "Dhanbad CBD"},
    1: {"name": "Baliapur"},
    2: {"name": "Govindpur"},
    3: {"name": "IIT-ISM"},
}

# ==============================
# LOAD DATA (UNCHANGED CORE)
# ==============================
def load_alerts():
    agg_path = Path("aggregated_traffic_all_new_ok.csv")
    if agg_path.exists():
        df = pd.read_csv(agg_path)
        df = df.rename(columns={"avg_flow": "predicted_flow"})
        df["timestamp"] = pd.Timestamp.now()
        df["location_name"] = df["location_name"].astype(str)
        df["location_id"] = df["location_name"]
        return df[["timestamp", "location_id", "location_name", "predicted_flow"]]

    if psycopg2 is None:
        return pd.DataFrame()

    sql = """
        SELECT
            timestamp,
            location_id,
            predicted_flow
        FROM traffic_predictions
        ORDER BY timestamp DESC
        LIMIT 300;
    """

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        df = pd.read_sql(sql, conn)
        conn.close()
    except Exception:
        return pd.DataFrame()

    # Map friendly names
    df["location_id_num"] = pd.to_numeric(df["location_id"], errors="coerce")
    df["location_name"] = df["location_id_num"].map(
        lambda x: LOCATION_POI.get(int(x), {}).get("name") if not pd.isna(x) and int(x) in LOCATION_POI else "Unknown"
    )

    return df[["timestamp", "location_id", "location_name", "predicted_flow"]]

# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(layout="wide")
st.title(" AI-Based Traffic Alerts")

df = load_alerts()

if df.empty:
    st.warning("No traffic data available for alerts.")
    st.stop()

# ==============================
# DATA CLEANING (SAFE)
# ==============================
df["predicted_flow"] = pd.to_numeric(df["predicted_flow"], errors="coerce")
df = df.dropna(subset=["predicted_flow"])

# ==============================
# AI ANOMALY DETECTION (CORE ADDITION)
# ==============================
baseline = (
    df.groupby("location_name")["predicted_flow"]
      .agg(["mean", "std"])
      .reset_index()
      .rename(columns={"mean": "baseline_mean", "std": "baseline_std"})
)

df = df.merge(baseline, on="location_name", how="left")
df["baseline_std"] = df["baseline_std"].replace(0, np.nan).fillna(1)

df["anomaly_score"] = (
    (df["predicted_flow"] - df["baseline_mean"]) / df["baseline_std"]
)

def classify_anomaly(z):
    if z >= 2.5:
        return "SPIKE_ANOMALY"
    elif z <= -2.5:
        return "DROP_ANOMALY"
    else:
        return "NORMAL_BEHAVIOR"

df["ai_alert_type"] = df["anomaly_score"].apply(classify_anomaly)

# Traditional alert level (kept for compatibility)
def traffic_level(v):
    if v > 400:
        return "HIGH TRAFFIC"
    elif v > 250:
        return "MODERATE TRAFFIC"
    return "NORMAL"

df["alert_level"] = df["predicted_flow"].apply(traffic_level)

# ==============================
# SIDEBAR FILTERS
# ==============================
st.sidebar.header("Filters")

locations = sorted(df["location_name"].unique())
selected_locations = st.sidebar.multiselect(
    "Locations",
    locations,
    default=locations
)

ai_alerts = ["SPIKE_ANOMALY", "DROP_ANOMALY", "NORMAL_BEHAVIOR"]
selected_ai_alerts = st.sidebar.multiselect(
    "AI Alert Type",
    ai_alerts,
    default=ai_alerts
)

df_filtered = df[
    (df["location_name"].isin(selected_locations)) &
    (df["ai_alert_type"].isin(selected_ai_alerts))
]

# ==============================
# KPI SUMMARY
# ==============================
st.subheader("Alert Summary")

c1, c2, c3 = st.columns(3)
c1.metric("Total Records", len(df_filtered))
c2.metric("Spike Anomalies", (df_filtered["ai_alert_type"] == "SPIKE_ANOMALY").sum())
c3.metric("Drop Anomalies", (df_filtered["ai_alert_type"] == "DROP_ANOMALY").sum())

st.markdown("---")

# ==============================
# AI ALERTS PANEL (NEW & IMPRESSIVE)
# ==============================
st.subheader(" AI-Detected Traffic Anomalies")

for _, r in df_filtered.iterrows():
    if r["ai_alert_type"] == "SPIKE_ANOMALY":
        st.error(
            f"Traffic spike in {r['location_name']} "
            f"(Flow={int(r['predicted_flow'])}, z={r['anomaly_score']:.2f})"
        )
    elif r["ai_alert_type"] == "DROP_ANOMALY":
        st.warning(
            f" Sudden traffic drop in {r['location_name']} "
            f"(Flow={int(r['predicted_flow'])}, z={r['anomaly_score']:.2f})"
        )

# ==============================
# ALERT TABLE
# ==============================
st.subheader(" Alert History")

st.dataframe(
    df_filtered[
        [
            "timestamp",
            "location_name",
            "predicted_flow",
            "baseline_mean",
            "anomaly_score",
            "ai_alert_type",
            "alert_level",
        ]
    ].sort_values("timestamp", ascending=False),
    width="stretch"
)

# ==============================
# DISTRIBUTION
# ==============================
st.subheader(" AI Alert Distribution")
st.bar_chart(df_filtered["ai_alert_type"].value_counts())

# ==============================
# LOCATION-WISE ANOMALIES
# ==============================
st.subheader(" Anomalies per Location")
st.bar_chart(
    df_filtered.groupby("location_name")["ai_alert_type"].count()
)
