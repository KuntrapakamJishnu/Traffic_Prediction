import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import tensorflow as tf
from datetime import datetime

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Traffic Prediction Dashboard",
    layout="wide"
)

st.title("🚦 AI-Based Traffic Prediction Dashboard")
st.caption("Streamlit-only deployment | Model-driven inference")

# -------------------------------
# LOAD MODEL (CACHED)
# -------------------------------
@st.cache_resource
def load_model_and_scaler():
    model = tf.keras.models.load_model("model/gru_lstm_model.h5")
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

# -------------------------------
# LOAD DATA FILES
# -------------------------------
@st.cache_data
def load_data():
    traffic_df = pd.read_csv("campus_traffic.csv")
    with open("location_coords.json", "r") as f:
        location_coords = json.load(f)
    return traffic_df, location_coords

traffic_df, location_coords = load_data()

# -------------------------------
# SIDEBAR CONTROLS
# -------------------------------
st.sidebar.header("⚙️ Prediction Controls")

location_id = st.sidebar.selectbox(
    "Select Road / Location ID",
    sorted(traffic_df["location"].unique())
)

hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
day_of_week = st.sidebar.slider("Day of Week (0=Mon)", 0, 6, 2)

# -------------------------------
# FEATURE PREPARATION
# -------------------------------
def prepare_features(location, hour, day):
    """
    Modify this feature vector if your model
    expects a different input format.
    """
    base_flow = traffic_df[traffic_df["location"] == location]["flow"].mean()
    base_flow = 0 if np.isnan(base_flow) else base_flow

    return [base_flow, hour, day]

# -------------------------------
# PREDICTION FUNCTION
# -------------------------------
def predict_traffic(features):
    features_scaled = scaler.transform([features])
    features_scaled = np.expand_dims(features_scaled, axis=2)
    prediction = model.predict(features_scaled, verbose=0)
    return float(prediction[0][0])

# -------------------------------
# MAIN DASHBOARD
# -------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Location Details")

    st.write(f"**Location ID:** `{location_id}`")

    if location_id in location_coords:
        st.json(location_coords[location_id])
    else:
        st.warning("Coordinates not available for this location")

with col2:
    st.subheader("🔮 Traffic Prediction")

    features = prepare_features(location_id, hour, day_of_week)
    predicted_flow = predict_traffic(features)

    st.metric(
        label="Predicted Traffic Flow",
        value=f"{predicted_flow:.2f}"
    )

    traffic_level = (
        "LOW" if predicted_flow < 30 else
        "MODERATE" if predicted_flow < 70 else
        "HIGH"
    )

    st.write(f"**Traffic Level:** `{traffic_level}`")

# -------------------------------
# HISTORICAL DATA VIEW
# -------------------------------
st.divider()
st.subheader("📈 Historical Traffic Snapshot")

filtered_df = traffic_df[traffic_df["location"] == location_id]

st.dataframe(
    filtered_df.tail(20),
    use_container_width=True
)

# -------------------------------
# FOOTER
# -------------------------------
st.divider()
st.caption(
    f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
    "Deployed on Streamlit Cloud"
)
