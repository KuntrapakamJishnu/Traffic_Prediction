import streamlit as st
import requests
import pandas as pd
import numpy as np
from requests.exceptions import ConnectionError, Timeout

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Traffic Intelligence Platform",
    layout="wide"
)

# If your backend is on a different port or machine, update this URL
API_URL = "http://127.0.0.1:8000/predict"

# =========================
# HEADER
# =========================
st.markdown(
    """
<div style="background:linear-gradient(90deg,#1e293b, #0f172a);padding:18px;border-radius:10px;margin-bottom:12px;">
  <h1 style="color:#ffffff;text-align:center;margin:0;">Traffic Prediction Platform</h1>
  <p style="color:rgba(255,255,255,0.9);text-align:center;margin:4px 0 0 0;">
    Understanding urban traffic with real-time prediction and actionable recommendations
  </p>
</div>
""",
    unsafe_allow_html=True,
)

# =========================
# BACKEND HEALTH CHECK
# =========================
backend_online = False
try:
    # Quick ping to see if server is even alive
    requests.get("http://127.0.0.1:8000/", timeout=1)
    backend_online = True
except:
    st.sidebar.error(" Backend API Offline (Port 8000)")

# =========================
# SIDEBAR – LOCATION
# =========================
st.sidebar.header(" Location")

location_value = None
try:
    df_locations = pd.read_csv("aggregated_traffic_all_new_ok.csv")
    loc_options = sorted(df_locations["location_name"].dropna().unique().tolist())
    location_value = st.sidebar.selectbox("Select Location", options=loc_options)
except Exception:
    location_value = st.sidebar.selectbox(
        "Select Location",
        options=[f"Location {i}" for i in range(1, 11)]
    )

# =========================
# INPUT FEATURES
# =========================
st.sidebar.markdown("###  Normalized Traffic Inputs")
flow = st.sidebar.slider("Flow Level", 0.0, 1.0, 0.3)
occupancy = st.sidebar.slider("Occupancy Level", 0.0, 1.0, 0.05)
speed = st.sidebar.slider("Speed Level", 0.0, 1.0, 0.6)

# Create sequence for GRU/LSTM (12 time steps)
sequence = [[flow, occupancy, speed] for _ in range(12)]

run_prediction = st.sidebar.button(" Run Live Prediction")

# =========================
# LIVE PREDICTION LOGIC
# =========================
if run_prediction:
    st.subheader("Live Model Inference")
    
    payload = {
        "sequence": sequence,
        "location": str(location_value)
    }

    result = None
    used_backend = False

    try:
        # INCREASED TIMEOUT to 15 seconds to handle heavy model processing
        with st.spinner('Contacting Model Server...'):
            response = requests.post(API_URL, json=payload, timeout=15)
            response.raise_for_status()
            result = response.json()
            used_backend = True
            st.success("Real-time prediction received.")

    except (ConnectionError, Timeout) as e:
        st.warning("Prediction service unreachable (Timeout/Connection) — using demo inference.")
        st.caption(f"**Debug Info:** {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

    # Fallback to Demo Data if Backend Fails
    if not used_backend:
        mean_flow = np.mean([s[0] for s in sequence])
        predicted_flow_demo = mean_flow * 500
        uncertainty_demo = max(0.01, 0.05 + 0.1 * np.std([s[0] for s in sequence]))
        
        result = {
            "predicted_flow": predicted_flow_demo,
            "uncertainty": uncertainty_demo,
            "confidence": "HIGH" if uncertainty_demo < 0.07 else "MEDIUM"
        }

    # Display results
    predicted_flow = float(result.get("predicted_flow", 0))
    uncertainty = float(result.get("uncertainty", 0))
    confidence = result.get("confidence", "N/A")

    # =========================
    # KPI SECTION
    # =========================
    st.subheader("Prediction Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Traffic Flow", f"{predicted_flow:.1f}")
    c2.metric("Model Uncertainty", f"{uncertainty:.4f}")

    if confidence == "HIGH":
        c3.success(" HIGH CONFIDENCE")
    elif confidence == "MEDIUM":
        c3.warning(" MEDIUM CONFIDENCE")
    else:
        c3.error(" LOW CONFIDENCE")

    st.markdown("---")

    # =========================
    # INTERPRETATION
    # =========================
    st.subheader(" Traffic Condition Interpretation")
    if predicted_flow > 400:
        st.error(f"Severe congestion predicted at **{location_value}**")
        st.info("**Action:** Deploy traffic wardens and reroute vehicles via secondary arteries.")
    elif predicted_flow > 250:
        st.warning(f"Moderate congestion predicted at **{location_value}**")
        st.info("**Action:** Adjust signal timings to favor high-flow directions.")
    else:
        st.success(f"Traffic normal at **{location_value}**")
        st.info("**Action:** No immediate intervention required.")

    # =========================
    # INPUT EVOLUTION
    # =========================
    st.subheader(" Input Feature Evolution (Last 12 Steps)")
    df_input = pd.DataFrame(sequence, columns=["Flow", "Occupancy", "Speed"])
    st.line_chart(df_input)

# =========================
# NETWORK-LEVEL ANALYTICS
# =========================
# =========================
# NETWORK-LEVEL ANALYTICS
# =========================
st.markdown("---")
# Use a clear heading for the entire section
st.header(" Network-Level Traffic Overview")

try:
    # Attempt to load the dataset
    df_agg = pd.read_csv("aggregated_traffic_all_new_ok.csv")
    
    # Process data for the bar chart
    location_data = (
        df_agg.groupby("location_name")["avg_flow"]
        .mean()
        .reset_index()
        .rename(columns={"location_name": "Location", "avg_flow": "Average Flow"})
        .sort_values("Average Flow", ascending=False)
    )

    # Display the table first for reference
    st.dataframe(location_data, use_container_width=True)

    # ADD TITLE HERE: This places the title directly above your bar graph
    st.subheader("Traffic Flow Review by Location")
    st.bar_chart(location_data.set_index("Location"))

except Exception as e:
    # Fallback if the CSV is missing or columns don't match
    st.info("Upload 'aggregated_traffic_all_new_ok.csv' to view the automated traffic overview.")
    # st.write(f"Error details: {e}") # Uncomment for debugging