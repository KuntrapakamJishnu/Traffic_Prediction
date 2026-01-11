import streamlit as st
import requests
import pandas as pd
import numpy as np

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Traffic Intelligence Platform",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000/predict"

st.markdown(
    """
<div style="background:linear-gradient(90deg,#0f172a);padding:18px;border-radius:10px;margin-bottom:12px;">
  <h1 style="color:#ffffff;text-align:center;margin:0;font-family:Segoe UI, Roboto, sans-serif;"> Traffic Prediction Platform</h1>
  <p style="color:rgba(255,255,255,0.95);text-align:center;margin:4px 0 0 0;font-size:14px;">Understanding urban traffic with real-time prediction and actionable recommendations</p>
</div>
""",
    unsafe_allow_html=True,
)


# =========================
# SIDEBAR – INPUT CONTROLS
# =========================
st.sidebar.header(" Location")

# Prefer real location names if CSV exists
use_text_location = False
location_name = None
location_id = None

try:
    df_locations = pd.read_csv("aggregated_traffic_all_new_ok.csv")
    loc_options = sorted(df_locations["location_name"].dropna().unique().tolist())
    if not loc_options:
        loc_options = ["Unknown"]
    # Provide an explicit key so Streamlit session-state is initialized predictably
    location_name = st.sidebar.selectbox("Select Location", options=loc_options, key="location_name")
    use_text_location = True
except Exception:
    # Fallback to numeric location ids; give the widget a stable key as well
    location_id = st.sidebar.selectbox(
        "Select Location",
        options=list(range(1, 11)),
        format_func=lambda x: f"Location {x}",
        key="location_id"
    )

st.sidebar.markdown("###  Normalized Traffic Inputs")

flow = st.sidebar.slider("Flow Level", 0.0, 1.0, 0.3)
occupancy = st.sidebar.slider("Occupancy Level", 0.0, 1.0, 0.05)
speed = st.sidebar.slider("Speed Level", 0.0, 1.0, 0.6)

# Temporal sequence (model input)
sequence = [[flow, occupancy, speed] for _ in range(12)]

run_prediction = st.sidebar.button(" Run Live Prediction")

# =========================
# LIVE PREDICTION
# =========================
if run_prediction:
    st.subheader(" Live Model Inference")

    payload = {"sequence": sequence}
    if use_text_location:
        payload["location"] = location_name
        location_label = location_name
    else:
        payload["location_id"] = int(location_id)
        location_label = f"Location {location_id}"

    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()
    except Exception:
        # Graceful fallback: use a local demo prediction when backend is unreachable
        st.warning("Unable to reach prediction service — using local demo prediction.")
        mean_flow = float(np.mean([s[0] for s in sequence]))
        # scale normalized flow [0,1] to a realistic range (0-500)
        predicted_flow_demo = mean_flow * 500.0
        uncertainty_demo = float(max(0.001, 0.05 + 0.1 * np.std([s[0] for s in sequence])))
        if uncertainty_demo > 0.15:
            confidence_demo = "LOW"
        elif uncertainty_demo > 0.07:
            confidence_demo = "MEDIUM"
        else:
            confidence_demo = "HIGH"
        result = {
            "predicted_flow": predicted_flow_demo,
            "uncertainty": uncertainty_demo,
            "confidence": confidence_demo,
        }

    predicted_flow = float(result.get("predicted_flow", 0))
    uncertainty = float(result.get("uncertainty", 0))
    confidence = result.get("confidence", "UNKNOWN")

    # =========================
    # KPI SECTION
    # =========================
    st.subheader(" Prediction Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric("Predicted Traffic Flow", f"{predicted_flow:.1f}")
    c2.metric("Model Uncertainty", f"{uncertainty:.4f}")

    if confidence == "HIGH":
        c3.success("HIGH CONFIDENCE")
    elif confidence == "MEDIUM":
        c3.warning("MEDIUM CONFIDENCE")
    else:
        c3.error("LOW CONFIDENCE")

    st.markdown("---")

    # =========================
    # DECISION INTERPRETATION
    # =========================
    st.subheader(" Traffic Condition Interpretation")

    if predicted_flow > 400:
        st.error(f" **Severe Congestion Detected** at **{location_label}**")
        st.markdown("Recommended Action: reroute traffic, adjust signals, issue alerts.")
    elif predicted_flow > 250:
        st.warning(f" **Moderate Congestion** at **{location_label}**")
        st.markdown("Recommended Action: monitor closely, prepare mitigation.")
    else:
        st.success(f" **Traffic Normal** at **{location_label}**")
        st.markdown("Recommended Action: no intervention required.")

    # =========================
    # INPUT EXPLANATION
    # =========================
    st.subheader(" Input Feature Evolution")

    df_input = pd.DataFrame(
        sequence,
        columns=["Flow", "Occupancy", "Speed"]
    )

    st.line_chart(df_input)

    st.caption(
        "The model consumes a short temporal sequence to capture recent traffic dynamics."
    )

# =========================
# SYSTEM-WIDE ANALYTICS
# =========================
st.markdown("---")
st.subheader(" Network-Level Traffic Overview")

try:
    df_agg = pd.read_csv("aggregated_traffic_all_new_ok.csv")
    location_data = (
        df_agg.groupby("location_name")["avg_flow"]
        .mean()
        .reset_index()
        .rename(columns={"location_name": "Location", "avg_flow": "Average Flow"})
        .sort_values("Average Flow", ascending=False)
    )
except Exception:
    location_data = pd.DataFrame({
        "Location": [f"Location {i}" for i in range(1, 11)],
        "Average Flow": np.random.randint(150, 500, 10)
    }).sort_values("Average Flow", ascending=False)

st.dataframe(location_data, width="stretch")

st.subheader(" Average Traffic by Location")
st.bar_chart(location_data.set_index("Location"))

