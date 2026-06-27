import streamlit as st
import requests
import pandas as pd
import numpy as np
from requests.exceptions import ConnectionError, Timeout
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

from location_catalog import normalize_location_label


st.set_page_config(
    page_title="Traffic Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)
bg_img = get_base64_image("maxresdefault.jpg")
st.markdown(
f"""
<style>

.stApp{{
    background:
        linear-gradient(rgba(5,9,20,0.82), rgba(5,9,20,0.82)),
        url("data:image/jpg;base64,{bg_img}");

    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

.hero {{
    padding: 2.1rem 2rem 1.8rem 2rem;
    border-radius: 28px;
    border: 1px solid rgba(255,255,255,0.08);
    background: linear-gradient(
        135deg,
        rgba(17,24,39,0.95),
        rgba(15,23,42,0.72)
    );
    box-shadow: 0 24px 70px rgba(0,0,0,0.35);
    margin-bottom: 1.2rem;
}}

</style>
""",
unsafe_allow_html=True
)

API_URL = "http://127.0.0.1:8000/predict"


def _safe_load_locations() -> list[str]:
    try:
        df_locations = pd.read_csv("aggregated_traffic_all_new_ok.csv")
        if "location_name" in df_locations.columns:
            values = df_locations["location_name"].dropna().astype(str).map(normalize_location_label)
            return sorted(values.unique().tolist())
    except Exception:
        pass
    return [f"Location {i}" for i in range(1, 11)]


st.markdown(
    """
<div class="hero">
  <div class="eyebrow">Executive traffic intelligence</div>
  <h1>Traffic Prediction Platform</h1>
  <p>
    A presentation-ready command center for live congestion prediction, hotspot review,
    uncertainty monitoring, and location-aware traffic planning across Amaravati and Vijayawada.
  </p>
  <div class="pill-row">
    <span class="pill">Real-time predictions</span>
    <span class="pill">Historical analytics</span>
    <span class="pill">Vijayawada street coverage</span>
    <span class="pill">VIT-AP campus coverage</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

backend_online = False
try:
    requests.get("http://127.0.0.1:8000/", timeout=1)
    backend_online = True
except Exception:
    st.sidebar.error("Backend API Offline (Port 8000)")

st.sidebar.header("Location")
location_value = st.sidebar.selectbox("Select Location", options=_safe_load_locations())

st.sidebar.markdown("---")
st.sidebar.markdown("### Presentation Mode")
st.sidebar.info(
    "Use the main dashboard and the live monitor page to demonstrate model behavior, uncertainty, and location-level congestion."
)

st.sidebar.markdown("### Normalized Traffic Inputs")
flow = st.sidebar.slider("Flow Level", 0.0, 1.0, 0.3)
occupancy = st.sidebar.slider("Occupancy Level", 0.0, 1.0, 0.05)
speed = st.sidebar.slider("Speed Level", 0.0, 1.0, 0.6)

sequence = [[flow, occupancy, speed] for _ in range(12)]
run_prediction = st.sidebar.button("Run Live Prediction")

top_a, top_b, top_c, top_d = st.columns(4)
top_a.metric("Backend", "Live" if backend_online else "Offline", "FastAPI service")
top_b.metric("Coverage", "Amaravati + Vijayawada", "Expanded names")
top_c.metric("Model Mode", "GRU-LSTM", "MC Dropout")
top_d.metric("UI Mode", "Presentation", "High-contrast theme")

if run_prediction:
    st.subheader("Live Model Inference")

    payload = {
        "sequence": sequence,
        "location": str(location_value),
    }

    result = None
    used_backend = False

    try:
        with st.spinner("Contacting Model Server..."):
            response = requests.post(API_URL, json=payload, timeout=15)
            response.raise_for_status()
            result = response.json()
            used_backend = True
            st.success("Real-time prediction received.")
    except (ConnectionError, Timeout) as e:
        st.warning("Prediction service unreachable (Timeout/Connection) — using demo inference.")
        st.caption(f"Debug Info: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

    if not used_backend:
        mean_flow = np.mean([s[0] for s in sequence])
        predicted_flow_demo = mean_flow * 500
        uncertainty_demo = max(0.01, 0.05 + 0.1 * np.std([s[0] for s in sequence]))

        result = {
            "predicted_flow": predicted_flow_demo,
            "uncertainty": uncertainty_demo,
            "confidence": "HIGH" if uncertainty_demo < 0.07 else "MEDIUM",
        }

    predicted_flow = float(result.get("predicted_flow", 0))
    uncertainty = float(result.get("uncertainty", 0))
    confidence = result.get("confidence", "N/A")

    st.subheader("Prediction Summary")
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

    st.subheader("Traffic Condition Interpretation")
    if predicted_flow > 400:
        st.error(f"Severe congestion predicted at **{location_value}**")
        st.info("Action: Deploy traffic wardens and reroute vehicles via secondary arteries.")
    elif predicted_flow > 250:
        st.warning(f"Moderate congestion predicted at **{location_value}**")
        st.info("Action: Adjust signal timings to favor high-flow directions.")
    else:
        st.success(f"Traffic normal at **{location_value}**")
        st.info("Action: No immediate intervention required.")

    st.subheader("Input Feature Evolution (Last 12 Steps)")
    df_input = pd.DataFrame(sequence, columns=["Flow", "Occupancy", "Speed"])
    st.line_chart(df_input)

st.markdown("---")
st.header("Network-Level Traffic Overview")
try:
    df_agg = pd.read_csv("aggregated_traffic_all_new_ok.csv")

    location_data = (
        df_agg.groupby("location_name")
        .agg(
            Average_Flow=("avg_flow", "mean"),
            Maximum_Flow=("avg_flow", "max")
        )
        .reset_index()
        .rename(columns={"location_name": "Location"})
    )

    location_data = location_data.sort_values(
        "Average_Flow",
        ascending=False
    )

    location_data["Rank"] = range(1, len(location_data) + 1)

    location_data["Traffic Status"] = location_data["Average_Flow"].apply(
        lambda x:
        "🔴 Heavy" if x > 400
        else "🟠 Moderate" if x > 250
        else "🟢 Smooth"
    )

    location_data = location_data[
        [
            "Rank",
            "Location",
            "Average_Flow",
            "Maximum_Flow",
            "Traffic Status"
        ]
    ]

    location_data = location_data.round(2)

    st.subheader("Location Ranking")

    col1, col2, col3 = st.columns([0.25, 4, 0.25])

    with col2:
        st.dataframe(
            location_data.head(12),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("Traffic Flow Review by Location")

    st.bar_chart(
        location_data.set_index("Location")[["Average_Flow"]],
        use_container_width=True
    )

    summary_a, summary_b, summary_c = st.columns(3)

    summary_a.metric(
        "Tracked Locations",
        location_data.shape[0]
    )

    summary_b.metric(
        "Top Flow",
        f"{location_data['Average_Flow'].max():.1f}"
    )

    summary_c.metric(
        "Median Flow",
        f"{location_data['Average_Flow'].median():.1f}"
    )

except Exception:
    st.info(
        "Upload 'aggregated_traffic_all_new_ok.csv' to view the automated traffic overview."
    )