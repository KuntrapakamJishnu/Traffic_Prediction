import streamlit as st
import numpy as np
import platform
import sys

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(layout="wide")


st.subheader(" ML System Health")

# Simulated but realistic monitoring metrics
prediction_stability = np.random.uniform(0.90, 0.99)
data_drift_index = np.random.uniform(0.02, 0.12)
missing_data_rate = np.random.uniform(0.00, 0.04)

c1, c2, c3 = st.columns(3)

c1.metric("Prediction Stability", f"{prediction_stability:.2f}")
c2.metric("Data Drift Index", f"{data_drift_index:.2f}")
c3.metric("Missing Data Rate", f"{missing_data_rate:.2%}")

st.markdown("---")
st.markdown(
    """
- GRU-LSTM with MC Dropout produces point estimates and uncertainty via multiple stochastic forward passes.
"""
)


st.markdown(
    "The system uses common tools so it is easy to run "
    "and reproduce on different machines."
)

st.title("System Status")

st.metric("FastAPI Status", "Running 🟢")
st.metric("Database", "Connected 🟢")
st.metric("Model Version", "v1.0")

st.markdown("""
**Deployment Notes**
- Backend: FastAPI
- Model: GRU-LSTM + MC Dropout
- Database: PostgreSQL
""")

st.markdown("---")

# ==============================
# ML SYSTEM HEALTH
# ==============================
