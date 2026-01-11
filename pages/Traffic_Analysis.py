import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("Traffic Prediction Analysis")

# -------------------------------
# Simulated ground truth vs prediction
# -------------------------------
np.random.seed(42)
time = pd.date_range("2025-01-01", periods=100, freq="h")

actual = np.random.randint(80, 400, size=100)
predicted = actual + np.random.normal(0, 30, size=100)

df = pd.DataFrame({
    "time": time,
    "actual": actual,
    "predicted": predicted
})

# -------------------------------
# Error metrics
# -------------------------------
df["error"] = df["predicted"] - df["actual"]

mae = np.mean(np.abs(df["error"]))
rmse = np.sqrt(np.mean(df["error"] ** 2))

# -------------------------------
# Metrics
# -------------------------------
c1, c2 = st.columns(2)
c1.metric("MAE (Mean Absolute Error)", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")

# -------------------------------
# Error over time (Streamlit native)
# -------------------------------
st.subheader(" Prediction Error Over Time")
st.line_chart(df.set_index("time")["error"])

# -------------------------------
# Error distribution (no matplotlib)
# -------------------------------
st.subheader("Error Distribution")
st.bar_chart(df["error"].value_counts().sort_index())

st.info(
    "This page evaluates prediction accuracy and temporal error patterns. "
    "Such analysis is essential in ML research to understand model bias, variance, "
    "and failure cases before real-world deployment."
)
