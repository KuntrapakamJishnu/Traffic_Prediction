import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title(" Model Explainability")

# -------------------------------
# Fake feature importance (replace later)
# -------------------------------
features = [
    "Previous Traffic Flow",
    "Time of Day",
    "Day of Week",
    "Weather Index",
    "Road Capacity"
]

importance = np.array([0.42, 0.25, 0.15, 0.10, 0.08])

df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values("Importance", ascending=False)

st.subheader(" Feature Importance")
st.bar_chart(df.set_index("Feature"))

st.markdown("""
###  Interpretation
- **Previous Traffic Flow** dominates predictions
- **Time of Day** strongly influences congestion
- External factors (weather) have secondary impact

This improves **trust and transparency** in AI systems.
""")
