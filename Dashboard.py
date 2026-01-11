import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("✅ Streamlit Cloud is working")

df = pd.DataFrame({
    "x": np.arange(5),
    "y": np.arange(5) ** 2
})

st.write("Platform test successful.")
st.dataframe(df)
