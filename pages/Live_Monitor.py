import os
import time
import streamlit as st
import requests
import numpy as np
from requests.exceptions import RequestException

DEFAULT_API_URL = os.getenv("LIVE_API_URL", "http://127.0.0.1:8000/predict")
DEFAULT_TIMEOUT = int(os.getenv("LIVE_API_TIMEOUT", "15"))
DEFAULT_RETRIES = int(os.getenv("LIVE_API_RETRIES", "3"))

def _health_url(api_url: str) -> str:
    if api_url.endswith('/predict'):
        return api_url.replace('/predict', '/openapi.json')
    base = api_url.rsplit('/', 1)[0]
    return base + '/openapi.json'

def check_backend(api_url: str, timeout: int, retries: int) -> (bool, str):
    url = _health_url(api_url)
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return True, "Backend reachable (OpenAPI found)."
            return False, f"Backend reachable but returned status {r.status_code}."
        except RequestException as e:
            if attempt < retries:
                time.sleep(1 * attempt)
                continue
            return False, str(e)

def send_prediction(api_url: str, payload: dict, timeout: int, retries: int):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            res = requests.post(api_url, json=payload, timeout=timeout)
            res.raise_for_status()
            return res
        except RequestException as e:
            last_exc = e
            if attempt < retries:
                time.sleep(1 * attempt)
                continue
            raise


st.title("Live Traffic Monitoring")

st.sidebar.header("Live Input Controls")

# Allow operator to override the API URL / timeout / retries from the UI
api_url = st.sidebar.text_input("Backend API URL", value=DEFAULT_API_URL)
timeout = st.sidebar.number_input("Request timeout (s)", min_value=1, value=DEFAULT_TIMEOUT)
retries = st.sidebar.number_input("Retry attempts", min_value=1, value=DEFAULT_RETRIES)

location = st.sidebar.text_input("Location (edge name)", value="n00_n01")
flow = st.sidebar.slider("Flow (scaled)", 0.0, 1.0, 0.3)
occupy = st.sidebar.slider("Occupancy (scaled)", 0.0, 1.0, 0.05)
speed = st.sidebar.slider("Speed (scaled)", 0.0, 1.0, 0.6)

sequence = [[flow, occupy, speed] for _ in range(12)]

uvicorn_cmd = "uvicorn app:app --reload --port 8000"

if st.button("Check backend"):
    with st.spinner("Checking backend..."):
        ok, msg = check_backend(api_url, timeout, retries)
    if ok:
        st.success(msg + " You can run live predictions.")
    else:
        st.error(f"Backend not reachable: {msg}")
        st.info(f"Start the backend locally with: {uvicorn_cmd}")

if st.button("Run Live Prediction"):
    payload = {"sequence": sequence, "location": str(location)}
    try:
        with st.spinner("Calling backend for prediction..."):
            res = send_prediction(api_url, payload, timeout, retries)
    except RequestException as e:
        st.error(f"Failed to call backend: {e}. Ensure the backend at {api_url} is running and reachable.")
        st.info(f"Start the backend locally with: {uvicorn_cmd}")
    else:
        try:
            data = res.json()
        except Exception:
            st.error("Backend returned invalid JSON.")
        else:
            c1, c2, c3 = st.columns(3)

            c1.metric("Predicted Flow", f"{data.get('predicted_flow', float('nan')):.1f}")
            c2.metric("Uncertainty", f"{data.get('uncertainty', float('nan')):.4f}")

            conf = data.get('confidence', '').upper()
            if conf == "HIGH":
                c3.success("HIGH CONFIDENCE")
            elif conf == "MEDIUM":
                c3.warning("MEDIUM CONFIDENCE")
            else:
                c3.error("LOW CONFIDENCE")

            st.subheader("Input Feature Snapshot")
            st.line_chart(np.array(sequence))
