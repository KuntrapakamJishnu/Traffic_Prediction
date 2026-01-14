# 🚦 Traffic Prediction Platform

An **industry-grade, research-ready, end-to-end AI Traffic Prediction & Analytics System**.

This single-file `README.md` contains the full project description plus **all example code blocks** (backend, dashboard, training, scripts, SQL, Docker) inside fenced Markdown code blocks so you can copy-paste files directly from this document.

---

## 📖 Executive Summary

Urban traffic management requires accurate, explainable, and reliable forecasts. This repo delivers a production-minded ML system that bridges research and deployment.

## 🎯 Objectives

* Predict short-horizon traffic flows (minutes → hours)
* Provide per-prediction uncertainty and confidence
* Persist predictions for analytics and alerting
* Provide a Streamlit dashboard for stakeholders

---

## 🏗️ System Architecture

```
[ Traffic Sensors / Historical Data ]
                ↓
        Data Preprocessing
                ↓
      Hybrid GRU + LSTM Model
      (MC Dropout inference)
                ↓
         FastAPI Backend
                ↓
            PostgreSQL
                ↓
       Streamlit Dashboard
```

---

## 📂 Repository Single-file Preview (copy sections into files as needed)

The sections below include ready-to-use code blocks. Each code block is intended to be saved into the filename shown in its header comment.

---

### `requirements.txt`

```text
fastapi==0.95.2
uvicorn[standard]==0.21.1
tensorflow==2.12.0
numpy==1.25.0
pandas==2.1.0
scikit-learn==1.2.2
joblib==1.3.2
psycopg2-binary==2.9.7
sqlalchemy==2.1.0
streamlit==1.24.0
python-dotenv==1.1.0
matplotlib==3.7.2
```

---

### `app.py`  (FastAPI backend)

```python
# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import os
import datetime
import psycopg2
from psycopg2.extras import execute_values
import tensorflow as tf

# Config via env
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://traffic_user:secure_password@localhost:5432/traffic_db')
MODEL_PATH = os.getenv('MODEL_PATH', './model/gru_lstm_model.h5')
SCALER_PATH = os.getenv('SCALER_PATH', './model/scaler.pkl')
MC_RUNS = int(os.getenv('MC_RUNS', '50'))

# Load artifacts
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

app = FastAPI(title="Traffic Prediction API")

class PredictRequest(BaseModel):
    location: str
    sequence: list[float]
    timestamp: str | None = None

class PredictResponse(BaseModel):
    predicted_flow: float
    uncertainty: float
    confidence: float
    model_name: str
    timestamp: str


def mc_predict(model, x, runs=50):
    # Ensure dropout is active during inference via `training=True`
    preds = []
    for _ in range(runs):
        preds.append(model(x, training=True).numpy())
    arr = np.concatenate(preds, axis=0)  # shape: (runs, batch, 1) or similar
    mean = arr.mean(axis=0).squeeze()
    std = arr.std(axis=0).squeeze()
    return mean, std


def compute_confidence(uncertainty, historical_std=1.0):
    # Simple mapping: lower uncertainty -> higher confidence
    conf = 1.0 / (1.0 + (uncertainty / (historical_std + 1e-9)))
    return float(np.clip(conf, 0.0, 1.0))


def persist_prediction(conn_str, row):
    # row: (timestamp, location, predicted_flow, uncertainty, confidence, model_name)
    insert_sql = """
    INSERT INTO traffic_predictions (timestamp, location, predicted_flow, uncertainty, confidence, model_name)
    VALUES %s
    """
    conn = psycopg2.connect(conn_str)
    try:
        with conn:
            with conn.cursor() as cur:
                execute_values(cur, insert_sql, [row])
    finally:
        conn.close()


@app.post('/predict', response_model=PredictResponse)
async def predict(req: PredictRequest):
    try:
        seq = np.array(req.sequence, dtype=float).reshape(-1, 1)
        # scale
        seq_scaled = scaler.transform(seq).reshape(1, seq.shape[0], 1)

        mean_scaled, std_scaled = mc_predict(model, seq_scaled, runs=MC_RUNS)
        # If model predicts single-step, mean_scaled likely shape (1,); adapt accordingly
        # Inverse transform if scaler is a univariate scaler
        mean_unscaled = scaler.inverse_transform(np.array(mean_scaled).reshape(-1, 1)).squeeze()[0]
        std_unscaled = std_scaled * (scaler.scale_[0] if hasattr(scaler, 'scale_') else 1.0)

        confidence = compute_confidence(std_unscaled, historical_std=5.0)

        timestamp = req.timestamp or datetime.datetime.utcnow().isoformat() + 'Z'
        model_name = os.path.basename(MODEL_PATH)

        persist_row = (timestamp, req.location, float(mean_unscaled), float(std_unscaled), confidence, model_name)
        # Persist -- do it asynchronously / fire-and-forget in production; here we persist synchronously
        persist_prediction(DATABASE_URL, persist_row)

        return {
            'predicted_flow': float(mean_unscaled),
            'uncertainty': float(std_unscaled),
            'confidence': confidence,
            'model_name': model_name,
            'timestamp': timestamp
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

> **Note:** This is a simple synchronous DB write for clarity. In production, use connection pooling and robust retry logic (e.g., SQLAlchemy with pool_pre_ping or pgBouncer).

---

### `Dashboard.py` (Streamlit)

```python
# Dashboard.py
import streamlit as st
import requests
import pandas as pd
import time

API_URL = st.secrets.get('API_URL', 'http://127.0.0.1:8000')

st.set_page_config(page_title='Traffic Prediction Dashboard', layout='wide')

st.title('Traffic Prediction — Live Monitor')

location = st.text_input('Location', value='Hirapur Road')

if st.button('Run Manual Prediction'):
    seq_str = st.text_area('Enter last N flow values as comma-separated', value='45,48,50,47,49')
    seq = [float(x.strip()) for x in seq_str.split(',') if x.strip()]
    payload = {'location': location, 'sequence': seq}
    with st.spinner('Requesting prediction...'):
        resp = requests.post(f'{API_URL}/predict', json=payload, timeout=10)
    if resp.ok:
        data = resp.json()
        st.success(f"Predicted: {data['predicted_flow']:.2f} (uncertainty: {data['uncertainty']:.2f}, conf: {data['confidence']:.2f})")
    else:
        st.error(f'Error: {resp.text}')

st.markdown('---')

st.subheader('Recent Predictions (from DB)')

@st.cache_data(ttl=30)
def load_recent():
    # For a simple demo, call a /recent endpoint or query the DB directly
    try:
        r = requests.get(f'{API_URL}/recent?limit=50', timeout=5)
        if r.ok:
            return pd.DataFrame(r.json())
    except Exception:
        return pd.DataFrame()

df = load_recent()
if df.empty:
    st.info('No recent predictions available or API unreachable.')
else:
    st.dataframe(df)
    st.line_chart(df.set_index('timestamp')['predicted_flow'])
```

---

### `scripts/preprocess.py`

```python
# scripts/preprocess.py
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

INPUT_CSV = 'data/aggregated_traffic_all_new_ok.csv'
OUT_SCALER = 'model/scaler.pkl'
OUT_PROC = 'data/processed_sequences.npz'
SEQUENCE_LENGTH = 10

 df = pd.read_csv(INPUT_CSV, parse_dates=['timestamp'])
# assume columns: timestamp, location, flow

sequences = []
for loc, g in df.groupby('location'):
    flows = g.sort_values('timestamp')['flow'].values
    for i in range(len(flows) - SEQUENCE_LENGTH):
        seq = flows[i:i + SEQUENCE_LENGTH]
        target = flows[i + SEQUENCE_LENGTH]
        sequences.append((seq, target))

X = np.stack([s for s, t in sequences])  # (N, seq_len)
y = np.array([t for s, t in sequences])

scaler = StandardScaler()
N, L = X.shape
X_flat = X.reshape(-1, 1)
scaler.fit(X_flat)
X_scaled = scaler.transform(X_flat).reshape(N, L)

joblib.dump(scaler, OUT_SCALER)
np.savez_compressed(OUT_PROC, X=X_scaled, y=y)
print('Preprocessing completed. Saved scaler and sequences.')
```

---

### `scripts/train.py` (Toy train loop)

```python
# scripts/train.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib

DATA = 'data/processed_sequences.npz'
MODEL_OUT = 'model/gru_lstm_model.h5'

arr = np.load(DATA)
X = arr['X']  # (N, seq_len)
y = arr['y']

X = X.reshape(X.shape[0], X.shape[1], 1)

inp = layers.Input(shape=(X.shape[1], 1))
# GRU block
x = layers.GRU(64, return_sequences=True)(inp)
# LSTM block
x = layers.LSTM(64)(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(1)(x)

model = models.Model(inp, out)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.fit(X, y, epochs=20, batch_size=64, validation_split=0.1)
model.save(MODEL_OUT)
print('Model saved to', MODEL_OUT)
```

---

### `Docker Compose` (dev with Postgres)

```yaml
version: '3.8'
services:
  db:
    image: postgres:14
    restart: unless-stopped
    environment:
      POSTGRES_USER: traffic_user
      POSTGRES_PASSWORD: secure_password
      POSTGRES_DB: traffic_db
    ports:
      - "5432:5432"
    volumes:
      - db_data:/var/lib/postgresql/data

volumes:
  db_data:
```

---

## 🗄️ SQL Schema (copy into a migration file)

```sql
CREATE TABLE IF NOT EXISTS traffic_predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    location TEXT NOT NULL,
    predicted_flow DOUBLE PRECISION NOT NULL,
    uncertainty DOUBLE PRECISION,
    confidence DOUBLE PRECISION,
    model_name TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_tp_timestamp ON traffic_predictions(timestamp);
CREATE INDEX idx_tp_location ON traffic_predictions(location);
```

---

## ✅ Quick Start

```bash
# clone
git clone https://github.com/<your-org>/Traffic_Prediction.git
cd Traffic_Prediction

# env
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# start DB (optional: docker-compose up -d)
# start backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# start dashboard
streamlit run Dashboard.py
```

---

## 🔎 Evaluation & Research Notes

* Report MAE, RMSE, MAPE on held-out test sets
* Provide calibration metrics for MC Dropout intervals
* Track experiments with MLflow or Weights & Biases

---

## 🔧 Maintenance & Extensibility

* Add streaming adapters, feature store, GNN spatial model variants
* Use model registry and immutable artifact storage

---

## 🧾 Licensing

Add `LICENSE` (MIT or Apache-2.0) to the repository root.

---

*End of README.md (single-file with embedded code blocks). Save, edit, or copy the contained code blocks into corresponding files in your repo.*
