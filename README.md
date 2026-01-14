🚦 Traffic Prediction Platform

An industry-grade, research-ready, end-to-end AI Traffic Prediction & Analytics System.
Performs short-term traffic flow forecasting using a hybrid GRU–LSTM model with uncertainty estimation, serves real-time predictions via a FastAPI backend, persists results to PostgreSQL, and exposes an interactive Streamlit dashboard for monitoring, alerts, and historical analysis.

📖 Executive Summary

This repository implements a robust pipeline for real-time traffic forecasting designed for deployment and academic evaluation. Key strengths:

Hybrid GRU + LSTM deep learning model that captures short- and long-term temporal patterns.

Monte Carlo Dropout inference for uncertainty and confidence estimation.

FastAPI backend for low-latency prediction serving and scalability.

PostgreSQL for reliable storage of predictions and analytics-ready history.

Streamlit dashboard for easy exploration, operational monitoring, and alerting.

Modular design enabling reproducible research and production deployment.

🎯 Objectives

Provide accurate short-horizon traffic predictions (minutes → hours).

Attach uncertainty metrics to each prediction for safer decision making.

Persist and analyze predictions to support historical analytics and alerts.

Offer a user-friendly dashboard for stakeholders (traffic engineers, city planners).

Ship a production-minded codebase: clear structure, environment setup, and deployment guidance.

🏗️ System Architecture
[Traffic Sensors / Data Sources]
            ↓ (ingest, batch, or RT stream)
     Data Preprocessing Layer
            ↓ (sequences, scaling)
      Hybrid GRU + LSTM Model
            ↓ (predictions + MC Dropout runs)
       FastAPI Prediction Service
            ↓ (persist)
       PostgreSQL (predictions/history)
            ↓
      Streamlit Dashboard (UI + Alerts)


🔧 Tech Stack

Language: Python 3.9+

ML: TensorFlow / Keras (GRU, LSTM), NumPy, scikit-learn (scalers)

Backend: FastAPI, Uvicorn

Dashboard: Streamlit, Pandas

DB: PostgreSQL 14+, psycopg2

Utilities: joblib, SQLAlchemy (optional), Docker (recommended)

📂 Project Structure
Traffic_Prediction/
│
├── app.py                         # FastAPI application (prediction endpoints + DB writes)
├── Dashboard.py                   # Streamlit app entry point (multi-page)
├── requirements.txt               # Python dependencies
├── README.md                      # This document
│
├── model/
│   ├── gru_lstm_model.h5          # Trained model weights
│   └── scaler.pkl                 # Preprocessing scaler (joblib)
│
├── pages/                         # Streamlit subpages (Live_Monitor, Alerts, History)
│   ├── Live_Monitor.py
│   ├── Alerts.py
│   └── Historical_Analytics.py
│
├── real_time/                     # Real-time ingestion & adapters (websocket/SSE helpers)
├── scripts/                       # Training, evaluation, preprocessing scripts
├── static/                        # Images, CSS, assets for dashboard
│
├── data/
│   ├── aggregated_traffic_all_new_ok.csv
│   └── aggressive_traffic_alerts_new_ok.csv
│
└── docs/                          # Optional: diagrams, experiment logs, model cards

🗄️ Database Design (Production Ready)

Table: traffic_predictions

Column	Type	Notes
id	SERIAL	Primary key
timestamp	TIMESTAMP	Prediction timestamp (UTC recommended)
location	TEXT	Location identifier (standardized)
predicted_flow	FLOAT	Predicted traffic flow (units consistent with input)
uncertainty	FLOAT	Estimated standard deviation or similar uncertainty metric
confidence	FLOAT	Derived confidence score (0–1)
model_name	TEXT	Model version / artifact id
created_at	TIMESTAMP	Record insert time (DEFAULT NOW())

Schema (SQL):

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


Indexing recommendations:

CREATE INDEX idx_tp_timestamp ON traffic_predictions (timestamp);

CREATE INDEX idx_tp_location ON traffic_predictions (location);
For high-query loads, consider partitioning by time (daily/monthly) and using connection pooling (pgbouncer).

⚙️ Environment & Setup
1. Clone
git clone https://github.com/<your-org>/Traffic_Prediction.git
cd Traffic_Prediction

2. Python environment
python -m venv venv
# Activate:
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

4. Environment variables (recommended)

Create a .env or pass environment variables for DB and runtime settings:

DATABASE_URL=postgresql://user:password@localhost:5432/traffic_db
MODEL_PATH=./model/gru_lstm_model.h5
SCALER_PATH=./model/scaler.pkl
APP_ENV=production
LOG_LEVEL=info

🐘 PostgreSQL Setup (detailed)

Install PostgreSQL 14+ (OS package manager, Docker, or cloud RDS).

Create DB and user:

CREATE DATABASE traffic_db;
CREATE USER traffic_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE traffic_db TO traffic_user;


Create schema/tables as above.

Configure connection string in .env and ensure psycopg2 present in requirements.txt.

Docker Compose snippet (recommended for dev):

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

🚀 Run the Backend (Development)

Start FastAPI with auto reload:

uvicorn app:app --reload --host 0.0.0.0 --port 8000


Open API docs:

http://127.0.0.1:8000/docs


Production note: run via Gunicorn + Uvicorn workers or containerize and use an orchestration platform; enable HTTPS and authentication.

📡 Prediction API (Design & Examples)
Endpoint: POST /predict

Accepts JSON payload (sequence + metadata), returns prediction with uncertainty.

Request

{
  "location": "Hirapur Road",
  "sequence": [45, 48, 50, 47, 49],   // past N flow measurements
  "timestamp": "2026-01-15T07:30:00Z"  // optional - ISO 8601 UTC recommended
}


Response

{
  "predicted_flow": 52.08,
  "uncertainty": 1.82,
  "confidence": 0.916,
  "model_name": "gru_lstm_v1",
  "timestamp": "2026-01-15T07:35:00Z"
}


Implementation notes:

Preprocess (scale, reshape) incoming sequences using scaler.pkl.

Run multiple stochastic forward passes (MC Dropout) and compute mean + stddev => predicted_flow, uncertainty.

Derive confidence as a function of uncertainty and historical error statistics.

Persist results into PostgreSQL with model_name and created_at.

📊 Streamlit Dashboard

Start:

streamlit run Dashboard.py


Pages

Overview: KPI cards (current flow, avg confidence, top anomalies).

Live Monitor: Real-time charting, manual prediction input.

Alerts: AI-driven alerts with severity (NORMAL / MODERATE / HIGH).

Historical Analytics: Query last 48 hours, compare predicted vs observed, confidence histograms.

Best practices

Use caching (st.cache_data) for costly DB queries.

Poll backend at a reasonable interval (e.g., 10–30s) or subscribe to SSE/WebSocket for push updates.

Show clear provenance: model version, data timestamp, and confidence on every chart.

🚨 Alerts & Anomaly Detection

Algorithm (example):

Compute Z-score of recent residuals or use rolling MAD.

Thresholds:

|z| < 2 → NORMAL

2 ≤ |z| < 3 → MODERATE

|z| ≥ 3 → HIGH

Integration

Alert records saved to alerts table (optional).

Dashboard highlights active alerts and allows acknowledgment.

Optionally push critical alerts to Slack/Telegram/email via webhook.

🧠 Model Details & Reproducibility

Architecture: stacked GRU + LSTM layers, dropout layers for MC inference.

Training pipeline: scripts in scripts/ handle preprocessing, sequence creation, model training, checkpointing, and evaluation.

Preprocessing: scaler.pkl (StandardScaler/MinMax) persisted via joblib—must be used at inference.

Evaluation metrics: MAE, RMSE, MAPE, and calibration metrics for uncertainty (e.g., prediction interval coverage).

Experiment tracking: recommended to use MLflow or Weights & Biases (store hyperparameters, seeds, artifact versions).

Training example (high level):

# preprocess
python scripts/preprocess.py --input data/aggregated_traffic.csv --out data/processed.pkl

# train
python scripts/train.py --config configs/train.yaml --output model/

✅ Deployment Recommendations

Containerize with Docker. Multi-stage build: install dependencies, copy model artifacts, and expose Uvicorn server.

Use an application server: Gunicorn + Uvicorn workers or FastAPI with uvicorn supervised by systemd / Docker orchestrator.

Scaling: Horizontal scale API servers behind a load balancer; use a managed PostgreSQL and connection pooling.

Observability: Prometheus + Grafana for metrics; centralize logs (ELK / EFK).

Security: Serve over HTTPS, secure DB credentials (vault / secrets manager), enable RBAC and token authentication on APIs if public.

CI/CD: Automated tests, linting, model artifact validation, and automated image builds on push.

🔎 Evaluation & Research Notes

Report MAE, RMSE, and MAPE on held-out test sets; include confidence intervals.

Include ablation studies: GRU only vs LSTM only vs hybrid; dropout rates; sequence lengths.

Publish a model card (docs/) describing limitations, intended use, and data provenance.

When claiming "real-time", specify latency SLOs (e.g., <200ms per request) and measure them.

🛠 Troubleshooting & Tips

If model fails to load: check MODEL_PATH, ensure TensorFlow version matches the model.

DB connection errors: verify DATABASE_URL, confirm PostgreSQL is reachable and credentials correct.

High latency: enable batching, increase workers, profile model inference (use TF-XLA or TensorRT where applicable).

Reproducibility issues: fix random seeds in training (numpy, tensorflow, python random) and log them.

♻️ Maintenance & Extensibility

Add more sources: integrate IoT sensor streams, external APIs, or historical datasets.

Model upgrades: treat model artifacts as immutable versions; store version with each prediction.

Feature store: consider introducing a feature store for feature engineering reuse.

Geo-aware modeling: incorporate spatial graphs (GNNs) for multi-location correlation.

📁 Useful Scripts (examples)

scripts/preprocess.py — data cleaning and sequence generation

scripts/train.py — model training loop with checkpoints and evaluation

scripts/evaluate.py — compute evaluation metrics and calibration plots

scripts/deploy_docker.sh — builds and pushes container image

📄 Author & Contact

Jishnu K.
AI / ML Engineer — Traffic Intelligence Systems
(Include email or professional link in repo profile if desired)

🧾 Licensing & Citations

Add a LICENSE (e.g., MIT or Apache-2.0) to clarify reuse and attribution.

Include references to any datasets or public APIs used; document data collection and consent if applicable.

✅ Quick Start Summary (commands)
# clone
git clone https://github.com/<your-org>/Traffic_Prediction.git
cd Traffic_Prediction

# env
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# start DB (optional: docker-compose)
# start backend
uvicorn app:app --reload --port 8000

# start dashboard
streamlit run Dashboard.py

Final Notes (research + production mindset)

This repository is structured to serve two audiences: researchers who require experiment reproducibility and engineers who need a deployable, maintainable system. Keep model artifacts, dataset splits, and experiment logs under version control (artifacts registry) and enforce rigorous testing and monitoring before any production rollout.
