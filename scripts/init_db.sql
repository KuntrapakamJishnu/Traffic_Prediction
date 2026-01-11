-- Initialize traffic database schema and helpful queries

CREATE DATABASE traffic_db;

\c traffic_db

CREATE TABLE IF NOT EXISTS traffic_predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    location TEXT,
    predicted_flow FLOAT,
    uncertainty FLOAT,
    confidence TEXT
);

CREATE INDEX IF NOT EXISTS idx_predictions_location
ON traffic_predictions(location);

CREATE INDEX IF NOT EXISTS idx_predictions_timestamp
ON traffic_predictions(timestamp);

CREATE TABLE IF NOT EXISTS traffic_alerts (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    location TEXT NOT NULL,
    alert_level TEXT,
    predicted_flow FLOAT
);

CREATE TABLE IF NOT EXISTS traffic_locations (
    location TEXT PRIMARY KEY,
    lat FLOAT,
    lon FLOAT,
    zone_id INTEGER
);

-- Helpful queries
-- Show latest predictions
SELECT *
FROM traffic_predictions
ORDER BY timestamp DESC
LIMIT 10;

-- Count predictions per location
SELECT location, COUNT(*) AS total_predictions
FROM traffic_predictions
GROUP BY location
ORDER BY total_predictions DESC;

-- Average flow by location
SELECT
    location,
    AVG(predicted_flow) AS avg_flow
FROM traffic_predictions
GROUP BY location
ORDER BY avg_flow DESC;

-- Latest alert per location
SELECT DISTINCT ON (location)
    location,
    alert_level,
    predicted_flow,
    timestamp
FROM traffic_alerts
ORDER BY location, timestamp DESC;
