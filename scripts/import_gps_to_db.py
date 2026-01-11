#!/usr/bin/env python3
"""Import gps_data.csv into PostgreSQL as `traffic_predictions`.

Usage:
  py -3 scripts/import_gps_to_db.py

Config via env vars (optional):
  DB_NAME, DB_USER, DB_PASS, DB_HOST, DB_PORT

The script will:
  - create `traffic_predictions` if it doesn't exist
  - bulk load `gps_data.csv` into a temp table via COPY
  - insert rows into `traffic_predictions` mapping `speed` -> `predicted_flow`
  - create a basic index on (lat, lon, timestamp)
"""
from pathlib import Path
import os
import sys
import psycopg2
from psycopg2 import sql

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "gps_data.csv"

DB_NAME = os.getenv("DB_NAME", "traffic_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "postgres123")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5433")

CREATE_TABLE_SQL = r"""
CREATE TABLE IF NOT EXISTS traffic_predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITHOUT TIME ZONE,
    vehicle_id INTEGER,
    location TEXT,
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    speed DOUBLE PRECISION,
    predicted_flow DOUBLE PRECISION,
    uncertainty DOUBLE PRECISION,
    confidence VARCHAR(16)
);
"""

def main():
    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}")
        sys.exit(1)

    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT)
    conn.autocommit = True
    cur = conn.cursor()

    print("Creating table traffic_predictions (if not exists)")
    cur.execute(CREATE_TABLE_SQL)

    # Create a temp table matching CSV columns
    cur.execute("DROP TABLE IF EXISTS _gps_raw_tmp")
    cur.execute(
        """
        CREATE TEMP TABLE _gps_raw_tmp (
            timestamp TEXT,
            vehicle_id INTEGER,
            lat DOUBLE PRECISION,
            lon DOUBLE PRECISION,
            speed DOUBLE PRECISION
        ) ON COMMIT DROP;
        """
    )

    print(f"COPYing CSV into temp table from {CSV_PATH}")
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        # Use COPY for speed; skip header
        cur.copy_expert("COPY _gps_raw_tmp FROM STDIN WITH CSV HEADER", f)

    print("Inserting into traffic_predictions (mapping speed -> predicted_flow)")
    insert_sql = sql.SQL(
        """
        INSERT INTO traffic_predictions (timestamp, vehicle_id, lat, lon, speed, predicted_flow, confidence)
        SELECT
            NULLIF(timestamp, '')::timestamp,
            vehicle_id,
            lat,
            lon,
            speed,
            speed AS predicted_flow,
            'LOW'::varchar
        FROM _gps_raw_tmp
        WHERE lat IS NOT NULL AND lon IS NOT NULL
        """
    )
    cur.execute(insert_sql)

    print("Creating index on lat, lon, timestamp")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tp_lat_lon_time ON traffic_predictions (lat, lon, timestamp)")

    cur.close()
    conn.close()
    print("Done. Inserted rows from CSV into traffic_predictions.")

if __name__ == '__main__':
    main()
