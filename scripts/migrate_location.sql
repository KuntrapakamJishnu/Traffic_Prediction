-- Migration: add textual `location` column and populate from existing `location_id` if present
BEGIN;

-- Add `location` column if missing
ALTER TABLE IF EXISTS traffic_predictions
    ADD COLUMN IF NOT EXISTS location TEXT;

-- If traffic_alerts exists, add location too
ALTER TABLE IF EXISTS traffic_alerts
    ADD COLUMN IF NOT EXISTS location TEXT;

-- Populate `location` from numeric `location_id` where empty
UPDATE traffic_predictions
SET location = location_id::text
WHERE (location IS NULL OR location = '') AND to_regclass('traffic_predictions.location_id') IS NOT NULL;

-- Also populate alerts
UPDATE traffic_alerts
SET location = location_id::text
WHERE (location IS NULL OR location = '') AND to_regclass('traffic_alerts.location_id') IS NOT NULL;

-- Create an index on the textual location for faster lookups
CREATE INDEX IF NOT EXISTS idx_tp_location ON traffic_predictions(location);
CREATE INDEX IF NOT EXISTS idx_ta_location ON traffic_alerts(location);

COMMIT;

-- Notes:
-- Run this file with psql or any SQL client connected to `traffic_db`:
-- psql -U postgres -d traffic_db -f scripts/migrate_location.sql

-- If your DB uses a different username/port, adjust the psql command accordingly.
