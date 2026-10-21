-- PlantWatch relational schema (Phase 3). Target: PostgreSQL (AWS RDS).
--
-- Layers:
--   machines      reference data for each monitored machine
--   features      the engineered "silver" layer: one row per (window, sensor, stat)
--   predictions   the model "gold" layer: one anomaly score per (window, model)
-- Views roll predictions up for the dashboard and monitoring.

-- ---------------------------------------------------------------------------
-- Reference data
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS machines (
    machine_id   TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    location     TEXT,
    installed_on DATE
);

-- ---------------------------------------------------------------------------
-- Engineered features (long format: tidy, and cheap to add new stats/sensors)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS features (
    machine_id  TEXT NOT NULL REFERENCES machines (machine_id),
    window_end  TIMESTAMPTZ NOT NULL,
    sensor      TEXT NOT NULL,
    stat        TEXT NOT NULL,                 -- mean | std | min | max
    value       DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (machine_id, window_end, sensor, stat)
);

CREATE INDEX IF NOT EXISTS idx_features_window
    ON features (machine_id, window_end);

-- ---------------------------------------------------------------------------
-- Model predictions (one row per window per model version)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions (
    machine_id    TEXT NOT NULL REFERENCES machines (machine_id),
    window_end    TIMESTAMPTZ NOT NULL,
    model_version TEXT NOT NULL,
    score         DOUBLE PRECISION NOT NULL,   -- reconstruction error / anomaly score
    is_anomaly    BOOLEAN NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (machine_id, window_end, model_version)
);

-- Covering index for the hot path: "recent anomalies for a machine".
CREATE INDEX IF NOT EXISTS idx_predictions_window
    ON predictions (machine_id, window_end DESC);

-- Partial index: anomalies are rare, so index only those rows.
CREATE INDEX IF NOT EXISTS idx_predictions_anomaly
    ON predictions (machine_id, window_end DESC)
    WHERE is_anomaly;

-- ---------------------------------------------------------------------------
-- Dashboard / monitoring views
-- ---------------------------------------------------------------------------

-- Hourly anomaly rate per machine and model — the monitoring time series.
CREATE OR REPLACE VIEW v_hourly_anomaly_rate AS
SELECT
    machine_id,
    model_version,
    date_trunc('hour', window_end)                       AS hour,
    count(*)                                             AS windows,
    count(*) FILTER (WHERE is_anomaly)                   AS anomalies,
    avg(CASE WHEN is_anomaly THEN 1.0 ELSE 0.0 END)      AS anomaly_rate,
    avg(score)                                           AS mean_score
FROM predictions
GROUP BY machine_id, model_version, date_trunc('hour', window_end);

-- Most recent prediction per window across models (latest write wins).
CREATE OR REPLACE VIEW v_latest_prediction AS
SELECT DISTINCT ON (machine_id, window_end)
    machine_id, window_end, model_version, score, is_anomaly, created_at
FROM predictions
ORDER BY machine_id, window_end, created_at DESC;
