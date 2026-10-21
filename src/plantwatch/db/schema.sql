-- PlantWatch relational schema (starter — expanded in Phase 3).
-- Target: PostgreSQL (AWS RDS). Holds machine metadata, engineered features, and
-- model predictions so the dashboard and monitoring can query them with plain SQL.

CREATE TABLE IF NOT EXISTS machines (
    machine_id   TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    location     TEXT,
    installed_on DATE
);

-- One row per sensor reading window (the "silver" feature layer).
CREATE TABLE IF NOT EXISTS features (
    machine_id   TEXT NOT NULL REFERENCES machines (machine_id),
    window_end   TIMESTAMPTZ NOT NULL,
    sensor       TEXT NOT NULL,
    stat         TEXT NOT NULL,          -- mean | std | min | max
    value        DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (machine_id, window_end, sensor, stat)
);

CREATE INDEX IF NOT EXISTS idx_features_window
    ON features (machine_id, window_end);

-- Model output (the "gold" layer the dashboard reads).
CREATE TABLE IF NOT EXISTS predictions (
    machine_id    TEXT NOT NULL REFERENCES machines (machine_id),
    window_end    TIMESTAMPTZ NOT NULL,
    model_version TEXT NOT NULL,
    score         DOUBLE PRECISION NOT NULL,   -- reconstruction error / anomaly score
    is_anomaly    BOOLEAN NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (machine_id, window_end, model_version)
);

CREATE INDEX IF NOT EXISTS idx_predictions_anomaly
    ON predictions (machine_id, window_end)
    WHERE is_anomaly;
