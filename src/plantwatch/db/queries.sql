-- PlantWatch analytical queries (PostgreSQL). Named blocks the dashboard and
-- monitoring jobs run. Each is written to use the indexes defined in schema.sql.

-- name: recent_anomalies
-- The operator's front page: latest anomalies for one machine.
SELECT window_end, model_version, score
FROM predictions
WHERE machine_id = :machine_id
  AND is_anomaly
ORDER BY window_end DESC
LIMIT :limit;                       -- uses idx_predictions_anomaly

-- name: anomaly_rate_last_24h
-- Hourly anomaly rate for the last day — drives the monitoring chart.
SELECT hour, anomalies, windows, anomaly_rate
FROM v_hourly_anomaly_rate
WHERE machine_id = :machine_id
  AND model_version = :model_version
  AND hour >= now() - INTERVAL '24 hours'
ORDER BY hour;

-- name: model_comparison
-- Side-by-side anomaly counts per model version (baseline vs challenger).
SELECT model_version,
       count(*)                           AS windows,
       count(*) FILTER (WHERE is_anomaly) AS anomalies,
       round(avg(score)::numeric, 4)      AS mean_score
FROM predictions
WHERE machine_id = :machine_id
GROUP BY model_version
ORDER BY anomalies DESC;

-- name: top_scoring_windows
-- The windows a reliability engineer should inspect first.
SELECT window_end, model_version, score
FROM predictions
WHERE machine_id = :machine_id
ORDER BY score DESC
LIMIT :limit;

-- name: feature_snapshot_for_window
-- Pull the engineered features behind a single flagged window (for drill-down).
SELECT sensor, stat, value
FROM features
WHERE machine_id = :machine_id
  AND window_end = :window_end
ORDER BY sensor, stat;           -- uses idx_features_window
