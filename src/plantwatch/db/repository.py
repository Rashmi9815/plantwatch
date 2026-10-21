"""Database access layer.

Defines the schema with SQLAlchemy Core so the same code runs on SQLite (tests, local
dev) and PostgreSQL/RDS (production) — only the connection URL changes. The hand-written
``schema.sql`` is the authoritative DDL applied to RDS (it carries the partial indexes
and views); this mirror keeps the Python path portable and testable.

    engine = get_engine("postgresql+psycopg://user:pw@host/plantwatch")  # real RDS
    engine = get_engine()                                                # in-memory SQLite
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

import numpy as np
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    MetaData,
    String,
    Table,
    create_engine,
    func,
    insert,
    select,
)
from sqlalchemy.engine import Engine

metadata = MetaData()

machines = Table(
    "machines",
    metadata,
    Column("machine_id", String, primary_key=True),
    Column("name", String, nullable=False),
    Column("location", String),
    Column("installed_on", DateTime(timezone=True)),
)

features = Table(
    "features",
    metadata,
    Column("machine_id", String, ForeignKey("machines.machine_id"), primary_key=True),
    Column("window_end", DateTime(timezone=True), primary_key=True),
    Column("sensor", String, primary_key=True),
    Column("stat", String, primary_key=True),
    Column("value", Float, nullable=False),
)
Index("idx_features_window", features.c.machine_id, features.c.window_end)

predictions = Table(
    "predictions",
    metadata,
    Column("machine_id", String, ForeignKey("machines.machine_id"), primary_key=True),
    Column("window_end", DateTime(timezone=True), primary_key=True),
    Column("model_version", String, primary_key=True),
    Column("score", Float, nullable=False),
    Column("is_anomaly", Boolean, nullable=False),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
)
Index("idx_predictions_window", predictions.c.machine_id, predictions.c.window_end)


def get_engine(url: str = "sqlite+pysqlite:///:memory:", *, echo: bool = False) -> Engine:
    return create_engine(url, echo=echo, future=True)


def create_all(engine: Engine) -> None:
    metadata.create_all(engine)


def upsert_machine(
    engine: Engine,
    machine_id: str,
    name: str,
    *,
    location: str | None = None,
    installed_on: datetime | None = None,
) -> None:
    with engine.begin() as conn:
        exists = conn.execute(
            select(machines.c.machine_id).where(machines.c.machine_id == machine_id)
        ).first()
        if exists:
            return
        conn.execute(
            insert(machines).values(
                machine_id=machine_id, name=name, location=location, installed_on=installed_on
            )
        )


def insert_predictions(engine: Engine, rows: Iterable[dict]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    with engine.begin() as conn:
        conn.execute(insert(predictions), rows)
    return len(rows)


def predictions_from_scores(
    machine_id: str,
    index,
    scores,
    *,
    threshold: float,
    model_version: str,
) -> list[dict]:
    """Build prediction rows from a detector's scores aligned to window-end timestamps."""
    rows = []
    for ts, score in zip(index, np.asarray(scores), strict=True):
        ts = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        rows.append(
            {
                "machine_id": machine_id,
                "window_end": ts,
                "model_version": model_version,
                "score": float(score),
                "is_anomaly": bool(score >= threshold),
            }
        )
    return rows


def recent_anomalies(
    engine: Engine, *, machine_id: str | None = None, limit: int = 50
) -> list[dict]:
    query = select(predictions).where(predictions.c.is_anomaly.is_(True))
    if machine_id is not None:
        query = query.where(predictions.c.machine_id == machine_id)
    query = query.order_by(predictions.c.window_end.desc()).limit(limit)
    with engine.connect() as conn:
        return [dict(row._mapping) for row in conn.execute(query)]


def anomaly_count_by_model(engine: Engine) -> dict[str, int]:
    query = (
        select(predictions.c.model_version, func.count().label("n"))
        .where(predictions.c.is_anomaly.is_(True))
        .group_by(predictions.c.model_version)
    )
    with engine.connect() as conn:
        return {row.model_version: row.n for row in conn.execute(query)}


def count_machines(engine: Engine) -> int:
    with engine.connect() as conn:
        return conn.execute(select(func.count()).select_from(machines)).scalar_one()
