"""ClickHouse storage (embedded engine via chDB).

Two things live here:
- `raw`: the frozen public series, loaded from the CSV snapshots.
- `action_log`: an append-only event log for proposed actions. Nothing is ever
  updated in place; the status of an action is the latest event for its id.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from chdb import session

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "data" / "eco2mix_national_tr_snapshot.csv"

# name -> (file, timestamp column, value column, readings expected per hour)
SERIES = {
    "fr_load": (SNAPSHOT, "date_heure", "consommation", 4),
    "paris_bike": (ROOT / "data" / "paris_bike_sebastopol_sn_snapshot.csv", "date", "sum_counts", 1),
}
GRID_START, GRID_END = "2026-09-01 00:00:00", "2026-11-19 00:00:00"  # UTC, end exclusive
DEFAULT_DB_PATH = ROOT / ".chdb"

EVENTS = ("proposed", "invalid", "approved", "rejected", "executed_dry_run")


class Store:
    def __init__(self, path: str | Path | None = None):
        """`path=None` gives a throwaway in-memory session (used by the backtest and tests)."""
        self.session = session.Session(str(path)) if path else session.Session()
        q = self.session.query
        q("CREATE DATABASE IF NOT EXISTS aot")
        q(
            "CREATE TABLE IF NOT EXISTS aot.raw "
            "(series LowCardinality(String), ts DateTime64(0, 'UTC'), value Float64) "
            "ENGINE = MergeTree ORDER BY (series, ts)"
        )
        q(
            "CREATE TABLE IF NOT EXISTS aot.action_log "
            "(id String, event LowCardinality(String), ts DateTime64(6, 'UTC'), "
            "actor String, payload String) ENGINE = MergeTree ORDER BY (id, ts)"
        )

    def close(self) -> None:
        self.session.close()

    # --- time series -----------------------------------------------------

    def load_series(self, name: str = "fr_load") -> int:
        """Idempotent load of one frozen CSV. Returns the row count for that series."""
        csv_path, ts_col, value_col, _ = SERIES[name]
        path = str(csv_path.resolve()).replace("'", "\\'")
        self.session.query(f"ALTER TABLE aot.raw DELETE WHERE series = '{name}' SETTINGS mutations_sync = 2")
        self.session.query(
            f"INSERT INTO aot.raw SELECT '{name}', {ts_col}, {value_col} "
            f"FROM file('{path}', CSVWithNames, "
            f"'{ts_col} DateTime64(0, \'\'UTC\'\'), {value_col} Float64')"
        )
        df = self.session.query(f"SELECT count() AS n FROM aot.raw WHERE series = '{name}'", "DataFrame")
        return int(df.n[0])

    def hourly(self, name: str = "fr_load") -> pd.Series:
        """Hourly values on a gap-free UTC grid covering the whole snapshot range.

        An hour is kept only if all its expected readings are present (4 quarter-hours
        for the load series). Missing hours are emitted as NaN by `WITH FILL`, never
        interpolated.
        """
        per_hour = SERIES[name][3]
        df = self.session.query(
            f"""
            SELECT h, v FROM (
                SELECT toStartOfHour(ts) AS h, toNullable(avg(value)) AS v
                FROM aot.raw WHERE series = '{name}' GROUP BY h HAVING count() = {per_hour}
            ) ORDER BY h WITH FILL
                FROM toDateTime('{GRID_START}', 'UTC') TO toDateTime('{GRID_END}', 'UTC')
                STEP toIntervalHour(1)
            """,
            "DataFrame",
        )
        return pd.Series(df.v.to_numpy(dtype=float), index=pd.DatetimeIndex(df.h), name=name)

    # --- action log --------------------------------------------------------

    def log_event(self, action_id: str, event: str, actor: str, payload: dict) -> None:
        if event not in EVENTS:
            raise ValueError(f"unknown event {event!r}")
        df = pd.DataFrame(  # noqa: F841 - read by chDB through the Python() table function
            [
                {
                    "id": action_id,
                    "event": event,
                    "ts": datetime.now(timezone.utc),
                    "actor": actor,
                    "payload": json.dumps(payload, sort_keys=True),
                }
            ]
        )
        self.session.query("INSERT INTO aot.action_log SELECT * FROM Python(df)")

    def events(self, action_id: str | None = None) -> pd.DataFrame:
        df = self.session.query(
            "SELECT id, event, ts, actor, payload FROM aot.action_log ORDER BY ts", "DataFrame"
        )
        return df[df.id == action_id] if action_id else df

    def status(self, action_id: str) -> str | None:
        ev = self.events(action_id)
        return None if ev.empty else str(ev.event.iloc[-1])

    def proposal(self, action_id: str) -> dict | None:
        ev = self.events(action_id)
        ev = ev[ev.event == "proposed"]
        return None if ev.empty else json.loads(ev.payload.iloc[0])
