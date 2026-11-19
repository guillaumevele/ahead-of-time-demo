"""ClickHouse storage (embedded engine via chDB).

Two things live here:
- `raw_load`: the 15-minute national load series, loaded from the CSV snapshot.
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
DEFAULT_DB_PATH = ROOT / ".chdb"

EVENTS = ("proposed", "invalid", "approved", "rejected", "executed_dry_run")


class Store:
    def __init__(self, path: str | Path | None = None):
        """`path=None` gives a throwaway in-memory session (used by the backtest and tests)."""
        self.session = session.Session(str(path)) if path else session.Session()
        q = self.session.query
        q("CREATE DATABASE IF NOT EXISTS aot")
        q(
            "CREATE TABLE IF NOT EXISTS aot.raw_load "
            "(ts DateTime64(0, 'UTC'), mw Float64) ENGINE = MergeTree ORDER BY ts"
        )
        q(
            "CREATE TABLE IF NOT EXISTS aot.action_log "
            "(id String, event LowCardinality(String), ts DateTime64(6, 'UTC'), "
            "actor String, payload String) ENGINE = MergeTree ORDER BY (id, ts)"
        )

    def close(self) -> None:
        self.session.close()

    # --- time series -----------------------------------------------------

    def load_snapshot(self, csv_path: Path = SNAPSHOT) -> int:
        """Idempotent load of the CSV snapshot. Returns the row count in the table."""
        path = str(Path(csv_path).resolve()).replace("'", "\\'")
        self.session.query("TRUNCATE TABLE aot.raw_load")
        self.session.query(
            "INSERT INTO aot.raw_load SELECT date_heure, consommation "
            f"FROM file('{path}', CSVWithNames, "
            "'date_heure DateTime64(0, ''UTC''), consommation Float64')"
        )
        return int(self.session.query("SELECT count() AS n FROM aot.raw_load", "DataFrame").n[0])

    def hourly(self) -> pd.Series:
        """Hourly mean load in MW on a gap-free UTC grid.

        An hour is kept only if its four quarter-hour readings are all present;
        missing hours are emitted as NaN by `WITH FILL` rather than interpolated.
        """
        df = self.session.query(
            """
            SELECT h, mw FROM (
                SELECT toStartOfHour(ts) AS h, toNullable(avg(mw)) AS mw
                FROM aot.raw_load GROUP BY h HAVING count() = 4
            ) ORDER BY h WITH FILL STEP toIntervalHour(1)
            """,
            "DataFrame",
        )
        return pd.Series(df.mw.to_numpy(dtype=float), index=pd.DatetimeIndex(df.h), name="mw")

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
