"""
SQLite database layer for storing seasonal analysis records.

Tables:
  - analysis_runs:       Metadata for each pipeline execution
  - displacement_records: Full processed records stored per analysis run
  - run_summary:          Aggregated statistics per run (country, type, month)
"""

import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional


def get_connection(db_path: Path) -> sqlite3.Connection:
    """Get a connection to the SQLite database, creating it if needed."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: Path) -> None:
    """Create all tables if they do not exist."""
    conn = get_connection(db_path)
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS analysis_runs (
                run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
                season          TEXT NOT NULL,
                year            INTEGER NOT NULL,
                executed_at     TEXT NOT NULL,
                records_count   INTEGER NOT NULL DEFAULT 0,
                total_displaced INTEGER NOT NULL DEFAULT 0,
                countries_count INTEGER NOT NULL DEFAULT 0,
                steps_executed  TEXT,
                status          TEXT NOT NULL DEFAULT 'completed',
                notes           TEXT,
                UNIQUE(season, year, executed_at)
            );

            CREATE TABLE IF NOT EXISTS displacement_records (
                record_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id          INTEGER NOT NULL REFERENCES analysis_runs(run_id),
                idmc_id         INTEGER,
                country         TEXT NOT NULL,
                displacement_type TEXT,
                disaster_type   TEXT,
                figure          INTEGER NOT NULL DEFAULT 0,
                event_name      TEXT,
                sources         TEXT,
                source_url      TEXT,
                locations_name  TEXT,
                month           TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_records_run
                ON displacement_records(run_id);
            CREATE INDEX IF NOT EXISTS idx_records_country
                ON displacement_records(country);
            CREATE INDEX IF NOT EXISTS idx_records_month
                ON displacement_records(month);

            CREATE TABLE IF NOT EXISTS run_summary (
                summary_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id          INTEGER NOT NULL REFERENCES analysis_runs(run_id),
                country         TEXT NOT NULL,
                month           TEXT NOT NULL,
                displacement_type TEXT,
                disaster_type   TEXT,
                total_figure    INTEGER NOT NULL DEFAULT 0,
                event_count     INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_summary_run
                ON run_summary(run_id);
        """)
        conn.commit()
    finally:
        conn.close()


def save_analysis_run(
    db_path: Path,
    season: str,
    year: int,
    df: pd.DataFrame,
    steps_executed: list,
    notes: Optional[str] = None,
) -> int:
    """
    Save a complete analysis run to the database.

    Returns the run_id of the inserted analysis_runs record.
    """
    conn = get_connection(db_path)
    try:
        now = datetime.now().isoformat()
        total_displaced = int(df['figure'].sum()) if 'figure' in df.columns else 0
        countries_count = df['country'].nunique() if 'country' in df.columns else 0

        cursor = conn.execute(
            """INSERT INTO analysis_runs
               (season, year, executed_at, records_count, total_displaced,
                countries_count, steps_executed, status, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'completed', ?)""",
            (season, year, now, len(df), total_displaced, countries_count,
             ','.join(steps_executed), notes)
        )
        run_id = cursor.lastrowid

        # Insert individual displacement records
        records = []
        for _, row in df.iterrows():
            records.append((
                run_id,
                row.get('id'),
                row.get('country', ''),
                row.get('displacement_type', ''),
                row.get('Disaster_Type', ''),
                int(row.get('figure', 0)),
                row.get('event_name', ''),
                row.get('sources', ''),
                row.get('source_url', ''),
                row.get('locations_name', ''),
                row.get('Month', ''),
            ))

        conn.executemany(
            """INSERT INTO displacement_records
               (run_id, idmc_id, country, displacement_type, disaster_type,
                figure, event_name, sources, source_url, locations_name, month)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            records
        )

        # Build and insert aggregated summary rows
        if len(df) > 0:
            group_cols = ['country', 'Month', 'displacement_type', 'Disaster_Type']
            available_cols = [c for c in group_cols if c in df.columns]

            summary_df = df.groupby(
                available_cols, dropna=False
            ).agg(
                total_figure=('figure', 'sum'),
                event_count=('event_name', 'nunique')
            ).reset_index()

            summary_records = []
            for _, row in summary_df.iterrows():
                summary_records.append((
                    run_id,
                    row.get('country', ''),
                    row.get('Month', ''),
                    row.get('displacement_type', ''),
                    row.get('Disaster_Type', ''),
                    int(row['total_figure']),
                    int(row['event_count']),
                ))

            conn.executemany(
                """INSERT INTO run_summary
                   (run_id, country, month, displacement_type, disaster_type,
                    total_figure, event_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                summary_records
            )

        conn.commit()
        return run_id
    finally:
        conn.close()


def get_latest_run(db_path: Path, season: str, year: int) -> Optional[dict]:
    """Retrieve the most recent analysis run for a given season and year."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """SELECT * FROM analysis_runs
               WHERE season = ? AND year = ?
               ORDER BY executed_at DESC LIMIT 1""",
            (season, year)
        )
        columns = [desc[0] for desc in cursor.description]
        row = cursor.fetchone()
        if row is None:
            return None
        return dict(zip(columns, row))
    finally:
        conn.close()


def get_run_records(db_path: Path, run_id: int) -> pd.DataFrame:
    """Load all displacement records for a given run as a DataFrame."""
    conn = get_connection(db_path)
    try:
        return pd.read_sql_query(
            "SELECT * FROM displacement_records WHERE run_id = ?",
            conn, params=(run_id,)
        )
    finally:
        conn.close()


def list_runs(db_path: Path) -> pd.DataFrame:
    """List all analysis runs."""
    conn = get_connection(db_path)
    try:
        return pd.read_sql_query(
            "SELECT run_id, season, year, executed_at, records_count, "
            "total_displaced, countries_count, steps_executed, status "
            "FROM analysis_runs ORDER BY executed_at DESC",
            conn
        )
    finally:
        conn.close()
