"""Coverage is measured, never assumed from a provider name."""
import sqlite3
from .store import utc_now


def audit(db: sqlite3.Connection) -> dict:
    seasons = []
    for record in db.execute("""
        SELECT f.season_start, COUNT(*) AS fixtures, COUNT(r.match_id) AS completed,
               SUM(v.kickoff_utc IS NULL) AS missing_kickoff_time,
               SUM(v.referee IS NULL) AS missing_referee
        FROM fixtures f
        JOIN latest_fixture_versions v ON f.match_id=v.match_id AND v.source='football_data'
        LEFT JOIN latest_match_results r ON r.match_id=f.match_id AND r.source='football_data'
        WHERE f.competition_id='ENG-PL'
        GROUP BY f.season_start ORDER BY f.season_start
    """):
        item = dict(record)
        year = item["season_start"]
        item["closing_1x2_by_bookmaker"] = {r[0]: r[1] for r in db.execute("""
            SELECT o.bookmaker, COUNT(*) FROM latest_odds_snapshots o
            JOIN fixtures f ON f.match_id=o.match_id
            WHERE f.season_start=? AND f.competition_id='ENG-PL'
              AND o.phase='closing' AND o.market='1x2'
            GROUP BY o.bookmaker
        """, (year,))}
        item["unflagged_market_average_closing"] = db.execute("""
            SELECT COUNT(*) FROM latest_odds_snapshots o JOIN fixtures f USING(match_id)
            WHERE f.season_start=? AND o.bookmaker='market_average' AND o.phase='closing'
              AND o.market='1x2' AND o.quality_flag IS NULL
        """, (year,)).fetchone()[0]
        item["complete_380_results"] = item["completed"] == 380
        seasons.append(item)
    tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    counts = {table: db.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] for table in tables}
    return {"generated_at_utc": utc_now(), "seasons": seasons, "table_row_counts": counts,
            "data_issues": [dict(r) for r in db.execute("SELECT * FROM data_issues ORDER BY match_id")],
            "foreign_key_errors": [list(row) for row in db.execute("PRAGMA foreign_key_check")],
            "integrity_check": db.execute("PRAGMA integrity_check").fetchone()[0],
            "warnings": [
                "Latest views are for audit/labels, not historical feature joins.",
                "No features or model evaluations have been run.",
                "Historical downloads do not prove historical publication times.",
                "EPL-only fixture data cannot measure full cup/European fixture congestion.",
                "No xG, player, injury or Elo coverage is inferred from missing rows.",
                "Pinnacle quotes on/after 2025-07-23 are flagged and must be excluded by default.",
            ]}
