"""Read the research snapshot without pretending latest revisions are historical archives."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

CLASS_NAMES = ("H", "D", "A")


def load_matches(path: str | Path) -> pd.DataFrame:
    if not Path(path).is_file():
        raise ValueError(f"Database not found: {path}")
    with sqlite3.connect(path) as db:
        frame = pd.read_sql_query("""
          WITH stats AS (
            SELECT *, ROW_NUMBER() OVER (
              PARTITION BY match_id,team_id,source ORDER BY available_at_utc DESC,rowid DESC) AS rn
            FROM team_match_stats WHERE source='football_data'
          )
          SELECT f.match_id,f.season_start AS season,f.home_team_id AS home_id,f.away_team_id AS away_id,
                 h.name AS home_team,a.name AS away_team,v.match_date,v.kickoff_utc,
                 r.home_goals,r.away_goals,r.result,
                 r.assumed_available_at_utc AS result_available,
                 hs.assumed_available_at_utc AS home_stats_available,
                 ast.assumed_available_at_utc AS away_stats_available,
                 hs.shots AS home_shots,ast.shots AS away_shots,
                 hs.shots_on_target AS home_sot,ast.shots_on_target AS away_sot
          FROM fixtures f
          JOIN latest_fixture_versions v ON v.match_id=f.match_id AND v.source='football_data'
          JOIN latest_match_results r ON r.match_id=f.match_id AND r.source='football_data'
          JOIN teams h ON h.team_id=f.home_team_id JOIN teams a ON a.team_id=f.away_team_id
          LEFT JOIN stats hs ON hs.match_id=f.match_id AND hs.team_id=f.home_team_id AND hs.rn=1
          LEFT JOIN stats ast ON ast.match_id=f.match_id AND ast.team_id=f.away_team_id AND ast.rn=1
          WHERE f.competition_id='ENG-PL' AND r.score_period='regulation'
          ORDER BY v.match_date,f.match_id
        """, db)
    if frame.empty or frame.match_id.duplicated().any():
        raise ValueError("Expected unique completed EPL matches")
    timestamps = frame[["result_available", "home_stats_available", "away_stats_available"]]
    if timestamps.isna().any().any():
        raise ValueError("Missing declared historical reporting delays; cannot build the retrospective baseline")
    frame["history_available_at"] = timestamps.apply(lambda col: pd.to_datetime(col, utc=True)).max(axis=1)
    frame["target"] = frame.result.map({c: i for i, c in enumerate(CLASS_NAMES)}).astype(int)
    if frame.history_available_at.isna().any():
        raise ValueError("Unparseable availability timestamps")
    return frame


def fingerprint(frame: pd.DataFrame) -> str:
    cols = ["match_id", "match_date", "home_goals", "away_goals", "home_shots", "away_shots",
            "home_sot", "away_sot", "history_available_at"]
    return hashlib.sha256(frame[cols].sort_values("match_id").to_csv(index=False).encode()).hexdigest()


def closing_benchmark(path: str | Path, matches: pd.DataFrame) -> pd.DataFrame:
    """One predeclared benchmark. Never optimize bookmaker choice against outcomes."""
    with sqlite3.connect(path) as db:
        quotes = pd.read_sql_query("""
          SELECT match_id,odds_home,odds_draw,odds_away,odds_over,odds_under,market
          FROM latest_odds_snapshots
          WHERE source='football_data' AND bookmaker='market_average'
            AND phase='closing' AND quality_flag IS NULL
        """, db)
    one = quotes[quotes.market == "1x2"].drop(columns=["market", "odds_over", "odds_under"])
    totals = quotes[quotes.market == "ou_2_5"][["match_id", "odds_over", "odds_under"]]
    out = matches[["match_id", "match_date", "target", "home_goals", "away_goals"]].merge(one, on="match_id", how="left", validate="one_to_one")
    out = out.merge(totals, on="match_id", how="left", validate="one_to_one")
    odds = out[["odds_home", "odds_draw", "odds_away"]].to_numpy(float)
    implied = 1 / odds
    out[["p_H", "p_D", "p_A"]] = implied / implied.sum(axis=1, keepdims=True)
    over, under = 1 / out.odds_over, 1 / out.odds_under
    out["p_over_2_5"] = over / (over + under)
    return out
