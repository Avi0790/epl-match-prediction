"""Load the research snapshot plus Understat xG, without touching the baseline loader.

xG enters only through the same declared retrospective assumption the project already
uses for results and shot statistics (`assumed_available_at_utc`). It is NOT an observed
historical publication time.
"""
from __future__ import annotations

from pathlib import Path
import sqlite3

import pandas as pd

from epl_model.data import fingerprint as _baseline_fingerprint, load_matches

XG_SOURCE = "understat"


def read_xg(path: str | Path, source: str = XG_SOURCE) -> pd.DataFrame:
    """One row per (match, team): the latest observation of each, from one provider."""
    with sqlite3.connect(path) as db:
        return pd.read_sql_query("""
          WITH ranked AS (
            SELECT *, ROW_NUMBER() OVER (
              PARTITION BY match_id, team_id, source ORDER BY available_at_utc DESC, rowid DESC) AS rn
            FROM advanced_team_stats WHERE source = ?
          )
          SELECT match_id, team_id, xg_for, xg_against, definition, assumed_available_at_utc
          FROM ranked WHERE rn = 1
        """, db, params=(source,))


def attach_xg(matches: pd.DataFrame, xg: pd.DataFrame, last_complete_season: int | None = None) -> pd.DataFrame:
    """Add home_xg / away_xg / xg_available without delaying baseline history.

    Rules (each raises instead of guessing):
      * every match must have xG for BOTH teams, except seasons after `last_complete_season`
        (those stay NaN, never 0);
      * the two sides must agree (home xg_against == away xg_for, and vice versa);
      * xG needs a declared availability assumption;
      * one xG definition only.
    """
    if xg.empty:
        raise ValueError("No xG rows found; import them first")
    if xg.definition.nunique() != 1:
        raise ValueError("Mixed xG definitions; do not combine providers or model versions")
    if xg.duplicated(["match_id", "team_id"]).any():
        raise ValueError("Duplicate xG rows for a match and team")
    if xg.assumed_available_at_utc.isna().any():
        raise ValueError("xG rows without a declared availability assumption")

    out = matches.copy()
    home = xg.rename(columns={"team_id": "home_id", "xg_for": "home_xg", "xg_against": "home_xg_against",
                              "assumed_available_at_utc": "home_xg_available"})[
        ["match_id", "home_id", "home_xg", "home_xg_against", "home_xg_available"]]
    away = xg.rename(columns={"team_id": "away_id", "xg_for": "away_xg", "xg_against": "away_xg_against",
                              "assumed_available_at_utc": "away_xg_available"})[
        ["match_id", "away_id", "away_xg", "away_xg_against", "away_xg_available"]]
    out = out.merge(home, on=["match_id", "home_id"], how="left", validate="one_to_one")
    out = out.merge(away, on=["match_id", "away_id"], how="left", validate="one_to_one")

    has_both = out.home_xg.notna() & out.away_xg.notna()
    if last_complete_season is not None:
        required = out.season <= last_complete_season
    else:
        required = pd.Series(True, index=out.index)
    if (required & ~has_both).any():
        missing = out.loc[required & ~has_both, "match_id"].head(5).tolist()
        raise ValueError(f"{int((required & ~has_both).sum())} matches are missing xG, e.g. {missing}")
    if (out.home_xg.notna() ^ out.away_xg.notna()).any():
        raise ValueError("A match has xG for only one team")
    both = out[has_both]
    if ((both.home_xg - both.away_xg_against).abs() > 1e-9).any() or ((both.away_xg - both.home_xg_against).abs() > 1e-9).any():
        raise ValueError("Home and away xG rows disagree for some match")

    avail_home = pd.to_datetime(out.home_xg_available, utc=True)
    avail_away = pd.to_datetime(out.away_xg_available, utc=True)
    out["xg_available"] = pd.concat([avail_home, avail_away], axis=1).max(axis=1)
    # Baseline result/statistic availability must not depend on the xG feed.
    # XgHistoryEngine gates xG separately, after both the result and xG qualify.
    return out.drop(columns=["home_xg_against", "away_xg_against", "home_xg_available", "away_xg_available"])


def load_matches_xg(path: str | Path, last_complete_season: int = 2025) -> pd.DataFrame:
    return attach_xg(load_matches(path), read_xg(path), last_complete_season)


def fingerprint_xg(frame: pd.DataFrame) -> str:
    """Baseline fingerprint plus the xG columns, so a changed xG import changes the snapshot hash."""
    import hashlib
    extra = frame[["match_id", "home_xg", "away_xg", "xg_available"]].sort_values("match_id").to_csv(index=False)
    return hashlib.sha256((_baseline_fingerprint(frame) + extra).encode()).hexdigest()
