#!/usr/bin/env python3
"""Optional Understat acquisition through soccerdata; offline runs need no download.

Install requirements-xg-data.txt in a separate acquisition environment. Historical
snapshots and observed download times do not establish historical publication times.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd


def validate_schedule(frame, expected_seasons, allow_incomplete=False):
    wanted = ["season", "date", "home_team", "away_team", "home_goals", "away_goals", "home_xg", "away_xg"]
    missing = set(wanted) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing Understat columns: {sorted(missing)}")
    out = frame[wanted].dropna(subset=["home_xg", "away_xg", "home_goals", "away_goals"]).copy()
    from understat_to_jsonl import season_start
    years = out.season.map(season_start)
    counts = years.value_counts().to_dict()
    if set(counts) != set(expected_seasons):
        raise ValueError(f"Unexpected season coverage: {counts}")
    if any(n > 380 or (n != 380 and not allow_incomplete) for n in counts.values()):
        raise ValueError(f"Expected 380 completed matches per season: {counts}")
    keys = out.assign(season_start=years)
    if keys.duplicated(["season_start", "home_team", "away_team"]).any():
        raise ValueError("Duplicate season/home/away fixture")
    numeric = out[["home_xg", "away_xg", "home_goals", "away_goals"]].to_numpy(float)
    if not np.isfinite(numeric).all() or (numeric < 0).any():
        raise ValueError("Goals and xG must be finite and nonnegative")
    return out, counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start-season", type=int, default=2014)
    p.add_argument("--end-season", type=int, default=2025)
    p.add_argument("--output", type=Path, default=Path("data/understat_xg.csv"))
    p.add_argument("--allow-incomplete", action="store_true", help="Permit a partial current season")
    a = p.parse_args()
    if a.end_season < a.start_season:
        p.error("end-season must be at least start-season")
    try:
        import soccerdata as sd
    except ImportError:
        p.error("Install optional requirements-xg-data.txt to download; bundled data works offline")
    years = list(range(a.start_season, a.end_season + 1))
    seasons = [f"{y}-{y + 1}" for y in years]
    provider = sd.Understat(leagues="ENG-Premier League", seasons=seasons)
    out, counts = validate_schedule(provider.read_schedule().reset_index(), years, a.allow_incomplete)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.output, index=False)
    metadata = {"source": "Understat via soccerdata", "soccerdata_version": version("soccerdata"),
                "observed_at_utc": datetime.now(timezone.utc).isoformat(), "requested_seasons": seasons,
                "matches_by_season": counts, "publication_times": "unknown; observation may reuse provider cache"}
    a.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {len(out)} completed matches to {a.output}; seasons: {counts}")


if __name__ == "__main__":
    main()
