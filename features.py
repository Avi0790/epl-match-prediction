#!/usr/bin/env python3
"""Build pre-match Elo, exponentially weighted form, and explicitly EPL-only rest.

This stage is a retrospective delayed-archive experiment, NOT an observed-time
historical archive. Original source publication times remain unknown.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from epl_model.data import fingerprint, load_matches

METRICS = ("points", "gf", "ga", "shots_for", "shots_against", "sot_for", "sot_against")
WINDOWS = (3, 5, 10)


@dataclass(frozen=True)
class FeatureConfig:
    history_mode: str = "retrospective_lagged"
    decay: float = 0.8
    elo_initial: float = 1500.0
    elo_new_entrant: float = 1450.0
    elo_home_advantage: float = 70.0
    elo_k: float = 20.0
    elo_season_retention: float = 0.75
    extra_history_lag_hours: float = 0.0


def day_cutoff(day: str | date) -> pd.Timestamp:
    d = date.fromisoformat(day) if isinstance(day, str) else day
    return pd.Timestamp(datetime.combine(d, time.min, ZoneInfo("Europe/London"))).tz_convert("UTC")


def _weighted(records, field: str, n: int, decay: float) -> float:
    items = list(records)[-n:]
    if not items:
        return np.nan
    values = np.array([item[field] for item in items], dtype=float)
    weights = decay ** np.arange(len(items) - 1, -1, -1)
    valid = np.isfinite(values)
    return float(np.average(values[valid], weights=weights[valid])) if valid.any() else np.nan


def feature_names() -> list[str]:
    cols = []
    for side in ("home", "away"):
        cols += [f"{side}_{key}" for key in ("elo", "promoted_or_returning", "epl_rest_days",
                 "epl_matches_14d", "epl_matches_30d", "epl_short_rest_3d", "epl_games_this_season")]
        for kind in ("form", "venue"):
            for n in WINDOWS:
                cols += [f"{side}_{kind}_n_{n}"] + [f"{side}_{kind}_{metric}_{n}" for metric in METRICS]
    cols += ["elo_difference", "epl_rest_difference"]
    for kind in ("form", "venue"):
        for n in WINDOWS:
            cols += [f"diff_{kind}_{metric}_{n}" for metric in METRICS]
    return cols


FEATURE_COLUMNS = feature_names()


class HistoryEngine:
    def __init__(self, matches: pd.DataFrame, config: FeatureConfig = FeatureConfig()):
        if config.history_mode != "retrospective_lagged":
            raise ValueError("Only explicitly labeled retrospective_lagged mode is supported by this baseline")
        if not 0 < config.decay <= 1 or config.extra_history_lag_hours < 0:
            raise ValueError("Invalid feature decay or history delay")
        self.config = config
        self.matches = matches.sort_values(["match_date", "match_id"]).copy()
        self.matches["usable_at"] = pd.to_datetime(self.matches.history_available_at, utc=True) + pd.Timedelta(hours=config.extra_history_lag_hours)
        self.events = self.matches.sort_values(["usable_at", "match_date", "match_id"]).to_dict("records")
        self.schedule = self.matches.to_dict("records")
        self.event_pos = self.schedule_pos = 0
        self.rating = {}
        self.form = defaultdict(lambda: deque(maxlen=10))
        self.venue = defaultdict(lambda: deque(maxlen=10))
        self.played_dates = defaultdict(list)
        self.members = defaultdict(set)
        self.season_games = defaultdict(int)
        self.initialized = set()
        self.first_season = int(matches.season.min()) if len(matches) else 2014
        self.current_season = self.first_season
        self.latest_available = None
        self.latest_history_date = None
        self.last_cutoff = None

    def _season(self, season):
        if season > self.current_season:
            factor = self.config.elo_season_retention ** (season - self.current_season)
            self.rating = {t: self.config.elo_initial + factor * (r - self.config.elo_initial)
                           for t, r in self.rating.items()}
            self.current_season = season

    def _team(self, team, season):
        self._season(season)
        if (team, season) not in self.initialized:
            newcomer = season > self.first_season and team not in self.members[season - 1]
            if newcomer:
                self.rating[team] = self.config.elo_new_entrant
                self.form[team].clear()
                self.venue[(team, "home")].clear()
                self.venue[(team, "away")].clear()
            else:
                self.rating.setdefault(team, self.config.elo_initial)
            self.initialized.add((team, season))

    def _result(self, rec):
        h, a, season = rec["home_id"], rec["away_id"], int(rec["season"])
        self._team(h, season)
        self._team(a, season)
        expected = 1 / (1 + 10 ** (-(self.rating[h] + self.config.elo_home_advantage - self.rating[a]) / 400))
        outcome = 1.0 if rec["home_goals"] > rec["away_goals"] else 0.5 if rec["home_goals"] == rec["away_goals"] else 0.0
        change = self.config.elo_k * (outcome - expected)
        self.rating[h] += change
        self.rating[a] -= change
        for team, venue, own, opp in ((h, "home", "home", "away"), (a, "away", "away", "home")):
            gf, ga = rec[f"{own}_goals"], rec[f"{opp}_goals"]
            record = {"points": 3 if gf > ga else 1 if gf == ga else 0, "gf": gf, "ga": ga,
                      "shots_for": rec[f"{own}_shots"], "shots_against": rec[f"{opp}_shots"],
                      "sot_for": rec[f"{own}_sot"], "sot_against": rec[f"{opp}_sot"]}
            self.form[team].append(record)
            self.venue[(team, venue)].append(record)
        self.latest_available = rec["usable_at"]
        self.latest_history_date = max(self.latest_history_date or "", rec["match_date"])

    def advance(self, day: str):
        cutoff = day_cutoff(day)
        if self.last_cutoff is not None and cutoff < self.last_cutoff:
            raise ValueError("History must advance chronologically")
        # Schedule history needs only a previously played date, not the target's future result.
        while self.schedule_pos < len(self.schedule) and self.schedule[self.schedule_pos]["match_date"] < day:
            rec = self.schedule[self.schedule_pos]
            for team in (rec["home_id"], rec["away_id"]):
                self.played_dates[team].append(date.fromisoformat(rec["match_date"]))
                self.members[int(rec["season"])].add(team)
                self.season_games[(team, int(rec["season"]))] += 1
            self.schedule_pos += 1
        while self.event_pos < len(self.events) and self.events[self.event_pos]["usable_at"] < cutoff:
            rec = self.events[self.event_pos]
            if rec["match_date"] >= day:
                raise ValueError("A result's declared availability precedes its match date")
            self._result(rec)
            self.event_pos += 1
        self.last_cutoff = cutoff

    def fixture(self, home: str, away: str, day: str, season: int) -> dict:
        if home == away:
            raise ValueError("Home and away teams must differ")
        self.advance(day)
        d, result = date.fromisoformat(day), {}
        for team, side in ((home, "home"), (away, "away")):
            self._team(team, season)
            dates = self.played_dates[team]
            rest = (d - dates[-1]).days if dates else np.nan
            result.update({f"{side}_elo": self.rating[team],
                f"{side}_promoted_or_returning": int(season > self.first_season and team not in self.members[season - 1]),
                f"{side}_epl_rest_days": float(rest),
                f"{side}_epl_matches_14d": sum(0 < (d - previous).days <= 14 for previous in dates),
                f"{side}_epl_matches_30d": sum(0 < (d - previous).days <= 30 for previous in dates),
                f"{side}_epl_short_rest_3d": int(np.isfinite(rest) and rest <= 3),
                f"{side}_epl_games_this_season": self.season_games[(team, season)]})
            for kind, records in (("form", self.form[team]), ("venue", self.venue[(team, side)])):
                for n in WINDOWS:
                    result[f"{side}_{kind}_n_{n}"] = min(len(records), n)
                    for field in METRICS:
                        result[f"{side}_{kind}_{field}_{n}"] = _weighted(records, field, n, self.config.decay)
        result["elo_difference"] = result["home_elo"] - result["away_elo"]
        result["epl_rest_difference"] = result["home_epl_rest_days"] - result["away_epl_rest_days"]
        for kind in ("form", "venue"):
            for n in WINDOWS:
                for field in METRICS:
                    result[f"diff_{kind}_{field}_{n}"] = result[f"home_{kind}_{field}_{n}"] - result[f"away_{kind}_{field}_{n}"]
        result["prediction_cutoff_utc"] = self.last_cutoff.isoformat()
        result["latest_history_available_at"] = self.latest_available.isoformat() if self.latest_available is not None else None
        result["latest_history_match_date"] = self.latest_history_date
        return result


def build_features(matches: pd.DataFrame, config: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
    engine = HistoryEngine(matches, config)
    output = []
    for rec in matches.sort_values(["match_date", "match_id"]).to_dict("records"):
        values = engine.fixture(rec["home_id"], rec["away_id"], rec["match_date"], int(rec["season"]))
        output.append({**rec, **values})
    return pd.DataFrame(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default="data/epl.sqlite")
    p.add_argument("--output", type=Path, default=Path("data/features.csv"))
    p.add_argument("--extra-history-lag-hours", type=float, default=0)
    args = p.parse_args()
    matches = load_matches(args.db)
    config = FeatureConfig(extra_history_lag_hours=args.extra_history_lag_hours)
    output = build_features(matches, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    args.output.with_suffix(".metadata.json").write_text(json.dumps({"config": asdict(config),
        "snapshot_sha256": fingerprint(matches), "rows": len(output), "feature_columns": FEATURE_COLUMNS,
        "warning": "Reconstructed history with declared lags; original publication/revision timing is unknown"}, indent=2) + "\n")
    print(f"Wrote {len(output)} rows and {len(FEATURE_COLUMNS)} numeric features to {args.output}")


if __name__ == "__main__":
    main()
