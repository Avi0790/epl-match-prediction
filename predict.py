#!/usr/bin/env python3
"""Predict a fixture at midnight London time using a trusted local model artifact."""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

from epl_model.data import CLASS_NAMES, load_matches
from epl_model.models import BoostedClassifier, EloClassifier, TeamPoisson
from epl_model.scores import summarize_score
from features import FEATURE_COLUMNS, FeatureConfig, HistoryEngine, day_cutoff
from epl_model.data_xg import load_matches_xg
from features_xg import FEATURE_COLUMNS_XG, XgFeatureConfig, XgHistoryEngine
from models_xg import LOGIT_XG_COLUMNS, BoostedClassifierXg, LogitXg


def resolve_team(name, matches):
    teams = pd.concat([matches[["home_id", "home_team"]].rename(columns={"home_id": "id", "home_team": "name"}),
                       matches[["away_id", "away_team"]].rename(columns={"away_id": "id", "away_team": "name"})]).drop_duplicates()
    exact = teams[(teams.name.str.casefold() == name.casefold()) | (teams.id == name)]
    if len(exact) != 1:
        raise ValueError(f"Unknown or ambiguous team {name!r}. Use --list-teams for exact names.")
    return exact.iloc[0].id, exact.iloc[0]["name"]


def check_horizon(bundle, fixture_date):
    cutoff = day_cutoff(fixture_date)
    if cutoff <= pd.Timestamp(bundle["information_through_utc"]):
        raise ValueError("This model used information at or after the requested prediction cutoff. Use an earlier research model or saved walk-forward predictions.")
    feature_set = bundle.get("feature_set", "baseline")
    if feature_set not in ("baseline", "xg"):
        raise ValueError("Unknown model feature set; retrain with this code")
    expected = FEATURE_COLUMNS_XG if feature_set == "xg" else FEATURE_COLUMNS
    if feature_set == "xg" and bundle.get("xg_history_version") != 2:
        raise ValueError("xG history version differs; retrain using the separate xG availability gate")
    if bundle.get("feature_columns") != expected:
        raise ValueError("Model and feature-code versions differ. Retrain with this version.")
    return cutoff


def key_factors(bundle, row, probabilities):
    model = bundle["classifier"].base
    selected_class = int(np.argmax(probabilities))
    if isinstance(model, (BoostedClassifier, BoostedClassifierXg)):
        import xgboost
        columns = FEATURE_COLUMNS_XG if isinstance(model, BoostedClassifierXg) else FEATURE_COLUMNS
        shap = model.model.get_booster().predict(xgboost.DMatrix(row[columns]), pred_contribs=True, strict_shape=True)[0, selected_class, :-1]
        indices = np.argsort(-np.abs(shap))[:5]
        return {"method": f"TreeSHAP for {CLASS_NAMES[selected_class]}, raw model margin before calibration; not causal",
                "factors": [{"feature": columns[i], "value": None if pd.isna(row.iloc[0][columns[i]]) else float(row.iloc[0][columns[i]]),
                             "contribution": float(shap[i])} for i in indices]}
    if isinstance(model, LogitXg):
        x = model._x(row)[0]
        factors = [{"feature": column,
                    "value": None if pd.isna(row.iloc[0][column]) else float(row.iloc[0][column]),
                    "scaled_model_input": float(x[i]),
                    "raw_class_margin_contribution": float(model.model.coef_[selected_class, i] * x[i])}
                   for i, column in enumerate(LOGIT_XG_COLUMNS)]
        return {"method": f"Linear contributions for {CLASS_NAMES[selected_class]} before calibration; missing differences use 0; not causal",
                "factors": sorted(factors, key=lambda f: -abs(f["raw_class_margin_contribution"]))}
    if isinstance(model, EloClassifier):
        value = float(row.iloc[0].elo_difference)
        return {"method": "Elo-only classifier: this is its only varying input; intercepts capture baseline home/draw/away rates",
                "factors": [{"feature": "elo_difference", "value": value,
                             "raw_class_margin_contribution": {label: float(model.model.coef_[k, 0] * value / 400) for k, label in enumerate(CLASS_NAMES)}}]}
    if isinstance(model, TeamPoisson):
        return {"method": "Regularized team attack/defence effects and home advantage",
                "factors": [{"feature": "home_attack_against_away_defence", "value": float(model.predict_rates(row)[0, 0])},
                            {"feature": "away_attack_against_home_defence", "value": float(model.predict_rates(row)[0, 1])}]}
    return {"method": "Unavailable", "factors": []}


def forecast(db, model_path, home, away, fixture_date):
    date.fromisoformat(fixture_date)
    # joblib/pickle can execute code: only load artifacts you created or trust.
    bundle = joblib.load(model_path)
    cutoff = check_horizon(bundle, fixture_date)
    uses_xg = bundle.get("feature_set", "baseline") == "xg"
    db = db or ("data/epl_xg.sqlite" if uses_xg else "data/epl.sqlite")
    matches = (load_matches_xg(db, last_complete_season=bundle.get("required_xg_through_season", 2025))
               if uses_xg else load_matches(db))
    home_id, home_name = resolve_team(home, matches)
    away_id, away_name = resolve_team(away, matches)
    config = (XgFeatureConfig if uses_xg else FeatureConfig)(**bundle["feature_config"])
    engine = (XgHistoryEngine if uses_xg else HistoryEngine)(matches, config)
    day = date.fromisoformat(fixture_date)
    season = day.year if day.month >= 7 else day.year - 1
    features = engine.fixture(home_id, away_id, fixture_date, season)
    row = pd.DataFrame([{**features, "home_id": home_id, "away_id": away_id}])
    p = bundle["classifier"].predict_proba(row)[0]
    rates = bundle["goals"].predict_rates(row)[0]
    scores, matrix = summarize_score(*rates, p)
    notes = []
    last = features["latest_history_match_date"]
    if last is None or (day - date.fromisoformat(last)).days > 21:
        notes.append("Match history is more than 21 days old or unavailable; refresh before relying on current strength estimates.")
    prior = matches[matches.match_date < fixture_date]
    for team, name in ((home_id, home_name), (away_id, away_name)):
        played = prior[(prior.home_id == team) | (prior.away_id == team)]
        if played.empty or (day - date.fromisoformat(played.match_date.max())).days > 120:
            notes.append(f"{name} has no recent EPL history; its form and rest are incomplete.")
    season_teams = set(matches.loc[matches.season == season, "home_id"]) | set(matches.loc[matches.season == season, "away_id"])
    if season_teams and (home_id not in season_teams or away_id not in season_teams):
        notes.append("At least one team has no observed EPL match in this season. This may be a hypothetical or out-of-competition fixture.")
    result = {"fixture": {"home": home_name, "away": away_name, "date": fixture_date,
                           "schedule_verified": False},
              "prediction_cutoff_utc": cutoff.isoformat(),
              "outcome_probabilities": dict(zip(CLASS_NAMES, map(float, p))),
              "most_likely_result": CLASS_NAMES[int(p.argmax())], **scores,
              "model": bundle["selection"]["classifier"], "goal_model": bundle["selection"]["goal_model"],
              "key_factors": key_factors(bundle, row, p),
              "context": {k: None if pd.isna(features[k]) else float(features[k]) for k in
                  ("home_elo", "away_elo", "home_form_points_5", "away_form_points_5", "home_epl_rest_days", "away_epl_rest_days")},
              "latest_available_history_match_date": last,
              "history_mode": bundle["history_mode"], "notes": notes,
              "scope": "Midnight forecast; EPL-only form/rest; no odds, injuries, xG or lineups. Input is not checked against a live schedule."}
    if uses_xg:
        xg_ready = pd.to_datetime(matches.xg_available, utc=True) + pd.Timedelta(hours=config.extra_xg_lag_hours)
        result_ready = pd.to_datetime(matches.history_available_at, utc=True) + pd.Timedelta(hours=config.extra_history_lag_hours)
        available = matches[xg_ready.lt(cutoff) & result_ready.lt(cutoff) & matches.match_date.lt(fixture_date)
                            & matches.home_xg.notna() & matches.away_xg.notna()]
        coverage = {}
        for team, name, side in ((home_id, home_name, "home"), (away_id, away_name, "away")):
            team_rows = available[(available.home_id == team) | (available.away_id == team)]
            latest = team_rows.match_date.max() if len(team_rows) else None
            count = int(features[f"{side}_xg_available_matches_10"])
            coverage[side] = {"team": name, "latest_available_xg_match_date": latest,
                              "available_xg_matches_in_last_10": count}
            if latest is None or (day - date.fromisoformat(latest)).days > 21:
                notes.append(f"{name}: latest available xG is {latest or 'missing'}; refresh xG before treating this as a current forecast.")
            if count < 3:
                notes.append(f"{name}: only {count} of its last 10 result-available matches have usable xG; form estimates are incomplete.")
        result["xg_coverage"] = coverage
        result["scope"] = ("Midnight forecast; lagged Understat xG with declared retrospective availability; "
                           "EPL-only form/rest; no odds, injuries or lineups. Goal regressor uses baseline features. "
                           "Input is not checked against a live schedule.")
    return result, matrix


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", help="Defaults to data/epl_xg.sqlite for an xG model; otherwise data/epl.sqlite")
    p.add_argument("--model", default="models/production.joblib")
    p.add_argument("--home")
    p.add_argument("--away")
    p.add_argument("--date")
    p.add_argument("--list-teams", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--matrix-json", type=Path)
    args = p.parse_args()
    try:
        if args.list_teams:
            matches = load_matches(args.db or "data/epl.sqlite")
            print("\n".join(sorted(set(matches.home_team) | set(matches.away_team))))
            return 0
        if not all([args.home, args.away, args.date]):
            p.error("--home, --away and --date are required unless using --list-teams")
        result, matrix = forecast(args.db, args.model, args.home, args.away, args.date)
        if args.matrix_json:
            args.matrix_json.parent.mkdir(parents=True, exist_ok=True)
            args.matrix_json.write_text(json.dumps({"axes": "rows=home goals, columns=away goals", "matrix": matrix.tolist(), "poisson_tail_mass_omitted": result["poisson_tail_mass_omitted"]}, indent=2) + "\n")
        if args.json:
            print(json.dumps(result, indent=2, allow_nan=False))
        else:
            f = result["fixture"]
            print(f"{f['home']} vs {f['away']} | {f['date']} | supplied fixture, schedule not verified")
            print("  ".join(f"{label}: {probability:.1%}" for label, probability in result["outcome_probabilities"].items()))
            print(f"Expected goals: {result['expected_home_goals']:.2f}–{result['expected_away_goals']:.2f}")
            print("Top scores: " + ", ".join(f"{s['home_goals']}–{s['away_goals']} ({s['probability']:.1%})" for s in result["top_5_scorelines"]))
            print(f"Over 2.5: {result['p_over_2_5']:.1%} | model: {result['model']}")
            print("Key factors: " + json.dumps(result["key_factors"]))
            print(f"Latest history: {result['latest_available_history_match_date']} | {result['scope']}")
            if "xg_coverage" in result:
                print("xG coverage: " + json.dumps(result["xg_coverage"]))
            for note in result["notes"]:
                print("Note: " + note)
        return 0
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
