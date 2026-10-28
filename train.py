#!/usr/bin/env python3
"""Expanding walk-forward selection, untouched final-season evaluation, and deployment fit."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from epl_model.data import closing_benchmark, fingerprint, load_matches
from epl_model.models import (CLASSIFIER_SPECS, GOAL_SPECS, CalibratedModel,
                              PlattCalibrator, fit_classifier, fit_goal_model)
from epl_model.scores import summarize_score
from epl_model.splits import describe, season_split
from evaluate import (calibration_plot, classification_metrics, goal_nll,
                      paired_comparison, score_metrics)
from features import FEATURE_COLUMNS, FeatureConfig, build_features, day_cutoff


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def probabilities_frame(frame, name, p):
    out = frame[["match_id", "season", "match_date", "target", "home_goals", "away_goals"]].copy()
    out["model"] = name
    out[["p_H", "p_D", "p_A"]] = p
    return out


def fit_candidates(training, calibration, evaluation):
    models, predictions, goals = {}, {}, {}
    for name in CLASSIFIER_SPECS:
        base = fit_classifier(name, training)
        raw = base.predict_proba(evaluation)
        platt = PlattCalibrator().fit(base.predict_proba(calibration), calibration.target)
        models[name] = CalibratedModel(base)
        models[name + "+platt"] = CalibratedModel(base, platt)
        predictions[name] = raw
        predictions[name + "+platt"] = platt.transform(raw)
        if name == "team_poisson":
            goals[name] = base
    goals["xgb_poisson"] = fit_goal_model("xgb_poisson", training)
    return models, predictions, goals


def make_bundle(classifier, goals, selection, training, calibration, config, snapshot, purpose):
    return {"format_version": 1, "classifier": classifier, "goals": goals,
            "selection": selection, "feature_config": asdict(config), "feature_columns": FEATURE_COLUMNS,
            "history_mode": config.history_mode, "training": describe(training), "calibration": describe(calibration),
            "information_through_utc": (pd.to_datetime(pd.concat([training, calibration]).history_available_at, utc=True).max() + pd.Timedelta(hours=config.extra_history_lag_hours)).isoformat(),
            "snapshot_sha256": snapshot, "purpose": purpose,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__, "sklearn": sklearn.__version__, "xgboost": xgboost.__version__}}


def atomic_dump(bundle, output):
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    joblib.dump(bundle, temporary, compress=3)
    os.replace(temporary, output)
    metadata = {k: v for k, v in bundle.items() if k not in ("classifier", "goals")}
    metadata["artifact_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    write_json(output.with_suffix(".json"), metadata)


def explain_xgb(model, calibration, output):
    """Native TreeSHAP on the pre-test calibration block; raw margins, not probabilities."""
    matrix = xgboost.DMatrix(calibration[FEATURE_COLUMNS])
    values = model.model.get_booster().predict(matrix, pred_contribs=True, strict_shape=True)
    # (matches, classes, features + bias). Aggregate classes without claiming causality.
    importance = np.abs(values[:, :, :-1]).mean(axis=(0, 1))
    pd.DataFrame({"feature": FEATURE_COLUMNS, "mean_abs_SHAP_raw_margin": importance}).sort_values(
        "mean_abs_SHAP_raw_margin", ascending=False).to_csv(output, index=False)


def fit_production(db, selection_path, output, as_of=None):
    selection = json.loads(Path(selection_path).read_text())
    config = FeatureConfig(**selection["feature_config"])
    matches = load_matches(db)
    now = pd.Timestamp(as_of) if as_of else pd.Timestamp.now(tz="UTC")
    if now.tzinfo is None:
        raise ValueError("--as-of must specify a timezone")
    matches = matches[pd.to_datetime(matches.history_available_at, utc=True) + pd.Timedelta(hours=config.extra_history_lag_hours) < now].copy()
    frame = build_features(matches, config)
    eligible = frame[frame.season >= 2015].sort_values(["match_date", "match_id"])
    if len(eligible) < 1000:
        raise ValueError("At least 1,000 eligible matches are needed for production fitting")
    cal_start = eligible.iloc[-380].match_date
    calibration = eligible[eligible.match_date >= cal_start].copy()
    training = eligible[(eligible.match_date < cal_start) &
        (pd.to_datetime(eligible.history_available_at, utc=True) + pd.Timedelta(hours=config.extra_history_lag_hours) < day_cutoff(cal_start))].copy()
    name = selection["classifier"].removesuffix("+platt")
    base = fit_classifier(name, training)
    platt = PlattCalibrator().fit(base.predict_proba(calibration), calibration.target) if selection["classifier"].endswith("+platt") else None
    classifier = CalibratedModel(base, platt)
    goals = base if name == selection["goal_model"] == "team_poisson" else fit_goal_model(selection["goal_model"], training)
    bundle = make_bundle(classifier, goals, selection, training, calibration, config,
                         fingerprint(matches), "production; final test report is unchanged")
    # Numerical smoke check is not a claim of better future performance.
    p = classifier.predict_proba(calibration.tail(5)); rates = goals.predict_rates(calibration.tail(5))
    for rate, probability in zip(rates, p):
        summarize_score(*rate, probability)
    atomic_dump(bundle, output)
    print(f"Production model saved to {output}; calibration starts {cal_start}", flush=True)
    return bundle


def backtest(db, report_dir, model_dir, final_season=2025, overwrite=False, extra_lag=0):
    reports, models_dir = Path(report_dir), Path(model_dir)
    if (reports / "test_report.json").exists() and not overwrite:
        raise ValueError("A final test report already exists. Use --overwrite only to reproduce, not to tune against it.")
    reports.mkdir(parents=True, exist_ok=True); models_dir.mkdir(parents=True, exist_ok=True)
    config = FeatureConfig(extra_history_lag_hours=extra_lag)
    all_matches = load_matches(db)
    matches = all_matches[all_matches.season <= final_season].copy()
    counts = matches.groupby("season").size()
    if any(counts.get(s, 0) != 380 for s in range(2014, final_season + 1)):
        raise ValueError("Backtest requires 380 completed matches in every season from 2014 through the final season")
    snapshot = fingerprint(matches)
    frame = build_features(matches, config)
    frame.to_csv(reports / "feature_snapshot.csv", index=False)
    validation_records, folds, goal_records = [], [], []
    # Last pre-test season belongs only to final calibration, never model selection.
    for year in range(2018, final_season - 1):
        training, calibration, validation = season_split(frame, year)
        print(f"Walk-forward: train through {year-2}, calibrate {year-1}/{str(year)[-2:]}, validate {year}/{str(year+1)[-2:]}", flush=True)
        _, predictions, goal_models = fit_candidates(training, calibration, validation)
        for name, p in predictions.items():
            validation_records.append(probabilities_frame(validation, name, p))
        for name, model in goal_models.items():
            goal_records.append({"season": year, "model": name, "goal_negative_log_likelihood": goal_nll(validation, model.predict_rates(validation))})
        folds.append({"validation_season": year, "train": describe(training), "calibration": describe(calibration), "validation": describe(validation)})
    if not validation_records:
        raise ValueError("Insufficient seasons for walk-forward selection")
    validation_predictions = pd.concat(validation_records, ignore_index=True)
    validation_predictions.to_csv(reports / "walk_forward_predictions.csv", index=False)
    validation_summary = {name: classification_metrics(group.target, group[["p_H", "p_D", "p_A"]])
        for name, group in validation_predictions.groupby("model")}
    selected = min(validation_summary, key=lambda key: validation_summary[key]["log_loss"])
    goal_summary = pd.DataFrame(goal_records).groupby("model").goal_negative_log_likelihood.mean().to_dict()
    chosen_goal = min(goal_summary, key=goal_summary.get)
    selection = {"classifier": selected, "goal_model": chosen_goal, "selected_by": "pooled walk-forward log loss; goal model by mean goal NLL",
                 "feature_config": asdict(config), "classifier_specs": CLASSIFIER_SPECS,
                 "validation_seasons": [fold["validation_season"] for fold in folds],
                 "final_test_season": final_season, "snapshot_sha256": snapshot,
                 "validation_metrics": validation_summary, "goal_validation_nll": goal_summary}
    # Selection is written before any final-test predictions or scores are computed.
    write_json(reports / "selection.json", selection)
    write_json(reports / "splits.json", folds)
    write_json(reports / "goal_walk_forward.json", goal_records)
    print(f"Frozen selection: {selected}; score model: {chosen_goal}", flush=True)
    training, calibration, test = season_split(frame, final_season)
    fitted, predictions, goal_models = fit_candidates(training, calibration, test)
    metrics = {name: classification_metrics(test.target, p) for name, p in predictions.items()}
    test_predictions = pd.concat([probabilities_frame(test, name, p) for name, p in predictions.items()], ignore_index=True)
    book = closing_benchmark(db, test)
    mask = book[["p_H", "p_D", "p_A"]].notna().all(axis=1).to_numpy()
    if not mask.any():
        raise ValueError("No market-average closing odds in final test")
    book_p = book.loc[mask, ["p_H", "p_D", "p_A"]].to_numpy()
    benchmark_metrics = classification_metrics(book.loc[mask, "target"], book_p)
    test_predictions = pd.concat([test_predictions, probabilities_frame(test.iloc[np.flatnonzero(mask)], "market_average_closing", book_p)], ignore_index=True)
    test_predictions.to_csv(reports / "test_predictions.csv", index=False)
    book.to_csv(reports / "closing_benchmark.csv", index=False)
    paired = paired_comparison(test.target.to_numpy()[mask], predictions[selected][mask], book_p, test.match_date.to_numpy()[mask])
    rates = goal_models[chosen_goal].predict_rates(test)
    scores = []
    for rec, rate, p in zip(test.to_dict("records"), rates, predictions[selected]):
        summary, _ = summarize_score(*rate, p)
        top = summary["most_likely_score"]
        scores.append({"match_id": rec["match_id"], "home_team": rec["home_team"], "away_team": rec["away_team"],
            "match_date": rec["match_date"], "home_goals": rec["home_goals"], "away_goals": rec["away_goals"],
            "pred_home_goals": top["home_goals"], "pred_away_goals": top["away_goals"],
            "p_over_2_5": summary["p_over_2_5"], "expected_home_goals": summary["expected_home_goals"],
            "expected_away_goals": summary["expected_away_goals"],
            "actual_in_top_5": int(any(s["home_goals"] == rec["home_goals"] and s["away_goals"] == rec["away_goals"] for s in summary["top_5_scorelines"])),
            "top_5_json": json.dumps(summary["top_5_scorelines"])})
    score_frame = pd.DataFrame(scores); score_frame.to_csv(reports / "score_predictions.csv", index=False)
    totals_mask = book.p_over_2_5.notna().to_numpy()
    book_over_truth = (book.home_goals + book.away_goals > 2).to_numpy()[totals_mask]
    book_over_p = book.p_over_2_5.to_numpy()[totals_mask]
    totals = {"matches": int(totals_mask.sum()),
              "bookmaker_accuracy": float(((book_over_p >= .5) == book_over_truth).mean()) if totals_mask.any() else None,
              "model_accuracy_same_matches": float(((score_frame.p_over_2_5.to_numpy()[totals_mask] >= .5) == book_over_truth).mean()) if totals_mask.any() else None}
    report = {"experiment": "retrospective delayed-archive; midnight Europe/London forecasts; no odds features",
        "selected_classifier": selected, "selected_goal_model": chosen_goal, "final_test_season": final_season,
        "train": describe(training), "calibration": describe(calibration), "test": describe(test),
        "test_model_metrics": metrics, "market_average_closing": benchmark_metrics,
        "selected_model_on_benchmark_matches": classification_metrics(test.target.to_numpy()[mask], predictions[selected][mask]),
        "paired_comparison": paired, "score_metrics": score_metrics(score_frame), "over_under_benchmark": totals,
        "beats_closing_on_log_loss": bool(paired["log_loss"]["model_minus_bookmaker"] < 0),
        "snapshot_sha256": snapshot,
        "limits": ["Closing odds have later information than midnight forecasts; quote times are not supplied",
                   "Original publication and revision times are unknown; availability uses declared conservative lags",
                   "Rest and congestion include EPL only", "Single final test season; no return-on-investment claim",
                   "H/D/A and scorelines are reconciled by scaling Poisson outcome regions; this does not validate independence"]}
    write_json(reports / "test_report.json", report)
    calibration_plot([(selected, test.target, predictions[selected]), ("Closing market", book.loc[mask, "target"], book_p)], reports / "calibration.png")
    explain_xgb(fitted["xgb_shallow"].base, calibration, reports / "xgb_shap.csv")
    bundle = make_bundle(fitted[selected], goal_models[chosen_goal], selection, training, calibration, config, snapshot, "final-test research model")
    atomic_dump(bundle, models_dir / "research.joblib")
    print(json.dumps({"selected": selected, "model": metrics[selected], "closing": benchmark_metrics,
                      "score_metrics": report["score_metrics"]}, indent=2), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default="data/epl.sqlite")
    p.add_argument("--reports", default="reports")
    p.add_argument("--models", default="models")
    p.add_argument("--final-season", type=int, default=2025)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--extra-history-lag-hours", type=float, default=0)
    p.add_argument("--production-only", action="store_true")
    p.add_argument("--selection", default="reports/selection.json")
    p.add_argument("--as-of", help="Production fit cutoff, e.g. 2026-10-15T00:00:00Z")
    args = p.parse_args()
    if not args.production_only:
        backtest(args.db, args.reports, args.models, args.final_season, args.overwrite, args.extra_history_lag_hours)
    else:
        fit_production(args.db, args.selection, Path(args.models) / "production.joblib", args.as_of)


if __name__ == "__main__":
    main()
