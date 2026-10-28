#!/usr/bin/env python3
"""Separate xG experiment: same splits, same selection rule, own reports folder.

Selection uses ONLY the walk-forward validation seasons and is written to disk before
the final test season is scored. The final test season was already looked at once by the
baseline run, so this run is a SECOND look at it and says so in its report.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from epl_model.data import closing_benchmark
from epl_model.data_xg import fingerprint_xg, load_matches_xg
from epl_model.models import GOAL_SPECS, CalibratedModel, PlattCalibrator, fit_goal_model
from epl_model.splits import describe, season_split
from evaluate import classification_metrics, paired_comparison
from features import day_cutoff
from features_xg import FEATURE_COLUMNS_XG, XgFeatureConfig, build_features_xg
from models_xg import ALL_SPECS, XG_CLASSIFIER_SPECS, fit_any
from train import atomic_dump, make_bundle, probabilities_frame, write_json

XG_NAMES = set(XG_CLASSIFIER_SPECS)


def fit_candidates(training, calibration, evaluation):
    predictions = {}
    for name in ALL_SPECS:
        base = fit_any(name, training)
        raw = base.predict_proba(evaluation)
        platt = PlattCalibrator().fit(base.predict_proba(calibration), calibration.target)
        predictions[name] = raw
        predictions[name + "+platt"] = platt.transform(raw)
    return predictions


def backtest(db, report_dir, final_season=2025, overwrite=False, extra_lag=0.0, extra_xg_lag=0.0):
    reports = Path(report_dir)
    if (reports / "test_report.json").exists() and not overwrite:
        raise ValueError("A test report already exists here. Use a new --reports folder or --overwrite to reproduce only.")
    reports.mkdir(parents=True, exist_ok=True)
    config = XgFeatureConfig(extra_history_lag_hours=extra_lag, extra_xg_lag_hours=extra_xg_lag)
    matches = load_matches_xg(db, last_complete_season=final_season)
    matches = matches[matches.season <= final_season].copy()
    counts = matches.groupby("season").size()
    if any(counts.get(s, 0) != 380 for s in range(2014, final_season + 1)):
        raise ValueError("Backtest requires 380 completed matches in every season from 2014 through the final season")
    snapshot = fingerprint_xg(matches)
    frame = build_features_xg(matches, config)

    validation_records, folds = [], []
    for year in range(2018, final_season - 1):
        training, calibration, validation = season_split(frame, year)
        print(f"Walk-forward: train through {year-2}, calibrate {year-1}/{str(year)[-2:]}, validate {year}/{str(year+1)[-2:]}", flush=True)
        for name, p in fit_candidates(training, calibration, validation).items():
            validation_records.append(probabilities_frame(validation, name, p))
        folds.append({"validation_season": year, "train": describe(training),
                      "calibration": describe(calibration), "validation": describe(validation)})
    val = pd.concat(validation_records, ignore_index=True)
    summary = {name: classification_metrics(g.target, g[["p_H", "p_D", "p_A"]]) for name, g in val.groupby("model")}
    is_xg = lambda n: n.removesuffix("+platt") in XG_NAMES
    best_all = min(summary, key=lambda k: summary[k]["log_loss"])
    best_base = min((k for k in summary if not is_xg(k)), key=lambda k: summary[k]["log_loss"])
    best_xg = min((k for k in summary if is_xg(k)), key=lambda k: summary[k]["log_loss"])
    selection = {"selected": best_all, "best_baseline_features": best_base, "best_xg_features": best_xg,
                 "selected_by": "pooled walk-forward log loss over validation seasons only",
                 "validation_seasons": [f["validation_season"] for f in folds],
                 "validation_metrics": summary, "feature_config": asdict(config),
                 "candidates": list(ALL_SPECS), "snapshot_sha256": snapshot,
                 "final_test_season": final_season,
                 "xg_assumption": "Declared retrospective xG availability, independently gated; "
                                  f"extra xG lag hours: {extra_xg_lag}; extra baseline history lag hours: {extra_lag}"}
    write_json(reports / "selection.json", selection)       # frozen before any test scoring
    write_json(reports / "splits.json", folds)
    val.to_csv(reports / "walk_forward_predictions.csv", index=False)
    print(f"Frozen selection: {best_all} | best baseline-feature model: {best_base} | best xG model: {best_xg}", flush=True)

    training, calibration, test = season_split(frame, final_season)
    predictions = fit_candidates(training, calibration, test)
    metrics = {name: classification_metrics(test.target, p) for name, p in predictions.items()}
    book = closing_benchmark(db, test)
    mask = book[["p_H", "p_D", "p_A"]].notna().all(axis=1).to_numpy()
    if not mask.any():
        raise ValueError("No market-average closing odds in the final test season")
    book_p = book.loc[mask, ["p_H", "p_D", "p_A"]].to_numpy()
    bench = classification_metrics(book.loc[mask, "target"], book_p)
    y, dates = test.target.to_numpy()[mask], test.match_date.to_numpy()[mask]
    rows = {}
    for label, name in (("selected_overall", best_all), ("best_baseline_features", best_base), ("best_xg_features", best_xg)):
        rows[label] = {"model": name,
                       "validation": summary[name],
                       "test_on_benchmark_matches": classification_metrics(y, predictions[name][mask]),
                       "paired_vs_closing": paired_comparison(y, predictions[name][mask], book_p, dates)}
    pd.concat([probabilities_frame(test, n, p) for n, p in predictions.items()] +
              [probabilities_frame(test.iloc[np.flatnonzero(mask)], "market_average_closing", book_p)],
              ignore_index=True).to_csv(reports / "test_predictions.csv", index=False)
    report = {"experiment": "xG feature experiment on the retrospective delayed-archive snapshot",
              "final_test_season": final_season, "train": describe(training), "calibration": describe(calibration),
              "test": describe(test), "market_average_closing": bench, "results": rows,
              "all_test_metrics": metrics, "snapshot_sha256": snapshot,
              "limits": ["The final test season was already scored once by the baseline run; this is a second look, "
                         "so the test numbers here are not an untouched holdout",
                         "xG availability is a declared assumption (equal to the result's), not observed publication time",
                         "Closing odds contain later information than midnight forecasts",
                         "Seven classifier families were compared on validation; selection uses validation only",
                         "Fourteen raw/calibrated variants were compared; validation intervals do not adjust for selection",
                         "extra_xg_lag_hours delays only xG; extra_history_lag_hours delays all result/statistic history",
                         "Single test season; no return-on-investment claim"]}
    write_json(reports / "test_report.json", report)
    table = pd.DataFrame([{"model": r["model"], "val_log_loss": r["validation"]["log_loss"],
                           "test_log_loss": r["test_on_benchmark_matches"]["log_loss"],
                           "test_accuracy": r["test_on_benchmark_matches"]["accuracy"], "role": k}
                          for k, r in rows.items()])
    print("\nRESULTS (test = final season, second look)\n" + table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"closing market: log_loss {bench['log_loss']:.4f}  accuracy {bench['accuracy']:.4f}")
    return report


def fit_production_xg(db, selection_path, goal_selection_path, output, as_of=None):
    """Fit future-prediction models from frozen choices without rescoring a test set.

    Raw classifiers and the goal regressor use all eligible post-warmup history.
    A calibrated classifier reserves the latest date-aligned ~380 matches for
    calibration. Goal family comes from the original pre-test selection report.
    """
    selection = json.loads(Path(selection_path).read_text())
    name = selection["selected"]
    base_name = name.removesuffix("+platt")
    if base_name not in ALL_SPECS:
        raise ValueError(f"Unknown selected classifier: {name}")
    goal_selection = json.loads(Path(goal_selection_path).read_text())
    goal_name = goal_selection["goal_model"]
    if goal_name not in GOAL_SPECS:
        raise ValueError(f"Unknown selected goal model: {goal_name}")
    config = XgFeatureConfig(**selection["feature_config"])
    now = pd.Timestamp(as_of) if as_of else pd.Timestamp.now(tz="UTC")
    if now.tzinfo is None:
        raise ValueError("--as-of must specify a timezone")
    matches = load_matches_xg(db, last_complete_season=selection.get("final_test_season", 2025))
    available = pd.to_datetime(matches.history_available_at, utc=True) + pd.Timedelta(hours=config.extra_history_lag_hours)
    matches = matches[available < now].copy()
    frame = build_features_xg(matches, config)
    eligible = frame[frame.season >= 2015].sort_values(["match_date", "match_id"])
    if len(eligible) < 1000:
        raise ValueError("At least 1,000 eligible matches are needed for production fitting")
    calibration = eligible.iloc[:0].copy()
    training = eligible
    if name.endswith("+platt"):
        cal_start = eligible.iloc[-380].match_date
        calibration = eligible[eligible.match_date >= cal_start].copy()
        fitting_available = pd.to_datetime(eligible.history_available_at, utc=True) + pd.Timedelta(hours=config.extra_history_lag_hours)
        training = eligible[(eligible.match_date < cal_start) & (fitting_available < day_cutoff(cal_start))].copy()
    base = fit_any(base_name, training)
    calibrator = (PlattCalibrator().fit(base.predict_proba(calibration), calibration.target)
                  if len(calibration) else None)
    classifier = CalibratedModel(base, calibrator)
    goals = fit_goal_model(goal_name, eligible)
    deployed_selection = {**selection, "classifier": name, "goal_model": goal_name,
                          "goal_selected_by": "original baseline walk-forward goal NLL; no xG goal-model selection"}
    bundle = make_bundle(classifier, goals, deployed_selection, training, calibration,
                         config, fingerprint_xg(matches), "xG production refit; historical test reports unchanged")
    bundle.update(format_version=2, feature_set="xg", feature_columns=FEATURE_COLUMNS_XG,
                  xg_history_version=2, required_xg_through_season=selection.get("final_test_season", 2025),
                  fit_as_of_utc=now.isoformat(), goal_training=describe(eligible))
    usable_xg = pd.to_datetime(matches.xg_available, utc=True) + pd.Timedelta(hours=config.extra_xg_lag_hours)
    observed = matches[usable_xg.lt(now) & matches.home_xg.notna() & matches.away_xg.notna()]
    bundle["xg_coverage"] = {"matches": len(observed), "latest_match_date": observed.match_date.max() if len(observed) else None,
                              "completed_matches_without_xg": int((matches.home_xg.isna() | matches.away_xg.isna()).sum())}
    from epl_model.scores import summarize_score
    for probabilities, rate in zip(classifier.predict_proba(eligible.tail(3)), goals.predict_rates(eligible.tail(3))):
        summarize_score(*rate, probabilities)
    atomic_dump(bundle, output)
    print(f"Saved {name} to {output}; classifier fitting matches: {len(training)}; calibration: {len(calibration)}")
    print(f"xG coverage: {bundle['xg_coverage']}")
    return bundle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default="data/epl_xg.sqlite")
    p.add_argument("--reports", default="reports/xg_run")
    p.add_argument("--final-season", type=int, default=2025)
    p.add_argument("--extra-history-lag-hours", type=float, default=0.0)
    p.add_argument("--extra-xg-lag-hours", type=float, default=0.0, help="Delay xG only; keep baseline history unchanged")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--production-only", action="store_true")
    p.add_argument("--selection", default="reports/xg_run/selection.json")
    p.add_argument("--goal-selection", default="reports/selection.json")
    p.add_argument("--models", default="models/xg")
    p.add_argument("--as-of", help="Timezone-aware production cutoff, e.g. 2026-10-28T00:00:00Z")
    a = p.parse_args()
    if a.production_only:
        if a.extra_history_lag_hours or a.extra_xg_lag_hours:
            p.error("Production uses the frozen feature_config in --selection; lag flags are for backtests")
        fit_production_xg(a.db, a.selection, a.goal_selection, Path(a.models) / "production.joblib", a.as_of)
    else:
        backtest(a.db, a.reports, a.final_season, a.overwrite, a.extra_history_lag_hours, a.extra_xg_lag_hours)


if __name__ == "__main__":
    main()
