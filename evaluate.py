#!/usr/bin/env python3
"""Metrics in explicit H/D/A order; saved-prediction evaluation and calibration plots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import poisson
from epl_model.scores import check_probabilities


def per_match_metrics(y, p):
    p = check_probabilities(p)
    y = np.asarray(y, dtype=int)
    if len(y) != len(p) or not np.isin(y, [0, 1, 2]).all():
        raise ValueError("Labels must be 0=H, 1=D, 2=A and match probability rows")
    truth = np.eye(3)[y]
    return {"log_loss": -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)),
            "brier": np.sum((p - truth) ** 2, axis=1),
            "rps": np.sum((p.cumsum(axis=1)[:, :2] - truth.cumsum(axis=1)[:, :2]) ** 2, axis=1) / 2,
            "accuracy": (p.argmax(axis=1) == y).astype(float)}


def classification_metrics(y, p):
    return {name: float(values.mean()) for name, values in per_match_metrics(y, p).items()}


def goal_nll(frame, rates):
    rates = np.asarray(rates)
    return float(-(poisson.logpmf(frame.home_goals, rates[:, 0]) + poisson.logpmf(frame.away_goals, rates[:, 1])).mean())


def score_metrics(frame):
    over_truth = (frame.home_goals + frame.away_goals > 2).to_numpy()
    p = frame.p_over_2_5.to_numpy()
    return {"exact_score_accuracy": float(((frame.pred_home_goals == frame.home_goals) & (frame.pred_away_goals == frame.away_goals)).mean()),
            "top_5_score_accuracy": float(frame.actual_in_top_5.mean()),
            "scoreline_result_accuracy": float((np.sign(frame.pred_home_goals - frame.pred_away_goals) == np.sign(frame.home_goals - frame.away_goals)).mean()),
            "over_under_2_5_accuracy": float(((p >= .5) == over_truth).mean()),
            "over_under_2_5_log_loss": float(-np.mean(over_truth * np.log(np.clip(p, 1e-15, 1)) + (~over_truth) * np.log(np.clip(1-p, 1e-15, 1))))}


def paired_comparison(y, model_p, bookmaker_p, dates, iterations=2000):
    """Resample calendar weeks, preserving matches in each block. Never used for fitting."""
    a, b = per_match_metrics(y, model_p), per_match_metrics(y, bookmaker_p)
    weeks = pd.to_datetime(pd.Series(dates)).dt.to_period("W").astype(str).to_numpy()
    groups = [np.where(weeks == week)[0] for week in np.unique(weeks)]
    rng, report = np.random.default_rng(2026), {}
    draws = rng.integers(0, len(groups), size=(iterations, len(groups)))
    counts = np.array([len(g) for g in groups])
    for key in ("log_loss", "rps"):
        diff = a[key] - b[key]
        sums = np.array([diff[g].sum() for g in groups])
        boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        lo, hi = np.quantile(boot, [.025, .975])
        report[key] = {"model_minus_bookmaker": float(diff.mean()), "ci_95_low": float(lo), "ci_95_high": float(hi)}
    return {"matches": len(y), "resampling": "calendar-week paired block bootstrap", "iterations": iterations,
            "interpretation": "Negative differences favor the model; intervals are descriptive, not a profitability test", **report}


def calibration_plot(series, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True, sharey=True)
    records = []
    for name, y, p in series:
        p = check_probabilities(p)
        for k, ax in enumerate(axes):
            bin_id = np.minimum((p[:, k] * 8).astype(int), 7)
            xs, ys = [], []
            for b in range(8):
                mask = bin_id == b
                if mask.any():
                    mean_p, actual = float(p[mask, k].mean()), float((np.asarray(y)[mask] == k).mean())
                    xs.append(mean_p); ys.append(actual)
                    records.append({"model": name, "class": "HDA"[k], "bin": b, "count": int(mask.sum()), "mean_probability": mean_p, "observed_frequency": actual})
            ax.plot(xs, ys, marker="o", label=name, markersize=4)
    for k, ax in enumerate(axes):
        ax.plot([0, 1], [0, 1], "--", color="#999999", linewidth=1)
        ax.set(title=("Home win", "Draw", "Away win")[k], xlabel="Predicted probability", xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Observed frequency")
    axes[-1].legend(fontsize=7, loc="upper left")
    fig.suptitle("2025/26 holdout calibration · 380 matches · small bins are noisy", fontsize=12)
    fig.tight_layout()
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160); plt.close(fig)
    pd.DataFrame(records).to_csv(output.with_suffix(".csv"), index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, default=Path("reports/test_predictions.csv"))
    args = p.parse_args()
    frame = pd.read_csv(args.predictions)
    report = {name: classification_metrics(group.target, group[["p_H", "p_D", "p_A"]]) for name, group in frame.groupby("model")}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
