#!/usr/bin/env python3
"""Paired comparison of two saved models (default: logit_xg vs elo) with a calendar-week
block bootstrap, on the walk-forward validation seasons and on the final test season.

    python compare_models.py --reports reports/xg_run
Negative log-loss / RPS differences favor the first model. Descriptive, not a profit test.
These fixed-prediction intervals do not adjust for validation model selection.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import paired_comparison


def compare(frame, a, b):
    wide = frame[frame.model.isin([a, b])].pivot(index="match_id", columns="model", values=["p_H", "p_D", "p_A"])
    meta = frame[frame.model == a].set_index("match_id").loc[wide.index]
    pa = np.column_stack([wide[(c, a)] for c in ("p_H", "p_D", "p_A")])
    pb = np.column_stack([wide[(c, b)] for c in ("p_H", "p_D", "p_A")])
    return paired_comparison(meta.target.to_numpy(), pa, pb, meta.match_date.to_numpy()), len(meta)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reports", type=Path, default=Path("reports/xg_run"))
    p.add_argument("--a", default="logit_xg")
    p.add_argument("--b", default="elo")
    args = p.parse_args()
    print("Descriptive fixed-prediction intervals; validation model selection is not repeated in the bootstrap.")
    for label, name in (("VALIDATION (2018/19-2023/24, pooled)", "walk_forward_predictions.csv"),
                        ("FINAL TEST SEASON (second look)", "test_predictions.csv")):
        result, n = compare(pd.read_csv(args.reports / name), args.a, args.b)
        print(f"\n{label}: {args.a} minus {args.b} on {n} matches")
        for key in ("log_loss", "rps"):
            r = result[key]
            print(f"  {key:9s} {r['model_minus_bookmaker']:+.4f}   95% interval [{r['ci_95_low']:+.4f}, {r['ci_95_high']:+.4f}]")


if __name__ == "__main__":
    main()
