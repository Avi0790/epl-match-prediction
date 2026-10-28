"""Regression tests for the xG experiment: joins, availability and no-leakage rules."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epl_model.data_xg import attach_xg                                    # noqa: E402
from features import FEATURE_COLUMNS, FeatureConfig, build_features         # noqa: E402
from features_xg import FEATURE_COLUMNS_XG, XG_FEATURE_COLUMNS, XgFeatureConfig, XgHistoryEngine, build_features_xg  # noqa: E402
from models_xg import LogitXg                                              # noqa: E402

TEAMS = ["a", "b", "c", "d"]


def make_matches(days=24, seed=3):
    rng = np.random.default_rng(seed)
    rows, day0, n = [], pd.Timestamp("2024-08-17"), 0
    pairs = [("a", "b"), ("c", "d"), ("a", "c"), ("b", "d"), ("a", "d"), ("b", "c")]
    for d in range(days):
        for h, a in (pairs[d % 6], ):
            h, a = (h, a) if (d // 6) % 2 == 0 else (a, h)
            date = (day0 + pd.Timedelta(days=d * 4)).date().isoformat()
            avail = pd.Timestamp(date, tz="UTC") + pd.Timedelta(days=1, hours=6)
            n += 1
            rows.append(dict(match_id=f"m{n:03d}", season=2024, home_id=h, away_id=a, match_date=date,
                             home_goals=int(rng.poisson(1.5)), away_goals=int(rng.poisson(1.2)),
                             home_shots=int(rng.integers(5, 20)), away_shots=int(rng.integers(5, 20)),
                             home_sot=int(rng.integers(1, 8)), away_sot=int(rng.integers(1, 8)),
                             history_available_at=avail, xg_available=avail, home_xg=float(rng.gamma(4, .35)),
                             away_xg=float(rng.gamma(4, .3))))
    return pd.DataFrame(rows)


def features(matches, **config):
    return build_features_xg(matches, XgFeatureConfig(**config)).set_index("match_id")


class XgFeatureLeakage(unittest.TestCase):
    def setUp(self):
        self.matches = make_matches()

    def test_baseline_columns_unchanged_by_xg_engine(self):
        base = build_features(self.matches.drop(columns=["home_xg", "away_xg"])).set_index("match_id")
        xg = features(self.matches)
        for column in FEATURE_COLUMNS:
            np.testing.assert_allclose(base[column].astype(float), xg[column].astype(float), equal_nan=True)

    def test_feature_list_is_complete_and_unique(self):
        self.assertEqual(len(set(FEATURE_COLUMNS_XG)), len(FEATURE_COLUMNS_XG))
        built = features(self.matches)
        self.assertTrue(set(XG_FEATURE_COLUMNS) <= set(built.columns))

    def test_target_and_future_xg_do_not_change_a_matchs_features(self):
        base = features(self.matches)
        target = self.matches.match_id.iloc[10]
        changed = self.matches.copy()
        later = changed.match_id >= target                   # the target match and everything after it
        changed.loc[later, ["home_xg", "away_xg"]] = [9.9, 8.8]
        after = features(changed)
        np.testing.assert_allclose(base.loc[target, XG_FEATURE_COLUMNS].astype(float),
                                   after.loc[target, XG_FEATURE_COLUMNS].astype(float), equal_nan=True)

    def test_earlier_xg_does_change_later_features(self):
        base = features(self.matches)
        changed = self.matches.copy()
        changed.loc[changed.index[0], ["home_xg", "away_xg"]] = [9.9, 8.8]
        after = features(changed)
        second = self.matches.match_id.iloc[1]
        involved = self.matches.loc[0, ["home_id", "away_id"]].tolist()
        # some later match involving one of the two teams must see the difference
        later = self.matches[self.matches.home_id.isin(involved) | self.matches.away_id.isin(involved)].match_id.iloc[1:]
        diffs = [(base.loc[m, XG_FEATURE_COLUMNS].astype(float) - after.loc[m, XG_FEATURE_COLUMNS].astype(float)).abs().sum()
                 for m in later]
        self.assertGreater(max(diffs), 1e-6)
        self.assertIn(second, base.index)

    def test_xg_is_excluded_until_its_declared_availability(self):
        delayed = self.matches.copy()
        delayed.loc[0, "history_available_at"] = pd.Timestamp("2030-01-01", tz="UTC")   # never usable in this season
        first_teams = delayed.loc[0, ["home_id", "away_id"]].tolist()
        out = features(delayed)
        later = delayed[(delayed.index > 0) & (delayed.home_id == first_teams[0])].match_id.iloc[0]
        control = features(self.matches)
        self.assertNotEqual(float(out.loc[later, "home_form_xg_for_3"]), float(control.loc[later, "home_form_xg_for_3"]))
        # and with the xG removed entirely the feature equals the delayed one
        removed = self.matches.drop(index=0)
        np.testing.assert_allclose(float(features(removed).loc[later, "home_form_xg_for_3"]),
                                   float(out.loc[later, "home_form_xg_for_3"]), equal_nan=True)

    def test_missing_xg_stays_nan_not_zero(self):
        matches = self.matches.copy()
        matches[["home_xg", "away_xg"]] = np.nan
        out = features(matches)
        self.assertTrue(out[XG_FEATURE_COLUMNS].isna().all().all())


class XgJoin(unittest.TestCase):
    def setUp(self):
        self.matches = make_matches(days=6)
        self.matches = self.matches.drop(columns=["home_xg", "away_xg"])
        self.xg = pd.concat([
            pd.DataFrame({"match_id": self.matches.match_id, "team_id": self.matches.home_id,
                          "xg_for": 1.5, "xg_against": 0.8}),
            pd.DataFrame({"match_id": self.matches.match_id, "team_id": self.matches.away_id,
                          "xg_for": 0.8, "xg_against": 1.5})], ignore_index=True)
        self.xg["definition"] = "test xG"
        self.xg["assumed_available_at_utc"] = "2024-09-01T06:00:00.000000Z"

    def test_clean_join_keeps_baseline_availability_unchanged(self):
        out = attach_xg(self.matches, self.xg)
        self.assertTrue((out.home_xg == 1.5).all() and (out.away_xg == .8).all())
        pd.testing.assert_series_equal(out.history_available_at, self.matches.history_available_at)
        self.assertTrue((out.xg_available == pd.Timestamp("2024-09-01T06:00:00Z")).all())

    def test_missing_xg_for_a_required_match_is_rejected(self):
        with self.assertRaises(ValueError):
            attach_xg(self.matches, self.xg.iloc[1:])

    def test_unplayed_future_season_may_lack_xg(self):
        matches = self.matches.copy()
        matches.loc[matches.index[-1], "season"] = 2026
        xg = self.xg[~self.xg.match_id.eq(matches.match_id.iloc[-1])]
        out = attach_xg(matches, xg, last_complete_season=2025)
        self.assertTrue(out.home_xg.iloc[-1] != out.home_xg.iloc[-1])  # NaN, never 0

    def test_mixed_definitions_and_disagreeing_sides_are_rejected(self):
        mixed = self.xg.copy(); mixed.loc[0, "definition"] = "other provider"
        with self.assertRaises(ValueError):
            attach_xg(self.matches, mixed)
        bad = self.xg.copy(); bad.loc[0, "xg_against"] = 9.0
        with self.assertRaises(ValueError):
            attach_xg(self.matches, bad)

    def test_missing_availability_assumption_is_rejected(self):
        bad = self.xg.copy(); bad.loc[0, "assumed_available_at_utc"] = None
        with self.assertRaises(ValueError):
            attach_xg(self.matches, bad)


class XgModelInputs(unittest.TestCase):
    def test_logit_handles_missing_differences_without_leaking_a_fill_value(self):
        frame = features(make_matches(days=40)).reset_index()
        frame["target"] = (np.arange(len(frame)) % 3)
        model = LogitXg().fit(frame)
        p = model.predict_proba(frame)
        self.assertTrue(np.isfinite(p).all())
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
