"""Regression checks for independent availability and saved-model prediction."""
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from epl_model.models import CalibratedModel, TeamPoisson
from epl_model.scores import matrix_outcomes
from features import FEATURE_COLUMNS, day_cutoff
from features_xg import FEATURE_COLUMNS_XG, XG_FEATURE_COLUMNS, XgFeatureConfig, XgHistoryEngine, build_features_xg
from models_xg import LogitXg
from predict import forecast
from test_xg import make_matches
from understat_to_jsonl import season_start


class IndependentXgTiming(unittest.TestCase):
    def test_only_xg_timestamp_changes_leave_all_baseline_columns_identical(self):
        matches = make_matches()
        original = build_features_xg(matches)
        delayed = matches.copy()
        delayed.loc[0, "xg_available"] = pd.Timestamp("2030-01-01", tz="UTC")
        later = build_features_xg(delayed)
        pd.testing.assert_frame_equal(original[FEATURE_COLUMNS], later[FEATURE_COLUMNS])
        self.assertFalse(original[XG_FEATURE_COLUMNS].equals(later[XG_FEATURE_COLUMNS]))

    def test_xg_lag_flag_changes_xg_but_not_baseline_features(self):
        matches = make_matches()
        original = build_features_xg(matches)
        delayed = build_features_xg(matches, XgFeatureConfig(extra_xg_lag_hours=72))
        pd.testing.assert_frame_equal(original[FEATURE_COLUMNS], delayed[FEATURE_COLUMNS])
        self.assertFalse(original[XG_FEATURE_COLUMNS].equals(delayed[XG_FEATURE_COLUMNS]))

    def test_xg_at_exact_cutoff_is_excluded_until_later(self):
        matches = make_matches(days=1)
        matches["xg_available"] = day_cutoff("2024-08-21")
        engine = XgHistoryEngine(matches)
        first = engine.fixture("a", "b", "2024-08-21", 2024)
        self.assertTrue(np.isnan(first["home_form_xg_for_3"]))
        self.assertEqual(first["home_xg_available_matches_10"], 0)
        self.assertNotEqual(first["home_elo"], 1500)
        later = engine.fixture("a", "b", "2024-08-22", 2024)
        self.assertAlmostEqual(later["home_form_xg_for_3"], matches.home_xg.iloc[0])

    def test_late_xg_keeps_match_order_in_rolling_weights(self):
        matches = make_matches(days=6)
        matches.loc[0, "xg_available"] = pd.Timestamp("2024-09-05T06:00:00Z")
        out = XgHistoryEngine(matches).fixture("a", "d", "2024-09-07", 2024)
        expected = np.average(matches.home_xg.iloc[[0, 2, 4]], weights=[.8**2, .8, 1])
        self.assertAlmostEqual(out["home_form_xg_for_3"], expected)

    def test_mutating_not_yet_available_past_xg_cannot_change_prediction_features(self):
        matches = make_matches()
        matches.loc[0, "xg_available"] = pd.Timestamp("2030-01-01", tz="UTC")
        original = build_features_xg(matches)
        matches.loc[0, ["home_xg", "away_xg"]] = [99, 88]
        changed = build_features_xg(matches)
        pd.testing.assert_frame_equal(original[FEATURE_COLUMNS_XG], changed[FEATURE_COLUMNS_XG])

    def test_negative_or_infinite_xg_delay_is_rejected(self):
        for delay in (-1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                XgHistoryEngine(make_matches(), XgFeatureConfig(extra_xg_lag_hours=delay))


class SavedXgPrediction(unittest.TestCase):
    def setUp(self):
        self.matches = make_matches(days=40)
        self.matches["home_team"] = self.matches.home_id
        self.matches["away_team"] = self.matches.away_id
        self.matches["target"] = np.where(self.matches.home_goals > self.matches.away_goals, 0,
                                          np.where(self.matches.home_goals == self.matches.away_goals, 1, 2))
        frame = build_features_xg(self.matches)
        self.classifier = CalibratedModel(LogitXg().fit(frame))
        goals = TeamPoisson().fit(frame)
        self.bundle = {"format_version": 2, "feature_set": "xg", "xg_history_version": 2,
                       "feature_columns": FEATURE_COLUMNS_XG, "feature_config": asdict(XgFeatureConfig()),
                       "information_through_utc": self.matches.history_available_at.max().isoformat(),
                       "classifier": self.classifier, "goals": goals,
                       "selection": {"classifier": "logit_xg", "goal_model": "team_poisson"},
                       "history_mode": "retrospective_lagged", "required_xg_through_season": 2024}
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "model.joblib"
        joblib.dump(self.bundle, self.path)

    def test_saved_model_uses_xg_features_and_auto_selects_database(self):
        with patch("predict.load_matches_xg", return_value=self.matches) as loader:
            out, matrix = forecast(None, self.path, "a", "b", "2025-02-01")
        loader.assert_called_once_with("data/epl_xg.sqlite", last_complete_season=2024)
        row = XgHistoryEngine(self.matches).fixture("a", "b", "2025-02-01", 2024)
        expected = self.classifier.predict_proba(pd.DataFrame([row]))[0]
        np.testing.assert_allclose(list(out["outcome_probabilities"].values()), expected)
        np.testing.assert_allclose(matrix_outcomes(matrix), expected, atol=1e-12)
        self.assertEqual(out["model"], "logit_xg")
        self.assertTrue(any("xg" in f["feature"] for f in out["key_factors"]["factors"]))
        self.assertEqual(out["xg_coverage"]["home"]["available_xg_matches_in_last_10"], 10)

    def test_stale_xg_is_reported(self):
        with patch("predict.load_matches_xg", return_value=self.matches):
            out, _ = forecast("example.sqlite", self.path, "a", "b", "2025-04-01")
        self.assertTrue(any("latest available xG" in n for n in out["notes"]))

    def test_prediction_before_fitting_cutoff_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "information at or after"):
            forecast(None, self.path, "a", "b", "2024-12-01")


class ImportSeasonLabels(unittest.TestCase):
    def test_supported_labels(self):
        for label in (1415, "1415", 2014, "2014/15", "2014-2015"):
            self.assertEqual(season_start(label), 2014)

    def test_invalid_labels(self):
        for label in ("2014/16", "2014-2017", "1416", "unknown"):
            with self.assertRaises(ValueError):
                season_start(label)


if __name__ == "__main__":
    unittest.main()
