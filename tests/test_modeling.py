import unittest
from collections import deque

import numpy as np
import pandas as pd

from epl_model.models import PlattCalibrator
from epl_model.scores import matrix_outcomes, score_matrix, summarize_score
from epl_model.splits import assert_order, season_split
from evaluate import classification_metrics
from features import FEATURE_COLUMNS, FeatureConfig, HistoryEngine, _weighted, build_features
from predict import check_horizon, resolve_team


def sample_matches():
    data = [("m1", "2015-08-01", "a", "b", 2, 0),
            ("m2", "2015-08-01", "c", "d", 1, 1),
            ("m3", "2015-08-04", "b", "a", 0, 1),
            ("m4", "2015-08-08", "a", "c", 3, 2),
            ("m5", "2015-08-10", "a", "b", 1, 0)]
    return pd.DataFrame([dict(match_id=i, match_date=d, home_id=h, away_id=a, home_team=h, away_team=a,
        home_goals=hg, away_goals=ag, home_shots=10, away_shots=8, home_sot=5, away_sot=3,
        season=2015, target=0 if hg > ag else 1 if hg == ag else 2,
        history_available_at=pd.Timestamp(d, tz="UTC") + pd.Timedelta(days=2)) for i, d, h, a, hg, ag in data])


class CausalFeatures(unittest.TestCase):
    def test_target_and_future_values_do_not_change_features(self):
        original = sample_matches()
        first = build_features(original)
        mutated = original.copy()
        mutated.loc[mutated.match_date >= "2015-08-04", ["home_goals", "away_goals", "home_shots", "away_sot"]] = 99
        second = build_features(mutated)
        pd.testing.assert_frame_equal(first.loc[first.match_date <= "2015-08-04", FEATURE_COLUMNS],
                                      second.loc[second.match_date <= "2015-08-04", FEATURE_COLUMNS])

    def test_future_rows_do_not_change_past_features(self):
        records = sample_matches()
        full = build_features(records)
        prefix = build_features(records.iloc[:3])
        pd.testing.assert_frame_equal(full.iloc[:3][FEATURE_COLUMNS], prefix[FEATURE_COLUMNS])

    def test_same_day_order_invariance(self):
        records = sample_matches()
        a = build_features(records).set_index("match_id")[FEATURE_COLUMNS].sort_index()
        b = build_features(records.iloc[[1, 0, 2, 4, 3]]).set_index("match_id")[FEATURE_COLUMNS].sort_index()
        pd.testing.assert_frame_equal(a, b)

    def test_elo_result_waits_for_availability(self):
        engine = HistoryEngine(sample_matches().iloc[:1])
        before = engine.fixture("a", "b", "2015-08-02", 2015)
        self.assertEqual(before["home_elo"], 1500)
        after = engine.fixture("a", "b", "2015-08-04", 2015)
        expected = 1500 + 20 * (1 - 1 / (1 + 10 ** (-70 / 400)))
        self.assertAlmostEqual(after["home_elo"], expected)
        self.assertAlmostEqual(after["home_elo"] + after["away_elo"], 3000)

    def test_strict_cutoff_excludes_equal_timestamp(self):
        records = sample_matches().iloc[:1].copy()
        records["history_available_at"] = pd.Timestamp("2015-08-03T23:00:00Z")
        self.assertEqual(HistoryEngine(records).fixture("a", "b", "2015-08-04", 2015)["home_elo"], 1500)

    def test_rest_counts_and_venue_form(self):
        row = build_features(sample_matches()).iloc[3]
        self.assertEqual(row.home_epl_rest_days, 4)
        self.assertEqual(row.home_epl_matches_14d, 2)
        self.assertEqual(row.home_form_n_3, 2)
        self.assertEqual(row.home_venue_n_3, 1)
        self.assertAlmostEqual(row.home_venue_gf_3, 2)

    def test_weighting_skips_missing_without_inventing_zero(self):
        records = deque([{"x": 1}, {"x": np.nan}, {"x": 3}])
        self.assertAlmostEqual(_weighted(records, "x", 3, .5), (1 * .25 + 3) / 1.25)

    def test_extra_lag_changes_only_future_eligibility(self):
        records = sample_matches().iloc[:1]
        row = HistoryEngine(records, FeatureConfig(extra_history_lag_hours=72)).fixture("a", "b", "2015-08-04", 2015)
        self.assertEqual(row["home_elo"], 1500)
        self.assertEqual(row["home_epl_rest_days"], 3)

    def test_new_season_promotion_prior(self):
        records = sample_matches()
        row = HistoryEngine(records).fixture("new", "a", "2016-08-10", 2016)
        self.assertEqual(row["home_elo"], 1450)
        self.assertEqual(row["home_promoted_or_returning"], 1)
        self.assertEqual(row["away_promoted_or_returning"], 0)
        self.assertEqual(row["home_form_n_3"], 0)


class ScoresAndMetrics(unittest.TestCase):
    def test_matrix_mass_tail_and_marginals(self):
        matrix, tail = score_matrix(1.7, .8, [.6, .25, .15])
        self.assertAlmostEqual(matrix.sum(), 1)
        np.testing.assert_allclose(matrix_outcomes(matrix), [.6, .25, .15], atol=1e-12)
        self.assertLess(tail["poisson_tail_mass_omitted"], 1e-10)
        self.assertTrue((matrix >= 0).all())

    def test_equal_rates_symmetry_and_expectation(self):
        summary, matrix = summarize_score(1.4, 1.4)
        self.assertAlmostEqual(summary["expected_home_goals"], 1.4, places=8)
        self.assertAlmostEqual(matrix_outcomes(matrix)[0], matrix_outcomes(matrix)[2])
        probabilities = [s["probability"] for s in summary["top_5_scorelines"]]
        self.assertEqual(probabilities, sorted(probabilities, reverse=True))

    def test_known_metrics_and_class_order(self):
        metrics = classification_metrics([0], [[.5, .25, .25]])
        self.assertAlmostEqual(metrics["log_loss"], np.log(2))
        self.assertAlmostEqual(metrics["brier"], .375)
        self.assertAlmostEqual(metrics["rps"], .15625)
        perfect = classification_metrics([0, 1, 2], np.eye(3))
        self.assertEqual(perfect["accuracy"], 1)
        self.assertEqual(perfect["rps"], 0)

    def test_bad_probabilities_fail(self):
        with self.assertRaises(ValueError):
            classification_metrics([0], [[.8, .3, .3]])
        with self.assertRaises(ValueError):
            score_matrix(-1, 2)

    def test_platt_stays_normalized(self):
        p = np.tile([[.6, .2, .2], [.2, .6, .2], [.2, .2, .6]], (10, 1))
        c = PlattCalibrator().fit(p, np.tile([0, 1, 2], 10))
        calibrated = c.transform(p[:3])
        np.testing.assert_allclose(calibrated.sum(axis=1), 1)
        self.assertTrue((calibrated >= 0).all())

    def test_chronological_partition_rejects_late_labels(self):
        rows = sample_matches().iloc[[0, 2, 3]].reset_index(drop=True).copy()
        rows.loc[0, "history_available_at"] = pd.Timestamp("2015-09-01T00:00:00Z")
        with self.assertRaises(ValueError):
            assert_order(rows.iloc[:1], rows.iloc[1:2], rows.iloc[2:])

    def test_prediction_rejects_model_trained_after_fixture(self):
        bundle = {"information_through_utc": "2025-05-26T18:00:00Z", "feature_columns": FEATURE_COLUMNS}
        with self.assertRaises(ValueError):
            check_horizon(bundle, "2025-05-25")
        self.assertEqual(str(check_horizon(bundle, "2025-08-01")), "2025-07-31 23:00:00+00:00")

    def test_unknown_team_is_not_silently_guessed(self):
        with self.assertRaises(ValueError):
            resolve_team("arsnal", sample_matches())

    def test_valid_season_splits_are_disjoint(self):
        rows = sample_matches().iloc[:1].copy()
        blocks = []
        for year in (2015, 2016, 2017):
            block = rows.copy(); block["season"] = year
            block["match_id"] = f"s{year}"; block["match_date"] = f"{year}-08-01"
            block["history_available_at"] = pd.Timestamp(f"{year}-08-03T00:00:00Z")
            blocks.append(block)
        training, calibration, evaluation = season_split(pd.concat(blocks), 2017)
        self.assertEqual(training.season.tolist(), [2015])
        self.assertEqual(calibration.season.tolist(), [2016])
        self.assertEqual(evaluation.season.tolist(), [2017])


if __name__ == "__main__":
    unittest.main()
