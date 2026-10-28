"""Offline regression tests for leakage-sensitive ingestion behavior and real source anomalies."""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epl_data.api_football import ApiFootball, normalize_lineups, normalize_fixtures
from epl_data.football_data import ingest_csv
from epl_data.http_client import AccessError, Download, HttpClient
from epl_data.imports import import_observations
from epl_data.store import add_artifact, common, connect, observe, utc_string

OBSERVED = "2026-10-15T06:00:00Z"
HEADERS = "Div Date Time HomeTeam AwayTeam FTHG FTAG FTR Referee HS AS HST AST HC AC HY AY HR AR B365H B365D B365A B365CH B365CD B365CA PSCH PSCD PSCA AvgCH AvgCD AvgCA".split()


def row(**updates):
    r = dict(zip(HEADERS, ["E0", "17/08/2024", "15:00", "Arsenal", "Everton", "2", "0", "H",
        "Example Referee", "15", "7", "6", "2", "5", "3", "1", "2", "0", "0",
        "1.5", "4", "7", "1.4", "4.5", "8", "1.41", "4.6", "8.1", "1.42", "4.4", "8"]))
    r.update(updates)
    return r


class IngestionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = connect(self.root / "test.sqlite")
        self.sequence = 0

    def tearDown(self):
        self.db.close()
        self.temp.cleanup()

    def download(self, rows, observed=OBSERVED, headers=HEADERS):
        self.sequence += 1
        target = self.root / f"raw-{self.sequence}.csv"
        with target.open("w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        return Download(target, observed, "https://example.invalid/2425/E0.csv", False)

    def core(self):
        d = self.download([row()])
        ingest_csv(self.db, d, 2024)
        return d

    def test_core_odds_and_date_timezone(self):
        self.core()
        f = self.db.execute("SELECT * FROM fixture_versions").fetchone()
        self.assertEqual(f["kickoff_utc"], "2024-08-17T14:00:00.000000Z")
        self.assertEqual(f["kickoff_precision"], "minute_assumed_zone")
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM match_results").fetchone()[0], 1)
        phases = {r[0] for r in self.db.execute("SELECT DISTINCT phase FROM odds_snapshots")}
        self.assertEqual(phases, {"pre_closing", "closing"})
        self.assertTrue(all(r[0] is None for r in self.db.execute("SELECT quote_time_utc FROM odds_snapshots")))

    def test_idempotent_repeat_and_safe_asof(self):
        d = self.core()
        before = self.db.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0]
        ingest_csv(self.db, d, 2024)
        self.assertEqual(before, self.db.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0])
        # Fetching history today does not magically create observed historical inputs.
        n = self.db.execute("SELECT COUNT(*) FROM team_match_stats WHERE available_at_utc < ?",
                            (utc_string("2024-08-18T15:00:00Z"),)).fetchone()[0]
        self.assertEqual(n, 0)

    def test_revisions_keep_match_identity_and_original_observation(self):
        self.core()
        ingest_csv(self.db, self.download([row(Date="18/08/2024", FTHG="3")],
                   "2026-10-16T06:00:00Z"), 2024)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM fixtures").fetchone()[0], 1)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM match_results").fetchone()[0], 2)
        self.assertEqual(self.db.execute("SELECT home_goals FROM latest_match_results").fetchone()[0], 3)
        old = self.db.execute("SELECT home_goals FROM match_results WHERE available_at_utc < ?",
                              (utc_string("2026-10-15T07:00:00Z"),)).fetchone()[0]
        self.assertEqual(old, 2)

    def test_missing_time_stays_missing(self):
        ingest_csv(self.db, self.download([row(Time="")]), 2024)
        r = self.db.execute("SELECT kickoff_utc,kickoff_precision FROM fixture_versions").fetchone()
        self.assertEqual(tuple(r), (None, "date_only"))
        self.assertIsNotNone(self.db.execute("SELECT assumed_available_at_utc FROM match_results").fetchone()[0])

    def test_invalid_score_rolls_back_entire_season(self):
        with self.assertRaises(ValueError):
            ingest_csv(self.db, self.download([row(), row(HomeTeam="Chelsea", FTR="A")]), 2024)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM fixtures").fetchone()[0], 0)

    def test_duplicate_pair_is_rejected(self):
        with self.assertRaises(ValueError):
            ingest_csv(self.db, self.download([row(), row(Date="18/08/2024")]), 2024)

    def test_source_anomalies_are_quarantined_without_losing_result(self):
        ingest_csv(self.db, self.download([row(AS="8", AST="9", B365CH="0")]), 2024)
        away = self.db.execute("SELECT shots,shots_on_target FROM team_match_stats WHERE team_id='football_data:everton'").fetchone()
        self.assertEqual(tuple(away), (None, None))
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM data_issues").fetchone()[0], 2)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM match_results").fetchone()[0], 1)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM odds_snapshots WHERE bookmaker='bet365' AND phase='closing'").fetchone()[0], 0)

    def test_no_false_closing_odds_from_earlier_columns(self):
        sample = row(B365CH="", B365CD="", B365CA="", PSCH="", PSCD="", PSCA="", AvgCH="", AvgCD="", AvgCA="")
        ingest_csv(self.db, self.download([sample]), 2024)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM odds_snapshots WHERE phase='closing'").fetchone()[0], 0)

    def test_pinnacle_warning_is_date_specific(self):
        ingest_csv(self.db, self.download([row(Date="16/08/2025")]), 2025)
        self.assertEqual(self.db.execute("SELECT quality_flag FROM odds_snapshots WHERE bookmaker='pinnacle'").fetchone()[0],
                         "pinnacle_unreliable_since_2025_07_23")

    def test_scheduled_match_has_no_result_or_match_stats(self):
        ingest_csv(self.db, self.download([row(FTHG="", FTAG="", FTR="")]), 2024)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM match_results").fetchone()[0], 0)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM team_match_stats").fetchone()[0], 0)

    def test_reject_html_response_and_incomplete_season(self):
        bad = self.root / "not.csv"
        bad.write_text("<html>access denied</html>")
        with self.assertRaises(ValueError):
            ingest_csv(self.db, Download(bad, OBSERVED, "https://example.invalid", False), 2024)
        with self.assertRaises(ValueError):
            ingest_csv(self.db, self.download([row()]), 2024, require_complete=True)

    def test_api_requires_key_and_does_not_log_one(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(AccessError):
            ApiFootball(self.db, HttpClient(self.root))

    def test_api_fixture_exact_match_join_and_historical_lineup(self):
        self.core()
        api_path = self.root / "api-synthetic.json"
        api_path.write_text('{"synthetic_test_only": true}')
        with self.db:
            artifact, _ = add_artifact(self.db, "api_football", "test:synthetic", api_path, OBSERVED)
        base = common("api_football", artifact, OBSERVED)
        api_rec = {"fixture": {"id": 123, "date": "2024-08-17T14:00:00Z", "status": {"short": "FT"}},
                   "league": {"id": 39, "season": 2024, "round": "Regular Season - 1"},
                   "teams": {"home": {"id": 42, "name": "Arsenal"}, "away": {"id": 45, "name": "Everton"}},
                   "goals": {"home": 2, "away": 0}}
        with self.db:
            matches = normalize_fixtures(self.db, [api_rec], base, {})
            line = {"team": {"id": 42, "name": "Arsenal"}, "formation": "4-3-3",
                    "startXI": [{"player": {"id": i, "name": f"Synthetic Player {i}", "pos": "M"}} for i in range(11)]}
            normalize_lineups(self.db, [line], base, matches[0][0], {})
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM fixtures").fetchone()[0], 1)
        n = self.db.execute("SELECT COUNT(*) FROM lineup_snapshots WHERE available_at_utc < ?",
                            (utc_string("2024-08-17T14:00:00Z"),)).fetchone()[0]
        self.assertEqual(n, 0)

    def test_wrong_team_join_and_naive_timestamp_rejected(self):
        self.core()
        artifact = self.db.execute("SELECT artifact_id FROM raw_artifacts").fetchone()[0]
        with self.assertRaises(ValueError):
            observe(self.db, "team_match_stats", {**common("football_data", artifact, OBSERVED),
                "match_id": "epl:2024:arsenal:everton", "team_id": "football_data:chelsea", "shots": 1})
        with self.assertRaises(ValueError):
            utc_string("2024-08-17T15:00:00")

    def test_historical_publication_needs_evidence(self):
        self.core()
        path = self.root / "xg.jsonl"
        record = {"table": "advanced_team_stats", "published_at_utc": "2024-08-18T00:00:00Z",
                  "data": {"match_id": "epl:2024:arsenal:everton", "team_id": "football_data:arsenal",
                           "xg_for": 1.5, "xg_against": 0.5, "definition": "SYNTHETIC TEST ONLY"}}
        path.write_text(json.dumps(record) + "\n")
        with self.assertRaises(ValueError):
            import_observations(self.db, path, self.root / "raw", "test_export", "test:synthetic")
        record["publication_evidence"] = "SYNTHETIC TEST ONLY: timestamp supplied by test provider"
        path.write_text(json.dumps(record) + "\n")
        import_observations(self.db, path, self.root / "raw", "test_export", "test:synthetic")
        self.assertEqual(self.db.execute("SELECT availability_basis FROM advanced_team_stats").fetchone()[0], "provider_published")


class CacheTest(unittest.TestCase):
    def test_cache_refresh_keeps_unchanged_observation_but_tracks_reversion(self):
        class Response:
            def __init__(self, payload): self.payload = payload
            def __enter__(self): return self
            def __exit__(self, *args): return None
            def read(self, limit): return self.payload
        class Opener:
            def __init__(self): self.payloads = iter([b"A", b"A", b"B", b"A"])
            def open(self, *args, **kwargs): return Response(next(self.payloads))
        with tempfile.TemporaryDirectory() as root:
            client = HttpClient(Path(root), interval=0)
            with patch("urllib.request.build_opener", return_value=Opener()):
                first = client.get("test", "https://example.invalid/data")
                cached = client.get("test", "https://example.invalid/data")
                same = client.get("test", "https://example.invalid/data", refresh=True)
                changed = client.get("test", "https://example.invalid/data", refresh=True)
                reverted = client.get("test", "https://example.invalid/data", refresh=True)
            self.assertTrue(cached.cached)
            self.assertEqual(first.observed_at, same.observed_at)
            self.assertEqual(first.path, reverted.path)
            self.assertGreater(reverted.observed_at, changed.observed_at)
            self.assertGreater(reverted.observed_at, first.observed_at)
            self.assertEqual(client.requests, 4)

    def test_credentials_are_not_permitted_in_urls(self):
        with tempfile.TemporaryDirectory() as root, self.assertRaises(AccessError):
            HttpClient(Path(root)).get("test", "https://example.invalid/?api_key=secret")


if __name__ == "__main__":
    unittest.main()
