import csv
import tempfile
import unittest
from pathlib import Path

from predict_week import read_fixtures, build_row


class BatchForecastTests(unittest.TestCase):
    def _file(self, content):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        p = Path(tmp.name) / 'week.csv'
        p.write_text(content)
        return p

    def test_valid_fixtures(self):
        path = self._file('date,home,away\n2026-10-17, Fulham , Hull \n')
        self.assertEqual(read_fixtures(path), [{'date': '2026-10-17', 'home': 'Fulham', 'away': 'Hull'}])

    def test_missing_columns(self):
        with self.assertRaisesRegex(ValueError, 'date,home,away'):
            read_fixtures(self._file('home,away\nFulham,Hull\n'))

    def test_invalid_date(self):
        with self.assertRaisesRegex(ValueError, 'Invalid date'):
            read_fixtures(self._file('date,home,away\n17/10/26,Fulham,Hull\n'))

    def test_duplicate_game(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate fixture'):
            read_fixtures(self._file('date,home,away\n2026-10-17,Fulham,Hull\n2026-10-17,fulham,hull\n'))

    def test_identical_team(self):
        with self.assertRaisesRegex(ValueError, 'Identical'):
            read_fixtures(self._file('date,home,away\n2026-10-17,Fulham,Fulham\n'))

    def test_full_output_row(self):
        pred = {'fixture': {'date': '2026-10-17','home':'Fulham','away':'Hull'},
                'outcome_probabilities': {'H':.45, 'D':.25, 'A':.30}, 'most_likely_result':'H',
                'most_likely_score': {'home_goals':1,'away_goals':1,'probability':.124},
                'expected_home_goals':1.5,'expected_away_goals':1.2,'p_over_2_5':.52,
                'latest_available_history_match_date':'2026-10-12','model':'elo',
                'goal_model':'xgb_poisson','notes':[]}
        row = build_row(pred)
        self.assertEqual(row['home_win_pct'],45.0)
        self.assertEqual(row['modal_score'],'1-1')
        self.assertEqual(row['most_likely_outcome'],'Home win')


if __name__ == '__main__':
    unittest.main()
