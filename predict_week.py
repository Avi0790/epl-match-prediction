#!/usr/bin/env python3
"""Batch EPL forecasts from an explicitly supplied fixture CSV; NEVER verifies schedules."""
from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path

from predict import forecast

REQUIRED = ('date', 'home', 'away')
LABELS = {'H': 'Home win', 'D': 'Draw', 'A': 'Away win'}
FIELDS = (
    'date', 'home', 'away', 'home_win_pct', 'draw_pct', 'away_win_pct',
    'most_likely_outcome', 'modal_score', 'modal_score_pct',
    'expected_home_goals', 'expected_away_goals', 'over_2_5_pct',
    'latest_available_history_match_date', 'model', 'goal_model', 'notes'
)


def read_fixtures(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f'Fixture CSV not found: {path}')
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(REQUIRED).issubset(reader.fieldnames):
            raise ValueError('CSV must have date,home,away columns')
        out = []
        seen = set()
        for line_num, row in enumerate(reader, start=2):
            if not all(row.get(key, '').strip() for key in REQUIRED):
                raise ValueError(f'Empty fixture field on CSV line {line_num}')
            item = {key: row[key].strip() for key in REQUIRED}
            try:
                if item['date'] != date.fromisoformat(item['date']).isoformat():
                    raise ValueError
            except ValueError:
                raise ValueError(f'Invalid date on line {line_num}: expected YYYY-MM-DD') from None
            if item['home'].casefold() == item['away'].casefold():
                raise ValueError(f'Identical home/away on CSV line {line_num}')
            fingerprint = tuple(item[key].casefold() for key in REQUIRED)
            if fingerprint in seen:
                raise ValueError(f'Duplicate fixture on CSV line {line_num}')
            seen.add(fingerprint)
            out.append(item)
    if not out:
        raise ValueError('Fixture CSV contains no games')
    return out


def build_row(pred: dict) -> dict:
    p = pred['outcome_probabilities']
    score = pred['most_likely_score']
    return {
        'date': pred['fixture']['date'],
        'home': pred['fixture']['home'], 'away': pred['fixture']['away'],
        'home_win_pct': round(p['H'] * 100, 1),
        'draw_pct': round(p['D'] * 100, 1),
        'away_win_pct': round(p['A'] * 100, 1),
        'most_likely_outcome': LABELS[pred['most_likely_result']],
        'modal_score': f"{score['home_goals']}-{score['away_goals']}",
        'modal_score_pct': round(score['probability'] * 100, 1),
        'expected_home_goals': round(pred['expected_home_goals'], 2),
        'expected_away_goals': round(pred['expected_away_goals'], 2),
        'over_2_5_pct': round(pred['p_over_2_5'] * 100, 1),
        'latest_available_history_match_date': pred['latest_available_history_match_date'],
        'model': pred['model'], 'goal_model': pred['goal_model'],
        'notes': ' | '.join(pred['notes']),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('fixtures', type=Path, help='CSV with date,home,away')
    p.add_argument('--output', type=Path, default=Path('reports/weekly_predictions.csv'))
    p.add_argument('--db', help='Defaults to the database appropriate for the saved model')
    p.add_argument('--model', default='models/production.joblib')
    args = p.parse_args()
    try:
        fixtures = read_fixtures(args.fixtures)
        results = []
        for item in fixtures:
            pred, _ = forecast(args.db, args.model, item['home'], item['away'], item['date'])
            results.append(build_row(pred))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(results)
        print(f'Wrote {len(results)} forecasts to {args.output}')
        print('These are model estimates, not guaranteed outcomes or verified schedules.')
    except (ValueError, FileNotFoundError) as e:
        p.exit(2, f'ERROR: {e}\n')


if __name__ == '__main__':
    main()
