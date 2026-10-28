# EPL prediction engine — Stage 1 (ready to run)

This is the continuation of the existing 12-season project. It is not a freshly invented dataset.

## What is in the archive

- `data/epl.sqlite`: 4,560 completed EPL matches from 2014/15–2025/26, plus 60 from 2026/27 through October 12, 2026 (4,620 total).
- `features.py`: timestamp-gated Elo, rolling team form, venue form and rest features.
- `train.py` / `evaluate.py`: expanding walk-forward validation and the frozen 2025/26 final test.
- `predict.py`: predict an individual user-supplied matchup.
- `predict_week.py`: predict a CSV of fixtures for a whole matchweek (new for this package).
- `fixtures/2026-10-17_matchweek.csv`: 10 upcoming matches; you can edit or replace the file for future weeks.
- `reports/2026-10-17_matchweek_predictions.csv`: output from the included production model.
- `reports/test_report.json` and `REPORT.md`: reproducible evaluation.
- `tests/`: 41 unit tests (35 original plus 6 batch-prediction tests), all passing when packaged.

## Mac setup

Open Terminal. From the folder containing the downloaded ZIP file:

```bash
unzip EPL_Prediction_Engine_Stage1_Ready.zip
cd epl-predictor
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Python 3.12 is the known-tested version for the original model. If macOS `python3` is different, use `python3.12` to create the venv. Only load the included model if you trust its source; joblib files are executable serialized data.

## Step 1: validate the data

```bash
python data_ingest.py audit
```

Check `complete_380_results: true` for each historical 2014–2025 season and that the total completed count is 4,620 with the extra 2026/27 matches. Historical availability timestamps are conservative *assumptions*, not independently verified publication timestamps.

## Step 2: run tests

```bash
python -m unittest discover -s tests -v
```

Expected at packaging time: `Ran 41 tests` and `OK`.

## Step 3: run the trained model for one match

```bash
python predict.py --home Fulham --away Hull --date 2026-10-17
```

The model returns probabilities for H (home win), D (draw), and A (away win), plus expected goals, five likely scores and data cutoff.

## Step 4: predict all fixtures in a CSV

```bash
python predict_week.py fixtures/2026-10-17_matchweek.csv --output reports/my_week.csv
```

Open `reports/my_week.csv` in Numbers or Excel. For the next week, make a new input CSV with precisely these columns:

```csv
date,home,away
2026-10-24,Aston Villa,Manchester City
```

All names must match `python predict.py --list-teams`. The program does not verify the actual schedule: confirm dates and pairings from an official fixture list.

## Step 5: baseline quality, not gambling promises

Held-out 2025/26 test: 380 matches.

| Forecast | H/D/A accuracy | Log loss (lower is better) |
|---|---:|---:|
| Selected Elo model | 48.2% | 1.0284 |
| XGBoost shallow classifier | 47.1% | 1.0443 |
| Market-average closing odds (benchmark only) | 49.5% | 1.0118 |

The model has not beaten market-average closing odds. No bookmaker odds were included as prediction inputs. Past odds have later information than the midnight prediction, so comparison is intentionally hard but not a proof of betting profitability. Most-likely *scoreline* can differ from the most-likely *match result*; H/D/A pools many different scores.

## Then: Stage 2, *one improvement at a time*

1. Preserve the frozen 2025/26 test; never repeatedly tune on it.
2. Collect truly future 2026/27 results and store actual pre-match availability times where possible.
3. Add new feature *groups* (e.g. opponent-adjusted attacking/defensive rates) and compare chronologically against Elo using log loss and calibration.
4. Keep only improvements demonstrated on new data; don't assume a complex model must win.
5. Don't infer profitable multi-game combos from single-match probabilities.

See `README.md`, `DATA_ACCESS.md`, `FEATURE_SCHEMA.md`, and `REPORT.md` for implementation details and limitations.
