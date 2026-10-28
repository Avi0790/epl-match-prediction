# Premier League match prediction

A reproducible Python project for pre-match home/draw/away probabilities, with a
separate goal model and an exploratory rolling-xG experiment. Start with
[START_HERE.md](START_HERE.md) for the Mac commands.

## Current status

- 4,560 historical matches across 2014/15–2025/26, with 9,120 Understat team-match xG observations.
- 60 additional result/statistic records from 2026/27, through October 12, 2026.
- Those 60 newer matches **do not have xG in the supplied snapshot**. The prediction
  command reports stale xG and incomplete rolling windows instead of implying a live feed.
- 64 passing unit/regression tests, including xG-only delays, strict cutoffs,
  late observations, model serialization and prediction.
- Models: Elo multinomial logistic regression, team Poisson, XGBoost, and Elo plus
  rolling xG logistic regression; raw and calibrated variants are compared.

| Experiment | Validation log loss | 2025/26 test log loss | Test accuracy |
| --- | ---: | ---: | ---: |
| Elo baseline | 0.967637 | 1.028352 | 48.2% |
| Elo + rolling xG | 0.963330 | 1.025677 | 49.2% |
| Closing market benchmark | — | 1.011767 | 49.5% |

The xG improvement is small and remains inconclusive on the test season. That
season was already inspected before the xG extension; these are exploratory
second-look results. The model has not demonstrated an advantage over closing odds.
See [XG_REPORT.md](XG_REPORT.md) for intervals and the corrected xG-only delay check.

## Run locally

Use Python 3.11+; the updated project was verified with Python 3.12. From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The updated archive includes a newly fitted xG model. This is an illustrative
pairing, not a verified fixture; the bundled history is not a live feed:

```bash
python predict.py --model models/xg/production.joblib --home Arsenal --away Chelsea --date 2026-10-31
```

The model metadata selects the baseline or xG feature builder and default database.
For explicit paths use `--db data/epl_xg.sqlite`. Both the individual and batch
prediction commands support xG artifacts:

```bash
python predict_week.py fixtures/demo.csv --model models/xg/production.joblib --output reports/demo_predictions.csv
```

Missing logistic xG differences use zero (a neutral difference); this is disclosed
in the explanations. Current output also shows each team's latest xG match date
and available xG count within its last ten result-available matches.

## Refit for future predictions

Use the classifier chosen by the earlier experiment and the independently selected
baseline goal-model family. This does not rescore or overwrite historical test reports:

```bash
python train_xg.py --production-only --selection reports/xg_verified/selection.json --models models/xg
```

The selected uncalibrated classifier and goal regressor fit all eligible post-2014/15
history. A calibrated classifier instead reserves a date-aligned block of approximately
380 matches for calibration. The output includes its fitting cutoff, feature configuration,
dataset fingerprint, dependency versions and xG coverage. Goal regressors use baseline
features; this release does not claim to have tested xG-based goal regressors.

## Reproduce the experiments

Choose fresh output directories; an existing test report is protected against
accidental overwrite. Reproduction is a verification step, not new evidence.

```bash
python train_xg.py --reports reports/xg_reproduction
python compare_models.py --reports reports/xg_reproduction
python train_xg.py --reports reports/xg_delay_reproduction --extra-xg-lag-hours 24
```

`--extra-xg-lag-hours` delays **only xG**. Baseline Elo, result, goal and shot
features remain identical. `--extra-history-lag-hours` is retained for deliberate
combined delays to result/statistic history; it does not isolate xG timing.
`reports/xg_lag24/` is the original combined-delay experiment and is retained for
traceability. `reports/xg_only_lag24/` contains the corrected experiment.

## Data and timing

`data/epl.sqlite` contains baseline data; `data/epl_xg.sqlite` adds xG. Each imported
xG row retains its first observation time and a separately declared historical
availability assumption. Historical xG is currently assumed available at the same
time as that match's result/statistics. These are reconstructed, latest-provider
snapshots, not independently verified historical publication/revision records.

The feature builder uses a strict midnight Europe/London cutoff. It forms windows
from the last 3, 5 or 10 result-available matches, preserves missing xG as missing,
and gates xG separately. A late observation retains its original match position.
No target-match or later-match xG enters that match's features under these assumptions.

`data/understat_xg.jsonl` and its archived import are supplied. To rebuild the xG
database offline under a new filename:

```bash
cp data/epl.sqlite data/epl_xg_rebuilt.sqlite
python data_ingest.py --db data/epl_xg_rebuilt.sqlite import-observations --file data/understat_xg.jsonl --source understat --source-url https://understat.com/league/EPL
```

For optional new acquisition, use a separate environment so its additional
packages do not change the modeling environment:

```bash
python3 -m venv .venv-data
.venv-data/bin/python -m pip install -r requirements-xg-data.txt
.venv-data/bin/python fetch_xg.py --start-season 2014 --end-season 2025
python understat_to_jsonl.py --csv data/understat_xg.csv --db data/epl.sqlite --output data/understat_xg_new.jsonl
```

The optional fetcher is derived from the user's earlier script, adds coverage and
value validation, and records the downloader version/observation time. It uses
soccerdata's documented Understat `read_schedule` API. Live downloading was not
rerun for this update; the provided database/JSONL support offline reproduction.
A new provider download can contain revisions and need not reproduce the archived
scores. See [DATA_ACCESS.md](DATA_ACCESS.md) for source notes.

## Code map

| File | Purpose |
| --- | --- |
| `data_ingest.py`, `epl_data/` | SQLite ingestion, provenance, validation, audits |
| `features.py`, `features_xg.py` | Baseline history and separately gated rolling xG |
| `epl_model/`, `models_xg.py` | Classifiers, goal models, probability matrices |
| `train.py`, `train_xg.py` | Chronological experiments and separate future-model fitting |
| `predict.py`, `predict_week.py` | Single-fixture and batch predictions |
| `evaluate.py`, `compare_models.py` | Metrics, calibration and paired weekly bootstrap |
| `fetch_xg.py`, `understat_to_jsonl.py` | Optional acquisition and canonical fixture joins |
| `tests/` | 64 unit/regression tests |

All active command modules are included in the installable package. The supplied
`xg_experiment/` directory and ZIP are historical copies; active code is in the
project root and `epl_model/`. `REPORT.md`, `BASELINE_README.md`, and
`MANIFEST_STAGE1.json` preserve the earlier baseline experiment/documentation.

## What the evidence supports

Use the original baseline and exploratory xG results as separate experiments.
Validation selected among 14 raw/calibrated variants. The paired bootstrap
resamples calendar weeks of fixed predictions; it does not repeat model selection
or estimate every source of training uncertainty. An interval containing zero is
inconclusive, not proof that two models are equivalent.

Future work: freeze the configurations, record predictions before kickoff, and
evaluate one predefined new block of matches. Do not select new variants by the
already examined 2025/26 test results. Team/match modeling is implemented; this
project does not yet measure individual player impact or establish betting profit.
