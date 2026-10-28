Historical Stage 1 documentation; see README.md for the current workflow.

# EPL outcome and scoreline predictor

A complete, free-data baseline with reproducible ingestion, chronological features,
trained models, a fixture CLI, and an honest bookmaker comparison. **No API key is
needed to run it.** Advanced data is deferred until it demonstrates incremental value.

The frozen **2025/26 test contains 380 matches**. The selected Elo classifier scored
**1.0284 log loss / 0.2092 RPS / 48.2% accuracy**, versus **1.0118 / 0.2045 / 49.5%**
for normalized market-average closing odds. It **did not beat the bookmaker benchmark**.
The selected XGBoost Poisson goal model, reconciled to the Elo probabilities, scored
11.6% exact-score accuracy and 53.2% over/under 2.5 accuracy. See [REPORT.md](REPORT.md).

## Quick start

Use Python **3.12** (tested); Python 3.11+ is supported. SQLite 3.37+ is required.
Unzip and open a terminal inside the `epl-predictor` folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python predict.py --list-teams
```

Windows activation is `.venv\Scripts\activate`. For platform-specific XGBoost
installation requirements, use its [official installation guide](https://xgboost.readthedocs.io/en/release_3.0.0/install.html).
Only load trusted `.joblib` files: Python model serialization can execute code.
The included models were trained with the dependency versions in `requirements.txt`.

Run a prediction immediately with the included production model:

```bash
# Illustrative user-supplied pairing; this is NOT a claim about the actual schedule.
python predict.py --home Arsenal --away Everton --date 2026-10-17
python predict.py --home Arsenal --away Everton --date 2026-10-17 --json \
  --matrix-json reports/my_score_matrix.json
```

Use exact names from `--list-teams`, or canonical IDs such as `football_data:arsenal`.
The CLI returns H/D/A, expected home and away goals, the modal score, top five
scorelines, over-2.5 probability, model factors, the data cutoff, and freshness notes.
A supplied fixture is **not verified against a live fixture schedule**. The CLI
rejects predictions dated before the model's fitting information was available.
For past matches use the saved walk-forward/test predictions, not today's model.

## What is included

| Component | Implementation |
|---|---|
| `data_ingest.py`, `epl_data/` | Cached, rate-limited public CSV ingestion, provenance, validation, optional API adapter |
| `features.py` | Own Elo; decayed 3/5/10-match form; separate venue form; EPL rest/congestion; newcomer indicator |
| `epl_model/models.py` | Elo multinomial logistic classifier, team Poisson, two XGBoost classifiers, two XGBoost Poisson goal regressors, Platt calibration |
| `train.py` | Six expanding validation seasons, model selection, final holdout, separate production fitting |
| `evaluate.py` | Log loss, Brier, RPS, accuracy, score/O-U metrics, calibration plots and paired uncertainty |
| `predict.py` | Fixture CLI, coherent score matrix and local explanations |
| `weekly_update.py` | Public-data refresh, fixed-model production refit, versioned outputs |
| `models/` | Frozen research model and refreshed production model, with metadata/checksums |
| `reports/` | Predictions, metrics, split membership, calibration plot/bins, SHAP, source checks, test output |
| `tests/` | 35 passing tests, including future-data mutation and cutoff checks |

The research snapshot has **4,560 matches across 2014/15–2025/26**. A subsequent
production refresh added **60 matches from 2026/27**, through **12 October 2026**,
for **4,620 matches total**. These extra matches do not enter historical model
selection or final-test evaluation. The database also retains basic match statistics,
referees and odds. It has no fabricated xG, injury, valuation or lineup observations.

## Prediction horizon and information timing

Every baseline forecast is made at **00:00 Europe/London on match day**. This works
with the 1,900 historical matches that lack kickoff clock times, and prevents
same-day results from leaking into another match's features.

This experiment is explicitly **`retrospective_lagged`**, not a certified historical
publication archive. Original CSV publication/revision times cannot be recovered
from a current download. The loader deliberately reads the latest reconstructed
archive and applies the separate research availability assumption:

- Known clock time: result/statistics available at kickoff plus 24 hours.
- Missing clock time: available at midnight two local calendar dates later.
- A record enters form/Elo only when its assumed availability is **strictly before**
  the prediction cutoff. All three result/home-stat/away-stat timestamps must qualify.
- Same-match outcomes/statistics never enter that match's features. Unavailable
  shot values remain missing. Schedule-derived rest uses previously played dates.
- Observation timestamps remain unchanged in SQLite. No code relabels these assumptions
  as verified historical publication times. Later corrections can still contaminate
  a reconstructed archive; this limitation cannot be solved by shifting rows alone.

`train.py --extra-history-lag-hours 24 --reports reports/lag_sensitivity --models models/lag_sensitivity`
adds a stricter reporting lag for a **separate research experiment**. Do not use the
known final test to choose a better lag or tune parameters.

See [FEATURE_SCHEMA.md](FEATURE_SCHEMA.md) for the 154 numeric features and
[SCHEMA.md](SCHEMA.md) for the normalized ingestion database.

## Chronological validation

2014/15 supplies warm-up history; its labels are not used to fit the statistical models.
For each validation season from **2018/19 through 2023/24**:

1. Fit on 2015/16 through the season two years before validation.
2. Fit optional one-versus-rest Platt calibration on the intervening season.
3. Score the next season with model parameters fixed. Features advance using only
   earlier, eligible matches, including eligible matches within the evaluation season.

Pool the six validation seasons to select the classifier by log loss. Choose the
goal model separately by independent Poisson goal negative log likelihood. Both raw
and calibrated classifiers are candidates. **Raw Elo won**, so the deployed outcome
model does not apply Platt scaling; calibration was implemented and tested, not
assumed beneficial. This is a match-by-match backtest, not a prediction of an entire
season before its first match.

Freeze selection before evaluating **2025/26**. Final fitting uses **2015/16–2023/24**;
**2024/25** is reserved for calibration, and **2025/26** is evaluated once. There is no
random training split, shuffled CV, random validation subset, or early stopping.
A seeded bootstrap is used only for uncertainty after evaluation.

```bash
# Reproduce in separate directories, preserving the supplied final test report.
python train.py --reports reports/reproduction --models models/reproduction
python evaluate.py --predictions reports/test_predictions.csv

# Generate the current feature table for inspection.
python features.py --output data/features.csv
```

The default `train.py` refuses to overwrite an existing final report without
`--overwrite`. That flag is for exact reproduction, not repeated tuning against
the test. Record new experiments and use genuinely future matches for new evidence.

## Models, calibration and scorelines

Elo starts at 1,500, uses K=20 and a 70-point home advantage, shrinks toward 1,500
by 25% each season, and assigns promoted/returning clubs 1,450 on re-entry.
These settings are fixed, not fitted against the final season. A multinomial logistic
model maps the pre-match Elo difference to H/D/A probabilities, including draws.

The team Poisson baseline fits regularized attacking and defending team effects
and home advantage; training weights have a 730-day half-life. Unknown teams use
league-level effects. It is **independent Poisson, not Dixon–Coles or bivariate Poisson**.
XGBoost classifiers use 154 causal numeric features with native missing-value handling.
Two `count:poisson` regressors predict home and away goal rates.

Scorelines start from independent Poisson distributions with an adaptive upper limit
and omitted tail mass below 1e-10. Each home-win/draw/away-win region is then rescaled
to the selected classifier's probabilities. This makes the returned score matrix and
H/D/A probabilities agree. Expected goals are calculated from the **final matrix**;
raw Poisson rates are also returned and can differ. After rescaling, the matrix is
not a pair of independent Poissons. Its most likely score need not imply the most
likely H/D/A outcome. Exact-score accuracy and result accuracy are reported separately.

SHAP in `reports/xgb_shap.csv` is native TreeSHAP for the shallow XGBoost classifier,
averaged over the **2024/25 calibration season**. It explains raw class margins,
not calibrated probability-point effects or causal effects. The selected Elo model
has only one varying input: Elo difference. The CLI explains that input directly;
rolling form/rest are labeled as context, not invented drivers of an Elo-only result.

## Bookmaker comparison

Use one predeclared series: **market-average closing 1X2 odds**, with inverse odds
normalized across H/D/A to remove the proportional margin. All 380 final-test matches
have complete benchmark quotes. Closing over/under 2.5 is evaluated separately.
No bookmaker is selected by which one makes the model look best.

Closing odds incorporate later information than the midnight forecast, so this is a
strong information-asymmetric benchmark. Exact quote timestamps are not supplied.
Closing odds therefore do **not** enter the forecast features. A future experiment
using them must be labeled closing-informed and separated from earlier predictions.
An honest early odds-feature model needs timestamped quotes at the chosen horizon.

Brier is the **sum over three classes** (range 0–2); RPS averages squared cumulative
errors over the first two boundaries in H/D/A order. Lower is better for both and
for log loss. Accuracy is argmax H/D/A. No profitability conclusion follows from
these metrics. `REPORT.md` includes the paired log-loss/RPS differences and intervals.

## Weekly refresh

```bash
python weekly_update.py
# Offline integration run, or retrain after a manual authorized import:
python weekly_update.py --skip-download
# Explicit season-start year, if needed:
python weekly_update.py --season 2026
```

The script refreshes the current and preceding seasons, audits data, and refits
**the already selected model family/configuration**. Its final 380-or-more matches
form a chronological calibration block without splitting a calendar date. Earlier
training labels must be available before that block starts. It validates probability
and score outputs, saves a version, then atomically replaces the production model
file. It does not rerun model selection, overwrite holdout reports, or claim that a
numerically valid update is more profitable. A lock prevents overlapping updates.
Original CSV requests are spaced by three seconds, cached, and bounded by retries.
401/403 responses stop access; rate limits are respected.

This is a script; no background schedule has been installed. To run weekly on your
own computer, schedule its absolute path with your scheduler using the virtual
environment's Python. A weekly job is insufficient for confirmed lineups or late
injury/odds updates; those would require separate timestamped matchday capture.

## Data ingestion and optional access

```bash
python data_ingest.py audit
python data_ingest.py schema
python data_ingest.py football-data --start-season 2014 --end-season 2025 --require-complete
python data_ingest.py football-data --start-season 2026 --end-season 2026 --refresh
```

Global options such as `--db` and `--raw-dir` precede the subcommand. Never use
`--require-complete` for a partial season. The default pipeline needs no API key.
Optional API-Football setup, normalized imports and access limitations are in
[DATA_ACCESS.md](DATA_ACCESS.md); the adapter reads `API_FOOTBALL_KEY` from your local
environment and does not automatically load `.env`. No account or paid plan was
created. Do not publish keys or raw licensed data. Source data remains subject to
its providers' terms; this project grants no redistribution license.

The original 17 ingestion tests passed before modeling; all 35 tests now pass.
Three stored scores were independently checked against official club/UEFA reports
in `reports/source_spot_checks.json`. This is a spot check, not verification of every
statistic. Two original source anomalies remain documented: Newcastle–West Ham
15 August 2021 has inconsistent away shot fields (normalized NULL), and one bet365
closing O/U quote for Newcastle–Manchester City 19 December 2021 is zero (excluded).
Pinnacle quotes from 23 July 2025 carry the source's reliability flag; the benchmark
does not use Pinnacle.

## Limitations and next experiments

- The baseline does not beat closing odds. One test season and a few thousand matches
  are modest evidence; calibration bins and exact-score metrics are noisy.
- EPL-only rest/congestion omits cups, Europe, international duty, travel and minutes.
  These variables are not presented as true total workload.
- Promoted clubs lack Championship history. The newcomer flag is inferred from prior
  EPL participation; it is not an official promotion feed. A first-season warm-up and
  missing-value handling do not remove all cold-start uncertainty.
- Goal counts can be dependent/overdispersed; independent Poisson plus region scaling
  does not solve every scoreline calibration problem. Dixon–Coles or a negative-binomial
  alternative should be evaluated on future validation blocks before deployment.
- xG, injuries, suspensions, likely/confirmed XI, player contributions, market value,
  head-to-head, referee tendencies, derbies and manager changes are not used in this
  baseline. Corners/cards/referees are retained in ingestion for later ablations.
- No current-season schedule verification, automatically inferred captain/GK/top-scorer,
  scraping workaround, or claim of historical injury coverage is included.
- Historical publication/revision timing is assumed. Start retaining actual timestamped
  snapshots now for a genuinely observed-time forward evaluation.

Next, test one feature group at a time against the frozen baseline: improve fixture
coverage first, then evaluate permitted xG and availability/lineup inputs only if
proper timestamps and coverage exist. Keep the existing final season as a report,
not a new tuning set. Do not pay for data before establishing incremental benefit.
