# Baseline feature schema

One row per completed EPL match in the feature export. A future fixture is passed
through the same `HistoryEngine.fixture` method. Feature names are enumerated by
`features.FEATURE_COLUMNS`; only those 154 columns enter the XGBoost models.
Targets and metadata never enter that list.

| Columns/pattern | Type | Meaning |
|---|---|---|
| `match_id`, `home_id`, `away_id` | string | Stable match and canonical team keys |
| `season`, `match_date`, `kickoff_utc` | integer/date/nullable UTC | Metadata; season is start year |
| `home_goals`, `away_goals`, `target` | integer | Labels only; H=0, D=1, A=2 |
| `prediction_cutoff_utc` | UTC timestamp | Midnight London on match day |
| `history_available_at` | UTC timestamp | This row's post-match availability assumption; never its prediction cutoff |
| `latest_history_available_at` | nullable UTC | Latest result/stat event consumed for this prediction |
| `latest_history_match_date` | nullable date | Latest consumed match date |
| `{home,away}_elo` | float | Elo before the fixture |
| `{home,away}_promoted_or_returning` | 0/1 | Absent from the previous EPL season; zero in warm-up season |
| `{home,away}_epl_rest_days` | nullable float | Calendar days since last **EPL** match |
| `{home,away}_epl_matches_{14d,30d}` | integer | Previous EPL matches within an inclusive 14/30 calendar-day window |
| `{home,away}_epl_short_rest_3d` | 0/1 | Previous EPL match at most three days earlier; missing history is zero, with rest itself still missing |
| `{home,away}_epl_games_this_season` | integer | Previously played EPL fixtures this season |
| `{home,away}_{form,venue}_n_{3,5,10}` | integer | Number of available matches in the requested rolling window |
| `{home,away}_{form,venue}_{metric}_{3,5,10}` | nullable float | Decayed weighted mean, newest weight 1 then 0.8, 0.64, ... |
| `elo_difference` | float | Home Elo minus away Elo |
| `epl_rest_difference` | nullable float | Home EPL rest minus away EPL rest |
| `diff_{form,venue}_{metric}_{3,5,10}` | nullable float | Home feature minus corresponding away feature |

`metric` is one of `points`, `gf`, `ga`, `shots_for`, `shots_against`, `sot_for`,
`sot_against`. `form` uses every eligible prior EPL match; `venue` uses home matches
for the home team and away matches for the away team. It uses the last N matches
in that venue, not a venue filter applied after taking N overall matches.

Rolling means renormalize weights over non-missing values; an entirely missing
window stays NaN. Windows contain at most ten matches and can cross season boundaries
for continuing EPL teams. Re-entering clubs have their form cleared because missing
lower-division fixtures make their old EPL form stale.

There are 110 side-specific columns, two direct differences, and 42 rolling
differences. No xG proxy is mislabeled as xG. Shots and shots on target retain their
own names. Learned team effects use team IDs in the Poisson baseline only; the boosted
models use the numeric feature whitelist.

The ingestion schema is unchanged. Locally computed Elo lives in feature/model
state rather than being inserted into the external-provider `elo_ratings` table.

## Model and prediction artifacts

`models/*.json` records feature settings, selected models, training/calibration
dates, fit-information cutoff, dependency versions, data SHA-256 and model checksum.
`models/*.joblib` contains the corresponding fitted objects. Weekly versions are
retained under `reports/updates/`.

`reports/test_predictions.csv` has one row per match and classifier/benchmark:
`match_id, season, match_date, target, home_goals, away_goals, model, p_H, p_D, p_A`.
`reports/score_predictions.csv` includes predicted score, expected goals, O/U
probability and the five explicit scores with probabilities.

CLI JSON includes a `fixture`, H/D/A probabilities, expected goals, raw Poisson rates,
modal score, top five scores, O/U probability, omitted tail mass, selected models,
key-factor method/contributions, descriptive context, cutoff, freshness and scope.
Optional matrix JSON uses **rows = home goals; columns = away goals**, with no
overflow bucket disguised as an exact score.
