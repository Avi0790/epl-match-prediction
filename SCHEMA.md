# Ingestion schema

The executable definition is `epl_data/schema.sql` (schema version 1).
The baseline feature/model schema is documented separately in `FEATURE_SCHEMA.md`.
SQLite STRICT tables enforce types, checks and foreign keys. All timestamps
use fixed-width UTC ISO-8601 with microseconds, such as
`2026-10-15T05:30:00.000000Z`. Calendar dates use `YYYY-MM-DD`.
NULL means unknown/unavailable, never zero. IDs are TEXT; season starts are INTEGER;
counts are INTEGER; decimal odds, xG, minutes, valuations and ratings are REAL.

## Shared provenance on every fact/snapshot

| Field | Meaning |
|---|---|
| `observation_id` | Deterministic observation key; repeated ingestion is idempotent |
| `source` | Provider identifier, retained when providers disagree |
| `artifact_id` | Foreign key to immutable raw bytes and acquisition metadata |
| `available_at_utc` | Supported publication time, otherwise when this version was first observed |
| `availability_basis` | `provider_published` or `first_observed` |
| `assumed_available_at_utc` | Present only on post-match results/statistics; a separate retrospective assumption |

`raw_artifacts.first_observed_at_utc` records retrieval/observation, not match time.
An import's publication time needs retained evidence in the raw JSONL. That evidence
must refer to the particular data version, not just the date of the event.

An unchanged refresh retains its first observation. A change appends a revision.
An A→B→A provider correction is a new observation even though the last byte blob
matches the original. Raw filenames are content-addressed; observation identity
also contains source URL and first-observed time.

## Tables and grain

| Table | One row represents | Important fields | Populated now? |
|---|---|---|---|
| `raw_artifacts` | One observed version of a source response | source URL, SHA-256, first-observed time, byte count, local raw path | Yes |
| `teams`, `team_aliases` | Canonical club / explicit provider alias | team ID, name, source, source team key | Yes |
| `players` | Provider-qualified player identity | player ID, source player key, display name | Awaiting provider |
| `fixtures`, `fixture_aliases` | Stable match identity / provider fixture mapping | competition, season, home/away IDs | Yes |
| `fixture_versions` | Observed schedule/context revision | match date, kickoff UTC, precision, status, matchweek, referee, neutral venue | Yes; unsupported context is NULL |
| `match_results` | Observed final-score revision | home/away goals, H/D/A result, regulation/extra-time indicator | Yes, regulation EPL |
| `team_match_stats` | Team × completed match × version | shots, shots on target, corners, fouls, yellows, reds, penalties awarded | Yes; penalties NULL |
| `advanced_team_stats` | Team × completed match × provider/version | xG for/against, provider metric definition | Awaiting authorized xG |
| `player_match_stats` | Player × team × completed match × provider/version | minutes, goals, assists, xG, xA, pressures, rating, position, definition | Awaiting provider |
| `odds_snapshots` | Match × bookmaker × market × phase × version | decimal prices, quote time, reliability flag | Yes; quote time NULL |
| `elo_ratings` | Team rating effective date × provider/version | Elo, rating date, original validity dates | Awaiting authorized ClubElo data |
| `availability_snapshots` | Team × target fixture × report version | coverage: reported/complete/unknown | Awaiting provider |
| `unavailable_players` | Player within an availability snapshot | out/doubtful/suspended/unknown, reason, expected return | Awaiting provider |
| `lineup_snapshots` | Team × target fixture × lineup version | likely/confirmed, formation | Awaiting provider |
| `lineup_players` | Player within a lineup snapshot | starter flag, start probability, position | Awaiting provider |
| `player_valuations` | Player valuation observed at a historical date | team, valuation date, amount, currency | Awaiting authorized data |
| `team_context_snapshots` | Historical team/season context version | manager/start date, captain, first-choice GK, promotion status | Awaiting curated/provider data |
| `fixture_context_snapshots` | Version of a fixture's defined context | derby flag, explicit definition | Awaiting curated/provider data |
| `competition_coverage` | Provider competition/season coverage response | provider coverage flags as JSON | Awaiting API key |
| `data_issues` | Detected source anomaly | artifact, match, issue code, raw values and normalization action | Yes |

## Match identity and schedule

`fixtures` contains `match_id`, `competition_id`, `season_start`, `home_team_id`,
and `away_team_id`. The EPL ID is based on season plus home/away pairing, e.g.
`epl:2024:arsenal:everton`. It never changes when the kickoff is postponed.
Non-league fixtures use provider fixture IDs because repeated pairings are possible.

The schedule is versioned separately. `kickoff_precision` is one of:

- `exact`: provider supplied a timezone-aware timestamp.
- `minute_assumed_zone`: source supplied clock time; the configured timezone was assumed.
- `date_only`: no clock time; `kickoff_utc` is NULL.

Fixture records contain no target goals. Historical referee names in a final CSV
are not evidence that the referee assignment was available at an earlier cutoff.
Matchweek is not inferred from sorted row number: postponements break that shortcut.

For full rest/congestion features, ingest each relevant EPL team's fixtures across
competitions and required seasons, including the previous season at the boundary.
Deduplicate with fixture aliases. Played FT/AET/PEN fixtures can contribute to rest
history; cancelled/postponed/unplayed matches must not. A coverage flag alone does
not establish complete collection. Reconcile counts against competition schedules
before treating an empty interval as genuine rest.

## Results, statistics and odds

Results enforce the H/D/A label against nonnegative final goals. League targets
are regulation time. The current API adapter preserves extra-time/shootout fixture
status for workload history but does not turn those totals into regulation labels.

Basic football-data mapping:

| Source columns | Canonical fields |
|---|---|
| `FTHG`, `FTAG`, `FTR` | home goals, away goals, H/D/A target |
| `HS`, `AS` | home/away shots |
| `HST`, `AST` | home/away shots on target |
| `HC`, `AC` | corners |
| `HF`, `AF` | fouls |
| `HY`, `AY`, `HR`, `AR` | yellow/red cards |
| `Referee` | observed referee on fixture revision |

Each odds row contains either a complete `(odds_home, odds_draw, odds_away)`
triple for `market='1x2'`, or `(odds_over, odds_under)` for `market='ou_2_5'`.
Prices must exceed 1. The other market's fields are NULL.

`phase` is `pre_closing`, `closing`, or `timestamped_prematch`. `B365H/D/A` are
pre-closing; `B365CH/CD/CA` are closing. `PSCH/PSCD/PSCA` are older Pinnacle
closing columns. Market maxima are stored but are not the default benchmark.
Provider odds timestamps are unknown in these CSVs, so no timestamp is fabricated.
Later evaluation should de-vig a complete market consistently, e.g.
`p_i = (1 / odds_i) / sum_j(1 / odds_j)`, and state that this is an approximation.

## Snapshot and player rules

- An availability snapshot with no returned players has `coverage_status='unknown'`
  in the API adapter. It does not become a declaration of a fully fit squad.
- Confirmed lineups require 11 distinct starters. Bench members are retained.
  Likely lineups may retain probabilistic starts; their forecast provenance must
  be timestamped. Missing lineups remain missing.
- Post-match performance rating is a result of that player's appearance. It is
  not a pre-match ability rating for the same game.
- Injury burden (past minutes, past xG+xA, historical value), top-scorer/key-player
  flags, manager-change windows, rolling form, Elo differences, and expected-XI
  strength are computed later. They are not filled using end-of-season aggregates.
- Values retain currency and valuation dates. Never sum currencies without a
  documented historical FX conversion. Final injury end dates are not backfilled
  as if the return date was known earlier.
- Preserve different xG/pressure definitions by provider. Do not silently combine
  incompatible measurement systems or treat unavailable pressures as zero.

## JSONL authorized-import contract

One UTF-8 JSON object per line. Required top-level keys are `table` and `data`.
Supported tables are advanced/team/player match stats, odds, Elo, availability,
lineups, valuations and context. Fixture identities must already exist.
Reserved source/timing/ID fields cannot be passed inside `data`.

Structural example only; placeholders must be replaced with real authorized data:

```json
{
  "table": "advanced_team_stats",
  "data": {
    "match_id": "CANONICAL_MATCH_ID",
    "team_id": "CANONICAL_TEAM_ID",
    "xg_for": 1.25,
    "xg_against": 0.75,
    "definition": "PROVIDER_AND_METRIC_VERSION"
  },
  "published_at_utc": "2024-08-18T08:00:00Z",
  "publication_evidence": "REFERENCE_TO_ARCHIVE_PROVING_PUBLICATION_OF_THIS_VERSION"
}
```

Put that object on one line for JSONL. The displayed numbers are examples, not
match observations. Omit the two publication fields when there is no supporting
timestamp; the importer then uses actual first observation. It never guesses.

For player-based records, declare identities in a top-level `players` array with
`source_player_key` and `name`. Canonical IDs become `SOURCE:SOURCE_PLAYER_KEY`.
For lineup/availability headers, a top-level `members` array contains the child
table fields (excluding `snapshot_id`, which is assigned automatically). See the
SQL definition for exact types and nullable fields. Duplicate snapshot headers or
duplicate members in an import are rejected transactionally.

## Point-in-time query contract for the next stage

For every target fixture, fix an explicit prediction cutoff earlier than kickoff.
Filter each source's observations by `available_at_utc < cutoff`, then choose
the latest eligible version within the full natural key (including source).
For match-derived histories, also require contributing match completion before
cutoff. Do not choose the latest revision first and filter afterward.

Strict observed-time data collected today cannot populate a past backtest.
Historical result-based research can separately use documented reporting-lag
assumptions and sensitivity analysis; that is a distinct, labeled research mode.
There is intentionally no universal SQL view called `training_features` that
hides this choice.

For rating histories, use a rating effective before the cutoff and a version
actually available then. In a retrospective daily ClubElo experiment, use at
least the prior day and document the assumption; never use a same-day rating
that may include the target match. `source_valid_to` can reveal a later update
and must not itself become a predictor.

The `latest_fixture_versions`, `latest_match_results`, and `latest_odds_snapshots`
views retain separate source versions. They support inspection and label selection,
not time-travel training joins. Prediction-time source precedence and all missingness
policies must be explicit in the future feature configuration.
