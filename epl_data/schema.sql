PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_metadata (
    version INTEGER PRIMARY KEY,
    description TEXT NOT NULL
) STRICT;
INSERT OR IGNORE INTO schema_metadata VALUES (1, 'Ingestion schema; no model features');

CREATE TABLE IF NOT EXISTS raw_artifacts (
    artifact_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_url TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    first_observed_at_utc TEXT NOT NULL,
    local_path TEXT NOT NULL,
    byte_count INTEGER NOT NULL CHECK(byte_count > 0),
    UNIQUE(source, source_url, sha256, first_observed_at_utc)
) STRICT;

CREATE TABLE IF NOT EXISTS teams (
    team_id TEXT PRIMARY KEY,
    name TEXT NOT NULL
) STRICT;
CREATE TABLE IF NOT EXISTS team_aliases (
    source TEXT NOT NULL,
    source_team_key TEXT NOT NULL,
    team_id TEXT NOT NULL REFERENCES teams(team_id),
    PRIMARY KEY(source, source_team_key)
) STRICT;
CREATE TABLE IF NOT EXISTS players (
    player_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_player_key TEXT NOT NULL,
    name TEXT NOT NULL,
    UNIQUE(source, source_player_key)
) STRICT;

CREATE TABLE IF NOT EXISTS data_issues (
    issue_id TEXT PRIMARY KEY,
    artifact_id TEXT NOT NULL REFERENCES raw_artifacts(artifact_id),
    match_id TEXT,
    code TEXT NOT NULL,
    detail_json TEXT NOT NULL CHECK(json_valid(detail_json))
) STRICT;

-- Match identity does not include a date: postponements must not create new matches.
CREATE TABLE IF NOT EXISTS fixtures (
    match_id TEXT PRIMARY KEY,
    competition_id TEXT NOT NULL,
    season_start INTEGER NOT NULL CHECK(season_start >= 1800),
    home_team_id TEXT NOT NULL REFERENCES teams(team_id),
    away_team_id TEXT NOT NULL REFERENCES teams(team_id),
    CHECK(home_team_id != away_team_id)
) STRICT;
CREATE TABLE IF NOT EXISTS fixture_aliases (
    source TEXT NOT NULL,
    source_fixture_key TEXT NOT NULL,
    match_id TEXT NOT NULL REFERENCES fixtures(match_id),
    PRIMARY KEY(source, source_fixture_key)
) STRICT;

CREATE TABLE IF NOT EXISTS fixture_versions (
    observation_id TEXT PRIMARY KEY,
    match_id TEXT NOT NULL REFERENCES fixtures(match_id),
    match_date TEXT NOT NULL,
    kickoff_utc TEXT,
    kickoff_precision TEXT NOT NULL CHECK(kickoff_precision IN ('exact','minute_assumed_zone','date_only')),
    source_timezone TEXT,
    status TEXT NOT NULL,
    matchweek INTEGER,
    referee TEXT,
    neutral_venue INTEGER CHECK(neutral_venue IN (0,1)),
    source TEXT NOT NULL,
    artifact_id TEXT NOT NULL REFERENCES raw_artifacts(artifact_id),
    available_at_utc TEXT NOT NULL,
    availability_basis TEXT NOT NULL CHECK(availability_basis IN ('first_observed','provider_published')),
    CHECK(kickoff_precision != 'date_only' OR kickoff_utc IS NULL)
) STRICT;

-- Full-time league results are labels, never same-match explanatory variables.
CREATE TABLE IF NOT EXISTS match_results (
    observation_id TEXT PRIMARY KEY,
    match_id TEXT NOT NULL REFERENCES fixtures(match_id),
    home_goals INTEGER NOT NULL CHECK(home_goals >= 0),
    away_goals INTEGER NOT NULL CHECK(away_goals >= 0),
    result TEXT NOT NULL CHECK(result IN ('H','D','A')),
    score_period TEXT NOT NULL CHECK(score_period IN ('regulation','after_extra_time')),
    source TEXT NOT NULL,
    artifact_id TEXT NOT NULL REFERENCES raw_artifacts(artifact_id),
    available_at_utc TEXT NOT NULL,
    availability_basis TEXT NOT NULL CHECK(availability_basis IN ('first_observed','provider_published')),
    assumed_available_at_utc TEXT,
    CHECK((home_goals > away_goals AND result = 'H') OR
          (home_goals = away_goals AND result = 'D') OR
          (home_goals < away_goals AND result = 'A'))
) STRICT;

CREATE TABLE IF NOT EXISTS team_match_stats (
    observation_id TEXT PRIMARY KEY,
    match_id TEXT NOT NULL REFERENCES fixtures(match_id),
    team_id TEXT NOT NULL REFERENCES teams(team_id),
    shots INTEGER CHECK(shots >= 0),
    shots_on_target INTEGER CHECK(shots_on_target >= 0),
    corners INTEGER CHECK(corners >= 0),
    fouls INTEGER CHECK(fouls >= 0),
    yellow_cards INTEGER CHECK(yellow_cards >= 0),
    red_cards INTEGER CHECK(red_cards >= 0),
    penalties_awarded INTEGER CHECK(penalties_awarded >= 0),
    source TEXT NOT NULL,
    artifact_id TEXT NOT NULL REFERENCES raw_artifacts(artifact_id),
    available_at_utc TEXT NOT NULL,
    availability_basis TEXT NOT NULL CHECK(availability_basis IN ('first_observed','provider_published')),
    assumed_available_at_utc TEXT,
    CHECK(shots_on_target IS NULL OR shots IS NULL OR shots_on_target <= shots)
) STRICT;

-- Provider and definition are retained; xG models are not silently interchangeable.
CREATE TABLE IF NOT EXISTS advanced_team_stats (
    observation_id TEXT PRIMARY KEY,
    match_id TEXT NOT NULL REFERENCES fixtures(match_id),
    team_id TEXT NOT NULL REFERENCES teams(team_id),
    xg_for REAL NOT NULL CHECK(xg_for >= 0),
    xg_against REAL NOT NULL CHECK(xg_against >= 0),
    definition TEXT NOT NULL,
    source TEXT NOT NULL,
    artifact_id TEXT NOT NULL REFERENCES raw_artifacts(artifact_id),
    available_at_utc TEXT NOT NULL,
    availability_basis TEXT NOT NULL CHECK(availability_basis IN ('first_observed','provider_published')),
    assumed_available_at_utc TEXT
) STRICT;

CREATE TABLE IF NOT EXISTS player_match_stats (
    observation_id TEXT PRIMARY KEY,
    match_id TEXT NOT NULL REFERENCES fixtures(match_id),
    team_id TEXT NOT NULL REFERENCES teams(team_id),
    player_id TEXT NOT NULL REFERENCES players(player_id),
    minutes REAL CHECK(minutes >= 0 AND minutes <= 150),
    goals INTEGER CHECK(goals >= 0),
    assists INTEGER CHECK(assists >= 0),
    xg REAL CHECK(xg >= 0),
    xa REAL CHECK(xa >= 0),
    pressures INTEGER CHECK(pressures >= 0),
    performance_rating REAL,
    position TEXT,
    definition TEXT NOT NULL,
    source TEXT NOT NULL,
    artifact_id TEXT NOT NULL REFERENCES raw_artifacts(artifact_id),
    available_at_utc TEXT NOT NULL,
    availability_basis TEXT NOT NULL CHECK(availability_basis IN ('first_observed','provider_published')),
    assumed_available_at_utc TEXT
) STRICT;

CREATE TABLE IF NOT EXISTS odds_snapshots (
    observation_id TEXT PRIMARY KEY,
    match_id TEXT NOT NULL REFERENCES fixtures(match_id),
    bookmaker TEXT NOT NULL,
    market TEXT NOT NULL CHECK(market IN ('1x2','ou_2_5')),
    phase TEXT NOT NULL CHECK(phase IN ('pre_closing','closing','timestamped_prematch')),
    quote_time_utc TEXT,
    odds_home REAL CHECK(odds_home > 1),
    odds_draw REAL CHECK(odds_draw > 1),
    odds_away REAL CHECK(odds_away > 1),
    odds_over REAL CHECK(odds_over > 1),
    odds_under REAL CHECK(odds_under > 1),
    quality_flag TEXT,
    source TEXT NOT NULL,
    artifact_id TEXT NOT NULL REFERENCES raw_artifacts(artifact_id),
    available_at_utc TEXT NOT NULL,
    availability_basis TEXT NOT NULL CHECK(availability_basis IN ('first_observed','provider_published')),
    CHECK((market = '1x2' AND odds_home IS NOT NULL AND odds_draw IS NOT NULL AND odds_away IS NOT NULL
           AND odds_over IS NULL AND odds_under IS NULL) OR
          (market = 'ou_2_5' AND odds_over IS NOT NULL AND odds_under IS NOT NULL
           AND odds_home IS NULL AND odds_draw IS NULL AND odds_away IS NULL))
) STRICT;

CREATE TABLE IF NOT EXISTS elo_ratings (
    observation_id TEXT PRIMARY KEY,
    team_id TEXT NOT NULL REFERENCES teams(team_id),
    rating_date TEXT NOT NULL,
    elo REAL NOT NULL CHECK(elo > 0),
    source_valid_from TEXT,
    source_valid_to TEXT,
    source TEXT NOT NULL,
    artifact_id TEXT NOT NULL REFERENCES raw_artifacts(artifact_id),
    available_at_utc TEXT NOT NULL,
    availability_basis TEXT NOT NULL CHECK(availability_basis IN ('first_observed','provider_published'))
) STRICT;

-- One whole snapshot per team/fixture. An empty successful snapshot differs from missing data.
CREATE TABLE IF NOT EXISTS availability_snapshots (
    observation_id TEXT PRIMARY KEY,
    match_id TEXT NOT NULL REFERENCES fixtures(match_id),
    team_id TEXT NOT NULL REFERENCES teams(team_id),
    coverage_status TEXT NOT NULL CHECK(coverage_status IN ('reported','complete','unknown')),
    source TEXT NOT NULL,
    artifact_id TEXT NOT NULL REFERENCES raw_artifacts(artifact_id),
    available_at_utc TEXT NOT NULL,
    availability_basis TEXT NOT NULL CHECK(availability_basis IN ('first_observed','provider_published'))
) STRICT;
CREATE TABLE IF NOT EXISTS unavailable_players (
    snapshot_id TEXT NOT NULL REFERENCES availability_snapshots(observation_id),
    player_id TEXT NOT NULL REFERENCES players(player_id),
    status TEXT NOT NULL CHECK(status IN ('out','doubtful','suspended','unknown')),
    reason TEXT,
    expected_return_date TEXT,
    PRIMARY KEY(snapshot_id, player_id)
) STRICT;

CREATE TABLE IF NOT EXISTS lineup_snapshots (
    observation_id TEXT PRIMARY KEY,
    match_id TEXT NOT NULL REFERENCES fixtures(match_id),
    team_id TEXT NOT NULL REFERENCES teams(team_id),
    kind TEXT NOT NULL CHECK(kind IN ('likely','confirmed')),
    formation TEXT,
    source TEXT NOT NULL,
    artifact_id TEXT NOT NULL REFERENCES raw_artifacts(artifact_id),
    available_at_utc TEXT NOT NULL,
    availability_basis TEXT NOT NULL CHECK(availability_basis IN ('first_observed','provider_published'))
) STRICT;
CREATE TABLE IF NOT EXISTS lineup_players (
    snapshot_id TEXT NOT NULL REFERENCES lineup_snapshots(observation_id),
    player_id TEXT NOT NULL REFERENCES players(player_id),
    is_starter INTEGER NOT NULL CHECK(is_starter IN (0,1)),
    start_probability REAL CHECK(start_probability BETWEEN 0 AND 1),
    position TEXT,
    PRIMARY KEY(snapshot_id, player_id)
) STRICT;

CREATE TABLE IF NOT EXISTS player_valuations (
    observation_id TEXT PRIMARY KEY,
    player_id TEXT NOT NULL REFERENCES players(player_id),
    team_id TEXT NOT NULL REFERENCES teams(team_id),
    valuation_date TEXT NOT NULL,
    market_value REAL NOT NULL CHECK(market_value >= 0),
    currency TEXT NOT NULL CHECK(length(currency) = 3),
    source TEXT NOT NULL,
    artifact_id TEXT NOT NULL REFERENCES raw_artifacts(artifact_id),
    available_at_utc TEXT NOT NULL,
    availability_basis TEXT NOT NULL CHECK(availability_basis IN ('first_observed','provider_published'))
) STRICT;

CREATE TABLE IF NOT EXISTS team_context_snapshots (
    observation_id TEXT PRIMARY KEY,
    team_id TEXT NOT NULL REFERENCES teams(team_id),
    season_start INTEGER NOT NULL,
    effective_date TEXT NOT NULL,
    manager_id TEXT,
    manager_started_at_utc TEXT,
    captain_player_id TEXT REFERENCES players(player_id),
    first_choice_gk_id TEXT REFERENCES players(player_id),
    promoted INTEGER CHECK(promoted IN (0,1)),
    source TEXT NOT NULL,
    artifact_id TEXT NOT NULL REFERENCES raw_artifacts(artifact_id),
    available_at_utc TEXT NOT NULL,
    availability_basis TEXT NOT NULL CHECK(availability_basis IN ('first_observed','provider_published'))
) STRICT;

CREATE TABLE IF NOT EXISTS fixture_context_snapshots (
    observation_id TEXT PRIMARY KEY,
    match_id TEXT NOT NULL REFERENCES fixtures(match_id),
    derby INTEGER NOT NULL CHECK(derby IN (0,1)),
    definition TEXT NOT NULL,
    source TEXT NOT NULL,
    artifact_id TEXT NOT NULL REFERENCES raw_artifacts(artifact_id),
    available_at_utc TEXT NOT NULL,
    availability_basis TEXT NOT NULL CHECK(availability_basis IN ('first_observed','provider_published'))
) STRICT;

CREATE TABLE IF NOT EXISTS competition_coverage (
    observation_id TEXT PRIMARY KEY,
    competition_id TEXT NOT NULL,
    season_start INTEGER NOT NULL,
    coverage_json TEXT NOT NULL CHECK(json_valid(coverage_json)),
    source TEXT NOT NULL,
    artifact_id TEXT NOT NULL REFERENCES raw_artifacts(artifact_id),
    available_at_utc TEXT NOT NULL,
    availability_basis TEXT NOT NULL CHECK(availability_basis IN ('first_observed','provider_published'))
) STRICT;

CREATE INDEX IF NOT EXISTS idx_fixture_time ON fixture_versions(match_id, available_at_utc);
CREATE INDEX IF NOT EXISTS idx_results_time ON match_results(match_id, available_at_utc);
CREATE INDEX IF NOT EXISTS idx_team_stats_time ON team_match_stats(team_id, available_at_utc);
CREATE INDEX IF NOT EXISTS idx_player_stats_time ON player_match_stats(player_id, available_at_utc);
CREATE INDEX IF NOT EXISTS idx_elo_time ON elo_ratings(team_id, rating_date, available_at_utc);
CREATE INDEX IF NOT EXISTS idx_odds_time ON odds_snapshots(match_id, bookmaker, market, phase, available_at_utc);

-- Latest views are for inspection/labels only. NEVER use them for historical feature joins.
CREATE VIEW IF NOT EXISTS latest_fixture_versions AS
SELECT * FROM (
 SELECT *, ROW_NUMBER() OVER (PARTITION BY match_id, source ORDER BY available_at_utc DESC, rowid DESC) AS revision_rank
 FROM fixture_versions
) WHERE revision_rank = 1;
CREATE VIEW IF NOT EXISTS latest_match_results AS
SELECT * FROM (
 SELECT *, ROW_NUMBER() OVER (PARTITION BY match_id, source ORDER BY available_at_utc DESC, rowid DESC) AS revision_rank
 FROM match_results
) WHERE revision_rank = 1;
CREATE VIEW IF NOT EXISTS latest_odds_snapshots AS
SELECT * FROM (
 SELECT *, ROW_NUMBER() OVER (
 PARTITION BY match_id, source, bookmaker, market, phase ORDER BY available_at_utc DESC, rowid DESC) AS revision_rank
 FROM odds_snapshots
) WHERE revision_rank = 1;
