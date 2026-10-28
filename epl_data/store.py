"""SQLite persistence, source identity and append-only observations."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3

UTC = timezone.utc
TABLES = frozenset({
    "fixture_versions", "match_results", "team_match_stats", "advanced_team_stats",
    "player_match_stats", "odds_snapshots", "elo_ratings", "availability_snapshots",
    "lineup_snapshots", "player_valuations", "team_context_snapshots", "fixture_context_snapshots", "competition_coverage",
})


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def utc_string(value: str) -> str:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("Timestamps must contain a timezone offset or Z")
    return dt.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def stable_id(*parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()[:32]


def connect(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    db.execute("PRAGMA busy_timeout = 10000")
    db.executescript(Path(__file__).with_name("schema.sql").read_text())
    return db


def slug(name: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")
    if not value:
        raise ValueError("Empty/non-ASCII team key; supply an explicit alias")
    return value


def ensure_team(db: sqlite3.Connection, source: str, key: str, name: str,
                canonical_id: str | None = None) -> str:
    old = db.execute("SELECT team_id FROM team_aliases WHERE source=? AND source_team_key=?",
                     (source, str(key))).fetchone()
    if old:
        if canonical_id is not None and old[0] != canonical_id:
            raise ValueError(f"Conflicting alias for {source}/{key}")
        return old[0]
    team_id = canonical_id or f"{source}:{slug(str(key))}"
    db.execute("INSERT INTO teams VALUES (?,?) ON CONFLICT(team_id) DO NOTHING", (team_id, name))
    db.execute("INSERT INTO team_aliases VALUES (?,?,?)", (source, str(key), team_id))
    return team_id


def ensure_player(db: sqlite3.Connection, source: str, key: str, name: str) -> str:
    player_id = f"{source}:{key}"
    db.execute("INSERT INTO players VALUES (?,?,?,?) ON CONFLICT(player_id) DO NOTHING",
               (player_id, source, str(key), name))
    return player_id


def ensure_fixture(db: sqlite3.Connection, match_id: str, competition_id: str,
                   season: int, home: str, away: str, source: str, source_key: str) -> None:
    identity = (match_id, competition_id, season, home, away)
    existing = db.execute("SELECT * FROM fixtures WHERE match_id=?", (match_id,)).fetchone()
    if existing and tuple(existing) != identity:
        raise ValueError(f"Conflicting fixture identity: {match_id}")
    db.execute("INSERT INTO fixtures VALUES (?,?,?,?,?) ON CONFLICT(match_id) DO NOTHING", identity)
    old = db.execute("SELECT match_id FROM fixture_aliases WHERE source=? AND source_fixture_key=?",
                     (source, source_key)).fetchone()
    if old and old[0] != match_id:
        raise ValueError(f"Conflicting fixture alias: {source}/{source_key}")
    db.execute("INSERT INTO fixture_aliases VALUES (?,?,?) ON CONFLICT DO NOTHING",
               (source, source_key, match_id))


def add_artifact(db: sqlite3.Connection, source: str, source_url: str, path: Path,
                 observed_at: str) -> tuple[str, str]:
    payload = path.read_bytes()
    if not payload:
        raise ValueError("Empty raw artifact")
    digest = hashlib.sha256(payload).hexdigest()
    artifact_id = stable_id(source, source_url, digest, utc_string(observed_at))
    db.execute("INSERT INTO raw_artifacts VALUES (?,?,?,?,?,?,?) ON CONFLICT(artifact_id) DO NOTHING",
               (artifact_id, source, source_url, digest, utc_string(observed_at), str(path), len(payload)))
    actual = db.execute("SELECT first_observed_at_utc FROM raw_artifacts WHERE artifact_id=?",
                        (artifact_id,)).fetchone()[0]
    return artifact_id, actual


def common(source: str, artifact: str, available: str) -> dict:
    return {"source": source, "artifact_id": artifact, "available_at_utc": utc_string(available),
            "availability_basis": "first_observed"}


def observe(db: sqlite3.Connection, table: str, values: dict) -> str:
    if table not in TABLES:
        raise ValueError(f"Unsupported observation table: {table}")
    values = dict(values)
    columns = {r[1] for r in db.execute(f"PRAGMA table_info({table})")}
    if set(values) - columns:
        raise ValueError(f"Unexpected columns for {table}: {set(values) - columns}")
    if "observation_id" in values:
        raise ValueError("Observation IDs are generated, not supplied")
    for key, value in values.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Non-finite number in {key}")
        if key.endswith("_utc") and value is not None:
            values[key] = utc_string(value)
    artifact = db.execute("SELECT first_observed_at_utc,source FROM raw_artifacts WHERE artifact_id=?",
                          (values["artifact_id"],)).fetchone()
    if not artifact:
        raise ValueError("Unknown raw artifact")
    if artifact[1] != values["source"]:
        raise ValueError("Observation source does not match raw artifact source")
    if values["availability_basis"] == "first_observed" and values["available_at_utc"] != artifact[0]:
        raise ValueError("First-observed timestamps must equal the raw artifact observation time")
    if values["available_at_utc"] > artifact[0]:
        raise ValueError("Availability cannot be later than retrieval")
    # Reject wrong-team joins rather than letting aliases silently contaminate training.
    if values.get("match_id") and values.get("team_id"):
        teams = db.execute("SELECT home_team_id,away_team_id FROM fixtures WHERE match_id=?",
                           (values["match_id"],)).fetchone()
        if not teams or values["team_id"] not in tuple(teams):
            raise ValueError("Team is not a participant in this fixture")
    oid = stable_id(table, values)
    values["observation_id"] = oid
    names = list(values)
    db.execute(f"INSERT INTO {table} ({','.join(names)}) VALUES ({','.join('?' for _ in names)}) "
               "ON CONFLICT(observation_id) DO NOTHING", tuple(values[n] for n in names))
    return oid
