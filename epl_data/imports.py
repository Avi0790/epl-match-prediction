"""Local imports for data the user is authorized to use; no restricted-site crawlers."""
from __future__ import annotations

import csv
from datetime import date
import hashlib
import json
from pathlib import Path
import re

from .http_client import Download
from .store import TABLES, add_artifact, common, ensure_player, observe, utc_now, utc_string

IMPORT_TABLES = TABLES - {"fixture_versions", "match_results", "competition_coverage"}
RESERVED = {"observation_id", "source", "artifact_id", "available_at_utc", "availability_basis"}


def archive_local(path: Path, raw_dir: Path, source: str, source_url: str | None = None) -> Download:
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,40}", source):
        raise ValueError("Use a simple lowercase source identifier, e.g. understat or licensed_export")
    payload = path.read_bytes()
    if not payload:
        raise ValueError("Empty import")
    digest = hashlib.sha256(payload).hexdigest()
    root = raw_dir / source
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"local-{digest}.raw"
    meta_path = root / f"local-{digest}.json"
    if not target.exists():
        target.write_bytes(payload)
    if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
        raise ValueError("Archived local import checksum mismatch")
    observed = utc_now()
    if meta_path.exists():
        observed = json.loads(meta_path.read_text())["observed_at"]
    else:
        meta_path.write_text(json.dumps({"original_filename": path.name, "sha256": digest,
                                        "observed_at": observed}, indent=2) + "\n")
    return Download(target, observed, source_url or f"local-import:{path.name}", True)


def import_clubelo(db, path: Path, raw_dir: Path, team_id: str) -> dict:
    if not db.execute("SELECT 1 FROM teams WHERE team_id=?", (team_id,)).fetchone():
        raise ValueError("Unknown canonical team_id; ingest core matches first")
    downloaded = archive_local(path, raw_dir, "clubelo", f"clubelo-authorized-export:{team_id}")
    reader = csv.DictReader(downloaded.path.read_text(encoding="utf-8-sig").splitlines())
    if not {"Club", "Elo", "From", "To"}.issubset(reader.fieldnames or []):
        raise ValueError("ClubElo CSV requires Club, Elo, From, To columns")
    count, clubs = 0, set()
    with db:
        artifact, first = add_artifact(db, "clubelo", downloaded.url, downloaded.path, downloaded.observed_at)
        for row in reader:
            clubs.add(row["Club"])
            if len(clubs) != 1:
                raise ValueError("Import one club's history at a time")
            start, end = date.fromisoformat(row["From"]), date.fromisoformat(row["To"])
            if end < start:
                raise ValueError("ClubElo validity interval is reversed")
            observe(db, "elo_ratings", {**common("clubelo", artifact, first), "team_id": team_id,
                "rating_date": start.isoformat(), "elo": float(row["Elo"]),
                "source_valid_from": start.isoformat(), "source_valid_to": end.isoformat()})
            count += 1
        if not count:
            raise ValueError("Empty ClubElo CSV")
    return {"source": "clubelo", "rows": count, "availability_basis": "first_observed"}


def import_observations(db, path: Path, raw_dir: Path, source: str, source_url: str) -> dict:
    downloaded = archive_local(path, raw_dir, source, source_url)
    count, seen = 0, set()
    with db:
        artifact, first = add_artifact(db, source, source_url, downloaded.path, downloaded.observed_at)
        for line_no, line in enumerate(downloaded.path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            table, data = row["table"], dict(row["data"])
            if table not in IMPORT_TABLES or RESERVED.intersection(data):
                raise ValueError(f"Line {line_no}: unsupported table or reserved metadata")
            for player in row.get("players", []):
                ensure_player(db, source, str(player["source_player_key"]), player["name"])
            meta = common(source, artifact, first)
            if row.get("published_at_utc"):
                if not row.get("publication_evidence"):
                    raise ValueError("Historical publication timestamp requires evidence in the raw JSONL")
                meta["available_at_utc"] = utc_string(row["published_at_utc"])
                meta["availability_basis"] = "provider_published"
            # Date parsing is explicit so malformed lexicographic dates cannot pass silently.
            for key in ("rating_date", "valuation_date", "effective_date"):
                if data.get(key):
                    data[key] = date.fromisoformat(data[key]).isoformat()
            oid = observe(db, table, {**data, **meta})
            if oid in seen:
                raise ValueError("Duplicate snapshot header in one import; provide distinct publication times")
            seen.add(oid)
            members = row.get("members", [])
            if len({m["player_id"] for m in members}) != len(members):
                raise ValueError("Duplicate player in snapshot")
            if table == "lineup_snapshots":
                if data["kind"] == "confirmed" and sum(int(m["is_starter"]) for m in members) != 11:
                    raise ValueError("A confirmed lineup requires exactly 11 starters")
                for member in members:
                    db.execute("INSERT INTO lineup_players VALUES (?,?,?,?,?) ON CONFLICT DO NOTHING",
                        (oid, member["player_id"], int(member["is_starter"]),
                         member.get("start_probability"), member.get("position")))
            elif table == "availability_snapshots":
                for member in members:
                    expected = member.get("expected_return_date")
                    if expected:
                        date.fromisoformat(expected)
                    db.execute("INSERT INTO unavailable_players VALUES (?,?,?,?,?) ON CONFLICT DO NOTHING",
                        (oid, member["player_id"], member["status"], member.get("reason"), expected))
            elif members:
                raise ValueError("Only lineup/availability snapshots accept members")
            count += 1
    return {"source": source, "observations": count}
