#!/usr/bin/env python3
"""Command-line entry point for ingestion only. Run with --help."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys

from epl_data.audit import audit
from epl_data.football_data import collect, ingest_csv
from epl_data.http_client import AccessError, Download, HttpClient
from epl_data.store import connect, utc_now


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=Path("data/epl.sqlite"))
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--interval", type=float, default=2.0, help="Seconds between HTTP requests")
    p.add_argument("--request-budget", type=int, default=100, help="Hard per-run request cap, including retries")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="Create an empty database with the complete ingestion schema")
    sub.add_parser("schema", help="Print the SQL schema")
    a = sub.add_parser("football-data", help="Download Premier League E0 CSV files")
    a.add_argument("--start-season", type=int, default=2014)
    a.add_argument("--end-season", type=int, default=2025, help="Inclusive start year; 2025 means 2025/26")
    a.add_argument("--refresh", action="store_true")
    a.add_argument("--require-complete", action="store_true", help="Require 380 completed results / 20 teams per season")
    a.add_argument("--source-timezone", default="Europe/London")
    a = sub.add_parser("import-football-data", help="Import a local, unmodified E0 CSV")
    a.add_argument("--file", type=Path, required=True)
    a.add_argument("--season", type=int, required=True)
    a.add_argument("--source-timezone", default="Europe/London")
    a = sub.add_parser("audit", help="Write a coverage and data-quality report")
    a.add_argument("--output", type=Path, default=Path("reports/ingestion_audit.json"))
    a = sub.add_parser("clubelo-import", help="Import an authorized ClubElo team-history CSV")
    a.add_argument("--file", type=Path, required=True)
    a.add_argument("--team-id", required=True)
    a = sub.add_parser("import-observations", help="Import authorized, normalized JSONL observations")
    a.add_argument("--file", type=Path, required=True)
    a.add_argument("--source", required=True)
    a.add_argument("--source-url", required=True)
    a = sub.add_parser("api-football", help="API-Sports adapter; key read from API_FOOTBALL_KEY")
    a.add_argument("--resource", choices=("coverage", "fixtures", "injuries", "lineups", "players"), required=True)
    a.add_argument("--league", type=int)
    a.add_argument("--season", type=int)
    a.add_argument("--team", type=int)
    a.add_argument("--fixture", type=int)
    a.add_argument("--refresh", action="store_true")
    a.add_argument("--team-map", type=Path, default=Path("config/api_team_map.json"))
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.command == "schema":
        print(Path(__file__).parent.joinpath("epl_data/schema.sql").read_text())
        return 0
    db = connect(args.db)
    try:
        client = HttpClient(args.raw_dir, interval=args.interval, request_budget=args.request_budget)
        if args.command == "init":
            print(f"Initialized {args.db}")
        elif args.command == "football-data":
            if args.end_season < args.start_season:
                raise ValueError("end-season must be >= start-season")
            for item in collect(db, client, range(args.start_season, args.end_season + 1), refresh=args.refresh,
                                timezone=args.source_timezone, require_complete=args.require_complete):
                print(json.dumps(item), flush=True)
        elif args.command == "import-football-data":
            from epl_data.imports import archive_local
            local = archive_local(args.file, args.raw_dir, "football_data")
            print(json.dumps(ingest_csv(db, local, args.season, args.source_timezone)))
        elif args.command == "audit":
            report = audit(db)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
        elif args.command == "clubelo-import":
            from epl_data.imports import import_clubelo
            print(json.dumps(import_clubelo(db, args.file, args.raw_dir, args.team_id)))
        elif args.command == "import-observations":
            from epl_data.imports import import_observations
            print(json.dumps(import_observations(db, args.file, args.raw_dir, args.source, args.source_url)))
        elif args.command == "api-football":
            from epl_data.api_football import ingest_api
            print(json.dumps(ingest_api(db, client, args)))
        return 0
    except (AccessError, ValueError, OSError, sqlite3.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
