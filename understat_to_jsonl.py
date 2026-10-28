#!/usr/bin/env python3
"""
Convert data/understat_xg.csv (from fetch_xg.py) into the JSONL import format that
`data_ingest.py import-observations` expects for the `advanced_team_stats` table.

Run from inside the epl-predictor folder:
    python understat_to_jsonl.py --csv data/understat_xg.csv --db data/epl.sqlite \
        --output data/understat_xg.jsonl

Design rules (they follow SCHEMA.md in your project):
  * Team IDs come from the team_aliases table (source='football_data'), and match IDs
    from the fixtures table, so every row joins to an existing fixture.
  * Any team or match that cannot be matched makes the script STOP and list the
    problems. Nothing is guessed and nothing is silently dropped.
  * No publication time is claimed. Without `published_at_utc` and evidence, the
    importer records the real first-observation time (today) as `available_at_utc`.
  * For the retrospective research mode, each row also carries
    `assumed_available_at_utc`, copied from the SAME match's stored result. That is the
    explicit assumption "xG became available when the result did", the same kind of
    declared delay your results and shot statistics already use. It is a labeled
    assumption, not an observed publication time.
  * Understat's goals are compared with your stored results as a join sanity check.
"""
import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

DEFINITION = "Understat match xG (via soccerdata read_schedule, home_xg/away_xg)"

# Understat name -> football-data.co.uk name (the name stored in team_aliases).
# Any name not listed here is assumed to be identical in both sources.
UNDERSTAT_TO_FOOTBALL_DATA = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Queens Park Rangers": "QPR",
    "West Bromwich Albion": "West Brom",
    "Wolverhampton Wanderers": "Wolves",
}


def season_start(value):
    """Accept provider labels 1415, 2014, 2014/15 and 2014-2015 explicitly."""
    label = str(value).strip()
    full = re.fullmatch(r"(\d{4})[-/](\d{2}|\d{4})", label)
    if full:
        start, end = int(full[1]), int(full[2])
        if end != (start + 1 if len(full[2]) == 4 else (start + 1) % 100):
            raise ValueError(f"Invalid season label: {label}")
        return start
    if re.fullmatch(r"\d{4}", label):
        number = int(label)
        if 2000 <= number <= 2099:
            return number
        start, end = int(label[:2]), int(label[2:])
        if end == (start + 1) % 100:
            return 2000 + start
    raise ValueError(f"Unsupported season label: {label}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=Path("data/understat_xg.csv"))
    ap.add_argument("--db", type=Path, default=Path("data/epl.sqlite"))
    ap.add_argument("--output", type=Path, default=Path("data/understat_xg.jsonl"))
    args = ap.parse_args()

    xg = pd.read_csv(args.csv)
    xg["season_start"] = xg["season"].map(season_start)

    db = sqlite3.connect(args.db)
    db.row_factory = sqlite3.Row
    alias = {r["source_team_key"]: r["team_id"] for r in db.execute(
        "SELECT source_team_key, team_id FROM team_aliases WHERE source = 'football_data'")}
    fixtures = {(r["season_start"], r["home_team_id"], r["away_team_id"]): r["match_id"]
                for r in db.execute("SELECT match_id, season_start, home_team_id, away_team_id FROM fixtures")}

    def team_id(name):
        return alias.get(UNDERSTAT_TO_FOOTBALL_DATA.get(name, name))

    problems = []
    unknown = sorted({n for n in set(xg.home_team) | set(xg.away_team) if team_id(n) is None})
    if unknown:
        problems.append(f"Team names with no match in team_aliases: {unknown}\n"
                        f"  Known football_data names: {sorted(alias)}")

    rows, seen = [], set()
    for r in xg.itertuples():
        h, a = team_id(r.home_team), team_id(r.away_team)
        if h is None or a is None:
            continue
        match_id = fixtures.get((r.season_start, h, a))
        if match_id is None:
            problems.append(f"No fixture for {r.season_start} {r.home_team} v {r.away_team}")
            continue
        if match_id in seen:
            problems.append(f"Duplicate Understat row for {match_id}")
            continue
        seen.add(match_id)
        rows.append((match_id, h, a, float(r.home_xg), float(r.away_xg), int(r.home_goals), int(r.away_goals)))

    # Sanity check: scores should agree with the results already in the database.
    try:
        stored = {r["match_id"]: (r["home_goals"], r["away_goals"]) for r in db.execute(
            "SELECT match_id, home_goals, away_goals FROM latest_match_results "
            "WHERE source = 'football_data'")}
        bad = [m for m, _, _, _, _, hg, ag in rows if m in stored and stored[m] != (hg, ag)]
        if bad:
            problems.append(f"{len(bad)} matches where Understat goals differ from stored results, e.g. {bad[:5]}")
        print(f"Score cross-check: {len(rows) - len(bad)} of {len(rows)} matches agree with stored results")
    except sqlite3.Error as exc:
        print(f"WARNING: score cross-check skipped ({exc})")

    assumed = {r["match_id"]: r["assumed_available_at_utc"] for r in db.execute(
        "SELECT match_id, assumed_available_at_utc FROM latest_match_results WHERE source = 'football_data'")}
    no_assumption = [m for m, *_ in rows if not assumed.get(m)]
    if no_assumption:
        problems.append(f"{len(no_assumption)} matches have no stored result-availability assumption, "
                        f"e.g. {no_assumption[:3]}")

    missing = set(fixtures.values()) - seen
    if missing:
        print(f"NOTE: {len(missing)} fixtures in your database have no xG row "
              f"(e.g. {sorted(missing)[:3]}). They will stay missing, not zero.")

    if problems:
        print("\nSTOPPED. Nothing written:\n- " + "\n- ".join(problems))
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        for match_id, h, a, hxg, axg, _, _ in rows:
            for team, xf, xa in ((h, hxg, axg), (a, axg, hxg)):
                out.write(json.dumps({"table": "advanced_team_stats", "data": {
                    "match_id": match_id, "team_id": team,
                    "xg_for": xf, "xg_against": xa, "definition": DEFINITION,
                    "assumed_available_at_utc": assumed[match_id]}}) + "\n")
    print(f"Wrote {2 * len(rows)} observations ({len(rows)} matches) to {args.output}")


if __name__ == "__main__":
    main()
