"""football-data.co.uk E0 CSV adapter; no web-page scraping."""
from __future__ import annotations

import csv
from datetime import datetime, time, timedelta
import io
import json
import math
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

from .http_client import Download, HttpClient
from .store import UTC, add_artifact, common, ensure_fixture, ensure_team, observe, stable_id

SOURCE = "football_data"
NAMES = {
    "Man City": "Manchester City", "Man United": "Manchester United",
    "Wolves": "Wolverhampton Wanderers", "Nott'm Forest": "Nottingham Forest",
    "West Brom": "West Bromwich Albion", "Sheffield United": "Sheffield United",
    "Tottenham": "Tottenham Hotspur", "Brighton": "Brighton and Hove Albion",
    "Newcastle": "Newcastle United", "West Ham": "West Ham United",
    "Leicester": "Leicester City", "Leeds": "Leeds United",
}
BOOKS = {
    "B365": "bet365", "BW": "bwin", "IW": "interwetten", "PS": "pinnacle",
    "P": "pinnacle", "WH": "william_hill", "VC": "betvictor", "BV": "betvictor",
    "LB": "ladbrokes", "GB": "gamebookers", "BS": "blue_square", "SJ": "stan_james",
    "BF": "betfair", "BFE": "betfair_exchange", "BFD": "betfred", "BMGM": "betmgm",
    "1XB": "1xbet", "CL": "coral", "PP": "paddy_power", "SK": "skybet",
    "SO": "sporting_odds", "SB": "sportingbet", "SY": "stanleybet",
    "Avg": "market_average", "Max": "market_maximum",
    "BbAv": "betbrain_average", "BbMx": "betbrain_maximum",
}


def number(value: object, *, integer: bool = False):
    if value is None or str(value).strip() in ("", "NA", "N/A", "null"):
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0 or (integer and not result.is_integer()):
        raise ValueError(f"Invalid nonnegative {'integer' if integer else 'number'}: {value!r}")
    return int(result) if integer else result


def parse_date(value: str):
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Invalid day-first match date: {value!r}")


def match_times(row: dict, source_timezone: str) -> tuple[str, str | None, str, str]:
    day = parse_date(row["Date"])
    zone = ZoneInfo(source_timezone)
    if (row.get("Time") or "").strip():
        clock = datetime.strptime(row["Time"].strip(), "%H:%M").time()
        kickoff = datetime.combine(day, clock, zone).astimezone(UTC)
        iso = kickoff.isoformat().replace("+00:00", "Z")
        assumed = kickoff + timedelta(hours=24)
        precision = "minute_assumed_zone"
    else:
        iso, precision = None, "date_only"
        # No invented kickoff. The retrospective option uses a conservative day-level lag.
        assumed = datetime.combine(day + timedelta(days=2), time.min, zone).astimezone(UTC)
    return day.isoformat(), iso, precision, assumed.isoformat().replace("+00:00", "Z")


def season_url(season: int) -> str:
    if not 1993 <= season <= 2098:
        raise ValueError("Season start must be between 1993 and 2098")
    code = f"{season % 100:02d}{(season + 1) % 100:02d}"
    return f"https://www.football-data.co.uk/mmz4281/{code}/E0.csv"


def result_label(home: int, away: int) -> str:
    return "H" if home > away else "A" if home < away else "D"


def _odds(db, row, match_id, day, base) -> int:
    count = 0
    emitted = set()
    for prefix, book in BOOKS.items():
        for suffix, phase in (("", "pre_closing"), ("C", "closing")):
            for market, outcomes in (("1x2", ("H", "D", "A")), ("ou_2_5", (">2.5", "<2.5"))):
                cols = [prefix + suffix + outcome for outcome in outcomes]
                if not all(col in row for col in cols):
                    continue
                values = [number(row[col]) for col in cols]
                if any(value is None for value in values):
                    continue  # Partial quotes are not valid benchmark distributions.
                if any(value <= 1 for value in values):
                    detail = {"columns": cols, "values": values,
                              "action": "entire invalid odds quote excluded"}
                    db.execute("INSERT INTO data_issues VALUES (?,?,?,?,?) ON CONFLICT DO NOTHING",
                        (stable_id(base["artifact_id"], match_id, cols, "invalid_odds"),
                         base["artifact_id"], match_id, "invalid_decimal_odds", json.dumps(detail, sort_keys=True)))
                    continue
                identity = (book, phase, market)
                if identity in emitted:
                    continue
                emitted.add(identity)
                fields = (dict(zip(("odds_home", "odds_draw", "odds_away"), values))
                          if market == "1x2" else dict(zip(("odds_over", "odds_under"), values)))
                flag = "pinnacle_unreliable_since_2025_07_23" if book == "pinnacle" and day >= "2025-07-23" else None
                observe(db, "odds_snapshots", {**base, "match_id": match_id, "bookmaker": book,
                    "market": market, "phase": phase, "quote_time_utc": None,
                    "quality_flag": flag, **fields})
                count += 1
    return count


def ingest_csv(db: sqlite3.Connection, download: Download, season: int,
               timezone: str = "Europe/London", require_complete: bool = False) -> dict:
    payload = download.path.read_bytes()
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = payload.decode("cp1252")
    reader = csv.DictReader(io.StringIO(text))
    required = {"Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
    if not reader.fieldnames or not required.issubset(reader.fieldnames):
        raise ValueError("Expected a football-data E0 CSV; required headers are missing")
    seen, teams = set(), set()
    counts = {"season_start": season, "fixtures": 0, "results": 0, "team_stat_rows": 0,
              "odds_rows": 0, "missing_kickoff_time": 0, "quarantined_stat_pairs": 0,
              "cached": download.cached}
    with db:
        artifact, first_seen = add_artifact(db, SOURCE, download.url, download.path, download.observed_at)
        base = common(SOURCE, artifact, first_seen)
        for line, row in enumerate(reader, start=2):
            if all(not str(value or "").strip() for value in row.values()):
                continue
            try:
                if row.get("Div") != "E0":
                    raise ValueError("Expected division E0")
                if not row["HomeTeam"].strip() or not row["AwayTeam"].strip():
                    raise ValueError("Empty team name")
                home = ensure_team(db, SOURCE, row["HomeTeam"], NAMES.get(row["HomeTeam"], row["HomeTeam"]))
                away = ensure_team(db, SOURCE, row["AwayTeam"], NAMES.get(row["AwayTeam"], row["AwayTeam"]))
                pair = (home, away)
                if pair in seen:
                    raise ValueError("Duplicate home/away pairing within this league season")
                seen.add(pair)
                teams.update(pair)
                day, kickoff, precision, assumed = match_times(row, timezone)
                if not f"{season}-07-01" <= day <= f"{season + 1}-08-31":
                    raise ValueError("Date outside supplied season (including the 2020 extension)")
                hg, ag = number(row["FTHG"], integer=True), number(row["FTAG"], integer=True)
                if (hg is None) != (ag is None):
                    raise ValueError("Partial final score")
                result = result_label(hg, ag) if hg is not None else None
                if (row["FTR"] or "").strip() and row["FTR"].strip() != result:
                    raise ValueError("FTR disagrees with the final goals")
                match_id = f"epl:{season}:{home.removeprefix(SOURCE + ':')}:{away.removeprefix(SOURCE + ':')}"
                ensure_fixture(db, match_id, "ENG-PL", season, home, away, SOURCE,
                               f"{season}:{row['HomeTeam']}:{row['AwayTeam']}")
                observe(db, "fixture_versions", {**base, "match_id": match_id,
                    "match_date": day, "kickoff_utc": kickoff, "kickoff_precision": precision,
                    "source_timezone": timezone if kickoff else None,
                    "status": "FT" if result else "scheduled", "referee": row.get("Referee") or None})
                counts["fixtures"] += 1
                counts["missing_kickoff_time"] += kickoff is None
                if result:
                    observe(db, "match_results", {**base, "match_id": match_id, "home_goals": hg,
                        "away_goals": ag, "result": result, "score_period": "regulation",
                        "assumed_available_at_utc": assumed})
                    counts["results"] += 1
                    for team, prefix in ((home, "H"), (away, "A")):
                        stats = {name: number(row.get(prefix + col), integer=True) for name, col in
                                 (("shots", "S"), ("shots_on_target", "ST"), ("corners", "C"),
                                  ("fouls", "F"), ("yellow_cards", "Y"), ("red_cards", "R"))}
                        if (stats["shots"] is not None and stats["shots_on_target"] is not None
                                and stats["shots_on_target"] > stats["shots"]):
                            # Preserve original bytes and record both values. We cannot know which is wrong.
                            detail = {"team_id": team, "shots": stats["shots"],
                                      "shots_on_target": stats["shots_on_target"], "csv_line": line,
                                      "action": "both shot fields set to NULL; all other fields retained"}
                            db.execute("INSERT INTO data_issues VALUES (?,?,?,?,?) ON CONFLICT DO NOTHING",
                                (stable_id(artifact, match_id, team, "sot_gt_shots"), artifact, match_id,
                                 "shots_on_target_exceed_shots", json.dumps(detail, sort_keys=True)))
                            stats["shots"] = stats["shots_on_target"] = None
                            counts["quarantined_stat_pairs"] += 1
                        observe(db, "team_match_stats", {**base, "match_id": match_id, "team_id": team,
                            "assumed_available_at_utc": assumed, **stats})
                        counts["team_stat_rows"] += 1
                counts["odds_rows"] += _odds(db, row, match_id, day, base)
            except (ValueError, sqlite3.IntegrityError) as exc:
                raise ValueError(f"Season {season}, CSV line {line}: {exc}") from exc
        if not counts["fixtures"]:
            raise ValueError("Empty season CSV")
        if require_complete and (counts["results"] != 380 or len(teams) != 20):
            raise ValueError(f"Expected 380 results / 20 teams, got {counts['results']} / {len(teams)}")
    counts["teams"] = len(teams)
    return counts


def collect(db, client: HttpClient, seasons: range, *, refresh: bool = False,
            timezone: str = "Europe/London", require_complete: bool = False):
    for season in seasons:
        downloaded = client.get(SOURCE, season_url(season), refresh=refresh)
        yield ingest_csv(db, downloaded, season, timezone, require_complete)
