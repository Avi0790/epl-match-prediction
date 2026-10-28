"""Direct API-Sports v3 adapter. Live use requires API_FOOTBALL_KEY and coverage."""
from __future__ import annotations

from collections import defaultdict
import json
import os
import re
import urllib.parse

from .football_data import NAMES, number, result_label
from .http_client import AccessError
from .store import (add_artifact, common, ensure_fixture, ensure_player, ensure_team,
                    observe, utc_string)

SOURCE = "api_football"
BASE_URL = "https://v3.football.api-sports.io"
EXACT_NAMES = {**NAMES, "Brighton & Hove Albion": "Brighton and Hove Albion",
               "Nottingham Forest": "Nottingham Forest", "Ipswich": "Ipswich Town",
               "Luton": "Luton Town", "Wolverhampton Wanderers": "Wolverhampton Wanderers"}


class ApiFootball:
    def __init__(self, db, client, refresh=False):
        self.db, self.client, self.refresh = db, client, refresh
        self.client.interval = max(self.client.interval, 6.1)
        self.key = os.environ.get("API_FOOTBALL_KEY", "").strip()
        if not self.key:
            raise AccessError("Set API_FOOTBALL_KEY locally; an API-Sports key is required. Do not paste it into chat.")

    def pages(self, endpoint, params):
        page, total = 1, 1
        while page <= total:
            query = {**params}
            if page > 1:
                query["page"] = page
            url = BASE_URL + "/" + endpoint + "?" + urllib.parse.urlencode(sorted(query.items()))
            downloaded = self.client.get(SOURCE, url, refresh=self.refresh,
                                         headers={"x-apisports-key": self.key})
            payload = json.loads(downloaded.path.read_text())
            if payload.get("errors"):
                # Do not echo provider errors: they may include account/request details.
                raise AccessError("API-Football returned an error; check plan, parameters, quota and dashboard")
            if not isinstance(payload.get("response"), list):
                raise ValueError("Unexpected API-Football response envelope")
            paging = payload.get("paging", {})
            total = int(paging.get("total", 1))
            if int(paging.get("current", page)) != page or total < page or total > 10000:
                raise ValueError("Invalid API pagination metadata")
            if int(payload.get("results", -1)) != len(payload["response"]):
                raise ValueError("API results count disagrees with response length")
            with self.db:
                artifact, first = add_artifact(self.db, SOURCE, url, downloaded.path, downloaded.observed_at)
            yield payload["response"], common(SOURCE, artifact, first)
            page += 1

    def coverage(self, *, league=None, team=None, season=None):
        params = {"season": season}
        if league is not None:
            params["id"] = league
        elif team is not None:
            params["team"] = team
        else:
            raise ValueError("Coverage requires a league or team")
        if season is None:
            raise ValueError("Coverage requires a season")
        result = {}
        for records, base in self.pages("leagues", params):
            with self.db:
                for record in records:
                    league_id = int(record["league"]["id"])
                    for entry in record.get("seasons", []):
                        if int(entry["year"]) != season:
                            continue
                        cov = entry.get("coverage", {})
                        result[(league_id, season)] = cov
                        observe(self.db, "competition_coverage", {**base,
                            "competition_id": "ENG-PL" if league_id == 39 else f"api_football:{league_id}",
                            "season_start": season, "coverage_json": json.dumps(cov, sort_keys=True)})
        if not result:
            raise AccessError("No coverage returned for this league/team and season; verify your subscription")
        return result


def resolve_team(db, record, team_map, require_mapping=False):
    key, name = str(record["id"]), record["name"]
    mapped = team_map.get(key)
    if mapped:
        if not db.execute("SELECT 1 FROM teams WHERE team_id=?", (mapped,)).fetchone():
            raise ValueError(f"Mapped team_id does not exist: {mapped}")
        return ensure_team(db, SOURCE, key, name, mapped)
    old = db.execute("SELECT team_id FROM team_aliases WHERE source=? AND source_team_key=?", (SOURCE, key)).fetchone()
    if old:
        return old[0]
    exact = db.execute("SELECT team_id FROM teams WHERE name=?", (EXACT_NAMES.get(name, name),)).fetchall()
    if len(exact) == 1:
        return ensure_team(db, SOURCE, key, name, exact[0][0])
    if require_mapping:
        raise ValueError(f"Missing explicit API team mapping: {key} ({name}). Ingest EPL CSVs first or edit config/api_team_map.json")
    return ensure_team(db, SOURCE, key, name)


def normalize_fixtures(db, records, base, team_map):
    output = []
    for rec in records:
        fixture, league = rec["fixture"], rec["league"]
        is_epl = int(league["id"]) == 39
        home = resolve_team(db, rec["teams"]["home"], team_map, is_epl)
        away = resolve_team(db, rec["teams"]["away"], team_map, is_epl)
        season = int(league["season"])
        comp = "ENG-PL" if is_epl else f"api_football:{league['id']}"
        if is_epl:
            rows = db.execute("SELECT match_id FROM fixtures WHERE competition_id=? AND season_start=? AND home_team_id=? AND away_team_id=?",
                              (comp, season, home, away)).fetchall()
            if len(rows) > 1:
                raise ValueError("Ambiguous EPL fixture mapping")
            match_id = rows[0][0] if rows else f"epl:{season}:{home.split(':',1)[1]}:{away.split(':',1)[1]}"
        else:
            match_id = f"api_football:{fixture['id']}"
        ensure_fixture(db, match_id, comp, season, home, away, SOURCE, str(fixture["id"]))
        kickoff = utc_string(fixture["date"]) if fixture.get("date") else None
        if kickoff is None:
            raise ValueError("API fixture lacks a scheduled date; cannot safely normalize it")
        status = fixture["status"]["short"]
        round_match = re.fullmatch(r"Regular Season - (\d+)", league.get("round", ""))
        observe(db, "fixture_versions", {**base, "match_id": match_id, "match_date": kickoff[:10],
            "kickoff_utc": kickoff, "kickoff_precision": "exact", "source_timezone": "UTC",
            "status": status, "referee": fixture.get("referee"),
            "matchweek": int(round_match[1]) if is_epl and round_match else None})
        # Cup fixtures ending AET/PEN contribute schedule history, not 90-minute goal labels here.
        if status == "FT":
            hg, ag = number(rec["goals"]["home"], integer=True), number(rec["goals"]["away"], integer=True)
            if hg is None or ag is None:
                raise ValueError("Finished API fixture missing final goals")
            observe(db, "match_results", {**base, "match_id": match_id, "home_goals": hg,
                "away_goals": ag, "result": result_label(hg, ag), "score_period": "regulation"})
        output.append((match_id, home, away, status))
    return output


def normalize_lineups(db, records, base, match_id, team_map):
    count = 0
    for rec in records:
        team = resolve_team(db, rec["team"], team_map)
        starters = rec.get("startXI", [])
        if len(starters) != 11:
            raise ValueError("Incomplete confirmed lineup; expected 11 starters")
        all_members = starters + rec.get("substitutes", [])
        if len({m["player"]["id"] for m in all_members}) != len(all_members):
            raise ValueError("Duplicate player in confirmed lineup")
        oid = observe(db, "lineup_snapshots", {**base, "match_id": match_id,
            "team_id": team, "kind": "confirmed", "formation": rec.get("formation")})
        for is_starter, members in ((1, starters), (0, rec.get("substitutes", []))):
            for member in members:
                player = member["player"]
                pid = ensure_player(db, SOURCE, str(player["id"]), player["name"])
                db.execute("INSERT INTO lineup_players VALUES (?,?,?,?,?) ON CONFLICT DO NOTHING",
                           (oid, pid, is_starter, float(is_starter), player.get("pos")))
        count += 1
    return count


def normalize_injuries(db, records, base, match_id, team_ids, team_map):
    by_team = defaultdict(list)
    for rec in records:
        team = resolve_team(db, rec["team"], team_map)
        if team not in team_ids:
            raise ValueError("Injury team does not match requested fixture")
        by_team[team].append(rec["player"])
    for team in team_ids:
        oid = observe(db, "availability_snapshots", {**base, "match_id": match_id, "team_id": team,
                      "coverage_status": "reported" if by_team[team] else "unknown"})
        for player in by_team[team]:
            pid = ensure_player(db, SOURCE, str(player["id"]), player["name"])
            reason, kind = player.get("reason") or "", player.get("type") or ""
            if "suspend" in reason.casefold() or "suspension" in reason.casefold():
                state = "suspended"
            elif kind.casefold() == "questionable":
                state = "doubtful"
            elif kind.casefold() == "missing fixture":
                state = "out"
            else:
                state = "unknown"
            db.execute("INSERT INTO unavailable_players VALUES (?,?,?,?,?) ON CONFLICT DO NOTHING",
                       (oid, pid, state, f"{kind}: {reason}", None))
    return len(records)


def normalize_players(db, records, base, match_id, team_map):
    count = 0
    for rec in records:
        team = resolve_team(db, rec["team"], team_map)
        for entry in rec.get("players", []):
            player = entry["player"]
            pid = ensure_player(db, SOURCE, str(player["id"]), player["name"])
            stats = entry.get("statistics", [])
            if len(stats) != 1:
                raise ValueError("Expected one match-stat record per player")
            st = stats[0]
            games, goals = st.get("games", {}), st.get("goals", {})
            observe(db, "player_match_stats", {**base, "match_id": match_id, "team_id": team,
                "player_id": pid, "minutes": number(games.get("minutes")),
                "goals": number(goals.get("total"), integer=True),
                "assists": number(goals.get("assists"), integer=True),
                "performance_rating": number(games.get("rating")), "position": games.get("position"),
                "definition": "API-Football per-fixture player statistics; xG/xA/pressures not inferred"})
            count += 1
    return count


def ingest_api(db, client, args):
    api = ApiFootball(db, client, args.refresh)
    team_map = json.loads(args.team_map.read_text()) if args.team_map.exists() else {}
    if args.resource == "coverage":
        cov = api.coverage(league=args.league, team=args.team, season=args.season)
        return {"coverage": {f"{k[0]}:{k[1]}": v for k, v in cov.items()}}
    if args.resource == "fixtures":
        if args.season is None or (args.league is None) == (args.team is None):
            raise ValueError("Provide --season and exactly one of --league or --team")
        api.coverage(league=args.league, team=args.team, season=args.season)
        params = {"season": args.season, "timezone": "UTC"}
        params["league" if args.league is not None else "team"] = args.league if args.league is not None else args.team
        count = 0
        for records, base in api.pages("fixtures", params):
            with db:
                count += len(normalize_fixtures(db, records, base, team_map))
        return {"resource": "fixtures", "rows": count,
                "scope": "league" if args.league is not None else "team across available competitions"}
    if args.fixture is None:
        raise ValueError("This resource requires --fixture")
    fixture_records = []
    normalized = []
    for records, base in api.pages("fixtures", {"id": args.fixture, "timezone": "UTC"}):
        fixture_records.extend(records)
        with db:
            normalized.extend(normalize_fixtures(db, records, base, team_map))
    if len(fixture_records) != 1 or len(normalized) != 1:
        raise ValueError("Requested fixture was not returned uniquely")
    match_id, home, away, status = normalized[0]
    league = fixture_records[0]["league"]
    cov = api.coverage(league=int(league["id"]), season=int(league["season"]))[(int(league["id"]), int(league["season"]))]
    field = {"lineups": "lineups", "players": "statistics_players"}.get(args.resource)
    supported = cov.get("injuries", False) if args.resource == "injuries" else cov.get("fixtures", {}).get(field, False)
    if not supported:
        raise AccessError(f"Provider reports no {args.resource} coverage for this league/season")
    if args.resource == "players" and status != "FT":
        raise ValueError("Only completed, regulation-time player performances are imported in stage 1")
    endpoint = "injuries" if args.resource == "injuries" else "fixtures/" + args.resource
    # These fixture endpoints are normally unpaginated. Reject multiple pages for complete snapshots.
    pages = list(api.pages(endpoint, {"fixture": args.fixture}))
    if len(pages) != 1:
        raise ValueError("Unexpected pagination on fixture snapshot; refusing to create partial snapshots")
    records, base = pages[0]
    with db:
        if args.resource == "lineups":
            count = normalize_lineups(db, records, base, match_id, team_map)
        elif args.resource == "injuries":
            count = normalize_injuries(db, records, base, match_id, (home, away), team_map)
        else:
            count = normalize_players(db, records, base, match_id, team_map)
    return {"resource": args.resource, "rows": count, "availability_basis": "first_observed",
            "note": "Historical lineups/injuries fetched today are NOT historical pre-match snapshots"}
