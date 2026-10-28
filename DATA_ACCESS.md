# Sources, coverage and access

## October 27, 2026 update

The supplied xG database now contains 9,120 Understat team-match observations for
4,560 matches from 2014/15–2025/26, imported from the user's JSONL export. The raw
import and its original observation metadata are retained. The 60 bundled 2026/27
matches have result/statistic data but no Understat xG in this snapshot.

`fetch_xg.py` now bundles the user's optional soccerdata acquisition route, with
separate optional dependencies, input checks and download-version metadata. It
uses https://soccerdata.readthedocs.io/en/stable/reference/understat.html . Its
live download was not rerun for this update. Offline model verification used the
included SQLite/JSONL files. Historical publication and revision times remain
unknown; the declared xG delay is an assumption. No xG data was fabricated to fill
the newer season. Use README.md for current commands.

The following section records the earlier Stage 1 access checks and limitations.

Checked on **2026-10-15**. Live access can change; these are the observed results,
not a promise about future service availability.

| Source | What the project actually does | Access / limits |
|---|---|---|
| football-data.co.uk | Downloaded 12 complete E0 seasons, 2014/15–2025/26, plus 60 matches from 2026/27 | No key used; requests cached and spaced |
| API-Football / API-Sports | Adapter and offline parser checks provided; no authenticated requests made | Your direct API-Sports key, plan and league-season coverage must be checked |
| Understat | No scraper included; authorized normalized export route available | `https://understat.com/robots.txt` returned `User-agent: *` and `Disallow: /`; this project does not crawl it |
| FBref | No scraper included; do not rely on former advanced-stat coverage | Sports Reference announced removal of its advanced provider's data in January 2026; authorized suitable exports would need inspection |
| ClubElo | Optional local CSV importer; baseline computes its own Elo | Test of `https://api.clubelo.com/Arsenal` returned HTTP 502; external Elo is not a dependency |
| Transfermarkt | No scraper included | Licensed/authorized data required; current values and injury pages are not historical pre-match snapshots |

## Official references

- Results, odds history and closing-odds coverage:
  https://www.football-data.co.uk/downloadm.php
- Column meanings and collection notes:
  https://football-data.co.uk/notes.txt
- Source warning about Pinnacle reliability from 2025-07-23:
  https://football-data.co.uk/data
- API-Football reference:
  https://www.api-football.com/documentation-v3
- API authentication, coverage flags, fixture endpoints and lineup timing:
  https://www.api-football.com/news/post/how-to-get-started-with-api-football-the-complete-beginners-guide
- API injury endpoint announcement (2021); do not infer complete ten-year coverage:
  https://www.api-football.com/news/post/new-endpoint-injuries
- Sports Reference's advanced-data removal announcement:
  https://www.sports-reference.com/blog/2026/01/fbref-stathead-data-update/
- Sports Reference data-use guidance:
  https://www.sports-reference.com/data_use.html
- Understat's published crawling directive:
  https://understat.com/robots.txt
- ClubElo:
  https://clubelo.com/

Broad closing-odds columns begin in 2019/20; Pinnacle closing H/D/A goes back
earlier. The included 2014/15–2018/19 files have no kickoff clock times. Those
facts are also verified directly by the normalized coverage report.

API coverage flags indicate intended collection, not a guarantee for every match.
An injury/lineup API returning a past match today does not establish when that
information was originally known. Player match statistics are appropriate for
lagged histories; final season totals must not be reused at earlier dates.

## Information needed before enrichment

**None of this access is needed for the working baseline.** No API key is currently
available. Keep enrichment deferred and do not purchase a plan for this stage.

1. Do you have an API-Football direct API-Sports account, and which plan? Set
   `API_FOOTBALL_KEY` locally. Never send the key in a chat message or source file.
2. Do you have authorized historical Understat/FBref xG and player match exports,
   or another permitted data feed with equivalent measurements?
3. External ClubElo is optional; local Elo already works from the retained results.
4. Historical market values, captain/GK/manager records and publication archives
   require their own evidence. We can proceed with a clearly labeled reduced
   feature set while collecting missing snapshots prospectively.

No subscription was purchased and no external account was created. No protected
site, rate limit, login wall, or bot restriction was bypassed. This project uses the
official downloadable results files and leaves unavailable enrichment fields NULL.
