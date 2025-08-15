from __future__ import annotations
import io, time
from typing import List
import pandas as pd
import requests
from requests import HTTPError

from ..utils.cache import RAW, SAMPLE
from ..utils.io import write_parquet
from ..utils.logging import get_logger
from ..config import settings

log = get_logger(__name__)

def bootstrap_raw_from_sample() -> None:
    fx = SAMPLE / "fixtures_sample.csv"
    if not fx.exists():
        log.info("No sample files present at %s; skipping demo bootstrap.", fx.parent)
        return
    fixtures = pd.read_csv(SAMPLE / "fixtures_sample.csv", parse_dates=["date"])
    players  = pd.read_csv(SAMPLE / "players_sample.csv")
    events   = pd.read_csv(SAMPLE / "events_sample.csv")
    write_parquet(fixtures, RAW / "sample" / "fixtures_2024-25.parquet")
    write_parquet(players,  RAW / "sample" / "players_2024-25.parquet")
    write_parquet(events,   RAW / "sample" / "events_2024-25.parquet")
    log.info("Demo raw parquet written from bundled samples.")

FD_COMP = {"EPL":"PL","LaLiga":"PD","Bundesliga":"BL1","SerieA":"SA","Ligue1":"FL1"}
FD_BASE = "https://api.football-data.org/v4"
FD_UK_CODE = {"EPL":"E0", "LaLiga":"SP1", "Bundesliga":"D1", "SerieA":"I1", "Ligue1":"F1"}
FD_UK_BASE = "https://www.football-data.co.uk/mmz4281"

def _fd_headers() -> dict:
    tok = settings.FOOTBALL_DATA_TOKEN
    return {"X-Auth-Token": tok} if tok else {}

def _fd_uk_csv(lg: str, season: int) -> pd.DataFrame:
    code = FD_UK_CODE[lg]
    yy = f"{season%100:02d}{(season+1)%100:02d}"
    url = f"{FD_UK_BASE}/{yy}/{code}.csv"
    log.info("Fallback CSV for %s %s: %s", lg, season, url)
    r = requests.get(url, timeout=45)
    if r.status_code != 200 or not r.content:
        log.warning("CSV fallback not available (%s %s): HTTP %s", lg, season, r.status_code)
        return pd.DataFrame()
    df = pd.read_csv(io.BytesIO(r.content))
    def pick(*c):
        for k in c:
            if k in df.columns: return k
        return None
    h = pick("HomeTeam","Home","HT"); a = pick("AwayTeam","Away","AT")
    fthg = pick("FTHG","HG","HomeGoals"); ftag = pick("FTAG","AG","AwayGoals")
    date_col = pick("Date","MatchDate")
    if not all([h,a,date_col]):
        log.warning("CSV missing expected cols for %s %s", lg, season); return pd.DataFrame()
    out = pd.DataFrame({
        "season": season,
        "date": pd.to_datetime(df[date_col], errors="coerce", dayfirst=True),
        "home_team": df[h],
        "away_team": df[a],
        "home_goals": df[fthg] if fthg in df.columns else None,
        "away_goals": df[ftag] if ftag in df.columns else None,
    }).dropna(subset=["date","home_team","away_team"])
    out["match_id"] = range(1, len(out)+1)
    return out

def _fd_matches(code: str, season: int, lg_name: str) -> pd.DataFrame:
    url = f"{FD_BASE}/competitions/{code}/matches"
    try:
        r = requests.get(url, headers=_fd_headers(), params={"season": season}, timeout=45)
        r.raise_for_status()
        js = r.json()
        rows = []
        for m in js.get("matches", []):
            rows.append({
                "season": season,
                "utcDate": m.get("utcDate"),
                "status": m.get("status"),
                "matchday": m.get("matchday"),
                "home_team": m.get("homeTeam", {}).get("name"),
                "away_team": m.get("awayTeam", {}).get("name"),
                "home_goals": m.get("score", {}).get("fullTime", {}).get("home"),
                "away_goals": m.get("score", {}).get("fullTime", {}).get("away"),
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df["date"] = pd.to_datetime(df["utcDate"], errors="coerce")
            df["match_id"] = range(1, len(df)+1)
        return df
    except HTTPError as e:
        if e.response is not None and e.response.status_code in (401,403):
            return _fd_uk_csv(lg_name, season)
        raise

def ingest_full(seasons: List[int] | None = None, leagues: List[str] | None = None) -> None:
    if seasons:
        start, end = seasons[0], seasons[-1]
    else:
        start, end = settings.seasons_window()
    seasons = seasons or list(range(start, end+1))
    leagues = leagues or ["EPL"]

    outdir = RAW / "football-data"
    outdir.mkdir(parents=True, exist_ok=True)
    epl_all = []

    for lg in leagues:
        code = FD_COMP[lg]
        for yr in seasons:
            log.info("Downloading %s %s…", lg, yr)
            df = _fd_matches(code, yr, lg)
            write_parquet(df, outdir / f"{lg}_{yr}_matches.parquet")
            if lg == "EPL" and not df.empty:
                epl_all.append(df)
            time.sleep(0.5)

    if epl_all:
        import pandas as pd
        write_parquet(pd.concat(epl_all, ignore_index=True), outdir / "EPL_all_matches.parquet")
        log.info("Wrote combined EPL matches.")
