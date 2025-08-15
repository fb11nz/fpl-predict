from __future__ import annotations
import re
import numpy as np
import pandas as pd
from ..utils.io import read_parquet, write_parquet
from ..utils.cache import PROC
from ..utils.logging import get_logger
from ..data.fpl_api import get_bootstrap, get_fixtures as fpl_fixtures
from .elo import update_elo, expected_score

log = get_logger(__name__)

def _team_key(name: str) -> str:
    s = str(name or "").lower()
    s = s.replace("&", "and")
    s = re.sub(r"\b(fc|afc)\b", "", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

def _apply_name_failsafe(df: pd.DataFrame, id2name: dict[int, str], short2name: dict[str, str]) -> None:
    """Convert any digits-only or 2–4 letter codes to full names in-place."""
    def fix(v):
        s = str(v)
        if s.isdigit():
            return id2name.get(int(s), s)
        if 2 <= len(s) <= 4 and s.isupper():  # e.g., 'ARS'
            return short2name.get(s, s)
        return s
    df["home_team"] = df["home_team"].map(fix)
    df["away_team"] = df["away_team"].map(fix)

def compute_fdr() -> pd.DataFrame:
    # 1) Past fixtures → Elo ratings on normalized keys
    fixtures_past = read_parquet(PROC / "fixtures.parquet").copy()
    # Create normalized keys for Elo only
    fp = fixtures_past.rename(columns={"home_team": "home_nm", "away_team": "away_nm"})
    fp["home_team"] = fp["home_nm"].map(_team_key)
    fp["away_team"] = fp["away_nm"].map(_team_key)

    ratings = update_elo(fp)  # dict keyed by normalized key

    # If Elo came back flat → build a rating prior from FPL team strengths
    boot = get_bootstrap()
    tdf = pd.DataFrame(boot["teams"])
    tdf["name"] = tdf["name"].astype("string")
    tdf["short_name"] = tdf["short_name"].astype("string")
    # robust id→name
    empty = tdf["name"].isna() | (tdf["name"].str.strip() == "")
    tdf.loc[empty, "name"] = tdf.loc[empty, "short_name"]
    id2name = dict(zip(tdf["id"].astype(int), tdf["name"].astype(str)))
    short2name = dict(zip(tdf["short_name"].astype(str), tdf["name"].astype(str)))

    # Elo flat? Seed from strengths
    if len(ratings) == 0 or np.std(list(ratings.values())) < 1e-6:
        # use overall strengths; scale to an "Elo-like" spread
        s = (tdf["strength_overall_home"].fillna(0) + tdf["strength_overall_away"].fillna(0)).astype(float)
        s = (s - s.mean()) / (s.std() if s.std() else 1.0)
        base = 1500.0 + 120.0 * s  # ~±120 pts spread
        for _id, nm, r in zip(tdf["id"].astype(int), tdf["name"].astype(str), base):
            ratings[_team_key(nm)] = float(r)
        log.info("Elo flat → using FPL team strengths as rating prior.")

    rows = []

    # 2) Past rows (display original names, lookup by normalized key)
    for _, m in fixtures_past.iterrows():
        hk = _team_key(m["home_team"]); ak = _team_key(m["away_team"])
        rh = ratings.get(hk, 1500.0); ra = ratings.get(ak, 1500.0)
        p_home = expected_score(rh, ra, home=True)
        rows.append({
            "match_id": int(m["match_id"]) if pd.notna(m["match_id"]) else None,
            "home_team": str(m["home_team"]),
            "away_team": str(m["away_team"]),
            "fdr_home": float(1.0 - p_home),
            "fdr_away": float(p_home),
            "is_future": False,
            "event": None,
            "kickoff_time": m.get("date", None),
        })

    # 3) Future fixtures (IDs → names; lookup by normalized key)
    upcoming = fpl_fixtures()
    for fx in upcoming:
        if fx.get("finished") or fx.get("finished_provisional"):
            continue
        th = fx.get("team_h"); ta = fx.get("team_a")
        home_nm = id2name.get(int(th), str(th))
        away_nm = id2name.get(int(ta), str(ta))
        hk = _team_key(home_nm); ak = _team_key(away_nm)
        rh = ratings.get(hk, 1500.0); ra = ratings.get(ak, 1500.0)
        p_home = expected_score(rh, ra, home=True)
        rows.append({
            "match_id": int(fx.get("id")) if fx.get("id") is not None else None,
            "home_team": str(home_nm),
            "away_team": str(away_nm),
            "fdr_home": float(1.0 - p_home),
            "fdr_away": float(p_home),
            "is_future": True,
            "event": fx.get("event"),
            "kickoff_time": fx.get("kickoff_time"),
        })

    fdr = pd.DataFrame(rows)

    # 4) FINAL FAILSAFE: replace any digits/short-codes with full names
    _apply_name_failsafe(fdr, id2name=id2name, short2name=short2name)

    # 5) Dtypes for Parquet
    fdr["home_team"] = fdr["home_team"].astype("string")
    fdr["away_team"] = fdr["away_team"].astype("string")
    for col in ("fdr_home", "fdr_away"):
        fdr[col] = pd.to_numeric(fdr[col], errors="coerce")
    for col in ("match_id", "event"):
        fdr[col] = pd.to_numeric(fdr[col], errors="coerce").astype("Int64")
    kt = pd.to_datetime(fdr["kickoff_time"], utc=True, errors="coerce")
    fdr["kickoff_time"] = kt.dt.strftime("%Y-%m-%dT%H:%M:%SZ").astype("string")
    fdr["is_future"] = fdr["is_future"].astype(bool)

    write_parquet(fdr, PROC / "fdr.parquet")
    log.info("FDR computed and written (past + future): %d rows", len(fdr))
    return fdr

def build_player_next5_fdr(current_event: int | None = None) -> pd.DataFrame:
    from ..data.fpl_api import get_bootstrap
    fdr = read_parquet(PROC / "fdr.parquet")
    fut = fdr[fdr["is_future"] == True].copy()

    if fut.empty:
        boot = get_bootstrap()
        out = pd.DataFrame({"player_id": [e["id"] for e in boot["elements"]],
                            "fdr_factor": 1.0})
        out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").astype("Int64")
        write_parquet(out, PROC / "player_next5_fdr.parquet")
        log.info("player_next5_fdr: no future fixtures; wrote neutral factors for %d players", len(out))
        return out

    fut["event"] = pd.to_numeric(fut["event"], errors="coerce").astype("Int64")
    if current_event is None:
        gws = fut["event"].dropna()
        current_event = int(gws.min()) if not gws.empty else None

    # team-level factor (inverse mean difficulty over next 5; normalized ~1.0)
    team_fac: dict[str, float] = {}
    for team_col, fdr_col in (("home_team", "fdr_home"), ("away_team", "fdr_away")):
        for tm, grp in fut.sort_values("event").groupby(team_col):
            sel = grp if current_event is None else grp[grp["event"] >= current_event]
            nxt = sel.head(5)
            if nxt.empty:
                fac = 1.0
            else:
                fac = float(1.0 / (pd.to_numeric(nxt[fdr_col], errors="coerce").mean() + 1e-9))
            team_fac[tm] = team_fac.get(tm, 0.0) + fac

    # average home/away contributions and normalize
    if team_fac:
        mean_fac = sum(team_fac.values()) / len(team_fac)
        for k in list(team_fac.keys()):
            team_fac[k] = team_fac[k] / (mean_fac if mean_fac else 1.0)

    # map to players via bootstrap team names
    boot = get_bootstrap()
    teams = pd.DataFrame(boot["teams"])[["id", "name", "short_name"]]
    teams["name"] = teams["name"].astype("string")
    teams["short_name"] = teams["short_name"].astype("string")
    empty = teams["name"].isna() | (teams["name"].str.strip() == "")
    teams.loc[empty, "name"] = teams.loc[empty, "short_name"]
    id_to_name = dict(zip(teams["id"].astype(int), teams["name"].astype(str)))

    elements = pd.DataFrame(boot["elements"])[["id", "team"]]
    elements = elements.rename(columns={"id": "player_id", "team": "team_id"})
    elements["team_name"] = elements["team_id"].map(id_to_name).astype("string")
    elements["fdr_factor"] = elements["team_name"].map(team_fac).fillna(1.0)

    out = elements[["player_id", "fdr_factor"]].copy()
    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").astype("Int64")
    out["fdr_factor"] = pd.to_numeric(out["fdr_factor"], errors="coerce")
    write_parquet(out, PROC / "player_next5_fdr.parquet")
    log.info("player_next5_fdr written for %d players", len(out))
    return out