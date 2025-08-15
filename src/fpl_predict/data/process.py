from __future__ import annotations
import requests
import pandas as pd
from ..utils.cache import RAW, PROC
from ..utils.io import read_parquet, write_parquet
from ..utils.logging import get_logger
from .fpl_api import get_bootstrap, get_element_summary
import math

log = get_logger(__name__)
FPL_BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"

def _seed_features_from_fpl() -> pd.DataFrame:
    js = get_bootstrap()
    els = js.get("elements", [])
    pos_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

    rows = []
    for i, e in enumerate(els, 1):
        pid = int(e["id"])

        # ---- per-player recent minutes (current season) ----
        mins_l5 = mins_l3 = 0.0
        starts_l5 = bench_l5 = 0
        # ---- previous-season priors (from history_past) ----
        mins_prev = starts_prev = 0.0
        prev_goals_per90 = prev_assists_per90 = 0.0
        prev_xg_per90 = prev_xa_per90 = 0.0

        try:
            summ = get_element_summary(pid)

            # current season history
            hist = (summ.get("history") or [])
            last5 = hist[-5:]
            mins_l5 = sum(h.get("minutes") or 0 for h in last5) / max(1, len(last5))
            last3 = last5[-3:] if last5 else []
            mins_l3 = sum(h.get("minutes") or 0 for h in last3) / max(1, len(last3))
            starts_l5 = sum(1 for h in last5 if (h.get("minutes") or 0) >= 60)
            bench_l5 = sum(1 for h in last5 if 1 <= (h.get("minutes") or 0) < 60)

            # previous season rollup (most recent)
            past = (summ.get("history_past") or [])
            last = past[-1] if past else {}
            mins_prev = float(last.get("minutes") or 0.0)
            # some seasons expose 'starts', some just 'appearances'
            starts_prev = float(last.get("starts") or last.get("appearances") or 0.0)
            g_prev = float(last.get("goals_scored") or 0.0)
            a_prev = float(last.get("assists") or 0.0)

            # xG/xA may be totals or per90 depending on vintage; handle both
            xg_prev_tot = last.get("expected_goals")
            xa_prev_tot = last.get("expected_assists")
            xg_prev_p90 = last.get("expected_goals_per_90")
            xa_prev_p90 = last.get("expected_assists_per_90")

            def per90(val):
                return (float(val) / mins_prev * 90.0) if mins_prev and float(mins_prev) > 0 else 0.0

            prev_goals_per90 = per90(g_prev)
            prev_assists_per90 = per90(a_prev)
            if xg_prev_p90 is not None and xa_prev_p90 is not None:
                prev_xg_per90 = float(xg_prev_p90 or 0.0)
                prev_xa_per90 = float(xa_prev_p90 or 0.0)
            else:
                prev_xg_per90 = per90(xg_prev_tot or 0.0)
                prev_xa_per90 = per90(xa_prev_tot or 0.0)

        except Exception:
            # leave defaults (zeros) if the per-player call fails
            pass

        chance_next = e.get("chance_of_playing_next_round")
        if chance_next is None:
            chance_next = 100 if (e.get("status") or "a") == "a" else 0

        rows.append({
            "player_id": pid,
            "position": pos_map.get(e.get("element_type")),
            "team": e.get("team"),
            "now_cost": e.get("now_cost"),
            "form": float(e.get("form") or 0.0),
            "ep_next": float(e.get("ep_next") or 0.0),
            "selected_by_percent": float(e.get("selected_by_percent") or 0.0),
            "status": e.get("status"),
            "chance_next": float(chance_next),

            # current-season signals
            "mins_l5": float(mins_l5),
            "mins_l3": float(mins_l3),
            "starts_l5": int(starts_l5),
            "bench_l5": int(bench_l5),

            # previous-season priors
            "prev_minutes": float(mins_prev),
            "prev_starts": float(starts_prev),
            "prev_goals_per90": float(prev_goals_per90),
            "prev_assists_per90": float(prev_assists_per90),
            "prev_xg_per90": float(prev_xg_per90),
            "prev_xa_per90": float(prev_xa_per90),

            # placeholders (zeros) until we wire richer sources
            "tackles": 0, "interceptions": 0, "blocks": 0, "clearances": 0, "recoveries": 0,
        })

        if i % 100 == 0:
            log.info("FPL per-player summaries processed: %d / %d", i, len(els))

    df = pd.DataFrame(rows)

    # minimal dtype hygiene
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df["position"] = df["position"].astype("string")
    df["team"] = pd.to_numeric(df["team"], errors="coerce").astype("Int64")

    write_parquet(df, PROC / "features.parquet")
    log.info("Processed features seeded from FPL bootstrap + per-player history: %d players", len(df))
    return df

def build_features() -> None:
    epl_all = RAW / "football-data" / "EPL_all_matches.parquet"
    if epl_all.exists():
        fx = read_parquet(epl_all).copy()
        for c in ["match_id","date","home_team","away_team","home_goals","away_goals"]:
            if c not in fx.columns: fx[c] = None
        write_parquet(fx[["match_id","date","home_team","away_team","home_goals","away_goals"]], PROC / "fixtures.parquet")
        _seed_features_from_fpl()
        return

    # DEMO fallback
    fixtures = read_parquet(RAW / "sample" / "fixtures_2024-25.parquet")
    events = read_parquet(RAW / "sample" / "events_2024-25.parquet")
    feats = (
        events.groupby("player_id").agg(
            minutes=("minutes","sum"), goals=("goals","sum"), assists=("assists","sum"),
            tackles=("tackles","sum"), interceptions=("interceptions","sum"),
            blocks=("blocks","sum"), clearances=("clearances","sum"),
            recoveries=("ball_recoveries","sum"),
        )
    ).reset_index()
    write_parquet(fixtures, PROC / "fixtures.parquet")
    write_parquet(feats, PROC / "features.parquet")
    log.info("Processed features written (DEMO).")
