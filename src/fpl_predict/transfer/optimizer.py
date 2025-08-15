# src/fpl_predict/transfer/optimizer.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Set, Optional
import pandas as pd
import numpy as np

import requests

from ..utils.logging import get_logger
from ..utils.io import read_parquet
from ..utils.cache import PROC

log = get_logger(__name__)

# Try to import optimization libraries
try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False
    log.debug("PuLP not available - LP optimization disabled")

# ---- Squad rules ----
POSITIONS = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_CLUB = 3
BUDGET = 1000  # tenths of a million, i.e. £100.0m

FORMATIONS = {
    "343": {"GKP": 1, "DEF": 3, "MID": 4, "FWD": 3},
    "352": {"GKP": 1, "DEF": 3, "MID": 5, "FWD": 2},
    "442": {"GKP": 1, "DEF": 4, "MID": 4, "FWD": 2},
    "451": {"GKP": 1, "DEF": 4, "MID": 5, "FWD": 1},
    "433": {"GKP": 1, "DEF": 4, "MID": 3, "FWD": 3},
    "541": {"GKP": 1, "DEF": 5, "MID": 4, "FWD": 1},
    "532": {"GKP": 1, "DEF": 5, "MID": 3, "FWD": 2},
}

# mild depth heuristics to avoid obvious backups
EXPECTED_TEAM_STARTERS = {"DEF": 4, "MID": 4, "FWD": 2}


@dataclass
class Player:
    id: int
    name: str
    pos: str
    team: int
    cost: int              # tenths of a million
    xmins: float           # 0..90
    ep_base: float         # single-GW, minutes-adjusted baseline
    xgi90: float           # optional per-90 attack involvement
    team_att: float        # optional team attack strength ~1.0
    ep_seq: List[float]    # EP per GW across horizon H
    ep1: float             # ep_seq[0]
    eph: float             # horizon-weighted sum
    
    # Additional fields for LP optimization
    form: float = 0.0
    selected_by: float = 0.0  # Ownership %
    value_score: float = 0.0
    is_differential: bool = False
    injury_risk: float = 0.0
    rotation_risk: float = 0.0


# ------------------- data loaders -------------------
def _fetch_bootstrap() -> dict:
    r = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=30)
    r.raise_for_status()
    return r.json()


def _fetch_fixtures() -> list:
    r = requests.get("https://fantasy.premierleague.com/api/fixtures/", timeout=30)
    r.raise_for_status()
    return r.json()


def _load_xmins() -> Dict[int, float]:
    try:
        df = read_parquet(PROC / "xmins.parquet")
        return {int(pid): float(xm) for pid, xm in zip(df["player_id"], df["xmins"])}
    except Exception as e:
        log.warning("xMins parquet missing; defaulting to 70 mins. (%s)", e)
        return {}


def _load_ep_extras() -> tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    """Return ep_adjusted (or ep_blend or ep_model), xgi90_est (optional), team_att (optional)."""
    try:
        df = read_parquet(PROC / "exp_points.parquet")
        # Prioritize ep_adjusted which accounts for playing time
        if "ep_adjusted" in df.columns:
            ep_col = "ep_adjusted"
        elif "ep_blend" in df.columns:
            ep_col = "ep_blend"
        else:
            ep_col = "ep_model"
        ep_map = {int(r.player_id): float(getattr(r, ep_col)) for _, r in df.iterrows()}
        xgi_map = {int(r.player_id): float(getattr(r, "xgi90_est", 0.0)) for _, r in df.iterrows()}
        tatt_map = {int(r.player_id): float(getattr(r, "team_att", 1.0)) for _, r in df.iterrows()}
        return ep_map, xgi_map, tatt_map
    except Exception:
        return {}, {}, {}


def _pergw_factors(team_id: int, H: int) -> list[float]:
    """Fixture difficulty → factors ~[0.84..1.16]."""
    try:
        fx = _fetch_fixtures()
    except Exception:
        return [1.0] * H
    ups = [f for f in fx if (not f.get("finished")) and f.get("event") and (f.get("team_h")==team_id or f.get("team_a")==team_id)]
    ups.sort(key=lambda f: f.get("event", 999))
    facs = []
    for f in ups[:H]:
        if f.get("team_h") == team_id:
            diff = int(f.get("team_h_difficulty") or 3)
        else:
            diff = int(f.get("team_a_difficulty") or 3)
        fac = 1.0 + (3 - diff) * 0.08
        facs.append(max(0.80, min(1.20, fac)))
    while len(facs) < H:
        facs.append(1.0)
    return facs


def _parse_weights(hweights: str, H: int) -> list[float]:
    if hweights:
        try:
            ws = [float(x.strip()) for x in hweights.split(",") if x.strip()]
            if len(ws) >= H:
                return ws[:H]
            while len(ws) < H:
                ws.append(ws[-1] if ws else 1.0)
            return ws
        except Exception:
            pass
    w = [1.0]
    for _ in range(1, H):
        w.append(w[-1] * 0.9)
    return w


def _candidate_pool(
    js: dict,
    xmins_map: dict[int, float],
    H: int,
    weights: list[float],
    fdr_weight: float,
) -> List[Player]:
    ep_blend, xgi90_est, team_att = _load_ep_extras()

    pool: List[Player] = []
    for e in js["elements"]:
        pid = int(e["id"])
        pos = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}[e["element_type"]]
        team_id = int(e["team"])
        # ep_blend now contains ep_adjusted which already accounts for minutes
        ep0 = float(ep_blend.get(pid, float(e.get("ep_next") or 0.0)))
        xm = float(xmins_map.get(pid, 70.0))
        
        # STRUCTURAL CONSTRAINT FIX: Prevent forcing weak bench players into XI
        # Key insight: We must field 11 players, so having 3+ cheap players from
        # weak teams forces us to start at least one (limiting flexibility)
        cost = int(e.get("now_cost") or 0)
        
        # Mark players as "bench-only" if they're cheap + low EP
        # This helps the optimizer understand they shouldn't be in the XI
        is_bench_quality = (cost <= 40 and ep0 < 2.0 and pos != "GKP")
        
        # Skip non-playing fodder entirely
        if cost <= 45 and xm < 60.0 and pos != "GKP":
            continue
        
        # Skip ultra-cheap players with very low EP (likely not playing)
        if cost <= 40 and ep0 < 1.2 and pos != "GKP":
            continue
            
        # Skip backup GKPs with 0 playing time (ep_adjusted handles this)
        if pos == "GKP" and ep0 == 0.0:
            continue
        
        # Note: ep0 is already minutes-adjusted via ep_adjusted column
        xgi90 = float(xgi90_est.get(pid, 0.0))
        t_att = float(team_att.get(pid, 1.0))

        facs = _pergw_factors(team_id, H)
        ep_seq = [max(0.0, ep0 * (1.0 + fdr_weight * (f - 1.0))) for f in facs]
        wsum = sum(w * v for w, v in zip(weights, ep_seq))

        # Add a penalty for bench-quality players to discourage starting them
        # This addresses the structural constraint issue
        ep_penalty = 1.0
        if is_bench_quality:
            # Apply 30% penalty to EP for optimization purposes
            # This makes the optimizer prefer other options for the starting XI
            ep_penalty = 0.7
            ep_seq = [ep * ep_penalty for ep in ep_seq]
            wsum = sum(w * v for w, v in zip(weights, ep_seq))
        
        pool.append(
            Player(
                id=pid,
                name=e["web_name"],
                pos=pos,
                team=team_id,
                cost=int(e.get("now_cost") or 0),
                xmins=xm,
                ep_base=ep0,
                xgi90=xgi90,
                team_att=t_att,
                ep_seq=ep_seq,
                ep1=ep_seq[0] if ep_seq else 0.0,
                eph=wsum,
            )
        )
    return pool


# ------------------- helpers -------------------
def _rank_key(p: Player) -> float:
    return 1.0 * p.ep_base + 0.2 * (p.xmins / 90.0) + 0.1 * p.xgi90


def _adjust_xmins_with_depth(pool: List[Player], nonstarter_xmins: float, gk_backup_xmins: float) -> None:
    by_team: Dict[int, List[Player]] = {}
    for p in pool:
        by_team.setdefault(p.team, []).append(p)

    for players in by_team.values():
        # GK: one clear starter → others trimmed
        gks = [p for p in players if p.pos == "GKP"]
        if gks:
            starter = max(gks, key=_rank_key)
            for g in gks:
                if g.id != starter.id:
                    g.xmins = min(g.xmins, gk_backup_xmins)

        # Outfield: trim deepest bench
        for pos in ["DEF", "MID", "FWD"]:
            ps = [p for p in players if p.pos == pos]
            if not ps:
                continue
            ps.sort(key=_rank_key, reverse=True)
            starters = EXPECTED_TEAM_STARTERS[pos]
            for idx, p in enumerate(ps):
                if idx >= starters:
                    p.xmins = min(p.xmins, nonstarter_xmins)

    # ultra-cheap nonstarters → tiny minutes
    for p in pool:
        if p.pos != "GKP" and p.cost <= 45 and p.ep_base <= 2.0:
            p.xmins = min(p.xmins, max(10.0, 0.5 * nonstarter_xmins))


def _club_counts(players: List[Player]) -> Dict[int, int]:
    cc: Dict[int, int] = {}
    for p in players:
        cc[p.team] = cc.get(p.team, 0) + 1
    return cc


def _total_spent(players: List[Player]) -> int:
    return sum(p.cost for p in players)


# ------------------- bench template (NEW: guarantees cap) -------------------
def _cheapest_by_pos(pool: List[Player], pos: str, exclude_ids: Set[int], used_clubs: Dict[int, int]) -> Optional[Player]:
    cands = [p for p in pool if p.pos == pos and p.id not in exclude_ids]
    cands.sort(key=lambda z: (z.cost, -z.ep1))
    for q in cands:
        if used_clubs.get(q.team, 0) >= MAX_PER_CLUB:
            continue
        return q
    return None


def _choose_cheap_bench_template(pool: List[Player], bench_budget: int) -> List[Player]:
    """
    Pick 4 cheap bench players under club caps:
      - exactly 1 GK (cheapest)
      - 3 outfielders across DEF/MID/FWD with minimum total cost
    Return list of 4 Players. Guarantees cost <= bench_budget if feasible in the pool.
    """
    used: List[Player] = []
    exclude: Set[int] = set()
    clubs: Dict[int, int] = {}

    # 1) cheapest GK
    gk = _cheapest_by_pos(pool, "GKP", exclude, clubs)
    if not gk:
        raise RuntimeError("No goalkeepers in pool.")
    used.append(gk); exclude.add(gk.id); clubs[gk.team] = clubs.get(gk.team, 0) + 1

    # 2) pick cheapest 3 outfielders (any of DEF/MID/FWD) under club caps
    # STRUCTURAL FIX: Prioritize better cheap defenders to avoid triple weak teams
    outfield = [p for p in pool if p.pos in {"DEF", "MID", "FWD"} and p.id not in exclude]
    
    # Sort with smarter logic: still cheap but prefer higher EP to avoid forcing weak players into XI
    # This addresses the "must field 11 players" constraint
    def bench_score(p):
        # For £4.0m players, heavily weight EP to avoid weak teams
        # For £4.5m+ players, still prefer cheaper
        if p.cost <= 40:
            return (p.cost, -p.ep1 * 10)  # Heavily weight EP for £4.0m
        else:
            return (p.cost, -p.ep1)
    
    outfield.sort(key=bench_score)
    
    # Track how many from potentially weak teams (low EP defenders)
    weak_team_players = 0
    
    for q in outfield:
        if len(used) == 4:
            break
        if clubs.get(q.team, 0) >= MAX_PER_CLUB:
            continue
        
        # STRUCTURAL CONSTRAINT: If this is a £4.0m defender with <2.0 EP
        # and we already have 2 such players, skip to avoid triple weak team
        if q.pos == "DEF" and q.cost <= 40 and q.ep1 < 2.0:
            if weak_team_players >= 2:
                continue  # Look for slightly better option
            weak_team_players += 1
        
        used.append(q)
        exclude.add(q.id)
        clubs[q.team] = clubs.get(q.team, 0) + 1

    # If still < 4, fill regardless of club caps (should be rare)
    if len(used) < 4:
        for q in outfield:
            if q.id in exclude:
                continue
            used.append(q)
            if len(used) == 4:
                break

    if len(used) < 4:
        raise RuntimeError("Unable to construct a 4-man bench template from pool.")

    cost = _total_spent(used)
    if cost > bench_budget:
        # Try to improve: swap the most expensive outfielder down if possible
        tries = 0
        while cost > bench_budget and tries < 50:
            tries += 1
            # find most expensive outfielder among the 3 outfielders
            ofs = [p for p in used if p.pos in {"DEF", "MID", "FWD"}]
            worst = max(ofs, key=lambda z: z.cost)
            # search a cheaper same-pos alt not yet used, under club caps
            cands = [p for p in outfield if p.pos == worst.pos and p.id not in {pp.id for pp in used}]
            cands.sort(key=lambda z: (z.cost, -z.ep1))
            swapped = False
            for q in cands:
                if q.cost >= worst.cost:
                    break
                # club caps
                clubs_now = _club_counts([pp for pp in used if pp.id != worst.id])
                if clubs_now.get(q.team, 0) >= MAX_PER_CLUB:
                    continue
                used = [q if pp.id == worst.id else pp for pp in used]
                cost = _total_spent(used)
                swapped = True
                break
            if not swapped:
                break

    return used


# ------------------- greedy selection (respects bench template) -------------------
def _min_costs_baseline(pool: List[Player]) -> Dict[str, int]:
    out = {}
    for pos in ["GKP", "DEF", "MID", "FWD"]:
        costs = [p.cost for p in pool if p.pos == pos]
        out[pos] = min(costs) if costs else 0
    return out


def _pick15_with_bench_template(pool: List[Player], bench_template: List[Player]) -> List[Player]:
    """Pick 15 ensuring the 4 bench-template players are in the squad; fill the rest greedily."""
    template_ids = {p.id for p in bench_template}

    def score(p: Player) -> float:
        # Slightly de-prioritize template guys for XI (they're meant to be cheap bench)
        bench_pen = 0.9 if p.id in template_ids else 1.0
        price = max(35.0, float(p.cost))
        return bench_pen * ((0.7 * p.eph + 0.3 * p.ep1) / price + 0.02 * (p.xmins / 90.0))

    cand = sorted(pool, key=score, reverse=True)

    need = dict(POSITIONS)
    picked: List[Player] = []
    clubs: Dict[int, int] = {}
    spent = 0
    min_cost = _min_costs_baseline(pool)

    # First, add the 4 bench-template players (they still count to totals)
    for p in bench_template:
        if need[p.pos] <= 0:
            # if template conflicts with totals (very rare), skip it
            continue
        if clubs.get(p.team, 0) >= MAX_PER_CLUB:
            continue
        if (spent + p.cost) > BUDGET:
            continue
        picked.append(p)
        need[p.pos] -= 1
        spent += p.cost
        clubs[p.team] = clubs.get(p.team, 0) + 1

    # Then fill the rest greedily
    def can_afford(cost_add: int, need_after: Dict[str, int]) -> bool:
        remaining_min = sum(need_after[pos] * min_cost.get(pos, 0) for pos in need_after)
        return (spent + cost_add + remaining_min) <= BUDGET

    for p in cand:
        if p in picked:
            continue
        if need[p.pos] <= 0:
            continue
        if clubs.get(p.team, 0) >= MAX_PER_CLUB:
            continue
        after = need.copy()
        after[p.pos] -= 1
        if not can_afford(p.cost, after):
            continue
        picked.append(p)
        need[p.pos] -= 1
        spent += p.cost
        clubs[p.team] = clubs.get(p.team, 0) + 1
        if sum(need.values()) == 0:
            break

    # fill if somehow short
    if sum(need.values()) > 0:
        cheap = sorted([q for q in pool if q not in picked], key=lambda q: (q.cost, -q.eph))
        for q in cheap:
            if need[q.pos] <= 0:
                continue
            if clubs.get(q.team, 0) >= MAX_PER_CLUB:
                continue
            if spent + q.cost > BUDGET:
                continue
            picked.append(q)
            need[q.pos] -= 1
            spent += q.cost
            clubs[q.team] = clubs.get(q.team, 0) + 1
            if sum(need.values()) == 0:
                break

    assert len(picked) == 15, f"picked={len(picked)}"
    assert sum(1 for p in picked if p.pos == "GKP") == 2
    assert sum(1 for p in picked if p.pos == "DEF") == 5
    assert sum(1 for p in picked if p.pos == "MID") == 5
    assert sum(1 for p in picked if p.pos == "FWD") == 3
    return picked


def _bench_cost(bench: List[Player]) -> int:
    return sum(p.cost for p in bench)


def _xi_for_formation(picked: List[Player], form_key: str) -> List[Player]:
    req = FORMATIONS[form_key]
    xi: List[Player] = []

    # GK: pick higher ep1 as starter
    gks = sorted([p for p in picked if p.pos == "GKP"], key=lambda p: p.ep1, reverse=True)
    xi.extend(gks[: req["GKP"]])

    # DEF/MID/FWD by ep1
    for pos in ["DEF", "MID", "FWD"]:
        needn = req[pos]
        cands = [p for p in picked if p.pos == pos and p not in xi]
        cands.sort(key=lambda p: p.ep1, reverse=True)
        xi.extend(cands[:needn])

    if len(xi) < 11:
        rem = [p for p in picked if p not in xi]
        rem.sort(key=lambda p: p.ep1, reverse=True)
        xi.extend(rem[: 11 - len(xi)])

    return xi[:11]


def _repair_bench_budget(xi: List[Player], picked: List[Player], bench_budget: int) -> Tuple[List[Player], List[Player]]:
    """
    Make bench ≤ bench_budget via same-position swaps (bench ↔ XI).
    Also ensure cheaper GK is on the bench.
    """
    bench = [p for p in picked if p not in xi][:4]

    # Ensure CHEAPER GK is on the bench
    xi_gk = [p for p in xi if p.pos == "GKP"]
    bench_gk = [p for p in bench if p.pos == "GKP"]
    if xi_gk and bench_gk and bench_gk[0].cost > xi_gk[0].cost:
        xi.remove(xi_gk[0])
        bench.remove(bench_gk[0])
        xi.append(bench_gk[0])
        bench.append(xi_gk[0])

    # Swap bench with XI (same-position) to push expensive pieces into XI
    tries = 0
    while _bench_cost(bench) > bench_budget and tries < 40:
        tries += 1
        best = None  # (cost_saved, -ep_loss, b_idx, i_idx)
        bench_by_pos: Dict[str, List[int]] = {}
        xi_by_pos: Dict[str, List[int]] = {}
        for idx, p in enumerate(bench):
            bench_by_pos.setdefault(p.pos, []).append(idx)
        for idx, p in enumerate(xi):
            xi_by_pos.setdefault(p.pos, []).append(idx)

        for pos in ["GKP", "DEF", "MID", "FWD"]:
            for b_idx in bench_by_pos.get(pos, []):
                b = bench[b_idx]
                for i_idx in xi_by_pos.get(pos, []):
                    i = xi[i_idx]
                    cost_saved = b.cost - i.cost  # reduce bench cost if positive
                    if cost_saved <= 0:
                        continue
                    ep_loss = max(0.0, i.ep1 - b.ep1)
                    cand = (cost_saved, -ep_loss, b_idx, i_idx)
                    if (best is None) or (cand > best):
                        best = cand

        if best is None:
            break
        _, _, b_idx, i_idx = best
        b = bench[b_idx]
        i = xi[i_idx]
        xi[i_idx] = b
        bench[b_idx] = i

    return xi, bench


def _downgrade_bench_until_cap(
    pool: List[Player],
    picked: List[Player],
    form: str,
    bench_budget: int,
) -> Tuple[List[Player], List[Player]]:
    """
    Replace expensive bench players with cheaper **same-position** pool candidates,
    respecting total budget and club caps. Rebuild XI/bench each time.
    """
    xi = _xi_for_formation(picked, form)
    xi, bench = _repair_bench_budget(xi, picked, bench_budget)
    if _bench_cost(bench) <= bench_budget:
        return xi, bench

    # pool by position (not already picked)
    pool_by_pos: Dict[str, List[Player]] = {}
    picked_ids = {p.id for p in picked}
    for p in pool:
        if p.id in picked_ids:
            continue
        pool_by_pos.setdefault(p.pos, []).append(p)
    for pos in pool_by_pos:
        pool_by_pos[pos].sort(key=lambda z: (z.cost, -z.ep1))

    tries = 0
    while _bench_cost(bench) > bench_budget and tries < 80:
        tries += 1
        b_idx_sorted = sorted(range(len(bench)), key=lambda i: bench[i].cost, reverse=True)
        made = False
        for b_idx in b_idx_sorted:
            b = bench[b_idx]
            current_spent = _total_spent(picked)
            clubs = _club_counts(picked)
            for q in pool_by_pos.get(b.pos, []):
                if q.cost >= b.cost:
                    break  # can't reduce spend with this q
                # club caps after swap
                clubs_after = dict(clubs)
                clubs_after[b.team] = clubs_after.get(b.team, 0) - 1
                if clubs_after[b.team] <= 0:
                    clubs_after.pop(b.team, None)
                clubs_after[q.team] = clubs_after.get(q.team, 0) + 1
                if clubs_after[q.team] > MAX_PER_CLUB:
                    continue
                if (current_spent - b.cost + q.cost) > BUDGET:
                    continue

                # perform replacement in picked
                picked = [q if (pp.id == b.id) else pp for pp in picked]
                xi = _xi_for_formation(picked, form)
                xi, bench = _repair_bench_budget(xi, picked, bench_budget)
                made = True
                break
            if made:
                break
        if not made:
            break

    return xi, bench


def _choose_xi_and_bench(
    picked: List[Player],
    formations_allowed: List[str],
    bench_budget: int,
    pool: Optional[List[Player]] = None,
) -> Tuple[List[Player], List[Player], str]:
    """
    Try each formation; repair bench; if still over, try downgrades using pool.
    """
    best = None  # (feasible(bool), xi_score, form, xi, bench)
    for form in formations_allowed:
        xi0 = _xi_for_formation(picked, form)
        xi, bench = _repair_bench_budget(xi0, picked, bench_budget)
        feasible = _bench_cost(bench) <= bench_budget
        score = sum(p.ep1 for p in xi)
        key = (feasible, score)
        if best is None or key > (best[0], best[1]):
            best = (feasible, score, form, xi, bench)

    feasible, _, form, xi, bench = best
    if feasible:
        return xi, bench, form

    if pool is not None:
        xi2, bench2 = _downgrade_bench_until_cap(pool, picked, form, bench_budget)
        return xi2, bench2, form

    return xi, bench, form


# ------------------- helpers for advanced optimizer -------------------
def _format_advanced_result(result: Dict[str, Any], horizon: int, bench_budget: int) -> Dict[str, Any]:
    """Format advanced optimizer result to match expected output format."""
    squad = result.get("squad", [])
    xi = result.get("starting_xi", [])
    bench = result.get("bench", [])
    captain = result.get("captain")
    formation = result.get("formation", "442")
    
    def fmt(p) -> str:
        return f" - {p.name} ({p.position}) — ep1 {p.ep_next:.2f}, next{horizon} {sum(p.ep_horizon[:horizon]):.2f}, xMins {p.xmins:.0f}, £{p.price:.1f}m"
    
    human = []
    human.append(f"Suggested 15-man squad (LP Optimized), formation {formation}:")
    human.append(f"Expected Points (GW1): {result.get('expected_points_gw1', 0):.1f}")
    human.append(f"Optimization Score: {result.get('optimization_score', 0):.1f}")
    human.append("")
    
    human.append("Full Squad:")
    for p in squad:
        human.append(fmt(p))
    human.append("")
    
    human.append("Starting XI:")
    for p in xi:
        marker = " (C)" if captain and p.id == captain.id else ""
        human.append(fmt(p) + marker)
    human.append("")
    
    human.append("Bench:")
    for i, p in enumerate(bench, 1):
        human.append(f"{i}. {p.name} ({p.position}) — {p.ep_next:.2f} pts")
    human.append("")
    
    bench_cost = sum(p.price for p in bench)
    human.append(f"Bench value: £{bench_cost:.1f}m (target max £{bench_budget/10:.1f}m)")
    
    if captain:
        human.append(f"Captain: {captain.name} ({captain.position}) - {captain.ep_next:.2f} pts")
    
    # Add transfer suggestions if present
    if result.get("transfers_in") or result.get("transfers_out"):
        human.append("")
        human.append("Recommended Transfers:")
        for p in result.get("transfers_in", []):
            human.append(f"  IN: {p.name}")
        for pid in result.get("transfers_out", []):
            human.append(f"  OUT: Player {pid}")
    
    return {
        "human_readable": "\n".join(human),
        "picked_ids": [p.id for p in squad],
        "xi_ids": [p.id for p in xi],
        "bench_ids": [p.id for p in bench],
        "formation": formation,
        "bench_cost": bench_cost,
        "bench_budget": bench_budget / 10.0,
        "captain_id": captain.id if captain else None,
        "expected_points": result.get("expected_points_gw1", 0),
        "optimization_score": result.get("optimization_score", 0),
        "table": [
            {"player_id": p.id, "name": p.name, "pos": p.position, "team_id": p.team,
             "cost": p.price, "xmins": p.xmins, "ep1": p.ep_next, 
             f"next{horizon}": sum(p.ep_horizon[:horizon])}
            for p in squad
        ],
    }


# ------------------- LP Optimization -------------------
def optimize_with_lp(
    horizon: int = 5,
    bench_weight: float = 0.1,
    bench_budget: int = 180,
    formations: str = "343,352,442,451,433",
    differential_bonus: float = 0.1,
    risk_penalty: float = 0.05,
    value_weight: float = 0.3,
    current_squad: Optional[List[int]] = None,
    max_transfers: int = 2,
    wildcard: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Linear Programming based optimization using PuLP.
    """
    if not HAS_PULP:
        return {"error": "PuLP not installed"}
    
    try:
        # Load player pool
        js = _fetch_bootstrap()
        xmins_map = _load_xmins()
        ep_map, xgi_map, team_att_map = _load_ep_extras()
        
        # Load FDR data for fixture difficulty
        try:
            from ..utils.cache import PROC
            from ..utils.io import read_parquet
            fdr_df = read_parquet(PROC / "player_next5_fdr.parquet")
            fdr_map = dict(zip(fdr_df['player_id'], fdr_df['fdr_factor']))
            log.info(f"Loaded FDR data for {len(fdr_map)} players")
        except Exception as e:
            log.warning(f"FDR data not available: {e}")
            fdr_map = {}
        
        # Create player objects with enhanced data
        players = []
        for e in js["elements"]:
            pid = int(e["id"])
            pos = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}[e["element_type"]]
            team_id = int(e["team"])
            cost = int(e.get("now_cost", 0))
            
            xmins = float(xmins_map.get(pid, 70.0))
            ep_base = float(ep_map.get(pid, float(e.get("ep_next", 0))))
            
            # Skip ultra-cheap fodder unless it's a GKP (need cheap backup GKP)
            # But keep some reasonable bench options
            if pos != "GKP":
                if cost < 40:  # Less than £4.0m
                    continue
                # Skip players with very low EP and minutes (not viable bench options)
                if cost <= 45 and (ep_base < 1.5 or xmins < 30):
                    continue
            
            # Calculate value score
            value_score = ep_base / (cost / 10) if cost > 0 else 0
            
            # Check if differential
            ownership = float(e.get("selected_by_percent", 0))
            is_differential = ownership < 10.0 and ep_base > 3.0
            
            # Risk assessments
            injury_risk = 0.1 if e.get("status") != "a" else 0.0
            rotation_risk = max(0, (90 - xmins) / 90) if xmins < 60 else 0.0
            
            # Apply FDR adjustment to expected points over horizon
            fdr_factor = fdr_map.get(pid, 1.0)  # Default to 1.0 if no FDR data
            # FDR factor typically ranges from 0.8 (hard fixtures) to 1.2 (easy fixtures)
            # For simplicity, apply a scaled version across the horizon
            adjusted_eph = ep_base * horizon * (0.9 + 0.1 * fdr_factor)  # Scale FDR impact
            
            players.append(Player(
                id=pid,
                name=e.get("web_name", f"Player_{pid}"),
                pos=pos,
                team=team_id,
                cost=cost,
                xmins=xmins,
                ep_base=ep_base,
                xgi90=float(xgi_map.get(pid, 0)),
                team_att=float(team_att_map.get(pid, 1.0)),
                ep_seq=[ep_base * (0.9 + 0.1 * fdr_factor) for _ in range(horizon)],
                ep1=ep_base,
                eph=adjusted_eph,
                form=float(e.get("form", 0)),
                selected_by=ownership,
                value_score=value_score,
                is_differential=is_differential,
                injury_risk=injury_risk,
                rotation_risk=rotation_risk
            ))
        
        if not players:
            return {"error": "No valid players found"}
        
        # Create LP problem
        prob = pulp.LpProblem("FPL_Squad", pulp.LpMaximize)
        
        # Decision variables
        squad_vars = {p.id: pulp.LpVariable(f"squad_{p.id}", cat="Binary") for p in players}
        xi_vars = {p.id: pulp.LpVariable(f"xi_{p.id}", cat="Binary") for p in players}
        captain_var = {p.id: pulp.LpVariable(f"cap_{p.id}", cat="Binary") for p in players}
        
        # Objective function
        objective_terms = []
        for p in players:
            # Base expected points
            ep_contrib = p.eph
            
            # Value bonus
            value_bonus = p.value_score * value_weight * 10
            
            # Differential bonus
            diff_bonus = differential_bonus * ep_contrib if p.is_differential else 0
            
            # Risk penalties
            risk_pen = (p.injury_risk + p.rotation_risk) * risk_penalty * ep_contrib
            
            # XI bonus (starters worth more)
            xi_bonus = ep_contrib * 0.2
            
            # Captain bonus (double points)
            cap_bonus = p.ep1
            
            # Only count EP for XI players, not squad
            # Bench players contribute nothing to objective
            objective_terms.append(
                xi_vars[p.id] * (ep_contrib + value_bonus + diff_bonus - risk_pen) +
                captain_var[p.id] * cap_bonus +
                squad_vars[p.id] * 0.01  # Tiny value for squad diversity
            )
        
        prob += pulp.lpSum(objective_terms)
        
        # Constraints
        # Squad size = 15
        prob += pulp.lpSum([squad_vars[p.id] for p in players]) == 15
        
        # Starting XI = 11
        prob += pulp.lpSum([xi_vars[p.id] for p in players]) == 11
        
        # XI must be in squad
        for p in players:
            prob += xi_vars[p.id] <= squad_vars[p.id]
        
        # Exactly 1 captain
        prob += pulp.lpSum([captain_var[p.id] for p in players]) == 1
        
        # Captain must be in XI
        for p in players:
            prob += captain_var[p.id] <= xi_vars[p.id]
        
        # Position constraints
        for pos, count in POSITIONS.items():
            prob += pulp.lpSum([squad_vars[p.id] for p in players if p.pos == pos]) == count
        
        # Formation constraints (exactly one valid formation)
        formation_vars = {f: pulp.LpVariable(f"form_{f}", cat="Binary") for f in FORMATIONS}
        prob += pulp.lpSum(formation_vars.values()) == 1
        
        # DEBUG: Force 442 to test
        # prob += formation_vars["442"] == 1
        
        for formation, reqs in FORMATIONS.items():
            for pos, required in reqs.items():
                prob += pulp.lpSum([xi_vars[p.id] for p in players if p.pos == pos]) >= required - 100 * (1 - formation_vars[formation])
                prob += pulp.lpSum([xi_vars[p.id] for p in players if p.pos == pos]) <= required + 100 * (1 - formation_vars[formation])
        
        # Budget constraint
        prob += pulp.lpSum([squad_vars[p.id] * p.cost for p in players]) <= BUDGET
        
        # Bench budget constraint
        bench_cost = pulp.lpSum([squad_vars[p.id] * p.cost for p in players]) - pulp.lpSum([xi_vars[p.id] * p.cost for p in players])
        prob += bench_cost <= bench_budget
        
        # Minimum bench quality - at least one bench player should have decent EP
        # This ensures viable substitution options
        bench_quality_vars = {p.id: pulp.LpVariable(f"bench_{p.id}", cat="Binary") for p in players}
        for p in players:
            # Player is on bench if in squad but not in XI
            prob += bench_quality_vars[p.id] <= squad_vars[p.id]
            prob += bench_quality_vars[p.id] <= 1 - xi_vars[p.id]
            prob += bench_quality_vars[p.id] >= squad_vars[p.id] - xi_vars[p.id]
        
        # At least 2 bench players should have EP > 1.5 (viable options)
        viable_bench = [p for p in players if p.ep_base > 1.5 and p.pos != "GKP"]
        if viable_bench:
            prob += pulp.lpSum([bench_quality_vars[p.id] for p in viable_bench]) >= 2
        
        # Max 3 per club
        for team_id in set(p.team for p in players):
            prob += pulp.lpSum([squad_vars[p.id] for p in players if p.team == team_id]) <= MAX_PER_CLUB
        
        # Transfer constraints if current squad provided
        if current_squad and not wildcard:
            for p in players:
                if p.id in current_squad:
                    # Encourage keeping current players
                    prob += squad_vars[p.id] >= 0.8
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Debug: Check which formation was selected and why
        selected_formation = None
        for f, var in formation_vars.items():
            if var.varValue and var.varValue > 0.5:
                selected_formation = f
                log.info(f"Selected formation: {f}")
                
                # Calculate what the XI EP would be for different formations
                xi_players = [p for p in players if xi_vars[p.id].varValue and xi_vars[p.id].varValue > 0.5]
                total_xi_ep = sum(p.ep1 for p in xi_players)
                log.info(f"Total XI EP with {f}: {total_xi_ep:.2f}")
                
                # Show position breakdown
                pos_breakdown = {}
                for p in xi_players:
                    pos_breakdown[p.pos] = pos_breakdown.get(p.pos, 0) + 1
                log.info(f"Position breakdown: {pos_breakdown}")
                break
        
        if prob.status != pulp.LpStatusOptimal:
            return {"error": f"Optimization failed with status: {pulp.LpStatus[prob.status]}"}
        
        # Extract solution
        squad = []
        xi = []
        captain_id = None
        
        for p in players:
            if squad_vars[p.id].varValue > 0.5:
                squad.append(p)
                if xi_vars[p.id].varValue > 0.5:
                    xi.append(p)
                if captain_var[p.id].varValue > 0.5:
                    captain_id = p.id
        
        # Sort XI by position
        xi.sort(key=lambda p: ["GKP", "DEF", "MID", "FWD"].index(p.pos))
        
        # Determine formation
        pos_counts = {"GKP": 0, "DEF": 0, "MID": 0, "FWD": 0}
        for p in xi:
            pos_counts[p.pos] += 1
        
        formation = "unknown"
        for form, reqs in FORMATIONS.items():
            if all(pos_counts[pos] == count for pos, count in reqs.items()):
                formation = form
                break
        
        # Format output
        bench = [p for p in squad if p not in xi]
        bench.sort(key=lambda p: -p.ep1)  # Best bench players first
        
        # Find vice captain (best non-captain in XI)
        vice = None
        for p in xi:
            if p.id != captain_id:
                if vice is None or p.ep1 > vice.ep1:
                    vice = p
        
        total_cost = sum(p.cost for p in squad)
        bench_value = sum(p.cost for p in bench)
        expected_points = sum(p.ep1 for p in xi) + (next(p.ep1 for p in xi if p.id == captain_id) if captain_id else 0)
        
        # Build human-readable output
        output = []
        output.append(f"Suggested 15-man squad (LP Optimized), formation {formation}:")
        output.append(f"Expected Points (GW1): {expected_points:.1f}")
        output.append(f"Optimization Score: {pulp.value(prob.objective):.1f}")
        output.append("")
        
        output.append("Full Squad:")
        for p in squad:
            status = ""
            if p.id == captain_id:
                status = " (C)"
            elif vice and p.id == vice.id:
                status = " (V)"
            output.append(f" - {p.name} ({p.pos}) — ep1 {p.ep1:.2f}, eph {p.eph:.2f}, xMins {p.xmins:.0f}, £{p.cost/10:.1f}m{status}")
        
        output.append(f"\nStarting XI:")
        for p in xi:
            status = ""
            if p.id == captain_id:
                status = " (C)"
            elif vice and p.id == vice.id:
                status = " (V)"
            output.append(f" - {p.name} ({p.pos}) — ep1 {p.ep1:.2f}, eph {p.eph:.2f}, xMins {p.xmins:.0f}, £{p.cost/10:.1f}m{status}")
        
        output.append(f"\nBench:")
        for i, p in enumerate(bench, 1):
            output.append(f"{i}. {p.name} ({p.pos}) — {p.ep1:.2f} pts")
        
        output.append(f"\nBench value: £{bench_value/10:.1f}m (target max £{bench_budget/10:.1f}m)")
        if captain_id:
            cap = next(p for p in xi if p.id == captain_id)
            output.append(f"Captain: {cap.name} ({cap.pos}) - {cap.ep1:.2f} pts")
        
        return {
            "squad": [
                {
                    "id": p.id,
                    "name": p.name,
                    "pos": p.pos,
                    "team": p.team,
                    "cost": p.cost / 10,
                    "ep1": p.ep1,
                    "eph": p.eph,
                    "xmins": p.xmins,
                    "is_captain": p.id == captain_id,
                    "is_vice": p.id == vice.id if vice else False,
                    "in_xi": p in xi
                }
                for p in squad
            ],
            "formation": formation,
            "total_cost": total_cost / 10,
            "bench_value": bench_value / 10,
            "expected_points_gw1": expected_points,
            "optimization_score": pulp.value(prob.objective),
            "solver": "Linear Programming (PuLP)",
            "human_readable": "\n".join(output)
        }
        
    except Exception as e:
        log.error(f"LP optimization error: {e}")
        return {"error": str(e)}


# ------------------- public API -------------------
def optimize_transfers(
    use_myteam: bool = False,               # (kept for CLI compat)
    horizon: int = 5,
    bench_weight: float = 0.10,             # (unused in greedy path)
    bench_budget: int = 180,                # <-- HARD CAP (tenths of a million)
    formations: str = "343,352,442,451,433",
    nonstarter_xmins: float = 20.0,
    gk_backup_xmins: float = 0.0,
    captain_positions: str = "MID,FWD",
    vice_positions: str = "MID,FWD,DEF,GKP",
    bench_min_xmins: float = 45.0,          # (depth trimming handles this loosely)
    use_model_ep: bool = True,
    fdr_weight: float = 0.10,
    hweights: str = "",
    explain: bool = True,
    json_out: Optional[str] = None,
    use_advanced: bool = False,             # Use LP-based optimization
    **kwargs,
) -> Dict[str, Any]:
    """
    Squad optimizer. Uses advanced LP solver if use_advanced=True,
    otherwise falls back to greedy approach.
    """
    # Use LP optimizer if requested and available
    if use_advanced and HAS_PULP:
        log.info("Using advanced LP-based optimization")
        
        result = optimize_with_lp(
            horizon=horizon,
            bench_weight=bench_weight,
            bench_budget=bench_budget,
            formations=formations,
            differential_bonus=kwargs.get('differential_bonus', 0.1),
            risk_penalty=kwargs.get('risk_penalty', 0.05),
            value_weight=kwargs.get('value_weight', 0.3),
            current_squad=kwargs.get('current_squad'),
            max_transfers=kwargs.get('max_transfers', 2),
            wildcard=kwargs.get('wildcard', False)
        )
        
        if result and "error" not in result:
            return result
        else:
            if result:
                log.warning(f"LP optimization failed: {result.get('error', 'Unknown error')}. Falling back to greedy.")
            else:
                log.warning("LP optimization returned None. Falling back to greedy.")
    elif use_advanced and not HAS_PULP:
        log.warning("PuLP not installed. Install with 'pip install pulp' for LP optimization. Using greedy approach.")
    
    # Original greedy implementation
    js = _fetch_bootstrap()
    xmins_map = _load_xmins()

    H = max(1, int(horizon))
    weights = _parse_weights(hweights, H)
    if explain:
        wtxt = ",".join(f"{w:.2f}" for w in weights)
        log.info(
            "Objective: next %d GWs, weights=[%s], bench_weight=%.2f, fdr_weight=%.2f, bench_budget=£%.1fm",
            H, wtxt, bench_weight, fdr_weight, bench_budget / 10.0
        )

    pool = _candidate_pool(js, xmins_map, H=H, weights=weights, fdr_weight=fdr_weight)
    _adjust_xmins_with_depth(pool, nonstarter_xmins=float(nonstarter_xmins), gk_backup_xmins=float(gk_backup_xmins))

    if not pool:
        return {"human_readable": "No candidates found from FPL bootstrap.", "picked_ids": []}

    # Build a cheap bench upfront (guarantees cap if feasible)
    bench_template = _choose_cheap_bench_template(pool, bench_budget)
    bt_cost = _total_spent(bench_template)
    if explain:
        log.info("Bench template cost: £%.1fm (cap £%.1fm)", bt_cost / 10.0, bench_budget / 10.0)

    allowed_forms = [f.strip() for f in formations.split(",") if f.strip() in FORMATIONS]
    if not allowed_forms:
        allowed_forms = ["343", "352", "442"]

    picked = _pick15_with_bench_template(pool, bench_template)
    xi, bench, form = _choose_xi_and_bench(picked, allowed_forms, bench_budget, pool=pool)

    # Final safety: if still over cap (should be rare), force cheapest same-pos replacements until ≤ cap
    if _bench_cost(bench) > bench_budget:
        bench_sorted_idx = sorted(range(len(bench)), key=lambda i: bench[i].cost, reverse=True)
        clubs = _club_counts(picked)
        picked_ids = {p.id for p in picked}
        for idx in bench_sorted_idx:
            if _bench_cost(bench) <= bench_budget:
                break
            b = bench[idx]
            candidates = [q for q in pool if (q.pos == b.pos and q.id not in picked_ids)]
            candidates.sort(key=lambda z: (z.cost, -z.ep1))
            for q in candidates:
                if q.cost >= b.cost:
                    break
                clubs_after = dict(clubs)
                clubs_after[b.team] = clubs_after.get(b.team, 0) - 1
                if clubs_after[b.team] <= 0:
                    clubs_after.pop(b.team, None)
                clubs_after[q.team] = clubs_after.get(q.team, 0) + 1
                if clubs_after[q.team] > MAX_PER_CLUB:
                    continue
                if (_total_spent(picked) - b.cost + q.cost) > BUDGET:
                    continue
                picked = [q if (pp.id == b.id) else pp for pp in picked]
                picked_ids = {p.id for p in picked}
                clubs = clubs_after
                xi = _xi_for_formation(picked, form)
                xi, bench = _repair_bench_budget(xi, picked, bench_budget)
                break

    # captain/vice by ep1 * availability
    # Simplified: Pick highest EP for captain, second highest for vice (considering position restrictions)
    def p_play(p: Player) -> float:
        return min(1.0, max(0.0, p.xmins / 60.0))

    allowed_c_pos = {p.strip().upper() for p in captain_positions.split(",") if p.strip()} or {"MID", "FWD"}
    allowed_v_pos = {p.strip().upper() for p in vice_positions.split(",") if p.strip()} or {"MID", "FWD", "DEF"}

    # Sort XI by expected points * playing probability
    xi_sorted = sorted(xi, key=lambda p: p.ep1 * p_play(p), reverse=True)
    
    # Pick captain from allowed positions
    cap = None
    for p in xi_sorted:
        if p.pos in allowed_c_pos:
            cap = p
            break
    if cap is None:
        cap = xi_sorted[0]  # Fallback to highest EP overall
    
    # Pick vice from allowed positions (excluding captain)
    vice = None
    for p in xi_sorted:
        if p.id != cap.id and p.pos in allowed_v_pos:
            vice = p
            break
    if vice is None:
        # Fallback to second highest EP overall
        for p in xi_sorted:
            if p.id != cap.id:
                vice = p
                break

    def fmt(p: Player) -> str:
        return f" - {p.name} ({p.pos}) — ep1 {p.ep1:.2f}, next{H} {p.eph:.2f}, xMins {p.xmins:.0f}, £{p.cost/10:.1f}m"

    human = []
    human.append(f"Suggested 15-man squad (budget 100.0m), formation {form}:")
    for p in picked:
        human.append(fmt(p))
    human.append("")
    human.append("Starting XI:")
    for p in xi:
        human.append(fmt(p))
    human.append("")
    human.append("Bench:")
    for p in bench:
        human.append(fmt(p))
    human.append("")
    human.append(f"Bench spend: £{_bench_cost(bench)/10:.1f}m (cap £{bench_budget/10:.1f}m)")
    human.append(f"Captain: {cap.name} ({cap.pos})")
    human.append(f"Vice: {vice.name} ({vice.pos})")

    return {
        "human_readable": "\n".join(human),
        "picked_ids": [p.id for p in picked],
        "xi_ids": [p.id for p in xi],
        "bench_ids": [p.id for p in bench],
        "formation": form,
        "bench_cost": _bench_cost(bench) / 10.0,
        "bench_budget": bench_budget / 10.0,
        "captain_id": cap.id,
        "vice_id": vice.id,
        "table": [
            {"player_id": p.id, "name": p.name, "pos": p.pos, "team_id": p.team,
             "cost": p.cost/10.0, "xmins": p.xmins, "ep1": p.ep1, f"next{H}": p.eph}
            for p in picked
        ],
    }