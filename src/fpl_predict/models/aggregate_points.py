from __future__ import annotations
import pandas as pd

# ---------- helpers ----------
def _rget(d: dict, *candidates, default=0):
    """
    Try a list of candidate keys or key-paths in rules dict.
    Each candidate can be:
      - a single string key, e.g. "assist"
      - a tuple path, e.g. ("goals", "GKP")
    Returns the first found non-dict value, else `default`.
    """
    for cand in candidates:
        cur = d
        if isinstance(cand, tuple):
            ok = True
            for k in cand:
                if isinstance(cur, dict) and k in cur:
                    cur = cur[k]
                else:
                    ok = False
                    break
            if ok and not isinstance(cur, dict):
                return cur
        else:
            if isinstance(cur, dict) and cand in cur and not isinstance(cur[cand], dict):
                return cur[cand]
    return default


def _goal_points_for_pos(rules: dict, pos: str) -> float:
    p = pos.upper()
    return float(_rget(
        rules,
        # your snapshot: rules["goals"]["GKP"/"DEF"/"MID"/"FWD"]
        ("goals", p),
        # fallbacks we support
        f"goal_{p.lower()}",
        f"goals_{p.lower()}",
        ("goal_points", p),
        ("goal_points_by_position", p),
        default=0,
    ))


def _cs_points_for_pos(rules: dict, pos: str) -> float:
    p = pos.upper()
    return float(_rget(
        rules,
        # your snapshot: flat key "clean_sheet" (may be dict-by-pos or scalar)
        ("clean_sheet", p),
        "clean_sheet",
        # fallbacks
        ("clean_sheets", p),
        f"cs_{p.lower()}",
        f"clean_sheet_{p.lower()}",
        ("clean_sheet_points", p),
        default=0,
    ))


def _assist_points(rules: dict) -> float:
    # your snapshot includes top-level "assist"
    return float(_rget(rules, "assist", "assists", "assist_points", default=0))


def _appearance_points(rules: dict) -> tuple[float, float]:
    """
    Returns (app1, app2): 1pt for any appearance, +1 for >=60 (so 2 total if 60+).
    Your snapshot groups this under rules["minutes"].
    We support multiple common shapes.
    """
    minutes = rules.get("minutes", {})
    # try common nested keys first
    app1 = _rget(minutes, "any", "played_any", "appearance_any", default=None)
    app2 = _rget(minutes, "sixty", "played_60", "appearance_60", default=None)

    # fallbacks to flat names if not found nested
    if app1 is None:
        app1 = _rget(rules, "appearance_1pt", "appearance_1", "minutes_played_1pt", default=0)
    if app2 is None:
        app2 = _rget(rules, "appearance_2pts", "appearance_2", "minutes_played_60", default=0)

    return float(app1 or 0), float(app2 or 0)
# --------------------------------------------


def expected_points_df(
    feats: pd.DataFrame,
    xmins: pd.Series,
    mu_g: pd.Series,
    mu_a: pd.Series,
    p_cs: pd.Series | float = 0.0,
) -> pd.DataFrame:
    # lazy import to avoid cycles
    from ..data.rules_fetcher import fetch_scoring_rules
    rules = fetch_scoring_rules()

    # Appearance expectations from minutes
    p60 = (xmins >= 60).astype(float)
    p1  = (xmins > 0).astype(float)

    # Points per position
    pos = feats["position"].astype(str).str.upper()
    g_pts = pos.map(lambda p: _goal_points_for_pos(rules, p)).astype(float)
    cs_pts = pos.map(lambda p: _cs_points_for_pos(rules, p)).astype(float)

    a_pts = _assist_points(rules)
    app1, app2 = _appearance_points(rules)

    # Clean-sheet probability may be a scalar or a Series
    if not isinstance(p_cs, pd.Series):
        p_cs = pd.Series(p_cs, index=feats.index)

    # Use real defensive stats from FPL API when available, else estimate
    def get_defensive_contribution(row_pos, row_xmins, player_id=None):
        """Get expected defensive contribution points per game"""
        if row_xmins < 60:
            return 0  # Need significant minutes for defensive contributions
        
        # Try to load real defensive stats
        try:
            import os
            if os.path.exists('data/processed/defensive_stats.parquet'):
                def_stats = pd.read_parquet('data/processed/defensive_stats.parquet')
                if player_id is not None and player_id in def_stats['player_id'].values:
                    player_stats = def_stats[def_stats['player_id'] == player_id].iloc[0]
                    # Use real data if we have enough sample size
                    if player_stats['minutes'] > 500 and player_stats['expected_defensive_points'] > 0:
                        # Scale by expected minutes
                        return player_stats['expected_defensive_points'] * (row_xmins / 90.0)
        except:
            pass
        
        # Fallback to estimates based on position and role
        # DEF need 10 CBIT, MID/FWD need 12 CBIRT for 2 points
        defensive_rates = {
            'DEF': 6.0,  # ~1.5 games per bonus (10/6.0)
            'MID': 3.0,  # ~4 games per bonus (12/3.0) 
            'FWD': 1.0,  # ~12 games per bonus
            'GKP': 0.0   # Goalkeepers don't get defensive contributions
        }
        
        cbit_per90 = defensive_rates.get(row_pos, 0)
        threshold = 10 if row_pos == 'DEF' else 12
        
        # Probability of hitting threshold in a game
        prob_per_game = min(cbit_per90 / threshold, 0.5)  # Cap at 50% chance
        
        # Points = 2 * probability * (xmins/90 to adjust for actual playing time)
        return 2.0 * prob_per_game * (row_xmins / 90.0)
    
    # Apply defensive contribution estimates
    # This is a rough estimate until we get real data
    try:
        # Check if we have ICT threat data to refine estimates
        from ..data.fpl_api import get_bootstrap
        bs = get_bootstrap()
        players_df = pd.DataFrame(bs['elements'])[['id', 'threat', 'minutes']]
        players_df = players_df.rename(columns={'id': 'player_id'})
        
        # Low threat players (CDMs) get bonus for more defensive actions
        players_df['threat_per90'] = (players_df['threat'] / players_df['minutes'].replace(0, 1)) * 90
        threat_data = feats[['player_id']].merge(players_df[['player_id', 'threat_per90']], 
                                                on='player_id', how='left')
        
        # CDMs and defensive players get higher defensive contribution
        is_defensive = (pos == 'MID') & (threat_data['threat_per90'].fillna(50) < 35)
        defensive_bonus = pd.Series(0.0, index=feats.index)
        
        for i in feats.index:
            player_id = feats.loc[i, 'player_id'] if 'player_id' in feats.columns else None
            base_contrib = get_defensive_contribution(pos.iloc[i], xmins.iloc[i], player_id)
            # CDMs get 50% more defensive contributions
            if is_defensive.iloc[i]:
                base_contrib *= 1.5
            defensive_bonus.iloc[i] = base_contrib
    except:
        # Fallback to simple position-based estimate
        defensive_bonus = pd.Series([
            get_defensive_contribution(p, xm, pid if 'player_id' in feats.columns else None) 
            for p, xm, pid in zip(pos, xmins, feats['player_id'] if 'player_id' in feats.columns else [None]*len(pos))
        ], index=feats.index)
    
    ep = (
        p1 * app1 +
        p60 * app2 +
        mu_g.astype(float) * g_pts +
        mu_a.astype(float) * float(a_pts) +
        p_cs.astype(float) * cs_pts +
        defensive_bonus  # Add defensive contribution points
    )

    out = pd.DataFrame({"player_id": feats["player_id"].values, "ep_model": ep.values})
    return out