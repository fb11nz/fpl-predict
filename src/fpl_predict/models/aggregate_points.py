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

    # Calculate defensive contribution points using real DC data
    def get_defensive_contribution_points(row):
        """Get expected defensive contribution points based on actual DC stats"""
        row_pos = row.get('position', '')
        row_xmins = row.get('xmins', 0)
        
        if row_xmins < 30:
            return 0  # Need minutes for defensive contributions
        
        # Use actual DC data if available
        dc_per90 = row.get('dc_per90', 0)
        dc_l5 = row.get('dc_l5', 0)
        
        # Thresholds: DEF need 10 CBIT, MID/FWD need 12 CBIRT for 2 points
        threshold = 10 if row_pos == 'DEF' else 12
        
        # Use recent form (dc_l5) if available, else use per90 rate
        dc_rate = dc_l5 if dc_l5 > 0 else dc_per90
        
        if dc_rate > 0:
            # Calculate probability of hitting threshold per game
            # DC includes all defensive actions (CBIT + recoveries + tackles)
            prob_per_game = min(dc_rate / threshold, 0.8)  # Cap at 80% chance
            
            # Expected points = 2 * probability * (xmins/90)
            return 2.0 * prob_per_game * (row_xmins / 90.0)
        
        # Fallback estimates if no DC data yet
        default_rates = {
            'DEF': 0.3,  # ~30% chance per game
            'MID': 0.15,  # ~15% chance per game
            'FWD': 0.05,  # ~5% chance per game
            'GKP': 0.0   # No DC points for goalkeepers
        }
        
        default_prob = default_rates.get(row_pos, 0)
        return 2.0 * default_prob * (row_xmins / 90.0)
    
    # Calculate defensive contribution bonus using actual DC data
    defensive_bonus = pd.Series(0.0, index=feats.index)
    
    # Create a DataFrame with the data we need for DC calculation
    dc_data = pd.DataFrame({
        'position': pos,
        'xmins': xmins,
        'dc_per90': feats.get('dc_per90', 0),
        'dc_l5': feats.get('dc_l5', 0),
        'dc_l3': feats.get('dc_l3', 0)
    })
    
    # Apply the DC calculation to each row
    defensive_bonus = dc_data.apply(get_defensive_contribution_points, axis=1)
    
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