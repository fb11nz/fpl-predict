from __future__ import annotations

import os
import numpy as np
import pandas as pd

from .features import load_model_features
from .minutes import MinutesModel
from .xgi import XGIModel
from .cleansheets import CleanSheetModel
from .ensemble import EnsembleModel, StackedEnsemble
from .feature_engineering import AdvancedFeatureEngineer
from .cards import CardsModel
from .aggregate_points import expected_points_df
from ..utils.cache import PROC
from ..utils.io import write_parquet, read_parquet
from ..utils.logging import get_logger
from ..data.fpl_api import get_bootstrap

log = get_logger(__name__)

# Configuration
FDR_GAMMA = 0.80
CS_PRIOR_DEF_GKP = 0.34
BETA_MIN = 0.35
BETA_MAX = 0.75
PREMIUM_COST = 110
PREMIUM_SEL = 20.0
PREMIUM_UPLIFT = 0.05

# Feature set
XCOLS = [
    "mins_l5", "mins_l3", "starts_l5", "bench_l5",
    "form", "selected_by_percent", "chance_next", "now_cost",
    "prev_minutes", "prev_starts", "prev_xg_per90", "prev_xa_per90",
    # Defensive contribution features (NEW!)
    "dc_l5", "dc_l3", "dc_per90", "dc_total",
    "tackles_l5", "tackles_total", "cbi_l5", "cbi_total", 
    "recoveries_l5", "recoveries_total",
    # ACTUAL PERFORMANCE DATA (CRITICAL!)
    "goals_l5", "assists_l5", "goals_l3", "assists_l3",
    "xg_l5", "xa_l5", "xg_l3", "xa_l3",
    "bonus_l5", "bps_l5", "ict_l5",
    "clean_sheets_l5", "goals_conceded_l5", "saves_l5",
    "goals_per90", "assists_per90", "xg_per90", "xa_per90",
    "home_goals", "away_goals", "home_xg", "away_xg",
    "total_minutes", "total_starts", "avg_points", "form_trend",
    # Additional features from advanced engineering
    "team_offensive_strength_normalized", "team_defensive_strength_normalized",
    "mins_trend", "consistency_score", "starter_confidence", "fitness_factor",
    "next5_fdr", "value_score", "active_form", "team_form"
]

PRIORS_XG90 = {
    "GKP": {0: 0.00},
    "DEF": {0: 0.04, 55: 0.05, 65: 0.06, 75: 0.07},
    "MID": {0: 0.15, 75: 0.22, 95: 0.28, 105: 0.32, 110: 0.35},
    "FWD": {0: 0.25, 70: 0.35, 85: 0.45, 100: 0.55, 110: 0.60},
}
PRIORS_XA90 = {
    "GKP": {0: 0.00},
    "DEF": {0: 0.05, 55: 0.06, 65: 0.07, 75: 0.08},
    "MID": {0: 0.10, 75: 0.15, 95: 0.20, 105: 0.22, 110: 0.25},
    "FWD": {0: 0.05, 70: 0.07, 85: 0.10, 100: 0.12, 110: 0.14},
}


def _build_X_advanced(feats: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix with all available features."""
    X = pd.DataFrame(index=feats.index)
    
    # Include all numeric columns except targets
    exclude_cols = {'player_id', 'minutes', 'goals', 'assists', 'clean_sheets', 
                   'goals_conceded', 'yellow_cards', 'red_cards', 'bonus', 'total_points'}
    
    for col in feats.columns:
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(feats[col]):
            X[col] = feats[col].fillna(0)
    
    # Ensure we have the essential columns
    for c in XCOLS:
        if c not in X.columns and c in feats.columns:
            X[c] = feats[c].fillna(0)
        elif c not in X.columns:
            X[c] = 0.0
    
    return X


def _tier_prior(prior_map: dict[int, float], cost: float) -> float:
    """Step function over cost (tenths of a million)."""
    keys = sorted(prior_map.keys())
    v = prior_map[keys[0]]
    for k in keys:
        if cost >= k:
            v = prior_map[k]
        else:
            break
    return float(v)


def _apply_xgi_priors_if_needed(feats: pd.DataFrame, mu_g: pd.Series, mu_a: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Apply FPL EP_next as baseline for new signings or low-signal players."""
    from ..data.fpl_api import get_bootstrap
    
    pos = feats.get("position", pd.Series("", index=feats.index)).astype(str)
    cost = feats.get("now_cost", pd.Series(0.0, index=feats.index)).fillna(0.0)
    prev_mins = feats.get("prev_minutes", pd.Series(0.0, index=feats.index)).fillna(0.0)
    mins_l5 = feats.get("mins_l5", pd.Series(0.0, index=feats.index)).fillna(0.0)
    mins_l3 = feats.get("mins_l3", pd.Series(0.0, index=feats.index)).fillna(0.0)

    # Identify players needing FPL EP baseline
    likely_new = (prev_mins < 300) & ((mins_l5 + mins_l3) < 120)
    near_zero = (mu_g.fillna(0).abs() + mu_a.fillna(0).abs()) < 1e-6
    needs_fpl_baseline = likely_new | near_zero

    if not needs_fpl_baseline.any():
        return mu_g, mu_a

    # Get FPL's expected points as baseline
    try:
        bootstrap = get_bootstrap()
        fpl_players = {p['id']: p for p in bootstrap['elements']}
    except Exception:
        log.warning("Could not fetch FPL data, falling back to price priors")
        # Fallback to original price-based priors
        mu_g2 = mu_g.copy()
        mu_a2 = mu_a.copy()
        for idx in feats.index[needs_fpl_baseline]:
            p = pos.at[idx] if idx in pos.index else ""
            c = float(cost.at[idx]) if idx in cost.index else 0.0
            p = p if p in PRIORS_XG90 else "MID"
            mu_g2.at[idx] = _tier_prior(PRIORS_XG90[p], c)
            mu_a2.at[idx] = _tier_prior(PRIORS_XA90[p], c)
        return mu_g2, mu_a2

    mu_g2 = mu_g.copy()
    mu_a2 = mu_a.copy()

    for idx in feats.index[needs_fpl_baseline]:
        player_id = feats.at[idx, 'player_id'] if 'player_id' in feats.columns else None
        
        if player_id and int(player_id) in fpl_players:
            fpl_player = fpl_players[int(player_id)]
            fpl_ep = float(fpl_player.get('ep_next', 0))
            position = pos.at[idx] if idx in pos.index else ""
            
            # Convert FPL EP to goals/assists estimates
            # Use conservative conversion: assume EP comes 60% from goals/assists, 40% from other sources
            if position == "FWD":
                # Forwards: 70% goals, 30% assists of attacking points
                attacking_points = fpl_ep * 0.6
                estimated_goals = (attacking_points * 0.7) / 4  # FWD goals worth 4pts
                estimated_assists = (attacking_points * 0.3) / 3  # Assists worth 3pts
            elif position == "MID":
                # Midfielders: 50% goals, 50% assists of attacking points  
                attacking_points = fpl_ep * 0.6
                estimated_goals = (attacking_points * 0.5) / 5  # MID goals worth 5pts
                estimated_assists = (attacking_points * 0.5) / 3
            elif position == "DEF":
                # Defenders: 30% goals, 70% assists of attacking points
                attacking_points = fpl_ep * 0.3  # Less attacking expected
                estimated_goals = (attacking_points * 0.3) / 6  # DEF goals worth 6pts
                estimated_assists = (attacking_points * 0.7) / 3
            else:  # GKP
                estimated_goals = 0.0
                estimated_assists = 0.0
            
            # Apply balanced weighting for new players:
            # 50% current model prediction + 50% FPL EP estimate (less harsh than 80/20)
            if mu_g.at[idx] == 0 and mu_a.at[idx] == 0:
                # Pure new player - use FPL baseline more heavily
                mu_g2.at[idx] = 0.3 * mu_g.at[idx] + 0.7 * estimated_goals
                mu_a2.at[idx] = 0.3 * mu_a.at[idx] + 0.7 * estimated_assists
            else:
                # Has some model signal - balanced approach
                mu_g2.at[idx] = 0.5 * mu_g.at[idx] + 0.5 * estimated_goals
                mu_a2.at[idx] = 0.5 * mu_a.at[idx] + 0.5 * estimated_assists
        else:
            # Fallback to price priors if no FPL data
            p = pos.at[idx] if idx in pos.index else ""
            c = float(cost.at[idx]) if idx in cost.index else 0.0
            p = p if p in PRIORS_XG90 else "MID"
            mu_g2.at[idx] = _tier_prior(PRIORS_XG90[p], c)
            mu_a2.at[idx] = _tier_prior(PRIORS_XA90[p], c)

    return mu_g2, mu_a2


def _team_attack_strength(boot: dict) -> dict[int, float]:
    """Build normalized team-attack index from bootstrap strengths."""
    teams = boot.get("teams", [])
    vals = []
    by_id = {}
    for t in teams:
        tid = int(t["id"])
        ah = float(t.get("strength_attack_home") or 0)
        aa = float(t.get("strength_attack_away") or 0)
        v = (ah + aa) / 2.0 if (ah or aa) else float(t.get("strength_attack", 0) or 0)
        by_id[tid] = v
        vals.append(v)
    mean = np.mean([v for v in vals if v > 0]) if vals else 1.0
    if mean <= 0:
        mean = 1.0
    return {tid: (v / mean if mean else 1.0) for tid, v in by_id.items()}




def train_all() -> dict:
    """
    Train lightweight baseline models (minutes, xGI, clean sheets) and
    produce model-based expected points for the next GW with less-conservative calibration.

    Enhancements:
      - Dynamic ep_blend weights: more FPL for uncertain players; more model for stable ones.
      - xGI priors from price/position for new signings or near-zero model signal.
      - Global calibration to counter systematic underestimation.
      - Slightly stronger FDR/CS effects to reflect real fixtures/defense impacts.
      - Persist xgi90_est and team_att for role-aware optimizer adjustments.
    """
    log.info("Starting advanced model training pipeline with 2025-26 season data (including GW1 results)")
    
    # Load features
    feats = load_model_features()
    
    if feats is None or len(feats) == 0:
        log.warning("No features found for training")
        return {"minutes": MinutesModel(), "xgi": XGIModel(), 
                "cs": CleanSheetModel(), "cards": CardsModel()}

    # Initialize feature engineer
    feature_engineer = AdvancedFeatureEngineer()
    
    # Apply advanced feature engineering
    try:
        fixtures_df = read_parquet(PROC / "fixtures.parquet")
        feats_enhanced = feature_engineer.fit_transform(feats, fixtures_df)
        log.info(f"Enhanced features from {len(feats.columns)} to {len(feats_enhanced.columns)} columns")
    except Exception as e:
        log.warning(f"Feature engineering failed: {e}, using basic features")
        feats_enhanced = feats
    
    # Build feature matrix and targets from ACTUAL MATCH DATA
    X = _build_X_advanced(feats_enhanced)
    
    # Use real training targets instead of non-existent columns
    y_mins = feats["target_minutes"].fillna(0) if "target_minutes" in feats.columns else pd.Series(0.0, index=feats.index)
    y_g = feats["target_goals"].fillna(0) if "target_goals" in feats.columns else pd.Series(0.0, index=feats.index)
    y_a = feats["target_assists"].fillna(0) if "target_assists" in feats.columns else pd.Series(0.0, index=feats.index)
    
    # Log training data quality
    log.info(f"Training targets: {(y_g > 0).sum()} players with goals, "
             f"{(y_a > 0).sum()} with assists, {(y_mins > 0).sum()} with minutes")
    log.info(f"Total goals in training: {y_g.sum():.1f}, assists: {y_a.sum():.1f}, "
             f"avg minutes: {y_mins.mean():.1f}")

    # Initialize models
    m_mins = EnsembleModel(use_models=['xgboost', 'lightgbm', 'ridge'], cv_folds=3)
    m_xgi = XGIModel(n_trials=10, cv_splits=3)  # Reduced trials for speed
    m_cs = CleanSheetModel()
    m_cards = CardsModel()
    
    # Train minutes model
    if len(X) > 0:
        try:
            log.info("Training ensemble minutes model...")
            m_mins.fit(X, y_mins)
        except Exception as e:
            log.warning(f"Minutes model failed: {e}, using fallback")
            m_mins = MinutesModel()
            m_mins.fit(X, y_mins)
    
    # Train XGI model with position-specific training
    if len(X) > 0:
        try:
            log.info("Training advanced XGI model with position-specific models...")
            positions = feats["position"] if "position" in feats.columns else None
            m_xgi.fit(X, y_g, y_a, positions)
        except Exception as e:
            log.warning(f"Advanced XGI model failed: {e}")
    
    # Train clean sheet model
    try:
        log.info("Training xG-based clean sheet model...")
        m_cs.fit(fixtures_df)
    except Exception as e:
        log.warning(f"Clean sheet model failed: {e}")

    # Generate predictions
    log.info("Generating predictions...")
    
    # Minutes prediction
    try:
        if hasattr(m_mins, 'predict'):
            xmins = pd.Series(m_mins.predict(X), index=feats.index)
        else:
            xmins = pd.Series(75.0, index=feats.index)
    except Exception:
        xmins = pd.Series(75.0, index=feats.index)
    
    # Apply minutes heuristics
    xm_heur = (
        0.6 * feats["mins_l5"].fillna(0) + 0.4 * feats["mins_l3"].fillna(0)
    ) * (feats["chance_next"].fillna(100) / 100.0)
    
    # CRITICAL FIX: Check if season hasn't started (all mins_l5 are 0)
    season_not_started = feats["mins_l5"].fillna(0).sum() == 0
    
    if xmins.nunique(dropna=True) <= 3 and not season_not_started:
        xmins = xm_heur
    
    # FIX for mid-season transfers: Use games available, not 38
    # If player has starts but low total minutes, likely joined mid-season
    prev_starts = feats["prev_starts"].fillna(0)
    prev_minutes = feats["prev_minutes"].fillna(0.0)
    
    # Detect mid-season transfers: started games but total minutes suggest <38 games available
    # If they averaged 60+ mins per start, they're likely regular when available
    mins_per_start = pd.Series(0.0, index=feats.index)
    mins_per_start[prev_starts > 0] = prev_minutes[prev_starts > 0] / prev_starts[prev_starts > 0]
    
    # Calculate xMins based on actual availability
    xm_prior = pd.Series(0.0, index=feats.index)
    
    # Regular calculation for full-season players
    full_season = (prev_starts >= 25) | (prev_minutes >= 2000)
    xm_prior[full_season] = (prev_minutes[full_season] / 38.0) * (feats["chance_next"].fillna(100.0)[full_season] / 100.0)
    
    # Mid-season/rotation players: use per-start average
    mid_season = (~full_season) & (prev_starts > 0) & (mins_per_start >= 60)
    xm_prior[mid_season] = mins_per_start[mid_season] * (feats["chance_next"].fillna(100.0)[mid_season] / 100.0)
    
    # Low-minute players
    low_mins = (~full_season) & (~mid_season)
    xm_prior[low_mins] = (prev_minutes[low_mins] / 38.0) * (feats["chance_next"].fillna(100.0)[low_mins] / 100.0)
    
    # When season hasn't started, prefer xm_prior over xm_heur
    if season_not_started:
        xmins = xm_prior
    else:
        xmins = xmins.where(xmins > 0, xm_prior)

    starter_boost = (feats["prev_starts"].fillna(0) >= 20).astype(float) * 10.0
    avail = (feats["chance_next"].fillna(100.0) / 100.0).clip(0, 1)
    
    # Improved fallback for gameweek 1: use previous season patterns instead of flat 60
    prev_starts = feats["prev_starts"].fillna(0)
    fallback_mins = pd.Series(0.0, index=feats.index)
    fallback_mins[prev_starts >= 25] = 75.0 * avail  # Regular starters
    fallback_mins[(prev_starts >= 10) & (prev_starts < 25)] = 45.0 * avail  # Squad players
    fallback_mins[(prev_starts >= 5) & (prev_starts < 10)] = 25.0 * avail  # Rotation players
    fallback_mins[prev_starts < 5] = 10.0 * avail  # Bench/new players
    
    # High-value players (>8.0m) without history likely to be new signings who will play
    # Use both price AND community selection % to identify marquee signings
    high_value_mask = (feats["now_cost"].fillna(0) >= 80) & (prev_starts < 5)
    
    # Marquee signings: high price AND high ownership = expected regular starters
    marquee_mask = ((feats["now_cost"].fillna(0) >= 85) | 
                    (feats["selected_by_percent"].fillna(0) >= 15.0)) & (prev_starts < 5)
    
    # Very high ownership (>25%) new signings are almost certainly expected starters
    high_ownership_mask = (feats["selected_by_percent"].fillna(0) >= 25.0) & (prev_starts < 5)
    
    fallback_mins[high_value_mask] = 65.0 * avail[high_value_mask]
    fallback_mins[marquee_mask] = 75.0 * avail[marquee_mask]  # Marquee signings
    fallback_mins[high_ownership_mask] = 85.0 * avail[high_ownership_mask]  # Community favorites
    
    xmins = (xmins + starter_boost).where(xmins > 0, fallback_mins)
    xmins = xmins.clip(lower=0, upper=90)
    
    # Create a copy to ensure modifications stick
    xmins = xmins.copy()
    
    # ---------------------- Handle Known Competition and Backup Players ---------------------- #
    # Get current FPL data for names and ownership
    log.info("Applying squad role adjustments to xMins...")
    try:
        bootstrap = get_bootstrap()
        player_lookup = {p['id']: p for p in bootstrap['elements']}
        team_lookup = {t['id']: t['name'] for t in bootstrap['teams']}
        
        # Track changes
        n_changes = 0
        
        # Define squad roles by team
        squad_roles = {
            # Arsenal (team 1)
            1: {
                'starters': ['Raya', 'Gabriel', 'Saliba', 'Timber', 'Calafiori', 'Partey', 
                            'Rice', 'Ødegaard', 'Saka', 'Martinelli', 'Havertz', 'Trossard'],
                'rotation': ['Jorginho', 'Merino', 'Jesus', 'Sterling'],
                'backup': ['Kiwior', 'Lewis-Skelly', 'White', 'Zinchenko', 'Setford', 
                          'Hein', 'Nwaneri', 'Clarke', 'Kacurri', 'Nichols', 'Mosquera']
            },
            # Manchester City (team 13)
            13: {
                'starters': ['Ederson', 'Walker', 'Dias', 'Akanji', 'Gvardiol', 
                            'Rodri', 'De Bruyne', 'B.Silva', 'Foden', 'Haaland', 'Doku'],
                'rotation': ['Stones', 'Aké', 'Kovacic', 'Grealish', 'Savinho'],
                'backup': ['Ortega', 'Ortega Moreno', 'Lewis', 'Simpson-Pusey']
            },
            # Liverpool (team 12)
            12: {
                'starters': ['Alisson', 'Alexander-Arnold', 'van Dijk', 'Konaté', 'Frimpong',
                            'Mac Allister', 'Gravenberch', 'Szoboszlai', 'M.Salah', 'Díaz', 'Gakpo'],
                'rotation': ['Robertson', 'Jones', 'Elliott', 'Núñez', 'Jota'],
                'backup': ['Kelleher', 'Gomez', 'Tsimikas', 'Quansah', 'Bradley', 'Endo', 'Morton']
            },
            # Chelsea (team 7)
            7: {
                'starters': ['Sánchez', 'Cucurella', 'Colwill', 'Fofana', 'Gusto',
                            'Enzo', 'Caicedo', 'Palmer', 'Neto', 'Jackson', 'Mudryk'],
                'rotation': ['James', 'Badiashile', 'Lavia', 'Madueke', 'João Félix', 'Nkunku'],
                'backup': ['Jörgensen', 'Bettinelli', 'Chalobah', 'Veiga', 'Chilwell', 'Dewsbury-Hall']
            },
        }
        
        # Create a mapping from player_id to index for faster lookups
        pid_to_idx = {pid: i for i, pid in enumerate(feats["player_id"].values)}
        
        # Process all players for squad roles
        for pid in player_lookup.keys():
            if pid not in pid_to_idx:
                continue
            i = pid_to_idx[pid]
            player = player_lookup[pid]
            name = player['web_name']
            pos_type = player['element_type']
            ownership = float(player['selected_by_percent'])
            cost = player['now_cost'] / 10.0
            team_id = player['team']
            
            # Check squad roles if defined for this team
            if team_id in squad_roles:
                roles = squad_roles[team_id]
                
                if name in roles['starters']:
                    # Boost starters if needed
                    if xmins.iloc[i] < 75:
                        log.debug(f"Boosting starter {name} (ID:{pid}) from {xmins.iloc[i]:.1f} to 85 xMins")
                        xmins.iloc[i] = 85.0
                        n_changes += 1
                elif name in roles['rotation']:
                    # Cap rotation players
                    if xmins.iloc[i] > 60:
                        log.debug(f"Capping rotation player {name} (ID:{pid}) from {xmins.iloc[i]:.1f} to 45 xMins")
                        xmins.iloc[i] = 45.0
                        n_changes += 1
                elif name in roles['backup']:
                    # Minimize backup players
                    if xmins.iloc[i] > 20:
                        new_mins = 0.0 if pos_type == 1 else 10.0
                        log.debug(f"Setting backup {name} (ID:{pid}) from {xmins.iloc[i]:.1f} to {new_mins} xMins")
                        xmins.iloc[i] = new_mins
                        n_changes += 1
            
            # General heuristics for all teams (especially those not in squad_roles)
            # Backup GKPs - comprehensive detection
            if pos_type == 1:
                # Multiple criteria for backup GKP detection
                is_backup = False
                
                # Known backup GKP names
                backup_gkp_names = ['Ortega', 'Travers', 'Kelleher', 'Ramsdale', 'Turner',
                                   'Setford', 'Hein', 'Neto', 'Virginia', 'Johnstone', 
                                   'Dennis', 'Vigouroux', 'Sinisalo', 'Darlow', 'Ward',
                                   'Whiteman', 'Bergström', 'Bettinelli', 'Carson', 'Gauci', 
                                   'Wharton', 'Jörgensen', 'Heaton', 'Bayindir']
                
                if any(backup in name for backup in backup_gkp_names):
                    is_backup = True
                elif ownership < 2.0 and cost < 5.0:
                    is_backup = True
                elif ownership < 0.5:  # Almost no ownership = definitely backup
                    is_backup = True
                elif cost < 4.5 and (i >= len(prev_minutes) or prev_minutes.iloc[i] < 500):
                    is_backup = True
                
                # Check team for known backup situations at big clubs
                if team_id in [1, 7, 12, 13, 22, 33] and cost < 5.0 and ownership < 5.0:
                    is_backup = True
                
                if is_backup and xmins.iloc[i] > 0:
                    log.debug(f"Setting backup GKP {name} (ID:{pid}, {ownership:.1f}% owned, £{cost:.1f}m) to 0 xMins")
                    xmins.iloc[i] = 0
                    n_changes += 1
            
            # Additional heuristics for defenders and midfielders not in squad_roles
            elif pos_type == 2 and team_id not in squad_roles:  # Defenders
                # Low ownership defenders at top clubs are likely backups
                if ownership < 2.0 and cost >= 5.0 and team_id in [1, 7, 12, 13, 22, 33]:
                    if xmins.iloc[i] > 45:
                        log.info(f"Reducing likely backup DEF {name} (ID:{pid}) to 20 xMins")
                        xmins.iloc[i] = 20.0
                # Very high ownership = likely starter
                elif ownership > 15.0:
                    if xmins.iloc[i] < 75:
                        log.info(f"Boosting high-ownership DEF {name} (ID:{pid}) to 85 xMins")
                        xmins.iloc[i] = 85.0
            
            elif pos_type == 3 and team_id not in squad_roles:  # Midfielders
                if ownership < 1.0 and cost >= 5.5 and team_id in [1, 7, 12, 13, 22, 33]:
                    if xmins.iloc[i] > 45:
                        log.info(f"Reducing likely backup MID {name} (ID:{pid}) to 25 xMins")
                        xmins.iloc[i] = 25.0
                elif ownership > 20.0:
                    if xmins.iloc[i] < 75:
                        log.info(f"Boosting high-ownership MID {name} (ID:{pid}) to 85 xMins")
                        xmins.iloc[i] = 85.0
            
            elif pos_type == 4 and team_id not in squad_roles:  # Forwards
                if ownership < 1.0 and cost >= 6.0 and team_id in [1, 7, 12, 13, 22, 33]:
                    if xmins.iloc[i] > 45:
                        log.info(f"Reducing likely backup FWD {name} (ID:{pid}) to 30 xMins")
                        xmins.iloc[i] = 30.0
                elif ownership > 15.0:
                    if xmins.iloc[i] < 75:
                        log.info(f"Boosting high-ownership FWD {name} (ID:{pid}) to 85 xMins")
                        xmins.iloc[i] = 85.0
        # Apply comprehensive backup GKP detection for ALL teams
        # This ensures we catch all backup GKPs regardless of team
        all_gkps = [(pid, p) for pid, p in player_lookup.items() if p['element_type'] == 1]
        
        for pid, player in all_gkps:
            if pid not in pid_to_idx:
                continue
            i = pid_to_idx[pid]
            name = player['web_name']
            ownership = float(player['selected_by_percent'])
            cost = player['now_cost'] / 10.0
            
            # Already handled by squad roles? Skip
            if player['team'] in squad_roles:
                continue
                
            # Apply general backup GKP heuristics
            is_backup = False
            
            # Very low ownership is a strong indicator
            if ownership < 1.0:
                is_backup = True
            # Low cost + low ownership
            elif cost < 5.0 and ownership < 3.0:
                is_backup = True
            # Known backup names (partial match)
            backup_indicators = ['Ortega', 'Travers', 'Kelleher', 'Turner', 'Ramsdale',
                               'Johnstone', 'Vigouroux', 'Darlow', 'Ward', 'Heaton']
            if any(ind in name for ind in backup_indicators):
                is_backup = True
                
            if is_backup and xmins.iloc[i] > 0:
                log.debug(f"Setting {name} to 0 xMins (backup GKP: {ownership:.1f}% owned, £{cost:.1f}m)")
                xmins.iloc[i] = 0
                n_changes += 1
        
        log.info(f"Applied {n_changes} squad role adjustments to xMins")
        
    except Exception as e:
        log.warning(f"Could not apply competition adjustments: {e}")
    
    # Final clip to ensure valid range
    xmins = xmins.clip(lower=0, upper=90)

    # Final save with all adjustments applied
    write_parquet(pd.DataFrame({"player_id": feats["player_id"].values, "xmins": xmins.values}), PROC / "xmins.parquet")
    log.info(f"Saved xMins for {len(xmins)} players with squad role adjustments")

    # XGI predictions with uncertainty
    try:
        positions = feats["position"] if "position" in feats.columns else None
        uncertainty_data = m_xgi.predict_with_uncertainty(X, positions, n_iterations=50)
        mu_g = uncertainty_data['goals']['mean']
        mu_a = uncertainty_data['assists']['mean']
        
        # Save uncertainty data
        uncertainty_df = pd.DataFrame({
            'player_id': feats['player_id'].values.astype(int),
            'goals_std': uncertainty_data['goals']['std'].values,
            'goals_lower': uncertainty_data['goals']['lower'].values,
            'goals_upper': uncertainty_data['goals']['upper'].values,
            'assists_std': uncertainty_data['assists']['std'].values,
            'assists_lower': uncertainty_data['assists']['lower'].values,
            'assists_upper': uncertainty_data['assists']['upper'].values
        })
        write_parquet(uncertainty_df, PROC / "prediction_uncertainty.parquet")
        log.info("Saved prediction uncertainty data")
    except Exception as e:
        log.warning(f"XGI prediction failed: {e}")
        mu_g = pd.Series(0.0, index=feats.index)
        mu_a = pd.Series(0.0, index=feats.index)
    
    # Apply priors if needed
    mu_g, mu_a = _apply_xgi_priors_if_needed(feats, mu_g, mu_a)
    
    # ---------------------- CDM Detection and Penalty ---------------------- #
    # Identify defensive midfielders using ICT Index data from FPL API
    # CDMs have low Threat scores compared to attacking mids/wingers
    
    is_mid = feats["position"] == "MID"
    
    # Get ICT data from FPL API for better role detection
    try:
        from ..data.fpl_api import get_bootstrap
        bs = get_bootstrap()
        players_df = pd.DataFrame(bs['elements'])[['id', 'threat', 'creativity', 'influence', 'minutes']]
        players_df = players_df.rename(columns={'id': 'player_id'})
        
        # Calculate threat per 90 minutes (normalized)
        players_df['threat_per90'] = (players_df['threat'] / players_df['minutes'].replace(0, 1)) * 90
        
        # Merge with features
        threat_data = feats[['player_id']].merge(players_df[['player_id', 'threat_per90']], 
                                                 on='player_id', how='left')
        threat_per90 = threat_data['threat_per90'].fillna(0)
        
        # CDMs typically have threat_per90 < 35 (based on Rice:~15, Enzo:~20)
        # Attacking mids have threat_per90 > 35 (Palmer:~40, Saka:~35)
        is_defensive_mid = is_mid & (threat_per90 < 35) & (feats["prev_minutes"].fillna(0) > 1500)
        
        # Also use xG as secondary check (defensive mids rarely have high xG)
        low_xg = feats["prev_xg_per90"].fillna(0) < 0.20
        is_likely_cdm = is_defensive_mid | (is_mid & low_xg & (threat_per90 < 30))
        
    except Exception as e:
        # Fallback to statistical approach if API data unavailable
        log.warning(f"Could not load ICT data for CDM detection: {e}")
        low_xg = feats["prev_xg_per90"].fillna(0) < 0.20
        low_combined_xa = (feats["prev_xg_per90"].fillna(0) + feats["prev_xa_per90"].fillna(0)) < 0.40
        high_minutes = feats["prev_minutes"].fillna(0) > 2000
        is_likely_cdm = is_mid & low_xg & low_combined_xa & high_minutes
    
    # Apply 30% penalty to CDM expected goals/assists
    cdm_penalty = 0.7  # Reduce to 70% of predicted value
    mu_g = mu_g.where(~is_likely_cdm, mu_g * cdm_penalty)
    mu_a = mu_a.where(~is_likely_cdm, mu_a * cdm_penalty)
    
    # Log how many CDMs were detected
    num_cdms = is_likely_cdm.sum()
    if num_cdms > 0:
        log.info(f"Applied CDM penalty to {num_cdms} defensive midfielders")


    # FDR scaling for xGI (boost good runs, trim bad ones)
    try:
        fdrp = pd.read_parquet(PROC / "player_next5_fdr.parquet")
        fdr_map = dict(zip(fdrp["player_id"].astype(int), fdrp["fdr_factor"].astype(float)))
        fac = feats["player_id"].map(lambda pid: fdr_map.get(int(pid), 1.0)).astype(float)
        fac = fac ** FDR_GAMMA
        mu_g = mu_g * fac
        mu_a = mu_a * fac
    except Exception:
        pass  # if file missing, skip

    # Clean sheet predictions
    try:
        p_cs = m_cs.predict_player_cs_probability(feats)
    except Exception:
        pos = feats["position"].fillna("")
        p_cs = pd.Series(0.0, index=feats.index)
        p_cs[(pos == "GKP") | (pos == "DEF")] = CS_PRIOR_DEF_GKP

    # Calculate expected points
    try:
        ep_df = expected_points_df(
            feats=feats,
            xmins=xmins,
            mu_g=mu_g,
            mu_a=mu_a,
            p_cs=p_cs,
        )
    except Exception as e:
        log.warning(f"expected_points_df failed: {e}")
        ep_df = pd.DataFrame({"player_id": feats["player_id"].values, "ep_model": 0.0})

    # Blend with FPL predictions
    try:
        boot = get_bootstrap()
        ep_map = {int(e["id"]): float(e.get("ep_next") or 0.0) for e in boot["elements"]}
        ep_fpl = pd.Series(ep_map, name="ep_fpl").rename_axis("player_id").reset_index()
        ep_df = ep_df.merge(ep_fpl, on="player_id", how="left")
    except Exception:
        ep_df["ep_fpl"] = 0.0

    # Global calibration: scale ep_model to better match FPL level for likely starters
    try:
        tmp = ep_df.merge(feats[["player_id"] + [c for c in ["now_cost"] if c in feats.columns]],
                          on="player_id", how="left")
        xm_s = pd.DataFrame({"player_id": feats["player_id"], "xmins": xmins}).merge(tmp, on="player_id", how="right")
        mask = (xm_s["xmins"] >= 60) & (xm_s["ep_fpl"] > 0)
        med_fpl = float(xm_s.loc[mask, "ep_fpl"].median()) if mask.any() else np.nan
        med_model = float(xm_s.loc[mask, "ep_model"].median()) if mask.any() else np.nan
        if np.isfinite(med_fpl) and np.isfinite(med_model) and med_model > 0:
            scale = np.clip(med_fpl / med_model, 0.90, 1.25)
            ep_df["ep_model"] = ep_df["ep_model"] * scale
            log.info(f"Calibrated ep_model by factor {scale:.3f}")
    except Exception as e:
        log.warning("Global calibration skipped: %s", e)

    # Reliability-based blending
    mins_recent = feats.get("mins_l5", pd.Series(0.0, index=feats.index)).fillna(0.0) + \
                  feats.get("mins_l3", pd.Series(0.0, index=feats.index)).fillna(0.0)
    prev_mins = feats.get("prev_minutes", pd.Series(0.0, index=feats.index)).fillna(0.0)
    reliability = (mins_recent / 180.0 + prev_mins / 3420.0).clip(0.0, 1.0)
    beta_series = (BETA_MIN + (BETA_MAX - BETA_MIN) * reliability).astype(float)

    ep_df = ep_df.merge(pd.DataFrame({"player_id": feats["player_id"].values, "beta": beta_series.values}),
                        on="player_id", how="left")
    ep_df["ep_blend"] = ep_df["beta"].fillna((BETA_MIN + BETA_MAX)/2) * ep_df["ep_model"].fillna(0.0) + \
                        (1.0 - ep_df["beta"].fillna((BETA_MIN + BETA_MAX)/2)) * ep_df["ep_fpl"].fillna(0.0)

    # Premium uplift
    try:
        meta = feats[["player_id", "now_cost", "selected_by_percent", "team"]].copy()
        meta["now_cost"] = meta["now_cost"].fillna(0.0)
        meta["selected_by_percent"] = meta["selected_by_percent"].fillna(0.0)
        ep_df = ep_df.merge(meta, on="player_id", how="left")
        prem_mask = (ep_df["now_cost"] >= PREMIUM_COST) | (ep_df["selected_by_percent"] >= PREMIUM_SEL)
        ep_df.loc[prem_mask, "ep_blend"] *= (1.0 + PREMIUM_UPLIFT)
    except Exception:
        pass

    # Calculate xgi90_est and team_att
    try:
        xgi_per_match = (mu_g.fillna(0) + mu_a.fillna(0))
        mins_factor = (xmins / 90.0).clip(lower=0.2, upper=1.0)
        mins_factor_safe = mins_factor.replace(0, np.nan).fillna(0.5)
        xgi90_est = (xgi_per_match / mins_factor_safe).clip(0.0, 1.20)
    except Exception:
        xgi90_est = pd.Series(0.0, index=feats.index)

    try:
        team_att_map = _team_attack_strength(get_bootstrap())
        team_ids = feats.get("team", pd.Series(np.nan, index=feats.index)).astype("Int64")
        team_att = team_ids.map(lambda t: float(team_att_map.get(int(t), 1.0)) if pd.notna(t) else 1.0)
    except Exception:
        team_att = pd.Series(1.0, index=feats.index)

    # Save expected points
    # Load xmins to create ep_adjusted
    xmins_df = read_parquet(PROC / "xmins.parquet")
    xmins_map = dict(zip(xmins_df["player_id"], xmins_df["xmins"]))
    player_xmins = feats["player_id"].map(xmins_map).fillna(0)
    
    # Calculate ep_adjusted (accounts for playing time)
    ep_adjusted = ep_df["ep_blend"].values * (player_xmins.values / 90.0)
    
    out_df = pd.DataFrame({
        "player_id": feats["player_id"].values.astype(int),
        "ep_model": ep_df["ep_model"].values,
        "ep_fpl": ep_df["ep_fpl"].fillna(0.0).values,
        "ep_blend": ep_df["ep_blend"].values,
        "ep_adjusted": ep_adjusted,
        "xgi90_est": xgi90_est.values,
        "team_att": team_att.values,
    })
    write_parquet(out_df, PROC / "exp_points.parquet")

    # Log results
    try:
        q = out_df["ep_blend"].quantile([0.1, 0.5, 0.9]).to_dict()
        log.info(f"EP blend quantiles: p10={q.get(0.1, float('nan')):.2f} " +
                f"p50={q.get(0.5, float('nan')):.2f} p90={q.get(0.9, float('nan')):.2f}; " +
                f">3 pts: {int((out_df['ep_blend'] > 3.0).sum())} " +
                f"(>4: {int((out_df['ep_blend'] > 4.0).sum())})")
    except Exception:
        pass

    log.info(f"Advanced models trained; xMins & exp_points written. rows={len(feats)}")
    
    return {
        "minutes": m_mins,
        "xgi": m_xgi,
        "cs": m_cs,
        "cards": m_cards,
        "feature_engineer": feature_engineer
    }