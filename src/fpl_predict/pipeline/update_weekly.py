from __future__ import annotations
from pathlib import Path
from ..utils.logging import get_logger
from ..utils.cache import DATA, MODELS, ROOT, PROC
from ..data.ingest import bootstrap_raw_from_sample, ingest_full
from ..data.process import build_features
from ..data.rules_fetcher import update_readme_scoring_table
from ..fixtures.fdr import compute_fdr, build_player_next5_fdr
from ..models.train import train_all
from ..utils.io import write_json, read_parquet, write_parquet
from ..utils.time import now_utc_str
from ..data.fpl_api import get_bootstrap
from .competition_detector import detect_position_competition
from .recent_transfers import apply_transfer_adjustments
import pandas as pd

log = get_logger(__name__)

def fix_squad_roles_post_training():
    """Fix xMins for backup players and update ep_adjusted after training."""
    log.info("Applying squad role fixes post-training...")
    
    try:
        # Load current data
        xmins_df = read_parquet(PROC / "xmins.parquet")
        bootstrap = get_bootstrap()
        
        # Track changes
        n_changes = 0
        
        # Process all GKPs for backup detection
        for player in bootstrap['elements']:
            if player['element_type'] != 1:  # Not a GKP
                continue
                
            pid = player['id']
            if pid not in xmins_df['player_id'].values:
                continue
                
            idx = xmins_df[xmins_df['player_id'] == pid].index[0]
            current_xmins = xmins_df.loc[idx, 'xmins']
            
            name = player['web_name']
            ownership = float(player['selected_by_percent'])
            cost = player['now_cost'] / 10.0
            team = player['team']
            
            # Identify backup GKPs
            is_backup = False
            
            # Known backup names
            backup_names = ['Ortega', 'Travers', 'Kelleher', 'Ramsdale', 'Turner',
                           'Setford', 'Hein', 'Neto', 'Virginia', 'Johnstone', 
                           'Dennis', 'Vigouroux', 'Sinisalo', 'Darlow', 'Ward',
                           'Whiteman', 'Bergström', 'Bettinelli', 'Carson', 'Gauci', 
                           'Wharton', 'Jörgensen', 'Heaton', 'Bayindir']
            
            if any(backup in name for backup in backup_names):
                is_backup = True
            elif ownership < 2.0 and cost < 5.0:
                is_backup = True
            elif ownership < 0.5:
                is_backup = True
            
            # Big clubs with clear #1 GKPs
            if team in [1, 7, 12, 13, 22, 33] and cost < 5.0 and ownership < 5.0:
                is_backup = True
            
            if is_backup and current_xmins > 0:
                xmins_df.loc[idx, 'xmins'] = 0
                n_changes += 1
                log.debug(f"Set {name} (ID:{pid}) to 0 xMins (backup GKP)")
        
        # Fix known backup defenders
        backup_defenders = {
            'Kiwior': 10,
            'Lewis-Skelly': 10,
            'White': 10,
            'Zinchenko': 10,
            'Chalobah': 10,
            'Chilwell': 10,
            'Tsimikas': 10,
            'Gomez': 10,
            'Bradley': 10,
            'Quansah': 10,
        }
        
        for player in bootstrap['elements']:
            if player['element_type'] != 2:  # Not a DEF
                continue
                
            name = player['web_name']
            if name in backup_defenders:
                pid = player['id']
                if pid in xmins_df['player_id'].values:
                    idx = xmins_df[xmins_df['player_id'] == pid].index[0]
                    if xmins_df.loc[idx, 'xmins'] > backup_defenders[name]:
                        xmins_df.loc[idx, 'xmins'] = backup_defenders[name]
                        n_changes += 1
                        log.debug(f"Set {name} (ID:{pid}) to {backup_defenders[name]} xMins (backup DEF)")
        
        # Apply competition-based adjustments
        log.info("Detecting position competition...")
        competition_adjustments = detect_position_competition()
        
        for pid, factor in competition_adjustments.items():
            if pid in xmins_df['player_id'].values:
                idx = xmins_df[xmins_df['player_id'] == pid].index[0]
                current_xmins = xmins_df.loc[idx, 'xmins']
                new_xmins = current_xmins * factor
                xmins_df.loc[idx, 'xmins'] = new_xmins
                n_changes += 1
                log.debug(f"Adjusted player {pid} from {current_xmins:.0f} to {new_xmins:.0f} xMins (competition factor {factor})")
        
        # Apply recent transfer adjustments (e.g., Grealish to Everton)
        log.info("Applying recent transfer adjustments...")
        xmins_map = dict(zip(xmins_df['player_id'], xmins_df['xmins']))
        xmins_map = apply_transfer_adjustments(xmins_map)
        for pid, new_xmins in xmins_map.items():
            if pid in xmins_df['player_id'].values:
                idx = xmins_df[xmins_df['player_id'] == pid].index[0]
                if xmins_df.loc[idx, 'xmins'] != new_xmins:
                    xmins_df.loc[idx, 'xmins'] = new_xmins
                    n_changes += 1
        
        # Apply availability adjustments for injured/unavailable players
        log.info("Applying availability adjustments...")
        from ..models.availability import apply_availability_adjustments
        
        # First update xmins based on availability
        xmins_df = apply_availability_adjustments(xmins_df, bootstrap)
        
        # Save updated xmins
        write_parquet(xmins_df, PROC / "xmins.parquet")
        
        # Update ep_adjusted with availability adjustments
        ep_df = read_parquet(PROC / "exp_points.parquet")
        ep_df = apply_availability_adjustments(ep_df, bootstrap)
        
        # Also recalculate ep_adjusted based on updated xmins
        xmins_map = dict(zip(xmins_df['player_id'], xmins_df['xmins']))
        ep_df['ep_adjusted'] = ep_df['ep_blend'] * ep_df['player_id'].map(xmins_map).fillna(0) / 90.0
        write_parquet(ep_df, PROC / "exp_points.parquet")
        
        log.info(f"Applied {n_changes} squad role and competition fixes")
        
    except Exception as e:
        log.warning(f"Squad role fixes failed: {e}")

def update_weekly_data(demo_mode: bool = False) -> None:
    log.info("Starting weekly update (demo=%s)", demo_mode)

    # Step 1: Ingest latest data from FPL API
    if demo_mode:
        bootstrap_raw_from_sample()
    else:
        ingest_full()

    # Step 2: Build features from the fresh data
    build_features()
    update_readme_scoring_table(ROOT / "README.md")
    compute_fdr()
    build_player_next5_fdr()

    # Step 3: IMPORTANT - Delete old training data to force rebuild with current stats
    # This fixes the bug where models were using months-old player statistics
    training_data_file = PROC / "training_data.parquet"
    if training_data_file.exists():
        log.info("Removing old training_data.parquet to force rebuild with current player stats")
        training_data_file.unlink()

    # Step 4: Train models (will now rebuild training data with current stats)
    models = train_all()
    write_json({"saved_at": now_utc_str(), "models": list(models.keys())}, MODELS / "latest.json")

    # Step 5: Apply post-training fixes for squad roles
    fix_squad_roles_post_training()

    write_json({"updated_at": now_utc_str()}, DATA / "processed" / "weekly_changelog.json")
    log.info("Weekly update complete.")

def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="store_true")
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()
    if args.run:
        update_weekly_data(demo_mode=args.demo)
    else:
        log.info("Use --run to execute the weekly update.")
