"""
Handle player availability based on injury status, suspensions, and transfers.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any
from ..utils.logging import get_logger

log = get_logger(__name__)


def apply_availability_adjustments(
    predictions_df: pd.DataFrame,
    bootstrap_data: Dict[str, Any],
    zero_unavailable: bool = True,
    reduce_doubtful: bool = True
) -> pd.DataFrame:
    """
    Adjust predictions based on player availability status.
    
    Sets predictions to 0 for:
    - Injured players with 0% chance of playing
    - Suspended players
    - Players who have left the league (status 'u')
    
    Reduces predictions for:
    - Players marked as doubtful (25-75% chance)
    
    Args:
        predictions_df: DataFrame with player predictions
        bootstrap_data: FPL bootstrap data containing player status
        zero_unavailable: Whether to zero out unavailable players
        reduce_doubtful: Whether to reduce predictions for doubtful players
        
    Returns:
        DataFrame with adjusted predictions
    """
    adjusted = predictions_df.copy()
    
    # Create player status mapping
    status_map = {}
    chance_map = {}
    
    for player in bootstrap_data['elements']:
        player_id = player['id']
        status = player.get('status', 'a')  # a=available, i=injured, d=doubtful, s=suspended, u=unavailable
        chance = player.get('chance_of_playing_this_round')
        news = player.get('news', '')
        
        status_map[player_id] = status
        chance_map[player_id] = chance
        
        # Log important cases
        if status in ['i', 's', 'u']:
            log.info(f"Player {player['web_name']} (ID {player_id}) is unavailable: status={status}, news={news[:50]}")
    
    # Apply adjustments
    adjusted_count = 0
    zeroed_count = 0
    
    for idx, row in adjusted.iterrows():
        player_id = row['player_id']
        
        if player_id not in status_map:
            continue
            
        status = status_map[player_id]
        chance = chance_map[player_id]
        
        # Handle completely unavailable players
        if zero_unavailable:
            should_zero = False
            
            # Injured with 0% chance
            if status == 'i' and (chance == 0 or pd.isna(chance)):
                should_zero = True
                
            # Suspended
            elif status == 's':
                should_zero = True
                
            # Left the league / unavailable
            elif status == 'u':
                should_zero = True
                
            if should_zero:
                # Zero out all predictions
                if 'xmins' in adjusted.columns:
                    adjusted.at[idx, 'xmins'] = 0
                if 'ep_model' in adjusted.columns:
                    adjusted.at[idx, 'ep_model'] = 0
                if 'ep_adjusted' in adjusted.columns:
                    adjusted.at[idx, 'ep_adjusted'] = 0
                if 'ep_blend' in adjusted.columns:
                    adjusted.at[idx, 'ep_blend'] = 0
                    
                zeroed_count += 1
                log.debug(f"Zeroed predictions for player {player_id} (status={status}, chance={chance})")
        
        # Handle doubtful players (partial availability)
        if reduce_doubtful and status == 'd' and chance is not None and not pd.isna(chance):
            if 0 < chance < 100:
                # Scale predictions by chance of playing
                multiplier = chance / 100.0
                
                if 'xmins' in adjusted.columns:
                    adjusted.at[idx, 'xmins'] *= multiplier
                if 'ep_model' in adjusted.columns:
                    adjusted.at[idx, 'ep_model'] *= multiplier
                if 'ep_adjusted' in adjusted.columns:
                    adjusted.at[idx, 'ep_adjusted'] *= multiplier
                if 'ep_blend' in adjusted.columns:
                    adjusted.at[idx, 'ep_blend'] *= multiplier
                    
                adjusted_count += 1
                log.debug(f"Reduced predictions for player {player_id} by {(1-multiplier)*100:.0f}% (chance={chance})")
    
    log.info(f"Availability adjustments: {zeroed_count} players zeroed, {adjusted_count} players reduced")
    
    return adjusted


def check_player_availability(player_id: int, bootstrap_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check detailed availability status for a specific player.
    
    Args:
        player_id: FPL player ID
        bootstrap_data: FPL bootstrap data
        
    Returns:
        Dictionary with availability details
    """
    for player in bootstrap_data['elements']:
        if player['id'] == player_id:
            return {
                'id': player_id,
                'name': player['web_name'],
                'status': player.get('status', 'a'),
                'chance': player.get('chance_of_playing_this_round'),
                'news': player.get('news', ''),
                'is_available': player.get('status') == 'a',
                'should_zero': (
                    player.get('status') in ['s', 'u'] or
                    (player.get('status') == 'i' and player.get('chance_of_playing_this_round', 100) == 0)
                )
            }
    
    return None