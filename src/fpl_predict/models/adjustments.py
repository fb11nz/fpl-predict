"""
Adjustments for model predictions to handle edge cases.
"""
import pandas as pd
import numpy as np
from typing import Optional
from ..utils.logging import get_logger

log = get_logger(__name__)


def apply_new_player_adjustments(
    predictions_df: pd.DataFrame,
    features_df: pd.DataFrame,
    bootstrap_data: dict,
    min_minutes_threshold: int = 180,  # 2 full games
    new_player_floor_multiplier: float = 0.5,  # Minimum 50% of position average
) -> pd.DataFrame:
    """
    Adjust predictions for players with limited Premier League data.
    
    The model can be overly harsh on new signings or players with limited minutes,
    especially if they were subbed early without returns in their first game(s).
    
    Args:
        predictions_df: DataFrame with model predictions
        features_df: DataFrame with player features
        bootstrap_data: FPL bootstrap data
        min_minutes_threshold: Minutes threshold below which we apply adjustments
        new_player_floor_multiplier: Minimum prediction as fraction of position average
    
    Returns:
        Adjusted predictions DataFrame
    """
    adjusted = predictions_df.copy()
    
    # Get player info from bootstrap
    players_map = {p['id']: p for p in bootstrap_data['elements']}
    
    # Calculate position averages for similar-priced players
    position_benchmarks = {}
    for pos in ['GKP', 'DEF', 'MID', 'FWD']:
        pos_players = [p for p in bootstrap_data['elements'] 
                      if p['element_type'] == {'GKP': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}[pos]]
        
        # Group by price bracket (±£1m)
        price_brackets = {}
        for p in pos_players:
            price = p['now_cost'] / 10  # Convert to millions
            bracket = round(price)  # Round to nearest million
            if bracket not in price_brackets:
                price_brackets[bracket] = []
            
            # Only include players with reasonable minutes
            if p['minutes'] > min_minutes_threshold:
                ppg = p.get('points_per_game', 0)
                if ppg > 0:
                    price_brackets[bracket].append(float(ppg))
        
        # Calculate average for each bracket
        position_benchmarks[pos] = {}
        for bracket, ppgs in price_brackets.items():
            if ppgs:
                position_benchmarks[pos][bracket] = np.mean(ppgs)
    
    # Apply adjustments
    adjustments_made = []
    
    for idx, row in adjusted.iterrows():
        player_id = row.get('id') or row.get('player_id')
        if player_id and player_id in players_map:
            player = players_map[player_id]
            
            # Check if player needs adjustment
            total_minutes = player.get('minutes', 0)
            if total_minutes < min_minutes_threshold:
                # Get player position and price
                pos_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
                pos = pos_map.get(player['element_type'])
                price = player['now_cost'] / 10
                price_bracket = round(price)
                
                if pos and pos in position_benchmarks:
                    # Get benchmark for similar-priced players in position
                    benchmark = position_benchmarks[pos].get(price_bracket)
                    if not benchmark:
                        # Try adjacent brackets
                        benchmark = (position_benchmarks[pos].get(price_bracket - 1, 0) + 
                                   position_benchmarks[pos].get(price_bracket + 1, 0)) / 2
                    
                    if benchmark > 0:
                        # Calculate minimum acceptable prediction
                        min_prediction = benchmark * new_player_floor_multiplier
                        
                        # Check EP columns and adjust if needed
                        for col in adjusted.columns:
                            if col.startswith('EP') or col == 'EPH':
                                current_val = adjusted.at[idx, col]
                                if pd.notna(current_val) and current_val < min_prediction:
                                    # Apply adjustment with explanation
                                    adjustment_factor = min_prediction / max(current_val, 0.1)
                                    adjusted.at[idx, col] = min_prediction
                                    
                                    if col == 'EPH' or col == 'EP_1':  # Log key adjustments
                                        adjustments_made.append({
                                            'player': player['web_name'],
                                            'position': pos,
                                            'price': price,
                                            'minutes': total_minutes,
                                            'original': current_val,
                                            'adjusted': min_prediction,
                                            'benchmark': benchmark
                                        })
    
    # Log adjustments
    if adjustments_made:
        log.info(f"Applied new player adjustments to {len(adjustments_made)} players:")
        for adj in adjustments_made[:5]:  # Show first 5
            log.info(f"  {adj['player']} ({adj['position']}, £{adj['price']:.1f}m): "
                    f"{adj['original']:.2f} -> {adj['adjusted']:.2f} pts/game "
                    f"(benchmark: {adj['benchmark']:.2f}, mins: {adj['minutes']})")
    
    return adjusted


def apply_rotation_risk_adjustments(
    predictions_df: pd.DataFrame,
    features_df: pd.DataFrame,
    early_sub_threshold: int = 65,  # Minutes threshold for early sub
    harsh_penalty_cap: float = 0.7,  # Don't reduce predictions by more than 30%
) -> pd.DataFrame:
    """
    Adjust for rotation risk without being overly harsh on tactical substitutions.
    
    Players subbed at 59-65 minutes are often tactical (preserving for next match)
    rather than performance-based, and shouldn't be heavily penalized.
    
    Args:
        predictions_df: DataFrame with model predictions
        features_df: DataFrame with player features  
        early_sub_threshold: Minutes below which we consider it an early sub
        harsh_penalty_cap: Maximum reduction factor (0.7 = max 30% reduction)
    
    Returns:
        Adjusted predictions DataFrame
    """
    adjusted = predictions_df.copy()
    
    # Check recent minutes patterns
    if 'mins_l3' in features_df.columns and 'starts_l3' in features_df.columns:
        for idx, row in features_df.iterrows():
            mins_l3 = row.get('mins_l3', 0)
            starts_l3 = row.get('starts_l3', 0)
            
            # Check if player is being subbed early consistently
            if starts_l3 > 0:
                avg_mins_when_started = mins_l3 / starts_l3
                
                # If consistently subbed around 60 mins, it's likely tactical
                if 55 <= avg_mins_when_started <= early_sub_threshold:
                    # Don't apply harsh penalty - they're still starting
                    player_id = row.get('id') or row.get('player_id')
                    
                    # Find corresponding prediction row
                    pred_idx = adjusted[
                        (adjusted.get('id') == player_id) | 
                        (adjusted.get('player_id') == player_id)
                    ].index
                    
                    if len(pred_idx) > 0:
                        pred_idx = pred_idx[0]
                        # Check if predictions are too low
                        for col in adjusted.columns:
                            if col.startswith('EP') or col == 'EPH':
                                current_val = adjusted.at[pred_idx, col]
                                if pd.notna(current_val):
                                    # If tactical sub pattern, ensure reasonable floor
                                    # 60 mins should still get ~70-80% of full game points
                                    min_acceptable = current_val / harsh_penalty_cap
                                    if current_val < min_acceptable:
                                        adjusted.at[pred_idx, col] = min_acceptable
    
    return adjusted