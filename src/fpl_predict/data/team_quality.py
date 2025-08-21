"""
Calculate team quality from actual historical match performance data.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from ..utils.logging import get_logger
from ..utils.cache import RAW, PROC
from ..utils.io import read_parquet, write_parquet

log = get_logger(__name__)


def calculate_team_quality_from_data() -> Dict[str, float]:
    """
    Calculate team quality scores from actual historical match data.
    
    Returns:
        Dict mapping team names to quality scores (0.5 = worst, 1.5 = best)
    """
    log.info("Calculating team quality from historical match data...")
    
    # Load historical match data
    matches = read_parquet(RAW / 'football-data' / 'EPL_all_matches.parquet')
    
    # Filter to recent seasons for relevance (2023-24, 2024-25)
    # Don't include 2025-26 as it's only 1 game so far
    recent_matches = matches[matches['season'].isin([2023, 2024])].copy()
    
    if len(recent_matches) == 0:
        log.warning("No recent match data found, using fallback team quality")
        return get_fallback_team_quality()
    
    log.info(f"Analyzing {len(recent_matches)} matches from {recent_matches['season'].nunique()} seasons")
    
    # Calculate team performance metrics
    team_stats = {}
    
    # Get all unique teams
    all_teams = set(recent_matches['home_team'].dropna()) | set(recent_matches['away_team'].dropna())
    
    for team in all_teams:
        # Home matches
        home_matches = recent_matches[recent_matches['home_team'] == team]
        away_matches = recent_matches[recent_matches['away_team'] == team]
        
        if len(home_matches) == 0 and len(away_matches) == 0:
            continue
            
        # Calculate points (3 for win, 1 for draw, 0 for loss)
        home_points = 0
        away_points = 0
        total_games = 0
        goals_for = 0
        goals_against = 0
        
        # Home performance
        for _, match in home_matches.iterrows():
            home_goals = match['home_goals']
            away_goals = match['away_goals']
            
            if pd.notna(home_goals) and pd.notna(away_goals):
                goals_for += home_goals
                goals_against += away_goals
                total_games += 1
                
                if home_goals > away_goals:
                    home_points += 3
                elif home_goals == away_goals:
                    home_points += 1
        
        # Away performance  
        for _, match in away_matches.iterrows():
            home_goals = match['home_goals'] 
            away_goals = match['away_goals']
            
            if pd.notna(home_goals) and pd.notna(away_goals):
                goals_for += away_goals
                goals_against += home_goals
                total_games += 1
                
                if away_goals > home_goals:
                    away_points += 3
                elif away_goals == home_goals:
                    away_points += 1
        
        if total_games > 0:
            points_per_game = (home_points + away_points) / total_games
            goals_per_game = goals_for / total_games
            goals_conceded_per_game = goals_against / total_games
            goal_difference_per_game = goals_per_game - goals_conceded_per_game
            
            # Calculate composite quality score
            # Weight: 50% points per game, 30% goal difference, 20% goals scored
            quality_score = (
                0.5 * (points_per_game / 3.0) +  # Normalize to 0-1 scale
                0.3 * max(0, min(1, (goal_difference_per_game + 2) / 4)) +  # GD from -2 to +2 → 0-1
                0.2 * min(1, goals_per_game / 3.0)  # Goals per game capped at 3
            )
            
            team_stats[team] = {
                'points_per_game': points_per_game,
                'goals_per_game': goals_per_game,
                'goals_conceded_per_game': goals_conceded_per_game,
                'goal_difference_per_game': goal_difference_per_game,
                'total_games': total_games,
                'quality_score': quality_score
            }
    
    # Normalize quality scores to 0.5-1.5 range
    if team_stats:
        scores = [stats['quality_score'] for stats in team_stats.values()]
        min_score = min(scores)
        max_score = max(scores)
        
        # Avoid division by zero
        if max_score > min_score:
            normalized_scores = {}
            for team, stats in team_stats.items():
                # Normalize to 0.5-1.5 range (0.5 = weakest, 1.0 = average, 1.5 = strongest)
                normalized = 0.5 + (stats['quality_score'] - min_score) / (max_score - min_score)
                normalized_scores[team] = normalized
        else:
            # All teams equal - set to 1.0 (average)
            normalized_scores = {team: 1.0 for team in team_stats.keys()}
    else:
        normalized_scores = {}
    
    # Handle promoted teams (no PL history)
    promoted_teams = {"Sunderland", "Burnley", "Leeds United"}
    for team in promoted_teams:
        if team not in normalized_scores:
            normalized_scores[team] = 0.6  # Slightly below average for promoted teams
    
    # Log results
    if normalized_scores:
        sorted_teams = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
        log.info("Team quality rankings (top 5):")
        for team, score in sorted_teams[:5]:
            points_pg = team_stats.get(team, {}).get('points_per_game', 0)
            log.info(f"  {team}: {score:.3f} ({points_pg:.2f} pts/game)")
        
        log.info("Team quality rankings (bottom 5):")
        for team, score in sorted_teams[-5:]:
            points_pg = team_stats.get(team, {}).get('points_per_game', 0)
            log.info(f"  {team}: {score:.3f} ({points_pg:.2f} pts/game)")
    
    # Save for caching
    quality_df = pd.DataFrame([
        {'team': team, 'quality_score': score, 'data_driven': True}
        for team, score in normalized_scores.items()
    ])
    write_parquet(quality_df, PROC / 'team_quality.parquet')
    
    log.info(f"Calculated data-driven quality scores for {len(normalized_scores)} teams")
    return normalized_scores


def get_fallback_team_quality() -> Dict[str, float]:
    """
    Fallback team quality when historical data is unavailable.
    Only use for promoted teams or emergency situations.
    """
    return {
        # Promoted teams (no PL history)
        "Sunderland": 0.6,
        "Burnley": 0.6, 
        "Leeds United": 0.6,
        # Fallback average for any other unknown teams
    }


def get_team_quality_scores() -> Dict[str, float]:
    """
    Get team quality scores, calculating from data if needed.
    
    Returns:
        Dict mapping team names to quality scores (0.5-1.5 range)
    """
    try:
        # Try to load cached scores
        quality_file = PROC / 'team_quality.parquet'
        if quality_file.exists():
            quality_df = read_parquet(quality_file)
            scores = dict(zip(quality_df['team'], quality_df['quality_score']))
            log.info(f"Loaded cached team quality scores for {len(scores)} teams")
            return scores
    except Exception as e:
        log.warning(f"Could not load cached team quality: {e}")
    
    # Calculate fresh scores
    return calculate_team_quality_from_data()


def get_team_tier(team_name: str, quality_scores: Dict[str, float] = None) -> int:
    """
    Get team tier (0-3) based on data-driven quality scores.
    
    Args:
        team_name: Name of the team
        quality_scores: Optional pre-calculated scores
        
    Returns:
        Tier: 0 = promoted/weak, 1 = lower-mid, 2 = upper-mid, 3 = top tier
    """
    if quality_scores is None:
        quality_scores = get_team_quality_scores()
    
    score = quality_scores.get(team_name, 1.0)  # Default to average
    
    # Convert quality score (0.5-1.5) to tier (0-3)
    if score >= 1.3:
        return 3  # Top tier (equivalent to old "top 6")
    elif score >= 1.1:
        return 2  # Upper-mid table (equivalent to old "good teams") 
    elif score >= 0.8:
        return 1  # Lower-mid table
    else:
        return 0  # Weak/promoted teams


if __name__ == "__main__":
    scores = calculate_team_quality_from_data()
    print(f"Calculated quality scores for {len(scores)} teams")