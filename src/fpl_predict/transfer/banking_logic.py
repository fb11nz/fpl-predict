"""
Smart banking decision logic for transfer recommendations.
"""
from typing import Dict, List, Any, Tuple
from ..utils.logging import get_logger

log = get_logger(__name__)


def evaluate_banking_decision(
    current_squad: List[Dict],
    banking_advantage: float,
    best_current_gain: float,
    banked_changes: List[Dict],
    current_changes: List[Dict],
    ep_predictions: Dict[int, float],
    xmins_predictions: Dict[int, float]
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Make an intelligent decision about whether to bank transfers or use them now.
    
    Args:
        current_squad: List of current squad players
        banking_advantage: EP advantage of banking (positive = banking better)
        best_current_gain: EP gain from using transfers now
        banked_changes: Transfer changes if banking
        current_changes: Transfer changes if using now
        ep_predictions: EP predictions by player ID
        xmins_predictions: Expected minutes by player ID
        
    Returns:
        Tuple of (should_bank, reasoning, metrics)
    """
    metrics = {
        'unavailable_players': 0,
        'low_ep_players': 0,
        'cant_field_xi': False,
        'banking_advantage': banking_advantage,
        'urgency_score': 0,
        'dead_players': []
    }
    
    # Check for unavailable/injured players
    for player in current_squad:
        # Handle both dict and int formats
        if isinstance(player, dict):
            player_id = player.get('element') or player.get('id')
        else:
            player_id = player
        ep = ep_predictions.get(player_id, 0)
        xmins = xmins_predictions.get(player_id, 90)
        
        if ep == 0 or xmins == 0:
            metrics['unavailable_players'] += 1
            metrics['dead_players'].append(player_id)
            log.info(f"Found unavailable player {player_id} with EP={ep:.2f}, xMins={xmins:.0f}")
        elif ep < 1.0:  # Very low EP threshold
            metrics['low_ep_players'] += 1
    
    # Calculate urgency score
    metrics['urgency_score'] = (
        metrics['unavailable_players'] * 10 +  # Heavy weight for dead players
        metrics['low_ep_players'] * 2          # Some weight for very poor players
    )
    
    # Check if we can field a valid XI
    available_count = len(current_squad) - metrics['unavailable_players']
    if available_count < 11:
        metrics['cant_field_xi'] = True
        log.warning(f"Cannot field XI! Only {available_count} available players")
    
    # Decision logic
    reasoning = []
    should_bank = False
    
    # RULE 1: If you can't field XI, must transfer now
    if metrics['cant_field_xi']:
        should_bank = False
        reasoning.append("CRITICAL: Cannot field 11 players - must transfer now")
        
    # RULE 2: If 2+ unavailable players, strongly prefer transferring now
    elif metrics['unavailable_players'] >= 2:
        should_bank = False
        reasoning.append(f"Have {metrics['unavailable_players']} dead players - need immediate transfers")
        
    # RULE 3: If 1 unavailable player, transfer now unless banking advantage is huge
    elif metrics['unavailable_players'] == 1:
        if banking_advantage > 5.0:  # Only bank if advantage is > 5 EP
            should_bank = True
            reasoning.append(f"1 dead player but banking gives +{banking_advantage:.1f} EP advantage")
        else:
            should_bank = False
            reasoning.append(f"1 dead player and banking advantage only +{banking_advantage:.1f} EP - transfer now")
            
    # RULE 4: No dead players - use standard banking logic
    else:
        # Consider banking if advantage is meaningful (>2 EP)
        if banking_advantage > 2.0:
            should_bank = True
            reasoning.append(f"No urgent issues and banking gives +{banking_advantage:.1f} EP")
        elif banking_advantage > 0:
            # Small advantage - consider other factors
            if best_current_gain < 5.0:
                should_bank = True
                reasoning.append(f"Current transfers low impact ({best_current_gain:.1f} EP), worth waiting")
            else:
                should_bank = False
                reasoning.append(f"Good transfers available now ({best_current_gain:.1f} EP), take them")
        else:
            should_bank = False
            reasoning.append(f"Better to transfer now (banking gives no advantage)")
    
    # Additional context
    if metrics['low_ep_players'] > 2:
        reasoning.append(f"Note: {metrics['low_ep_players']} players with very low EP")
    
    # Check if current best transfer would fix a dead player
    if not should_bank and current_changes:
        for change in current_changes:
            if change.get('out') in metrics['dead_players']:
                reasoning.append(f"✓ Removes dead player ID {change['out']}")
    
    # Final reasoning string
    final_reasoning = " | ".join(reasoning)
    
    log.info(f"Banking decision: {'BANK' if should_bank else 'TRANSFER NOW'} - {final_reasoning}")
    
    return should_bank, final_reasoning, metrics


def format_banking_recommendation(
    should_bank: bool,
    reasoning: str,
    metrics: Dict[str, Any],
    banking_advantage: float,
    current_ft: int,
    next_week_ft: int
) -> str:
    """
    Format the banking recommendation for display.
    
    Args:
        should_bank: Whether to recommend banking
        reasoning: Explanation for the decision
        metrics: Decision metrics
        banking_advantage: EP advantage from banking
        current_ft: Current free transfers
        next_week_ft: Free transfers if banking
        
    Returns:
        Formatted recommendation string
    """
    lines = []
    
    # Add urgency indicators
    if metrics['cant_field_xi']:
        lines.append("⚠️  URGENT: Cannot field full XI without transfers!")
    elif metrics['unavailable_players'] >= 2:
        lines.append(f"⚠️  WARNING: {metrics['unavailable_players']} players unavailable (injured/transferred)")
    elif metrics['unavailable_players'] == 1:
        lines.append(f"⚠  Note: 1 player unavailable")
    
    # Main recommendation
    if should_bank:
        lines.append(f"✓ Recommendation: Bank your transfer for {next_week_ft} FT next week")
        if banking_advantage > 0:
            lines.append(f"   Banking gives +{banking_advantage:.2f} EP advantage")
    else:
        lines.append(f"✓ Recommendation: Use your {current_ft} FT now")
        if metrics['unavailable_players'] > 0:
            lines.append(f"   Remove {metrics['unavailable_players']} dead player(s) immediately")
    
    # Add reasoning
    lines.append(f"   Reasoning: {reasoning}")
    
    return "\n".join(lines)