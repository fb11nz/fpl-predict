"""Transfer recommendation system for existing FPL teams."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import click

from ..utils.logging import get_logger
from ..utils.cache import PROC
from .optimizer import optimize_transfers

log = get_logger(__name__)

def load_current_team(entry_id: Optional[int] = None) -> List[int]:
    """
    Load current team from myteam snapshot.
    
    Returns list of player IDs currently in the squad.
    """
    myteam_file = Path("data/processed/myteam_latest.json")
    
    if not myteam_file.exists():
        raise FileNotFoundError(
            "No team data found. Run 'fpl myteam sync --entry YOUR_ID' first to download your current team."
        )
    
    with open(myteam_file) as f:
        data = json.load(f)
    
    # Extract player IDs from picks
    player_ids = [pick["element"] for pick in data.get("picks", [])]
    
    if len(player_ids) != 15:
        log.warning(f"Expected 15 players but found {len(player_ids)}")
    
    return player_ids


def recommend_weekly_transfers(
    max_transfers: int = 1,
    planning_horizon: int = 5,
    consider_hits: bool = False,
    entry_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Recommend transfers for the upcoming gameweek.
    
    Args:
        max_transfers: Maximum transfers to consider (1 or 2)
        planning_horizon: How many GWs ahead to optimize for
        consider_hits: Whether to consider taking a -4 hit for 2 transfers
        entry_id: FPL team ID (optional, uses myteam_latest.json)
    
    Returns:
        Dictionary with transfer recommendations
    """
    # Load current team
    current_squad = load_current_team(entry_id)
    log.info(f"Loaded current squad with {len(current_squad)} players")
    
    # Get recommendations for different scenarios
    scenarios = []
    
    # Common optimization parameters
    opt_params = {
        "horizon": planning_horizon,
        "use_advanced": True,  # Use LP optimization
        "bench_weight": 0.1,
        "differential_bonus": 0.05,
        "risk_penalty": 0.1,
        "value_weight": 0.2,
    }
    
    # No transfer scenario (baseline)
    no_transfer = optimize_transfers(
        current_squad=current_squad,
        max_transfers=0,
        wildcard=False,
        **opt_params
    )
    if "error" not in no_transfer:
        # Extract squad IDs from result
        squad_ids = [p["id"] for p in no_transfer.get("squad", [])]
        scenarios.append({
            "transfers": 0,
            "cost": 0,
            "expected_points": no_transfer.get("expected_points_gw1", 0),
            "net_points": no_transfer.get("expected_points_gw1", 0),
            "squad": squad_ids,
            "changes": []
        })
    
    # One transfer (free)
    one_transfer = optimize_transfers(
        current_squad=current_squad,
        max_transfers=1,
        wildcard=False,
        **opt_params
    )
    if "error" not in one_transfer:
        squad_ids = [p["id"] for p in one_transfer.get("squad", [])]
        changes = identify_changes(current_squad, squad_ids)
        scenarios.append({
            "transfers": 1,
            "cost": 0,
            "expected_points": one_transfer.get("expected_points_gw1", 0),
            "net_points": one_transfer.get("expected_points_gw1", 0),
            "squad": squad_ids,
            "changes": changes
        })
    
    # Two transfers (if considering hits)
    if consider_hits and max_transfers >= 2:
        two_transfers = optimize_transfers(
            current_squad=current_squad,
            max_transfers=2,
            wildcard=False,
            **opt_params
        )
        if "error" not in two_transfers:
            squad_ids = [p["id"] for p in two_transfers.get("squad", [])]
            changes = identify_changes(current_squad, squad_ids)
            expected_pts = two_transfers.get("expected_points_gw1", 0)
            scenarios.append({
                "transfers": 2,
                "cost": 4,  # -4 hit
                "expected_points": expected_pts,
                "net_points": expected_pts - 4,
                "squad": squad_ids,
                "changes": changes
            })
    
    # Find best scenario by net points
    if scenarios:
        best_scenario = max(scenarios, key=lambda x: x["net_points"])
        
        # Format recommendation
        recommendation = {
            "recommended_transfers": best_scenario["transfers"],
            "transfer_cost": best_scenario["cost"],
            "expected_gain": best_scenario["net_points"] - scenarios[0]["net_points"] if scenarios else 0,
            "changes": best_scenario["changes"],
            "scenarios": scenarios,
            "current_squad": current_squad,
            "new_squad": best_scenario["squad"]
        }
        
        return recommendation
    
    return {"error": "Could not generate transfer recommendations"}


def identify_changes(current_squad: List[int], new_squad: List[int]) -> List[Dict[str, Any]]:
    """Identify transfers between two squads."""
    current_set = set(current_squad)
    new_set = set(new_squad)
    
    transfers_out = list(current_set - new_set)
    transfers_in = list(new_set - current_set)
    
    changes = []
    for i, pid_out in enumerate(transfers_out):
        if i < len(transfers_in):
            changes.append({
                "out": pid_out,
                "in": transfers_in[i]
            })
    
    return changes


def format_recommendation_output(recommendation: Dict[str, Any]) -> str:
    """Format recommendation for CLI output."""
    from ..data.fpl_api import get_bootstrap
    
    # Get player names
    bootstrap = get_bootstrap()
    players_map = {p["id"]: p for p in bootstrap["elements"]}
    teams_map = {t["id"]: t["name"] for t in bootstrap["teams"]}
    
    output = []
    output.append("=" * 60)
    output.append("TRANSFER RECOMMENDATIONS")
    output.append("=" * 60)
    
    if recommendation.get("error"):
        output.append(f"Error: {recommendation['error']}")
        return "\n".join(output)
    
    # Summary
    rec_transfers = recommendation["recommended_transfers"]
    expected_gain = recommendation["expected_gain"]
    cost = recommendation["transfer_cost"]
    
    if rec_transfers == 0:
        output.append("\n✅ HOLD - No transfers recommended this week")
        output.append(f"Your current team is well-optimized for the upcoming fixtures.")
    else:
        output.append(f"\n📊 Recommended: {rec_transfers} transfer{'s' if rec_transfers > 1 else ''}")
        if cost > 0:
            output.append(f"Hit cost: -{cost} points")
        output.append(f"Expected net gain: +{expected_gain:.1f} points")
        
        # Show specific transfers
        output.append("\n" + "-" * 40)
        output.append("TRANSFERS:")
        for change in recommendation["changes"]:
            p_out = players_map.get(change["out"], {})
            p_in = players_map.get(change["in"], {})
            
            out_name = p_out.get("web_name", f"Player {change['out']}")
            out_team = teams_map.get(p_out.get("team"), "")
            in_name = p_in.get("web_name", f"Player {change['in']}")
            in_team = teams_map.get(p_in.get("team"), "")
            
            output.append(f"OUT: {out_name} ({out_team})")
            output.append(f" IN: {in_name} ({in_team})")
            output.append("")
    
    # Scenario comparison
    output.append("\n" + "-" * 40)
    output.append("SCENARIO ANALYSIS:")
    for scenario in recommendation["scenarios"]:
        transfers = scenario["transfers"]
        net_pts = scenario["net_points"]
        cost = scenario["cost"]
        
        if transfers == 0:
            output.append(f"Hold (0 transfers): {net_pts:.1f} pts")
        else:
            output.append(f"{transfers} transfer{'s' if transfers > 1 else ''}: {net_pts:.1f} pts" + 
                         (f" (after -{cost} hit)" if cost > 0 else ""))
    
    output.append("=" * 60)
    return "\n".join(output)