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

def load_current_team(entry_id: Optional[int] = None) -> tuple[List[int], Dict[str, Any]]:
    """
    Load current team from myteam snapshot.

    Returns tuple of (player_ids, team_info) where team_info contains transfers data.
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

    # Extract transfer info
    transfer_info = data.get("transfers", {})

    return player_ids, transfer_info


def match_transfers_by_position(transfers_out, transfers_in, bootstrap_data: Dict) -> List[Dict]:
    """
    Match transfers OUT and IN by position so each pair is the same position.

    Args:
        transfers_out: List of player IDs or Player objects being transferred out
        transfers_in: List of player IDs or Player objects being transferred in
        bootstrap_data: FPL bootstrap data with player info

    Returns:
        List of dicts with 'out' and 'in' keys, matched by position
    """
    from ..data.fpl_api import get_bootstrap

    if not bootstrap_data:
        bootstrap_data = get_bootstrap()

    players_map = {p["id"]: p for p in bootstrap_data["elements"]}

    # Convert Player objects to IDs if needed
    def get_id(p):
        if hasattr(p, 'id'):
            return p.id  # Player object
        elif isinstance(p, dict) and 'id' in p:
            return p['id']  # Dict
        else:
            return p  # Already an ID

    def get_pos_from_id(pid):
        player = players_map.get(pid)
        if player:
            return ["GKP", "DEF", "MID", "FWD"][player["element_type"] - 1]
        return None

    # Group by position
    out_by_pos = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
    in_by_pos = {"GKP": [], "DEF": [], "MID": [], "FWD": []}

    for p in transfers_out:
        pid = get_id(p)
        pos = get_pos_from_id(pid)
        if pos:
            out_by_pos[pos].append(pid)

    for p in transfers_in:
        pid = get_id(p)
        pos = get_pos_from_id(pid)
        if pos:
            in_by_pos[pos].append(pid)

    # Match by position
    matched_changes = []
    for pos in ["GKP", "DEF", "MID", "FWD"]:
        out_pids = out_by_pos[pos]
        in_pids = in_by_pos[pos]

        # Pair up players in this position
        for i in range(min(len(out_pids), len(in_pids))):
            matched_changes.append({
                "out": out_pids[i],
                "in": in_pids[i]
            })

    return matched_changes


def recommend_weekly_transfers(
    max_transfers: Optional[int] = None,
    planning_horizon: int = 5,
    consider_hits: bool = False,
    entry_id: Optional[int] = None,
    evaluate_banking: bool = True
) -> Dict[str, Any]:
    """
    Recommend transfers for the upcoming gameweek.
    
    Args:
        max_transfers: Override max transfers to consider (uses free transfers if None)
        planning_horizon: How many GWs ahead to optimize for
        consider_hits: Whether to consider taking hits for additional transfers
        entry_id: FPL team ID (optional, uses myteam_latest.json)
        evaluate_banking: Compare banking vs using transfer now
    
    Returns:
        Dictionary with transfer recommendations
    """
    # Load current team and transfer info
    current_squad, transfer_info = load_current_team(entry_id)
    log.info(f"Loaded current squad with {len(current_squad)} players")
    
    # Get actual free transfers available
    free_transfers = transfer_info.get('limit', 1)  # Default to 1 if not found
    bank = transfer_info.get('bank', 0) / 10  # Convert to millions
    team_value = transfer_info.get('value', 1000) / 10
    
    log.info(f"Free transfers available: {free_transfers}, Bank: £{bank:.1f}m, Team value: £{team_value:.1f}m")
    
    # Use actual free transfers unless overridden
    if max_transfers is None:
        max_transfers = free_transfers
    
    # Get recommendations for different scenarios
    scenarios = []
    
    # Common optimization parameters
    opt_params = {
        "horizon": planning_horizon,
        "use_advanced": True,  # Use LP optimization
        "bench_weight": 0.1,
        "differential_bonus": 0.02,  # Reduced to avoid chasing differentials
        "risk_penalty": 0.15,  # Increased to avoid injury-prone players
        "value_weight": 0.0,  # No value weighting - we care about points!
        # Note: bank is now handled internally by reading myteam_latest.json for accurate budget calculation
    }
    
    # Always evaluate current squad (baseline)
    baseline = optimize_transfers(
        current_squad=current_squad,
        max_transfers=0,
        wildcard=False,
        **opt_params
    )
    if "error" not in baseline:
        squad_ids = [p["id"] for p in baseline.get("squad", [])]
        # Get the expected points - try different fields
        baseline_ep = baseline.get("expected_points_total", 0) or baseline.get("expected_points_gw1", 0)
        scenarios.append({
            "transfers": 0,
            "cost": 0,
            "expected_points": baseline_ep,
            "net_points": baseline_ep,
            "squad": squad_ids,
            "changes": [],
            "description": "Hold current squad",
            "full_result": baseline  # Store full result for later reference
        })
    
    # Evaluate transfers up to free transfers available
    for num_transfers in range(1, free_transfers + 1):
        result = optimize_transfers(
            current_squad=current_squad,
            max_transfers=num_transfers,
            wildcard=False,
            **opt_params
        )
        if "error" not in result:
            # Load bootstrap data for position matching
            from ..data.fpl_api import get_bootstrap
            bootstrap = get_bootstrap()

            # Handle two-stage optimization result structure
            if "transfers_out" in result and "transfers_in" in result:
                # Two-stage optimization returns transfers directly
                # Match transfers by position so each pair makes sense
                changes = match_transfers_by_position(
                    result["transfers_out"],
                    result["transfers_in"],
                    bootstrap
                )
                squad_ids = result.get("squad", [p["id"] for p in result.get("squad", [])])
            else:
                # Regular optimization returns squad
                squad_ids = [p["id"] for p in result.get("squad", [])]
                changes = identify_changes(current_squad, squad_ids)
            
            # Calculate EP gain if available
            ep_gain = None
            if "optimization_score" in result:
                ep_gain = result["optimization_score"]
            elif "expected_points_gw1" in result and "expected_points_gw1" in baseline:
                ep_gain = result["expected_points_gw1"] - baseline["expected_points_gw1"]
            
            # Validate transfer quality
            if not validate_transfer_quality(changes, current_squad, ep_gain=ep_gain):
                log.info(f"Skipping {num_transfers} transfer(s) - failed quality check")
                continue
                
            # Get expected points - handle different result structures
            expected_pts = result.get("expected_points_gw1", 0)
            if expected_pts == 0:
                expected_pts = result.get("expected_points_total", 0)
            
            scenarios.append({
                "transfers": num_transfers,
                "cost": 0,  # Free transfers
                "expected_points": expected_pts,
                "net_points": expected_pts,
                "squad": squad_ids,
                "changes": changes,
                "description": f"Use {num_transfers} free transfer{'s' if num_transfers > 1 else ''}",
                "full_result": result  # Store full result for lineup details
            })
    
    # Consider taking hits if enabled
    if consider_hits:
        # Only consider 1-2 extra transfers beyond free ones
        for extra_transfers in range(1, min(3, 15 - free_transfers)):
            num_transfers = free_transfers + extra_transfers
            hit_cost = extra_transfers * 4  # -4 per extra transfer
            
            result = optimize_transfers(
                current_squad=current_squad,
                max_transfers=num_transfers,
                wildcard=False,
                **opt_params
            )
            if "error" not in result:
                # Load bootstrap data for position matching
                from ..data.fpl_api import get_bootstrap
                bootstrap = get_bootstrap()

                # Handle two-stage optimization results
                if "transfers_out" in result and "transfers_in" in result:
                    # Match transfers by position
                    changes = match_transfers_by_position(
                        result["transfers_out"],
                        result["transfers_in"],
                        bootstrap
                    )
                    squad_ids = result.get("squad", [p["id"] for p in result.get("squad", [])])
                else:
                    # Regular optimization
                    squad_ids = [p["id"] for p in result.get("squad", [])]
                    changes = identify_changes(current_squad, squad_ids)
                
                # Calculate EP gain if available
                ep_gain = None
                if "optimization_score" in result:
                    ep_gain = result["optimization_score"]
                elif "expected_points_gw1" in result and "expected_points_gw1" in baseline:
                    ep_gain = result["expected_points_gw1"] - baseline["expected_points_gw1"]
                
                # Validate transfer quality - be stricter for hits
                if not validate_transfer_quality(changes, current_squad, taking_hit=True, ep_gain=ep_gain):
                    log.info(f"Skipping {num_transfers} transfers with hit - failed quality check")
                    continue
                
                expected_pts = result.get("expected_points_total", 0)
                # Skip if optimization failed or returned invalid points
                if expected_pts <= 0:
                    log.info(f"Skipping {num_transfers} transfers with hit - optimization returned no valid points")
                    continue
                scenarios.append({
                    "transfers": num_transfers,
                    "cost": hit_cost,
                    "expected_points": expected_pts,
                    "net_points": expected_pts - hit_cost,
                    "squad": squad_ids,
                    "changes": changes,
                    "description": f"{num_transfers} transfers (-{hit_cost} hit)",
                    "full_result": result  # Store full result for lineup details
                })
    
    # Evaluate banking strategy if requested and we have less than 5 free transfers (FPL cap)
    banking_analysis = None
    if evaluate_banking and free_transfers < 5:  # FPL now caps at 5, not 2
        next_week_ft = min(free_transfers + 1, 5)  # Cap at 5
        log.info(f"Evaluating banking strategy (save transfer for {next_week_ft} FT next week)")
        
        # Find best scenario with current free transfers
        current_ft_scenarios = [s for s in scenarios if s["transfers"] <= free_transfers and s["cost"] == 0]
        best_current = max(current_ft_scenarios, key=lambda x: x["net_points"]) if current_ft_scenarios else None
        
        # Calculate actual EP gain for current best option (improvement over baseline)
        current_best_ep_gain = 0
        if best_current:
            # ALWAYS calculate as improvement over baseline, not absolute score
            baseline_scenario = next((s for s in scenarios if s["transfers"] == 0), None)
            if baseline_scenario:
                current_best_ep_gain = best_current["net_points"] - baseline_scenario["net_points"]
            elif best_current["transfers"] > 0 and "full_result" in best_current:
                # Fallback: if no baseline, try to use objective_improvement if available
                current_best_ep_gain = best_current["full_result"].get("objective_improvement", 0)
        
        # Simulate having more free transfers next week if we bank
        log.info(f"Simulating {next_week_ft} free transfers for banking comparison")
        banked_result = optimize_transfers(
            current_squad=current_squad,
            max_transfers=next_week_ft,
            wildcard=False,
            **opt_params
        )
        
        if "error" not in banked_result:
            # Load bootstrap data for position matching
            from ..data.fpl_api import get_bootstrap
            bootstrap = get_bootstrap()

            # Handle two-stage optimization results
            if "transfers_out" in banked_result and "transfers_in" in banked_result:
                # Two-stage optimization returns transfers directly
                # Match transfers by position so each pair makes sense
                banked_changes = match_transfers_by_position(
                    banked_result["transfers_out"],
                    banked_result["transfers_in"],
                    bootstrap
                )
                banked_squad_ids = banked_result.get("squad", [])
            else:
                # Regular optimization returns squad
                banked_squad_ids = [p["id"] for p in banked_result.get("squad", [])]
                banked_changes = identify_changes(current_squad, banked_squad_ids)
            
            # Calculate EP gain for banking scenario
            banked_ep_gain = None
            
            # Check if this is from two-stage optimization (has transfers_out/transfers_in)
            if "transfers_out" in banked_result and "transfers_in" in banked_result and "optimization_score" in banked_result:
                # Two-stage optimization - optimization_score is the EP gain
                banked_ep_gain = banked_result["optimization_score"]
            else:
                # Regular LP optimization or fallback - need to compare against baseline
                # Get the baseline expected points (no transfer scenario)
                baseline_scenario = next((s for s in scenarios if s["transfers"] == 0), None)
                if baseline_scenario:
                    banked_expected = banked_result.get("expected_points_gw1", 0) or banked_result.get("expected_points_total", 0)
                    baseline_expected = baseline_scenario.get("expected_points", 0) or baseline_scenario.get("net_points", 0)
                    
                    if banked_expected > 0 and baseline_expected > 0:
                        # This is the actual EP gain from the transfers
                        banked_ep_gain = banked_expected - baseline_expected
                    else:
                        # Try using the original baseline result
                        if baseline and "expected_points_total" in baseline:
                            baseline_expected = baseline["expected_points_total"]
                            banked_ep_gain = banked_expected - baseline_expected
            
            # Add player names to changes
            from ..data.fpl_api import get_bootstrap
            bootstrap = get_bootstrap()
            players_map = {p["id"]: p for p in bootstrap["elements"]}
            
            # Enhance changes with names
            enhanced_changes = []
            for change in banked_changes:
                p_out = players_map.get(change["out"], {})
                p_in = players_map.get(change["in"], {})
                enhanced_changes.append({
                    "out": change["out"],
                    "in": change["in"],
                    "out_name": p_out.get("web_name", "Unknown"),
                    "in_name": p_in.get("web_name", "Unknown"),
                    "out_price": p_out.get("now_cost", 0),
                    "in_price": p_in.get("now_cost", 0)
                })
            
            # Check if current best transfer is part of the banked combination
            overlapping_transfer = False
            if best_current and len(best_current.get("changes", [])) > 0:
                current_change = best_current["changes"][0]
                for banked_change in banked_changes:
                    if banked_change["out"] == current_change["out"] and banked_change["in"] == current_change["in"]:
                        overlapping_transfer = True
                        break
            
            # Adjust banking advantage calculation
            if overlapping_transfer:
                # If the best single transfer is part of the 2-transfer combo,
                # then banking doesn't give you that transfer benefit earlier
                # The real comparison is: do transfer now + flexibility vs wait for both
                banking_note = "Note: Best single transfer is part of 2-transfer combination"
                # Banking advantage should be negative since you delay the benefit
                actual_banking_advantage = -1.0  # Slight penalty for delaying
            else:
                banking_note = None
                actual_banking_advantage = (banked_ep_gain or 0) - current_best_ep_gain
            
            # Smart banking decision logic
            from .banking_logic import evaluate_banking_decision, format_banking_recommendation
            from ..utils.io import read_parquet
            from pathlib import Path
            
            # Load EP and xmins predictions for decision logic
            try:
                ep_df = read_parquet(Path("data/processed/exp_points.parquet"))
                xmins_df = read_parquet(Path("data/processed/xmins.parquet"))
                ep_map = dict(zip(ep_df['player_id'], ep_df['ep_adjusted']))
                xmins_map = dict(zip(xmins_df['player_id'], xmins_df['xmins']))
            except:
                log.warning("Could not load predictions for banking logic")
                ep_map = {}
                xmins_map = {}
            
            # Get current changes from best scenario
            current_changes_for_logic = []
            if best_current and 'changes' in best_current:
                current_changes_for_logic = best_current['changes']
            
            # Make intelligent banking decision
            should_bank, reasoning, metrics = evaluate_banking_decision(
                current_squad=current_squad,
                banking_advantage=actual_banking_advantage,
                best_current_gain=current_best_ep_gain,
                banked_changes=banked_changes,
                current_changes=current_changes_for_logic,
                ep_predictions=ep_map,
                xmins_predictions=xmins_map
            )
            
            banking_analysis = {
                "evaluated": True,
                "strategy": "bank",
                "current_week_gain": 0,  # No transfer this week if banking
                "next_week_gain": banked_ep_gain or 0,
                "total_gain": banked_ep_gain or 0,
                "banked_changes": enhanced_changes,
                "banked_result": banked_result,
                "overlapping_transfer": overlapping_transfer,
                "banking_note": banking_note,
                "current_free_transfers": free_transfers,
                "next_week_free_transfers": next_week_ft,
                "comparison": {
                    "best_now": current_best_ep_gain,  # Best option with current FTs
                    "bank_for_next": banked_ep_gain or 0,  # Best option with more FTs next week
                    "banking_advantage": actual_banking_advantage
                },
                "decision": {
                    "should_bank": should_bank,
                    "reasoning": reasoning,
                    "metrics": metrics,
                    "formatted": format_banking_recommendation(
                        should_bank=should_bank,
                        reasoning=reasoning,
                        metrics=metrics,
                        banking_advantage=actual_banking_advantage,
                        current_ft=free_transfers,
                        next_week_ft=next_week_ft
                    )
                }
            }
            
            log.info(f"Banking analysis: Best with {free_transfers} FT now = {banking_analysis['comparison']['best_now']:.2f} EP gain, "
                    f"Banking for {next_week_ft} FT = {banking_analysis['comparison']['bank_for_next']:.2f} EP gain, "
                    f"Advantage = {banking_analysis['comparison']['banking_advantage']:.2f} EP")
    
    # Find best scenario by net points
    if scenarios:
        best_scenario = max(scenarios, key=lambda x: x["net_points"])
        baseline_points = scenarios[0]["net_points"] if scenarios else 0  # No transfer baseline
        baseline_scenario = scenarios[0] if scenarios else None  # Hold scenario

        # Decide what to actually recommend
        recommended_scenario = best_scenario

        # Check if banking was evaluated and recommends banking
        if banking_analysis and "decision" in banking_analysis:
            if banking_analysis["decision"].get("should_bank", False):
                # Banking recommended - use the hold scenario (current squad)
                recommended_scenario = baseline_scenario
                log.info("Banking recommended - using current squad lineup")
        else:
            # No banking recommendation, check if transfers are worth making
            if best_scenario["transfers"] > free_transfers:
                # Taking a hit - be stricter
                net_gain = best_scenario["net_points"] - baseline_points
                if net_gain < 4.0:  # Less than 4 points for a hit, not worth it
                    recommended_scenario = baseline_scenario
            elif best_scenario["transfers"] > 0:
                # Free transfers - still check if worthwhile
                net_gain = best_scenario["net_points"] - baseline_points
                if net_gain < 1.0:  # Less than 1 point gain, just hold
                    recommended_scenario = baseline_scenario

        # Use the lineup from the scenario we're actually recommending
        full_result = recommended_scenario.get("full_result", {})

        # Get actual EP gain (improvement over baseline, NOT absolute score)
        ep_gain = 0
        if recommended_scenario["transfers"] > 0 and baseline_scenario:
            # Calculate as improvement: new EP - baseline EP
            ep_gain = recommended_scenario["expected_points"] - baseline_scenario["expected_points"]
            # Note: ep_gain is the raw points gain before considering hit costs
            # The net gain after hits is shown in the scenario analysis

        # Add names to changes
        from ..data.fpl_api import get_bootstrap
        bootstrap = get_bootstrap()
        players_map = {p["id"]: p for p in bootstrap["elements"]}

        enhanced_changes = []
        for change in recommended_scenario["changes"]:
            p_out = players_map.get(change["out"], {})
            p_in = players_map.get(change["in"], {})
            enhanced_changes.append({
                "out": change["out"],
                "in": change["in"],
                "out_name": p_out.get("web_name", "Unknown"),
                "in_name": p_in.get("web_name", "Unknown"),
                "out_price": p_out.get("now_cost", 0),
                "in_price": p_in.get("now_cost", 0)
            })

        # Build human-readable output with clear transfer section
        output_lines = []

        # Header with transfer count
        if recommended_scenario["transfers"] > 0:
            output_lines.append("=" * 60)
            output_lines.append(f"RECOMMENDED TRANSFERS: {recommended_scenario['transfers']}")
            output_lines.append("=" * 60)
            output_lines.append("")

            # Show transfers clearly
            for i, change in enumerate(enhanced_changes, 1):
                out_name = change["out_name"]
                in_name = change["in_name"]
                out_price = change["out_price"] / 10
                in_price = change["in_price"] / 10
                diff = (change["in_price"] - change["out_price"]) / 10
                diff_str = f"(+£{diff:.1f}m)" if diff > 0 else f"(£{diff:.1f}m)" if diff < 0 else ""

                output_lines.append(f"{i}. OUT: {out_name} (£{out_price:.1f}m)")
                output_lines.append(f"   IN:  {in_name} (£{in_price:.1f}m) {diff_str}")
                output_lines.append("")

            # Show EP gain
            if ep_gain > 0:
                output_lines.append(f"Expected points gain: +{ep_gain:.1f} over {planning_horizon} gameweeks")
            if recommended_scenario["cost"] > 0:
                output_lines.append(f"Hit cost: -{recommended_scenario['cost']} points")
                net_gain = ep_gain - recommended_scenario['cost']
                output_lines.append(f"Net gain: {net_gain:+.1f} points")
            output_lines.append("")

        # Add the squad/lineup details from optimizer
        squad_output = full_result.get("human_readable", "")
        if squad_output:
            output_lines.append(squad_output)

        # Format recommendation
        recommendation = {
            "recommended_transfers": recommended_scenario["transfers"],
            "transfer_cost": recommended_scenario["cost"],
            "expected_gain": ep_gain,
            "changes": enhanced_changes,
            "scenarios": scenarios,
            "current_squad": current_squad,
            "new_squad": recommended_scenario["squad"],
            "planning_horizon": planning_horizon,
            "free_transfers": free_transfers,
            "bank": bank,
            # Add lineup details from the RECOMMENDED scenario, not necessarily the best
            "optimization_result": full_result,
            "human_readable": "\n".join(output_lines),
            "banking_analysis": banking_analysis
        }
        
        return recommendation
    
    return {"error": "Could not generate transfer recommendations"}


def identify_changes(current_squad: List[int], new_squad: List[int]) -> List[Dict[str, Any]]:
    """Identify transfers between two squads, matched by position."""
    current_set = set(current_squad)
    new_set = set(new_squad)

    transfers_out = list(current_set - new_set)
    transfers_in = list(new_set - current_set)

    # Match transfers by position so each pair is the same position
    from ..data.fpl_api import get_bootstrap
    bootstrap = get_bootstrap()

    changes = match_transfers_by_position(transfers_out, transfers_in, bootstrap)

    return changes


def validate_transfer_quality(
    changes: List[Dict[str, Any]], 
    current_squad: List[int],
    taking_hit: bool = False,
    ep_gain: Optional[float] = None
) -> bool:
    """
    Validate if transfers make sense using common sense rules.
    
    Returns True if transfers are sensible, False otherwise.
    """
    if not changes:
        return True
    
    # If there's a significant EP gain (5+ points over horizon), allow the transfer
    # This handles cases like Frimpong -> Virgil with 8+ EP gain
    if ep_gain and ep_gain >= 5.0:
        log.info(f"Allowing transfer with significant EP gain: {ep_gain:.2f}")
        return True
        
    from ..data.fpl_api import get_bootstrap
    from ..data.team_quality import get_team_tier
    
    bootstrap = get_bootstrap()
    players_map = {p["id"]: p for p in bootstrap["elements"]}
    teams_map = {t["id"]: t["name"] for t in bootstrap["teams"]}
    
    for change in changes:
        p_out = players_map.get(change["out"], {})
        p_in = players_map.get(change["in"], {})
        
        if not p_out or not p_in:
            continue
            
        team_out = teams_map.get(p_out.get("team"), "")
        team_in = teams_map.get(p_in.get("team"), "")
        
        # Use data-driven team tiers based on historical performance
        tier_out = get_team_tier(team_out)  # Returns 0-3 (0=weak, 3=top)
        tier_in = get_team_tier(team_in)
        
        # Block downgrades without good justification
        if tier_out > tier_in:
            form_out = float(p_out.get("form", 0))
            form_in = float(p_in.get("form", 0))
            cost_diff = p_out.get("now_cost", 0) - p_in.get("now_cost", 0)
            
            # For big tier drops (e.g., top 6 to promoted), be very strict
            if tier_out - tier_in >= 2:
                if form_in < form_out * 3.0 or cost_diff < 15:
                    log.info(f"Blocking major downgrade: {p_out['web_name']} ({team_out}, tier {tier_out}) "
                            f"-> {p_in['web_name']} ({team_in}, tier {tier_in})")
                    return False
            
            # For minor downgrades, require form improvement or significant savings
            elif tier_out - tier_in == 1:
                if form_in <= form_out and cost_diff < 10:
                    log.info(f"Blocking minor downgrade: {p_out['web_name']} ({team_out}) "
                            f"-> {p_in['web_name']} ({team_in}) - form: {form_out:.1f} -> {form_in:.1f}")
                    return False
        
        # Don't make sideways transfers (similar price, similar team quality)
        cost_diff = abs(p_out.get("now_cost", 0) - p_in.get("now_cost", 0))
        if cost_diff < 5:  # Less than £0.5m difference
            # Check if teams are of similar quality using data-driven tiers
            similar_quality = abs(tier_out - tier_in) <= 1  # Within 1 tier
            
            if similar_quality:
                form_diff = abs(float(p_out.get("form", 0)) - float(p_in.get("form", 0)))
                if form_diff < 3.0:  # Not enough form difference
                    log.info(f"Blocking sideways transfer: {p_out['web_name']} (tier {tier_out}) -> {p_in['web_name']} (tier {tier_in})")
                    return False
        
        # If taking a hit, require significant improvement
        if taking_hit:
            # Expected points should be at least 4 points better over planning horizon
            # This check would need access to EP data - simplified for now
            form_improvement = float(p_in.get("form", 0)) - float(p_out.get("form", 0))
            if form_improvement < 4.0:
                log.info(f"Blocking hit transfer: Not enough improvement for -4 hit")
                return False
    
    return True


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
    horizon = recommendation.get("planning_horizon", 5)
    free_transfers = recommendation.get("free_transfers", 1)
    bank = recommendation.get("bank", 0)
    
    output.append(f"\nFree transfers available: {free_transfers}")
    output.append(f"Bank: £{bank:.1f}m")
    
    if rec_transfers == 0:
        output.append(f"\n✅ HOLD - No transfers recommended")
        output.append(f"Your current team is well-optimized for the next {horizon} gameweeks.")
    else:
        output.append(f"\n📊 Recommended: {rec_transfers} transfer{'s' if rec_transfers > 1 else ''}")
        if cost > 0:
            output.append(f"Hit cost: -{cost} points")
        # Show net gain after accounting for hit cost
        net_gain = expected_gain - cost
        if net_gain > 0:
            output.append(f"Expected net gain over next {horizon} GWs: +{net_gain:.1f} points")
        else:
            output.append(f"Expected net gain over next {horizon} GWs: {net_gain:.1f} points")
        
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
    
    # Show proposed lineup
    if "human_readable" in recommendation and recommendation["human_readable"]:
        hr = recommendation["human_readable"]
        if "OPTIMAL LINEUP" in hr:
            output.append("\n" + "-" * 40)
            # Show appropriate title based on recommendation
            if recommendation["recommended_transfers"] == 0:
                # Check if this is due to banking
                if "banking_analysis" in recommendation and recommendation["banking_analysis"]:
                    if recommendation["banking_analysis"].get("decision", {}).get("should_bank", False):
                        output.append("CURRENT LINEUP (BANKING TRANSFER):")
                    else:
                        output.append("CURRENT LINEUP (OPTIMIZED):")
                else:
                    output.append("CURRENT LINEUP (OPTIMIZED):")
            else:
                output.append("PROPOSED LINEUP AFTER TRANSFERS:")
            
            # Extract the lineup section from human_readable
            lines = hr.split('\n')
            in_lineup = False
            lineup_lines = []
            
            for line in lines:
                if "OPTIMAL LINEUP" in line:
                    in_lineup = True
                    lineup_lines.append(line.replace("=== ", "").replace(" ===", ""))
                elif in_lineup:
                    lineup_lines.append(line)
                    if "Vice-Captain:" in line:
                        break
            
            # Add the lineup to output
            for line in lineup_lines:
                output.append(line)
    
    # Banking decision (if evaluated)
    if "banking_analysis" in recommendation and recommendation["banking_analysis"]:
        banking_info = recommendation["banking_analysis"]
        if "decision" in banking_info:
            decision = banking_info["decision"]
            output.append("\n" + "-" * 40)
            output.append("BANKING ANALYSIS:")
            output.append(decision.get("formatted", "Banking analysis not available"))
    
    # Scenario comparison
    output.append("\n" + "-" * 40)
    output.append(f"SCENARIO ANALYSIS (cumulative over {horizon} GWs):")
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