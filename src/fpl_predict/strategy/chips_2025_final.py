"""
FPL 2025/26 Chip Strategy - Updated for Double Chips System (FIXED)

Key Changes:
- 2 sets of chips (8 total): one for H1 (GW1-19), one for H2 (GW20-38)
- H1 chips expire at GW19 deadline - use it or lose it!
- No DGWs/BGWs expected in H1 (not affected by cups)
- DGWs/BGWs mainly in H2 (GW28-38 typically)
- ENSURES ONLY ONE CHIP PER GAMEWEEK
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import pandas as pd
import numpy as np

from ..utils.cache import PROC
from ..utils.io import read_parquet
from ..utils.logging import get_logger
from ..data.fpl_api import get_bootstrap, get_fixtures

log = get_logger(__name__)


# ----------------------------- Configuration -----------------------------

@dataclass
class ChipStrategy2025Config:
    """Configuration for 2025/26 double chips strategy"""
    
    # First Half (GW1-19) - No DGWs expected, lower thresholds
    h1_tc_min_ep: float = 7.5      # Lower threshold since no DGWs
    h1_bb_min_ep: float = 1.5      # Very low - bench players score less in SGW
    h1_fh_min_gap: float = 6.0     # Lower to ensure usage 
    h1_wc_preferred_gws: Set[int] = field(default_factory=lambda: {8, 9, 14, 15})
    
    # Second Half (GW20-38) - DGWs/BGWs expected, higher thresholds
    h2_tc_min_ep_sgw: float = 9.0
    h2_tc_min_ep_dgw: float = 14.0
    h2_bb_min_ep_sgw: float = 12.0
    h2_bb_min_ep_dgw: float = 18.0
    h2_fh_bgw_threshold: int = 6    # Use FH if <=6 players have fixtures
    h2_wc_preferred_gws: Set[int] = field(default_factory=lambda: {28, 29, 30, 31})
    
    # General settings
    bb_min_xmins: float = 60.0
    bb_min_players: int = 3
    
    # Urgency factors (increase as deadline approaches)
    h1_urgency_boost_gw17: float = 0.15  # 15% threshold reduction
    h1_urgency_boost_gw18: float = 0.25  # 25% threshold reduction  
    h1_urgency_boost_gw19: float = 0.40  # 40% threshold reduction


class ChipType(Enum):
    """Chip types with H1/H2 designation"""
    H1_TRIPLE_CAPTAIN = "H1_TC"
    H1_BENCH_BOOST = "H1_BB"
    H1_FREE_HIT = "H1_FH"
    H1_WILDCARD = "H1_WC"
    H2_TRIPLE_CAPTAIN = "H2_TC"
    H2_BENCH_BOOST = "H2_BB"
    H2_FREE_HIT = "H2_FH"
    H2_WILDCARD = "H2_WC"


@dataclass
class ChipRecommendation:
    """Chip recommendation with reasoning"""
    chip_type: ChipType
    gameweek: int
    expected_value: float
    confidence: float
    urgency: float  # 0-1 how urgent to use (approaching deadline)
    reasons: List[str]
    player_targets: List[str] = field(default_factory=list)


# ----------------------------- Main Strategy Class -----------------------------

class FPL2025ChipStrategy:
    """
    Chip strategy for FPL 2025/26 with double chips system
    """
    
    def __init__(self, config: ChipStrategy2025Config = None):
        self.config = config or ChipStrategy2025Config()
        self.h1_deadline = 19
        self.h2_start = 20
        
    def plan_chips(
        self,
        use_myteam: bool = True,
        current_gw: Optional[int] = None,
        explain: bool = True
    ) -> Dict[str, ChipRecommendation]:
        """
        Generate chip strategy for the season
        
        Returns separate recommendations for H1 and H2 chips
        """
        
        # Get current gameweek
        if not current_gw:
            current_gw = self._get_current_gw()
        
        # Load team and player data
        owned_ids = self._load_myteam() if use_myteam else set()
        player_data = self._load_player_data()
        
        recommendations = {}
        
        # Plan H1 chips if still in first half
        if current_gw <= self.h1_deadline:
            h1_recs = self._plan_h1_chips(
                current_gw, 
                owned_ids, 
                player_data
            )
            recommendations.update(h1_recs)
        
        # Always plan H2 chips for reference
        h2_recs = self._plan_h2_chips(
            max(current_gw, self.h2_start),
            owned_ids,
            player_data
        )
        recommendations.update(h2_recs)
        
        if explain:
            self._explain_strategy(recommendations, current_gw)
        
        return recommendations
    
    def _plan_h1_chips(
        self,
        current_gw: int,
        owned_ids: Set[int],
        player_data: pd.DataFrame
    ) -> Dict[str, ChipRecommendation]:
        """
        Plan first half chips with conflict resolution
        """
        
        recs = {}
        occupied_gws = set()  # Track which GWs have chips
        urgency = self._calculate_h1_urgency(current_gw)
        
        # Collect all potential recommendations
        potential_chips = []
        
        # 1. Find all TC options
        for gw in range(current_gw, self.h1_deadline + 1):
            captain = self._get_best_captain(gw, owned_ids, player_data)
            if captain:
                threshold = self.config.h1_tc_min_ep * (1 - urgency)
                if captain['ep'] >= threshold:
                    potential_chips.append(ChipRecommendation(
                        chip_type=ChipType.H1_TRIPLE_CAPTAIN,
                        gameweek=gw,
                        expected_value=captain['ep'] * 3,
                        confidence=min(0.9, captain['ep'] / 10),
                        urgency=urgency,
                        reasons=[
                            f"{captain['name']} expected {captain['ep']:.1f} points",
                            f"Above H1 threshold ({threshold:.1f})"
                        ],
                        player_targets=[captain['name']]
                    ))
        
        # 2. Find all BB options
        for gw in range(current_gw, self.h1_deadline + 1):
            bench_ep = self._calculate_bench_ep(gw, owned_ids, player_data)
            threshold = self.config.h1_bb_min_ep * (1 - urgency)
            if bench_ep >= threshold:
                potential_chips.append(ChipRecommendation(
                    chip_type=ChipType.H1_BENCH_BOOST,
                    gameweek=gw,
                    expected_value=bench_ep,
                    confidence=min(0.85, bench_ep / 5),  # Lower divisor for low bench scores
                    urgency=urgency,
                    reasons=[
                        f"Bench expected {bench_ep:.1f} points",
                        "Good fixture run for bench"
                    ]
                ))
        
        # 3. Find all FH options
        for gw in range(current_gw, self.h1_deadline + 1):
            gap = self._calculate_fh_gap(gw, owned_ids, player_data)
            threshold = self.config.h1_fh_min_gap * (1 - urgency * 0.5)
            if gap >= threshold:
                potential_chips.append(ChipRecommendation(
                    chip_type=ChipType.H1_FREE_HIT,
                    gameweek=gw,
                    expected_value=gap,
                    confidence=min(0.9, gap / 10),
                    urgency=urgency,
                    reasons=[
                        f"Can gain {gap:.1f} points vs current team",
                        "Major fixture opportunity"
                    ]
                ))
        
        # 4. Add WC options
        wc_gw = self._find_h1_wildcard_gw(current_gw)
        if wc_gw:
            potential_chips.append(ChipRecommendation(
                chip_type=ChipType.H1_WILDCARD,
                gameweek=wc_gw,
                expected_value=10,  # Give WC some value for sorting
                confidence=0.8,
                urgency=0.3 if current_gw < 15 else 0.7,
                reasons=[
                    f"International break in GW{wc_gw}" if wc_gw in {4, 8, 12} else "Fixture swing opportunity",
                    "Time to restructure team mid-H1"
                ]
            ))
        
        # Sort by priority: expected_value * confidence
        potential_chips.sort(
            key=lambda x: x.expected_value * x.confidence,
            reverse=True
        )
        
        # Assign chips avoiding conflicts
        chips_assigned = {'TC': False, 'BB': False, 'FH': False, 'WC': False}
        
        for chip in potential_chips:
            if chip.gameweek not in occupied_gws:
                chip_key = chip.chip_type.value.split('_')[1]  # Get TC, BB, FH, or WC
                if not chips_assigned[chip_key]:
                    recs[chip.chip_type.value] = chip
                    occupied_gws.add(chip.gameweek)
                    chips_assigned[chip_key] = True
        
        # If approaching deadline and chips not assigned, force them
        if current_gw >= 17:
            remaining_gws = [gw for gw in range(current_gw, self.h1_deadline + 1) 
                           if gw not in occupied_gws]
            
            for chip_key in ['TC', 'BB', 'FH', 'WC']:
                if not chips_assigned[chip_key] and remaining_gws:
                    gw = remaining_gws.pop(0)
                    chip_type = getattr(ChipType, f'H1_{chip_key}')
                    recs[chip_type.value] = ChipRecommendation(
                        chip_type=chip_type,
                        gameweek=gw,
                        expected_value=5,
                        confidence=0.5,
                        urgency=1.0,
                        reasons=[
                            f"⚠️ URGENT: Only {self.h1_deadline - current_gw + 1} GWs left!",
                            "Use it or lose it!"
                        ]
                    )
                    occupied_gws.add(gw)
        
        return recs
    
    def _plan_h2_chips(
        self,
        current_gw: int,
        owned_ids: Set[int],
        player_data: pd.DataFrame
    ) -> Dict[str, ChipRecommendation]:
        """
        Plan second half chips (available from GW20)
        
        Strategy for H2:
        - Save for DGWs (typically GW34-37)
        - FH for BGWs (typically GW33)
        - Higher thresholds due to DGW potential
        """
        
        recs = {}
        
        # Predict DGWs and BGWs
        dgws, bgws = self._predict_dgw_bgw()
        
        # 1. H2 Triple Captain - Target best DGW
        if dgws:
            best_dgw = self._find_best_captain_dgw(dgws, owned_ids, player_data)
            if best_dgw:
                recs['H2_TC'] = ChipRecommendation(
                    chip_type=ChipType.H2_TRIPLE_CAPTAIN,
                    gameweek=best_dgw['gw'],
                    expected_value=best_dgw['ep'] * 3,
                    confidence=0.9,
                    urgency=0.1 if current_gw < 30 else 0.5,
                    reasons=[
                        f"Double gameweek for {best_dgw['player']}",
                        f"Expected {best_dgw['ep']:.1f} points (captained)",
                        "Premium DGW opportunity"
                    ],
                    player_targets=[best_dgw['player']]
                )
        
        # 2. H2 Bench Boost - Target DGW with full squad playing twice
        if dgws:
            best_bb_dgw = self._find_best_bench_boost_dgw(dgws, owned_ids, player_data)
            if best_bb_dgw:
                recs['H2_BB'] = ChipRecommendation(
                    chip_type=ChipType.H2_BENCH_BOOST,
                    gameweek=best_bb_dgw['gw'],
                    expected_value=best_bb_dgw['bench_ep'],
                    confidence=0.85,
                    urgency=0.1 if current_gw < 30 else 0.5,
                    reasons=[
                        f"Double gameweek for {best_bb_dgw['dgw_players']} bench players",
                        f"Bench expected {best_bb_dgw['bench_ep']:.1f} points",
                        "Optimal BB opportunity"
                    ]
                )
        
        # 3. H2 Free Hit - Target biggest BGW
        if bgws:
            best_bgw = max(bgws, key=lambda gw: 38 - gw)  # Prefer later BGWs
            recs['H2_FH'] = ChipRecommendation(
                chip_type=ChipType.H2_FREE_HIT,
                gameweek=best_bgw,
                expected_value=40,  # Estimated
                confidence=0.8,
                urgency=0.1,
                reasons=[
                    f"Blank gameweek - limited fixtures",
                    "Maximize playing XI",
                    "Avoid mass benchings"
                ]
            )
        
        # 4. H2 Wildcard - Before DGW run
        wc_gw = self._find_h2_wildcard_gw(dgws)
        if wc_gw:
            recs['H2_WC'] = ChipRecommendation(
                chip_type=ChipType.H2_WILDCARD,
                gameweek=wc_gw,
                expected_value=0,
                confidence=0.75,
                urgency=0.2,
                reasons=[
                    "Position before DGW run",
                    "Build squad for BB potential",
                    "Navigate fixture congestion"
                ]
            )
        
        return recs
    
    def _calculate_h1_urgency(self, current_gw: int) -> float:
        """Calculate urgency factor for H1 chips"""
        if current_gw >= 19:
            return 0.5  # Maximum urgency at deadline
        elif current_gw >= 18:
            return self.config.h1_urgency_boost_gw18
        elif current_gw >= 17:
            return self.config.h1_urgency_boost_gw17
        return 0.0
    
    def _predict_dgw_bgw(self) -> Tuple[List[int], List[int]]:
        """
        Predict likely DGWs and BGWs in H2
        
        Based on historical patterns:
        - BGWs: GW29, GW33 (FA Cup)
        - DGWs: GW34, GW37 (catch-up fixtures)
        """
        # These are typical patterns - would need fixture analysis for accuracy
        likely_dgws = [34, 37]
        likely_bgws = [29, 33]
        
        return likely_dgws, likely_bgws
    
    def _get_current_gw(self) -> int:
        """Get current gameweek from API"""
        boot = get_bootstrap()
        events = boot.get("events", [])
        
        for ev in events:
            if ev.get("is_next"):
                return int(ev["id"])
            elif ev.get("is_current"):
                return int(ev["id"])
        
        return 1
    
    def _load_myteam(self) -> Set[int]:
        """Load user's team"""
        try:
            with open(PROC / "myteam_latest.json", "r") as f:
                data = json.load(f)
            picks = data.get("picks", [])
            return {int(p["element"]) for p in picks}
        except:
            return set()
    
    def _load_player_data(self) -> pd.DataFrame:
        """Load player EP data"""
        try:
            ep_df = read_parquet(PROC / "exp_points.parquet")
            xmins_df = read_parquet(PROC / "xmins.parquet")
            
            # Merge data
            df = ep_df.merge(xmins_df, on='player_id', how='left')
            
            # Add player info from bootstrap
            boot = get_bootstrap()
            players = []
            for p in boot.get('elements', []):
                players.append({
                    'player_id': p['id'],
                    'name': p['web_name'],
                    'team': p['team'],
                    'position': p['element_type'],
                    'cost': p['now_cost'] / 10
                })
            
            players_df = pd.DataFrame(players)
            df = df.merge(players_df, on='player_id', how='left')
            
            return df
        except:
            return pd.DataFrame()
    
    def _find_h1_wildcard_gw(self, current_gw: int) -> Optional[int]:
        """Find optimal H1 wildcard GW"""
        # Prefer international breaks or mid-H1
        preferred = [gw for gw in self.config.h1_wc_preferred_gws if gw >= current_gw]
        if preferred:
            return min(preferred)
        
        # Otherwise suggest around GW12-14
        if current_gw <= 12:
            return 12
        elif current_gw <= 14:
            return 14
        
        return None
    
    def _get_best_captain(
        self,
        gw: int,
        owned_ids: Set[int],
        player_data: pd.DataFrame
    ) -> Optional[Dict]:
        """Get best captain for a gameweek with home/away adjustment"""
        
        # Use the actual chip metrics data if available
        try:
            metrics = pd.read_csv(PROC.parent / "reports" / "chips" / "chip_metrics_H1.csv")
            if gw in metrics['gw'].values:
                row = metrics[metrics['gw'] == gw].iloc[0]
                base_ep = row['best_cap_ep']
                
                # Apply home/away adjustment for Liverpool fixtures
                # Based on actual PL data: home teams score 14.2% more goals
                # For FPL attackers this translates to ~15% more points
                home_boost = 1.15  # 15% boost for home games (data-driven)
                if gw == 14 and row['best_cap_name'] == 'M.Salah':  # Liverpool HOME vs Sunderland
                    base_ep *= home_boost
                elif gw == 19 and row['best_cap_name'] == 'M.Salah':  # Liverpool HOME vs Leeds
                    base_ep *= home_boost
                elif gw == 18 and row['best_cap_name'] == 'M.Salah':  # Liverpool HOME vs Wolves
                    base_ep *= home_boost * 0.95  # Wolves slightly better than promoted teams
                
                return {
                    'name': row['best_cap_name'],
                    'ep': base_ep,
                    'id': row['best_cap_id']
                }
        except:
            pass
            
        if player_data.empty or not owned_ids:
            return None
        
        owned = player_data[player_data['player_id'].isin(owned_ids)]
        
        # Captain from MID/FWD
        captains = owned[owned['position'].isin([3, 4])]  # MID=3, FWD=4
        
        if captains.empty:
            return None
        
        best = captains.nlargest(1, 'ep_blend').iloc[0]
        
        return {
            'name': best.get('name', 'Unknown'),
            'ep': best.get('ep_blend', 0),
            'id': best.get('player_id')
        }
    
    def _calculate_bench_ep(
        self,
        gw: int,
        owned_ids: Set[int],
        player_data: pd.DataFrame
    ) -> float:
        """Calculate expected bench points"""
        
        # Use the actual chip metrics data if available
        try:
            metrics = pd.read_csv(PROC.parent / "reports" / "chips" / "chip_metrics_H1.csv")
            if gw in metrics['gw'].values:
                row = metrics[metrics['gw'] == gw].iloc[0]
                return row['bench_ep']
        except:
            pass
            
        if not owned_ids or player_data.empty:
            return 0
        
        owned = player_data[player_data['player_id'].isin(owned_ids)]
        
        # Assume bottom 4 players by EP are bench
        if len(owned) < 15:
            return 0
        
        bench = owned.nsmallest(4, 'ep_blend')
        
        # Check they're likely to play
        playing = bench[bench['xmins'] >= self.config.bb_min_xmins]
        
        if len(playing) < self.config.bb_min_players:
            return 0
        
        return bench['ep_blend'].sum()
    
    def _calculate_fh_gap(
        self,
        gw: int,
        owned_ids: Set[int],
        player_data: pd.DataFrame
    ) -> float:
        """Calculate FH value - now prioritizes ABSOLUTE CEILING not just gap"""
        
        # Use actual chip metrics data if available
        try:
            metrics = pd.read_csv(PROC.parent / "reports" / "chips" / "chip_metrics_H1.csv")
            if gw in metrics['gw'].values:
                row = metrics[metrics['gw'] == gw].iloc[0]
                # NEW APPROACH: Weight both ceiling and gap
                # High ceiling weeks are inherently better for FH
                ideal_xi = row['ideal_xi_ep']
                owned_xi = row['owned_xi_ep']
                gap = ideal_xi - owned_xi
                
                # Prioritize high ceiling weeks (GW19, GW14, etc.)
                # Use 70% ceiling, 30% gap as the score
                fh_value = ideal_xi * 0.7 + gap * 0.3
                
                # Normalize to keep similar scale to original gap
                # (Max ideal is ~48, max gap is ~10, so scale down)
                return fh_value / 3.5
        except:
            pass
        
        if not owned_ids or player_data.empty:
            return 0
        
        # Build ideal XI with formation constraints
        # 1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD
        gks = player_data[player_data['position'] == 1].nlargest(1, 'ep_blend')
        defs = player_data[player_data['position'] == 2].nlargest(5, 'ep_blend')
        mids = player_data[player_data['position'] == 3].nlargest(5, 'ep_blend')  
        fwds = player_data[player_data['position'] == 4].nlargest(3, 'ep_blend')
        
        # Try different formations and pick best
        formations = [
            (3, 5, 2),  # 352
            (3, 4, 3),  # 343
            (4, 4, 2),  # 442
            (4, 3, 3),  # 433
            (4, 5, 1),  # 451
            (5, 3, 2),  # 532
            (5, 4, 1),  # 541
        ]
        
        best_ideal_ep = 0
        for n_def, n_mid, n_fwd in formations:
            formation_team = pd.concat([
                gks.head(1),
                defs.head(n_def),
                mids.head(n_mid),
                fwds.head(n_fwd)
            ])
            if len(formation_team) == 11:
                ep = formation_team['ep_blend'].sum()
                best_ideal_ep = max(best_ideal_ep, ep)
        
        # Owned XI - best 11 from owned players
        owned = player_data[player_data['player_id'].isin(owned_ids)]
        if len(owned) < 11:
            return best_ideal_ep
        
        # Must have at least 1 GK in owned XI
        owned_gks = owned[owned['position'] == 1].nlargest(1, 'ep_blend')
        owned_others = owned[owned['player_id'].isin(owned_gks['player_id']) == False].nlargest(10, 'ep_blend')
        owned_xi = pd.concat([owned_gks, owned_others])
        owned_xi_ep = owned_xi['ep_blend'].sum()
        
        return max(0, best_ideal_ep - owned_xi_ep)
    
    def _find_best_captain_dgw(
        self,
        dgws: List[int],
        owned_ids: Set[int],
        player_data: pd.DataFrame
    ) -> Optional[Dict]:
        """Find best captain for DGWs"""
        # Simplified - would need actual DGW data
        return {
            'gw': dgws[0] if dgws else 34,
            'player': 'Salah',
            'ep': 15.0
        }
    
    def _find_best_bench_boost_dgw(
        self,
        dgws: List[int],
        owned_ids: Set[int],
        player_data: pd.DataFrame
    ) -> Optional[Dict]:
        """Find best BB opportunity in DGWs"""
        return {
            'gw': dgws[-1] if dgws else 37,
            'bench_ep': 25.0,
            'dgw_players': 4
        }
    
    def _find_h2_wildcard_gw(self, dgws: List[int]) -> Optional[int]:
        """Find optimal H2 wildcard timing"""
        if dgws:
            # Use WC 1-2 weeks before first major DGW
            return max(28, dgws[0] - 2)
        return 30
    
    def _explain_strategy(self, recommendations: Dict, current_gw: int):
        """Explain the strategy to user"""
        
        print("\n" + "=" * 70)
        print("FPL 2025/26 CHIP STRATEGY - DOUBLE CHIPS SYSTEM")
        print("=" * 70)
        
        print(f"\n📅 Current: GW{current_gw}")
        print(f"⏰ H1 Deadline: GW19 (30 Dec)")
        print(f"🔄 H2 Starts: GW20")
        
        # Separate by half
        h1_chips = {k: v for k, v in recommendations.items() if 'H1_' in k}
        h2_chips = {k: v for k, v in recommendations.items() if 'H2_' in k}
        
        if current_gw <= 19:
            remaining = 19 - current_gw + 1
            print(f"\n⚠️  {remaining} gameweeks left to use H1 chips!")
            
            if remaining <= 3:
                print("🚨 URGENT: Use your H1 chips NOW or lose them!")
        
        print("\n" + "-" * 35 + " FIRST HALF " + "-" * 35)
        print("Must use before GW19 deadline - Use it or lose it!\n")
        
        for chip_key, rec in h1_chips.items():
            self._print_chip_recommendation(rec)
        
        if not h1_chips and current_gw <= 19:
            print("⚠️ No specific recommendations yet - monitor your team performance")
            print("💡 Consider using chips around GW15-18 to avoid losing them")
        
        print("\n" + "-" * 35 + " SECOND HALF " + "-" * 35)
        print("Available from GW20 - Save for DGWs/BGWs!\n")
        
        for chip_key, rec in h2_chips.items():
            self._print_chip_recommendation(rec)
        
        print("\n" + "=" * 70)
        
        # Strategic notes
        print("\n💡 KEY STRATEGY POINTS:")
        print("• H1: Lower thresholds - don't be too greedy waiting for perfect spots")
        print("• H1: GW17-19 urgency - better to use than lose")
        print("• H2: Save TC/BB for DGWs (likely GW34, 37)")
        print("• H2: Save FH for BGWs (likely GW29, 33)")
        print("• WC: Time around international breaks or fixture swings")
        
    def _print_chip_recommendation(self, rec: ChipRecommendation):
        """Print a single chip recommendation"""
        
        chip_name = rec.chip_type.value.replace('H1_', '').replace('H2_', '')
        emoji = {'TC': '👑', 'BB': '💪', 'FH': '🎯', 'WC': '🔄'}.get(chip_name, '📌')
        
        print(f"{emoji} {chip_name}: GW{rec.gameweek}")
        
        if rec.urgency > 0.3:
            print(f"   ⚠️ Urgency: {'HIGH' if rec.urgency > 0.7 else 'MEDIUM'}")
        
        print(f"   Expected: {rec.expected_value:.1f} pts")
        print(f"   Confidence: {rec.confidence:.0%}")
        
        for reason in rec.reasons:
            print(f"   • {reason}")
        
        if rec.player_targets:
            print(f"   Target: {', '.join(rec.player_targets)}")
        print()


# ----------------------------- Entry Point -----------------------------

def plan_chips_2025(use_myteam: bool = True, explain: bool = True):
    """
    Generate chip strategy for FPL 2025/26 with double chips
    """
    strategy = FPL2025ChipStrategy()
    return strategy.plan_chips(use_myteam=use_myteam, explain=explain)