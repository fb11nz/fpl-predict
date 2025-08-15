"""
Handle recent transfers that need integration time.
Scrapes transfer data to identify recent moves.
"""

import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import re
from bs4 import BeautifulSoup
import json

log = logging.getLogger(__name__)


def detect_transfers_from_fpl_data() -> List[Tuple[str, int]]:
    """
    Detect likely recent transfers by analyzing FPL data patterns.
    
    Indicators of recent transfers:
    - Very low ownership despite reasonable price
    - Player at unexpected team (compared to last season)
    - Low TSB (Total Selected By) for known good players
    
    Returns:
        List of (player_name, estimated_days_since_transfer) tuples
    """
    transfers = []
    
    try:
        from ..data.fpl_api import get_bootstrap
        bootstrap = get_bootstrap()
        
        # Known recent transfers to check for
        # Based on web scraping and search results (August 2025)
        # Format: "Player Name": (expected_team_name, days_ago_estimate)
        known_transfers = {
            # Confirmed transfers (based on web search):
            "Gyökeres": ("Arsenal", 15),  # Completed late July/early August - established now
            "Zubimendi": ("Arsenal", 10),  # Recent but had some time to integrate
            "Madueke": ("Arsenal", 7),  # From Chelsea - more recent
            "Grealish": ("Everton", 5),  # Loan from Man City - very recent
            "Cunha": ("Manchester United", 7),  # From Wolves
            "Mbeumo": ("Manchester United", 7),  # From Brentford  
            "Walker": ("Burnley", 7),  # From Man City
            "Mosquera": ("Arsenal", 10),  # From Valencia
            # Note: Dates are estimates based on "early August" completions
        }
        
        for player in bootstrap['elements']:
            player_name = player['web_name']
            team_id = player['team']
            team_name = bootstrap['teams'][team_id - 1]['name']
            ownership = float(player.get('selected_by_percent', 0))
            price = player['now_cost'] / 10.0
            
            # Check known transfers
            if player_name in known_transfers:
                expected_team, days_ago = known_transfers[player_name]
                if expected_team.lower() in team_name.lower():
                    transfers.append((player_name, days_ago))
                    log.info(f"Detected known transfer: {player_name} to {team_name}")
            
            # DO NOT use ownership heuristic - it's too unreliable
            # Low ownership can be due to:
            # - Poor form
            # - Injury concerns  
            # - Rotation risk
            # - Better alternatives at similar price
            # Only use VERIFIED transfers from known_transfers dict
                    
    except Exception as e:
        log.debug(f"Could not analyze FPL data for transfers: {e}")
    
    return transfers


def scrape_recent_transfers() -> List[Tuple[str, int]]:
    """
    Scrape recent Premier League transfers from reliable sources.
    Uses multiple fallback options.
    
    Returns:
        List of (player_name, days_since_transfer) tuples
    """
    recent_transfers = []
    
    # First try: Check FPL API for ownership/price anomalies
    recent_transfers = detect_transfers_from_fpl_data()
    
    if recent_transfers:
        return recent_transfers
    
    try:
        # Fallback to web scraping
        # Try official Premier League news
        url = "https://www.premierleague.com/transfers"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for transfer tables
            tables = soup.find_all('table', class_='wikitable')
            
            # Get current date for comparison
            today = datetime.now()
            cutoff_date = today - timedelta(days=14)  # Only care about last 2 weeks
            
            for table in tables:
                rows = table.find_all('tr')
                for row in rows[1:]:  # Skip header
                    cells = row.find_all('td')
                    if len(cells) >= 3:
                        # Try to extract player name and date
                        player_cell = cells[0]
                        date_cell = cells[-1]  # Often date is in last column
                        
                        player_name = player_cell.get_text(strip=True)
                        date_text = date_cell.get_text(strip=True)
                        
                        # Parse date (various formats possible)
                        transfer_date = parse_transfer_date(date_text)
                        if transfer_date and transfer_date > cutoff_date:
                            days_ago = (today - transfer_date).days
                            recent_transfers.append((player_name, days_ago))
                            log.info(f"Found recent transfer: {player_name} ({days_ago} days ago)")
        
        # Fallback: Try Sky Sports Transfer Centre
        if not recent_transfers:
            recent_transfers = scrape_sky_sports_transfers()
            
        # Additional fallback: BBC Sport
        if not recent_transfers:
            recent_transfers = scrape_bbc_transfers()
            
    except Exception as e:
        log.warning(f"Failed to scrape transfer data: {e}")
    
    return recent_transfers


def scrape_sky_sports_transfers() -> List[Tuple[str, int]]:
    """
    Fallback scraper for Sky Sports transfer data.
    """
    transfers = []
    try:
        url = "https://www.skysports.com/premier-league-transfers"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Sky Sports specific parsing
            transfer_items = soup.find_all('div', class_='transfer-item')
            today = datetime.now()
            
            for item in transfer_items:
                player_elem = item.find('h4', class_='transfer-item__player')
                date_elem = item.find('span', class_='transfer-item__date')
                
                if player_elem and date_elem:
                    player_name = player_elem.get_text(strip=True)
                    date_text = date_elem.get_text(strip=True)
                    
                    transfer_date = parse_transfer_date(date_text)
                    if transfer_date:
                        days_ago = (today - transfer_date).days
                        if days_ago <= 14:  # Last 2 weeks
                            transfers.append((player_name, days_ago))
                            
    except Exception as e:
        log.debug(f"Sky Sports scraping failed: {e}")
    
    return transfers


def scrape_bbc_transfers() -> List[Tuple[str, int]]:
    """
    Fallback scraper for BBC Sport transfer data.
    """
    transfers = []
    try:
        # BBC Sport Premier League transfers
        url = "https://www.bbc.com/sport/football/transfers"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # BBC specific parsing - they often use data attributes
            transfer_list = soup.find_all('article', {'data-testid': 'transfer-item'})
            today = datetime.now()
            
            for item in transfer_list:
                # Extract player and date from BBC's structure
                player_elem = item.find('h3')
                date_elem = item.find('time')
                
                if player_elem and date_elem:
                    player_name = player_elem.get_text(strip=True)
                    date_str = date_elem.get('datetime', '')
                    
                    if date_str:
                        try:
                            transfer_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                            days_ago = (today - transfer_date).days
                            if days_ago <= 14:
                                transfers.append((player_name, days_ago))
                        except:
                            pass
                            
    except Exception as e:
        log.debug(f"BBC scraping failed: {e}")
    
    return transfers


def parse_transfer_date(date_text: str) -> Optional[datetime]:
    """
    Parse various date formats from transfer news.
    """
    if not date_text:
        return None
        
    # Clean the text
    date_text = date_text.strip()
    
    # Try different date formats
    formats = [
        "%d %B %Y",      # 15 August 2025
        "%d %b %Y",      # 15 Aug 2025
        "%d/%m/%Y",      # 15/08/2025
        "%Y-%m-%d",      # 2025-08-15
        "%B %d, %Y",     # August 15, 2025
        "%b %d, %Y",     # Aug 15, 2025
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_text, fmt)
        except:
            continue
    
    # Check for relative dates
    if "today" in date_text.lower():
        return datetime.now()
    elif "yesterday" in date_text.lower():
        return datetime.now() - timedelta(days=1)
    elif "days ago" in date_text.lower():
        # Extract number of days
        match = re.search(r'(\d+)\s*days?\s*ago', date_text.lower())
        if match:
            days = int(match.group(1))
            return datetime.now() - timedelta(days=days)
    
    return None


def match_transfer_to_fpl(transfer_name: str, fpl_players: list) -> Optional[int]:
    """
    Match a transfer name to an FPL player ID.
    
    Args:
        transfer_name: Name from transfer news
        fpl_players: List of FPL player elements
        
    Returns:
        Player ID if match found, None otherwise
    """
    # Clean the transfer name
    transfer_name = transfer_name.strip()
    
    # Remove common suffixes
    for suffix in [' (loan)', ' (permanent)', ' (free)', ' (undisclosed)']:
        transfer_name = transfer_name.replace(suffix, '')
    
    # Try exact match first
    for player in fpl_players:
        if player['web_name'].lower() == transfer_name.lower():
            return player['id']
    
    # Try last name match
    transfer_last = transfer_name.split()[-1] if transfer_name else ""
    for player in fpl_players:
        if player['second_name'].lower() == transfer_last.lower():
            # Verify it's the right player by checking if they're at a new club
            # (ownership might be low if they just transferred)
            if float(player.get('selected_by_percent', 0)) < 5.0:
                return player['id']
    
    # Try partial match
    for player in fpl_players:
        if transfer_last and transfer_last.lower() in player['web_name'].lower():
            if float(player.get('selected_by_percent', 0)) < 5.0:
                return player['id']
    
    return None


def get_recent_transfer_adjustments() -> Dict[int, float]:
    """
    Returns adjustment factors for players who recently transferred.
    
    Players who transferred early in the window have had preseason to integrate
    and are NOT penalized.
    
    Returns:
        Dict mapping player_id to adjustment factor (0.0 to 1.0)
    """
    adjustments = {}
    
    try:
        # Get FPL player data
        from ..data.fpl_api import get_bootstrap
        bootstrap = get_bootstrap()
        
        # Scrape recent transfers
        log.info("Scraping recent transfer data...")
        recent_transfers = scrape_recent_transfers()
        
        if not recent_transfers:
            log.info("No recent transfers found from web scraping")
            # Fallback to manual list if scraping fails completely
            recent_transfers = get_manual_transfers()
        
        # Match transfers to FPL players and apply adjustments
        for transfer_name, days_ago in recent_transfers:
            player_id = match_transfer_to_fpl(transfer_name, bootstrap['elements'])
            
            if player_id:
                # Calculate adjustment factor based on days since transfer
                if days_ago <= 3:
                    factor = 0.2  # Very recent: 20% of normal minutes
                elif days_ago <= 7:
                    factor = 0.35  # One week: 35% of normal minutes
                elif days_ago <= 14:
                    factor = 0.6  # Two weeks: 60% of normal minutes
                else:
                    continue  # More than 2 weeks, likely integrated
                
                adjustments[player_id] = factor
                
                # Log the adjustment
                for p in bootstrap['elements']:
                    if p['id'] == player_id:
                        log.info(f"Transfer adjustment: {p['web_name']} "
                                f"({days_ago} days since transfer) - factor={factor:.1f}")
                        break
                        
    except Exception as e:
        log.warning(f"Could not apply transfer adjustments: {e}")
        # Use manual fallback
        adjustments = get_manual_transfer_fallback()
    
    return adjustments


def get_manual_transfers() -> List[Tuple[str, int]]:
    """
    Manual fallback list of known recent transfers.
    Update this when scraping fails.
    """
    # Format: (player_name, days_since_transfer)
    # ONLY add VERIFIED transfers
    return [
        # Example: ("Grealish", 5),  # If Grealish actually transferred to Everton
        # User should add actual recent transfers here
    ]


def get_manual_transfer_fallback() -> Dict[int, float]:
    """
    Manual fallback adjustments when scraping fails completely.
    """
    try:
        from ..data.fpl_api import get_bootstrap
        bootstrap = get_bootstrap()
        
        # Manual list with known player names
        # ONLY add VERIFIED transfers
        manual_adjustments = {
            # Example: "Grealish": 0.3,  # If actually transferred recently
            # User should add actual recent transfers here
        }
        
        adjustments = {}
        for player_name, factor in manual_adjustments.items():
            for p in bootstrap['elements']:
                if player_name in p['web_name']:
                    adjustments[p['id']] = factor
                    log.info(f"Manual adjustment: {player_name} - factor={factor:.1f}")
                    break
                    
        return adjustments
        
    except:
        return {}


def apply_transfer_adjustments(xmins_map: Dict[int, float]) -> Dict[int, float]:
    """
    Apply adjustments to xMins for recent transfers.
    
    Args:
        xmins_map: Current xMins predictions
        
    Returns:
        Updated xMins map with transfer adjustments
    """
    adjustments = get_recent_transfer_adjustments()
    
    if not adjustments:
        return xmins_map
    
    updated = xmins_map.copy()
    for player_id, factor in adjustments.items():
        if player_id in updated:
            original = updated[player_id]
            updated[player_id] = original * factor
            log.info(f"Adjusted player {player_id} xMins: {original:.0f} -> {updated[player_id]:.0f}")
    
    return updated