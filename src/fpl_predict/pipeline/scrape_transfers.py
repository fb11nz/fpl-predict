"""
Dedicated transfer scraping module with working implementations.
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import logging
import re

log = logging.getLogger(__name__)


def scrape_premier_league_transfers() -> List[Tuple[str, str, str, Optional[datetime]]]:
    """
    Scrape transfers from official Premier League website.
    
    Returns:
        List of (player_name, transfer_type, club, date) tuples
    """
    transfers = []
    
    try:
        url = 'https://www.premierleague.com/transfers'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            log.warning(f"Premier League returned status {response.status_code}")
            return transfers
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all tables (each club has its own table)
        tables = soup.find_all('table')
        
        for table in tables:
            # Try to find club name (usually in a heading above the table)
            club_name = None
            prev_sibling = table.find_previous_sibling(['h2', 'h3', 'div'])
            if prev_sibling:
                club_text = prev_sibling.get_text(strip=True)
                # Extract club name (remove "Transfers" suffix if present)
                club_name = club_text.replace(' Transfers', '').strip()
            
            # Process table rows
            rows = table.find_all('tr')[1:]  # Skip header
            
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 3:
                    player_name = cells[0].get_text(strip=True)
                    transfer_type = cells[1].get_text(strip=True)
                    other_club = cells[2].get_text(strip=True) if len(cells) > 2 else ""
                    
                    # Try to extract date if available
                    date = None
                    if len(cells) > 3:
                        date_text = cells[3].get_text(strip=True)
                        date = parse_date(date_text)
                    
                    # Determine which club (for transfers in)
                    if 'in' in transfer_type.lower():
                        transfers.append((player_name, transfer_type, club_name or other_club, date))
                    
        log.info(f"Scraped {len(transfers)} transfers from Premier League site")
        
    except Exception as e:
        log.error(f"Error scraping Premier League: {e}")
    
    return transfers


def scrape_bbc_transfers() -> List[Tuple[str, str, str, Optional[datetime]]]:
    """
    Scrape transfers from BBC Sport.
    
    Returns:
        List of (player_name, transfer_type, club, date) tuples
    """
    transfers = []
    
    try:
        # BBC Sport transfer deadline day or summer transfers page
        urls = [
            'https://www.bbc.com/sport/football/transfers',
            'https://www.bbc.com/sport/football/premier-league/transfers'
        ]
        
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        for url in urls:
            try:
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code != 200:
                    continue
                    
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # BBC often uses article tags for transfer news
                articles = soup.find_all('article')
                
                for article in articles:
                    # Look for player name in heading
                    heading = article.find(['h2', 'h3', 'h4'])
                    if not heading:
                        continue
                    
                    text = heading.get_text(strip=True)
                    
                    # Look for transfer patterns
                    patterns = [
                        r'([A-Z][a-z]+ [A-Z][a-z]+) (?:joins|signs for|moves to) ([A-Z][a-z]+)',
                        r'([A-Z][a-z]+) (?:joins|signs for|moves to) ([A-Z][a-z]+)',
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, text)
                        if match:
                            player_name = match.group(1)
                            club = match.group(2)
                            
                            # Try to find date
                            time_elem = article.find('time')
                            date = None
                            if time_elem:
                                date_str = time_elem.get('datetime')
                                if date_str:
                                    try:
                                        date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                                    except:
                                        pass
                            
                            transfers.append((player_name, "Transfer", club, date))
                            break
                            
            except Exception as e:
                log.debug(f"Error with BBC URL {url}: {e}")
                
        log.info(f"Scraped {len(transfers)} transfers from BBC")
        
    except Exception as e:
        log.error(f"Error scraping BBC: {e}")
    
    return transfers


def parse_date(date_text: str) -> Optional[datetime]:
    """Parse various date formats."""
    if not date_text:
        return None
        
    date_text = date_text.strip()
    
    # Common date formats
    formats = [
        "%d %B %Y",      # 15 August 2025
        "%d %b %Y",      # 15 Aug 2025  
        "%d/%m/%Y",      # 15/08/2025
        "%d/%m/%y",      # 15/08/25
        "%Y-%m-%d",      # 2025-08-15
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_text, fmt)
        except:
            continue
    
    # Relative dates
    today = datetime.now()
    if 'today' in date_text.lower():
        return today
    elif 'yesterday' in date_text.lower():
        return today - timedelta(days=1)
    elif 'ago' in date_text.lower():
        # Extract days ago
        match = re.search(r'(\d+)\s*days?\s*ago', date_text.lower())
        if match:
            days = int(match.group(1))
            return today - timedelta(days=days)
    
    return None


def get_recent_premier_league_transfers(days_back: int = 14) -> List[Tuple[str, int]]:
    """
    Get recent Premier League transfers within specified days.
    
    Args:
        days_back: How many days back to look for transfers
        
    Returns:
        List of (player_name, days_since_transfer) tuples
    """
    all_transfers = []
    
    # Try Premier League official site
    pl_transfers = scrape_premier_league_transfers()
    all_transfers.extend(pl_transfers)
    
    # Try BBC as backup
    if len(all_transfers) < 5:  # If we didn't get many from PL site
        bbc_transfers = scrape_bbc_transfers()
        all_transfers.extend(bbc_transfers)
    
    # Filter to recent transfers and calculate days
    recent = []
    today = datetime.now()
    cutoff = today - timedelta(days=days_back)
    
    for player_name, transfer_type, club, date in all_transfers:
        if date and date > cutoff:
            days_ago = (today - date).days
            recent.append((player_name, days_ago))
            log.info(f"Recent transfer: {player_name} to {club} ({days_ago} days ago)")
        elif not date and 'in' in transfer_type.lower():
            # If no date but it's a transfer in, assume it's recent
            # Estimate 7 days as a middle ground
            recent.append((player_name, 7))
            log.info(f"Recent transfer (no date): {player_name} to {club} (estimated 7 days ago)")
    
    return recent


if __name__ == "__main__":
    # Test the scraper
    logging.basicConfig(level=logging.INFO)
    transfers = get_recent_premier_league_transfers()
    print(f"\nFound {len(transfers)} recent transfers:")
    for player, days in transfers[:10]:
        print(f"  - {player}: {days} days ago")