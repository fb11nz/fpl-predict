# FPL Predict

A comprehensive machine learning system for Fantasy Premier League predictions and squad optimization.

## Features

- **Advanced ML Models**: XGBoost/LightGBM ensemble for expected points, minutes, and clean sheets
- **Transfer Optimization**: Linear Programming solver for mathematically optimal squad selection with lineup/captain selection
- **Weekly Transfer Recommendations**: Analyzes your existing team and suggests optimal transfers with banking strategy comparison
- **Fixture-Based EP Predictions**: Per-gameweek expected points using FDR and venue adjustments
- **Free Hit Team Generator**: Builds optimal 15-man Free Hit squads with budget optimization and haul-factor captaincy
- **Chip Strategy**: Plans all 8 chips (2 sets of 4) using the availability windows the API reports
- **Competition Detection**: Data-driven identification of backup players and rotation risks
- **Recent Transfer Detection**: Web scraping to adjust for new signings
- **New Player Adjustments**: Smart handling of players with limited data to avoid harsh penalties
- **Automated Pipeline**: Fully integrated data update, training, and post-processing

## Quick Start

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env

# Authenticate (get token from FPL DevTools)
fpl auth set-token --token "Bearer YOUR_TOKEN_HERE"

# Advanced workflow with ensemble models
fpl update --run --advanced
fpl transfers optimize --use-lp --horizon 5 --bench-budget 180

# Get transfer recommendations for existing team
fpl myteam sync --entry YOUR_TEAM_ID
fpl transfers recommend --consider-hits

# Chip strategy and Free Hit planning
fpl chips plan --use-myteam                   # See when to use chips
fpl chips free-hit --gw 9                     # Generate optimal Free Hit team
fpl chips free-hit-analysis                   # Compare all gameweeks
```

## Installation

1. Clone the repository
2. Create virtual environment: `python -m venv .venv`
3. Activate: `source .venv/bin/activate` (Linux/Mac) or `.venv\Scripts\activate` (Windows)
4. Install: `pip install -e .`
5. Configure: Copy `.env.example` to `.env` and set your credentials

## Authentication

FPL requires authentication for accessing team data. Get your token from browser DevTools:

1. Login to fantasy.premierleague.com
2. Open DevTools (F12) → Network tab
3. Find any API request to fantasy.premierleague.com
4. Copy the `x-api-authorization` header value (including "Bearer")
5. Set token: `fpl auth set-token --token "Bearer YOUR_TOKEN"`

Alternative methods:
```bash
# Direct token (no prompt)
fpl auth set-token --token "Bearer eyJ..."

# Via echo/pipe (for scripts)
echo "Bearer eyJ..." | fpl auth set-token

# Manual .env edit
echo 'FPL_AUTH_TOKEN="eyJ..."' >> .env
```

## CLI Commands Reference

### Data & Model Commands

#### `fpl update`
Updates FPL data and optionally trains models.

Options:
- `--run`: Actually run the update (otherwise just shows what would happen)
- `--advanced`: Use XGBoost/LightGBM ensemble models instead of basic models
- `--rebuild-features`: Force rebuild feature engineering
- `--rebuild-models`: Force retrain all models

Examples:
```bash
fpl update --run                    # Basic update and training
fpl update --run --advanced         # Use advanced models
fpl update --run --rebuild-models   # Force retrain everything
```

### Transfer Commands

#### `fpl transfers optimize`
Generates optimal 15-player squad from scratch using Linear Programming.

Options:
- `--use-lp`: Use Linear Programming optimizer (recommended)
- `--horizon N`: Planning horizon in gameweeks (default: 1)
- `--bench-budget N`: Bench budget in £0.1m (default: 180 = £18m)
- `--exclude`: Exclude specific players

Examples:
```bash
fpl transfers optimize --use-lp                           # Basic optimization
fpl transfers optimize --use-lp --horizon 5               # 5-week horizon
fpl transfers optimize --use-lp --bench-budget 200        # £20m bench
fpl transfers optimize --use-lp --exclude "Haaland,Salah" # Exclude players
```

#### `fpl transfers recommend`
Suggests transfers for your existing team with optimal lineup and captain selection.

Options:
- `--consider-hits`: Allow point hits for transfers
- `--max-transfers N`: Maximum transfers to suggest
- `--horizon N`: Planning horizon
- `--no-banking`: Disable banking strategy comparison

Features:
- Recommends optimal starting XI and bench order
- Selects captain and vice-captain based on expected points
- Compares making transfers now vs banking for next week (up to 5 FT cap)
- Shows expected point gains for each strategy

Examples:
```bash
fpl transfers recommend                      # Basic recommendations with banking
fpl transfers recommend --consider-hits      # Allow -4 point hits
fpl transfers recommend --max-transfers 2    # Limit to 2 transfers
fpl transfers recommend --no-banking         # Disable banking comparison
```

### Team Management Commands

#### `fpl myteam sync`
Downloads your current FPL team data.

Options:
- `--entry ID`: Your FPL team ID (or set FPL_ENTRY_ID in .env)

Examples:
```bash
fpl myteam sync --entry 5436936    # Download specific team
fpl myteam sync                    # Use entry from .env
```

#### `fpl myteam prices`
Shows price changes for your players.

```bash
fpl myteam prices    # Check if any players changed price
```

### Chip Strategy Commands

#### `fpl chips plan`
Plans all 8 chips (2 sets of 4, one per half of the season).

Features:
- **H1 Planning**: Use-it-or-lose-it chips with urgency tracking, windows read from the API
- **H2 Planning**: DGW/BGW predictions for optimal timing
- **Fixture-Based Analysis**: Per-gameweek EP predictions using FDR and venue
- **Personalized Recommendations**: Based on your actual squad
- **Haul Factor Captaincy**: Prioritizes high-ceiling players for Triple Captain

Options:
- `--use-myteam`: Base recommendations on your synced team (default: True)
- `--explain`: Show detailed reasoning (default: True)
- `--show-teams`: Display full Free Hit XI for recommended gameweeks

Examples:
```bash
fpl chips plan                             # General chip strategy
fpl chips plan --use-myteam                # Personalized for your team
fpl chips plan --show-teams                # Include Free Hit team previews
```

#### `fpl chips free-hit`
Generates optimal Free Hit team for a specific gameweek.

Features:
- **15-Man Squad**: Full starting XI + 4-player bench
- **Budget Optimization**: Uses your actual squad selling value
- **Upgrade Logic**: Maximizes squad quality (no wasted funds)
- **Smart Captaincy**: Captain + vice-captain using haul factor (prioritizes attackers)
- **All FPL Constraints**: 2 GKP, 5 DEF, 5 MID, 3 FWD, max 3 per club
- **Fixture-Aware**: EP predictions based on opponent difficulty and venue

Options:
- `--gw N`: Gameweek to generate team for (required)

Examples:
```bash
fpl chips free-hit --gw 9              # Generate Free Hit team for GW9
fpl chips free-hit --gw 19             # H1 deadline week team
```

#### `fpl chips free-hit-analysis`
Analyzes Free Hit value across multiple gameweeks.

Features:
- **EP Comparison**: Your XI vs optimal XI for each gameweek
- **Delta Calculation**: Shows point gain for each week
- **Worth It Assessment**: Recommends if FH is worth using (≥6 point threshold)
- **Best Week Identification**: Highlights optimal Free Hit gameweek

Options:
- `--gw-start N`: Starting gameweek (default: current GW)
- `--gw-end N`: Ending gameweek (default: the H1 chip deadline from the API)

Examples:
```bash
fpl chips free-hit-analysis                    # Analyze current GW to GW19
fpl chips free-hit-analysis --gw-start 8       # Start from GW8
fpl chips free-hit-analysis --gw-end 12        # Short-term analysis
```

### Authentication Commands

#### `fpl auth set-token`
Sets the FPL authentication token.

Options:
- `--token TOKEN`: Provide token directly (no prompt)
- `--save-env/--no-save-env`: Save to .env file (default: yes)

Examples:
```bash
fpl auth set-token --token "Bearer eyJ..."    # Direct token
echo "Bearer eyJ..." | fpl auth set-token     # Via pipe
fpl auth set-token                            # Interactive prompt
```

#### `fpl auth test`
Tests if authentication is working.

```bash
fpl auth test --entry 5436936    # Test with specific team ID
```

## Pipeline Architecture

### 1. Data Sources & Ingestion

| Source | What it gives | Where it lands |
|:-------|:--------------|:---------------|
| FPL API `bootstrap-static` | prices, ownership, status, scoring rules, chip windows | in memory, cached per run |
| FPL API `element-summary` | per-gameweek history for the current season, previous-season totals | `data/processed/features.parquet` |
| FPL API `fixtures` | fixture list, results once played | `data/raw/football-data/EPL_<season>_matches.parquet` |
| `vaastav/Fantasy-Premier-League` | per-player-per-gameweek history, 2022-23 onward | `data/raw/fpl_history/<season>/player_gw.parquet` |
| football-data.org | match results, fallback when the archive is unavailable | `data/raw/football-data/` |
| Web scraping | mid-season transfer news | applied as xMins adjustments |

The gameweek archive is the training set: 113,260 player-gameweek rows across four seasons,
with xG, xA and xGC from 2022-23 and the defensive-contribution components from 2025-26 (the
season the rule was introduced). Rows are keyed on FPL's stable player `code`, because
element ids are renumbered every summer. It also carries `xP`, FPL's own pre-deadline
expected points, which is the baseline any model here has to beat.

**Season handling.** `config.current_season()` flips in July, so nothing is pinned to a
season number. Completed seasons are cached and reused; only the season in progress is
re-pulled. A fetch that returns nothing leaves the stored file alone, which matters in
August: the new season has no finished fixtures, and an earlier version rebuilt the combined
match file from that empty result and dropped a whole season of history.

**Team names.** Every source spells clubs differently ("Manchester City FC", "Man City",
"Man Utd", "Man United"), so `data/teams.py` maps all of them onto one slug. Eight of twenty
clubs used to resolve to two different keys depending on the source, which broke every join
keyed on team name: Elo ratings, clean-sheet rates, team form and team quality were all
computed under one spelling and looked up under another.

- **Scoring rules** come from `bootstrap-static.game_config.scoring`, not from scraping the
  help page, so a rule change is picked up on the next run. 2026/27 raised goalkeeper goals
  from 6 to 10 points. The only values not in the API are the defensive-contribution
  thresholds (10 CBIT for defenders, 12 CBIRT for midfielders and forwards), which are
  scraped with those constants as the fallback.

### 2. Preprocessing Pipeline

Data flows through several preprocessing steps:

1. **Feature Engineering** (`build_features.py`):
   - Aggregates last N games statistics
   - Calculates rolling averages and form metrics
   - Merges FPL and external data sources

2. **Team Strength Calculation**:
   - Uses historical results to compute team offensive/defensive ratings
   - Adjusts for home/away performance

3. **Fixture Difficulty Rating (FDR)**:
   - Combines official FPL FDR with computed team strengths
   - Creates custom difficulty metrics

### 3. Model Training

The system trains multiple specialized models:

#### Basic Models (default):
- **Linear Regression**: Simple baseline predictions
- **Random Forest**: Non-linear patterns

#### Advanced Models (`--advanced` flag):
- **XGBoost**: Gradient boosting for complex patterns
- **LightGBM**: Fast gradient boosting
- **Ensemble**: Weighted average of multiple models

#### Prediction Targets:
- **Expected Points (EP)**: Total points prediction
  - `ep_base`: Raw model prediction
  - `ep_blend`: 50/50 blend with FPL's official EP
  - `ep_adjusted`: Post-processed with competition/transfer adjustments

- **Expected Minutes (xMins)**: Playing time prediction (0-90)
  - Critical for identifying rotation risks
  - Post-processed for backup players

- **Clean Sheets (CS)**: Probability of defensive returns
  - Separate models for GKP/DEF

- **Expected Goal Involvement (xGI)**: Goals + Assists prediction

### 4. Post-Processing Pipeline

After training, several adjustments are applied:

1. **Backup Player Detection** (`fix_squad_roles_post_training`):
   - Identifies backup GKPs (ownership < 2%, price < £5.0m)
   - Sets their xMins to 0

2. **Competition Detection** (`competition_detector.py`):
   - Data-driven approach using:
     - Ownership percentages
     - Player prices
     - Position depth at each club
   - Reduces xMins for players with competition

3. **Recent Transfer Adjustments** (`recent_transfers.py`):
   - Web scrapes for transfers in last 14 days
   - Applies penalties:
     - 0-3 days: 80% of original xMins
     - 4-7 days: 65% of original xMins
     - 8-14 days: 40% of original xMins

4. **New Player Adjustments** (`adjustments.py`):
   - Prevents harsh penalties for players with limited minutes (<180 mins)
   - Sets minimum floor at 60% of position average for same price bracket
   - Ensures new signings aren't unfairly penalized

5. **EP Recalculation**:
   - After xMins adjustments, recalculates `ep_adjusted`
   - Ensures consistency between playing time and points

### 5. Squad Optimization

The Linear Programming optimizer (`optimizer.py`) solves:

#### Objective Function:
Maximize: Σ(EP × captain_multiplier + value_bonus + difficulty_bonus - risk_penalty)

#### Constraints:
- **Budget**: Total cost ≤ £100m
- **Squad Composition**: Exactly 2 GKP, 5 DEF, 5 MID, 3 FWD
- **Club Limit**: Max 3 players per club
- **Formation**: Valid formation for starting XI
- **Bench Budget**: Bench players cost ≤ specified budget

#### Key Features:
- **Two-Stage Optimization**: Evaluates transfers then optimizes lineup
- **Formation Flexibility**: Automatically selects optimal formation
- **Captain/Vice-Captain Selection**: Identifies best captain and vice-captain choices
- **Bench Optimization**: Orders bench by expected points
- **Banking Strategy**: Compares immediate transfers vs saving for multiple transfers (up to 5 FT)
- **Multi-horizon Planning**: Considers future fixtures

### 6. Integration & Automation

The entire pipeline is integrated via `update_weekly.py`:

1. **Data Update**: Fetches latest FPL/external data
2. **Feature Engineering**: Rebuilds features with new data
3. **Model Training**: Retrains all models
4. **Post-Processing**: Applies all adjustments automatically
5. **Saves Results**: Outputs ready for optimizer

Running `fpl update --run` triggers the complete pipeline, ensuring all fixes and adjustments are applied without manual intervention.

## Model Performance

Not currently measured. The figures previously quoted here (MAE ~2.5, 85% minutes accuracy,
CS AUC 0.72) were never produced by any code in this repo — `models/backtest.py` returns
hardcoded constants and nothing calls it.

A rolling-origin backtest over `data/raw/fpl_history/` is the next piece of work. Until it
exists, treat the expected points as unvalidated, and note that FPL's own `ep_next` is
blended into every prediction (see `ep_blend` below), so a large part of the output is not
this project's own signal.

Two known calibration problems, measured against 45,787 player-gameweek appearances from
2022-23 to 2025-26 under the current scoring table:

| Component            | Modelled | Actual (mean pts/appearance)          |
|:---------------------|:---------|:--------------------------------------|
| Saves                | no       | 0.66 for GKP (17% of their points)    |
| Goals conceded       | no       | -0.50 GKP, -0.40 DEF                  |
| Bonus                | no       | 0.17-0.36, and the strongest single predictor of a score after goals |
| Cards                | no       | -0.07 to -0.17                        |
| Defensive contribution | yes    | over-credited 3x for DEF, 4.5x for MID, 69x for FWD |

Net effect: expected points over-rate defenders by roughly 68% and under-rate goalkeepers by
roughly 20%.

## Recent Updates

### Free Hit Team Generator (New!)
- **Complete 15-Man Squads**: Generates full starting XI + bench with legal formations
- **Budget Optimization**: Uses your actual squad selling value (not hardcoded £100m)
- **Upgrade Algorithm**: Iteratively improves squad to maximize quality with remaining budget
- **Smart Captaincy**: Recommends captain + vice-captain using haul factor
  - Prioritizes high-ceiling players (FWD 1.8x, MID 1.5x, DEF 0.9x, GKP 0.6x)
- **Fixture Analysis**: Compare Free Hit value across all gameweeks with EP deltas

### Chip Strategy Improvements
- **Double Chips**: Plans all 8 chips (2 sets of TC/BB/FH/WC)
- **Fixture-Based EP**: Per-gameweek predictions using FDR and home/away adjustments
- **Urgency Tracking**: H1 chips (GW1-19) have deadline pressure, H2 saved for DGWs
- **FDR Bug Fix**: Corrected inverted FDR multiplier (low FDR now correctly = easier fixture)
- **Personalized Analysis**: Based on your actual owned players, not generic top scorers

### Transfer Recommender Enhancements
- **Lineup Optimization**: Recommends optimal starting XI with captain/vice-captain
- **Banking Strategy**: Compares making transfers now vs banking for next week
- **FPL 5 FT Cap**: Updated to reflect FPL's current 5 free transfer maximum (was 2)
- **New Player Adjustments**: Smart handling of players with limited data (<180 mins)

### System Improvements
The system automatically handles:
- Players who recently transferred clubs (via web scraping)
- Backup goalkeepers and rotation risks
- New signings who need time to settle
- Banking decisions for up to 5 free transfers
- Max 3 players per club constraint in all optimizations

## Configuration

### Environment Variables (.env)

```bash
# Authentication (required for myteam features)
FPL_AUTH_TOKEN=your_token_here    # From browser DevTools

# Optional
FPL_ENTRY_ID=5436936              # Your team ID
FPL_EMAIL=your_email              # For password login
FPL_PASSWORD=your_password        # For password login
FOOTBALL_DATA_TOKEN=api_token     # For historical data

# Behavior toggles
ALLOW_RULES_FALLBACK=true        # Use rule-based models as fallback
ALLOW_ODDS_FALLBACK=true         # Use odds data if available
```

## Troubleshooting

### Authentication Issues
- Ensure token includes "Bearer " prefix or is just the token value
- Token expires after ~1 hour, refresh from browser
- Check .env file is in project root

### Model Training Issues
- Run with `--rebuild-features` if data seems stale
- Use `--rebuild-models` to force retrain
- Check `data/processed/` for intermediate files

### Optimizer Issues
- Ensure models trained successfully first
- Check bench budget is reasonable (150-200 = £15-20m)
- Verify no impossible constraints (e.g., excluding too many players)

### Free Hit Team Generator Issues
- **"No valid formation found"**: Usually due to budget constraints or missing fixture data
  - Run `fpl update --run` to ensure latest data
  - Check your squad value with `fpl myteam sync --entry YOUR_ID`
- **Budget seems wrong**: Free Hit uses your squad's selling value (not buying price)
  - Selling value = what you'd get if you sold all players now
  - Check `data/processed/myteam_latest.json` for selling prices
- **Team seems suboptimal**: Analysis uses fixture-based EP predictions
  - EP varies by opponent and home/away status
  - Use `fpl chips free-hit-analysis` to see EP for each gameweek

## License

MIT License - see LICENSE file for details

<!-- SCORING_TABLE_START -->

| Action                                     | Points       |
|:-------------------------------------------|:-------------|
| Minutes 1-59                               | 1            |
| Minutes 60+                                | 2            |
| Goal (GKP)                                 | 10           |
| Goal (DEF)                                 | 6            |
| Goal (MID)                                 | 5            |
| Goal (FWD)                                 | 4            |
| Assist                                     | 3            |
| Clean sheet (GKP)                          | 4            |
| Clean sheet (DEF)                          | 4            |
| Clean sheet (MID)                          | 1            |
| Goals conceded (GKP/DEF, per 2)            | -1           |
| Saves (per 3)                              | 1            |
| Penalty save                               | 5            |
| Penalty miss                               | -2           |
| Yellow card                                | -1           |
| Red card                                   | -3           |
| Own goal                                   | -2           |
| Defensive contribution (DEF threshold)     | 2 @ 10 CBIT  |
| Defensive contribution (MID/FWD threshold) | 2 @ 12 CBIRT |

<!-- SCORING_TABLE_END -->