# FPL Predict

A machine learning system for Fantasy Premier League predictions and optimization.

## Features

- **Advanced ML Models**: XGBoost-based prediction models for expected points, minutes, and clean sheets
- **Transfer Optimization**: Linear Programming solver for optimal squad selection  
- **Weekly Transfer Recommendations**: Analyzes your existing team and suggests optimal transfers
- **Competition Detection**: Identifies backup players and rotation risks
- **Chip Strategy**: Plans optimal usage of chips with 2025/26 double chips system
- **Automated Weekly Updates**: Fetches latest FPL data and retrains models

## Quick Start

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env

# Update data and train models
fpl update --run --advanced

# Get optimal squad from scratch
fpl transfers optimize --use-lp --horizon 5 --bench-budget 18

# Get transfer recommendations for existing team
fpl myteam sync --entry YOUR_TEAM_ID
fpl transfers recommend --consider-hits

# Plan chip strategy
fpl chips plan-2025 --use-myteam
```

## Installation

1. Clone the repository
2. Create virtual environment: `python -m venv .venv`
3. Activate: `source .venv/bin/activate` (Linux/Mac) or `.venv\Scripts\activate` (Windows)
4. Install: `pip install -e .`
5. Configure: Copy `.env.example` to `.env` and add your credentials

## Configuration

Set these in your `.env` file:
- `FPL_EMAIL`: Your FPL login email
- `FPL_PASSWORD`: Your FPL password  
- `FPL_ENTRY_ID`: Your team ID
- `FOOTBALL_DATA_TOKEN`: Optional API token for football-data.org

## Core Commands

### Data Updates
- `fpl update --run`: Fetch latest data and train models
- `fpl update --run --advanced`: Use XGBoost models with ensemble predictions

### Squad Optimization
- `fpl transfers optimize`: Get optimal 15-player squad
- `fpl transfers recommend`: Get transfer suggestions for your team

### Team Management
- `fpl myteam sync --entry ID`: Download your current team
- `fpl myteam prices`: Check price changes

### Chip Strategy
- `fpl chips plan-2025`: Plan chip usage for 2025/26 season

## Model Architecture

The system uses ensemble machine learning models for predictions:
- **Expected Points (EP)**: Combines FPL's official EP with ML predictions
- **Expected Minutes (xMins)**: Predicts playing time with competition detection
- **Clean Sheets**: XGBoost model for defensive returns
- **Expected Goals/Assists (xGI)**: Attacking threat predictions

## Transfer Optimizer

The Linear Programming optimizer considers:
- Expected points over planning horizon (default 5 GWs)
- Budget constraints (£100m)
- Squad rules (2 GKP, 5 DEF, 5 MID, 3 FWD, max 3 per club)
- Formation flexibility
- Bench value optimization
- Competition and rotation risks

## License

MIT License - see LICENSE file for details

<!-- SCORING_TABLE_START -->

| Action                                     | Points       |
|:-------------------------------------------|:-------------|
| Minutes 1-59                               | 1            |
| Minutes 60+                                | 2            |
| Goal (GKP)                                 | 6            |
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
