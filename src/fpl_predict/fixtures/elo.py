from __future__ import annotations
import pandas as pd

HOME_ADV = 65.0
K = 20.0

def expected_score(rating_a: float, rating_b: float, home: bool) -> float:
    ra = rating_a + (HOME_ADV if home else 0.0)
    return 1.0 / (1.0 + 10 ** (-(ra - rating_b) / 400))

def update_elo(df_matches: pd.DataFrame) -> dict[str, float]:
    ratings: dict[str, float] = {}
    def get(team: str) -> float: return ratings.get(team, 1500.0)
    for _, m in df_matches.sort_values("date").iterrows():
        h, a = m.home_team, m.away_team
        rh, ra = get(h), get(a)
        exp_h = expected_score(rh, ra, home=True)
        if (m.home_goals or 0) > (m.away_goals or 0): out = 1.0
        elif (m.home_goals or 0) < (m.away_goals or 0): out = 0.0
        else: out = 0.5
        margin = abs((m.home_goals or 0) - (m.away_goals or 0))
        k = K * (1 + 0.1 * margin)
        ratings[h] = rh + k * (out - exp_h)
        ratings[a] = ra + k * ((1 - out) - (1 - exp_h))
    return ratings
