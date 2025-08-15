from __future__ import annotations
import re
from pathlib import Path
from typing import Any
import requests
from bs4 import BeautifulSoup
import pandas as pd
import yaml

from ..utils.cache import RULES_CACHE, DATA
from ..utils.logging import get_logger
from ..utils.io import write_json

log = get_logger(__name__)

PL_DEFCON_URL = "https://www.premierleague.com/en/news/4361991"
FPL_RULES_URL = "https://fantasy.premierleague.com/help/rules"

def _get(url: str) -> str:
    headers = {"User-Agent": "fpl-predict/0.1 (+https://example.org)"}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.text

def parse_defensive_contribution(html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" \n")
    points = 2 if re.search(r"two FPL points", text, re.I) else 2
    def_thr = 10 if re.search(r"defender.*?combined total of\s*10.*?clearances.*?blocks.*?interceptions.*?tackles", text, re.I|re.S) else 10
    mf_thr = 12 if re.search(r"midfielders and forwards.*?12", text, re.I|re.S) else 12
    return { "defensive_contribution": {
        "points": points, "defender_threshold_cbit": def_thr,
        "mid_fwd_threshold_cbirt": mf_thr, "capped_per_match": True
    }}

def parse_rules_table(html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" \n")
    rules: dict[str, Any] = {
        "minutes": {}, "goals": {}, "assist": None, "clean_sheet": {},
        "saves_per_3": None, "penalty_save": None, "penalty_miss": None,
        "goals_conceded_per_2": {"GKP_DEF": None}, "yellow_card": None,
        "red_card": None, "own_goal": None, "bonus": [1,2,3],
    }
    def m(p): return re.search(p, text, re.I)
    k = m(r"1-59 minutes\s*(\+?)(\d+)");   rules["minutes"]["1-59"] = int(k.group(2)) if k else 1
    k = m(r"60 minutes or more\s*(\+?)(\d+)"); rules["minutes"]["60+"] = int(k.group(2)) if k else 2
    for pos, lab in [("GKP","Goalkeeper"),("DEF","Defender"),("MID","Midfielder"),("FWD","Forward")]:
        g = m(fr"{lab} goal\s*(\+?)(\d+)")
        if g: rules["goals"][pos] = int(g.group(2))
    for pos, lab in [("GKP","Goalkeeper"),("DEF","Defender"),("MID","Midfielder")]:
        c = m(fr"{lab} clean sheet\s*(\+?)(\d+)")
        if c: rules["clean_sheet"][pos] = int(c.group(2))
    a = m(r"Assist\s*(\+?)(\-?\d+)");             rules["assist"] = int(a.group(2)) if a else 3
    s = m(r"Save\s*\(every\s*3\)\s*(\+?)(\-?\d+)"); rules["saves_per_3"] = int(s.group(2)) if s else 1
    ps = m(r"Penalty save\s*(\+?)(\-?\d+)");      rules["penalty_save"] = int(ps.group(2)) if ps else 5
    pm = m(r"Penalty miss\s*(\+?)(\-?\d+)");      rules["penalty_miss"] = int(pm.group(2)) if pm else -2
    gc = m(r"Goalkeeper/Defender goals conceded\s*\(every 2\)\s*(\-?\d+)")
    rules["goals_conceded_per_2"]["GKP_DEF"] = int(gc.group(1)) if gc else -1
    y = m(r"Yellow card\s*(\+?)(\-?\d+)");        rules["yellow_card"] = int(y.group(2)) if y else -1
    r = m(r"Red card\s*(\+?)(\-?\d+)");           rules["red_card"] = int(r.group(2)) if r else -3
    og = m(r"Own goal\s*(\+?)(\-?\d+)");          rules["own_goal"] = int(og.group(2)) if og else -2
    return rules

def fetch_scoring_rules() -> dict[str, Any]:
    compiled: dict[str, Any] = {}
    try:
        compiled.update(parse_defensive_contribution(_get(PL_DEFCON_URL)))
    except Exception as e:
        log.warning("Defensive contribution parse failed: %s", e)
    try:
        compiled.update(parse_rules_table(_get(FPL_RULES_URL)))
    except Exception as e:
        log.warning("Rules table parse failed: %s", e)

    essentials = [compiled.get("minutes"), compiled.get("goals"), compiled.get("assist"),
                  compiled.get("clean_sheet"), compiled.get("defensive_contribution")]
    if not all(essentials):
        snap = RULES_CACHE / "2025-07-21.yaml"
        log.warning("Using cached rules snapshot: %s", snap)
        import yaml as _y
        compiled = _y.safe_load(Path(snap).read_text())

    write_json(compiled, DATA / "processed" / "rules.json")
    return compiled

def render_scoring_table_md(rules: dict[str, Any]) -> str:
    rows = []
    mins = rules.get("minutes", {})
    rows.append(["Minutes 1-59", mins.get("1-59","?")])
    rows.append(["Minutes 60+", mins.get("60+","?")])
    for pos in ["GKP","DEF","MID","FWD"]:
        if pos in rules.get("goals", {}):
            rows.append([f"Goal ({pos})", rules["goals"][pos]])
    if rules.get("assist") is not None: rows.append(["Assist", rules["assist"]])
    for pos in ["GKP","DEF","MID"]:
        if pos in rules.get("clean_sheet", {}):
            rows.append([f"Clean sheet ({pos})", rules["clean_sheet"][pos]])
    gc = rules.get("goals_conceded_per_2",{}).get("GKP_DEF")
    if gc is not None: rows.append(["Goals conceded (GKP/DEF, per 2)", gc])
    for k in ["saves_per_3","penalty_save","penalty_miss","yellow_card","red_card","own_goal"]:
        if rules.get(k) is not None:
            label = {"saves_per_3":"Saves (per 3)","penalty_save":"Penalty save","penalty_miss":"Penalty miss",
                     "yellow_card":"Yellow card","red_card":"Red card","own_goal":"Own goal"}[k]
            rows.append([label, rules[k]])
    dc = rules.get("defensive_contribution",{})
    if dc:
        rows.append(["Defensive contribution (DEF threshold)", f"{dc.get('points','?')} @ {dc.get('defender_threshold_cbit','?')} CBIT"])
        rows.append(["Defensive contribution (MID/FWD threshold)", f"{dc.get('points','?')} @ {dc.get('mid_fwd_threshold_cbirt','?')} CBIRT"])
    df = pd.DataFrame(rows, columns=["Action","Points"])
    return df.to_markdown(index=False)

def update_readme_scoring_table(readme_path: Path) -> None:
    start, end = "<!-- SCORING_TABLE_START -->", "<!-- SCORING_TABLE_END -->"
    rules = fetch_scoring_rules()
    table_md = render_scoring_table_md(rules)
    md = Path(readme_path).read_text(encoding="utf-8") if Path(readme_path).exists() else ""
    if start not in md:
        md += f"\n\n{start}\n{end}\n"
    import re
    pattern = re.compile(f"{re.escape(start)}.*?{re.escape(end)}", re.S)
    new_md = pattern.sub(f"{start}\n\n{table_md}\n\n{end}", md)
    Path(readme_path).write_text(new_md, encoding="utf-8")
    log.info("README scoring table updated.")
