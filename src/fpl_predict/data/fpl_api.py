from __future__ import annotations
from functools import lru_cache
import time
import requests

UA = {"User-Agent": "fpl-predict/0.1 (training pipeline)"}
BASE = "https://fantasy.premierleague.com/api"

@lru_cache(maxsize=1)
def get_bootstrap() -> dict:
    r = requests.get(f"{BASE}/bootstrap-static/", headers=UA, timeout=30)
    r.raise_for_status()
    return r.json()

@lru_cache(maxsize=1)
def get_fixtures() -> list[dict]:
    r = requests.get(f"{BASE}/fixtures/", headers=UA, timeout=30)
    r.raise_for_status()
    return r.json()

def get_element_summary(pid: int) -> dict:
    # simple throttle to be polite
    time.sleep(0.05)
    r = requests.get(f"{BASE}/element-summary/{pid}/", headers=UA, timeout=30)
    r.raise_for_status()
    return r.json()
