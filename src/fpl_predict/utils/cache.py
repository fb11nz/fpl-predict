from __future__ import annotations
from pathlib import Path

def _project_root() -> Path:
    p = Path(__file__).resolve()
    for _ in range(8):
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
        p = p.parent
    return Path(__file__).resolve().parents[3]

ROOT = _project_root()
DATA = ROOT / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"
RULES_CACHE = DATA / "rules_cache"
SAMPLE = DATA / "sample"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

for d in [RAW, PROC, RULES_CACHE, SAMPLE, MODELS, REPORTS]:
    d.mkdir(parents=True, exist_ok=True)
