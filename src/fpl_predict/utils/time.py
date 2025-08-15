from __future__ import annotations
from datetime import datetime, timezone
def now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
