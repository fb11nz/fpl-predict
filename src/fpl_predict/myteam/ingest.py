from __future__ import annotations
import os, json, requests
from ..utils.logging import get_logger
from ..config import settings
from ..auth.login import get_auth_headers

log = get_logger(__name__)
API = "https://fantasy.premierleague.com/api"


def sync_myteam(entry_id: int | None = None) -> None:
    entry = entry_id or settings.FPL_ENTRY_ID
    if not entry:
        raise RuntimeError("No entry id. Pass --entry or set FPL_ENTRY_ID in .env.")

    headers = get_auth_headers()
    url = f"{API}/my-team/{int(entry)}/"
    r = requests.get(url, headers=headers, timeout=20)

    if r.status_code == 403:
        raise RuntimeError("Forbidden (403). Your token/cookie is invalid or expired.")
    r.raise_for_status()

    data = r.json()
    out = "data/processed/myteam_latest.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    log.info("My-team snapshot written → %s", out)