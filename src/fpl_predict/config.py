# src/fpl_predict/config.py
from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
import datetime as _dt

# Number of completed seasons of history to ingest by default. Five seasons is roughly the
# useful horizon: further back and squads, tactics and the FPL scoring rules have all moved
# far enough that the data hurts more than it helps.
DEFAULT_HISTORY_SEASONS = 5


def current_season(today: _dt.date | None = None) -> int:
    """Start year of the season in progress, e.g. 2026 for 2026/27.

    A Premier League season starts in August, so anything from July onward belongs to the
    season starting that calendar year.
    """
    today = today or _dt.date.today()
    return today.year if today.month >= 7 else today.year - 1


def season_label(start_year: int) -> str:
    """Render a season start year the way FPL does: 2026 -> '2026-27'."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


class Settings(BaseSettings):
    # --- Auth / convenience ---
    FPL_SESSION: str | None = None
    FPL_AUTH_TOKEN: str | None = None  # x-api-authorization Bearer (optional)
    FPL_EMAIL: str | None = None
    FPL_PASSWORD: str | None = None
    FPL_ENTRY_ID: int | None = None

    # --- External API tokens (optional) ---
    FOOTBALL_DATA_TOKEN: str | None = None  # Football-Data.org

    # --- Ingestion window (optional; defaults if unset) ---
    # Example: 2023 means the 2023/24 season
    FD_START_SEASON: int | None = Field(default=None, description="Start season, e.g. 2023")
    FD_END_SEASON: int | None = Field(default=None, description="End season, e.g. 2024")
    HISTORY_SEASONS: int = Field(
        default=DEFAULT_HISTORY_SEASONS,
        description="How many completed seasons of history to ingest when the window is not pinned.",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("FPL_ENTRY_ID", "FD_START_SEASON", "FD_END_SEASON", mode="before")
    @classmethod
    def _blank_to_none(cls, v):
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    def current_season(self) -> int:
        """Start year of the season in progress."""
        return current_season()

    def seasons_window(self) -> tuple[int, int]:
        """(start, end) inclusive window of *completed* seasons to ingest.

        Ends at last season, because the season in progress has no complete result set and
        is fetched separately. FD_START_SEASON still anchors the start, but the end is
        always carried forward to last season: a pinned end older than that is a leftover
        from a previous season rather than a real instruction, and honouring it drops a
        whole season of results on the floor.
        """
        cur = current_season()
        span = max(1, int(self.HISTORY_SEASONS))
        latest_complete = cur - 1

        start = int(self.FD_START_SEASON) if self.FD_START_SEASON else None
        end = int(self.FD_END_SEASON) if self.FD_END_SEASON else None

        if end is None or end < latest_complete:
            end = latest_complete
        if start is None:
            start = end - span + 1
        return start, max(start, end)

    def history_seasons(self) -> list[int]:
        """Completed seasons to ingest, oldest first."""
        start, end = self.seasons_window()
        return list(range(start, end + 1))


settings = Settings()
