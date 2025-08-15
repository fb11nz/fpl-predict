# src/fpl_predict/config.py
from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import datetime as _dt

class Settings(BaseSettings):
    # --- Auth / convenience ---
    FPL_SESSION: str | None = None
    FPL_AUTH_TOKEN: str | None = None  # x-api-authorization Bearer (optional)
    FPL_EMAIL: str | None = None
    FPL_PASSWORD: str | None = None
    FPL_ENTRY_ID: int | None = None

    # --- External API tokens (optional) ---
    FOOTBALL_DATA_TOKEN: str | None = None   # Football-Data.org
    ODDS_API_TOKEN: str | None = None        # Any odds API you use

    # --- Ingestion window (optional; defaults if unset) ---
    # Example: 2023 means the 2023/24 season
    FD_START_SEASON: int | None = Field(default=None, description="Start season, e.g. 2023")
    FD_END_SEASON: int | None = Field(default=None, description="End season, e.g. 2024")

    # --- Toggles ---
    ALLOW_RULES_FALLBACK: bool = True
    ALLOW_ODDS_FALLBACK: bool = True
    AFCON_GW16_5FT: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def seasons_window(self) -> tuple[int, int]:
        """
        Returns (start, end) seasons. If not provided in .env,
        default to a safe two-season window ending in the current season.
        """
        if self.FD_START_SEASON and self.FD_END_SEASON:
            return int(self.FD_START_SEASON), int(self.FD_END_SEASON)

        today = _dt.date.today()
        # If after July, current season starts this year; else last year.
        start_guess = today.year if today.month >= 7 else today.year - 1
        # Use two-season window by default (start-1 .. start)
        start = (self.FD_START_SEASON or (start_guess - 1))
        end = (self.FD_END_SEASON or start_guess)
        return int(start), int(end)

settings = Settings()