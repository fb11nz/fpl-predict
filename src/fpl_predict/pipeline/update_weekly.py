from __future__ import annotations

from ..config import current_season, season_label
from ..data.fpl_api import get_bootstrap
from ..data.fpl_history import build_history
from ..data.ingest import bootstrap_raw_from_sample, ingest_full
from ..data.process import build_features
from ..data.rules_fetcher import update_readme_scoring_table
from ..fixtures.fdr import build_player_next5_fdr, compute_fdr
from ..models.train import train_all
from ..utils.cache import DATA, MODELS, PROC, ROOT
from ..utils.io import read_parquet, write_json, write_parquet
from ..utils.logging import get_logger
from ..utils.time import now_utc_str
from .competition_detector import detect_position_competition
from .recent_transfers import apply_transfer_adjustments

log = get_logger(__name__)


def fix_squad_roles_post_training():
    """Apply pecking-order, transfer and availability adjustments to xMins, then rescale EP.

    Everything here is derived from the current squad rather than from hand-maintained name
    lists. The previous version carried a dict of backup keeper surnames and a dict of
    backup defenders with fixed minute caps, both captured from the 2024/25 squads; they
    survived two transfer windows and were pinning the wrong players to 0 and 10 minutes.
    """
    log.info("Applying squad role fixes post-training...")

    try:
        xmins_df = read_parquet(PROC / "xmins.parquet")
        bootstrap = get_bootstrap()
        xmins_df = xmins_df.set_index("player_id")

        n_changes = 0

        log.info("Detecting position competition...")
        competition_adjustments = detect_position_competition()
        for pid, factor in competition_adjustments.items():
            if pid not in xmins_df.index:
                continue
            before = float(xmins_df.at[pid, "xmins"])
            after = before * factor
            if abs(after - before) > 1e-9:
                xmins_df.at[pid, "xmins"] = after
                n_changes += 1
                log.debug(
                    "Player %s: %.0f -> %.0f xMins (competition %.2f)", pid, before, after, factor
                )

        log.info("Applying recent transfer adjustments...")
        xmins_map = apply_transfer_adjustments(dict(xmins_df["xmins"]))
        for pid, new_xmins in xmins_map.items():
            if pid in xmins_df.index and float(xmins_df.at[pid, "xmins"]) != float(new_xmins):
                xmins_df.at[pid, "xmins"] = float(new_xmins)
                n_changes += 1

        xmins_df = xmins_df.reset_index()

        log.info("Applying availability adjustments...")
        from ..models.availability import apply_availability_adjustments

        xmins_df = apply_availability_adjustments(xmins_df, bootstrap)
        write_parquet(xmins_df, PROC / "xmins.parquet")

        ep_df = read_parquet(PROC / "exp_points.parquet")
        ep_df = apply_availability_adjustments(ep_df, bootstrap)

        # ep_blend is a per-90 style figure, so scale it by expected minutes to get the
        # points actually expected from the player this gameweek.
        xmins_map = dict(zip(xmins_df["player_id"], xmins_df["xmins"]))
        ep_df["ep_adjusted"] = (
            ep_df["ep_blend"] * ep_df["player_id"].map(xmins_map).fillna(0) / 90.0
        )
        write_parquet(ep_df, PROC / "exp_points.parquet")

        log.info("Applied %d squad role and competition fixes", n_changes)

    except Exception as e:
        log.warning("Squad role fixes failed: %s", e, exc_info=True)


def update_weekly_data(demo_mode: bool = False) -> None:
    log.info("Starting weekly update (demo=%s)", demo_mode)

    # Step 1: Ingest match results
    if demo_mode:
        bootstrap_raw_from_sample()
    else:
        ingest_full()

        # Player-gameweek history for previous seasons. Completed seasons are cached on disk
        # and reused; only the season in progress is re-pulled.
        try:
            cur = current_season()
            build_history()
            build_history(seasons=[cur - 1], refresh=True)
        except Exception as e:
            log.warning("Could not refresh player-gameweek history: %s", e)

    # Step 2: Build features from the fresh data
    build_features()
    update_readme_scoring_table(ROOT / "README.md")
    compute_fdr()
    build_player_next5_fdr()

    # Step 3: Drop stale training data so it is rebuilt against current player stats
    training_data_file = PROC / "training_data.parquet"
    if training_data_file.exists():
        log.info("Removing old training_data.parquet to force rebuild with current player stats")
        training_data_file.unlink()

    # Step 4: Train models (will now rebuild training data with current stats)
    models = train_all()
    write_json({"saved_at": now_utc_str(), "models": list(models.keys())}, MODELS / "latest.json")

    # Step 5: Apply post-training fixes for squad roles
    fix_squad_roles_post_training()

    write_json(
        {"updated_at": now_utc_str(), "season": season_label(current_season())},
        DATA / "processed" / "weekly_changelog.json",
    )
    log.info("Weekly update complete for %s.", season_label(current_season()))


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--run", action="store_true")
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()
    if args.run:
        update_weekly_data(demo_mode=args.demo)
    else:
        log.info("Use --run to execute the weekly update.")


if __name__ == "__main__":
    main()
