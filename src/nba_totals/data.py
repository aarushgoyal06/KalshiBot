"""Download and normalize NBA team game logs."""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

BOX_SCORE_COLUMNS = [
    "GAME_ID",
    "GAME_DATE",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "MATCHUP",
    "MIN",
    "FGM",
    "FGA",
    "FG3M",
    "FG3A",
    "FTM",
    "FTA",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PF",
    "PTS",
]


def season_id(start_year: int) -> str:
    """Return the NBA API season representation, e.g. 2023 -> ``2023-24``."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def download_team_game_logs(
    seasons: Iterable[str],
    output_path: str | Path,
    pause_seconds: float = 0.7,
) -> pd.DataFrame:
    """Download regular-season team box scores and cache them as CSV.

    The NBA stats service occasionally throttles requests. Completed seasons are
    fetched one at a time and the output is written only after all responses
    have been validated.
    """
    try:
        from nba_api.stats.endpoints import leaguegamelog
    except ImportError as exc:  # pragma: no cover - gives a useful CLI error
        raise RuntimeError("Install project dependencies before downloading data") from exc

    frames: list[pd.DataFrame] = []
    for season in seasons:
        response = leaguegamelog.LeagueGameLog(
            player_or_team_abbreviation="T",
            season=season,
            season_type_all_star="Regular Season",
            timeout=60,
        )
        frame = response.get_data_frames()[0]
        missing = set(BOX_SCORE_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"NBA API response for {season} omitted columns: {sorted(missing)}")
        frame = frame[BOX_SCORE_COLUMNS].copy()
        frame["SEASON"] = season
        frames.append(frame)
        time.sleep(pause_seconds)

    if not frames:
        raise ValueError("At least one season is required")

    result = pd.concat(frames, ignore_index=True)
    result["GAME_ID"] = result["GAME_ID"].astype(str).str.zfill(10)
    result["GAME_DATE"] = pd.to_datetime(result["GAME_DATE"]).dt.strftime("%Y-%m-%d")
    result = result.drop_duplicates(["GAME_ID", "TEAM_ID"]).sort_values(
        ["GAME_DATE", "GAME_ID", "TEAM_ID"]
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


def load_team_game_logs(path: str | Path) -> pd.DataFrame:
    """Read and validate a cached team-game CSV."""
    frame = pd.read_csv(path, dtype={"GAME_ID": str})
    missing = set(BOX_SCORE_COLUMNS + ["SEASON"]) - set(frame.columns)
    if missing:
        raise ValueError(f"Team game log is missing columns: {sorted(missing)}")
    frame["GAME_ID"] = frame["GAME_ID"].str.zfill(10)
    frame["GAME_DATE"] = pd.to_datetime(frame["GAME_DATE"])
    numeric = [column for column in BOX_SCORE_COLUMNS if column not in {
        "GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION", "MATCHUP"
    }]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    return frame.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ID"]).reset_index(drop=True)
