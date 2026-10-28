"""Leakage-safe, pregame feature engineering."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

ROLLING_WINDOWS = (5, 10)
TEAM_METRICS = (
    "PTS",
    "POSS",
    "OFF_RATING",
    "DEF_RATING",
    "PACE",
    "EFG",
    "TOV_RATE",
    "OREB_RATE",
    "FT_RATE",
)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.replace(0, np.nan))


def _game_rows(team_logs: pd.DataFrame) -> pd.DataFrame:
    logs = team_logs.copy()
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
    keys = ["GAME_ID", "GAME_DATE", "SEASON"]
    team_counts = logs.groupby("GAME_ID")["TEAM_ID"].transform("size")
    if not team_counts.eq(2).all():
        invalid_ids = logs.loc[team_counts.ne(2), "GAME_ID"].unique()[:5]
        raise ValueError(f"Expected two team rows per game; invalid GAME_ID values: {invalid_ids}")

    marked_home = logs["MATCHUP"].str.contains(r"\bvs\.?", regex=True, na=False)
    marked_home_count = marked_home.groupby(logs["GAME_ID"]).transform("sum")
    # Neutral-site NBA logs occasionally mark both teams with "@". Since a total
    # is symmetric, assign a stable side for those games rather than dropping them.
    fallback_home = logs["TEAM_ID"].eq(logs.groupby("GAME_ID")["TEAM_ID"].transform("max"))
    logs["_IS_HOME"] = np.where(marked_home_count.eq(1), marked_home, fallback_home)
    home = logs[logs["_IS_HOME"]].drop(columns="_IS_HOME")
    away = logs[~logs["_IS_HOME"]].drop(columns="_IS_HOME")

    duplicated = logs.duplicated(["GAME_ID", "TEAM_ID"]).any()
    if duplicated:
        raise ValueError("Expected one row per team per game")
    games = home.merge(away, on=keys, how="inner", suffixes=("_HOME", "_AWAY"), validate="one_to_one")
    expected_games = logs["GAME_ID"].nunique()
    if len(games) != expected_games:
        raise ValueError(
            f"Paired {len(games)} games but found {expected_games} unique GAME_ID values"
        )
    return games.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def _team_history(team_logs: pd.DataFrame) -> pd.DataFrame:
    """Create one row per team-game with realized efficiency metrics."""
    games = _game_rows(team_logs)
    rows: list[pd.DataFrame] = []
    for side, opponent in (("HOME", "AWAY"), ("AWAY", "HOME")):
        frame = pd.DataFrame(
            {
                "GAME_ID": games["GAME_ID"],
                "GAME_DATE": games["GAME_DATE"],
                "SEASON": games["SEASON"],
                "TEAM_ID": games[f"TEAM_ID_{side}"],
                "IS_HOME": float(side == "HOME"),
            }
        )
        own = {name: games[f"{name}_{side}"] for name in (
            "MIN", "FGM", "FGA", "FTM", "FTA", "OREB", "DREB", "TOV", "PTS"
        )}
        opp = {name: games[f"{name}_{opponent}"] for name in (
            "MIN", "FGM", "FGA", "FTM", "FTA", "OREB", "DREB", "TOV", "PTS"
        )}

        own_poss = (
            own["FGA"]
            + 0.4 * own["FTA"]
            - 1.07
            * _safe_divide(own["OREB"], own["OREB"] + opp["DREB"])
            * (own["FGA"] - own["FGM"])
            + own["TOV"]
        )
        opp_poss = (
            opp["FGA"]
            + 0.4 * opp["FTA"]
            - 1.07
            * _safe_divide(opp["OREB"], opp["OREB"] + own["DREB"])
            * (opp["FGA"] - opp["FGM"])
            + opp["TOV"]
        )
        frame["PTS"] = own["PTS"].to_numpy()
        frame["POSS"] = (0.5 * (own_poss + opp_poss)).to_numpy()
        frame["OFF_RATING"] = 100 * _safe_divide(own["PTS"], frame["POSS"]).to_numpy()
        frame["DEF_RATING"] = 100 * _safe_divide(opp["PTS"], frame["POSS"]).to_numpy()
        frame["PACE"] = (48 * _safe_divide(frame["POSS"], own["MIN"] / 5)).to_numpy()
        frame["EFG"] = _safe_divide(own["FGM"] + 0.5 * games[f"FG3M_{side}"], own["FGA"]).to_numpy()
        frame["TOV_RATE"] = _safe_divide(own["TOV"], own_poss).to_numpy()
        frame["OREB_RATE"] = _safe_divide(own["OREB"], own["OREB"] + opp["DREB"]).to_numpy()
        frame["FT_RATE"] = _safe_divide(own["FTA"], own["FGA"]).to_numpy()
        rows.append(frame)
    return pd.concat(rows, ignore_index=True).sort_values(
        ["TEAM_ID", "GAME_DATE", "GAME_ID"]
    ).reset_index(drop=True)


def _pregame_team_features(history: pd.DataFrame) -> pd.DataFrame:
    """Shift every statistic before rolling so the current game cannot leak."""
    output_parts: list[pd.DataFrame] = []
    for _, group in history.groupby("TEAM_ID", sort=False):
        group = group.sort_values(["GAME_DATE", "GAME_ID"]).copy()
        group["REST_DAYS"] = group["GAME_DATE"].diff().dt.days.clip(upper=7)
        group["GAMES_PLAYED"] = np.arange(len(group), dtype=float)
        for metric in TEAM_METRICS:
            prior = group[metric].shift(1)
            for window in ROLLING_WINDOWS:
                group[f"{metric}_L{window}"] = prior.rolling(window, min_periods=3).mean()
            group[f"{metric}_EWM"] = prior.ewm(span=10, min_periods=3, adjust=False).mean()
        output_parts.append(group)
    return pd.concat(output_parts, ignore_index=True)


def _feature_names() -> list[str]:
    names = ["REST_DAYS", "GAMES_PLAYED"]
    names.extend(
        f"{metric}_{suffix}"
        for metric in TEAM_METRICS
        for suffix in ("L5", "L10", "EWM")
    )
    return names


def _add_matchup_features(frame: pd.DataFrame) -> pd.DataFrame:
    for name in _feature_names():
        frame[f"DIFF_{name}"] = frame[f"HOME_{name}"] - frame[f"AWAY_{name}"]
    frame["EXPECTED_PACE"] = (
        frame["HOME_PACE_EWM"] + frame["AWAY_PACE_EWM"]
    ) / 2
    frame["EXPECTED_HOME_RATING"] = (
        frame["HOME_OFF_RATING_EWM"] + frame["AWAY_DEF_RATING_EWM"]
    ) / 2
    frame["EXPECTED_AWAY_RATING"] = (
        frame["AWAY_OFF_RATING_EWM"] + frame["HOME_DEF_RATING_EWM"]
    ) / 2
    frame["EFFICIENCY_TOTAL"] = (
        frame["EXPECTED_HOME_RATING"] + frame["EXPECTED_AWAY_RATING"]
    )
    frame["PACE_X_EFFICIENCY"] = (
        frame["EXPECTED_PACE"] * frame["EFFICIENCY_TOTAL"] / 100
    )
    return frame


def feature_columns() -> list[str]:
    """Return the stable model feature schema."""
    team_features = _feature_names()
    columns = [f"{side}_{name}" for side in ("HOME", "AWAY") for name in team_features]
    columns.extend(f"DIFF_{name}" for name in team_features)
    columns.extend(
        [
            "EXPECTED_PACE",
            "EXPECTED_HOME_RATING",
            "EXPECTED_AWAY_RATING",
            "EFFICIENCY_TOTAL",
            "PACE_X_EFFICIENCY",
        ]
    )
    return columns


def build_training_table(team_logs: pd.DataFrame) -> pd.DataFrame:
    """Build one feature row per completed game."""
    games = _game_rows(team_logs)
    pregame = _pregame_team_features(_team_history(team_logs))
    selected = ["GAME_ID", "TEAM_ID", *_feature_names()]
    home = pregame[pregame["IS_HOME"] == 1][selected].rename(
        columns={column: f"HOME_{column}" for column in selected if column not in {"GAME_ID"}}
    )
    away = pregame[pregame["IS_HOME"] == 0][selected].rename(
        columns={column: f"AWAY_{column}" for column in selected if column not in {"GAME_ID"}}
    )
    table = games[["GAME_ID", "GAME_DATE", "SEASON", "TEAM_ID_HOME", "TEAM_ID_AWAY"]].copy()
    table["TARGET_TOTAL"] = games["PTS_HOME"] + games["PTS_AWAY"]
    table = table.merge(home, on="GAME_ID", validate="one_to_one")
    table = table.merge(away, on="GAME_ID", validate="one_to_one")
    table = _add_matchup_features(table)
    return table.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def _snapshot(
    history: pd.DataFrame, team_id: int, game_date: pd.Timestamp
) -> dict[str, float]:
    past = history[
        (history["TEAM_ID"] == team_id) & (history["GAME_DATE"] < game_date)
    ].sort_values(["GAME_DATE", "GAME_ID"])
    values: dict[str, float] = {
        "REST_DAYS": np.nan,
        "GAMES_PLAYED": float(len(past)),
    }
    if not past.empty:
        values["REST_DAYS"] = float(min((game_date - past["GAME_DATE"].iloc[-1]).days, 7))
    for metric in TEAM_METRICS:
        series = past[metric]
        for window in ROLLING_WINDOWS:
            values[f"{metric}_L{window}"] = (
                float(series.tail(window).mean()) if len(series) >= 3 else np.nan
            )
        values[f"{metric}_EWM"] = (
            float(series.ewm(span=10, min_periods=3, adjust=False).mean().iloc[-1])
            if len(series) >= 3
            else np.nan
        )
    return values


def build_upcoming_table(
    schedule: pd.DataFrame,
    completed_team_logs: pd.DataFrame,
    passthrough_columns: Iterable[str] = ("MARKET_TOTAL",),
) -> pd.DataFrame:
    """Build features for a schedule containing date, home team ID and away team ID."""
    required = {"GAME_DATE", "HOME_TEAM_ID", "AWAY_TEAM_ID"}
    missing = required - set(schedule.columns)
    if missing:
        raise ValueError(f"Upcoming schedule is missing columns: {sorted(missing)}")

    history = _team_history(completed_team_logs)
    rows: list[dict[str, object]] = []
    for item in schedule.to_dict(orient="records"):
        game_date = pd.Timestamp(item["GAME_DATE"])
        row: dict[str, object] = {
            "GAME_DATE": game_date,
            "HOME_TEAM_ID": int(item["HOME_TEAM_ID"]),
            "AWAY_TEAM_ID": int(item["AWAY_TEAM_ID"]),
        }
        for column in passthrough_columns:
            if column in item:
                row[column] = item[column]
        for side in ("HOME", "AWAY"):
            team_id = int(item[f"{side}_TEAM_ID"])
            for name, value in _snapshot(history, team_id, game_date).items():
                row[f"{side}_{name}"] = value
        rows.append(row)

    table = _add_matchup_features(pd.DataFrame(rows))
    return table.sort_values("GAME_DATE").reset_index(drop=True)
