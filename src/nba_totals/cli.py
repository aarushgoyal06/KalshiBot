"""Command-line entry points for the forecasting workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nba_totals.data import download_team_game_logs, load_team_game_logs, season_id
from nba_totals.features import build_training_table, build_upcoming_table
from nba_totals.model import TotalsEnsemble


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nba-totals")
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="download regular-season team game logs")
    download.add_argument("--start-year", type=int, required=True)
    download.add_argument("--end-year", type=int, required=True)
    download.add_argument("--output", default="data/raw/team_game_logs.csv")

    train = commands.add_parser("train", help="feature engineer, validate, and train ensemble")
    train.add_argument("--data", default="data/raw/team_game_logs.csv")
    train.add_argument("--model", default="artifacts/totals_ensemble.joblib")
    train.add_argument("--report", default="artifacts/validation_report.json")
    train.add_argument("--features-output", default="data/processed/training_features.csv")
    train.add_argument("--min-train-seasons", type=int, default=3)
    train.add_argument(
        "--market-history",
        help="optional CSV with GAME_ID and timestamp-correct MARKET_TOTAL columns",
    )

    predict = commands.add_parser("predict", help="predict totals for an upcoming schedule")
    predict.add_argument("--data", default="data/raw/team_game_logs.csv")
    predict.add_argument("--model", default="artifacts/totals_ensemble.joblib")
    predict.add_argument("--schedule", required=True)
    predict.add_argument("--output", default="predictions/upcoming_totals.csv")

    teams = commands.add_parser("team-ids", help="print current NBA team IDs")
    teams.add_argument("--output", help="optional CSV output path")
    return parser


def _download(args: argparse.Namespace) -> None:
    if args.end_year < args.start_year:
        raise ValueError("--end-year must be at least --start-year")
    seasons = [season_id(year) for year in range(args.start_year, args.end_year + 1)]
    result = download_team_game_logs(seasons, args.output)
    print(f"Wrote {len(result):,} team-game rows across {len(seasons)} seasons to {args.output}")


def _train(args: argparse.Namespace) -> None:
    logs = load_team_game_logs(args.data)
    table = build_training_table(logs)
    if args.market_history:
        market = pd.read_csv(args.market_history, dtype={"GAME_ID": str})
        required = {"GAME_ID", "MARKET_TOTAL"}
        missing = required - set(market.columns)
        if missing:
            raise ValueError(f"Market history is missing columns: {sorted(missing)}")
        if market["GAME_ID"].duplicated().any():
            raise ValueError("Market history must have one row per GAME_ID")
        market["GAME_ID"] = market["GAME_ID"].str.zfill(10)
        table = table.merge(
            market[["GAME_ID", "MARKET_TOTAL"]],
            on="GAME_ID",
            how="inner",
            validate="one_to_one",
        )
    feature_path = Path(args.features_output)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(feature_path, index=False)

    model = TotalsEnsemble().fit(table, min_train_seasons=args.min_train_seasons)
    model.save(args.model)
    model.write_report(args.report)
    metrics = model.validation_report["oof_ensemble"]
    print(
        f"OOF ensemble: MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}; "
        f"model={args.model}; report={args.report}"
    )


def _predict(args: argparse.Namespace) -> None:
    logs = load_team_game_logs(args.data)
    schedule = pd.read_csv(args.schedule)
    table = build_upcoming_table(schedule, logs)
    predictions = TotalsEnsemble.load(args.model).predict_frame(table)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)
    print(f"Wrote {len(predictions):,} predictions to {output}")


def _team_ids(args: argparse.Namespace) -> None:
    try:
        from nba_api.stats.static import teams
    except ImportError as exc:
        raise RuntimeError("Install project dependencies before listing teams") from exc
    frame = pd.DataFrame(teams.get_teams())[["id", "abbreviation", "full_name"]]
    frame = frame.rename(columns={"id": "TEAM_ID"}).sort_values("abbreviation")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        print(f"Wrote team IDs to {output}")
    else:
        print(frame.to_string(index=False))


def main() -> None:
    args = _parser().parse_args()
    handlers = {
        "download": _download,
        "train": _train,
        "predict": _predict,
        "team-ids": _team_ids,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
