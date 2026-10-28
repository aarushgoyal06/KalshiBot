# NBA totals ensemble

A leakage-safe NBA game-total forecasting pipeline. It downloads official team
box scores, creates rolling pregame features, validates on future seasons, and
trains a weighted ensemble of:

- Ridge regression
- Extremely randomized trees
- Histogram gradient boosting

The blend weights are learned only from expanding-window, out-of-fold season
predictions. Predictions include an 80% empirical interval and, when a market
line is provided, an empirical probability of the over.

> This is a research tool, not a guarantee of profit. Backtests that ignore
> fees, liquidity, line movement, or time leakage will look much better than
> live results.

## Setup

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## 1. Download historical data

Use at least six completed seasons so the expanding-window report has several
honest test folds:

```bash
nba-totals download \
  --start-year 2018 \
  --end-year 2024 \
  --output data/raw/team_game_logs.csv
```

`2018` means the 2018-19 NBA season. NBA Stats sometimes rate-limits requests;
rerun the command later if the service rejects a request.

## 2. Train and validate

```bash
nba-totals train \
  --data data/raw/team_game_logs.csv \
  --model artifacts/totals_ensemble.joblib \
  --report artifacts/validation_report.json
```

The report contains each future-season fold's MAE/RMSE, individual model
scores, an equal-weight ensemble, a season-mean baseline, and learned final
weights. The headline metric is `oof_ensemble`, not in-sample fit.

If you have timestamp-correct historical lines, provide a CSV with one
`GAME_ID,MARKET_TOTAL` row per game:

```bash
nba-totals train \
  --data data/raw/team_game_logs.csv \
  --market-history data/raw/historical_totals.csv
```

The line becomes an ensemble feature and each fold reports
`market_line_baseline`. Do not mix opening and closing lines, or train on
closing lines when your live strategy acts earlier; that introduces
look-ahead bias.

Useful checks before risking money:

1. Beat the season-mean baseline consistently across folds.
2. Add historical closing totals and compare against the closing line.
3. Include fees and realistic fill assumptions in any strategy backtest.
4. Choose an edge threshold once on validation data; do not tune it repeatedly
   on the same test seasons.

## 3. Predict an upcoming slate

Find team IDs:

```bash
nba-totals team-ids
```

Create `data/upcoming.csv`:

```csv
GAME_DATE,HOME_TEAM_ID,AWAY_TEAM_ID,MARKET_TOTAL
2026-10-28,1610612738,1610612752,224.5
```

`MARKET_TOTAL` is optional. Then run:

```bash
nba-totals predict \
  --data data/raw/team_game_logs.csv \
  --model artifacts/totals_ensemble.joblib \
  --schedule data/upcoming.csv \
  --output predictions/upcoming_totals.csv
```

Output fields:

- `PREDICTED_TOTAL`: ensemble point forecast
- `P10_TOTAL` / `P90_TOTAL`: interval based on walk-forward residuals
- `P_OVER`: probability the total exceeds `MARKET_TOTAL`
- `MODEL_EDGE_POINTS`: forecast minus market total

`P_OVER` is an empirical estimate from historical out-of-fold errors. It is
more defensible than treating every prediction error as normally distributed,
but it still requires live calibration monitoring.

## Feature design

Every rolling statistic is shifted by one game before it is computed. The
current game's result therefore cannot enter its own features. Features include
5-game, 10-game, and exponentially weighted:

- possessions and pace
- offensive and defensive rating
- effective field-goal percentage
- turnover, offensive-rebound, and free-throw rates
- points, rest, team differences, and pace/efficiency interactions

Early-season rows naturally use the end of the previous season. The model's
imputers handle teams with fewer than three prior games.

## Tests

```bash
pytest
ruff check .
```

The tests include an explicit leakage check: changing a game's final score
must not change that game's pregame feature values.

## Recommended next data upgrades

The included model deliberately uses reliable free game logs. The highest-value
extensions are:

1. Historical opening and closing totals, timestamped before the prediction.
2. Confirmed starters, injuries, and minutes-weighted player availability.
3. Travel distance, altitude, and dense schedule indicators.
4. A true market backtest with Kalshi prices, order-book liquidity, and fees.

Keep raw forecasts separate from the decision policy. A positive point edge
does not automatically imply a positive expected-value trade.
