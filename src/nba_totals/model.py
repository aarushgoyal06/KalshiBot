"""Time-aware validation and weighted ensemble training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nba_totals.features import feature_columns


def _base_models(random_state: int) -> dict[str, RegressorMixin]:
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    return {
        "ridge": make_pipeline(
            clone(imputer),
            StandardScaler(),
            Ridge(alpha=25.0),
        ),
        "extra_trees": make_pipeline(
            clone(imputer),
            ExtraTreesRegressor(
                n_estimators=500,
                min_samples_leaf=4,
                max_features=0.7,
                n_jobs=-1,
                random_state=random_state,
            ),
        ),
        "hist_gradient_boosting": make_pipeline(
            clone(imputer),
            HistGradientBoostingRegressor(
                learning_rate=0.04,
                max_iter=350,
                max_leaf_nodes=15,
                l2_regularization=5.0,
                early_stopping=True,
                random_state=random_state,
            ),
        ),
    }


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def _season_folds(
    table: pd.DataFrame, min_train_seasons: int
) -> list[tuple[list[str], str]]:
    season_order = (
        table.groupby("SEASON", as_index=False)["GAME_DATE"]
        .min()
        .sort_values("GAME_DATE")["SEASON"]
        .astype(str)
        .tolist()
    )
    if len(season_order) <= min_train_seasons:
        raise ValueError(
            f"Need more than {min_train_seasons} seasons for walk-forward validation; "
            f"found {len(season_order)}"
        )
    return [
        (season_order[:index], season_order[index])
        for index in range(min_train_seasons, len(season_order))
    ]


@dataclass
class TotalsEnsemble:
    """A weighted blend trained from walk-forward out-of-fold predictions."""

    models: dict[str, RegressorMixin] | None = None
    weights: dict[str, float] | None = None
    bias: float = 0.0
    residuals: np.ndarray | None = None
    validation_report: dict[str, Any] | None = None
    model_features: list[str] | None = None
    random_state: int = 42

    def fit(
        self,
        table: pd.DataFrame,
        min_train_seasons: int = 3,
    ) -> TotalsEnsemble:
        columns = feature_columns()
        uses_market_line = "MARKET_TOTAL" in table.columns
        if uses_market_line:
            columns = [*columns, "MARKET_TOTAL"]
        self.model_features = columns
        required = {"GAME_DATE", "SEASON", "TARGET_TOTAL", *columns}
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"Training table is missing columns: {sorted(missing)}")

        clean = table.copy()
        clean["GAME_DATE"] = pd.to_datetime(clean["GAME_DATE"])
        clean = clean.dropna(subset=["TARGET_TOTAL", "PACE_X_EFFICIENCY", *(
            ["MARKET_TOTAL"] if uses_market_line else []
        )])
        clean = clean.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
        folds = _season_folds(clean, min_train_seasons)
        model_templates = _base_models(self.random_state)
        oof_parts: list[pd.DataFrame] = []
        fold_reports: list[dict[str, Any]] = []

        for train_seasons, test_season in folds:
            train = clean[clean["SEASON"].astype(str).isin(train_seasons)]
            test = clean[clean["SEASON"].astype(str) == test_season]
            x_train, y_train = train[columns], train["TARGET_TOTAL"].to_numpy()
            x_test, y_test = test[columns], test["TARGET_TOTAL"].to_numpy()
            predictions: dict[str, np.ndarray] = {}
            model_metrics: dict[str, dict[str, float]] = {}
            for name, template in model_templates.items():
                model = clone(template).fit(x_train, y_train)
                prediction = model.predict(x_test)
                predictions[name] = prediction
                model_metrics[name] = _metrics(y_test, prediction)

            fold_frame = pd.DataFrame(predictions, index=test.index)
            fold_frame["actual"] = y_test
            oof_parts.append(fold_frame)
            equal_blend = np.mean(np.column_stack(list(predictions.values())), axis=1)
            fold_report = {
                "test_season": test_season,
                "train_seasons": train_seasons,
                "games": len(test),
                "models": model_metrics,
                "equal_blend": _metrics(y_test, equal_blend),
                "season_mean_baseline": _metrics(
                    y_test, np.full(len(test), y_train.mean())
                ),
            }
            if uses_market_line:
                fold_report["market_line_baseline"] = _metrics(
                    y_test, test["MARKET_TOTAL"].to_numpy()
                )
            fold_reports.append(fold_report)

        oof = pd.concat(oof_parts).sort_index()
        model_names = list(model_templates)
        meta = Ridge(alpha=10.0, positive=True, fit_intercept=False)
        meta.fit(oof[model_names], oof["actual"])
        raw_weights = np.maximum(meta.coef_, 0)
        if raw_weights.sum() == 0:
            raw_weights = np.ones(len(model_names))
        normalized = raw_weights / raw_weights.sum()
        self.weights = dict(zip(model_names, normalized, strict=True))
        blended = oof[model_names].to_numpy() @ normalized
        self.bias = float(np.mean(oof["actual"].to_numpy() - blended))
        final_oof = blended + self.bias
        self.residuals = oof["actual"].to_numpy() - final_oof

        self.models = {}
        x_all, y_all = clean[columns], clean["TARGET_TOTAL"].to_numpy()
        for name, template in model_templates.items():
            self.models[name] = clone(template).fit(x_all, y_all)

        self.validation_report = {
            "strategy": "expanding-window season folds",
            "training_games": len(clean),
            "uses_market_line": uses_market_line,
            "folds": fold_reports,
            "learned_weights": self.weights,
            "bias": self.bias,
            "oof_ensemble": _metrics(oof["actual"].to_numpy(), final_oof),
            "oof_residual_std": float(np.std(self.residuals, ddof=1)),
        }
        return self

    def _check_fitted(self) -> None:
        if not self.models or not self.weights or self.residuals is None:
            raise RuntimeError("Fit or load the ensemble before predicting")

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        columns = self.model_features or feature_columns()
        missing = set(columns) - set(table.columns)
        if missing:
            raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
        predictions = sum(
            self.weights[name] * model.predict(table[columns])
            for name, model in self.models.items()
        )
        return np.asarray(predictions) + self.bias

    def predict_frame(self, table: pd.DataFrame) -> pd.DataFrame:
        """Predict totals, empirical intervals, and over probability if a line exists."""
        self._check_fitted()
        output_columns = [
            column
            for column in ("GAME_DATE", "HOME_TEAM_ID", "AWAY_TEAM_ID", "MARKET_TOTAL")
            if column in table
        ]
        output = table[output_columns].copy()
        output["PREDICTED_TOTAL"] = self.predict(table)
        lower, upper = np.quantile(self.residuals, [0.1, 0.9])
        output["P10_TOTAL"] = output["PREDICTED_TOTAL"] + lower
        output["P90_TOTAL"] = output["PREDICTED_TOTAL"] + upper
        if "MARKET_TOTAL" in output:
            threshold = (
                output["MARKET_TOTAL"].to_numpy() - output["PREDICTED_TOTAL"].to_numpy()
            )
            # Smoothed empirical CDF avoids assuming normally distributed scoring errors.
            over_count = (self.residuals[:, None] > threshold[None, :]).sum(axis=0)
            output["P_OVER"] = (over_count + 1) / (len(self.residuals) + 2)
            output["MODEL_EDGE_POINTS"] = (
                output["PREDICTED_TOTAL"] - output["MARKET_TOTAL"]
            )
        return output

    def save(self, path: str | Path) -> None:
        self._check_fitted()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    @classmethod
    def load(cls, path: str | Path) -> TotalsEnsemble:
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"{path} does not contain a TotalsEnsemble")
        return model

    def write_report(self, path: str | Path) -> None:
        if self.validation_report is None:
            raise RuntimeError("No validation report is available")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.validation_report, indent=2) + "\n")
