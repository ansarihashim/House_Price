"""Reproduces the notebook's Gradient Boosting training workflow exactly
(notebooks/House_PRICE.ipynb, cells 10-12) and serializes the fitted model
plus its preprocessing parameters as a single artifact for inference.

Run with: python -m model.train
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from model.pipeline import HousePriceModel

ARTIFACT_PATH = Path(__file__).resolve().parent / "artifact.joblib"

COLS_TO_WINSORIZE = ["AveRooms", "AveBedrms", "AveOccup", "Population"]
TARGET = "MedHouseVal"

# Reported in the notebook's results summary (cell 18) — used only to sanity
# check that this script reproduces the same trained model.
EXPECTED_METRICS = {"MAE": 0.3300, "RMSE": 0.4899, "R2": 0.8168}


def build_training_artifact():
    housing = fetch_california_housing(as_frame=True)
    df_clean = housing.frame.copy()

    winsorize_bounds = {}
    for col in COLS_TO_WINSORIZE:
        lower = df_clean[col].quantile(0.01)
        upper = df_clean[col].quantile(0.99)
        winsorize_bounds[col] = (lower, upper)
        df_clean[col] = df_clean[col].clip(lower, upper)

    df_clean["RoomsPerBedroom"] = df_clean["AveRooms"] / df_clean["AveBedrms"].replace(0, 1)
    df_clean["BedroomRatio"] = df_clean["AveBedrms"] / df_clean["AveRooms"].replace(0, 1)
    df_clean["PopulationPerHousehold"] = df_clean["Population"] / df_clean["AveOccup"].replace(0, 1)
    df_clean["MedIncSq"] = df_clean["MedInc"] ** 2

    feature_order = [c for c in df_clean.columns if c != TARGET]
    X = df_clean[feature_order]
    y = df_clean[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    gbm = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42
    )
    gbm.fit(X_train, y_train)

    artifact = HousePriceModel(
        winsorize_bounds=winsorize_bounds, feature_order=feature_order, model=gbm
    )

    test_pred = gbm.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, test_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, test_pred)),
        "R2": r2_score(y_test, test_pred),
    }
    return artifact, metrics


def main():
    artifact, metrics = build_training_artifact()
    joblib.dump(artifact, ARTIFACT_PATH)

    print(f"Saved artifact to {ARTIFACT_PATH}")
    for name, value in metrics.items():
        expected = EXPECTED_METRICS[name]
        print(f"Test {name}: {value:.4f} (notebook: {expected:.4f})")
        assert abs(value - expected) < 1e-3, (
            f"{name} mismatch vs notebook: {value:.4f} != {expected:.4f}"
        )


if __name__ == "__main__":
    main()
