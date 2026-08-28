"""Inference entry point for the Streamlit app.

Loads the pre-trained artifact (see model/train.py) once and exposes
predict_house_price() for single-row predictions. Does not retrain.
"""

import math
import numbers
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from model.pipeline import HousePriceModel  # noqa: F401  (required for joblib.load)

ARTIFACT_PATH = Path(__file__).resolve().parent / "artifact.joblib"

ORIGINAL_FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

# Basic domain sanity checks — not new ML logic, just guards against
# nonsensical input (negative counts, out-of-range coordinates, etc.).
VALIDATION_RULES = {
    "MedInc": (lambda v: v > 0, "must be positive"),
    "HouseAge": (lambda v: v >= 0, "must be non-negative"),
    "AveRooms": (lambda v: v > 0, "must be positive"),
    "AveBedrms": (lambda v: v > 0, "must be positive"),
    "Population": (lambda v: v >= 0, "must be non-negative"),
    "AveOccup": (lambda v: v > 0, "must be positive"),
    "Latitude": (lambda v: -90 <= v <= 90, "must be between -90 and 90"),
    "Longitude": (lambda v: -180 <= v <= 180, "must be between -180 and 180"),
}


@lru_cache(maxsize=1)
def _load_artifact() -> HousePriceModel:
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {ARTIFACT_PATH}. "
            "Run `python -m model.train` first to generate it."
        )
    return joblib.load(ARTIFACT_PATH)


def _validate(name, value):
    if value is None:
        raise ValueError(f"{name} is required")
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")

    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")

    is_valid, message = VALIDATION_RULES[name]
    if not is_valid(value):
        raise ValueError(f"{name}={value} is invalid: {message}")

    return value


def predict_house_price(
    MedInc,
    HouseAge,
    AveRooms,
    AveBedrms,
    Population,
    AveOccup,
    Latitude,
    Longitude,
):
    """Predict MedHouseVal (in $100,000s) for a single district.

    Applies the exact preprocessing/feature engineering the notebook's
    Gradient Boosting model was trained with, then runs the pre-trained
    model. Raises ValueError/TypeError on invalid input.
    """
    raw_inputs = {
        "MedInc": MedInc,
        "HouseAge": HouseAge,
        "AveRooms": AveRooms,
        "AveBedrms": AveBedrms,
        "Population": Population,
        "AveOccup": AveOccup,
        "Latitude": Latitude,
        "Longitude": Longitude,
    }
    validated = {name: _validate(name, value) for name, value in raw_inputs.items()}

    input_df = pd.DataFrame([validated], columns=ORIGINAL_FEATURES)

    artifact = _load_artifact()
    prediction = artifact.predict(input_df)

    return float(prediction[0])
