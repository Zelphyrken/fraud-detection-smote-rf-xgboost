"""Shared helpers for the fraud-detection training script and Streamlit app."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

RAW_FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET = "Class"
ENGINEERED_FEATURES = RAW_FEATURES + ["Amount_log1p"]
RANDOM_STATE = 42


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create the report's limited feature-engineering feature: log1p(Amount).

    The original Amount feature is retained. Time is retained as the source's
    relative elapsed-seconds field; it is not mislabeled as a clock hour.
    """
    missing = [column for column in RAW_FEATURES if column not in data.columns]
    if missing:
        raise ValueError(
            "Kolom input tidak lengkap. Kolom yang belum ada: " + ", ".join(missing)
        )
    result = data.loc[:, RAW_FEATURES].copy()
    amount = pd.to_numeric(result["Amount"], errors="coerce")
    if amount.isna().any():
        raise ValueError("Kolom Amount memiliki nilai kosong atau bukan angka.")
    # Amount should be non-negative in this dataset; clip prevents log1p errors
    # if a malformed negative value is encountered, while preserving Amount.
    result["Amount_log1p"] = np.log1p(amount.clip(lower=0))
    return result.loc[:, ENGINEERED_FEATURES]


def validate_dataset_columns(data: pd.DataFrame) -> None:
    missing = [column for column in RAW_FEATURES + [TARGET] if column not in data.columns]
    if missing:
        raise ValueError(
            "Dataset tidak memiliki kolom wajib: " + ", ".join(missing)
            + ". Pastikan file berasal dari Credit Card Fraud Detection Dataset."
        )


def find_threshold_max_f1(y_true: Iterable[int], scores: Iterable[float]) -> tuple[float, float]:
    """Choose a binary decision threshold using validation labels only."""
    from sklearn.metrics import precision_recall_curve

    y_array = np.asarray(y_true, dtype=int)
    score_array = np.asarray(scores, dtype=float)
    precision, recall, thresholds = precision_recall_curve(y_array, score_array)
    if thresholds.size == 0:
        return 0.5, 0.0
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    best_idx = int(np.nanargmax(f1))
    return float(thresholds[best_idx]), float(f1[best_idx])
