"""
Preprocessing pipeline for the carOBD anomaly detection model.

The pipeline is built as a scikit-learn Pipeline that can be:
    - Saved to disk with joblib.dump
    - Loaded at inference time and applied to one new window
    - Versioned and tested like any other artifact

The exact same transformation runs at training and inference time,
which is the only way to avoid train-serve skew (the bug where the
model behaves differently in production than in testing).
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def build_preprocessing_pipeline() -> Pipeline:
    """
    Construct the sklearn pipeline.

    Steps:
        1. SimpleImputer fills NaN with the median. We only have a tiny
           amount of NaN in our data, but we want the pipeline to handle
           it gracefully at inference time when one sensor might glitch.
        2. StandardScaler centers each feature at mean 0 and scales to
           std 1. Critical for distance-based methods and for keeping
           features with different units on equal footing.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])


def filter_warmup_period(df: pd.DataFrame, min_run_time: float) -> pd.DataFrame:
    """
    Drop rows from the first `min_run_time` seconds of each session.

    During the first minute or so after engine start, fuel trims are
    not yet stable, coolant is below operating temp, and the catalyst
    has not lit off. The 'normal' patterns during this period are
    different from steady-state operation, and including them would
    blur the model's notion of normal.

    For a production system we'd build a separate cold-start model.
    For now, we filter and document the limitation: our anomaly detector
    is a STEADY-STATE detector and should not be applied during the
    first minute of operation.
    """
    return df[df["ENGINE_RUN_TIME"] >= min_run_time].copy()


def split_by_session(df: pd.DataFrame, val_fraction: float, seed: int):
    """
    Split into train/validation by SESSION FILE, not by row.

    This is the correct way to split time-series data: entire trips
    go to train or validation, never both. Otherwise the model sees
    consecutive seconds in train and val, which is essentially data
    leakage that inflates validation metrics.
    """
    rng = np.random.default_rng(seed)
    # Convert to plain numpy str array so np.random.shuffle works
    # correctly. Without this, pandas/Arrow-backed string types trigger
    # a warning and can produce non-deterministic shuffling.
    all_files = np.array(df["session_file"].unique(), dtype=str)
    rng.shuffle(all_files)

    n_val = max(1, int(len(all_files) * val_fraction))
    val_files = set(all_files[:n_val])
    train_files = set(all_files[n_val:])

    train_df = df[df["session_file"].isin(train_files)].copy()
    val_df = df[df["session_file"].isin(val_files)].copy()

    return train_df, val_df, sorted(train_files), sorted(val_files)


def get_window_feature_names(sensors: list) -> list:
    """
    Return the full list of windowed feature column names.

    For each sensor we have 6 summary statistics, giving
    len(sensors) * 6 features total. The exact list is generated
    dynamically so adding a sensor to the windowing module
    automatically extends the feature set.
    """
    statistics = ["mean", "std", "min", "max", "range", "slope"]
    return [f"{s}_{stat}" for s in sensors for stat in statistics]