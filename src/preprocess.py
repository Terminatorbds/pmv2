"""
Preprocessing pipeline for the carOBD anomaly detection model.

The pipeline is built as a scikit-learn ColumnTransformer + Pipeline
combo. This means it can be:
    - Saved to disk with joblib.dump
    - Loaded at inference time and applied to one new row
    - Versioned and tested like any other artifact

The exact same transformation runs at training and inference time,
which is the only way to avoid train-serve skew (the bug where the
model behaves differently in production than in testing).
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Columns to drop entirely - these have no information for our model.
# This is more aggressive than the COLUMNS_TO_DROP from Phase 2 because
# we also discovered FUEL_AIR_COMMANDED_EQUIV_RATIO is all-zero and
# RELATIVE_THROTTLE_POSITION is 59% zero (sparsely populated).
ZERO_INFORMATION_COLUMNS = [
    "FUEL_AIR_COMMANDED_EQUIV_RATIO",     # 100% zero
    "RELATIVE_THROTTLE_POSITION",         # 59% zero, unreliable
]

# Final feature set the model will consume.
# Listed explicitly so we never silently include something we didn't
# mean to. New features added in features.py must be added here too.
MODEL_FEATURES = [
    # Raw sensor readings the model finds informative
    "ENGINE_RPM",
    "VEHICLE_SPEED",
    "THROTTLE",
    "ENGINE_LOAD",
    "COOLANT_TEMPERATURE",
    "LONG_TERM_FUEL_TRIM_BANK_1",
    "SHORT_TERM_FUEL_TRIM_BANK_1",
    "INTAKE_MANIFOLD_PRESSURE",
    "ABSOLUTE_THROTTLE_B",
    "PEDAL_D",
    "PEDAL_E",
    "ABSOLUTE_BAROMETRIC_PRESSURE",
    "INTAKE_AIR_TEMP",
    "TIMING_ADVANCE",
    "CATALYST_TEMPERATURE_BANK1_SENSOR1",
    "CATALYST_TEMPERATURE_BANK1_SENSOR2",
    "CONTROL_MODULE_VOLTAGE",
    "ENGINE_RUN_TIME",
    # Engineered diagnostic features
    "FUEL_TRIM_TOTAL",
    "THROTTLE_VS_ABS_DELTA",
    "PEDAL_D_VS_E_DELTA",
    "CATALYST_DELTA",
    "LOAD_PER_RPM",
]


def build_preprocessing_pipeline() -> Pipeline:
    """
    Construct the sklearn pipeline.

    Steps:
        1. SimpleImputer fills NaN with the median (we only have 0.001%
           NaN, but we want the pipeline to handle it gracefully at
           inference time when one sensor might glitch).
        2. StandardScaler centers each feature at mean 0 and scales to
           std 1. Critical for distance-based methods and for keeping
           features with different units (RPM at thousands vs fuel
           trim at single digits) on equal footing.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])


def filter_warmup_period(df: pd.DataFrame, min_run_time: float = 60) -> pd.DataFrame:
    """
    Drop rows from the first 60 seconds of each session.

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


def split_by_session(df: pd.DataFrame, val_fraction: float = 0.2,
                     seed: int = 42):
    """
    Split into train/validation by SESSION FILE, not by row.

    This is the correct way to split time-series data: entire trips
    go to train or validation, never both. Otherwise the model sees
    consecutive seconds in train and val, which is essentially data
    leakage that inflates validation metrics.
    """
    rng = np.random.default_rng(seed)
    all_files = df["session_file"].unique()
    rng.shuffle(all_files)

    n_val = max(1, int(len(all_files) * val_fraction))
    val_files = set(all_files[:n_val])
    train_files = set(all_files[n_val:])

    train_df = df[df["session_file"].isin(train_files)].copy()
    val_df = df[df["session_file"].isin(val_files)].copy()

    return train_df, val_df, sorted(train_files), sorted(val_files)