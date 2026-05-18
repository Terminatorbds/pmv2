"""
Sliding-window feature engineering for time-series OBD data.

Why this matters:
    Engine faults don't manifest as single-second snapshots - they
    manifest as sustained patterns over tens of seconds. A vacuum
    leak shows as elevated fuel trim sustained for a minute, not
    as a single high reading. Single-row anomaly detection catches
    transient driving events; windowed anomaly detection catches
    behavioral patterns.

This module produces one feature vector per window, where each window
summarizes WINDOW_SIZE seconds of engine operation.

Public API:
    create_windows(df, window_size, step) -> DataFrame of window features
"""
import numpy as np
import pandas as pd


# Sensors we'll compute windowed statistics for.
# We exclude metadata (session_*), the regime label (categorical),
# and ENGINE_RUN_TIME (monotonic - mean/std meaningless).
SENSORS_FOR_WINDOWING = [
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
    "FUEL_TRIM_TOTAL",
    "THROTTLE_VS_ABS_DELTA",
    "PEDAL_D_VS_E_DELTA",
    "CATALYST_DELTA",
    "LOAD_PER_RPM",
]


def _compute_window_features(window: pd.DataFrame) -> dict | None:
    """
    Compute aggregate features for a single window.

    For each sensor, we extract:
        mean   - the typical value during the window
        std    - the variability (high std = unstable/transient)
        min    - the lowest reading
        max    - the highest reading
        range  - max - min (how widely it varied)
        slope  - linear trend (positive = rising, negative = falling)

    The slope is the key feature for detecting developing problems:
        - Coolant temp slope > 0 sustained = warming up or overheating
        - LTFT slope drifting upward = developing vacuum leak
        - Catalyst delta slope downward = catalyst degrading

    Returns None if the window is too short to compute a slope.
    """
    features: dict = {}

    # Window length in samples - used for slope calculation
    n = len(window)
    if n < 2:
        return None  # Can't compute slope from < 2 points

    # Time axis for slope (just 0, 1, 2, ... n-1 since data is 1 Hz)
    t = np.arange(n, dtype=float)
    t_mean = float(t.mean())
    t_var = float(((t - t_mean) ** 2).sum())

    for sensor in SENSORS_FOR_WINDOWING:
        if sensor not in window.columns:
            continue

        # Force conversion to a plain numpy float array so numpy
        # functions see a fully-typed input. Pandas may return an
        # Arrow-backed array otherwise, which numpy handles correctly
        # at runtime but type checkers flag as ambiguous.
        values = np.asarray(window[sensor].values, dtype=float)

        # Skip if all NaN (rare, but possible for sensors with glitches)
        if np.isnan(values).all():
            features[f"{sensor}_mean"] = np.nan
            features[f"{sensor}_std"] = np.nan
            features[f"{sensor}_min"] = np.nan
            features[f"{sensor}_max"] = np.nan
            features[f"{sensor}_range"] = np.nan
            features[f"{sensor}_slope"] = np.nan
            continue

        features[f"{sensor}_mean"] = float(np.nanmean(values))
        features[f"{sensor}_std"] = float(np.nanstd(values))
        features[f"{sensor}_min"] = float(np.nanmin(values))
        features[f"{sensor}_max"] = float(np.nanmax(values))
        features[f"{sensor}_range"] = features[f"{sensor}_max"] - features[f"{sensor}_min"]

        # Linear slope via least-squares.
        # slope = sum((t - t_mean) * (v - v_mean)) / sum((t - t_mean)^2)
        # We use a manual computation rather than np.polyfit because
        # this is ~10x faster when called thousands of times.
        v = values - float(np.nanmean(values))
        cov = float(np.nansum((t - t_mean) * v))
        features[f"{sensor}_slope"] = cov / t_var if t_var > 0 else 0.0

    return features


def _window_regime(window: pd.DataFrame) -> str:
    """
    Determine the regime label for a window.

    Returns the regime if all samples in the window agree on regime,
    otherwise returns 'MIXED' so the caller can decide what to do
    with it.
    """
    regimes = window["regime"].unique()
    if len(regimes) == 1:
        return regimes[0]
    return "MIXED"


def create_windows(
    df: pd.DataFrame,
    window_size: int = 60,
    step: int = 15,
    drop_mixed: bool = True,
) -> pd.DataFrame:
    """
    Slide a window over each session and emit one feature vector per window.

    Args:
        df: cleaned OBD data with 'session_file' column and 'regime'
            column already added by engineer_all_features
        window_size: window length in samples (= seconds at 1 Hz)
        step: shift between consecutive windows in samples
        drop_mixed: if True, discard windows that span multiple regimes

    Returns:
        DataFrame with one row per emitted window. Columns:
            - session_file, session_type, window_start_idx, window_end_idx
            - regime (the single regime label, or 'MIXED' if drop_mixed=False)
            - <sensor>_mean, _std, _min, _max, _range, _slope  for each sensor
    """
    # Sort by session and original index to guarantee correct temporal ordering.
    # Without this, the slide could produce windows that mix samples from
    # different points in time.
    df = df.sort_values(["session_file", "ENGINE_RUN_TIME"]).reset_index(drop=True)

    all_windows = []
    sessions = df["session_file"].unique()

    for session in sessions:
        session_df = df[df["session_file"] == session].reset_index(drop=True)

        # Walk the window across the session
        for start in range(0, len(session_df) - window_size + 1, step):
            end = start + window_size
            window = session_df.iloc[start:end]

            regime = _window_regime(window)
            if drop_mixed and regime == "MIXED":
                continue

            features = _compute_window_features(window)
            if features is None:
                continue

            features["session_file"] = session
            features["session_type"] = session_df["session_type"].iloc[start]
            features["window_start_idx"] = start
            features["window_end_idx"] = end - 1
            features["window_start_runtime"] = session_df["ENGINE_RUN_TIME"].iloc[start]
            features["regime"] = regime

            all_windows.append(features)

    return pd.DataFrame(all_windows)