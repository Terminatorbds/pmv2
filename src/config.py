"""
Central configuration for the predictive maintenance project.

Why this file exists:
    Hardcoding paths and constants throughout the codebase is a maintenance
    nightmare. When you move the project to a new machine, you'd have to
    hunt down every hardcoded path. Centralizing them here means one place
    to change.
"""
from pathlib import Path

# Project root = parent of the src/ folder this file lives in
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

# Session type detection from filename prefix
SESSION_PREFIXES = {
    "drive": "highway",
    "idle":  "idle",
    "live":  "commute",
    "ufpe":  "campus_low_speed",
    "long":  "long_trip",
}

# Columns dropped during loading - these contribute nothing useful for
# anomaly detection because they are either constants across all sessions,
# sensors the firmware never populated, or MIL/diagnostic counters that
# reflect historical state rather than real-time engine behaviour.
COLUMNS_TO_DROP = [
    "COMMANDED_THROTTLE_ACTUATOR",
    "COMMANDED_EVAPORATIVE_PURGE",
    "TIME_SINCE_TROUBLE_CODES_CLEARED",
    "DISTANCE_TRAVELED_WITH_MIL_ON",
    "WARM_UPS_SINCE_CODES_CLEARED",
    "TIME_RUN_WITH_MIL_ON",
]

# Columns with no information beyond the constants above, identified
# during Phase 3 EDA on properly aligned data.
ZERO_INFORMATION_COLUMNS = [
    "FUEL_AIR_COMMANDED_EQUIV_RATIO",   # 100% zero
    "RELATIVE_THROTTLE_POSITION",       # 59% zero, unreliable
]

# Windowing parameters. Used by run_phase4 to slice the time-series
# data into overlapping windows. See src/windowing.py for the algorithm.
WINDOW_SIZE = 60       # samples = seconds at 1 Hz sampling rate
WINDOW_STEP = 15       # 75% overlap between consecutive windows
WINDOW_DROP_MIXED = True   # discard windows that span multiple regimes

# Warmup filter: drop the first 60 seconds of each session.
# During warmup, fuel trims aren't stable and the catalyst hasn't
# lit off. Including this period would blur the model's notion of
# normal steady-state operation.
WARMUP_FILTER_SECONDS = 60

# Train/validation split parameters.
# val_fraction: 20% of sessions held out for evaluation.
# random_seed: fixed for reproducibility.
VAL_FRACTION = 0.2
RANDOM_SEED = 42