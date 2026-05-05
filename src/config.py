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

# Columns to drop from the modeling dataset.
#
# These contribute nothing useful to anomaly detection because they are
# either constant across all sessions, sensors that the firmware never
# populated, or MIL/diagnostic counters that reflect historical state
# rather than real-time engine behaviour.
#
# We will revisit this list after Phase 3's re-EDA on properly aligned
# data - some of these may behave differently than they did in the
# initial buggy view.
COLUMNS_TO_DROP = [
    "COMMANDED_THROTTLE_ACTUATOR",        # was constant 0 in initial check
    "COMMANDED_EVAPORATIVE_PURGE",        # was constant 0 in initial check
    "TIME_SINCE_TROUBLE_CODES_CLEARED",   # was constant 0
    "DISTANCE_TRAVELED_WITH_MIL_ON",      # was constant 255 (sensor max code)
    "WARM_UPS_SINCE_CODES_CLEARED",       # was all-NaN in initial check
    "TIME_RUN_WITH_MIL_ON",               # MIL counter, not real-time state
]