"""
Data loader for the carOBD dataset.

Handles two responsibilities:
    1. Read CSVs without triggering pandas' false-index-column inference
    2. Combine multiple session files into a single tagged DataFrame

Public API:
    load_session_file(path) -> DataFrame   # one file
    load_all_sessions(raw_dir) -> DataFrame # everything in a folder
"""
from pathlib import Path
import pandas as pd
import re

from src.config import SESSION_PREFIXES, COLUMNS_TO_DROP


def _detect_session_type(filename: str) -> str:
    """
    Map a filename like 'drive2.csv' to a session label like 'highway'.

    Files we don't recognize are tagged 'unknown' rather than crashing,
    so a stray CSV in data/raw/ doesn't blow up the whole pipeline.
    """
    name = filename.lower()
    for prefix, session_type in SESSION_PREFIXES.items():
        if name.startswith(prefix):
            return session_type
    return "unknown"


def _extract_session_number(filename: str) -> int:
    """Pull the digit out of 'drive2.csv' -> 2. Defaults to 0 if none found."""
    match = re.search(r"(\d+)", Path(filename).stem)
    return int(match.group(1)) if match else 0


def _clean_column_names(columns) -> list:
    """
    Strip the trailing ' ()' artifact from header names and fix the
    'ENGINE_RUN_TINE' typo present in the original CSVs.
    """
    cleaned = []
    for col in columns:
        c = col.strip().replace(" ()", "").strip()
        if c == "ENGINE_RUN_TINE":
            c = "ENGINE_RUN_TIME"
        cleaned.append(c)
    return cleaned


def load_session_file(path: Path) -> pd.DataFrame:
    """
    Load a single carOBD CSV with column alignment correctly preserved.

    The bug:
        Each row in the source CSV ends with a trailing comma, creating
        28 fields per data row but only 27 names in the header. Pandas'
        default behaviour interprets this mismatch as "the first field
        must be a row index" and silently consumes the leftmost data
        column, shifting every column label one position to the right
        of its actual data.

    The fix:
        index_col=False tells pandas to NEVER infer an index column,
        keeping data and headers correctly aligned. The trailing empty
        field becomes a phantom 'Unnamed' column that we drop.

    Type coercion:
        Some files contain non-numeric strings in otherwise numeric
        columns (whitespace, formatting artifacts). We force everything
        to numeric and convert un-parseable values to NaN, which lets
        downstream code handle bad data uniformly.
    """
    path = Path(path)

    # index_col=False is the key fix - it disables pandas' broken
    # auto-detection of index columns when the data has more fields
    # than the header has names.
    df = pd.read_csv(path, index_col=False)

    # Drop the phantom column created by the trailing comma in each row
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Clean up the header names: strip ' ()' and fix the 'TINE' typo
    df.columns = _clean_column_names(df.columns)

    # Verify we got the expected 27 columns. If a future file has a
    # different layout we want to know immediately, not produce silent
    # garbage downstream.
    if df.shape[1] != 27:
        raise ValueError(
            f"{path.name}: expected 27 columns after cleanup, got {df.shape[1]}. "
            f"This file may have a different format - inspect manually."
        )

    # Force every sensor column to numeric. Some source files contain
    # whitespace-padded values or other artifacts that cause pandas to
    # store entire columns as strings. errors='coerce' converts any
    # unparseable value to NaN rather than crashing.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Tag with provenance so we can analyze by session type later
    df["session_type"] = _detect_session_type(path.name)
    df["session_file"] = path.name
    df["session_number"] = _extract_session_number(path.name)

    return df


def load_all_sessions(raw_dir: Path) -> pd.DataFrame:
    """
    Load every CSV in raw_dir and concatenate into one tagged DataFrame.

    Returns a DataFrame with the 27 sensor columns plus session_type,
    session_file, and session_number metadata columns.
    """
    raw_dir = Path(raw_dir)
    csv_files = sorted(raw_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    print(f"Found {len(csv_files)} CSV file(s) in {raw_dir}")

    frames = []
    for csv_path in csv_files:
        try:
            df = load_session_file(csv_path)
            frames.append(df)
            print(f"  [OK] {csv_path.name:25s} {len(df):>6,} rows  "
                  f"({_detect_session_type(csv_path.name)})")
        except Exception as e:
            print(f"  [SKIP] {csv_path.name:25s} {e}")

    if not frames:
        raise RuntimeError("No files loaded successfully.")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nCombined: {len(combined):,} total rows from {len(frames)} file(s)")
    return combined


def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that contribute nothing to anomaly detection.
    Returns a NEW DataFrame - the original is not modified.
    """
    cols_present = [c for c in COLUMNS_TO_DROP if c in df.columns]
    return df.drop(columns=cols_present)
