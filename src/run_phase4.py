"""
Phase 4 runner: windowed feature engineering.

Builds 60-second windows with 15-second step over each session,
computes per-sensor statistics, and produces a windowed train/val
split ready for model training.

Run from project root:
    python -m src.run_phase4

Inputs:
    data/processed/all_sessions_clean.parquet  (from Phase 2)

Outputs:
    data/processed/windows_train.parquet
    data/processed/windows_val.parquet
    models/preprocessor_windowed.joblib
    outputs/phase4/preprocessing_metadata.json
    outputs/phase4/window_summary.csv
"""
import json

import joblib
import pandas as pd

from src.config import (
    DATA_PROCESSED, MODELS_DIR, OUTPUTS_DIR,
    ZERO_INFORMATION_COLUMNS,
    WINDOW_SIZE, WINDOW_STEP, WINDOW_DROP_MIXED,
    WARMUP_FILTER_SECONDS,
    VAL_FRACTION, RANDOM_SEED,
)
from src.features import engineer_all_features
from src.preprocess import (
    build_preprocessing_pipeline,
    filter_warmup_period,
    split_by_session,
    get_window_feature_names,
)
from src.windowing import create_windows, SENSORS_FOR_WINDOWING


def main():
    print("=" * 70)
    print("PHASE 4: WINDOWED FEATURE ENGINEERING")
    print("=" * 70)
    print(f"Window size: {WINDOW_SIZE}s   Step: {WINDOW_STEP}s   "
          f"Overlap: {(1 - WINDOW_STEP/WINDOW_SIZE)*100:.0f}%")
    print(f"Mixed-regime windows: {'DROPPED' if WINDOW_DROP_MIXED else 'KEPT'}")

    out_dir = OUTPUTS_DIR / "phase4"
    out_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Phase 2 clean data
    df = pd.read_parquet(DATA_PROCESSED / "all_sessions_clean.parquet")
    print(f"\nLoaded {len(df):,} rows from all_sessions_clean")

    # 2. Drop zero-information columns
    df = df.drop(columns=[c for c in ZERO_INFORMATION_COLUMNS if c in df.columns])

    # 3. Add row-level engineered features (regime, fuel trim sum, etc.)
    # These are NOT our final model features - they're the inputs that
    # the windowing layer will summarize over 60-second windows.
    df = engineer_all_features(df)

    # 4. Filter warmup period
    n_before = len(df)
    df = filter_warmup_period(df, min_run_time=WARMUP_FILTER_SECONDS)
    print(f"After warmup filter: {n_before:,} -> {len(df):,} rows")

    # 5. Build windows.
    # This is the heavy step - we iterate every session and slide the
    # window across it. For 129 sessions x ~2200 rows / 15-sample step
    # we expect ~9000 windows after dropping mixed-regime ones.
    print(f"\nBuilding windows from {df['session_file'].nunique()} sessions...")
    windows = create_windows(
        df,
        window_size=WINDOW_SIZE,
        step=WINDOW_STEP,
        drop_mixed=WINDOW_DROP_MIXED,
    )
    print(f"Created {len(windows):,} windows")

    # 6. Report window counts per regime and session type
    print(f"\nWindow distribution by regime:")
    print(windows["regime"].value_counts().to_string())
    print(f"\nWindow distribution by session type:")
    print(windows["session_type"].value_counts().to_string())

    # 7. Build the model feature list dynamically.
    # If a future sensor is added to SENSORS_FOR_WINDOWING, this
    # picks it up automatically without code changes here.
    model_features = get_window_feature_names(SENSORS_FOR_WINDOWING)
    model_features = [f for f in model_features if f in windows.columns]
    print(f"\nModel feature count: {len(model_features)}")

    # 8. Train/val split by session - prevents temporal leakage.
    # Same logic as before, but operating on windows now.
    train_df, val_df, train_files, val_files = split_by_session(
        windows, val_fraction=VAL_FRACTION, seed=RANDOM_SEED
    )
    print(f"\nTrain: {len(train_df):,} windows from {len(train_files)} sessions")
    print(f"Val:   {len(val_df):,} windows from {len(val_files)} sessions")

    # 9. Fit the preprocessing pipeline on TRAIN ONLY.
    # The scaler learns mean and std from training data. Applying it
    # to validation later uses those frozen stats - which is correct,
    # because we want to know how the model performs on data with
    # exactly the distribution we'll see at inference time.
    pipeline = build_preprocessing_pipeline()
    X_train_raw = train_df[model_features]
    pipeline.fit(X_train_raw)
    print(f"\nPipeline fitted on {len(model_features)} windowed features")

    # 10. Transform both splits
    X_train_scaled = pd.DataFrame(
        pipeline.transform(X_train_raw),
        columns=model_features,
        index=train_df.index,
    )
    X_val_scaled = pd.DataFrame(
        pipeline.transform(val_df[model_features]),
        columns=model_features,
        index=val_df.index,
    )

    # 11. Save train/val splits with metadata attached.
    # The metadata columns let us trace any sample back to its source
    # session and time location, which is essential for interpretation.
    metadata_cols = [
        "session_file", "session_type", "regime",
        "window_start_idx", "window_end_idx", "window_start_runtime",
    ]
    train_out = X_train_scaled.copy()
    for c in metadata_cols:
        train_out[c] = train_df[c].values

    val_out = X_val_scaled.copy()
    for c in metadata_cols:
        val_out[c] = val_df[c].values

    train_path = DATA_PROCESSED / "windows_train.parquet"
    val_path = DATA_PROCESSED / "windows_val.parquet"
    train_out.to_parquet(train_path, index=False)
    val_out.to_parquet(val_path, index=False)
    print(f"\n[saved] {train_path}")
    print(f"[saved] {val_path}")

    # 12. Save the fitted pipeline for inference-time reuse
    pipeline_path = MODELS_DIR / "preprocessor_windowed.joblib"
    joblib.dump(pipeline, pipeline_path)
    print(f"[saved] {pipeline_path}")

    # 13. Save run metadata. Documents exactly what configuration
    # produced these artifacts - essential for reproducibility.
    metadata = {
        "window_size": WINDOW_SIZE,
        "window_step": WINDOW_STEP,
        "drop_mixed": WINDOW_DROP_MIXED,
        "warmup_filter_seconds": WARMUP_FILTER_SECONDS,
        "val_fraction": VAL_FRACTION,
        "random_seed": RANDOM_SEED,
        "model_features": model_features,
        "n_train_windows": len(train_df),
        "n_val_windows": len(val_df),
        "train_session_files": list(train_files),
        "val_session_files": list(val_files),
    }
    metadata_path = out_dir / "preprocessing_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[saved] {metadata_path}")

    # 14. Scaling sanity check
    summary = pd.DataFrame({
        "raw_mean": X_train_raw.mean().round(3),
        "raw_std": X_train_raw.std().round(3),
        "scaled_mean": X_train_scaled.mean().round(4),
        "scaled_std": X_train_scaled.std().round(4),
    })
    summary.to_csv(out_dir / "window_summary.csv")
    print(f"[saved] {out_dir / 'window_summary.csv'}")

    print("\nFirst 10 windowed features (scaled_mean ~0, scaled_std ~1):")
    print(summary.head(10).to_string())

    print("\n" + "=" * 70)
    print("PHASE 4 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()