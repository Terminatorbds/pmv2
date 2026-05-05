"""
Phase 4 runner: feature engineering, train/val split, fit the
preprocessing pipeline, save artifacts.

Run from project root:
    python -m src.run_phase4

Outputs:
    data/processed/train.parquet
    data/processed/val.parquet
    models/preprocessor.joblib
    outputs/phase4/feature_summary.csv
"""
import json
from pathlib import Path

import joblib
import pandas as pd

from src.config import DATA_PROCESSED, MODELS_DIR, OUTPUTS_DIR
from src.features import engineer_all_features
from src.preprocess import (
    ZERO_INFORMATION_COLUMNS,
    MODEL_FEATURES,
    build_preprocessing_pipeline,
    filter_warmup_period,
    split_by_session,
)


def main():
    print("=" * 70)
    print("PHASE 4: FEATURE ENGINEERING & PREPROCESSING")
    print("=" * 70)

    # Make output directories
    out_dir = OUTPUTS_DIR / "phase4"
    out_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Phase 2's clean dataset
    parquet_path = DATA_PROCESSED / "all_sessions_clean.parquet"
    csv_path = DATA_PROCESSED / "all_sessions_clean.csv"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    else:
        df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows from {parquet_path.name}")

    # 2. Drop the zero-information columns we identified in Phase 3
    df = df.drop(columns=[c for c in ZERO_INFORMATION_COLUMNS if c in df.columns])
    print(f"Dropped {len(ZERO_INFORMATION_COLUMNS)} zero-information columns")

    # 3. Engineer derived features
    df = engineer_all_features(df)
    print(f"After feature engineering: {df.shape[1]} columns")
    print(f"Regime distribution:")
    print(df["regime"].value_counts().to_string())

    # 4. Filter the warmup period
    n_before = len(df)
    df = filter_warmup_period(df, min_run_time=60)
    print(f"\nFiltered warmup: {n_before:,} -> {len(df):,} rows "
          f"({n_before - len(df):,} removed)")

    # 5. Verify we have all the model features we expect
    missing = [f for f in MODEL_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing expected features: {missing}")

    # 6. Train/val split BY SESSION FILE - no temporal leakage
    train_df, val_df, train_files, val_files = split_by_session(
        df, val_fraction=0.2, seed=42
    )
    print(f"\nTrain: {len(train_df):,} rows from {len(train_files)} sessions")
    print(f"Val:   {len(val_df):,} rows from {len(val_files)} sessions")

    # 7. Fit the preprocessing pipeline ON TRAIN ONLY.
    # The scaler learns the mean and std from training data. Applying
    # it to validation later uses those frozen stats - which is correct,
    # because we want to know how the model performs on data with
    # exactly the distribution we'll see at inference time.
    pipeline = build_preprocessing_pipeline()

    X_train_raw = train_df[MODEL_FEATURES]
    pipeline.fit(X_train_raw)
    print(f"\nPipeline fitted on {len(MODEL_FEATURES)} features")

    # 8. Transform both train and val.
    # X_train_scaled is what the Isolation Forest will train on in
    # Phase 5. We save it as a DataFrame for readability.
    X_train_scaled = pd.DataFrame(
        pipeline.transform(X_train_raw),
        columns=MODEL_FEATURES,
        index=train_df.index,
    )
    X_val_scaled = pd.DataFrame(
        pipeline.transform(val_df[MODEL_FEATURES]),
        columns=MODEL_FEATURES,
        index=val_df.index,
    )

    # 9. Save the train/val splits.
    # We save the scaled feature matrix AND the metadata (regime,
    # session_file) side by side, so downstream phases can analyze
    # results by regime without re-running preprocessing.
    train_out = X_train_scaled.copy()
    train_out["regime"] = train_df["regime"].values
    train_out["session_file"] = train_df["session_file"].values

    val_out = X_val_scaled.copy()
    val_out["regime"] = val_df["regime"].values
    val_out["session_file"] = val_df["session_file"].values

    train_out.to_parquet(DATA_PROCESSED / "train.parquet", index=False)
    val_out.to_parquet(DATA_PROCESSED / "val.parquet", index=False)
    print(f"\n[saved] {DATA_PROCESSED / 'train.parquet'}")
    print(f"[saved] {DATA_PROCESSED / 'val.parquet'}")

    # 10. Save the fitted pipeline.
    # joblib is sklearn's recommended format - faster than pickle for
    # numpy-heavy objects.
    pipeline_path = MODELS_DIR / "preprocessor.joblib"
    joblib.dump(pipeline, pipeline_path)
    print(f"[saved] {pipeline_path}")

    # 11. Save metadata: feature list, train/val files, scaler stats.
    # This is documentation - so future-you knows exactly what the
    # model was trained on without re-reading the code.
    scaler = pipeline.named_steps["scale"]
    metadata = {
        "model_features": MODEL_FEATURES,
        "n_train_rows": len(train_df),
        "n_val_rows": len(val_df),
        "train_session_files": train_files,
        "val_session_files": val_files,
        "feature_means_train": dict(zip(MODEL_FEATURES, scaler.mean_.tolist())),
        "feature_stds_train": dict(zip(MODEL_FEATURES, scaler.scale_.tolist())),
    }
    metadata_path = out_dir / "preprocessing_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[saved] {metadata_path}")

    # 12. Quick feature summary - check the scaling worked.
    # After StandardScaler, every feature should have mean ~0 and std ~1.
    summary = pd.DataFrame({
        "raw_mean": X_train_raw.mean().round(3),
        "raw_std": X_train_raw.std().round(3),
        "scaled_mean": X_train_scaled.mean().round(4),
        "scaled_std": X_train_scaled.std().round(4),
    })
    summary.to_csv(out_dir / "feature_summary.csv")
    print(f"[saved] {out_dir / 'feature_summary.csv'}")

    print("\nFeature scaling sanity check (scaled_mean should be ~0, scaled_std ~1):")
    print(summary.to_string())

    print("\n" + "=" * 70)
    print("PHASE 4 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()