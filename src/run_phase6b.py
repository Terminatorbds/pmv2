"""
Phase 6b runner: inspect the top anomalies, categorize them, and
produce a human-readable report.

Run from project root:
    python -m src.run_phase6b

Inputs:
    data/processed/all_sessions_clean.parquet  (raw values for context)
    data/processed/val_scored_v2.parquet
    models/isolation_forest.joblib

Outputs:
    outputs/phase6b/top_anomalies_report.txt
    outputs/phase6b/anomaly_categories_distribution.png
    outputs/phase6b/anomalies_by_session_file.csv
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import DATA_PROCESSED, MODELS_DIR, OUTPUTS_DIR
from src.preprocess import MODEL_FEATURES
from src.features import engineer_all_features
from src.inspect_anomalies import (
    get_top_anomalies,
    per_sample_contributions,
    categorize_anomaly,
    explain_anomaly,
)


plt.rcParams.update({
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 9,
})


def main():
    print("=" * 70)
    print("PHASE 6b: ANOMALY INSPECTION")
    print("=" * 70)

    out_dir = OUTPUTS_DIR / "phase6b"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load: the corrected scored validation set + the model.
    # We also need the raw (engineering-units) data to inspect what
    # the engine was actually doing at each anomalous moment.
    val_scored = pd.read_parquet(DATA_PROCESSED / "val_scored_v2.parquet")
    model = joblib.load(MODELS_DIR / "isolation_forest.joblib")
    print(f"Loaded scored validation set ({len(val_scored):,} rows)")
    print(f"Loaded trained Isolation Forest")

    # 2. The scored validation set has SCALED features. We need the
    # ORIGINAL values to interpret what the engine was doing. Reload
    # and re-engineer to get raw values aligned with our val indices.
    raw = pd.read_parquet(DATA_PROCESSED / "all_sessions_clean.parquet")
    raw = raw.drop(columns=[c for c in [
        "FUEL_AIR_COMMANDED_EQUIV_RATIO",
        "RELATIVE_THROTTLE_POSITION",
    ] if c in raw.columns])
    raw = engineer_all_features(raw)
    raw = raw[raw["ENGINE_RUN_TIME"] >= 60].copy()
    print(f"Loaded raw features ({len(raw):,} rows)")

    # The val_scored DataFrame uses the same indices as the raw frame
    # (we preserved them through Phase 4). Subset to the validation
    # rows by matching session files.
    val_files = val_scored["session_file"].unique()
    raw_val = raw[raw["session_file"].isin(val_files)].copy()

    # Merge raw values into the scored frame.
    # We use index-based alignment since both share the original
    # DataFrame index from the cleaned dataset.
    merged = val_scored.copy()
    raw_only_cols = [c for c in raw_val.columns if c not in merged.columns]
    raw_aligned = raw_val[raw_only_cols].reindex(merged.index)
    for c in raw_only_cols:
        merged[c] = raw_aligned[c].values

    # 3. Get the top 50 anomalies (highest scores among flagged samples).
    # We need the SCALED features to recompute contributions through
    # the model, and the RAW features for human interpretation.
    flagged = merged[merged["is_anomaly_v2"]].copy()
    print(f"\nValidation set has {len(flagged):,} flagged anomalies")
    print(f"Inspecting top 50...")

    top50 = flagged.nlargest(50, "anomaly_score")

    # 4. Compute per-sample feature contributions.
    # We use the scaled values (model features) from val_scored,
    # which are already normalized.
    X_top_scaled = top50[MODEL_FEATURES]

    # Use the median of TRAINING scaled features as the neutral value.
    # Since features are standard-scaled, median is approximately 0.
    train_scored = pd.read_parquet(DATA_PROCESSED / "train_scored_v2.parquet")
    medians = train_scored[MODEL_FEATURES].median()

    contributions = per_sample_contributions(
        model, X_top_scaled, medians, MODEL_FEATURES
    )
    print(f"Computed feature contributions for top 50")

    # 5. Categorize each anomaly using our heuristic mapping
    categories = contributions.apply(categorize_anomaly, axis=1)
    top50["category"] = categories.values

    # 6. Write a human-readable report
    report_path = out_dir / "top_anomalies_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("TOP 50 ANOMALIES - INSPECTION REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("Category distribution:\n")
        f.write(categories.value_counts().to_string())
        f.write("\n\n")

        for rank, (idx, row) in enumerate(top50.iterrows(), 1):
            f.write(f"--- Rank #{rank} ---\n")
            f.write(f"Session: {row['session_file']}  (row index {idx})\n")
            explanation = explain_anomaly(row, contributions.loc[idx], top_k=5)
            f.write(explanation + "\n\n")

    print(f"\n[saved] {report_path}")

    # 7. Plot: distribution of anomaly categories
    fig, ax = plt.subplots(figsize=(9, 5))
    cat_counts = categories.value_counts()
    ax.bar(cat_counts.index, cat_counts.values, color="steelblue")
    ax.set_ylabel("Count (out of top 50 anomalies)")
    ax.set_title("Anomaly category distribution (top 50 in validation)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / "anomaly_categories_distribution.png",
                bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_dir / 'anomaly_categories_distribution.png'}")

    # 8. Save per-session anomaly counts.
    # If anomalies cluster heavily in 2-3 specific sessions, those
    # sessions may have been recorded under unusual conditions
    # (cold weather, high load, etc.) rather than indicating engine
    # faults.
    per_session = (
        flagged.groupby("session_file")
               .size()
               .sort_values(ascending=False)
    )
    per_session.to_csv(out_dir / "anomalies_by_session_file.csv",
                      header=["anomaly_count"])
    print(f"[saved] {out_dir / 'anomalies_by_session_file.csv'}")

    # 9. Print summary to console
    print("\n" + "=" * 70)
    print("SUMMARY OF TOP 50 ANOMALIES")
    print("=" * 70)
    print(f"\nCategory distribution:")
    print(categories.value_counts().to_string())

    print(f"\nDistribution by regime:")
    print(top50["regime"].value_counts().to_string())

    print(f"\nTop 5 sessions contributing flagged anomalies:")
    print(per_session.head().to_string())

    print(f"\nScore range of top 50: {top50['anomaly_score'].min():.4f} "
          f"to {top50['anomaly_score'].max():.4f}")

    print("\n" + "=" * 70)
    print("PHASE 6b COMPLETE")
    print("=" * 70)
    print(f"\nOpen {report_path} to read each anomaly's details.")


if __name__ == "__main__":
    main()