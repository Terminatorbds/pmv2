"""
Phase 5 runner: train the Isolation Forest, generate anomaly scores,
produce diagnostic plots.

Run from project root:
    python -m src.run_phase5
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import DATA_PROCESSED, MODELS_DIR, OUTPUTS_DIR
from src.preprocess import MODEL_FEATURES
from src.anomaly_model import (
    train_isolation_forest,
    score_samples,
    feature_contributions,
)


# Use the same plot style as Phase 3
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
    print("PHASE 5: ISOLATION FOREST ANOMALY DETECTION")
    print("=" * 70)

    out_dir = OUTPUTS_DIR / "phase5"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load the preprocessed train/val splits from Phase 4
    train_df = pd.read_parquet(DATA_PROCESSED / "train.parquet")
    val_df = pd.read_parquet(DATA_PROCESSED / "val.parquet")
    print(f"Train: {len(train_df):,} rows")
    print(f"Val:   {len(val_df):,} rows")

    # Separate features from metadata
    X_train = train_df[MODEL_FEATURES]
    X_val = val_df[MODEL_FEATURES]

    # 2. Train the model
    print("\nTraining Isolation Forest...")
    model = train_isolation_forest(X_train, contamination=0.01)
    print(f"Trained {len(model.estimators_)} trees on {len(X_train):,} samples")
    # OLD: print(f"Trained {model.n_estimators} trees on {len(X_train):,} samples")

    # 3. Save the trained model
    model_path = MODELS_DIR / "isolation_forest.joblib"
    joblib.dump(model, model_path)
    print(f"[saved] {model_path}")

    # 4. Generate anomaly scores
    train_scores = score_samples(model, X_train)
    val_scores = score_samples(model, X_val)
    print(f"\nScore range (train): {train_scores.min():.4f} to {train_scores.max():.4f}")
    print(f"Score range (val):   {val_scores.min():.4f} to {val_scores.max():.4f}")

    # 5. Choose a decision threshold.
    # We use the 99th percentile of TRAINING scores. Above this value
    # = anomaly. Rationale: we set contamination=0.01, so the model
    # already considers 1% of training to be "outlier." Using the same
    # threshold for inference matches the training assumption.
    threshold = np.percentile(train_scores, 99)
    print(f"\nDecision threshold (99th percentile of train): {threshold:.4f}")

    train_anomaly_pct = (train_scores > threshold).mean() * 100
    val_anomaly_pct = (val_scores > threshold).mean() * 100
    print(f"Anomaly rate (train): {train_anomaly_pct:.2f}%")
    print(f"Anomaly rate (val):   {val_anomaly_pct:.2f}%")

    # 6. Save scores back into the dataframes for downstream analysis
    train_df["anomaly_score"] = train_scores
    train_df["is_anomaly"] = train_scores > threshold
    val_df["anomaly_score"] = val_scores
    val_df["is_anomaly"] = val_scores > threshold

    train_df.to_parquet(DATA_PROCESSED / "train_scored.parquet", index=False)
    val_df.to_parquet(DATA_PROCESSED / "val_scored.parquet", index=False)
    print(f"\n[saved] {DATA_PROCESSED / 'train_scored.parquet'}")
    print(f"[saved] {DATA_PROCESSED / 'val_scored.parquet'}")

    # 7. Plot 1: Score distributions for train vs val.
    # If val scores look very different from train scores, our split
    # captured genuinely different conditions and we should investigate.
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(train_scores, bins=80, alpha=0.5, density=True,
            label=f"Train (n={len(train_scores):,})", color="steelblue")
    ax.hist(val_scores, bins=80, alpha=0.5, density=True,
            label=f"Val (n={len(val_scores):,})", color="darkorange")
    ax.axvline(threshold, color="red", ls="--", lw=1,
               label=f"Threshold = {threshold:.3f}")
    ax.set_xlabel("Anomaly score (higher = more anomalous)")
    ax.set_ylabel("Density")
    ax.set_title("Anomaly score distribution: train vs validation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "score_distributions.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_dir / 'score_distributions.png'}")

    # 8. Plot 2: Anomaly rate by regime.
    # Tells us whether the model is biased toward calling certain
    # regimes anomalous. If 'highway' has a much higher anomaly rate
    # than 'city', the model may just be confused by the smaller
    # highway sample size.
    fig, ax = plt.subplots(figsize=(8, 5))
    by_regime = (
        val_df.groupby("regime")["is_anomaly"].mean() * 100
    ).sort_values(ascending=False)
    ax.bar(by_regime.index, by_regime.to_numpy(), color="steelblue")
    # OLD: ax.bar(by_regime.index, by_regime.values, color="steelblue")
    ax.axhline(val_anomaly_pct, color="red", ls="--", lw=1,
               label=f"Overall = {val_anomaly_pct:.2f}%")
    ax.set_ylabel("Anomaly rate (%)")
    ax.set_title("Anomaly rate by regime (validation set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "anomaly_rate_by_regime.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_dir / 'anomaly_rate_by_regime.png'}")

    # 9. Feature contributions for the top anomalies.
    # We pick the 100 highest-scoring validation samples and ask the
    # model: which features drove these verdicts? This is the
    # interpretation step - for each top anomaly, we can report
    # 'RPM and timing advance contributed most to the anomaly score.'
    print("\nComputing feature contributions for top 100 anomalies...")
    top_idx = val_df["anomaly_score"].nlargest(100).index
    X_top = X_val.loc[top_idx]
    contribs = feature_contributions(model, X_top, MODEL_FEATURES)

    # Average contribution across the top anomalies tells us which
    # features the model finds most discriminating overall
    avg_contrib = contribs.mean().sort_values(ascending=False)
    avg_contrib.to_csv(out_dir / "feature_contributions.csv")
    print(f"[saved] {out_dir / 'feature_contributions.csv'}")

    fig, ax = plt.subplots(figsize=(10, 7))
    avg_contrib.plot(kind="barh", ax=ax, color="steelblue")
    ax.invert_yaxis()
    ax.set_xlabel("Average contribution to anomaly score")
    ax.set_title("Feature contributions to the top 100 anomalies (validation)")
    ax.axvline(0, color="black", lw=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "feature_contributions.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_dir / 'feature_contributions.png'}")

    # 10. Save metadata
    metadata = {
        "n_train_rows": len(train_df),
        "n_val_rows": len(val_df),
        "model_n_estimators": len(model.estimators_),
        "model_contamination": 0.01,
        # OLD: "model_n_estimators": model.n_estimators,
        # OLD: "model_contamination": float(model.contamination),
        "decision_threshold": float(threshold),
        "train_anomaly_rate_pct": float(train_anomaly_pct),
        "val_anomaly_rate_pct": float(val_anomaly_pct),
        "top_5_contributing_features": avg_contrib.head(5).index.tolist(),
    }
    with open(out_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[saved] {out_dir / 'model_metadata.json'}")

    # 11. Top contributing features summary
    print("\nTop 5 features driving anomaly detections:")
    for feat, contrib in avg_contrib.head(5).items():
        print(f"  {feat:40s} {contrib:+.4f}")

    print("\n" + "=" * 70)
    print("PHASE 5 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()