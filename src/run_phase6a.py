"""
Phase 6a runner: per-regime thresholding to correct regime imbalance.

Run from project root:
    python -m src.run_phase6a

Inputs:
    data/processed/train_scored.parquet  (from Phase 5)
    data/processed/val_scored.parquet
    models/isolation_forest.joblib

Outputs:
    data/processed/train_scored_v2.parquet
    data/processed/val_scored_v2.parquet
    models/regime_thresholds.json
    outputs/phase6a/anomaly_rate_by_regime_corrected.png
    outputs/phase6a/threshold_comparison.png
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import DATA_PROCESSED, MODELS_DIR, OUTPUTS_DIR
from src.anomaly_model import (
    compute_per_regime_thresholds,
    apply_per_regime_thresholds,
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
    print("PHASE 6a: PER-REGIME THRESHOLDING")
    print("=" * 70)

    out_dir = OUTPUTS_DIR / "phase6a"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Phase 5 outputs
    train_df = pd.read_parquet(DATA_PROCESSED / "train_scored.parquet")
    val_df = pd.read_parquet(DATA_PROCESSED / "val_scored.parquet")
    print(f"Loaded train ({len(train_df):,}) and val ({len(val_df):,})")

    # 2. Compute per-regime thresholds from TRAINING scores only.
    # We never use validation data to set thresholds - that would
    # contaminate the evaluation.
    train_scores = train_df["anomaly_score"].values
    train_regimes = train_df["regime"]

    thresholds = compute_per_regime_thresholds(
        train_scores, train_regimes, percentile=99
    )
    global_threshold = float(np.percentile(train_scores, 99))

    print("\nPer-regime thresholds (99th percentile of training scores):")
    print(f"  {'GLOBAL (old)':<15s} {global_threshold:.4f}")
    for regime, thresh in sorted(thresholds.items()):
        delta = thresh - global_threshold
        print(f"  {regime:<15s} {thresh:.4f}  (delta vs global: {delta:+.4f})")

    # 3. Apply per-regime thresholds to BOTH train and val.
    # The training anomaly rates per regime should be ~1% by
    # construction. The validation rates are the real evaluation.
    train_df["is_anomaly_v2"] = apply_per_regime_thresholds(
        train_scores, train_regimes, thresholds, global_threshold
    )
    val_df["is_anomaly_v2"] = apply_per_regime_thresholds(
        val_df["anomaly_score"].values,
        val_df["regime"],
        thresholds,
        global_threshold,
    )

    # 4. Compare old vs new anomaly rates per regime
    print("\nAnomaly rates - VALIDATION (the real test):")
    print(f"  {'Regime':<15s} {'Old rate':>10s} {'New rate':>10s} {'Change':>10s}")
    print("  " + "-" * 50)

    comparison = []
    for regime in sorted(val_df["regime"].unique()):
        sub = val_df[val_df["regime"] == regime]
        old_rate = sub["is_anomaly"].mean() * 100
        new_rate = sub["is_anomaly_v2"].mean() * 100
        comparison.append({
            "regime": regime,
            "old_rate_pct": old_rate,
            "new_rate_pct": new_rate,
            "n_samples": len(sub),
        })
        print(f"  {regime:<15s} {old_rate:>9.2f}% {new_rate:>9.2f}% "
              f"{new_rate - old_rate:>+9.2f}%")

    overall_old = val_df["is_anomaly"].mean() * 100
    overall_new = val_df["is_anomaly_v2"].mean() * 100
    print("  " + "-" * 50)
    print(f"  {'OVERALL':<15s} {overall_old:>9.2f}% {overall_new:>9.2f}% "
          f"{overall_new - overall_old:>+9.2f}%")

    # 5. Save the corrected scored dataframes
    train_df.to_parquet(DATA_PROCESSED / "train_scored_v2.parquet", index=False)
    val_df.to_parquet(DATA_PROCESSED / "val_scored_v2.parquet", index=False)
    print(f"\n[saved] {DATA_PROCESSED / 'train_scored_v2.parquet'}")
    print(f"[saved] {DATA_PROCESSED / 'val_scored_v2.parquet'}")

    # 6. Save thresholds for inference time use.
    # The live inference code will load this dict to know what
    # threshold to apply for each detected regime.
    thresholds_path = MODELS_DIR / "regime_thresholds.json"
    with open(thresholds_path, "w") as f:
        json.dump({
            "per_regime": thresholds,
            "global_fallback": global_threshold,
            "percentile": 99,
        }, f, indent=2)
    print(f"[saved] {thresholds_path}")

    # 7. Plot: side-by-side comparison of old vs new anomaly rates
    comp_df = pd.DataFrame(comparison)
    comp_df = comp_df.sort_values("old_rate_pct", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(comp_df))
    width = 0.35

    ax.bar(x - width/2, comp_df["old_rate_pct"], width,
           label="Global threshold", color="indianred")
    ax.bar(x + width/2, comp_df["new_rate_pct"], width,
           label="Per-regime threshold", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(comp_df["regime"])
    ax.set_ylabel("Anomaly rate (%)")
    ax.set_title("Validation anomaly rate by regime: before vs after correction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "threshold_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_dir / 'threshold_comparison.png'}")

    # 8. Plot: corrected anomaly rate by regime
    fig, ax = plt.subplots(figsize=(8, 5))
    by_regime = (
        val_df.groupby("regime")["is_anomaly_v2"].mean() * 100
    ).sort_values(ascending=False)
    ax.bar(by_regime.index, by_regime.values, color="steelblue")
    ax.axhline(overall_new, color="red", ls="--", lw=1,
               label=f"Overall = {overall_new:.2f}%")
    ax.set_ylabel("Anomaly rate (%)")
    ax.set_title("Anomaly rate by regime - corrected with per-regime thresholds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "anomaly_rate_by_regime_corrected.png",
                bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_dir / 'anomaly_rate_by_regime_corrected.png'}")

    print("\n" + "=" * 70)
    print("PHASE 6a COMPLETE")
    print("=" * 70)
    print("\nKey numbers to verify:")
    print("  - Each regime's NEW rate should be ~1% (matches contamination)")
    print("  - Overall NEW rate may rise slightly (was suppressed by")
    print("    over-flagging highway under the global threshold)")


if __name__ == "__main__":
    main()