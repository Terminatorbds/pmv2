"""
Phase 6b runner: inspect the top anomalies, categorize them, and
produce a human-readable report.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import DATA_PROCESSED, MODELS_DIR, OUTPUTS_DIR
from src.preprocess import MODEL_FEATURES, ZERO_INFORMATION_COLUMNS
from src.features import engineer_all_features
from src.preprocess import filter_warmup_period
from src.inspect_anomalies import (
    per_sample_contributions,
    categorize_anomaly,
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

    # 1. Load model and the scored validation set
    val_scored = pd.read_parquet(DATA_PROCESSED / "val_scored_v2.parquet")
    train_scored = pd.read_parquet(DATA_PROCESSED / "train_scored_v2.parquet")
    model = joblib.load(MODELS_DIR / "isolation_forest.joblib")
    print(f"Loaded scored validation ({len(val_scored):,} rows)")

    # 2. Rebuild the RAW feature frame using the EXACT same pipeline
    # as Phase 4, so indices align perfectly. We reset_index(drop=True)
    # at every step to ensure deterministic alignment.
    raw = pd.read_parquet(DATA_PROCESSED / "all_sessions_clean.parquet")
    raw = raw.drop(columns=[c for c in ZERO_INFORMATION_COLUMNS if c in raw.columns])
    raw = engineer_all_features(raw)
    raw = filter_warmup_period(raw, min_run_time=60)
    raw = raw.reset_index(drop=True)
    print(f"Rebuilt raw feature frame ({len(raw):,} rows)")

    # 3. The val_scored frame contains 'session_file' but not the
    # row index from raw. We rebuild val by filtering raw to the
    # validation session_files in the same order val_scored has them.
    val_files = set(val_scored["session_file"].unique())
    raw_val = raw[raw["session_file"].isin(val_files)].reset_index(drop=True)

    # CRITICAL alignment check: raw_val and val_scored must have the
    # same number of rows AND the same session_file in the same order.
    # If this fails we know immediately rather than producing garbage.
    assert len(raw_val) == len(val_scored), (
        f"Row count mismatch: raw_val={len(raw_val)}, "
        f"val_scored={len(val_scored)}"
    )
    file_match = (np.asarray(raw_val["session_file"].values) == np.asarray(val_scored["session_file"].values)).all()
    # OLD: file_match = (raw_val["session_file"].values == val_scored["session_file"].values).all()
    assert file_match, "session_file order mismatch between raw_val and val_scored"
    print("Alignment check passed: raw_val and val_scored are row-aligned")

    # 4. Identify the top 50 flagged anomalies from val_scored
    flagged_mask = np.asarray(val_scored["is_anomaly_v2"].values)
    print(f"\nValidation set has {int(flagged_mask.sum()):,} flagged anomalies")
    # OLD: flagged_mask = val_scored["is_anomaly_v2"].values
    # OLD: print(f"\nValidation set has {flagged_mask.sum():,} flagged anomalies")

    flagged_df = val_scored[flagged_mask].copy()
    top50_positions = flagged_df["anomaly_score"].nlargest(50).index.values
    top50_scaled = val_scored.loc[top50_positions]
    top50_raw = raw_val.loc[top50_positions]
    print(f"Inspecting top 50 anomalies")

    # 5. Compute per-sample feature contributions on the SCALED features
    medians = train_scored[MODEL_FEATURES].median()
    contributions = per_sample_contributions(
        model, top50_scaled[MODEL_FEATURES], medians, MODEL_FEATURES
    )
    print(f"Computed feature contributions for top 50")

    # 6. Categorize each anomaly
    categories = contributions.apply(categorize_anomaly, axis=1)

    # 7. Write the report using BOTH raw and scaled values for
    # transparency. This way we never confuse them again.
    report_path = out_dir / "top_anomalies_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("TOP 50 ANOMALIES - INSPECTION REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("Category distribution (top 50):\n")
        f.write(categories.value_counts().to_string())
        f.write("\n\n")

        for rank, pos in enumerate(top50_positions, 1):
            scaled_row = top50_scaled.loc[pos]
            raw_row = top50_raw.loc[pos]
            contrib_row = contributions.loc[pos]

            f.write(f"--- Rank #{rank} ---\n")
            f.write(f"Session: {raw_row['session_file']}  ")
            f.write(f"(global row {pos}, "
                    f"local row in session ~{int(raw_row['ENGINE_RUN_TIME'])}s into drive)\n")
            f.write(f"Score: {scaled_row['anomaly_score']:.4f}  "
                    f"Regime: {scaled_row['regime']}\n")
            f.write(f"Category: {categories.loc[pos]}\n")

            f.write("\nEngine state at this moment (RAW values):\n")
            engine_state = [
                "ENGINE_RPM", "VEHICLE_SPEED", "THROTTLE", "ENGINE_LOAD",
                "COOLANT_TEMPERATURE", "INTAKE_MANIFOLD_PRESSURE",
                "LONG_TERM_FUEL_TRIM_BANK_1", "SHORT_TERM_FUEL_TRIM_BANK_1",
                "FUEL_TRIM_TOTAL", "TIMING_ADVANCE",
                "ABSOLUTE_THROTTLE_B", "PEDAL_D", "PEDAL_E",
                "THROTTLE_VS_ABS_DELTA", "PEDAL_D_VS_E_DELTA",
                "CATALYST_TEMPERATURE_BANK1_SENSOR1",
                "CATALYST_TEMPERATURE_BANK1_SENSOR2",
                "CATALYST_DELTA", "CONTROL_MODULE_VOLTAGE",
            ]
            for feat in engine_state:
                if feat in raw_row.index:
                    f.write(f"  {feat:38s} {float(raw_row[feat]):>10.3f}\n")
            # OLD: f.write(f"  {feat:38s} {raw_row[feat]:>10.3f}\n")

            f.write("\nTop 5 features driving the anomaly verdict:\n")
            top_contribs = contrib_row.nlargest(5)
            for feat, contrib in top_contribs.items():
                raw_val = float(raw_row.get(feat, float("nan")))
                f.write(f"  {feat:38s} raw={raw_val:>10.3f}  "
                        f"contrib={contrib:+.4f}\n")
            # OLD: raw_val = raw_row.get(feat, float("nan"))
            # OLD: f.write(f"  {feat:38s} raw={raw_val:>10.3f}  "
            f.write("\n")

    print(f"\n[saved] {report_path}")

    # 8. Plots
    fig, ax = plt.subplots(figsize=(9, 5))
    cat_counts = categories.value_counts()
    ax.bar(cat_counts.index, cat_counts.to_numpy(), color="steelblue")
    # OLD: ax.bar(cat_counts.index, cat_counts.values, color="steelblue")
    ax.set_ylabel("Count (out of top 50 anomalies)")
    ax.set_title("Anomaly category distribution (top 50 in validation)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / "anomaly_categories_distribution.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_dir / 'anomaly_categories_distribution.png'}")

    # 9. Per-session anomaly counts (uses ALL flagged samples, not just top 50)
    per_session = (
        val_scored[flagged_mask]
            .groupby("session_file")
            .size()
            .sort_values(ascending=False)
    )
    per_session.to_csv(out_dir / "anomalies_by_session_file.csv",
                      header=["anomaly_count"])
    print(f"[saved] {out_dir / 'anomalies_by_session_file.csv'}")

    # 10. Console summary
    print("\n" + "=" * 70)
    print("SUMMARY OF TOP 50 ANOMALIES")
    print("=" * 70)
    print(f"\nCategory distribution:")
    print(categories.value_counts().to_string())
    print(f"\nDistribution by regime:")
    print(top50_scaled["regime"].value_counts().to_string())
    print(f"\nTop 5 sessions contributing flagged anomalies:")
    print(per_session.head().to_string())
    print(f"\nScore range of top 50: "
          f"{top50_scaled['anomaly_score'].min():.4f} to "
          f"{top50_scaled['anomaly_score'].max():.4f}")

    print("\n" + "=" * 70)
    print("PHASE 6b COMPLETE")
    print("=" * 70)
    print(f"\nOpen {report_path} to read each anomaly's details.")


if __name__ == "__main__":
    main()