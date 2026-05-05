"""
Phase 3 runner: exploratory data analysis on the corrected dataset.

Run from project root:
    python -m src.run_phase3

This re-does the EDA from Phase 1 but on properly aligned data,
combined across all 129 session files. The goal is to inform feature
selection and preprocessing decisions for the model.
"""
import pandas as pd

from src.config import DATA_PROCESSED, OUTPUTS_DIR
from src.eda import (
    statistical_summary,
    plot_distributions_by_session,
    plot_correlation_heatmap,
    find_strong_correlations,
    plot_outlier_boxes,
    plot_temporal_patterns,
    get_sensor_columns,
)


def main():
    print("=" * 70)
    print("PHASE 3: EXPLORATORY DATA ANALYSIS (corrected data)")
    print("=" * 70)

    out_dir = OUTPUTS_DIR / "phase3"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load the cleaned dataset from Phase 2.
    # Try parquet first (faster); fall back to CSV.
    parquet_path = DATA_PROCESSED / "all_sessions_clean.parquet"
    csv_path = DATA_PROCESSED / "all_sessions_clean.csv"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        print(f"Loaded from {parquet_path.name}")
    else:
        df = pd.read_csv(csv_path)
        print(f"Loaded from {csv_path.name}")

    print(f"Shape: {df.shape}")
    print(f"Sensor columns: {len(get_sensor_columns(df))}")

    # 2. Statistical summary (printed and saved)
    print("\n" + "=" * 70)
    print("STATISTICAL SUMMARY")
    print("=" * 70)
    summary = statistical_summary(df)
    print(summary.to_string())
    summary.to_csv(out_dir / "statistical_summary.csv")
    print(f"\n[saved] {out_dir / 'statistical_summary.csv'}")

    # 3. Distributions per session type
    print("\n" + "=" * 70)
    print("DISTRIBUTIONS BY SESSION TYPE")
    print("=" * 70)
    plot_distributions_by_session(df, out_dir / "distributions_by_session.png")
    print(f"[saved] {out_dir / 'distributions_by_session.png'}")

    # 4. Correlation analysis
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)
    corr = plot_correlation_heatmap(df, out_dir / "correlation_heatmap.png")
    print(f"[saved] {out_dir / 'correlation_heatmap.png'}")

    strong = find_strong_correlations(corr, threshold=0.85)
    if strong:
        print(f"\nStrongly correlated pairs (|r| >= 0.85):")
        for a, b, r in strong:
            print(f"  {r:+.2f}   {a}  <-->  {b}")
    else:
        print("\nNo strongly correlated pairs above 0.85.")

    # 5. Box plots per session
    print("\n" + "=" * 70)
    print("OUTLIERS BY SESSION TYPE")
    print("=" * 70)
    plot_outlier_boxes(df, out_dir / "outliers_by_session.png")
    print(f"[saved] {out_dir / 'outliers_by_session.png'}")

    # 6. Temporal warmup analysis
    print("\n" + "=" * 70)
    print("TEMPORAL / WARMUP PATTERNS")
    print("=" * 70)
    plot_temporal_patterns(df, out_dir / "warmup_patterns.png")
    print(f"[saved] {out_dir / 'warmup_patterns.png'}")

    print("\n" + "=" * 70)
    print("PHASE 3 COMPLETE")
    print("=" * 70)
    print(f"All outputs in: {out_dir}")


if __name__ == "__main__":
    main()