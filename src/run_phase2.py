"""
Phase 2 runner: load all sessions, clean, save processed dataset.

Run from project root:
    python -m src.run_phase2

The -m flag tells Python "run this as a module," which is what makes
the `from src.config import ...` imports work.
"""
import pandas as pd

from src.config import DATA_RAW, DATA_PROCESSED
from src.loader import load_all_sessions, drop_useless_columns


def main():
    print("=" * 70)
    print("PHASE 2: DATA LOADING & CLEANING")
    print("=" * 70)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # 1. Load every session file with the column-alignment fix applied
    df = load_all_sessions(DATA_RAW)

    # 2. Sanity check: do the values now look like real engineering units?
    print("\n" + "=" * 70)
    print("SANITY CHECK - do values look physically plausible?")
    print("=" * 70)
    sanity_cols = [
        "ENGINE_RPM",
        "VEHICLE_SPEED",
        "COOLANT_TEMPERATURE",
        "CONTROL_MODULE_VOLTAGE",
        "INTAKE_AIR_TEMP",
    ]
    print(df[sanity_cols].describe().round(2).to_string())
    print("\nExpected ranges:")
    print("  ENGINE_RPM             : ~700 (idle) to ~6000 (redline)")
    print("  VEHICLE_SPEED          : 0 to ~120 km/h")
    print("  COOLANT_TEMPERATURE    : 80-105 C when warm")
    print("  CONTROL_MODULE_VOLTAGE : 12-14.5 V")
    print("  INTAKE_AIR_TEMP        : roughly ambient, 0-50 C")

    # 3. Per-session breakdown
    print("\n" + "=" * 70)
    print("ROWS BY SESSION TYPE")
    print("=" * 70)
    print(df.groupby("session_type").size().to_string())

    # 4. Data quality report - missing values per column.
    # After type coercion, any string garbage in numeric columns has
    # become NaN. Reporting this tells us which sensors are unreliable.
    print("\n" + "=" * 70)
    print("DATA QUALITY - missing values per column")
    print("=" * 70)
    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(3)
    quality = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    quality = quality[quality["missing_count"] > 0].sort_values(
        "missing_count", ascending=False
    )
    if quality.empty:
        print("No missing values - all columns are clean.")
    else:
        print(quality.to_string())

    # 5. Drop the useless columns
    df_clean = drop_useless_columns(df)
    print(f"\nDropped {df.shape[1] - df_clean.shape[1]} columns. "
          f"Final shape: {df_clean.shape}")

    # 6. Save to processed/.
    # Parquet is faster and preserves dtypes, but requires clean data.
    # CSV is the readable backup.
    csv_path = DATA_PROCESSED / "all_sessions_clean.csv"
    df_clean.to_csv(csv_path, index=False)
    print(f"\n[saved] {csv_path}")

    try:
        parquet_path = DATA_PROCESSED / "all_sessions_clean.parquet"
        df_clean.to_parquet(parquet_path, index=False)
        print(f"[saved] {parquet_path}")
    except Exception as e:
        print(f"[WARN] Parquet save failed: {e}")
        print("       CSV was saved successfully and can be used instead.")

    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()