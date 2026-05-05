"""
EDA utilities for the carOBD dataset.

These are reusable building blocks - each function does one thing and
returns a figure or a DataFrame. The Phase 3 runner orchestrates them.

Why split into functions?
    - Each function can be tested independently
    - We can call them from notebooks for interactive exploration
    - Re-running one analysis doesn't require re-running everything
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Default style settings - applied once at import time
plt.rcParams.update({
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 9,
})


def get_sensor_columns(df: pd.DataFrame) -> list:
    """
    Return only the numeric sensor columns, excluding our metadata
    (session_type, session_file, session_number).
    """
    metadata = {"session_type", "session_file", "session_number"}
    return [c for c in df.columns if c not in metadata]


def statistical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-feature summary table with mean, std, quartiles, range,
    skewness, and zero-percentage. More informative than df.describe()
    alone because it adds the columns most relevant to anomaly detection.
    """
    sensor_cols = get_sensor_columns(df)
    summary = df[sensor_cols].describe().T
    summary["range"] = summary["max"] - summary["min"]
    summary["skew"] = df[sensor_cols].skew()
    summary["zero_pct"] = (df[sensor_cols] == 0).mean() * 100
    return summary.round(3)


def plot_distributions_by_session(df: pd.DataFrame, save_path: Path):
    """
    For each sensor, overlay distributions from each session type.

    Why this matters for anomaly detection:
        A reading that's "normal" during highway driving may be very
        unusual during idle. The model needs to be regime-aware.
        Sensors whose distributions overlap heavily across regimes
        are good universal signals; sensors that look different in
        every regime need regime-specific handling.
    """
    sensor_cols = get_sensor_columns(df)
    n = len(sensor_cols)
    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 2.4 * n_rows))
    axes = axes.flatten()

    session_types = sorted(df["session_type"].unique())
    palette = sns.color_palette("Set2", n_colors=len(session_types))

    for i, col in enumerate(sensor_cols):
        ax = axes[i]
        for stype, color in zip(session_types, palette):
            data = df.loc[df["session_type"] == stype, col].dropna()  # type: ignore[attr-defined]
            # OLD: data = df.loc[df["session_type"] == stype, col].dropna()
            if len(data) < 10 or data.std() == 0:
                continue
            ax.hist(data, bins=40, alpha=0.45, label=stype,
                    color=color, density=True)
        ax.set_title(col, fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=6, loc="upper right")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distributions by session type (density-normalized)",
                 fontsize=11, y=1.001)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, save_path: Path) -> pd.DataFrame:
    """
    Pearson correlation among sensors. Returns the correlation matrix
    so the caller can extract specific pairs.
    """
    sensor_cols = get_sensor_columns(df)
    # Drop any columns with zero variance - they break correlation
    valid_cols = [c for c in sensor_cols if df[c].std() > 0]
    corr = df[valid_cols].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.3,
                cbar_kws={"shrink": 0.7}, annot_kws={"size": 6}, ax=ax)
    ax.set_title("Correlation heatmap (Pearson)", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return corr


def find_strong_correlations(corr: pd.DataFrame, threshold: float = 0.85):
    """
    Extract feature pairs with |r| > threshold. These are candidates
    for either dropping (one feature is redundant) or combining (the
    pair carries the same signal so we can sum/average them).
    """
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            r = corr.iloc[i, j]
            if abs(r) >= threshold:  # type: ignore[operator]
            # OLD: if abs(r) >= threshold:
                pairs.append((corr.columns[i], corr.columns[j], r))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs


def plot_outlier_boxes(df: pd.DataFrame, save_path: Path):
    """
    Box plots per session type for each sensor. Shows the 'normal' range
    and how many points fall outside it per regime.
    """
    sensor_cols = get_sensor_columns(df)
    n = len(sensor_cols)
    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 2.4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(sensor_cols):
        ax = axes[i]
        # boxplot per session type
        groups = df.groupby("session_type")[col].apply(
            lambda s: s.dropna().values
        )
        ax.boxplot(groups.values, labels=groups.index,
                   flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
                   widths=0.5)
        ax.set_title(col, fontsize=8)
        ax.tick_params(axis="x", labelsize=6, rotation=45)
        ax.tick_params(axis="y", labelsize=7)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Box plots per session type", fontsize=11, y=1.001)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_temporal_patterns(df: pd.DataFrame, save_path: Path):
    """
    Use ENGINE_RUN_TIME to plot how key sensors evolve from cold start
    to steady state. Helps identify warmup behaviour, which is a big
    source of 'normal but different' patterns the model must accept.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    key_signals = [
        ("COOLANT_TEMPERATURE", "Coolant warmup curve"),
        ("INTAKE_AIR_TEMP", "Intake air warmup"),
        ("LONG_TERM_FUEL_TRIM_BANK_1", "Long-term fuel trim"),
        ("CATALYST_TEMPERATURE_BANK1_SENSOR1", "Catalyst light-off"),
    ]

    # Sample a few sessions across types to keep the plot readable
    sample_sessions = (
        df.groupby("session_type")["session_file"]
          .first()
          .tolist()
    )
    palette = sns.color_palette("Set2", n_colors=len(sample_sessions))

    for ax, (col, title) in zip(axes.flatten(), key_signals):
        for sess, color in zip(sample_sessions, palette):
            sub = df[df["session_file"] == sess]
            if col not in sub.columns or sub[col].dropna().empty:
                continue
            ax.plot(sub["ENGINE_RUN_TIME"], sub[col],
                    alpha=0.7, lw=0.7, color=color, label=sess)
        ax.set_xlabel("Engine run time (s)", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="best")

    fig.suptitle("Sensor evolution from cold start (one sample session per type)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)