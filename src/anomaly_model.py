"""
Isolation Forest anomaly detection model.

The model learns a baseline of "normal" engine behaviour from the
training set, then scores any new sample by how unusual it looks.

Public API:
    train_isolation_forest(X) -> fitted model
    score_samples(model, X) -> anomaly scores (higher = more anomalous)
    feature_contributions(model, X, feature_names) -> per-feature anomaly
        contribution for interpretation
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def train_isolation_forest(
    X: pd.DataFrame,
    contamination: float = 0.01,
    n_estimators: int = 200,
    max_samples: int = 256,
    random_state: int = 42,
) -> IsolationForest:
    """
    Fit an Isolation Forest to the training feature matrix.

    Hyperparameter rationale:
        contamination: expected fraction of anomalies in training. We use
            0.01 (1%) - the dataset is allegedly all-healthy, but real
            sensor data always has some genuine glitches and edge cases
            we should let the model treat as outliers rather than
            stretching its notion of normal to include them.
        n_estimators: 200 trees gives stable scores. Default is 100;
            paying the small training cost for less variance is worth it.
        max_samples: 256 per tree is the standard sub-sampling size from
            the original Isolation Forest paper. Each tree only sees
            this many random samples, which is what makes the algorithm
            efficient and prevents over-isolation.
        random_state: fixed seed makes results reproducible across runs.
    """
    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        max_samples=max_samples,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X)
    return model


def score_samples(model: IsolationForest, X: pd.DataFrame) -> np.ndarray:
    """
    Return anomaly scores where HIGHER means MORE ANOMALOUS.

    Why we negate the raw score:
        sklearn's score_samples returns higher values for normal points
        (because internally the score is negated path length). We flip
        it here so 'higher = more anomalous,' which is the convention
        most engineers expect.
    """
    return -model.score_samples(X)


def feature_contributions(
    model: IsolationForest,
    X: pd.DataFrame,
    feature_names: list,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    For each row, report which features contributed most to its
    anomaly score.

    The technique: replace one feature at a time with its training
    median (a "neutral" value), recompute the score, and measure
    how much the score dropped. Features whose removal reduces the
    score most are the ones driving the anomaly verdict.

    This gives us interpretable explanations - critical for any
    real diagnostic system. When the model flags a sample, we can
    say "RPM and intake pressure contributed 70% of the anomaly
    score" instead of just "the model says it's wrong."

    Returns a DataFrame with one row per input sample, listing the
    top_k features and their contribution scores.
    """
    base_scores = score_samples(model, X)
    contributions = np.zeros((len(X), len(feature_names)))

    medians = X.median()

    for i, feat in enumerate(feature_names):
        X_modified = X.copy()
        X_modified[feat] = medians[feat]
        new_scores = score_samples(model, X_modified)
        # If neutralizing this feature drops the score, the feature
        # was contributing positively to the anomaly verdict
        contributions[:, i] = base_scores - new_scores

    contrib_df = pd.DataFrame(contributions, columns=feature_names, index=X.index)
    return contrib_df

def compute_per_regime_thresholds(
    scores: np.ndarray,
    regimes: pd.Series,
    percentile: float = 99,
) -> dict:
    """
    Compute a separate decision threshold for each regime.

    Why this matters:
        A global threshold treats every regime equally, but our training
        data is imbalanced (highway = 5% of rows, idle = 35%). The model
        sees lots of idle examples and learns idle's "normal" tightly,
        but sees few highway examples and is uncertain about highway -
        causing it to over-flag highway samples. Per-regime thresholds
        correct this by asking 'is this sample unusual FOR ITS REGIME?'
        rather than 'is this sample unusual GLOBALLY?'

    Returns a dict mapping regime name -> threshold value.
    """
    thresholds = {}
    for regime in regimes.unique():
        regime_scores = scores[regimes == regime]
        if len(regime_scores) == 0:
            continue
        thresholds[regime] = float(np.percentile(regime_scores, percentile))
    return thresholds


def apply_per_regime_thresholds(
    scores: np.ndarray,
    regimes: pd.Series,
    thresholds: dict,
    fallback_threshold: float,
) -> np.ndarray:
    """
    Flag each sample as anomalous if its score exceeds the threshold
    for ITS regime.

    The fallback_threshold handles regimes that exist at inference time
    but weren't in training (defensive programming - shouldn't happen
    given our regime taxonomy, but better safe than crashing in
    production).
    """
    is_anomaly = np.zeros(len(scores), dtype=bool)
    regimes_arr = np.asarray(regimes)
    # OLD: regimes_arr = regimes.values if hasattr(regimes, "values") else np.asarray(regimes)

    for regime, thresh in thresholds.items():
        mask = regimes_arr == regime
        is_anomaly[mask] = scores[mask] > thresh

    unknown_regimes = set(np.unique(regimes_arr).tolist()) - set(thresholds.keys())
    # OLD: unknown_regimes = set(np.unique(regimes_arr)) - set(thresholds.keys())
    for regime in unknown_regimes:
        mask = regimes_arr == regime
        is_anomaly[mask] = scores[mask] > fallback_threshold

    return is_anomaly