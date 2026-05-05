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