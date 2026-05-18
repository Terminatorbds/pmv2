"""
Anomaly inspection utilities.

These functions help us look at flagged samples in their raw form -
not as scaled feature vectors, but as actual OBD readings a mechanic
would recognize. This is the validation step that confirms the model
is finding meaningful patterns rather than noise.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from src.preprocess import MODEL_FEATURES


def get_top_anomalies(
    scored_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    n: int = 50,
) -> pd.DataFrame:
    """
    Return the top n anomalous samples merged with their raw (un-scaled)
    OBD readings.

    The scored_df has scaled features (mean 0, std 1). For inspection
    we want the original engineering units. We get them by joining on
    index back to the cleaned dataset.
    """
    top = scored_df[scored_df["is_anomaly_v2"]].nlargest(n, "anomaly_score")
    raw_at_top = raw_df.loc[top.index]

    # Combine: raw values + anomaly metadata
    combined = raw_at_top.copy()
    combined["anomaly_score"] = top["anomaly_score"]
    combined["regime"] = top["regime"]
    return combined


def per_sample_contributions(
    model,
    X_scaled: pd.DataFrame,
    medians: pd.Series,
    feature_names: list,
) -> pd.DataFrame:
    """
    For each row in X_scaled, compute how much each feature contributes
    to its anomaly score by neutralization.

    This is a more efficient version of the function in anomaly_model.py:
    we precompute the median once and reuse it.

    Returns a DataFrame with same index as X_scaled, one column per
    feature.
    """
    base_scores = -model.score_samples(X_scaled)
    contributions = np.zeros((len(X_scaled), len(feature_names)))

    for i, feat in enumerate(feature_names):
        X_mod = X_scaled.copy()
        X_mod[feat] = medians[feat]
        new_scores = -model.score_samples(X_mod)
        contributions[:, i] = base_scores - new_scores

    return pd.DataFrame(contributions, columns=feature_names, index=X_scaled.index)


def categorize_anomaly(contribution_row: pd.Series) -> str:
    """
    Heuristic mapping from feature contribution pattern to a fault
    category label.

    Categories:
        FUELING       - fuel trim deviations dominate
        THROTTLE_PEDAL - throttle/pedal disagreement dominates
        CATALYST      - catalyst delta dominates
        THERMAL       - coolant or intake air temp dominates
        ELECTRICAL    - control module voltage dominates
        COMPLEX       - no single category dominates
        TRANSIENT     - all contributions small (likely a driving event)

    The thresholds are heuristic for now. Phase 6c will refine these
    into a proper classifier.
    """
    # Group features by fault category
    fueling_feats = [
        "FUEL_TRIM_TOTAL",
        "LONG_TERM_FUEL_TRIM_BANK_1",
        "SHORT_TERM_FUEL_TRIM_BANK_1",
    ]
    throttle_feats = [
        "THROTTLE_VS_ABS_DELTA",
        "PEDAL_D_VS_E_DELTA",
    ]
    catalyst_feats = ["CATALYST_DELTA"]
    thermal_feats = ["COOLANT_TEMPERATURE", "INTAKE_AIR_TEMP"]
    electrical_feats = ["CONTROL_MODULE_VOLTAGE"]

    # Sum positive contributions per category
    def cat_score(feats):
        return sum(max(0, contribution_row.get(f, 0)) for f in feats)

    scores = {
        "FUELING": cat_score(fueling_feats),
        "THROTTLE_PEDAL": cat_score(throttle_feats),
        "CATALYST": cat_score(catalyst_feats),
        "THERMAL": cat_score(thermal_feats),
        "ELECTRICAL": cat_score(electrical_feats),
    }

    total_positive = sum(scores.values())

    if total_positive < 0.001:
        return "TRANSIENT"

    top_cat = max(scores.keys(), key=lambda k: scores[k])
    # OLD: top_cat = max(scores, key=scores.get)
    top_score = scores[top_cat]

    # If the top category accounts for less than 50% of positive
    # contribution, the anomaly involves multiple subsystems
    if top_score / total_positive < 0.5:
        return "COMPLEX"

    return top_cat


def explain_anomaly(
    raw_row: pd.Series,
    contribution_row: pd.Series,
    top_k: int = 5,
) -> str:
    """
    Build a human-readable explanation of one anomalous sample.
    """
    lines = []
    lines.append(f"Score: {raw_row['anomaly_score']:.4f}  Regime: {raw_row['regime']}")

    # Top contributing features with their raw values
    top_contribs = contribution_row.nlargest(top_k)
    lines.append("Top contributors:")
    for feat, contrib in top_contribs.items():
        if feat in raw_row.index:
            raw_val = float(raw_row.loc[feat])  # type: ignore[index]
            # OLD: raw_val = float(raw_row[feat])
            lines.append(f"  {feat:38s} value={raw_val:>10.3f}  contrib={contrib:+.4f}")
        else:
            lines.append(f"  {feat:38s}                      contrib={contrib:+.4f}")

    lines.append(f"Category: {categorize_anomaly(contribution_row)}")
    return "\n".join(lines)