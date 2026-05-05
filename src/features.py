"""
Feature engineering for the carOBD dataset.

Each function takes a DataFrame and returns a NEW DataFrame with one or
more derived columns added. The functions are pure - they don't modify
their input. This makes them safe to compose in any order.

The engineered features encode known fault signatures that are more
diagnostic than the raw sensor values. For example, two pedal sensors
should always agree; their disagreement is a stronger fault signal
than either sensor alone.
"""
import numpy as np
import pandas as pd


# Threshold definitions for the derived regime label.
# These are based on observed ranges in our cleaned data and standard
# automotive engineering conventions.
REGIME_THRESHOLDS = {
    "rpm_idle_max":     1100,    # below this with no speed -> idle
    "speed_idle_max":   2,       # km/h, accounts for sensor noise at zero
    "speed_city_max":   60,      # km/h, urban driving threshold
    "load_decel_max":   15,      # %, deceleration has very low load
}


def add_derived_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute an operating regime label from real-time signals.

    Why this is preferable to the filename session_type:
        At inference time there's no filename - just live OBD readings.
        This function works from the same signals available in production.

    Regime categories:
        idle      : engine running, vehicle stopped
        decel     : moving but throttle closed (engine braking)
        city      : low-speed driving with throttle input
        highway   : sustained higher-speed driving
    """
    out = df.copy()

    rpm = out["ENGINE_RPM"]
    speed = out["VEHICLE_SPEED"]
    load = out["ENGINE_LOAD"]

    # Default to city - we'll overwrite the others
    regime = pd.Series("city", index=out.index)

    # Idle: low RPM, vehicle stopped
    is_idle = (
        (rpm <= REGIME_THRESHOLDS["rpm_idle_max"]) &
        (speed <= REGIME_THRESHOLDS["speed_idle_max"])
    )
    regime[is_idle] = "idle"

    # Deceleration: moving but engine load very low (foot off pedal)
    is_decel = (
        (speed > REGIME_THRESHOLDS["speed_idle_max"]) &
        (load <= REGIME_THRESHOLDS["load_decel_max"])
    )
    regime[is_decel] = "decel"

    # Highway: sustained higher speed
    is_highway = speed > REGIME_THRESHOLDS["speed_city_max"]
    regime[is_highway] = "highway"

    out["regime"] = regime
    return out


def add_fuel_trim_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combined fuel trim is more diagnostic than either trim alone.

    The ECU adjusts fuel using both short-term (immediate) and long-term
    (learned) corrections. Their sum is the total correction the engine
    is applying right now. Persistent values above +-10% indicate a
    fueling fault (vacuum leak, injector issue, MAF sensor drift).
    """
    out = df.copy()
    out["FUEL_TRIM_TOTAL"] = (
        out["LONG_TERM_FUEL_TRIM_BANK_1"] +
        out["SHORT_TERM_FUEL_TRIM_BANK_1"]
    )
    return out


def add_throttle_disagreement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cars have multiple throttle position sensors for safety. They should
    all agree within a few percent. Disagreement = sensor fault or
    drive-by-wire actuator problem.

    We compute three deltas:
        - THROTTLE vs ABSOLUTE_THROTTLE_B (primary cross-check)
        - PEDAL_D vs PEDAL_E (the dual pedal-position sensors)
        - THROTTLE vs PEDAL_D normalized (driver intent vs throttle response)

    Note: these sensors are reported in different scales. The raw delta
    isn't directly meaningful, but its variation across time IS. The model
    will learn the typical delta and flag deviations.
    """
    out = df.copy()
    out["THROTTLE_VS_ABS_DELTA"] = (
        out["THROTTLE"] - out["ABSOLUTE_THROTTLE_B"]
    )
    out["PEDAL_D_VS_E_DELTA"] = out["PEDAL_D"] - out["PEDAL_E"]
    return out


def add_catalyst_delta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Difference between upstream and downstream catalyst temperatures.

    A healthy catalyst converts pollutants exothermically, so downstream
    temp (S2) should be near or slightly above upstream (S1) at steady
    state. A degraded catalyst will show a smaller delta. This feature
    is a direct catalyst-efficiency proxy.
    """
    out = df.copy()
    out["CATALYST_DELTA"] = (
        out["CATALYST_TEMPERATURE_BANK1_SENSOR1"] -
        out["CATALYST_TEMPERATURE_BANK1_SENSOR2"]
    )
    return out


def add_load_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engine load per unit RPM. At a given RPM, a healthy engine produces
    a predictable amount of load. Higher-than-expected load indicates
    parasitic drag, compression loss, or accessory issues. We protect
    against division by zero by adding a small epsilon.
    """
    out = df.copy()
    out["LOAD_PER_RPM"] = out["ENGINE_LOAD"] / (out["ENGINE_RPM"] + 1)
    return out


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply every feature engineering step in sequence. This is the
    single function called by the preprocessing pipeline.
    """
    df = add_derived_regime(df)
    df = add_fuel_trim_features(df)
    df = add_throttle_disagreement(df)
    df = add_catalyst_delta(df)
    df = add_load_efficiency(df)
    return df