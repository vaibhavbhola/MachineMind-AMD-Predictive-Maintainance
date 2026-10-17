"""
Automated tests for Synthetic Industrial Telemetry Engine (data_generator.py).
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from io import StringIO

from data_generator import (
    generate_scenario_telemetry,
    TelemetryStreamer,
    validate_and_normalize_csv
)
from config import FEATURE_COLUMNS, SENSOR_CONFIGS


def test_telemetry_schema_and_types():
    """Verifies that generated telemetry contains all required columns with correct types."""
    df = generate_scenario_telemetry("scenario_a", duration_seconds=50, seed=42)
    assert len(df) == 50
    for col in FEATURE_COLUMNS:
        assert col in df.columns
        assert pd.api.types.is_numeric_dtype(df[col])
        assert not df[col].isna().any()

    assert "timestamp" in df.columns
    assert "operating_state" in df.columns
    assert "fault_state" in df.columns
    assert "machine_id" in df.columns


def test_telemetry_reproducibility():
    """Verifies deterministic output when replayed with identical seeds."""
    df1 = generate_scenario_telemetry("scenario_b", duration_seconds=80, seed=123)
    df2 = generate_scenario_telemetry("scenario_b", duration_seconds=80, seed=123)
    pd.testing.assert_frame_equal(df1, df2)

    # Different seeds should produce distinct noise profiles
    df3 = generate_scenario_telemetry("scenario_b", duration_seconds=80, seed=999)
    assert not df1["vibration"].equals(df3["vibration"])


def test_scenario_a_normal_envelope():
    """Verifies that Scenario A stays within nominal operating tolerances without extreme spikes."""
    df = generate_scenario_telemetry("scenario_a", duration_seconds=120, seed=42)
    
    # Temperature should remain between 65 and 76
    assert df["temperature"].min() >= 66.0
    assert df["temperature"].max() <= 75.5

    # Vibration should remain between 1.5 and 2.8
    assert df["vibration"].min() >= 1.6
    assert df["vibration"].max() <= 2.8

    # Pressure should remain between 4.7 and 5.4
    assert df["pressure"].min() >= 4.7
    assert df["pressure"].max() <= 5.4

    # All operating states must be NORMAL
    assert (df["operating_state"] == "NORMAL").all()
    assert (df["fault_state"] == "NONE").all()


def test_scenario_b_bearing_degradation_and_lead_time():
    """
    Verifies that Scenario B exhibits progressive vibration growth
    and delivers approximately 45s warning lead time.
    """
    df = generate_scenario_telemetry("scenario_b", duration_seconds=160, seed=42)
    
    # Vibration should increase significantly over time
    initial_vib = df.loc[0:20, "vibration"].mean()
    final_vib = df.loc[140:159, "vibration"].mean()
    assert final_vib > initial_vib + 4.0

    # Verify warning and critical threshold crossings
    warn_thresh = SENSOR_CONFIGS["vibration"].warning_threshold   # 5.0
    crit_thresh = SENSOR_CONFIGS["vibration"].critical_threshold  # 7.0

    warn_indices = np.where(df["vibration"] >= warn_thresh)[0]
    crit_indices = np.where(df["vibration"] >= crit_thresh)[0]

    assert len(warn_indices) > 0, "Scenario B should cross warning vibration threshold"
    assert len(crit_indices) > 0, "Scenario B should cross critical vibration threshold"

    w_tick = warn_indices[0]
    c_tick = crit_indices[0]
    lead_time = c_tick - w_tick

    # Lead time should be exactly 45 seconds under default settings
    assert 40 <= lead_time <= 48, f"Expected ~45s lead time, got {lead_time}s"
    assert lead_time == 45, f"Lead time should be exactly 45s, got {lead_time}s"


def test_scenario_c_thermal_runaway_dynamics():
    """
    Verifies that Scenario C demonstrates rapid pressure decrease
    accompanied by sharp thermal escalation.
    """
    df = generate_scenario_telemetry("scenario_c", duration_seconds=160, seed=42)

    # Initial baseline healthy
    assert df.loc[10, "temperature"] < 75.0
    assert df.loc[10, "pressure"] > 4.8

    # After fault onset at t=40, pressure drops and temperature surges
    final_temp = df.loc[140:159, "temperature"].mean()
    final_press = df.loc[140:159, "pressure"].mean()

    assert final_temp >= 95.0, f"Expected final temp >= 95°C, got {final_temp}"
    assert final_press <= 2.8, f"Expected final pressure <= 2.8 bar, got {final_press}"


def test_telemetry_streamer_stepping():
    """Verifies that TelemetryStreamer steps deterministically and maintains buffer bounds."""
    streamer = TelemetryStreamer("scenario_a", duration_seconds=60, buffer_size=15, seed=42)
    assert streamer.current_index == 0

    # Step 10 times
    for _ in range(10):
        streamer.step(1)
    assert streamer.current_index == 10

    hist = streamer.get_visible_history()
    assert len(hist) == 11

    # Step beyond buffer size
    for _ in range(25):
        streamer.step(1)
    assert streamer.current_index == 35
    hist2 = streamer.get_visible_history()
    assert len(hist2) == 15, "Visible history should be bounded by buffer_size"


def test_csv_validation_success_and_normalization():
    """Verifies that valid CSV telemetry with aliases is parsed and normalized properly."""
    raw_csv = """time,temp_c,vibe,psi,rpm
2026-10-17 12:00:00,71.2,2.3,5.1,1802
2026-10-17 12:00:01,71.5,2.4,5.0,1799
2026-10-17 12:00:02,72.0,2.5,4.9,1805
"""
    df = pd.read_csv(StringIO(raw_csv))
    is_valid, msg, norm_df = validate_and_normalize_csv(df)
    assert is_valid is True
    assert norm_df is not None
    assert len(norm_df) == 3
    assert "temperature" in norm_df.columns
    assert "vibration" in norm_df.columns
    assert "pressure" in norm_df.columns
    assert "speed" in norm_df.columns


def test_csv_validation_failure_on_missing_columns():
    """Verifies that missing critical sensor columns produces an informative error."""
    bad_csv = """timestamp,temperature,pressure
2026-10-17 12:00:00,70.5,5.1
2026-10-17 12:00:01,70.8,5.0
"""
    df = pd.read_csv(StringIO(bad_csv))
    is_valid, msg, norm_df = validate_and_normalize_csv(df)
    assert is_valid is False
    assert norm_df is None
    assert "Missing required sensor columns" in msg
