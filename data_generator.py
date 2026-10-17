"""
Synthetic Industrial Telemetry Engine for MachineMind.
Generates realistic multi-sensor telemetry for industrial rotating machinery,
supporting 3 deterministic scenarios:
  1. Normal Operation (Scenario A)
  2. Bearing Degradation with ~45s lead time (Scenario B)
  3. Coolant Leak / Thermal Runaway (Scenario C)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional, List

from config import (
    SENSOR_CONFIGS,
    FEATURE_COLUMNS,
    SCENARIOS,
    ML_CONFIG,
    DEFAULT_MACHINE_ID
)


def generate_scenario_telemetry(
    scenario_id: str = "scenario_a",
    duration_seconds: int = 160,
    seed: int = 42,
    start_time: Optional[datetime] = None,
    machine_id: str = DEFAULT_MACHINE_ID,
) -> pd.DataFrame:
    """
    Generates a deterministic synthetic telemetry time series for a specified scenario.

    Parameters:
        scenario_id: 'scenario_a', 'scenario_b', or 'scenario_c'
        duration_seconds: Total duration of the simulation in seconds (1 tick = 1 second)
        seed: Random seed for reproducible physics & noise
        start_time: Starting timestamp (defaults to current time minus duration)
        machine_id: Machine identifier string

    Returns:
        pd.DataFrame containing simulated sensor readings and ground truth annotations
    """
    rng = np.random.RandomState(seed)
    
    if start_time is None:
        start_time = datetime(2026, 1, 1, 12, 0, 0)
        
    timestamps = [start_time + timedelta(seconds=i) for i in range(duration_seconds)]
    t = np.arange(duration_seconds, dtype=float)

    # 1. Base physics for all scenarios (nominal healthy operating envelope)
    # Slow operational oscillations + Gaussian sensor noise
    cycle_slow = np.sin(2 * np.pi * t / 75.0)
    cycle_fast = np.cos(2 * np.pi * t / 25.0)

    # Baseline sensor signals with physical noise
    temp_base = 70.0 + 0.45 * cycle_slow + rng.normal(0, 0.35, size=duration_seconds)
    vib_base = 2.15 + 0.08 * cycle_fast + rng.normal(0, 0.08, size=duration_seconds)
    press_base = 5.02 + 0.06 * cycle_slow + rng.normal(0, 0.07, size=duration_seconds)
    speed_base = 1800.0 + 3.5 * cycle_slow + rng.normal(0, 4.5, size=duration_seconds)

    operating_state = ["NORMAL"] * duration_seconds
    fault_state = ["NONE"] * duration_seconds

    # 2. Scenario specific modulations
    if scenario_id == "scenario_a":
        # Normal Operation: Stable, no injected anomalies
        temperature = temp_base
        vibration = vib_base
        pressure = press_base
        speed = speed_base

    elif scenario_id == "scenario_b":
        # Bearing Degradation:
        # Ticks 0-40: Healthy baseline
        # Tick 40 onwards: Progressive bearing spalling and friction wear
        # Calibrated so that vibration crosses warning threshold (5.0 mm/s) at tick ~85,
        # and critical threshold (7.0 mm/s) at tick ~130.
        # This yields an actual measured warning lead time of 130 - 85 = 45 seconds.
        temperature = temp_base.copy()
        vibration = vib_base.copy()
        pressure = press_base.copy()
        speed = speed_base.copy()

        fault_start = 40
        warning_tick_target = 85
        critical_tick_target = 130

        for i in range(duration_seconds):
            if i >= fault_start:
                fault_state[i] = "BEARING_DEGRADATION"
                elapsed = i - fault_start
                
                # Smooth progressive degradation curve
                # Vibration rises smoothly from 2.15 to ~5.0 at t=88, ~7.0 at t=133 (yielding exactly 45s lead time)
                if i < 88:
                    vib_offset = (5.02 - 2.15) * (((i - 40) / 48.0) ** 1.15)
                else:
                    vib_offset = (5.02 - 2.15) + (7.05 - 5.02) * (((i - 88) / 45.0) ** 1.20)
                    
                vibration[i] += vib_offset
                
                # Associated friction-induced thermal rise (mild, ~7-8°C increase, staying below 80°C)
                temperature[i] += 0.065 * elapsed
                
                # Slight rotational speed drag under extreme mechanical friction
                if i >= critical_tick_target:
                    speed[i] -= 15.0 * ((i - critical_tick_target) / 30.0)

            # Assign operating state based on thresholds
            if vibration[i] >= SENSOR_CONFIGS["vibration"].critical_threshold:
                operating_state[i] = "CRITICAL"
            elif vibration[i] >= SENSOR_CONFIGS["vibration"].warning_threshold:
                operating_state[i] = "WARNING"
            elif fault_state[i] != "NONE":
                operating_state[i] = "ELEVATED"
            else:
                operating_state[i] = "NORMAL"

    elif scenario_id == "scenario_c":
        # Coolant Leak / Thermal Runaway:
        # Ticks 0-40: Healthy baseline
        # Tick 40 onwards: Sudden coolant line pressure loss followed by rapid thermal escalation
        temperature = temp_base.copy()
        vibration = vib_base.copy()
        pressure = press_base.copy()
        speed = speed_base.copy()

        fault_start = 40
        for i in range(duration_seconds):
            if i >= fault_start:
                fault_state[i] = "COOLANT_LEAK"
                elapsed = i - fault_start
                
                # Pressure drops swiftly towards 2.2 bar
                press_drop = 2.8 * (1.0 - np.exp(-elapsed / 22.0))
                pressure[i] -= press_drop
                
                # Temperature climbs rapidly towards 98°C+
                temp_rise = 29.0 * (1.0 - np.exp(-elapsed / 38.0))
                temperature[i] += temp_rise
                
                # Thermal expansion and cavitation introduce modest vibration increase
                vibration[i] += 0.85 * (1.0 - np.exp(-elapsed / 45.0))
                
                # Stator winding resistance increases, reducing speed slightly
                speed[i] -= 55.0 * (1.0 - np.exp(-elapsed / 50.0))

            # Assign operating state
            if (temperature[i] >= SENSOR_CONFIGS["temperature"].critical_threshold or 
                pressure[i] <= SENSOR_CONFIGS["pressure"].critical_threshold):
                operating_state[i] = "CRITICAL"
            elif (temperature[i] >= SENSOR_CONFIGS["temperature"].warning_threshold or 
                  pressure[i] <= SENSOR_CONFIGS["pressure"].warning_threshold):
                operating_state[i] = "WARNING"
            elif fault_state[i] != "NONE":
                operating_state[i] = "ELEVATED"
            else:
                operating_state[i] = "NORMAL"
    elif scenario_id == "custom":
        temperature = temp_base.copy()
        vibration = vib_base.copy()
        pressure = press_base.copy()
        speed = speed_base.copy()
    else:
        raise ValueError(f"Unknown scenario_id: {scenario_id}. Choose 'scenario_a', 'scenario_b', 'scenario_c', or 'custom'.")

    # Assemble comprehensive telemetry dataframe
    df = pd.DataFrame({
        "timestamp": timestamps,
        "machine_id": machine_id,
        "scenario": scenario_id,
        "temperature": np.round(temperature, 2),
        "vibration": np.round(vibration, 3),
        "pressure": np.round(pressure, 3),
        "speed": np.round(speed, 1),
        "operating_state": operating_state,
        "fault_state": fault_state,  # Ground truth (evaluation only, not fed to unsupervised model)
    })

    return df


class TelemetryStreamer:
    """
    Manages stateful, bounded step-by-step telemetry replay for interactive dashboard use.
    Provides deterministic stepping, reset, and historical window buffering.
    """
    def __init__(
        self,
        scenario_id: str = "scenario_a",
        duration_seconds: int = 160,
        buffer_size: int = 120,
        seed: int = 42
    ):
        self.scenario_id = scenario_id
        self.duration_seconds = duration_seconds
        self.buffer_size = buffer_size
        self.seed = seed
        self.current_index = 0
        self.full_telemetry = generate_scenario_telemetry(
            scenario_id=self.scenario_id,
            duration_seconds=self.duration_seconds,
            seed=self.seed
        )

    def reset(self, scenario_id: Optional[str] = None):
        """Resets the streamer to tick 0, optionally switching scenario."""
        if scenario_id is not None and scenario_id != self.scenario_id:
            self.scenario_id = scenario_id
            self.full_telemetry = generate_scenario_telemetry(
                scenario_id=self.scenario_id,
                duration_seconds=self.duration_seconds,
                seed=self.seed
            )
        self.current_index = 0

    def step(self, num_steps: int = 1) -> Optional[pd.Series]:
        """Advances simulation by num_steps and returns the latest reading."""
        if self.current_index + num_steps >= len(self.full_telemetry):
            self.current_index = len(self.full_telemetry) - 1
            return self.full_telemetry.iloc[self.current_index]
        self.current_index += num_steps
        return self.full_telemetry.iloc[self.current_index]

    def get_latest_reading(self) -> pd.Series:
        """Returns the current reading at the current simulation pointer."""
        return self.full_telemetry.iloc[self.current_index]

    def get_visible_history(self, window_size: Optional[int] = None) -> pd.DataFrame:
        """Returns visible telemetry window up to the current simulation pointer."""
        window = window_size or self.buffer_size
        start_idx = max(0, self.current_index - window + 1)
        return self.full_telemetry.iloc[start_idx : self.current_index + 1].copy()

    def is_finished(self) -> bool:
        """Checks if simulation has reached the end of the scenario."""
        return self.current_index >= len(self.full_telemetry) - 1


def validate_and_normalize_csv(df: pd.DataFrame) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    """
    Validates uploaded historical CSV telemetry.
    Checks required sensor columns, parses timestamps, checks missing data,
    and applies reasonable column name normalization.

    Returns:
        (is_valid: bool, message: str, normalized_df: Optional[pd.DataFrame])
    """
    if df is None or df.empty:
        return False, "Uploaded CSV file is empty.", None

    # Normalize column names: lower-case, stripped, standard aliases
    column_mapping = {
        "temp": "temperature",
        "temperature_c": "temperature",
        "temp_c": "temperature",
        "vib": "vibration",
        "vibration_rms": "vibration",
        "vibe": "vibration",
        "press": "pressure",
        "pressure_bar": "pressure",
        "psi": "pressure",
        "rpm": "speed",
        "rotational_speed": "speed",
        "motor_speed": "speed",
        "time": "timestamp",
        "datetime": "timestamp",
        "date_time": "timestamp"
    }

    norm_cols = {}
    for col in df.columns:
        clean_col = str(col).strip().lower()
        norm_cols[col] = column_mapping.get(clean_col, clean_col)

    renamed_df = df.rename(columns=norm_cols)

    # Check for required sensor features
    missing_features = [feat for feat in FEATURE_COLUMNS if feat not in renamed_df.columns]
    if missing_features:
        return (
            False,
            f"Missing required sensor columns: {missing_features}. Accepted schema requires: {FEATURE_COLUMNS}",
            None
        )

    # Validate and parse timestamp column
    if "timestamp" not in renamed_df.columns:
        # If timestamp is missing, create a synthetic sequential timestamp series
        renamed_df["timestamp"] = [
            datetime.now() - timedelta(seconds=len(renamed_df) - i)
            for i in range(len(renamed_df))
        ]
    else:
        try:
            renamed_df["timestamp"] = pd.to_datetime(renamed_df["timestamp"])
        except Exception as e:
            return False, f"Failed to parse 'timestamp' column as datetime: {str(e)}", None

    # Check and clean numeric sensor values
    cleaned_df = renamed_df.copy()
    for col in FEATURE_COLUMNS:
        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")
        missing_count = cleaned_df[col].isna().sum()
        if missing_count > 0:
            if missing_count > len(cleaned_df) * 0.2:
                return (
                    False,
                    f"Column '{col}' has {missing_count} invalid or missing values (>20% of dataset). Upload rejected.",
                    None
                )
            # Impute small amounts of missing data using forward-fill then backward-fill
            cleaned_df[col] = cleaned_df[col].ffill().bfill()

    # Ensure auxiliary columns exist
    if "machine_id" not in cleaned_df.columns:
        cleaned_df["machine_id"] = "UPLOADED-DATASET"
    if "scenario" not in cleaned_df.columns:
        cleaned_df["scenario"] = "historical_upload"
    if "operating_state" not in cleaned_df.columns:
        cleaned_df["operating_state"] = "UNKNOWN"
    if "fault_state" not in cleaned_df.columns:
        cleaned_df["fault_state"] = "UNLABELED"

    return True, f"Successfully validated and normalized {len(cleaned_df)} telemetry records.", cleaned_df
