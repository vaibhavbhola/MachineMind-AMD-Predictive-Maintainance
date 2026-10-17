"""
Configuration and constants for MachineMind: Predictive Maintenance Assistant.
Defines sensor operating ranges, demonstration scenario parameters, ML settings,
alert thresholds, and filesystem paths.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

# -----------------------------------------------------------------------------
# Filesystem Paths
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
EXPORTS_DIR = BASE_DIR / "exports"

for dir_path in [MODELS_DIR, DATA_DIR, EXPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Equipment Information
# -----------------------------------------------------------------------------
DEFAULT_MACHINE_ID = "PUMP-MOTOR-UNIT-04B"
EQUIPMENT_TYPE = "Centrifugal Pump & AC Induction Motor (45kW)"
FACILITY_LOCATION = "Industrial Plant Area 3 - Feedwater Subsystem"

# -----------------------------------------------------------------------------
# Sensor Definitions & Operating Limits
# Illustrative demonstration parameters, not universal industrial safety standards.
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class SensorConfig:
    name: str
    unit: str
    baseline_mean: float
    baseline_std: float
    normal_min: float
    normal_max: float
    warning_threshold: float
    critical_threshold: float
    description: str

SENSOR_CONFIGS: Dict[str, SensorConfig] = {
    "temperature": SensorConfig(
        name="Bearing & Stator Temperature",
        unit="°C",
        baseline_mean=70.0,
        baseline_std=2.2,
        normal_min=65.0,
        normal_max=75.0,
        warning_threshold=85.0,
        critical_threshold=95.0,
        description="Core casing and bearing temperature measurement"
    ),
    "vibration": SensorConfig(
        name="Overall Vibration",
        unit="mm/s RMS",
        baseline_mean=2.2,
        baseline_std=0.25,
        normal_min=1.5,
        normal_max=3.0,
        warning_threshold=5.0,
        critical_threshold=7.0,
        description="Overall radial RMS vibration measured on drive-end bearing"
    ),
    "pressure": SensorConfig(
        name="Lubrication & Hydraulic Pressure",
        unit="bar",
        baseline_mean=5.0,
        baseline_std=0.18,
        normal_min=4.5,
        normal_max=5.5,
        warning_threshold=3.8,  # Under-pressure warning
        critical_threshold=2.8,  # Severe pressure drop
        description="Lube-oil and cooling hydraulic loop pressure"
    ),
    "speed": SensorConfig(
        name="Rotational Shaft Speed",
        unit="RPM",
        baseline_mean=1800.0,
        baseline_std=12.0,
        normal_min=1750.0,
        normal_max=1850.0,
        warning_threshold=1680.0,
        critical_threshold=1580.0,
        description="Main motor shaft rotational speed"
    ),
}

FEATURE_COLUMNS = ["temperature", "vibration", "pressure", "speed"]

# -----------------------------------------------------------------------------
# Simulation Scenarios
# -----------------------------------------------------------------------------
SCENARIOS = {
    "scenario_a": {
        "id": "scenario_a",
        "name": "Scenario A: Normal Operation",
        "description": "Stable baseline operation within nominal tolerances. Minor natural fluctuations, no injected fault, no repeated false alarms.",
        "duration_seconds": 160,
    },
    "scenario_b": {
        "id": "scenario_b",
        "name": "Scenario B: Bearing Degradation",
        "description": "Progressive mechanical bearing wear. Gradual vibration increase with subtle thermal rise, providing ~45s lead time before critical vibration.",
        "duration_seconds": 160,
    },
    "scenario_c": {
        "id": "scenario_c",
        "name": "Scenario C: Coolant Leak / Thermal Runaway",
        "description": "Hydraulic seal breach and coolant loss. Rapid pressure reduction accompanied by sharp temperature spike and acute risk escalation.",
        "duration_seconds": 160,
    },
    "custom": {
        "id": "custom",
        "name": "Scenario Custom: Interactive Operator Tuning",
        "description": "User-controlled live telemetry workbench. Manually dial individual sensor parameters, inject industrial faults, and observe real-time AI detection and prescriptive fixes.",
        "duration_seconds": 160,
    },
}

# -----------------------------------------------------------------------------
# Anomaly Detection & Risk Engine Settings
# -----------------------------------------------------------------------------
ML_CONFIG: Dict[str, Any] = {
    "random_seed": 42,
    "training_samples": 400,
    "contamination": 0.05,
    "n_estimators": 100,
    "max_samples": "auto",
    "rolling_window_size": 15,
    "z_score_threshold": 2.5,
    "health_decay_rate": 0.15,
    "health_recovery_rate": 0.08,
}

# Risk categories and explicit assignment bounds
RISK_CATEGORIES = {
    "LOW": "Low",
    "MODERATE": "Moderate",
    "ELEVATED": "Elevated",
    "CRITICAL": "Critical"
}

# Hysteresis & persistence to prevent alert flickering
ALERT_PERSISTENCE_TICKS = 3  # Required consecutive anomalous ticks to raise alert
ALERT_RECOVERY_TICKS = 4     # Required consecutive normal ticks to resolve alert

# -----------------------------------------------------------------------------
# UI Styling & Theme Palette
# Industrial dark theme compliant with hackathon design constraints
# -----------------------------------------------------------------------------
THEME_COLORS = {
    "bg_dark": "#0d1117",
    "card_bg": "#161b22",
    "card_border": "#30363d",
    "text_main": "#c9d1d9",
    "text_muted": "#8b949e",
    "text_highlight": "#f0f6fc",
    "status_normal": "#2ea043",    # Green
    "status_moderate": "#e3b341",  # Amber / Yellow
    "status_elevated": "#db6d28",  # Orange
    "status_critical": "#f85149",  # Red
    "accent_blue": "#58a6ff",      # Industrial Blue
    "accent_purple": "#8957e5",
}
