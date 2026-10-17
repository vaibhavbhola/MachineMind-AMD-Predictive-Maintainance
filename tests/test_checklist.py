"""
Automated tests for Prescriptive Maintenance Recommendation Engine (checklist.py).
"""

import pytest
from datetime import datetime

from checklist import get_recommendation_for_telemetry, MaintenanceRecommendation
from model import AnomalyInferenceResult


def test_bearing_fault_pattern_recommendation():
    """High vibration with normal temperature should map to bearing pattern."""
    inf = AnomalyInferenceResult(
        timestamp=datetime.now(),
        is_anomaly=True,
        raw_anomaly_score=-0.2,
        normalized_anomaly_score=0.8,
        max_z_score=4.5,
        feature_z_scores={"temperature": 0.8, "vibration": 5.2, "pressure": 0.3, "speed": 0.1},
        feature_deviations={"temperature": 1.5, "vibration": 3.1, "pressure": 0.0, "speed": 0.0},
        feature_values={"temperature": 71.5, "vibration": 5.3, "pressure": 5.0, "speed": 1800.0},
        health_score=40.0,
        risk_category="Elevated",
        risk_score=0.7,
        breached_thresholds=["vibration:WARNING_HIGH"],
        model_backend="IsolationForest (CPU Scikit-Learn)"
    )
    rec: MaintenanceRecommendation = get_recommendation_for_telemetry(inf)
    assert rec.pattern_id == "PATTERN_VIBRATION_BEARING"
    assert "Bearing" in rec.pattern_name
    assert rec.severity == "WARNING"
    assert len(rec.inspection_steps) >= 3
    # Check for bearing inspection items
    titles = [s.title for s in rec.inspection_steps]
    assert any("Bearing" in t for t in titles)
    assert any("Alignment" in t for t in titles)


def test_coolant_leak_pattern_recommendation():
    """High temperature with low pressure should map to coolant leak pattern."""
    inf = AnomalyInferenceResult(
        timestamp=datetime.now(),
        is_anomaly=True,
        raw_anomaly_score=-0.3,
        normalized_anomaly_score=0.9,
        max_z_score=6.0,
        feature_z_scores={"temperature": 7.5, "vibration": 1.0, "pressure": 8.0, "speed": 2.0},
        feature_deviations={"temperature": 26.0, "vibration": 0.4, "pressure": -2.6, "speed": -40.0},
        feature_values={"temperature": 96.0, "vibration": 2.5, "pressure": 2.4, "speed": 1760.0},
        health_score=15.0,
        risk_category="Critical",
        risk_score=0.95,
        breached_thresholds=["temperature:CRITICAL_HIGH", "pressure:CRITICAL_LOW"],
        model_backend="IsolationForest (CPU Scikit-Learn)"
    )
    rec = get_recommendation_for_telemetry(inf)
    assert rec.pattern_id == "PATTERN_COOLANT_THERMAL"
    assert rec.severity == "CRITICAL"
    assert rec.priority == "P1 - CRITICAL"
    assert rec.escalation_notes is not None
    # Check safety and LOTO protocol
    assert "LOTO" in rec.safety_advisory or "Lockout/Tagout" in rec.safety_advisory


def test_motor_speed_drop_pattern():
    """Speed drop with elevated temperature should map to motor overload pattern."""
    inf = AnomalyInferenceResult(
        timestamp=datetime.now(),
        is_anomaly=True,
        raw_anomaly_score=-0.15,
        normalized_anomaly_score=0.7,
        max_z_score=4.0,
        feature_z_scores={"temperature": 4.0, "vibration": 1.0, "pressure": 0.5, "speed": 4.5},
        feature_deviations={"temperature": 16.0, "vibration": 0.2, "pressure": -0.1, "speed": -150.0},
        feature_values={"temperature": 86.0, "vibration": 2.3, "pressure": 4.9, "speed": 1650.0},
        health_score=35.0,
        risk_category="Elevated",
        risk_score=0.75,
        breached_thresholds=["speed:WARNING_SPEED", "temperature:WARNING_HIGH"],
        model_backend="IsolationForest (CPU Scikit-Learn)"
    )
    rec = get_recommendation_for_telemetry(inf)
    assert rec.pattern_id == "PATTERN_SPEED_THERMAL"
    assert "Overload" in rec.pattern_name or "Speed" in rec.pattern_name


def test_ambiguous_pattern_fallback():
    """Unclassified multi-sensor anomaly should fall back to general inspection with uncertainty notice."""
    inf = AnomalyInferenceResult(
        timestamp=datetime.now(),
        is_anomaly=True,
        raw_anomaly_score=-0.1,
        normalized_anomaly_score=0.5,
        max_z_score=2.2,
        feature_z_scores={"temperature": 2.1, "vibration": 2.1, "pressure": 2.1, "speed": 2.1},
        feature_deviations={"temperature": 4.0, "vibration": 0.5, "pressure": 0.4, "speed": 25.0},
        feature_values={"temperature": 74.0, "vibration": 2.65, "pressure": 5.4, "speed": 1825.0},
        health_score=70.0,
        risk_category="Moderate",
        risk_score=0.35,
        breached_thresholds=[],
        model_backend="IsolationForest (CPU Scikit-Learn)"
    )
    rec = get_recommendation_for_telemetry(inf)
    assert rec.pattern_id == "PATTERN_STATISTICAL_ANOMALY"
    assert "uncertain" in rec.escalation_notes.lower()
