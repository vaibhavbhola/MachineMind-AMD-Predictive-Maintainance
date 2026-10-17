"""
Automated tests for Machine Learning and Risk Engine (model.py).
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from model import (
    PredictiveMaintenanceModel,
    AnomalyInferenceResult,
    evaluate_model_on_dataset
)
from data_generator import generate_scenario_telemetry
from config import RISK_CATEGORIES


@pytest.fixture(scope="module")
def trained_model():
    """Provides a trained baseline model instance."""
    m = PredictiveMaintenanceModel()
    m.fit_baseline()
    return m


def test_model_training_and_baseline_stats(trained_model):
    """Verifies that model fits and extracts valid baseline statistics."""
    assert trained_model.is_trained is True
    assert trained_model.scaler is not None
    assert trained_model.isolation_forest is not None

    for s in ["temperature", "vibration", "pressure", "speed"]:
        assert s in trained_model.baseline_stats
        stat = trained_model.baseline_stats[s]
        assert "mean" in stat
        assert "std" in stat
        assert stat["std"] > 0


def test_predict_nominal_reading(trained_model):
    """Verifies that normal readings yield high health and Low risk category."""
    nominal_reading = pd.Series({
        "timestamp": datetime.now(),
        "temperature": 70.1,
        "vibration": 2.18,
        "pressure": 5.02,
        "speed": 1801.0
    })
    res: AnomalyInferenceResult = trained_model.predict_reading(nominal_reading)
    assert 0.0 <= res.health_score <= 100.0
    assert 0.0 <= res.risk_score <= 1.0
    assert res.risk_category in (RISK_CATEGORIES["LOW"], RISK_CATEGORIES["MODERATE"])
    assert res.is_anomaly is False


def test_predict_anomalous_reading(trained_model):
    """Verifies that extreme sensor readings are correctly flagged as anomalies."""
    anomalous_reading = pd.Series({
        "timestamp": datetime.now(),
        "temperature": 98.5,  # Critical thermal breach
        "vibration": 7.4,   # Critical vibration breach
        "pressure": 2.3,    # Critical pressure drop
        "speed": 1720.0
    })
    res: AnomalyInferenceResult = trained_model.predict_reading(anomalous_reading)
    assert res.is_anomaly is True
    assert res.risk_category == RISK_CATEGORIES["CRITICAL"]
    assert res.max_z_score >= 3.0
    assert len(res.breached_thresholds) >= 2


def test_alert_persistence_and_hysteresis(trained_model):
    """
    Verifies that alert creation requires consecutive anomalous ticks
    and does not flicker on single noisy readings.
    """
    trained_model.reset_state()
    assert trained_model.active_alert is None

    normal_reading = pd.Series({"temperature": 70.0, "vibration": 2.15, "pressure": 5.0, "speed": 1800.0})
    anom_reading = pd.Series({"temperature": 89.0, "vibration": 5.5, "pressure": 4.8, "speed": 1800.0})

    # Tick 1: anomalous -> should not trigger alert yet (needs 3 ticks)
    trained_model.predict_reading(anom_reading)
    assert trained_model.active_alert is None

    # Tick 2: anomalous
    trained_model.predict_reading(anom_reading)
    assert trained_model.active_alert is None

    # Tick 3: anomalous -> should trigger active alert now
    trained_model.predict_reading(anom_reading)
    assert trained_model.active_alert is not None
    assert trained_model.active_alert.status == "ACTIVE"


def test_scenario_retrospective_evaluation(trained_model):
    """Verifies evaluation metrics computation on complete scenario datasets."""
    # Scenario A: Normal Operation
    df_a = generate_scenario_telemetry("scenario_a", duration_seconds=100, seed=42)
    metrics_a = evaluate_model_on_dataset(trained_model, df_a)
    assert metrics_a["false_alarm_rate"] <= 0.10
    assert metrics_a["final_health_score"] >= 90.0

    # Scenario B: Bearing Degradation
    df_b = generate_scenario_telemetry("scenario_b", duration_seconds=160, seed=42)
    metrics_b = evaluate_model_on_dataset(trained_model, df_b)
    assert metrics_b["anomaly_detection_rate"] >= 0.85
    assert metrics_b["warning_lead_time_seconds"] == 45
    assert metrics_b["final_health_score"] <= 20.0


def test_model_persistence(tmp_path):
    """Verifies saving and loading of model artifacts."""
    save_path = tmp_path / "test_model.joblib"
    m1 = PredictiveMaintenanceModel(model_path=save_path)
    m1.fit_baseline()
    m1.save()

    assert save_path.exists()

    m2 = PredictiveMaintenanceModel(model_path=save_path)
    loaded = m2.load()
    assert loaded is True
    assert m2.is_trained is True
    assert "vibration" in m2.baseline_stats
