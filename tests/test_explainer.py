"""
Automated tests for Explainability and Root-Cause Attribution Engine (explainer.py).
"""

import pytest
import pandas as pd
from datetime import datetime

from explainer import explain_prediction, create_attribution_chart, SensorAttributionReport
from model import AnomalyInferenceResult
from config import SENSOR_CONFIGS


@pytest.fixture
def sample_inference_result():
    """Constructs a sample bearing degradation inference result."""
    return AnomalyInferenceResult(
        timestamp=datetime.now(),
        is_anomaly=True,
        raw_anomaly_score=-0.18,
        normalized_anomaly_score=0.72,
        max_z_score=4.2,
        feature_z_scores={"temperature": 1.2, "vibration": 4.2, "pressure": 0.4, "speed": 0.2},
        feature_deviations={"temperature": 2.6, "vibration": 1.05, "pressure": -0.07, "speed": -2.4},
        feature_values={"temperature": 72.6, "vibration": 5.25, "pressure": 4.95, "speed": 1797.6},
        health_score=45.0,
        risk_category="Elevated",
        risk_score=0.68,
        breached_thresholds=["vibration:WARNING_HIGH"],
        model_backend="IsolationForest (CPU Scikit-Learn)"
    )


def test_explain_prediction_structure(sample_inference_result):
    """Verifies attribution report contents and primary contributor identification."""
    report: SensorAttributionReport = explain_prediction(sample_inference_result)

    assert report.is_anomaly is True
    assert "Overall Vibration" in report.primary_contributor
    assert len(report.attribution_items) == 4

    # Verify sum of contributions is approximately 100%
    total_pct = sum(item.contribution_percentage for item in report.attribution_items)
    assert 99.0 <= total_pct <= 101.0

    # Primary contributor should have the highest contribution
    top_item = report.attribution_items[0]
    assert top_item.sensor_id == "vibration"
    assert top_item.contribution_percentage > 40.0


def test_plain_language_summary_and_disclaimer(sample_inference_result):
    """Verifies that plain-language diagnostics and uncertainty disclaimers are populated."""
    report = explain_prediction(sample_inference_result)
    assert "primary driver" in report.plain_language_summary
    assert "Overall Vibration" in report.plain_language_summary
    assert "empirical statistical deviations" in report.uncertainty_statement
    assert "Physical inspection is required" in report.uncertainty_statement


def test_create_attribution_chart(sample_inference_result):
    """Verifies that Plotly horizontal bar chart renders without error."""
    report = explain_prediction(sample_inference_result)
    fig = create_attribution_chart(report)
    assert fig is not None
    assert len(fig.data) == 1
    assert fig.data[0].orientation == "h"
    assert len(fig.data[0].y) == 4
