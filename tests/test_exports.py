"""
Automated tests for Work Order generation and Exports (JSON, Markdown, PDF).
"""

import pytest
import json
from datetime import datetime

from utils import (
    create_work_order,
    export_work_order_json,
    export_work_order_markdown,
    export_work_order_pdf,
    WorkOrderData
)
from model import AnomalyInferenceResult
from explainer import explain_prediction
from checklist import get_recommendation_for_telemetry


@pytest.fixture
def sample_work_order():
    """Generates a complete WorkOrderData object with realistic incident payload."""
    inf = AnomalyInferenceResult(
        timestamp=datetime.now(),
        is_anomaly=True,
        raw_anomaly_score=-0.25,
        normalized_anomaly_score=0.85,
        max_z_score=5.5,
        feature_z_scores={"temperature": 1.5, "vibration": 5.5, "pressure": 0.2, "speed": 0.5},
        feature_deviations={"temperature": 3.0, "vibration": 3.2, "pressure": -0.05, "speed": -5.0},
        feature_values={"temperature": 73.0, "vibration": 5.4, "pressure": 4.97, "speed": 1795.0},
        health_score=38.0,
        risk_category="Elevated",
        risk_score=0.72,
        breached_thresholds=["vibration:WARNING_HIGH (5.40 >= 5.0)"],
        model_backend="IsolationForest (CPU Scikit-Learn)"
    )
    attr = explain_prediction(inf)
    rec = get_recommendation_for_telemetry(inf, {"step_vib_1": True, "step_vib_2": False})
    
    return create_work_order(
        inference_result=inf,
        attribution_report=attr,
        recommendation=rec,
        scenario_id="scenario_b",
        operator_notes="Verified elevated acoustic noise on drive-end bearing casing."
    )


def test_json_export(sample_work_order):
    """Verifies that JSON export produces valid, complete JSON."""
    json_str = export_work_order_json(sample_work_order)
    parsed = json.loads(json_str)

    assert parsed["work_order_id"] == sample_work_order.work_order_id
    assert parsed["priority"] == sample_work_order.priority
    assert parsed["health_score"] == 38.0
    assert "vibration" in parsed["sensor_readings"]
    assert len(parsed["inspection_checklist"]) > 0
    assert "safety_advisory" in parsed
    assert "disclaimer" in parsed


def test_markdown_export(sample_work_order):
    """Verifies that Markdown export contains formatted headers, tables, and notes."""
    md_str = export_work_order_markdown(sample_work_order)

    assert f"# MAINTENANCE WORK ORDER: {sample_work_order.work_order_id}" in md_str
    assert "## Executive Summary" in md_str
    assert "## Sensor Telemetry at Trigger Event" in md_str
    assert "## Prescriptive Inspection Checklist" in md_str
    assert "Verified elevated acoustic noise" in md_str
    assert "DISCLAIMER" in md_str


def test_pdf_export(sample_work_order):
    """Verifies that PDF export generates valid PDF binary document with ReportLab."""
    pdf_bytes = export_work_order_pdf(sample_work_order)

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    # PDF files must start with %PDF header
    assert pdf_bytes.startswith(b"%PDF")
