"""
Utility functions for MachineMind: Predictive Maintenance Assistant.
Handles structured maintenance work-order generation, JSON/Markdown/PDF exports,
timestamp formatting, and logging.
"""

import json
import logging
from io import BytesIO
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional

from config import (
    EXPORTS_DIR,
    DEFAULT_MACHINE_ID,
    EQUIPMENT_TYPE,
    FACILITY_LOCATION,
    SENSOR_CONFIGS,
    THEME_COLORS
)
from model import AnomalyInferenceResult
from explainer import SensorAttributionReport
from checklist import MaintenanceRecommendation

# Configure standard logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("MachineMind")


@dataclass
class WorkOrderData:
    work_order_id: str
    machine_id: str
    equipment_type: str
    facility_location: str
    created_at: str
    scenario_source: str
    priority: str
    severity: str
    risk_category: str
    health_score: float
    triggering_sensors: List[str]
    sensor_readings: Dict[str, Dict[str, Any]]
    pattern_name: str
    primary_reason: str
    plain_language_attribution: str
    safe_order_of_operations: List[str]
    inspection_checklist: List[Dict[str, Any]]
    operator_notes: str
    model_backend: str
    safety_advisory: str
    disclaimer: str


def create_work_order(
    inference_result: AnomalyInferenceResult,
    attribution_report: SensorAttributionReport,
    recommendation: MaintenanceRecommendation,
    scenario_id: str,
    operator_notes: str = "",
    machine_id: str = DEFAULT_MACHINE_ID
) -> WorkOrderData:
    """Assembles a comprehensive work order object from active application state."""
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    wo_id = f"WO-{datetime.now().strftime('%Y%m%d')}-{int(datetime.now().timestamp()) % 10000:04d}"

    # Build sensor readings detail
    sensor_details: Dict[str, Dict[str, Any]] = {}
    for item in attribution_report.attribution_items:
        sensor_details[item.sensor_id] = {
            "name": item.sensor_name,
            "reading": item.current_value,
            "unit": item.unit,
            "baseline_mean": item.baseline_mean,
            "deviation": item.absolute_deviation,
            "z_score": item.relative_z_score,
            "contribution_pct": item.contribution_percentage,
            "status": item.status,
        }

    # Checklist details
    checklist = [
        {
            "step_id": step.step_id,
            "order": step.order,
            "title": step.title,
            "description": step.description,
            "is_critical": step.is_critical,
            "completed": step.completed,
        }
        for step in recommendation.inspection_steps
    ]

    # Triggering sensors
    triggering = [
        item.sensor_name for item in attribution_report.attribution_items
        if item.status in ("WARNING", "CRITICAL", "DEVIATED")
    ]

    return WorkOrderData(
        work_order_id=wo_id,
        machine_id=machine_id,
        equipment_type=EQUIPMENT_TYPE,
        facility_location=FACILITY_LOCATION,
        created_at=timestamp_str,
        scenario_source=scenario_id,
        priority=recommendation.priority,
        severity=recommendation.severity,
        risk_category=inference_result.risk_category,
        health_score=inference_result.health_score,
        triggering_sensors=triggering,
        sensor_readings=sensor_details,
        pattern_name=recommendation.pattern_name,
        primary_reason=recommendation.primary_reason,
        plain_language_attribution=attribution_report.plain_language_summary,
        safe_order_of_operations=recommendation.safe_order_of_operations,
        inspection_checklist=checklist,
        operator_notes=operator_notes or "No operator notes recorded.",
        model_backend=inference_result.model_backend,
        safety_advisory=recommendation.safety_advisory,
        disclaimer=recommendation.uncertainty_disclaimer
    )


def export_work_order_json(wo: WorkOrderData) -> str:
    """Serializes work order data to formatted JSON string."""
    return json.dumps(asdict(wo), indent=2)


def export_work_order_markdown(wo: WorkOrderData) -> str:
    """Formats work order data as structured industrial Markdown report."""
    md = []
    md.append(f"# MAINTENANCE WORK ORDER: {wo.work_order_id}")
    md.append(f"**Facility Location:** {wo.facility_location} | **Machine ID:** {wo.machine_id}")
    md.append(f"**Equipment:** {wo.equipment_type} | **Generated At:** {wo.created_at}")
    md.append("")
    md.append("---")
    md.append("## Executive Summary")
    md.append(f"- **Priority:** {wo.priority}")
    md.append(f"- **Severity:** {wo.severity}")
    md.append(f"- **Inspection Risk Category:** {wo.risk_category}")
    md.append(f"- **Machine Health Score:** {wo.health_score:.1f} / 100")
    md.append(f"- **Identified Fault Pattern:** {wo.pattern_name}")
    md.append(f"- **Inference Backend:** {wo.model_backend}")
    md.append("")
    md.append("### Diagnostic Reason")
    md.append(f"> {wo.primary_reason}")
    md.append("")
    md.append("### Root-Cause Attribution Summary")
    md.append(f"{wo.plain_language_attribution}")
    md.append("")
    md.append("---")
    md.append("## Sensor Telemetry at Trigger Event")
    md.append("| Sensor | Current Value | Baseline Mean | Deviation | Z-Score (σ) | Contribution | Status |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for s_id, d in wo.sensor_readings.items():
        md.append(
            f"| **{d['name']}** | {d['reading']} {d['unit']} | {d['baseline_mean']} {d['unit']} | "
            f"{d['deviation']:+.2f} | {d['z_score']}σ | {d['contribution_pct']:.1f}% | `{d['status']}` |"
        )
    md.append("")
    md.append("---")
    md.append("## Prescriptive Inspection Checklist")
    for step in wo.inspection_checklist:
        status_box = "[x]" if step["completed"] else "[ ]"
        crit_tag = " **[CRITICAL STEP]**" if step["is_critical"] else ""
        md.append(f"{status_box} **Step {step['order']}: {step['title']}**{crit_tag}")
        md.append(f"    *{step['description']}*")
        md.append("")
    md.append("### Safe Order of Operations")
    for op in wo.safe_order_of_operations:
        md.append(f"- {op}")
    md.append("")
    md.append("---")
    md.append("## Operator Notes")
    md.append(f"> {wo.operator_notes}")
    md.append("")
    md.append("---")
    md.append("### Safety Advisory")
    md.append(f"**WARNING:** {wo.safety_advisory}")
    md.append("")
    md.append("### Decision Support Disclaimer")
    md.append(f"*{wo.disclaimer}*")
    md.append("")
    return "\n".join(md)


def export_work_order_pdf(wo: WorkOrderData) -> bytes:
    """
    Generates a printable, professional PDF maintenance ticket using ReportLab.
    Returns raw PDF bytes suitable for direct Streamlit download.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors

        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#0d1117")
        )
        h2_style = ParagraphStyle(
            "DocH2",
            parent=styles["Heading2"],
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#1f2937"),
            spaceBefore=8,
            spaceAfter=4
        )
        body_style = ParagraphStyle(
            "DocBody",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#374151")
        )
        badge_style = ParagraphStyle(
            "DocBadge",
            parent=styles["Normal"],
            fontSize=9,
            leading=11,
            textColor=colors.white
        )
        disclaimer_style = ParagraphStyle(
            "DocDisclaimer",
            parent=styles["Italic"],
            fontSize=7.5,
            leading=10,
            textColor=colors.HexColor("#6b7280")
        )

        elements = []

        # Document Header
        elements.append(Paragraph(f"MachineMind | Maintenance Work Order: {wo.work_order_id}", title_style))
        elements.append(Paragraph(
            f"<b>Facility:</b> {wo.facility_location} &nbsp;|&nbsp; <b>Equipment:</b> {wo.equipment_type} ({wo.machine_id}) &nbsp;|&nbsp; <b>Issued:</b> {wo.created_at}",
            body_style
        ))
        elements.append(Spacer(1, 8))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0d1117"), spaceAfter=10))

        # KPI Summary Table
        kpi_data = [
            [
                Paragraph("<b>Priority:</b> " + wo.priority, body_style),
                Paragraph("<b>Severity:</b> " + wo.severity, body_style),
                Paragraph(f"<b>Health Score:</b> {wo.health_score:.1f}/100", body_style),
                Paragraph("<b>Inspection Risk:</b> " + wo.risk_category, body_style),
            ]
        ]
        kpi_table = Table(kpi_data, colWidths=[135, 135, 135, 135])
        kpi_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f3f4f6")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#d1d5db")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(kpi_table)
        elements.append(Spacer(1, 10))

        # Diagnostic Reason
        elements.append(Paragraph("<b>Primary Recommendation & Reason:</b>", h2_style))
        elements.append(Paragraph(f"<b>{wo.pattern_name}:</b> {wo.primary_reason}", body_style))
        elements.append(Spacer(1, 4))
        elements.append(Paragraph(f"<b>Attribution:</b> {wo.plain_language_attribution}", body_style))
        elements.append(Spacer(1, 10))

        # Sensor Table
        elements.append(Paragraph("<b>Sensor Telemetry Snapshot:</b>", h2_style))
        sensor_rows = [
            ["Sensor Name", "Reading", "Baseline", "Deviation", "Z-Score", "Contribution", "Condition"]
        ]
        for s_id, d in wo.sensor_readings.items():
            sensor_rows.append([
                d["name"],
                f"{d['reading']} {d['unit']}",
                f"{d['baseline_mean']} {d['unit']}",
                f"{d['deviation']:+.2f}",
                f"{d['z_score']} σ",
                f"{d['contribution_pct']:.1f}%",
                d["status"]
            ])

        sensor_table = Table(sensor_rows, colWidths=[130, 75, 75, 65, 65, 70, 60])
        sensor_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(sensor_table)
        elements.append(Spacer(1, 10))

        # Inspection Checklist
        elements.append(Paragraph("<b>Prescriptive Maintenance Checklist:</b>", h2_style))
        chk_rows = [["Done", "Step #", "Procedure & Action Items"]]
        for step in wo.inspection_checklist:
            box = "[ X ]" if step["completed"] else "[   ]"
            desc = f"<b>{step['title']}</b>" + (" [CRITICAL]" if step["is_critical"] else "") + f"<br/>{step['description']}"
            chk_rows.append([box, str(step["order"]), Paragraph(desc, body_style)])

        chk_table = Table(chk_rows, colWidths=[40, 45, 455])
        chk_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#374151")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
            ("ALIGN", (0, 0), (1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(chk_table)
        elements.append(Spacer(1, 8))

        # Operator Notes
        elements.append(Paragraph(f"<b>Operator Notes:</b> {wo.operator_notes}", body_style))
        elements.append(Spacer(1, 6))

        # Safety & Legal Disclaimers
        elements.append(Paragraph(f"<b>MANDATORY SAFETY:</b> {wo.safety_advisory}", disclaimer_style))
        elements.append(Paragraph(f"<b>NOTICE:</b> {wo.disclaimer} Backend: {wo.model_backend}", disclaimer_style))

        doc.build(elements)
        return buffer.getvalue()
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        raise
