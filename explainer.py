"""
Explainability and Root-Cause Attribution Engine for MachineMind.
Translates multidimensional model deviations into transparent, operator-readable
diagnostic attributions, relative feature contribution percentages, and interactive charts.
"""

import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from config import SENSOR_CONFIGS, THEME_COLORS, FEATURE_COLUMNS
from model import AnomalyInferenceResult


@dataclass
class SensorAttributionItem:
    sensor_id: str
    sensor_name: str
    unit: str
    current_value: float
    baseline_mean: float
    baseline_std: float
    expected_range_str: str
    absolute_deviation: float
    relative_z_score: float
    contribution_percentage: float
    status: str  # 'NORMAL', 'DEVIATED', 'WARNING', 'CRITICAL'
    direction: str  # 'HIGH', 'LOW', 'NOMINAL'


@dataclass
class SensorAttributionReport:
    timestamp_str: str
    is_anomaly: bool
    risk_category: str
    health_score: float
    primary_contributor: str
    secondary_contributor: Optional[str]
    plain_language_summary: str
    uncertainty_statement: str
    attribution_items: List[SensorAttributionItem]


def explain_prediction(
    inference_result: AnomalyInferenceResult,
    baseline_stats: Optional[Dict[str, Dict[str, float]]] = None
) -> SensorAttributionReport:
    """
    Computes normalized relative feature contributions and plain-language
    attributions for a given inference result.
    """
    if baseline_stats is None:
        baseline_stats = {}

    z_scores = inference_result.feature_z_scores
    deviations = inference_result.feature_deviations
    values = inference_result.feature_values

    # Compute sum of absolute Z-scores for relative contribution normalization
    total_z = sum(max(0.01, z) for z in z_scores.values())

    attribution_items: List[SensorAttributionItem] = []

    for sensor_id in FEATURE_COLUMNS:
        cfg = SENSOR_CONFIGS[sensor_id]
        cur_val = values.get(sensor_id, cfg.baseline_mean)
        abs_dev = deviations.get(sensor_id, 0.0)
        z = z_scores.get(sensor_id, 0.0)
        
        stat = baseline_stats.get(sensor_id, {})
        base_mean = stat.get("mean", cfg.baseline_mean)
        base_std = stat.get("std", cfg.baseline_std)
        expected_range = f"{base_mean - 2*base_std:.1f} to {base_mean + 2*base_std:.1f} {cfg.unit}"

        # Relative contribution percentage
        contribution_pct = (max(0.01, z) / total_z) * 100.0

        # Sensor status determination
        direction = "NOMINAL"
        if abs(abs_dev) > 0.5 * base_std:
            direction = "HIGH" if abs_dev > 0 else "LOW"

        # Check limits
        status = "NORMAL"
        if sensor_id == "pressure":
            if cur_val <= cfg.critical_threshold:
                status = "CRITICAL"
            elif cur_val <= cfg.warning_threshold:
                status = "WARNING"
            elif z >= 2.0:
                status = "DEVIATED"
        elif sensor_id == "speed":
            if cur_val <= cfg.critical_threshold:
                status = "CRITICAL"
            elif cur_val <= cfg.warning_threshold:
                status = "WARNING"
            elif z >= 2.0:
                status = "DEVIATED"
        else:
            if cur_val >= cfg.critical_threshold:
                status = "CRITICAL"
            elif cur_val >= cfg.warning_threshold:
                status = "WARNING"
            elif z >= 2.0:
                status = "DEVIATED"

        attribution_items.append(SensorAttributionItem(
            sensor_id=sensor_id,
            sensor_name=cfg.name,
            unit=cfg.unit,
            current_value=cur_val,
            baseline_mean=base_mean,
            baseline_std=base_std,
            expected_range_str=expected_range,
            absolute_deviation=round(abs_dev, 3),
            relative_z_score=round(z, 2),
            contribution_percentage=round(contribution_pct, 1),
            status=status,
            direction=direction
        ))

    # Sort sensors by contribution percentage descending
    attribution_items.sort(key=lambda x: x.contribution_percentage, reverse=True)

    primary = attribution_items[0].sensor_name
    secondary = attribution_items[1].sensor_name if len(attribution_items) > 1 and attribution_items[1].relative_z_score >= 1.5 else None

    # Construct Plain-Language Diagnostic Summary
    top_item = attribution_items[0]
    if not inference_result.is_anomaly and inference_result.max_z_score < 2.0:
        summary = (
            f"Machine telemetry is operating stably within normal design tolerances. "
            f"All sensor channels demonstrate nominal correlation with baseline profiles."
        )
    else:
        parts = [
            f"{top_item.sensor_name} is the primary driver of this deviation ({top_item.contribution_percentage:.1f}% contribution, "
            f"current {top_item.current_value:.2f} {top_item.unit} vs baseline {top_item.baseline_mean:.2f} {top_item.unit}, Z={top_item.relative_z_score:.1f})."
        ]
        if secondary:
            sec_item = attribution_items[1]
            parts.append(
                f"{sec_item.sensor_name} exhibits secondary divergence ({sec_item.contribution_percentage:.1f}% contribution, "
                f"current {sec_item.current_value:.2f} {sec_item.unit}, Z={sec_item.relative_z_score:.1f})."
            )
        summary = " ".join(parts)

    uncertainty_statement = (
        "Note: Attribution scores represent empirical statistical deviations relative to learned normal baselines, "
        "not deterministic physical causality. Sensor cross-coupling or mechanical resonance may influence readings. "
        "Physical inspection is required before performing maintenance."
    )

    return SensorAttributionReport(
        timestamp_str=inference_result.timestamp.strftime("%Y-%m-%d %H:%M:%S") if isinstance(inference_result.timestamp, (datetime,)) else str(inference_result.timestamp),
        is_anomaly=inference_result.is_anomaly,
        risk_category=inference_result.risk_category,
        health_score=inference_result.health_score,
        primary_contributor=primary,
        secondary_contributor=secondary,
        plain_language_summary=summary,
        uncertainty_statement=uncertainty_statement,
        attribution_items=attribution_items
    )


def create_attribution_chart(report: SensorAttributionReport) -> go.Figure:
    """
    Renders an industrial horizontal bar chart showing sensor deviation contributions.
    Adheres strictly to the hackathon dark theme and high contrast styling.
    """
    items = list(reversed(report.attribution_items))  # Reverse so highest is at top of vertical axis
    names = [item.sensor_name for item in items]
    percentages = [item.contribution_percentage for item in items]
    z_scores = [item.relative_z_score for item in items]
    units = [item.unit for item in items]
    cur_vals = [item.current_value for item in items]

    # Bar colors reflecting severity
    bar_colors = []
    for item in items:
        if item.status == "CRITICAL":
            bar_colors.append(THEME_COLORS["status_critical"])
        elif item.status == "WARNING":
            bar_colors.append(THEME_COLORS["status_elevated"])
        elif item.status == "DEVIATED":
            bar_colors.append(THEME_COLORS["status_moderate"])
        else:
            bar_colors.append(THEME_COLORS["status_normal"])

    # Hover text
    hover_texts = [
        f"<b>{item.sensor_name}</b><br>"
        f"Contribution: {item.contribution_percentage:.1f}%<br>"
        f"Reading: {item.current_value:.2f} {item.unit}<br>"
        f"Baseline: {item.baseline_mean:.2f} {item.unit}<br>"
        f"Deviation Z-Score: {item.relative_z_score:.2f} σ<br>"
        f"Condition: {item.status}"
        for item in items
    ]

    fig = go.Figure(go.Bar(
        x=percentages,
        y=names,
        orientation="h",
        marker=dict(
            color=bar_colors,
            line=dict(color=THEME_COLORS["card_border"], width=1)
        ),
        text=[f"{pct:.1f}% (Z={z:.1f})" for pct, z in zip(percentages, z_scores)],
        textposition="inside",
        textfont=dict(color="#ffffff", size=12, family="sans-serif"),
        hoverinfo="text",
        hovertext=hover_texts
    ))

    fig.update_layout(
        title=dict(
            text="Root-Cause Attribution: Relative Sensor Deviation Contribution",
            font=dict(color=THEME_COLORS["text_highlight"], size=14),
            x=0.0,
            y=0.98
        ),
        dragmode=False,
        xaxis=dict(
            title=dict(text="Normalized Contribution (%)", font=dict(color=THEME_COLORS["text_muted"], size=12)),
            range=[0, max(max(percentages) * 1.15, 40.0)],
            gridcolor="#21262d",
            zerolinecolor="#30363d",
            tickfont=dict(color=THEME_COLORS["text_muted"]),
            fixedrange=True
        ),
        yaxis=dict(
            tickfont=dict(color=THEME_COLORS["text_main"], size=12),
            gridcolor="#21262d",
            fixedrange=True
        ),
        paper_bgcolor=THEME_COLORS["card_bg"],
        plot_bgcolor=THEME_COLORS["card_bg"],
        margin=dict(l=10, r=20, t=35, b=35),
        height=240,
    )

    return fig
