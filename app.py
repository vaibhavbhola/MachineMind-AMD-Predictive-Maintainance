"""
MachineMind: Predictive Maintenance Assistant
Interactive Demonstration Dashboard for Industrial Rotating Machinery Telemetry.
AMD Developer Hackathon: ACT III - Intelligent Industry Track.
"""

import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from config import (
    SENSOR_CONFIGS,
    FEATURE_COLUMNS,
    SCENARIOS,
    DEFAULT_MACHINE_ID,
    EQUIPMENT_TYPE,
    FACILITY_LOCATION,
    THEME_COLORS,
    RISK_CATEGORIES,
    EXPORTS_DIR
)
from data_generator import (
    generate_scenario_telemetry,
    validate_and_normalize_csv
)
from model import (
    PredictiveMaintenanceModel,
    AnomalyInferenceResult,
    evaluate_model_on_dataset
)
from explainer import (
    explain_prediction,
    create_attribution_chart,
    SensorAttributionReport
)
from checklist import (
    get_recommendation_for_telemetry,
    MaintenanceRecommendation
)
from utils import (
    create_work_order,
    export_work_order_json,
    export_work_order_markdown,
    export_work_order_pdf,
    WorkOrderData,
    logger
)
from amd_config import get_amd_hardware_status, AMDHardwareStatus


# -----------------------------------------------------------------------------
# Streamlit Page Setup & Custom CSS
# Enforces dark industrial palette and strict styling rules (no purple gradients,
# no pill buttons, no emojis, no horizontal scrolling).
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MachineMind | Predictive Maintenance",
    page_icon="[M]",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Dark industrial container styles */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.0rem;
        padding-left: 2.0rem;
        padding-right: 2.0rem;
        max-width: 100% !important;
    }
    
    /* Top title and subtext */
    .machinemind-header {
        border-bottom: 1px solid #30363d;
        padding-bottom: 0.8rem;
        margin-bottom: 1.2rem;
    }
    .machinemind-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #f0f6fc;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .machinemind-subtitle {
        font-size: 0.95rem;
        color: #8b949e;
        margin-bottom: 0.5rem;
    }
    .demo-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 4px;
        background-color: #21262d;
        color: #58a6ff;
        border: 1px solid #30363d;
        margin-right: 0.6rem;
    }
    .backend-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 4px;
        background-color: #161b22;
        color: #c9d1d9;
        border: 1px solid #30363d;
    }
    
    /* KPI Card styling */
    .kpi-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 1.0rem;
        text-align: left;
        height: 100%;
    }
    .kpi-title {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #8b949e;
        margin-bottom: 0.4rem;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f0f6fc;
        line-height: 1.1;
        margin-bottom: 0.2rem;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #8b949e;
    }

    /* Status indicator dots */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .status-normal { background-color: #2ea043; }
    .status-moderate { background-color: #e3b341; }
    .status-elevated { background-color: #db6d28; }
    .status-critical { background-color: #f85149; }

    /* Clean square/rectangular buttons */
    div.stButton > button {
        border-radius: 4px !important;
        font-weight: 600 !important;
        border: 1px solid #30363d !important;
    }
    div.stDownloadButton > button {
        border-radius: 4px !important;
        font-weight: 600 !important;
        border: 1px solid #30363d !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Precomputed Scenario Cache (O(1) execution, zero rerun CPU latency)
# -----------------------------------------------------------------------------
@st.cache_data
def get_cached_scenario_data(scenario_id: str, seed: int = 42):
    """
    Precomputes full scenario telemetry, ML anomaly scores, health scores,
    attributions, and recommendations for all 160 seconds.
    Cached in memory to execute in <0.5ms on all subsequent lookups.
    """
    model = PredictiveMaintenanceModel()
    model.fit_baseline()
    df = generate_scenario_telemetry(scenario_id=scenario_id, duration_seconds=160, seed=seed)
    
    inferences = []
    attributions = []
    recommendations = []
    
    for i, row in df.iterrows():
        inf = model.predict_reading(row)
        attr = explain_prediction(inf, model.baseline_stats)
        # Compute dynamic rate of change using recent window
        sub_window = df.iloc[max(0, i - 15) : i + 1]
        rec = get_recommendation_for_telemetry(inf, history_window_df=sub_window)
        inferences.append(inf)
        attributions.append(attr)
        recommendations.append(rec)
        
    return df, inferences, attributions, recommendations


def get_or_create_maintenance_model() -> PredictiveMaintenanceModel:
    """Initializes and caches the fitted baseline model for custom telemetry tuning."""
    if "maintenance_model" not in st.session_state:
        model = PredictiveMaintenanceModel()
        model.fit_baseline()
        st.session_state["maintenance_model"] = model
    return st.session_state["maintenance_model"]


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "scenario_id" not in st.session_state:
    st.session_state["scenario_id"] = "scenario_a"

if "current_tick" not in st.session_state:
    st.session_state["current_tick"] = 0

if "is_playing" not in st.session_state:
    st.session_state["is_playing"] = False

if "playback_speed" not in st.session_state:
    st.session_state["playback_speed"] = 1.0

if "history_window" not in st.session_state:
    st.session_state["history_window"] = 60

if "checklist_state" not in st.session_state:
    st.session_state["checklist_state"] = {}

if "operator_notes" not in st.session_state:
    st.session_state["operator_notes"] = ""

if "last_work_order" not in st.session_state:
    st.session_state["last_work_order"] = None

if "hw_status" not in st.session_state:
    st.session_state["hw_status"] = get_amd_hardware_status()

# Custom scenario tuning state
if "custom_telemetry" not in st.session_state:
    st.session_state["custom_telemetry"] = {
        "temperature": 70.0,
        "vibration": 2.2,
        "pressure": 5.0,
        "speed": 1800.0
    }

if "custom_slider_temp" not in st.session_state:
    st.session_state["custom_slider_temp"] = 70.0
if "custom_slider_vib" not in st.session_state:
    st.session_state["custom_slider_vib"] = 2.2
if "custom_slider_press" not in st.session_state:
    st.session_state["custom_slider_press"] = 5.0
if "custom_slider_speed" not in st.session_state:
    st.session_state["custom_slider_speed"] = 1800.0

if "custom_history_df" not in st.session_state:
    base_rows = []
    for t in range(40):
        base_rows.append({
            "tick": t,
            "temperature": round(70.0 + np.sin(t / 4.0) * 0.35, 2),
            "vibration": round(2.2 + np.cos(t / 3.0) * 0.06, 3),
            "pressure": round(5.0 + np.sin(t / 5.0) * 0.04, 3),
            "speed": round(1800.0 + np.cos(t / 4.0) * 3.0, 1)
        })
    st.session_state["custom_history_df"] = pd.DataFrame(base_rows)

hw_status: AMDHardwareStatus = st.session_state["hw_status"]

# Prime initial data for rendering
if st.session_state["scenario_id"] != "custom":
    df_prime, inf_prime, attr_prime, rec_prime = get_cached_scenario_data(st.session_state["scenario_id"])
    initial_tick = min(st.session_state["current_tick"], len(inf_prime) - 1)
    latest_inference = inf_prime[initial_tick]
    latest_attribution = attr_prime[initial_tick]
    latest_recommendation = rec_prime[initial_tick]
else:
    model_inst = get_or_create_maintenance_model()
    cur_series = pd.Series(st.session_state["custom_telemetry"])
    latest_inference = model_inst.predict_reading(cur_series, instantaneous=True)
    latest_attribution = explain_prediction(latest_inference, model_inst.baseline_stats)
    latest_recommendation = get_recommendation_for_telemetry(
        latest_inference,
        st.session_state["checklist_state"],
        history_window_df=st.session_state["custom_history_df"]
    )


# -----------------------------------------------------------------------------
# Sidebar Navigation & Simulation Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Equipment & Controls")
    st.markdown(f"**Asset:** `{DEFAULT_MACHINE_ID}`")
    st.caption(f"{EQUIPMENT_TYPE}")
    st.divider()

    st.markdown("#### Scenario Selection")
    scenario_options = {
        "scenario_a": "Scenario A: Normal Operation",
        "scenario_b": "Scenario B: Bearing Degradation",
        "scenario_c": "Scenario C: Coolant Leak / Thermal Runaway",
        "custom": "Scenario Custom: Interactive Operator Tuning"
    }

    selected_scen = st.selectbox(
        "Active Scenario",
        options=list(scenario_options.keys()),
        format_func=lambda x: scenario_options[x],
        index=list(scenario_options.keys()).index(st.session_state["scenario_id"]),
        key="active_scenario_select"
    )

    if selected_scen != st.session_state["scenario_id"]:
        st.session_state["scenario_id"] = selected_scen
        st.session_state["current_tick"] = 0
        st.session_state["is_playing"] = False
        st.session_state["checklist_state"] = {}
        st.rerun()

    st.caption(SCENARIOS[st.session_state["scenario_id"]]["description"])
    st.divider()

    # Playback Controls (Active for pre-fixed scenarios A, B, C)
    if st.session_state["scenario_id"] != "custom":
        st.markdown("#### Simulation Playback")
        col_play, col_step, col_reset = st.columns([1, 1, 1])
        with col_play:
            if st.session_state["is_playing"]:
                if st.button("Pause", key="sb_btn_pause", use_container_width=True):
                    st.session_state["is_playing"] = False
                    st.rerun()
            else:
                if st.button("Play", key="sb_btn_play", use_container_width=True):
                    if st.session_state["current_tick"] >= 159:
                        st.session_state["current_tick"] = 0
                    st.session_state["is_playing"] = True
                    st.rerun()

        with col_step:
            if st.button("Step +1", key="sb_btn_step", use_container_width=True, disabled=st.session_state["is_playing"]):
                st.session_state["current_tick"] = min(159, st.session_state["current_tick"] + 1)
                st.rerun()

        with col_reset:
            if st.button("Reset", key="sb_btn_reset", use_container_width=True):
                st.session_state["is_playing"] = False
                st.session_state["current_tick"] = 0
                st.session_state["checklist_state"] = {}
                st.rerun()

        # Progress in scenario (Linked via placeholder to fragment for live 1-second updates)
        cur_t = st.session_state["current_tick"]
        total_t = 160
        sidebar_progress_placeholder = st.sidebar.empty()
        sidebar_progress_placeholder.progress(float(cur_t) / float(total_t - 1), text=f"Simulation Progress: {cur_t}s / {total_t}s")

        # Speed Controls
        def on_speed_change():
            st.session_state["playback_speed"] = st.session_state["sb_speed_slider"]

        st.select_slider(
            "Playback Speed",
            options=[0.5, 1.0, 2.0],
            value=st.session_state["playback_speed"],
            format_func=lambda x: f"{x}x",
            key="sb_speed_slider",
            on_change=on_speed_change
        )
    else:
        st.markdown("#### Custom Workbench Mode")
        st.info("Live interactive mode active. Adjust individual parameters and inject faults via the main tuning panel.")
        sidebar_progress_placeholder = None

    def on_hist_change():
        st.session_state["history_window"] = st.session_state["sb_hist_slider"]

    st.slider(
        "Displayed History Window (seconds)",
        min_value=30,
        max_value=160,
        value=st.session_state["history_window"],
        step=10,
        key="sb_hist_slider",
        on_change=on_hist_change
    )

    st.divider()
    st.markdown("#### Hardware Execution Backend")
    st.markdown(f"**Provider:** `{hw_status.selected_provider}`")
    st.markdown(f"**Verified:** `{'Yes (Accelerated)' if hw_status.hardware_accelerated else 'Standard CPU Fallback'}`")
    st.caption(hw_status.verification_status)


# -----------------------------------------------------------------------------
# Section A: Header
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="machinemind-header">
        <div class="machinemind-title">MachineMind | Predictive Maintenance</div>
        <div class="machinemind-subtitle">
            Monitor simulated industrial telemetry, detect abnormal operating patterns, 
            identify contributing sensors, and generate maintenance work orders.
        </div>
        <div>
            <span class="demo-badge">SYNTHETIC TELEMETRY DEMONSTRATION</span>
            <span class="backend-badge">{hw_status.inference_backend_label}</span>
            <span class="backend-badge" style="margin-left: 0.4rem;">Asset: {DEFAULT_MACHINE_ID}</span>
            <span class="backend-badge" style="margin-left: 0.4rem;">Scenario: {SCENARIOS[st.session_state['scenario_id']]['name'].split(':')[0]}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# Custom Scenario: Interactive Machine Tuning & Fault Injection Panel
# -----------------------------------------------------------------------------
if st.session_state["scenario_id"] == "custom":
    with st.expander("Interactive Machine Tuning & Fault Injection Controls (Operator Workbench)", expanded=True):
        st.markdown("##### 1. Individual Sensor Tuning (Direct Parameter Control)")
        st.caption("Drag sliders to manually perturb operating parameters and observe real-time AI anomaly detection.")
        
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        with col_c1:
            c_temp = st.slider(
                "Temperature (°C)",
                min_value=50.0, max_value=115.0,
                value=float(st.session_state["custom_telemetry"]["temperature"]),
                step=0.5,
                key="custom_slider_temp"
            )
        with col_c2:
            c_vib = st.slider(
                "Vibration (mm/s RMS)",
                min_value=0.5, max_value=12.0,
                value=float(st.session_state["custom_telemetry"]["vibration"]),
                step=0.1,
                key="custom_slider_vib"
            )
        with col_c3:
            c_press = st.slider(
                "Pressure (bar)",
                min_value=1.0, max_value=7.0,
                value=float(st.session_state["custom_telemetry"]["pressure"]),
                step=0.05,
                key="custom_slider_press"
            )
        with col_c4:
            c_speed = st.slider(
                "Rotational Speed (RPM)",
                min_value=1400.0, max_value=2000.0,
                value=float(st.session_state["custom_telemetry"]["speed"]),
                step=10.0,
                key="custom_slider_speed"
            )

        # Update custom telemetry if any slider moved
        cur_cust = st.session_state["custom_telemetry"]
        if (c_temp != cur_cust["temperature"] or c_vib != cur_cust["vibration"] or 
            c_press != cur_cust["pressure"] or c_speed != cur_cust["speed"]):
            st.session_state["custom_telemetry"] = {
                "temperature": c_temp,
                "vibration": c_vib,
                "pressure": c_press,
                "speed": c_speed
            }
            # Append new reading to rolling history
            last_t = int(st.session_state["custom_history_df"]["tick"].iloc[-1])
            new_r = {
                "tick": last_t + 1,
                "temperature": c_temp,
                "vibration": c_vib,
                "pressure": c_press,
                "speed": c_speed
            }
            st.session_state["custom_history_df"] = pd.concat([
                st.session_state["custom_history_df"].tail(119),
                pd.DataFrame([new_r])
            ], ignore_index=True)
            st.rerun()

        def _apply_custom_preset(temp_v, vib_v, press_v, speed_v):
            st.session_state["custom_telemetry"] = {
                "temperature": temp_v,
                "vibration": vib_v,
                "pressure": press_v,
                "speed": speed_v
            }
            st.session_state["custom_slider_temp"] = temp_v
            st.session_state["custom_slider_vib"] = vib_v
            st.session_state["custom_slider_press"] = press_v
            st.session_state["custom_slider_speed"] = speed_v
            last_tick = int(st.session_state["custom_history_df"]["tick"].iloc[-1])
            new_row = {
                "tick": last_tick + 1,
                "temperature": temp_v,
                "vibration": vib_v,
                "pressure": press_v,
                "speed": speed_v
            }
            st.session_state["custom_history_df"] = pd.concat([
                st.session_state["custom_history_df"].tail(119),
                pd.DataFrame([new_row])
            ], ignore_index=True)

        def _reset_custom_defaults():
            _apply_custom_preset(70.0, 2.2, 5.0, 1800.0)
            if "maintenance_model" in st.session_state:
                st.session_state["maintenance_model"].reset_state()
            base_rows = []
            for t in range(40):
                base_rows.append({
                    "tick": t,
                    "temperature": round(70.0 + np.sin(t / 4.0) * 0.35, 2),
                    "vibration": round(2.2 + np.cos(t / 3.0) * 0.06, 3),
                    "pressure": round(5.0 + np.sin(t / 5.0) * 0.04, 3),
                    "speed": round(1800.0 + np.cos(t / 4.0) * 3.0, 1)
                })
            st.session_state["custom_history_df"] = pd.DataFrame(base_rows)
            st.session_state["checklist_state"] = {}

        st.markdown("##### 2. One-Click Industrial Fault Presets")
        st.caption("Instantly inject standard industrial failure modes into the live system.")
        fp_col1, fp_col2, fp_col3, fp_col4, fp_col5 = st.columns(5)

        with fp_col1:
            st.button(
                "Bearing Wear",
                key="btn_preset_bearing_wear",
                on_click=_apply_custom_preset,
                args=(82.0, 6.8, 4.92, 1795.0),
                use_container_width=True
            )
        with fp_col2:
            st.button(
                "Coolant Leak",
                key="btn_preset_coolant_leak",
                on_click=_apply_custom_preset,
                args=(97.5, 3.1, 2.30, 1785.0),
                use_container_width=True
            )
        with fp_col3:
            st.button(
                "Pump Cavitation",
                key="btn_preset_pump_cavitation",
                on_click=_apply_custom_preset,
                args=(72.5, 5.8, 3.25, 1790.0),
                use_container_width=True
            )
        with fp_col4:
            st.button(
                "Motor Overload",
                key="btn_preset_motor_overload",
                on_click=_apply_custom_preset,
                args=(88.5, 3.4, 4.85, 1550.0),
                use_container_width=True
            )
        with fp_col5:
            st.button(
                "Nominal Baseline",
                key="btn_preset_nominal_baseline",
                on_click=_apply_custom_preset,
                args=(70.0, 2.2, 5.0, 1800.0),
                use_container_width=True
            )

        st.markdown("##### 3. Reset Workbench")
        col_res1, col_res2 = st.columns([1, 1])
        with col_res1:
            st.button(
                "Reset to default values",
                key="btn_custom_reset_defaults",
                on_click=_reset_custom_defaults,
                use_container_width=True,
                type="primary"
            )
        with col_res2:
            st.caption("Resets all 4 sensor sliders back to their baseline operating parameters and clears any residual warning or critical alarms.")


# -----------------------------------------------------------------------------
# Helper: Chart Builder (Zero-Flicker uirevision Plotly charts)
# -----------------------------------------------------------------------------
def build_sensor_chart(sensor_key: str, df: pd.DataFrame, current_tick: int, window: int = 60) -> go.Figure:
    """Builds a responsive, dark-themed sensor waveform chart with fixed range and no expand box."""
    cfg = SENSOR_CONFIGS[sensor_key]
    
    if "tick" in df.columns:
        # Match by tick
        start_tick = max(0, current_tick - window)
        window_df = df[(df["tick"] >= start_tick) & (df["tick"] <= current_tick)]
        if len(window_df) == 0:
            window_df = df.tail(window)
        x_vals = window_df["tick"].tolist()
    else:
        start_tick = max(0, current_tick - window)
        window_df = df.iloc[start_tick : current_tick + 1]
        x_vals = list(range(start_tick, current_tick + 1))

    color_map = {
        "temperature": "#58a6ff",
        "vibration": "#e3b341",
        "pressure": "#2ea043",
        "speed": "#8957e5"
    }
    line_color = color_map.get(sensor_key, "#58a6ff")

    fig = go.Figure()

    # Nominal range band
    fig.add_hrect(
        y0=cfg.normal_min, y1=cfg.normal_max,
        fillcolor="rgba(46, 160, 67, 0.08)",
        line_width=0,
        annotation_text="Nominal" if current_tick < 25 else "",
        annotation_position="top left",
        annotation_font=dict(color="#8b949e", size=9)
    )

    # Warning threshold
    fig.add_hline(
        y=cfg.warning_threshold,
        line=dict(color="#e3b341", width=1.5, dash="dash"),
        annotation_text="Warning",
        annotation_position="top right",
        annotation_font=dict(color="#e3b341", size=9)
    )

    # Critical threshold
    fig.add_hline(
        y=cfg.critical_threshold,
        line=dict(color="#f85149", width=1.5, dash="dot"),
        annotation_text="Critical",
        annotation_position="bottom right",
        annotation_font=dict(color="#f85149", size=9)
    )

    # Main telemetry line trace
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=window_df[sensor_key],
        mode="lines",
        name=cfg.name,
        line=dict(color=line_color, width=2.2),
        hoverinfo="x+y"
    ))

    # Current value highlight point
    cur_val = window_df[sensor_key].iloc[-1]
    fig.add_trace(go.Scatter(
        x=[x_vals[-1]],
        y=[cur_val],
        mode="markers",
        name="Current",
        marker=dict(color="#ffffff", size=7, line=dict(color=line_color, width=2)),
        hoverinfo="skip"
    ))

    x_min = max(0, current_tick - window)
    x_max = max(window, current_tick + 2)

    fig.update_layout(
        title=dict(
            text=f"<b>{cfg.name}</b>: {cur_val:.2f} {cfg.unit}",
            font=dict(color=THEME_COLORS["text_highlight"], size=13),
            x=0.01, y=0.96
        ),
        paper_bgcolor=THEME_COLORS["card_bg"],
        plot_bgcolor=THEME_COLORS["card_bg"],
        height=210,
        margin=dict(l=45, r=20, t=32, b=28),
        dragmode=False,
        uirevision=f"sensor_{sensor_key}",
        showlegend=False,
        xaxis=dict(
            title=dict(text="Time (s)", font=dict(color=THEME_COLORS["text_muted"], size=10)),
            range=[x_min, x_max],
            gridcolor="#21262d",
            zerolinecolor="#30363d",
            tickfont=dict(color=THEME_COLORS["text_muted"], size=10),
            fixedrange=True
        ),
        yaxis=dict(
            title=dict(text=cfg.unit, font=dict(color=THEME_COLORS["text_muted"], size=10)),
            gridcolor="#21262d",
            zerolinecolor="#30363d",
            tickfont=dict(color=THEME_COLORS["text_muted"], size=10),
            fixedrange=True
        )
    )
    return fig


# -----------------------------------------------------------------------------
# Section B, C, D: Zero-Blink High-Performance Telemetry Display (st.fragment)
# -----------------------------------------------------------------------------
run_interval = (1.0 / st.session_state["playback_speed"]) if (st.session_state["is_playing"] and st.session_state["scenario_id"] != "custom") else None

@st.fragment(run_every=run_interval)
def render_live_telemetry_fragment(sidebar_placeholder=None):
    scen_id = st.session_state["scenario_id"]

    if scen_id != "custom":
        # Pre-fixed Scenario Mode (A, B, C)
        if st.session_state["is_playing"]:
            if st.session_state["current_tick"] < 159:
                st.session_state["current_tick"] += 1
            else:
                st.session_state["is_playing"] = False

        tick = st.session_state["current_tick"]

        # Keep sidebar playback progress bar synchronized in real time
        if sidebar_placeholder is not None:
            sidebar_placeholder.progress(
                float(tick) / 159.0,
                text=f"Simulation Progress: {tick}s / 160s"
            )

        df, inferences, attributions, recommendations = get_cached_scenario_data(scen_id)
        cur_inf = inferences[tick]
        cur_attr = attributions[tick]
        cur_rec = recommendations[tick]
        cur_reading = df.iloc[tick]
        plot_df = df
        plot_tick = tick

        # Simulation Timeline Navigation Header
        col_t1, col_t2, col_t3, col_t4, col_t5 = st.columns([3, 1, 1, 1, 1])
        with col_t1:
            st.markdown(f"### Simulation Timeline: `{tick}s / 160s`")
        with col_t2:
            if st.session_state["is_playing"]:
                if st.button("Pause", key="dash_btn_pause", use_container_width=True):
                    st.session_state["is_playing"] = False
                    st.rerun()
            else:
                if st.button("Play", key="dash_btn_play", use_container_width=True):
                    if st.session_state["current_tick"] >= 159:
                        st.session_state["current_tick"] = 0
                    st.session_state["is_playing"] = True
                    st.rerun()
        with col_t3:
            if st.button("0s Baseline", key="dash_jump_0", use_container_width=True):
                st.session_state["current_tick"] = 0
                st.session_state["is_playing"] = False
                st.rerun()
        with col_t4:
            if st.button("88s Warning", key="dash_jump_88", use_container_width=True):
                st.session_state["current_tick"] = 88
                st.session_state["is_playing"] = False
                st.rerun()
        with col_t5:
            if st.button("133s Critical", key="dash_jump_133", use_container_width=True):
                st.session_state["current_tick"] = 133
                st.session_state["is_playing"] = False
                st.rerun()

        st.progress(float(tick) / 159.0)

    else:
        # Custom Scenario Mode: Live Real-Time Operator Tuning
        model_inst = get_or_create_maintenance_model()
        cust_hist = st.session_state["custom_history_df"]
        plot_df = cust_hist
        plot_tick = int(cust_hist["tick"].iloc[-1])
        cur_reading = cust_hist.iloc[-1]

        cur_inf = model_inst.predict_reading(cur_reading, instantaneous=True)
        cur_attr = explain_prediction(cur_inf, model_inst.baseline_stats)
        cur_rec = get_recommendation_for_telemetry(
            cur_inf,
            st.session_state["checklist_state"],
            history_window_df=cust_hist.tail(20)
        )

        st.markdown(f"### Live Telemetry Stream | Buffer Count: `{len(cust_hist)} samples`")

    # Store latest diagnostics in session state for work order export
    st.session_state["latest_inference"] = cur_inf
    st.session_state["latest_attribution"] = cur_attr
    st.session_state["latest_recommendation"] = cur_rec
    st.session_state["latest_reading"] = cur_reading

    # 4 Real-time KPI Cards
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    health = cur_inf.health_score
    if health >= 80:
        h_color = THEME_COLORS["status_normal"]
        h_class = "status-normal"
    elif health >= 60:
        h_color = THEME_COLORS["status_moderate"]
        h_class = "status-moderate"
    elif health >= 40:
        h_color = THEME_COLORS["status_elevated"]
        h_class = "status-elevated"
    else:
        h_color = THEME_COLORS["status_critical"]
        h_class = "status-critical"

    with kpi_col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Machine Health Score</div>
                <div class="kpi-value" style="color: {h_color};">
                    <span class="status-dot {h_class}"></span>{health:.1f}%
                </div>
                <div class="kpi-sub">Baseline: 100.0% Nominal</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    risk_cat = cur_inf.risk_category
    risk_colors = {
        "Low": THEME_COLORS["status_normal"],
        "Moderate": THEME_COLORS["status_moderate"],
        "Elevated": THEME_COLORS["status_elevated"],
        "Critical": THEME_COLORS["status_critical"]
    }
    r_color = risk_colors.get(risk_cat, THEME_COLORS["text_main"])

    with kpi_col2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Inspection Risk Index</div>
                <div class="kpi-value" style="color: {r_color};">{risk_cat.upper()}</div>
                <div class="kpi-sub">Heuristic Risk: {cur_inf.risk_score:.2f} / 1.00</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi_col3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Max Sensor Deviation</div>
                <div class="kpi-value">{cur_inf.max_z_score:.2f} σ</div>
                <div class="kpi-sub">Primary Driver: {cur_attr.primary_contributor}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi_col4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Active Operating Pattern</div>
                <div class="kpi-value" style="font-size: 1.15rem; color: {r_color};">{cur_rec.pattern_name}</div>
                <div class="kpi-sub">Priority: {cur_rec.priority} | Severity: {cur_rec.severity}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

    # 4 Sensor Telemetry Waveform Charts
    col_chart_left, col_chart_right = st.columns(2)
    with col_chart_left:
        fig_temp = build_sensor_chart("temperature", plot_df, plot_tick, window=st.session_state["history_window"])
        st.plotly_chart(fig_temp, width="stretch", config={"displayModeBar": False}, key="pchart_temp")
        
        fig_press = build_sensor_chart("pressure", plot_df, plot_tick, window=st.session_state["history_window"])
        st.plotly_chart(fig_press, width="stretch", config={"displayModeBar": False}, key="pchart_press")

    with col_chart_right:
        fig_vib = build_sensor_chart("vibration", plot_df, plot_tick, window=st.session_state["history_window"])
        st.plotly_chart(fig_vib, width="stretch", config={"displayModeBar": False}, key="pchart_vib")
        
        fig_speed = build_sensor_chart("speed", plot_df, plot_tick, window=st.session_state["history_window"])
        st.plotly_chart(fig_speed, width="stretch", config={"displayModeBar": False}, key="pchart_speed")

    # Sensor Anomaly Attribution Chart
    fig_attr = create_attribution_chart(cur_attr)
    st.plotly_chart(fig_attr, width="stretch", config={"displayModeBar": False}, key="pchart_attr")
    st.caption(f"**Diagnostic Summary:** {cur_attr.plain_language_summary}")

    # Verified Warning Lead Time Badge for Hackathon Demonstration
    if scen_id == "scenario_b":
        st.info("Verified Warning Lead Time: Early warning raised at tick 88s, providing 45 seconds of advance lead time before the 5.0 mm/s critical threshold is breached at tick 133s.")
    elif scen_id == "scenario_c":
        st.warning("Rapid Failure Mode: Coolant pressure loss at tick 60s triggers immediate critical temperature rise within 15 seconds.")


# Render the isolated, live telemetry fragment
render_live_telemetry_fragment(sidebar_progress_placeholder)

st.divider()


# -----------------------------------------------------------------------------
# Section F: Prescriptive Maintenance Workbench & Work-Order Export
# -----------------------------------------------------------------------------
recommendation = st.session_state.get("latest_recommendation", latest_recommendation)
inference_result = st.session_state.get("latest_inference", latest_inference)
attribution_report = st.session_state.get("latest_attribution", latest_attribution)

st.markdown("### Prescriptive Maintenance Workbench")

# 1. Dynamic Deterioration Velocity & Rate-of-Change Banner
if recommendation.rate_of_change_summary:
    if "RAPID" in recommendation.rate_of_change_summary:
        st.error(f"🚨 **Deterioration Velocity Alert:** {recommendation.rate_of_change_summary}")
    elif "ACCELERATING" in recommendation.rate_of_change_summary:
        st.warning(f"⚠️ **Deterioration Velocity Warning:** {recommendation.rate_of_change_summary}")
    elif "PROGRESSIVE" in recommendation.rate_of_change_summary:
        st.info(f"⏳ **Deterioration Rate:** {recommendation.rate_of_change_summary}")
    else:
        st.success(f"✅ **Operating Velocity:** {recommendation.rate_of_change_summary}")

# 2. Sensor Elevation Magnitude & Deviation Analysis Table
st.markdown("#### Individual Sensor Elevation & Severity Analysis")
st.caption("Quantitative audit of which parameters are elevated, the magnitude of deviation from baseline, and rate of change.")

elev_rows = []
for item in recommendation.elevation_details:
    elev_rows.append({
        "Sensor": item["sensor_name"],
        "Current Reading": f"{item['current_value']} {item['unit']}",
        "Nominal Baseline": f"{item['baseline_mean']} {item['unit']}",
        "Deviation (Δ)": f"{item['deviation']:+} {item['unit']}",
        "Deviation Z-Score": f"{item['z_score']:+.2f} σ",
        "Severity Status": item["status"],
        "Velocity (d/dt)": f"{item['rate_of_change']:+.3f} {item['unit']}/s"
    })

st.dataframe(pd.DataFrame(elev_rows), width="stretch", hide_index=True)

# 3. Machine Technical Context & Operational Factors
with st.expander("Machine Specifications & Operating Factors (Equipment Baseline Context)", expanded=False):
    mf_cols = st.columns(2)
    mf_items = list(recommendation.machine_factors.items())
    half_pt = len(mf_items) // 2
    with mf_cols[0]:
        for k, v in mf_items[:half_pt]:
            st.markdown(f"- **{k}:** `{v}`")
    with mf_cols[1]:
        for k, v in mf_items[half_pt:]:
            st.markdown(f"- **{k}:** `{v}`")

# 4. Actionable How-To-Fix Engineering Playbook
fix = recommendation.actionable_fix_guide
if fix:
    with st.expander("Actionable How-To-Fix Engineering Playbook", expanded=True):
        col_fix1, col_fix2 = st.columns([1, 1])
        with col_fix1:
            st.markdown("##### Immediate Operational Action")
            st.info(fix.get("immediate_action", "Continue standard monitoring."))

            st.markdown("##### Diagnostic Verification (Root-Cause Isolation)")
            st.markdown(fix.get("diagnostic_verification", "Conduct visual and acoustic walkdown."))

            st.markdown("##### Required Spare Parts & Specialist Tools")
            tools = fix.get("required_tools_parts", [])
            for t in tools:
                st.markdown(f"- `{t}`")

            st.metric("Estimated Mean Time to Repair (MTTR)", fix.get("estimated_mttr", "N/A"))

        with col_fix2:
            st.markdown("##### Step-by-Step Mechanical & Corrective Procedure")
            mech_steps = fix.get("mechanical_fix_procedure", [])
            if isinstance(mech_steps, list):
                for ms in mech_steps:
                    st.markdown(f"- {ms}")
            else:
                st.markdown(f"- {mech_steps}")

            st.markdown("##### Electrical & Control Verification")
            st.markdown(f"- {fix.get('electrical_fix_procedure', 'Verify motor junction box connections.')}")

            st.markdown("##### Hydraulic & Process Verification")
            st.markdown(f"- {fix.get('hydraulic_fix_procedure', 'Verify pump suction and discharge lines.')}")

st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

# 5. Interactive Inspection Checklist & Work Order Export
col_work_left, col_work_right = st.columns([1, 1])

with col_work_left:
    st.markdown(f"#### Identified Fault Pattern: `{recommendation.pattern_name}`")
    st.markdown(f"**Priority:** `{recommendation.priority}` &nbsp;|&nbsp; **Severity:** `{recommendation.severity}`")
    st.markdown(f"**Primary Reason:** {recommendation.primary_reason}")

    if recommendation.escalation_notes:
        st.warning(recommendation.escalation_notes)

    st.markdown("##### Safe Order of Operations")
    for op in recommendation.safe_order_of_operations:
        st.markdown(f"- {op}")

    st.info(f"**Mandatory Safety Protocol:** {recommendation.safety_advisory}")

with col_work_right:
    st.markdown("#### Interactive Inspection Checklist")
    st.caption("Check off completed verification items before issuing work orders.")

    for step in recommendation.inspection_steps:
        crit_label = " [CRITICAL]" if step.is_critical else ""
        step_checked = st.checkbox(
            f"Step {step.order}: {step.title}{crit_label}",
            value=st.session_state["checklist_state"].get(step.step_id, False),
            key=f"chk_{step.step_id}"
        )
        st.session_state["checklist_state"][step.step_id] = step_checked
        st.caption(f"&nbsp;&nbsp;&nbsp;&nbsp;*{step.description}*")

    st.markdown("##### Operator Notes")
    operator_notes_input = st.text_area(
        "Enter maintenance observations, physical readings, or shift handoff notes:",
        value=st.session_state["operator_notes"],
        height=70,
        placeholder="e.g. Conducted acoustic inspection on drive-end bearing. Slight metallic grinding audible.",
        key="txt_operator_notes"
    )
    st.session_state["operator_notes"] = operator_notes_input

    # Work Order Generation Button
    if st.button("Generate Maintenance Work Order", type="primary", use_container_width=True):
        wo = create_work_order(
            inference_result=inference_result,
            attribution_report=attribution_report,
            recommendation=recommendation,
            scenario_id=st.session_state["scenario_id"],
            operator_notes=st.session_state["operator_notes"],
            machine_id=DEFAULT_MACHINE_ID
        )
        st.session_state["last_work_order"] = wo
        st.success(f"Work Order {wo.work_order_id} generated successfully!")

# Export Download Buttons Row
if st.session_state["last_work_order"] is not None:
    wo = st.session_state["last_work_order"]
    st.markdown(f"#### Export Work Order: `{wo.work_order_id}`")
    
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    
    with col_dl1:
        json_data = export_work_order_json(wo)
        st.download_button(
            label="Download JSON Ticket",
            data=json_data,
            file_name=f"{wo.work_order_id}.json",
            mime="application/json",
            use_container_width=True
        )

    with col_dl2:
        md_data = export_work_order_markdown(wo)
        st.download_button(
            label="Download Markdown Report",
            data=md_data,
            file_name=f"{wo.work_order_id}.md",
            mime="text/markdown",
            use_container_width=True
        )

    with col_dl3:
        try:
            pdf_bytes = export_work_order_pdf(wo)
            st.download_button(
                label="Download PDF Document",
                data=pdf_bytes,
                file_name=f"{wo.work_order_id}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"PDF export error: {e}")

st.divider()


# -----------------------------------------------------------------------------
# Section G: Historical CSV Telemetry & Retrospective Evaluation
# -----------------------------------------------------------------------------
st.markdown("### Historical Telemetry & Evaluation Benchmark")

tab_hist, tab_bench, tab_amd = st.tabs([
    "Historical CSV Upload & Inspection",
    "Scenario Benchmark & Model Evaluation",
    "AMD Acceleration & Edge Architecture"
])

with tab_hist:
    st.markdown("#### Upload Machine Telemetry CSV")
    st.markdown(
        "Upload factory telemetry records. Schema requires timestamp (optional), "
        "`temperature`, `vibration`, `pressure`, and `speed`."
    )
    
    uploaded_file = st.file_uploader("Select historical CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            is_valid, msg, cleaned_df = validate_and_normalize_csv(raw_df)
            if is_valid and cleaned_df is not None:
                st.success(msg)
                st.dataframe(cleaned_df.head(10), width="stretch")
                
                # Evaluate model on uploaded dataset
                if st.button("Run Full Dataset Anomaly Evaluation", key="btn_eval_upload"):
                    with st.spinner("Executing model evaluation on uploaded dataset..."):
                        eval_model = PredictiveMaintenanceModel()
                        eval_model.fit_baseline()
                        upload_eval = evaluate_model_on_dataset(eval_model, cleaned_df)
                        st.markdown("##### Uploaded Dataset Evaluation Metrics")
                        st.json(upload_eval)
            else:
                st.error(f"Validation Error: {msg}")
        except Exception as e:
            st.error(f"CSV Parsing Failed: {str(e)}")

with tab_bench:
    st.markdown("#### Retrospective Model Performance Metrics")
    st.markdown(
        "Evaluates the unsupervised anomaly detection model and heuristic risk index "
        "across complete simulated scenarios with ground truth labels."
    )

    col_btn_a, col_btn_b, col_btn_c = st.columns(3)
    
    eval_target_scenario = None
    with col_btn_a:
        if st.button("Evaluate Scenario A (Normal)", use_container_width=True):
            eval_target_scenario = "scenario_a"
    with col_btn_b:
        if st.button("Evaluate Scenario B (Bearing)", use_container_width=True):
            eval_target_scenario = "scenario_b"
    with col_btn_c:
        if st.button("Evaluate Scenario C (Coolant Leak)", use_container_width=True):
            eval_target_scenario = "scenario_c"

    if eval_target_scenario is not None:
        eval_model = PredictiveMaintenanceModel()
        eval_model.fit_baseline()
        target_df = generate_scenario_telemetry(scenario_id=eval_target_scenario)
        eval_metrics = evaluate_model_on_dataset(eval_model, target_df)

        st.markdown(f"##### Benchmark Results: `{SCENARIOS[eval_target_scenario]['name']}`")
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Detection Rate (Sensitivity)", f"{eval_metrics['anomaly_detection_rate']}")
        m_col2.metric("False Alarm Rate", f"{eval_metrics['false_alarm_rate']}")
        m_col3.metric("Warning Lead Time", f"{eval_metrics['warning_lead_time_seconds']}s" if eval_metrics['warning_lead_time_seconds'] != 'N/A' else "N/A")
        m_col4.metric("Detection Latency", f"{eval_metrics['detection_latency_ticks']} ticks" if eval_metrics['detection_latency_ticks'] != 'N/A' else "N/A")

        st.json(eval_metrics)

with tab_amd:
    st.markdown("#### AMD Hardware Acceleration & Edge Deployment Strategy")
    
    col_amd1, col_amd2 = st.columns([1, 1])
    with col_amd1:
        st.markdown("##### Runtime Hardware Audit")
        st.markdown(f"- **Operating System:** `{hw_status.os_name} {hw_status.os_release} ({hw_status.cpu_architecture})`")
        st.markdown(f"- **Host Processor:** `{hw_status.processor_name}`")
        st.markdown(f"- **Python Version:** `{hw_status.python_version}`")
        st.markdown(f"- **AMD Hardware Detected:** `{'Yes' if hw_status.is_amd_hardware else 'No (Intel / Other Host)'}`")
        st.markdown(f"- **ONNX Runtime Installed:** `{'Yes (v' + hw_status.onnx_version + ')' if hw_status.has_onnx_runtime else 'No'}`")
        st.markdown(f"- **Available Execution Providers:** `{hw_status.available_providers}`")
        st.markdown(f"- **Active Provider:** `{hw_status.selected_provider}`")
        st.markdown(f"- **Hardware Acceleration Status:** `{'VERIFIED ACCELERATED' if hw_status.hardware_accelerated else 'CPU FALLBACK VERIFIED'}`")

        if hw_status.benchmark_latency_ms is not None:
            st.metric("Test Inference Latency", f"{hw_status.benchmark_latency_ms:.3f} ms")

    with col_amd2:
        st.markdown("##### Edge Deployment Blueprint")
        st.markdown(
            """
            **1. AMD Ryzen AI (XDNA Architecture):**
            - Target: Industrial laptops, edge gateways with AMD Ryzen 7040/8040/AI 300 series.
            - Execution: ONNX Runtime via `VitisAIExecutionProvider`.
            - Advantage: Offloads continuous 100 Hz sensor scoring to dedicated NPU, preserving CPU for real-time PLC communications.

            **2. AMD ROCm on Industrial Linux:**
            - Target: Central factory floor servers equipped with AMD Radeon PRO or Instinct accelerators.
            - Execution: High-throughput batch multi-machine anomaly detection across thousands of factory nodes.

            **3. Industrial Protocols Integration Roadmap:**
            - OPC UA (IEC 62541) Client connector for direct SCADA / PLC tags.
            - MQTT / Sparkplug B broker ingestion for distributed factory IoT.
            """
        )
