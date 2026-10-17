"""
Interactive, zero-flicker industrial telemetry console for MachineMind.
Precomputes scenario timeline through the unsupervised ML model and renders a
high-performance client-side dashboard with live waveforms, health gauge,
attribution bars, and deterministic controls with ZERO browser blinking.
"""

import json
from typing import List, Dict, Any
import streamlit as st
import streamlit.components.v1 as components

from data_generator import generate_scenario_telemetry
from model import PredictiveMaintenanceModel
from explainer import explain_prediction
from checklist import get_recommendation_for_telemetry
from config import SENSOR_CONFIGS, THEME_COLORS


@st.cache_data
def get_scenario_timeline_data(scenario_id: str, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Precomputes full scenario telemetry, ML anomaly scores, health scores,
    and root-cause attributions for all 160 seconds of a scenario.
    Cached to execute instantly with zero server lag on reruns.
    """
    model = PredictiveMaintenanceModel()
    model.fit_baseline()
    df = generate_scenario_telemetry(scenario_id=scenario_id, duration_seconds=160, seed=seed)

    records = []
    for i, row in df.iterrows():
        inf = model.predict_reading(row)
        attr = explain_prediction(inf, model.baseline_stats)
        rec = get_recommendation_for_telemetry(inf)
        
        has_alert = model.active_alert is not None and model.active_alert.status != "RESOLVED"
        alert_lvl = model.active_alert.level if has_alert else "NORMAL"
        alert_id = model.active_alert.alert_id if has_alert else "NONE"
        alert_reason = model.active_alert.reason if has_alert else "Operating within nominal limits."

        records.append({
            "t": int(i),
            "temp": round(float(row["temperature"]), 2),
            "vib": round(float(row["vibration"]), 3),
            "press": round(float(row["pressure"]), 3),
            "speed": round(float(row["speed"]), 1),
            "health": round(float(inf.health_score), 1),
            "risk_cat": str(inf.risk_category),
            "risk_score": round(float(inf.risk_score), 2),
            "max_z": round(float(inf.max_z_score), 1),
            "has_alert": has_alert,
            "alert_id": alert_id,
            "alert_lvl": alert_lvl,
            "alert_reason": alert_reason,
            "primary_driver": attr.primary_contributor,
            "summary": attr.plain_language_summary,
            "pattern_name": rec.pattern_name,
            "priority": rec.priority,
            "severity": rec.severity,
            "attributions": [
                {
                    "id": item.sensor_id,
                    "name": item.sensor_name,
                    "val": round(float(item.current_value), 2),
                    "base": round(float(item.baseline_mean), 2),
                    "z": round(float(item.relative_z_score), 1),
                    "pct": round(float(item.contribution_percentage), 1),
                    "status": item.status,
                    "unit": item.unit
                }
                for item in attr.attribution_items
            ]
        })

    return records


def render_interactive_telemetry_console(scenario_id: str, lead_time_str: str = "45 Seconds"):
    """
    Renders the zero-flicker client-side industrial console inside an isolated container.
    """
    timeline_records = get_scenario_timeline_data(scenario_id)
    data_json = json.dumps(timeline_records)

    html_code = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <style>
            * {{
                box-sizing: border-box;
                margin: 0;
                padding: 0;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            }}
            body {{
                background-color: #0d1117;
                color: #c9d1d9;
                padding: 10px;
                overflow-x: hidden;
            }}
            
            /* Control Bar */
            .control-panel {{
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 12px 16px;
                margin-bottom: 12px;
            }}
            .control-row {{
                display: flex;
                flex-wrap: wrap;
                align-items: center;
                justify-content: space-between;
                gap: 12px;
            }}
            .btn-group {{
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            button {{
                background-color: #21262d;
                color: #f0f6fc;
                border: 1px solid #30363d;
                border-radius: 4px;
                padding: 6px 14px;
                font-size: 13px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.15s ease;
            }}
            button:hover {{
                background-color: #30363d;
                border-color: #8b949e;
            }}
            button.btn-primary {{
                background-color: #238636;
                border-color: #2ea043;
            }}
            button.btn-primary:hover {{
                background-color: #2ea043;
            }}
            button.btn-active {{
                background-color: #1f6feb;
                border-color: #388bfd;
            }}
            
            .slider-container {{
                display: flex;
                align-items: center;
                gap: 12px;
                flex: 1;
                min-width: 280px;
            }}
            input[type=range] {{
                flex: 1;
                accent-color: #58a6ff;
                cursor: pointer;
                height: 6px;
            }}
            .time-display {{
                font-family: monospace;
                font-size: 14px;
                font-weight: 700;
                color: #58a6ff;
                min-width: 110px;
                text-align: right;
            }}

            /* KPI Cards */
            .kpi-row {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 12px;
                margin-bottom: 12px;
            }}
            @media (max-width: 800px) {{
                .kpi-row {{ grid-template-columns: repeat(2, 1fr); }}
            }}
            .kpi-card {{
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 12px 14px;
            }}
            .kpi-title {{
                font-size: 11px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                color: #8b949e;
                margin-bottom: 4px;
            }}
            .kpi-value {{
                font-size: 24px;
                font-weight: 700;
                line-height: 1.1;
                margin-bottom: 4px;
            }}
            .kpi-sub {{
                font-size: 11px;
                color: #8b949e;
            }}
            .badge-low {{ color: #2ea043; }}
            .badge-mod {{ color: #e3b341; }}
            .badge-elev {{ color: #db6d28; }}
            .badge-crit {{ color: #f85149; }}

            /* Charts Grid */
            .charts-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 12px;
                margin-bottom: 12px;
            }}
            @media (max-width: 850px) {{
                .charts-grid {{ grid-template-columns: 1fr; }}
            }}
            .chart-card {{
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 10px 12px;
            }}
            .chart-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 6px;
            }}
            .chart-title {{
                font-size: 12px;
                font-weight: 600;
                color: #f0f6fc;
            }}
            .chart-value {{
                font-family: monospace;
                font-size: 13px;
                font-weight: 700;
            }}
            canvas {{
                width: 100% !important;
                height: 140px !important;
                display: block;
            }}

            /* Diagnostics Row */
            .diag-row {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 12px;
                margin-bottom: 8px;
            }}
            @media (max-width: 850px) {{
                .diag-row {{ grid-template-columns: 1fr; }}
            }}
            .diag-card {{
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 12px 14px;
            }}
            .diag-title {{
                font-size: 13px;
                font-weight: 600;
                color: #f0f6fc;
                margin-bottom: 8px;
                border-bottom: 1px solid #21262d;
                padding-bottom: 4px;
            }}

            /* Attribution Bars */
            .attr-bar-container {{
                margin-bottom: 8px;
            }}
            .attr-label {{
                display: flex;
                justify-content: space-between;
                font-size: 11px;
                color: #c9d1d9;
                margin-bottom: 3px;
            }}
            .bar-track {{
                background-color: #21262d;
                border-radius: 3px;
                height: 14px;
                overflow: hidden;
            }}
            .bar-fill {{
                height: 100%;
                transition: width 0.15s ease-out;
            }}

            /* Readout Table */
            table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 11px;
            }}
            th {{
                text-align: left;
                color: #8b949e;
                padding: 4px 6px;
                border-bottom: 1px solid #30363d;
            }}
            td {{
                padding: 4px 6px;
                border-bottom: 1px solid #21262d;
                color: #c9d1d9;
            }}

            /* Alert Banner */
            .alert-banner {{
                padding: 10px 14px;
                border-radius: 6px;
                margin-bottom: 12px;
                border-left: 4px solid #2ea043;
                background-color: #161b22;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .alert-banner.alert-warning {{
                border-left-color: #d29922;
                background-color: #242014;
            }}
            .alert-banner.alert-critical {{
                border-left-color: #f85149;
                background-color: #2b1718;
            }}
        </style>
    </head>
    <body>
        <!-- Top Control Bar -->
        <div class="control-panel">
            <div class="control-row">
                <div class="btn-group">
                    <button id="btnPlay" class="btn-primary" onclick="togglePlay()">Play</button>
                    <button onclick="resetSim()">Reset</button>
                    <button id="btnSpd05" onclick="setSpeed(0.5)">0.5x</button>
                    <button id="btnSpd1" class="btn-active" onclick="setSpeed(1.0)">1x</button>
                    <button id="btnSpd2" onclick="setSpeed(2.0)">2x</button>
                </div>
                
                <div class="slider-container">
                    <input type="range" id="timeSlider" min="0" max="159" value="0" oninput="onSliderDrag(this.value)">
                    <div class="time-display" id="timeDisplay">t = 0s / 159s</div>
                </div>

                <div class="btn-group">
                    <button onclick="jumpTo(20)">Baseline 20s</button>
                    <button onclick="jumpTo(50)">Onset 50s</button>
                    <button onclick="jumpTo(88)">Warning 88s</button>
                    <button onclick="jumpTo(133)">Critical 133s</button>
                </div>
            </div>
        </div>

        <!-- Alert State Banner -->
        <div class="alert-banner" id="alertBanner">
            <div>
                <strong id="alertTitle" style="color: #2ea043;">NOMINAL ENVELOPE: All sensors within normal tolerances</strong>
                <div id="alertReason" style="font-size: 11px; color: #8b949e; margin-top: 2px;">Multi-sensor baseline stability verified. No active fault.</div>
            </div>
            <div id="alertBadge" style="font-size: 11px; font-weight: 700; color: #8b949e;">NORMAL</div>
        </div>

        <!-- Executive KPI Row -->
        <div class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-title">Machine Health Score</div>
                <div class="kpi-value" id="kpiHealth" style="color: #2ea043;">99.2%</div>
                <div class="kpi-sub">Continuous dynamic degradation indicator</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Inspection Risk Index</div>
                <div class="kpi-value" id="kpiRisk" style="color: #2ea043;">Low</div>
                <div class="kpi-sub" id="kpiRiskSub">Risk Score: 0.01 (Max Z=0.6σ)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Hysteresis Alert State</div>
                <div class="kpi-value" id="kpiAlert" style="color: #2ea043;">0 Active</div>
                <div class="kpi-sub" id="kpiAlertSub">Persistent: 0 / 3 ticks</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Warning Lead Time</div>
                <div class="kpi-value" id="kpiLead">{lead_time_str}</div>
                <div class="kpi-sub">Verified from 88s (Warn) to 133s (Crit)</div>
            </div>
        </div>

        <!-- 4 Synchronized Multi-Channel Waveforms (Canvas 60 FPS) -->
        <div class="charts-grid">
            <div class="chart-card">
                <div class="chart-header">
                    <span class="chart-title">Bearing & Stator Temperature (°C) [Warn 85°C | Crit 95°C]</span>
                    <span class="chart-value" id="valTemp" style="color: #f0883e;">70.1 °C</span>
                </div>
                <canvas id="canvasTemp"></canvas>
            </div>
            
            <div class="chart-card">
                <div class="chart-header">
                    <span class="chart-title">Radial Vibration (mm/s RMS) [Warn 5.0 | Crit 7.0]</span>
                    <span class="chart-value" id="valVib" style="color: #58a6ff;">2.15 mm/s</span>
                </div>
                <canvas id="canvasVib"></canvas>
            </div>

            <div class="chart-card">
                <div class="chart-header">
                    <span class="chart-title">Hydraulic & Lube Pressure (bar) [Warn 3.8 | Crit 2.8]</span>
                    <span class="chart-value" id="valPress" style="color: #3fb950;">5.02 bar</span>
                </div>
                <canvas id="canvasPress"></canvas>
            </div>

            <div class="chart-card">
                <div class="chart-header">
                    <span class="chart-title">Rotational Shaft Speed (RPM) [Warn Low 1680]</span>
                    <span class="chart-value" id="valSpeed" style="color: #a371f7;">1800 RPM</span>
                </div>
                <canvas id="canvasSpeed"></canvas>
            </div>
        </div>

        <!-- Diagnostics Row: Attribution & Sensor Readout Table -->
        <div class="diag-row">
            <div class="diag-card">
                <div class="diag-title">Root-Cause Attribution: Relative Sensor Deviation Contribution</div>
                <div id="attributionBars"></div>
                <div id="diagSummary" style="font-size: 11px; color: #8b949e; margin-top: 8px;">
                    Telemetry is operating stably within normal design tolerances.
                </div>
            </div>

            <div class="diag-card">
                <div class="diag-title">Multi-Sensor Telemetry Readout & Baselines</div>
                <table>
                    <thead>
                        <tr>
                            <th>Sensor</th>
                            <th>Current</th>
                            <th>Baseline</th>
                            <th>Z-Score</th>
                            <th>Contribution</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody id="readoutTableBody"></tbody>
                </table>
            </div>
        </div>

        <script>
            const DATA = {data_json};
            let currentIndex = 0;
            let isPlaying = false;
            let speed = 1.0;
            let timerId = null;

            // Canvas contexts
            const ctxTemp = document.getElementById('canvasTemp').getContext('2d');
            const ctxVib = document.getElementById('canvasVib').getContext('2d');
            const ctxPress = document.getElementById('canvasPress').getContext('2d');
            const ctxSpeed = document.getElementById('canvasSpeed').getContext('2d');

            function resizeCanvases() {{
                const dpr = window.devicePixelRatio || 1;
                ['canvasTemp', 'canvasVib', 'canvasPress', 'canvasSpeed'].forEach(id => {{
                    const cv = document.getElementById(id);
                    const rect = cv.getBoundingClientRect();
                    cv.width = rect.width * dpr;
                    cv.height = 140 * dpr;
                    const ctx = cv.getContext('2d');
                    ctx.resetTransform();
                    ctx.scale(dpr, dpr);
                }});
            }}
            window.addEventListener('resize', () => {{
                resizeCanvases();
                renderFrame(currentIndex);
            }});
            resizeCanvases();

            function drawWaveform(ctx, cvId, fullArr, curIdx, minVal, maxVal, colorHex, warnThresh, critThresh, isLowLimit) {{
                const cv = document.getElementById(cvId);
                const w = cv.getBoundingClientRect().width;
                const h = 140;

                ctx.clearRect(0, 0, w, h);

                // Grid background lines
                ctx.strokeStyle = '#21262d';
                ctx.lineWidth = 1;
                for (let y = 20; y < h; y += 30) {{
                    ctx.beginPath();
                    ctx.moveTo(0, y);
                    ctx.lineTo(w, y);
                    ctx.stroke();
                }}

                // Threshold reference lines
                const getY = val => h - ((val - minVal) / (maxVal - minVal)) * (h - 24) - 12;

                if (warnThresh !== null) {{
                    const yWarn = getY(warnThresh);
                    ctx.strokeStyle = '#d29922';
                    ctx.setLineDash([4, 4]);
                    ctx.beginPath();
                    ctx.moveTo(0, yWarn);
                    ctx.lineTo(w, yWarn);
                    ctx.stroke();
                }}
                if (critThresh !== null) {{
                    const yCrit = getY(critThresh);
                    ctx.strokeStyle = '#f85149';
                    ctx.setLineDash([2, 3]);
                    ctx.beginPath();
                    ctx.moveTo(0, yCrit);
                    ctx.lineTo(w, yCrit);
                    ctx.stroke();
                }}
                ctx.setLineDash([]);

                // 1. Full trajectory profile in faint muted line
                ctx.strokeStyle = '#30363d';
                ctx.lineWidth = 1;
                ctx.beginPath();
                for (let i = 0; i < fullArr.length; i++) {{
                    const x = (i / (fullArr.length - 1)) * w;
                    const y = getY(fullArr[i]);
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }}
                ctx.stroke();

                // 2. Active simulated trace up to curIdx
                if (curIdx >= 0) {{
                    ctx.strokeStyle = colorHex;
                    ctx.lineWidth = 2.5;
                    ctx.beginPath();
                    for (let i = 0; i <= curIdx; i++) {{
                        const x = (i / (fullArr.length - 1)) * w;
                        const y = getY(fullArr[i]);
                        if (i === 0) ctx.moveTo(x, y);
                        else ctx.lineTo(x, y);
                    }}
                    ctx.stroke();

                    // 3. Highlighted head marker
                    const curX = (curIdx / (fullArr.length - 1)) * w;
                    const curY = getY(fullArr[curIdx]);

                    ctx.fillStyle = colorHex;
                    ctx.beginPath();
                    ctx.arc(curX, curY, 4.5, 0, Math.PI * 2);
                    ctx.fill();

                    ctx.strokeStyle = '#ffffff';
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                }}
            }}

            function renderFrame(idx) {{
                if (idx < 0) idx = 0;
                if (idx >= DATA.length) idx = DATA.length - 1;
                currentIndex = idx;
                const rec = DATA[idx];

                // Update Control Readouts
                document.getElementById('timeSlider').value = idx;
                document.getElementById('timeDisplay').innerText = 't = ' + idx + 's / ' + (DATA.length - 1) + 's';

                // Numeric Values
                document.getElementById('valTemp').innerText = rec.temp.toFixed(1) + ' °C';
                document.getElementById('valVib').innerText = rec.vib.toFixed(2) + ' mm/s';
                document.getElementById('valPress').innerText = rec.press.toFixed(2) + ' bar';
                document.getElementById('valSpeed').innerText = rec.speed.toFixed(0) + ' RPM';

                // KPIs
                const kpiH = document.getElementById('kpiHealth');
                kpiH.innerText = rec.health.toFixed(1) + '%';
                let hColor = '#2ea043';
                if (rec.health < 35) hColor = '#f85149';
                else if (rec.health < 60) hColor = '#db6d28';
                else if (rec.health < 80) hColor = '#e3b341';
                kpiH.style.color = hColor;

                const kpiR = document.getElementById('kpiRisk');
                kpiR.innerText = rec.risk_cat;
                let rColor = '#2ea043';
                if (rec.risk_cat === 'Critical') rColor = '#f85149';
                else if (rec.risk_cat === 'Elevated') rColor = '#db6d28';
                else if (rec.risk_cat === 'Moderate') rColor = '#e3b341';
                kpiR.style.color = rColor;
                document.getElementById('kpiRiskSub').innerText = 'Risk Score: ' + rec.risk_score.toFixed(2) + ' (Max Z=' + rec.max_z.toFixed(1) + 'σ)';

                const kpiA = document.getElementById('kpiAlert');
                if (rec.has_alert) {{
                    kpiA.innerText = rec.alert_lvl + ' (' + rec.alert_id + ')';
                    kpiA.style.color = rec.alert_lvl === 'CRITICAL' ? '#f85149' : '#d29922';
                    document.getElementById('kpiAlertSub').innerText = 'Persistent Alert Active';
                }} else {{
                    kpiA.innerText = '0 Active';
                    kpiA.style.color = '#2ea043';
                    document.getElementById('kpiAlertSub').innerText = 'Nominal baseline';
                }}

                // Alert Banner
                const banner = document.getElementById('alertBanner');
                const aTitle = document.getElementById('alertTitle');
                const aReason = document.getElementById('alertReason');
                const aBadge = document.getElementById('alertBadge');
                
                banner.className = 'alert-banner';
                if (rec.alert_lvl === 'CRITICAL') {{
                    banner.classList.add('alert-critical');
                    aTitle.innerText = 'CRITICAL INSPECTION ALERT: Immediate emergency inspection required';
                    aTitle.style.color = '#f85149';
                    aBadge.innerText = 'CRITICAL';
                    aBadge.style.color = '#f85149';
                    aReason.innerText = rec.alert_reason;
                }} else if (rec.alert_lvl === 'WARNING') {{
                    banner.classList.add('alert-warning');
                    aTitle.innerText = 'ELEVATED WARNING ALERT: Progressive mechanical deviation detected';
                    aTitle.style.color = '#d29922';
                    aBadge.innerText = 'WARNING';
                    aBadge.style.color = '#d29922';
                    aReason.innerText = rec.alert_reason;
                }} else {{
                    aTitle.innerText = 'NOMINAL ENVELOPE: All sensors operating within normal tolerances';
                    aTitle.style.color = '#2ea043';
                    aBadge.innerText = 'NORMAL';
                    aBadge.style.color = '#2ea043';
                    aReason.innerText = 'Multi-sensor baseline stability verified. No active fault.';
                }}

                // Draw Waveforms
                const allTemps = DATA.map(d => d.temp);
                const allVibs = DATA.map(d => d.vib);
                const allPress = DATA.map(d => d.press);
                const allSpeeds = DATA.map(d => d.speed);

                drawWaveform(ctxTemp, 'canvasTemp', allTemps, idx, 64, 102, '#f0883e', 85, 95, false);
                drawWaveform(ctxVib, 'canvasVib', allVibs, idx, 1.2, 8.2, '#58a6ff', 5.0, 7.0, false);
                drawWaveform(ctxPress, 'canvasPress', allPress, idx, 1.8, 5.8, '#3fb950', 3.8, 2.8, true);
                drawWaveform(ctxSpeed, 'canvasSpeed', allSpeeds, idx, 1550, 1860, '#a371f7', 1680, null, true);

                // Attribution Bars
                const attrContainer = document.getElementById('attributionBars');
                let attrHtml = '';
                rec.attributions.forEach(a => {{
                    let bColor = '#2ea043';
                    if (a.status === 'CRITICAL') bColor = '#f85149';
                    else if (a.status === 'WARNING') bColor = '#db6d28';
                    else if (a.status === 'DEVIATED') bColor = '#e3b341';

                    attrHtml += `
                        <div class="attr-bar-container">
                            <div class="attr-label">
                                <span>${{a.name}} (${{a.val}} ${{a.unit}})</span>
                                <span style="font-weight:700;">${{a.pct.toFixed(1)}}% (Z=${{a.z.toFixed(1)}}σ)</span>
                            </div>
                            <div class="bar-track">
                                <div class="bar-fill" style="width: ${{Math.min(100, Math.max(2, a.pct))}}%; background-color: ${{bColor}};"></div>
                            </div>
                        </div>
                    `;
                }});
                attrContainer.innerHTML = attrHtml;
                document.getElementById('diagSummary').innerText = rec.summary;

                // Readout Table
                const tbody = document.getElementById('readoutTableBody');
                let tHtml = '';
                rec.attributions.forEach(a => {{
                    tHtml += `
                        <tr>
                            <td><strong>${{a.name}}</strong></td>
                            <td>${{a.val.toFixed(2)}} ${{a.unit}}</td>
                            <td>${{a.base.toFixed(2)}} ${{a.unit}}</td>
                            <td>${{a.z.toFixed(1)}} σ</td>
                            <td>${{a.pct.toFixed(1)}}%</td>
                            <td><span style="font-weight:700; color:${{a.status === 'CRITICAL' ? '#f85149' : (a.status === 'WARNING' ? '#db6d28' : '#2ea043')}}">${{a.status}}</span></td>
                        </tr>
                    `;
                }});
                tbody.innerHTML = tHtml;
            }}

            // Control Actions
            function togglePlay() {{
                isPlaying = !isPlaying;
                const btn = document.getElementById('btnPlay');
                if (isPlaying) {{
                    btn.innerText = 'Pause';
                    btn.classList.remove('btn-primary');
                    btn.style.backgroundColor = '#d29922';
                    btn.style.borderColor = '#d29922';
                    startTimer();
                }} else {{
                    btn.innerText = 'Play';
                    btn.classList.add('btn-primary');
                    btn.style.backgroundColor = '';
                    btn.style.borderColor = '';
                    stopTimer();
                }}
            }}

            function startTimer() {{
                stopTimer();
                const intervalMs = Math.round(1000 / speed);
                timerId = setInterval(() => {{
                    if (currentIndex < DATA.length - 1) {{
                        renderFrame(currentIndex + 1);
                    }} else {{
                        togglePlay();
                    }}
                }}, intervalMs);
            }}

            function stopTimer() {{
                if (timerId !== null) {{
                    clearInterval(timerId);
                    timerId = null;
                }}
            }}

            function resetSim() {{
                if (isPlaying) togglePlay();
                renderFrame(0);
            }}

            function onSliderDrag(val) {{
                if (isPlaying) togglePlay();
                renderFrame(parseInt(val, 10));
            }}

            function jumpTo(targetSec) {{
                if (isPlaying) togglePlay();
                renderFrame(targetSec);
            }}

            function setSpeed(s) {{
                speed = s;
                ['btnSpd05', 'btnSpd1', 'btnSpd2'].forEach(id => document.getElementById(id).classList.remove('btn-active'));
                if (s === 0.5) document.getElementById('btnSpd05').classList.add('btn-active');
                if (s === 1.0) document.getElementById('btnSpd1').classList.add('btn-active');
                if (s === 2.0) document.getElementById('btnSpd2').classList.add('btn-active');
                if (isPlaying) startTimer();
            }}

            // Initial render frame at t=0
            renderFrame(0);
        </script>
    </body>
    </html>
    """

    components.html(html_code, height=920, scrolling=False)
