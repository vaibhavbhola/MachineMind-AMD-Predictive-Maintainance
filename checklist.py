"""
Prescriptive Maintenance Recommendation Engine for MachineMind.
Maps multi-sensor abnormal patterns to practical, safe, prioritized industrial inspection procedures.
Provides interactive checklists, rate-of-change degradation analytics, machine factors, and actionable how-to-fix playbooks.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from config import SENSOR_CONFIGS, DEFAULT_MACHINE_ID, EQUIPMENT_TYPE, FACILITY_LOCATION
from model import AnomalyInferenceResult


@dataclass
class InspectionStep:
    step_id: str
    order: int
    title: str
    description: str
    is_critical: bool = False
    completed: bool = False


@dataclass
class MaintenanceRecommendation:
    pattern_id: str
    pattern_name: str
    priority: str                     # 'P1 - CRITICAL', 'P2 - URGENT', 'P3 - STANDARD', 'P4 - ROUTINE'
    severity: str                     # 'CRITICAL', 'WARNING', 'INFORMATIONAL', 'NORMAL'
    primary_reason: str
    relevant_sensors: List[str]
    safe_order_of_operations: List[str]
    inspection_steps: List[InspectionStep]
    escalation_notes: Optional[str]
    safety_advisory: str
    uncertainty_disclaimer: str
    rate_of_change_summary: str = ""
    elevation_details: List[Dict[str, Any]] = field(default_factory=list)
    machine_factors: Dict[str, str] = field(default_factory=dict)
    actionable_fix_guide: Dict[str, Any] = field(default_factory=dict)


# Standard safety protocols mandatory for industrial rotating machinery
DEFAULT_SAFETY_ADVISORY = (
    "SAFETY FIRST: Always execute facility Lockout/Tagout (LOTO) protocols and verify zero mechanical "
    "and electrical energy state before opening junction boxes, removing guards, or touching rotating shafts."
)

DEFAULT_UNCERTAINTY_DISCLAIMER = (
    "DISCLAIMER: MachineMind provides advisory decision support based on statistical telemetry deviations. "
    "Sensor anomalies indicate abnormal behavior but do not constitute a definitive physical diagnosis. "
    "Qualified maintenance personnel must perform on-site verification."
)


def get_recommendation_for_telemetry(
    inference_result: AnomalyInferenceResult,
    operator_checklist_state: Optional[Dict[str, bool]] = None,
    rates_of_change: Optional[Dict[str, float]] = None,
    history_window_df: Optional[pd.DataFrame] = None,
    **kwargs
) -> MaintenanceRecommendation:
    """
    Evaluates sensor deviations, rate of change, and machine parameters to return
    a prioritized maintenance recommendation, interactive checklists, and a concrete fix playbook.
    """
    if operator_checklist_state is None:
        operator_checklist_state = {}

    vals = inference_result.feature_values
    temp = vals.get("temperature", SENSOR_CONFIGS["temperature"].baseline_mean)
    vib = vals.get("vibration", SENSOR_CONFIGS["vibration"].baseline_mean)
    press = vals.get("pressure", SENSOR_CONFIGS["pressure"].baseline_mean)
    speed = vals.get("speed", SENSOR_CONFIGS["speed"].baseline_mean)

    # -------------------------------------------------------------------------
    # 1. Rate of Change / Velocity Analysis (d/dt)
    # -------------------------------------------------------------------------
    if rates_of_change is None:
        rates_of_change = {}
        if history_window_df is not None and len(history_window_df) >= 3:
            for skey in ["temperature", "vibration", "pressure", "speed"]:
                if skey in history_window_df.columns:
                    s_series = history_window_df[skey].dropna()
                    if len(s_series) >= 2:
                        dt = max(1, len(s_series) - 1)
                        rates_of_change[skey] = float(s_series.iloc[-1] - s_series.iloc[0]) / float(dt)

    temp_slope = rates_of_change.get("temperature", 0.0)
    vib_slope = rates_of_change.get("vibration", 0.0)
    press_slope = rates_of_change.get("pressure", 0.0)
    speed_slope = rates_of_change.get("speed", 0.0)

    # Velocity classification
    if temp_slope > 0.35 or press_slope < -0.06:
        rate_summary = (
            f"RAPID RUNAWAY VELOCITY: Temperature rising at +{temp_slope:.2f}°C/s, "
            f"Pressure dropping at {press_slope:.2f} bar/s. Severe fast-onset failure in progress. "
            "Controlled emergency shutdown required within 30-60 seconds."
        )
    elif vib_slope > 0.07:
        rate_summary = (
            f"ACCELERATING MECHANICAL SEVERITY: Vibration rate +{vib_slope:.3f} mm/s/s. "
            "Dynamic unbalance or bearing spalling is accelerating rapidly. Inspect within current hour."
        )
    elif vib_slope > 0.015:
        rate_summary = (
            f"PROGRESSIVE WEAR SLOPE: Vibration rate +{vib_slope:.3f} mm/s/s. "
            "Steady subsurface fatigue propagation (~45s lead time before critical threshold). Plan bearing maintenance."
        )
    elif abs(temp_slope) > 0.05 or abs(press_slope) > 0.02:
        rate_summary = (
            f"MODERATE DRIFT: Thermal velocity {temp_slope:+.2f}°C/s, "
            f"Pressure velocity {press_slope:+.2f} bar/s. Continuous operating deviation detected."
        )
    else:
        rate_summary = "STABLE OPERATING VELOCITY: Telemetry slope is near-zero (d/dt ~= 0). No rapid divergence detected."

    # -------------------------------------------------------------------------
    # 2. Elevation & Severity Details for Individual Sensors
    # -------------------------------------------------------------------------
    elevation_details = []
    for skey, cfg in SENSOR_CONFIGS.items():
        val = vals.get(skey, cfg.baseline_mean)
        dev = val - cfg.baseline_mean
        z = inference_result.feature_z_scores.get(skey, dev / max(cfg.baseline_std, 1e-3))
        
        if skey in ["temperature", "vibration"]:
            if val >= cfg.critical_threshold:
                status = "CRITICAL (Limit Breached)"
            elif val >= cfg.warning_threshold:
                status = "WARNING (Limit Breached)"
            elif abs(z) >= 2.0:
                status = "ELEVATED (Statistical Drift)"
            else:
                status = "NORMAL (Nominal)"
        elif skey == "pressure":
            if val <= cfg.critical_threshold:
                status = "CRITICAL (Severe Pressure Loss)"
            elif val <= cfg.warning_threshold:
                status = "WARNING (Low Pressure Warning)"
            elif abs(z) >= 2.0:
                status = "ELEVATED (Under-Pressure Drift)"
            else:
                status = "NORMAL (Nominal)"
        else:  # speed
            if val <= cfg.critical_threshold:
                status = "CRITICAL (Severe Speed Loss)"
            elif val <= cfg.warning_threshold:
                status = "WARNING (Speed Drop)"
            elif abs(z) >= 2.0:
                status = "ELEVATED (Rotational Slip)"
            else:
                status = "NORMAL (Nominal)"

        slope_val = rates_of_change.get(skey, 0.0)
        elevation_details.append({
            "sensor_id": skey,
            "sensor_name": cfg.name,
            "current_value": round(float(val), 2),
            "baseline_mean": round(float(cfg.baseline_mean), 2),
            "deviation": round(float(dev), 2),
            "z_score": round(float(z), 2),
            "unit": cfg.unit,
            "status": status,
            "rate_of_change": round(float(slope_val), 3)
        })

    # -------------------------------------------------------------------------
    # 3. Machine Asset Context & Engineering Specifications
    # -------------------------------------------------------------------------
    machine_factors = {
        "Equipment Type": "Centrifugal Single-Stage Pump coupled to AC Induction Motor",
        "Motor Power & Rating": "45 kW (60 HP), 4-Pole, 480V 3-Phase, 60 Hz, 68A FLA, 1780 RPM rated",
        "Drive-End Bearing": "SKF 6312-2Z Deep-Groove Ball Bearing (Radial Load Support)",
        "Non-Drive-End Bearing": "SKF 6210 Single-Row Deep-Groove Ball Bearing",
        "Lubrication Specification": "Mobil Polyrex EM (Polyurea NLGI-2) / ISO VG 46 Synthetic Turbine Oil",
        "Shaft Mechanical Seal": "Cartridge Balanced Mechanical Seal (Silicon Carbide vs Carbon Faces)",
        "Shaft Flexible Coupling": "Falk Steelflex / Grid Pack flexible coupling (Max angular runout: 0.05 mm)",
        "Operating Environment": "Industrial Feedwater Subsystem (Continuous duty, 40°C max ambient)"
    }

    # Threshold checks
    vib_high = vib >= SENSOR_CONFIGS["vibration"].warning_threshold
    vib_crit = vib >= SENSOR_CONFIGS["vibration"].critical_threshold
    temp_high = temp >= SENSOR_CONFIGS["temperature"].warning_threshold
    temp_crit = temp >= SENSOR_CONFIGS["temperature"].critical_threshold
    press_low = press <= SENSOR_CONFIGS["pressure"].warning_threshold
    press_crit = press <= SENSOR_CONFIGS["pressure"].critical_threshold
    speed_low = speed <= SENSOR_CONFIGS["speed"].warning_threshold

    # -------------------------------------------------------------------------
    # Pattern 1: High Vibration with Normal/Mild Temperature (Bearing Wear)
    # -------------------------------------------------------------------------
    if vib_high and not temp_high:
        priority = "P1 - CRITICAL" if vib_crit else "P2 - URGENT"
        severity = "CRITICAL" if vib_crit else "WARNING"
        escalation = (
            "CRITICAL VIBRATION LEVEL (>7.0 mm/s RMS): Risk of catastrophic mechanical failure. "
            "Initiate controlled machine shutdown if vibration persists above 7.0 mm/s for >30s."
        ) if vib_crit else "Elevated vibration exceeding 5.0 mm/s RMS. Schedule physical inspection within current shift."

        steps = [
            InspectionStep(
                step_id="step_vib_1",
                order=1,
                title="Verify Vibration via Handheld Accelerometer",
                description="Cross-check fixed sensor reading at drive-end and non-drive-end bearing housings using calibrated handheld vibration meter.",
                is_critical=True
            ),
            InspectionStep(
                step_id="step_vib_2",
                order=2,
                title="Inspect Bearing Lubrication & Acoustic Emission",
                description="Examine grease/oil quality for metallic particulate contamination, discoloration, or emulsion. Perform ultrasonic acoustic check for rolling-element cage distress.",
                is_critical=True
            ),
            InspectionStep(
                step_id="step_vib_3",
                order=3,
                title="Check Shaft Alignment & Flexible Coupling",
                description="Inspect laser alignment between motor shaft and pump impeller shaft. Check elastomeric coupling insert for wear or angular runout.",
                is_critical=False
            ),
            InspectionStep(
                step_id="step_vib_4",
                order=4,
                title="Verify Foundation Mounting Bolts & Mechanical Looseness",
                description="Torque-check bedplate anchor bolts, motor foot fasteners, and structural baseplate for soft-foot condition or looseness.",
                is_critical=False
            )
        ]

        safe_order = [
            "1. Perform external acoustic and vibration cross-check while running.",
            "2. If readings exceed 7.0 mm/s, trip equipment to safe standby.",
            "3. Lock out / Tag out (LOTO) breaker 480V-M4B.",
            "4. Remove coupling guard and inspect mechanical alignment.",
            "5. Regrease bearings per OEM lubrication spec NLGI-2."
        ]

        fix_guide = {
            "immediate_action": "Reduce motor speed/load by 15-20% if process permits. Monitor vibration for 30s. If vibration crosses 7.0 mm/s RMS, initiate controlled machine stop immediately.",
            "diagnostic_verification": "Deploy portable FFT vibration analyzer on DE bearing housing. Look for dominant peaks at 1X (unbalance: check impeller fouling/cavitation), 2X (coupling misalignment: check laser alignment), or bearing ball pass frequencies (BPFO=3.58X, BPFI=5.42X for SKF 6312). Use ultrasound acoustic sensor to detect metal-to-metal rubbing.",
            "mechanical_fix_procedure": [
                "1. Execute 480V electrical LOTO at breaker MCC-4B.",
                "2. Remove coupling guard; measure shaft angular runout and radial offset with dial indicator or laser alignment tool.",
                "3. Inspect bearing grease condition: if dark, emulsified, or containing metallic debris, flush cavity and recharge with 15g of Mobil Polyrex EM grease.",
                "4. If bearing raceway roughness or pitting is detected during manual shaft spin, pull bearing using hydraulic 3-jaw puller; heat new SKF 6312-2Z bearing to 110°C on induction heater and press onto shaft shoulder.",
                "5. Re-torque baseplate anchor bolts to 120 Nm to eliminate soft-foot resonance."
            ],
            "electrical_fix_procedure": "Inspect motor ground bonding wire; verify no VFD common-mode shaft currents causing electrical discharge machining (EDM) fluting on bearing raceways.",
            "hydraulic_fix_procedure": "Verify pump suction pressure is above minimum NPSHa (Net Positive Suction Head) to prevent cavitation-induced impeller vibration.",
            "required_tools_parts": [
                "Replacement Bearing: SKF 6312-2Z",
                "Mobil Polyrex EM Grease (NLGI-2, Polyurea)",
                "Laser Shaft Alignment System (Prüftechnik or Easy-Laser)",
                "Hydraulic 3-Jaw Bearing Puller",
                "Portable Induction Bearing Heater",
                "Calibrated Torque Wrench (19mm socket, 120 Nm rating)"
            ],
            "estimated_mttr": "2.5 Hours (Alignment & Re-greasing) / 4.0 Hours (Bearing Replacement)"
        }

        return MaintenanceRecommendation(
            pattern_id="PATTERN_VIBRATION_BEARING",
            pattern_name="Bearing Degradation & Mechanical Vibration",
            priority=priority,
            severity=severity,
            primary_reason=f"Overall vibration ({vib:.2f} mm/s) significantly exceeds baseline envelope while temperature remains moderate ({temp:.1f}°C).",
            relevant_sensors=["vibration", "temperature"],
            safe_order_of_operations=safe_order,
            inspection_steps=_apply_completion(steps, operator_checklist_state),
            escalation_notes=escalation,
            safety_advisory=DEFAULT_SAFETY_ADVISORY,
            uncertainty_disclaimer=DEFAULT_UNCERTAINTY_DISCLAIMER,
            rate_of_change_summary=rate_summary,
            elevation_details=elevation_details,
            machine_factors=machine_factors,
            actionable_fix_guide=fix_guide
        )

    # -------------------------------------------------------------------------
    # Pattern 2: High Temperature with Pressure Drop (Coolant Leak)
    # -------------------------------------------------------------------------
    if temp_high and press_low:
        priority = "P1 - CRITICAL" if (temp_crit or press_crit) else "P2 - URGENT"
        severity = "CRITICAL" if (temp_crit or press_crit) else "WARNING"
        escalation = (
            "CRITICAL THERMAL/PRESSURE EVENT: Temperature > 95°C or Pressure < 2.8 bar. "
            "High risk of stator insulation breakdown, seal seizure, or runaway bearing welding. Immediate emergency shutdown recommended."
        ) if (temp_crit or press_crit) else "Coolant loss with thermal rise detected. Investigate fluid loops immediately."

        steps = [
            InspectionStep(
                step_id="step_cool_1",
                order=1,
                title="Inspect Coolant / Lubrication Fluid Level",
                description="Check external reservoir sight glass and reservoir telemetry. Look for active external hydraulic puddle or leaking seals.",
                is_critical=True
            ),
            InspectionStep(
                step_id="step_cool_2",
                order=2,
                title="Examine Cooling Flow & Heat Exchanger / Radiator",
                description="Verify heat exchanger coolant inlet/outlet delta-T, radiator fin blockage, and auxiliary circulating pump discharge.",
                is_critical=True
            ),
            InspectionStep(
                step_id="step_cool_3",
                order=3,
                title="Check Lube-Oil Pressure Relief Valve & Filters",
                description="Inspect differential pressure indicator across oil filters and check for stuck-open pressure relief bypass valve.",
                is_critical=False
            ),
            InspectionStep(
                step_id="step_cool_4",
                order=4,
                title="Inspect Mechanical Seals & Gasket Integrity",
                description="Inspect drive shaft mechanical seal cartridge for weeping, extrusion, or thermal cracking.",
                is_critical=False
            )
        ]

        safe_order = [
            "1. Verify pressure sensor reading against local analog Bourdon tube gauge.",
            "2. If pressure is below 2.8 bar or temperature is climbing through 95°C, trip unit immediately.",
            "3. LOTO electrical drive prior to approaching hydraulic piping.",
            "4. Inspect fluid reservoir and replace ruptured hoses or fittings.",
            "5. Refill coolant/lube fluid and pressure test prior to restart."
        ]

        fix_guide = {
            "immediate_action": "Immediate trip recommended if pressure is below 2.8 bar or temperature is climbing through 95°C to prevent mechanical seal seizure, stator winding breakdown, and permanent shaft scoring.",
            "diagnostic_verification": "Inspect mechanical seal gland and drain port for active steam or liquid weeping. Check differential pressure indicator across oil/coolant filter. Check fluid level in coolant reservoir.",
            "mechanical_fix_procedure": [
                "1. Isolate suction and discharge isolation valves and execute 480V electrical LOTO.",
                "2. Depressurize and drain cooling/lube loop into containment drum.",
                "3. Disassemble seal gland and inspect mechanical seal cartridge faces (check for thermal cracking, silicon carbide blistering, or Viton O-ring extrusion).",
                "4. Install replacement cartridge seal assembly with fresh Viton secondary seals.",
                "5. Flush and refill lube/coolant loop with ISO VG 46 synthetic oil, vent air from top casing plug, and hydro-test loop at 6.0 bar."
            ],
            "electrical_fix_procedure": "Conduct 1000V DC insulation resistance (Megger) test across motor stator windings phase-to-phase and phase-to-ground (minimum acceptance: 5 MΩ at 40°C).",
            "hydraulic_fix_procedure": "Check auxiliary coolant pump discharge head and inspect heat exchanger plate pack for scale clogging or restricted cooling water flow.",
            "required_tools_parts": [
                "Cartridge Mechanical Seal Kit (SiC/Carbon, Viton O-rings)",
                "ISO VG 46 Synthetic Turbine/Hydraulic Oil (20L Pail)",
                "Replacement Hydraulic Hoses (1/2\" SAE 100R2AT)",
                "1000V DC Megger Insulation Resistance Tester",
                "Bourdon Pressure Test Gauge (0-10 bar)"
            ],
            "estimated_mttr": "3.0 Hours (Seal & Hose Replacement)"
        }

        return MaintenanceRecommendation(
            pattern_id="PATTERN_COOLANT_THERMAL",
            pattern_name="Coolant Leak / Thermal Runaway",
            priority=priority,
            severity=severity,
            primary_reason=f"Acute hydraulic pressure drop ({press:.2f} bar) coupled with rapid thermal escalation ({temp:.1f}°C).",
            relevant_sensors=["temperature", "pressure"],
            safe_order_of_operations=safe_order,
            inspection_steps=_apply_completion(steps, operator_checklist_state),
            escalation_notes=escalation,
            safety_advisory=DEFAULT_SAFETY_ADVISORY,
            uncertainty_disclaimer=DEFAULT_UNCERTAINTY_DISCLAIMER,
            rate_of_change_summary=rate_summary,
            elevation_details=elevation_details,
            machine_factors=machine_factors,
            actionable_fix_guide=fix_guide
        )

    # -------------------------------------------------------------------------
    # Pattern 3: Speed Drop with High Temperature (Motor Overload)
    # -------------------------------------------------------------------------
    if speed_low and temp_high:
        priority = "P1 - CRITICAL" if temp_crit else "P2 - URGENT"
        severity = "CRITICAL" if temp_crit else "WARNING"
        escalation = "Motor slipping under high thermal load. Risk of stator winding burnout."

        steps = [
            InspectionStep(
                step_id="step_spd_1",
                order=1,
                title="Inspect Motor Drive Electrical Load & Phase Balance",
                description="Check variable frequency drive (VFD) output current, 3-phase current balance, and line voltage.",
                is_critical=True
            ),
            InspectionStep(
                step_id="step_spd_2",
                order=2,
                title="Examine Driven Machine for Mechanical Binding",
                description="Check for impeller jamming, foreign object obstruction in pump volute, or excessive process backpressure.",
                is_critical=True
            ),
            InspectionStep(
                step_id="step_spd_3",
                order=3,
                title="Perform Motor Stator Insulation Resistance Check",
                description="Megger stator windings phase-to-phase and phase-to-ground after isolation.",
                is_critical=False
            )
        ]

        safe_order = [
            "1. Check drive fault codes on local HMI.",
            "2. De-energize and LOTO motor circuit.",
            "3. Perform manual shaft turn-over by hand (feel for binding).",
            "4. Inspect motor ventilation fan cowl."
        ]

        fix_guide = {
            "immediate_action": "Throttle pump discharge throttle valve to reduce volumetric flow and pump hydraulic horsepower demand. Verify motor current does not exceed 68A FLA.",
            "diagnostic_verification": "Check VFD fault log (look for fault codes F0001 overcurrent, F0011 motor stall, or phase imbalance). Use true-RMS clamp meter to verify balanced phase currents (imbalance < 2%).",
            "mechanical_fix_procedure": [
                "1. De-energize and LOTO motor circuit breaker MCC-4B.",
                "2. Rotate pump and motor shafts by hand to verify free rotation and rule out mechanical binding or foreign object jamming in pump volute.",
                "3. Inspect external cooling fan cowl and stator fins; clear dust or debris matting restricting air velocity."
            ],
            "electrical_fix_procedure": "Inspect motor terminal junction box for loose terminal lugs, phase discoloration, or degraded crimp connections. Measure stator winding phase resistance with micro-ohmmeter.",
            "hydraulic_fix_procedure": "Check pumped liquid temperature and specific gravity against design pump curve to verify process fluid is not causing excessive hydraulic drag.",
            "required_tools_parts": [
                "True-RMS Clamp Meter (Fluke 376)",
                "Low-Resistance Micro-Ohmmeter",
                "1000V Insulation Tester",
                "Thermal Imaging Infrared Camera (FLIR)"
            ],
            "estimated_mttr": "1.5 Hours (Debris clearance / Electrical inspection)"
        }

        return MaintenanceRecommendation(
            pattern_id="PATTERN_SPEED_THERMAL",
            pattern_name="Motor Overload & Speed Slip",
            priority=priority,
            severity=severity,
            primary_reason=f"Rotational shaft speed drop ({speed:.1f} RPM) combined with elevated thermal readings ({temp:.1f}°C).",
            relevant_sensors=["speed", "temperature"],
            safe_order_of_operations=safe_order,
            inspection_steps=_apply_completion(steps, operator_checklist_state),
            escalation_notes=escalation,
            safety_advisory=DEFAULT_SAFETY_ADVISORY,
            uncertainty_disclaimer=DEFAULT_UNCERTAINTY_DISCLAIMER,
            rate_of_change_summary=rate_summary,
            elevation_details=elevation_details,
            machine_factors=machine_factors,
            actionable_fix_guide=fix_guide
        )

    # -------------------------------------------------------------------------
    # Pattern 4: High Temperature without Pressure Drop (Isolated Overheating)
    # -------------------------------------------------------------------------
    if temp_high:
        priority = "P1 - CRITICAL" if temp_crit else "P2 - URGENT"
        severity = "CRITICAL" if temp_crit else "WARNING"
        
        steps = [
            InspectionStep(
                step_id="step_thm_1",
                order=1,
                title="Inspect Motor Ventilation Cowl & Cooling Fins",
                description="Check for heavy dust accumulation, blocked airflow louvers, or damaged external cooling fan blades.",
                is_critical=True
            ),
            InspectionStep(
                step_id="step_thm_2",
                order=2,
                title="Check for Over-Greasing or Bearing Churning",
                description="Verify if recent relubrication occurred; excessive grease fill causes viscous friction and high temperatures.",
                is_critical=False
            )
        ]

        fix_guide = {
            "immediate_action": "Inspect cooling airflow and clean stator fins immediately. If temperature continues past 95°C, trip unit.",
            "diagnostic_verification": "Use infrared thermal camera to locate hotspot (stator core vs DE bearing vs NDE bearing). Check grease purge valve for over-greasing.",
            "mechanical_fix_procedure": [
                "1. Clean motor cooling cowl louvers and stator cooling fins with compressed air.",
                "2. Check if recent lubrication occurred: open grease relief port and run motor for 30 minutes to vent excess grease churn.",
                "3. Inspect cooling fan blades on motor non-drive end for broken or missing blades."
            ],
            "electrical_fix_procedure": "Verify supply voltage balance across phases (voltage imbalance > 1% can cause a 6-10% increase in stator temperature).",
            "hydraulic_fix_procedure": "Verify heat exchanger cooling medium flow rate.",
            "required_tools_parts": [
                "FLIR Thermal Imaging Camera",
                "Compressed Air Lance / Soft Bristle Brush",
                "Digital Multimeter"
            ],
            "estimated_mttr": "1.0 Hour"
        }

        return MaintenanceRecommendation(
            pattern_id="PATTERN_THERMAL_ONLY",
            pattern_name="High Temperature (Isolated)",
            priority=priority,
            severity=severity,
            primary_reason=f"Elevated operating temperature ({temp:.1f}°C) with nominal hydraulic pressure.",
            relevant_sensors=["temperature"],
            safe_order_of_operations=["1. Check airflow paths.", "2. Use infrared thermography to isolate hotspots."],
            inspection_steps=_apply_completion(steps, operator_checklist_state),
            escalation_notes="Sustained operation >95°C rapidly degrades bearing grease and stator insulation life.",
            safety_advisory=DEFAULT_SAFETY_ADVISORY,
            uncertainty_disclaimer=DEFAULT_UNCERTAINTY_DISCLAIMER,
            rate_of_change_summary=rate_summary,
            elevation_details=elevation_details,
            machine_factors=machine_factors,
            actionable_fix_guide=fix_guide
        )

    # -------------------------------------------------------------------------
    # Pattern 5: Pressure Instability / Drop without Thermal Rise
    # -------------------------------------------------------------------------
    if press_low:
        priority = "P2 - URGENT" if press_crit else "P3 - STANDARD"
        severity = "CRITICAL" if press_crit else "WARNING"
        
        steps = [
            InspectionStep(
                step_id="step_prs_1",
                order=1,
                title="Check Hydraulic Suction Strainer & In-line Filter",
                description="Inspect differential pressure gauge across suction filter for clogging or debris.",
                is_critical=True
            ),
            InspectionStep(
                step_id="step_prs_2",
                order=2,
                title="Inspect Pressure Regulating Valve & Bleed Ports",
                description="Check if regulator diaphragm has ruptured or relief valve is leaking back to reservoir.",
                is_critical=False
            )
        ]

        fix_guide = {
            "immediate_action": "Confirm suction shutoff valve is 100% open and locked. Monitor suction pressure gauge for cavitation threshold.",
            "diagnostic_verification": "Check differential pressure indicator across duplex suction basket filter. Inspect hydraulic circuit for air entrainment or pump suction vortexing.",
            "mechanical_fix_procedure": [
                "1. Switch flow to clean standby duplex filter basket.",
                "2. Remove and clean clogged suction strainer mesh; remove debris and scale deposits.",
                "3. Inspect pressure regulating valve diaphragm and relief valve seat for sticking or leakage."
            ],
            "electrical_fix_procedure": "Verify pressure transmitter 4-20mA loop integrity.",
            "hydraulic_fix_procedure": "Calculate Net Positive Suction Head Available (NPSHa) vs NPSHr. Bleed air from casing vent cock until liquid discharges without air bubbles.",
            "required_tools_parts": [
                "Replacement Filter Cartridge Element (10 micron)",
                "Suction Strainer Cover Gasket (EPDM)",
                "Calibrated Pressure Transmitter"
            ],
            "estimated_mttr": "1.0 Hour"
        }

        return MaintenanceRecommendation(
            pattern_id="PATTERN_PRESSURE_ONLY",
            pattern_name="Hydraulic Pressure Deviation",
            priority=priority,
            severity=severity,
            primary_reason=f"Low hydraulic/lube pressure ({press:.2f} bar) without accompanying thermal surge.",
            relevant_sensors=["pressure"],
            safe_order_of_operations=["1. Check analog pressure gauge.", "2. Inspect suction lines for air leaks."],
            inspection_steps=_apply_completion(steps, operator_checklist_state),
            escalation_notes=None,
            safety_advisory=DEFAULT_SAFETY_ADVISORY,
            uncertainty_disclaimer=DEFAULT_UNCERTAINTY_DISCLAIMER,
            rate_of_change_summary=rate_summary,
            elevation_details=elevation_details,
            machine_factors=machine_factors,
            actionable_fix_guide=fix_guide
        )

    # -------------------------------------------------------------------------
    # Pattern 6: Ambiguous / Statistical Anomaly (Unsupervised Detection)
    # -------------------------------------------------------------------------
    if inference_result.is_anomaly or inference_result.max_z_score >= 2.0:
        steps = [
            InspectionStep(
                step_id="step_gen_1",
                order=1,
                title="Perform Comprehensive Walkdown & Sensory Inspection",
                description="Check for abnormal noise, unusual odor, oil weeping around seals, or localized heating with an infrared camera.",
                is_critical=False
            ),
            InspectionStep(
                step_id="step_gen_2",
                order=2,
                title="Verify Sensor Calibration & Transmitter Wiring",
                description="Ensure transmitter signal integrity, verify grounding shields, and cross-check against reference gauges.",
                is_critical=False
            )
        ]

        fix_guide = {
            "immediate_action": "Increase sensor polling rate to 1 Hz. Monitor trend divergence across multi-sensor correlation matrix.",
            "diagnostic_verification": "Conduct complete walkdown: acoustic ultrasound inspection on bearing housings, thermal FLIR scan, and vibration overall check.",
            "mechanical_fix_procedure": [
                "1. Perform physical inspection of motor/pump skid.",
                "2. Check baseplate anchor bolt torque (120 Nm) to eliminate structural looseness.",
                "3. Verify coupling elastomer insert condition and shaft alignment."
            ],
            "electrical_fix_procedure": "Verify 4-20mA instrumentation loop calibration, verify cable shield grounding at control cabinet to eliminate electromagnetic interference (EMI).",
            "hydraulic_fix_procedure": "Check for minor cavitation noise (crackling marble sound) in pump suction elbow.",
            "required_tools_parts": ["Handheld Vibration Meter", "FLIR Thermal Camera", "Process Calibrator (4-20mA)"],
            "estimated_mttr": "1.0 Hour"
        }

        return MaintenanceRecommendation(
            pattern_id="PATTERN_STATISTICAL_ANOMALY",
            pattern_name="Multi-Sensor Statistical Anomaly",
            priority="P3 - STANDARD",
            severity="WARNING",
            primary_reason="Unsupervised anomaly detected. Multi-sensor cross-correlation deviates from learned normal baseline.",
            relevant_sensors=list(inference_result.feature_z_scores.keys()),
            safe_order_of_operations=["1. Review recent trends.", "2. Conduct visual and acoustic inspection."],
            inspection_steps=_apply_completion(steps, operator_checklist_state),
            escalation_notes="Root cause uncertain. Recommend comprehensive physical inspection before equipment distress accelerates.",
            safety_advisory=DEFAULT_SAFETY_ADVISORY,
            uncertainty_disclaimer=DEFAULT_UNCERTAINTY_DISCLAIMER,
            rate_of_change_summary=rate_summary,
            elevation_details=elevation_details,
            machine_factors=machine_factors,
            actionable_fix_guide=fix_guide
        )

    # -------------------------------------------------------------------------
    # Nominal Operation: Routine Monitoring
    # -------------------------------------------------------------------------
    steps = [
        InspectionStep(
            step_id="step_nom_1",
            order=1,
            title="Perform Scheduled Shift Walkdown",
            description="Log standard operating parameters into shift logbook. Check lube sight glass.",
            is_critical=False
        )
    ]

    fix_guide = {
        "immediate_action": "No corrective intervention required. Unit operating normally within nominal boundaries.",
        "diagnostic_verification": "Log telemetry in regular shift maintenance log. Verify normal oil sight glass level (middle of bullseye).",
        "mechanical_fix_procedure": ["Continue routine predictive monitoring."],
        "electrical_fix_procedure": "Routine monthly thermal scan on MCC bucket contacts.",
        "hydraulic_fix_procedure": "Normal discharge pressure and flow maintained.",
        "required_tools_parts": ["Standard Shift Logbook / Maintenance Tablet"],
        "estimated_mttr": "0.0 Hours"
    }

    return MaintenanceRecommendation(
        pattern_id="PATTERN_NORMAL",
        pattern_name="Nominal Operating Envelope",
        priority="P4 - ROUTINE",
        severity="NORMAL",
        primary_reason="All sensor streams conform to normal statistical operating envelope.",
        relevant_sensors=[],
        safe_order_of_operations=["1. Continue continuous monitoring."],
        inspection_steps=_apply_completion(steps, operator_checklist_state),
        escalation_notes=None,
        safety_advisory=DEFAULT_SAFETY_ADVISORY,
        uncertainty_disclaimer=DEFAULT_UNCERTAINTY_DISCLAIMER,
        rate_of_change_summary=rate_summary,
        elevation_details=elevation_details,
        machine_factors=machine_factors,
        actionable_fix_guide=fix_guide
    )


def _apply_completion(
    steps: List[InspectionStep],
    checklist_state: Dict[str, bool]
) -> List[InspectionStep]:
    """Updates completed attribute of each step based on session state."""
    for step in steps:
        step.completed = checklist_state.get(step.step_id, False)
    return steps
