"""
Machine Learning and Risk Engine for MachineMind.
Combines Scikit-learn's Isolation Forest with a rolling statistical baseline
and Z-score deviation tracker to deliver robust, unsupervised industrial anomaly detection.
Produces continuous health scores (0-100), categorical inspection risk indexes,
and persistent hysteresis-managed alerts.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from config import (
    MODELS_DIR,
    FEATURE_COLUMNS,
    SENSOR_CONFIGS,
    ML_CONFIG,
    RISK_CATEGORIES,
    ALERT_PERSISTENCE_TICKS,
    ALERT_RECOVERY_TICKS,
)
from data_generator import generate_scenario_telemetry


@dataclass
class AnomalyInferenceResult:
    """Encapsulates detailed inference diagnostics for a single telemetry reading."""
    timestamp: datetime
    is_anomaly: bool
    raw_anomaly_score: float         # Isolation forest decision function value
    normalized_anomaly_score: float  # [0, 1] scaled score (higher = more anomalous)
    max_z_score: float               # Highest absolute Z-score among all sensors
    feature_z_scores: Dict[str, float]
    feature_deviations: Dict[str, float]
    feature_values: Dict[str, float]
    health_score: float              # [0, 100] scale
    risk_category: str               # 'Low', 'Moderate', 'Elevated', 'Critical'
    risk_score: float                # Continuous heuristic risk indicator [0.0 - 1.0]
    breached_thresholds: List[str]   # Any sensors that crossed warning or critical limits
    model_backend: str               # 'IsolationForest (CPU Scikit-Learn)' or fallback


@dataclass
class AlertRecord:
    """Represents a discrete equipment alert with state tracking."""
    alert_id: str
    timestamp: str
    level: str                       # 'INFORMATIONAL', 'WARNING', 'CRITICAL', 'RESOLVED'
    triggering_sensors: List[str]
    reason: str
    health_score: float
    status: str                      # 'ACTIVE', 'ACKNOWLEDGED', 'RESOLVED'
    details: Dict[str, Any] = field(default_factory=dict)


class PredictiveMaintenanceModel:
    """
    Unsupervised predictive maintenance model combining Isolation Forest with
    rolling multi-sensor statistical deviation modeling.
    """
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or (MODELS_DIR / "isolation_forest_model.joblib")
        self.scaler: Optional[StandardScaler] = None
        self.isolation_forest: Optional[IsolationForest] = None
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
        self.is_trained: bool = False
        
        # State tracking for health score and alert hysteresis
        self.current_health: float = 100.0
        self.consecutive_anomalies: int = 0
        self.consecutive_normals: int = 0
        self.active_alert: Optional[AlertRecord] = None
        self.alert_history: List[AlertRecord] = []
        self._alert_counter: int = 0

    def fit_baseline(self, normal_df: Optional[pd.DataFrame] = None) -> "PredictiveMaintenanceModel":
        """
        Trains the unsupervised Isolation Forest and computes sensor baselines
        using representative normal-operation telemetry (Scenario A).
        """
        if normal_df is None or normal_df.empty:
            # Generate synthetic normal baseline dataset
            normal_df = generate_scenario_telemetry(
                scenario_id="scenario_a",
                duration_seconds=ML_CONFIG["training_samples"],
                seed=ML_CONFIG["random_seed"]
            )

        X = normal_df[FEATURE_COLUMNS].copy()

        # 1. Compute empirical baseline statistics (mean, std, min, max, median)
        self.baseline_stats = {}
        for col in FEATURE_COLUMNS:
            mean_val = float(X[col].mean())
            std_val = float(X[col].std())
            # Ensure standard deviation is non-zero to prevent division by zero
            std_val = max(std_val, 1e-4)
            self.baseline_stats[col] = {
                "mean": mean_val,
                "std": std_val,
                "min": float(X[col].min()),
                "max": float(X[col].max()),
                "q25": float(X[col].quantile(0.25)),
                "q75": float(X[col].quantile(0.75)),
            }

        # 2. Fit Standard Scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 3. Fit Isolation Forest
        self.isolation_forest = IsolationForest(
            n_estimators=ML_CONFIG["n_estimators"],
            contamination=ML_CONFIG["contamination"],
            random_state=ML_CONFIG["random_seed"],
            max_samples=ML_CONFIG["max_samples"],
            n_jobs=-1
        )
        self.isolation_forest.fit(X_scaled)
        self.is_trained = True

        # Persist model artifacts
        self.save()
        return self

    def save(self, target_path: Optional[Path] = None):
        """Persists trained model and baseline statistics to disk."""
        path = target_path or self.model_path
        joblib.dump({
            "scaler": self.scaler,
            "isolation_forest": self.isolation_forest,
            "baseline_stats": self.baseline_stats,
            "feature_columns": FEATURE_COLUMNS,
        }, path)

    def load(self, source_path: Optional[Path] = None) -> bool:
        """Loads fitted model state from disk if available."""
        path = source_path or self.model_path
        if not path.exists():
            return False
        try:
            state = joblib.load(path)
            self.scaler = state.get("scaler")
            self.isolation_forest = state.get("isolation_forest")
            self.baseline_stats = state.get("baseline_stats", {})
            self.is_trained = self.scaler is not None and self.isolation_forest is not None
            return self.is_trained
        except Exception:
            self.is_trained = False
            return False

    def predict_reading(self, reading: pd.Series, instantaneous: bool = False) -> AnomalyInferenceResult:
        """
        Executes sequential inference on an incoming telemetry record.
        Combines statistical Z-scores with Isolation Forest score and evaluates
        operating limits and alert hysteresis.
        """
        if not self.is_trained:
            # Automatic on-demand fitting if not yet trained
            self.fit_baseline()

        timestamp = reading.get("timestamp", datetime.now())
        if isinstance(timestamp, str):
            try:
                timestamp = pd.to_datetime(timestamp)
            except Exception:
                timestamp = datetime.now()

        # Extract sensor values
        sensor_values: Dict[str, float] = {}
        deviations: Dict[str, float] = {}
        z_scores: Dict[str, float] = {}
        breached_thresholds: List[str] = []

        for col in FEATURE_COLUMNS:
            val = float(reading.get(col, SENSOR_CONFIGS[col].baseline_mean))
            sensor_values[col] = val
            base_mean = self.baseline_stats.get(col, {}).get("mean", SENSOR_CONFIGS[col].baseline_mean)
            base_std = self.baseline_stats.get(col, {}).get("std", SENSOR_CONFIGS[col].baseline_std)
            base_std = max(base_std, 1e-4)

            dev = val - base_mean
            deviations[col] = dev
            z_score = abs(dev) / base_std
            z_scores[col] = z_score

            # Check threshold breaches
            cfg = SENSOR_CONFIGS[col]
            if col == "pressure":
                if val <= cfg.critical_threshold:
                    breached_thresholds.append(f"{col}:CRITICAL_LOW ({val:.2f} <= {cfg.critical_threshold})")
                elif val <= cfg.warning_threshold:
                    breached_thresholds.append(f"{col}:WARNING_LOW ({val:.2f} <= {cfg.warning_threshold})")
            elif col == "speed":
                if val <= cfg.critical_threshold:
                    breached_thresholds.append(f"{col}:CRITICAL_SPEED ({val:.1f} <= {cfg.critical_threshold})")
                elif val <= cfg.warning_threshold:
                    breached_thresholds.append(f"{col}:WARNING_SPEED ({val:.1f} <= {cfg.warning_threshold})")
            else:
                if val >= cfg.critical_threshold:
                    breached_thresholds.append(f"{col}:CRITICAL_HIGH ({val:.2f} >= {cfg.critical_threshold})")
                elif val >= cfg.warning_threshold:
                    breached_thresholds.append(f"{col}:WARNING_HIGH ({val:.2f} >= {cfg.warning_threshold})")

        max_z = max(z_scores.values())

        # Model Inference via Isolation Forest
        try:
            X_input = pd.DataFrame([[sensor_values[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
            X_scaled = self.scaler.transform(X_input)
            raw_score = float(self.isolation_forest.decision_function(X_scaled)[0])
            pred_label = int(self.isolation_forest.predict(X_scaled)[0])  # -1 = anomaly, 1 = normal
            
            # Map raw decision function score (typically ~ -0.3 to +0.2) to normalized anomaly intensity [0, 1]
            # Lower decision function means more anomalous
            norm_anomaly_score = float(np.clip((0.15 - raw_score) / 0.40, 0.0, 1.0))
            is_ml_anomaly = pred_label == -1
            backend_label = "IsolationForest (CPU Scikit-Learn)"
        except Exception:
            # Fallback heuristic if ML prediction fails
            raw_score = 0.0
            norm_anomaly_score = float(np.clip((max_z - 1.5) / 4.0, 0.0, 1.0))
            is_ml_anomaly = max_z >= ML_CONFIG["z_score_threshold"]
            backend_label = "Heuristic Fallback (Statistical Z-Score)"

        # Combined anomaly criterion: ML flag OR high statistical Z-score OR threshold breach
        is_anomaly = is_ml_anomaly or (max_z >= ML_CONFIG["z_score_threshold"]) or (len(breached_thresholds) > 0)

        # ---------------------------------------------------------------------
        # Machine Health Score & Risk Calculation
        # ---------------------------------------------------------------------
        if instantaneous:
            # Immediate responsive calculation for custom operator tuning
            if not is_anomaly:
                self.current_health = 100.0
                self.consecutive_anomalies = 0
                self.consecutive_normals = ALERT_RECOVERY_TICKS + 1
                risk_score = 0.0
                risk_category = RISK_CATEGORIES["LOW"]
                if self.active_alert is not None:
                    self.active_alert.status = "RESOLVED"
                    self.active_alert = None
            else:
                self.consecutive_normals = 0
                self.consecutive_anomalies = ALERT_PERSISTENCE_TICKS + 1
                if any("CRITICAL" in b for b in breached_thresholds):
                    self.current_health = max(10.0, 30.0 - 5.0 * max(0.0, max_z - 4.0))
                    risk_score = 0.95
                    risk_category = RISK_CATEGORIES["CRITICAL"]
                elif any("WARNING" in b for b in breached_thresholds):
                    self.current_health = max(35.0, 60.0 - 8.0 * max(0.0, max_z - 2.5))
                    risk_score = 0.68
                    risk_category = RISK_CATEGORIES["ELEVATED"]
                elif max_z >= 2.0 or is_ml_anomaly:
                    self.current_health = max(60.0, 80.0 - 10.0 * max(0.0, max_z - 1.5))
                    risk_score = 0.40
                    risk_category = RISK_CATEGORIES["MODERATE"]
                else:
                    self.current_health = 90.0
                    risk_score = 0.15
                    risk_category = RISK_CATEGORIES["LOW"]
        else:
            # Dynamic decaying / recovery formula for continuous time series:
            if is_anomaly:
                severity_factor = max(norm_anomaly_score, (max_z - 1.5) / 3.0)
                severity_factor = max(severity_factor, 0.25)
                # Critical breaches accelerate health degradation
                if any("CRITICAL" in b for b in breached_thresholds):
                    severity_factor = max(severity_factor, 0.85)
                elif any("WARNING" in b for b in breached_thresholds):
                    severity_factor = max(severity_factor, 0.50)

                decay = 18.0 * severity_factor * ML_CONFIG["health_decay_rate"]
                self.current_health = max(0.0, self.current_health - decay)
                self.consecutive_anomalies += 1
                self.consecutive_normals = 0
            else:
                recovery = 6.0 * ML_CONFIG["health_recovery_rate"]
                self.current_health = min(100.0, self.current_health + recovery)
                self.consecutive_normals += 1
                self.consecutive_anomalies = 0

            # Heuristic Risk Score [0.0, 1.0]
            # Inversely related to health, weighted by active threshold breaches
            base_risk = (100.0 - self.current_health) / 100.0
            if any("CRITICAL" in b for b in breached_thresholds):
                risk_score = max(base_risk, 0.90)
            elif any("WARNING" in b for b in breached_thresholds):
                risk_score = max(base_risk, 0.65)
            else:
                risk_score = base_risk

            # Categorical Risk Classification
            if risk_score >= 0.80 or any("CRITICAL" in b for b in breached_thresholds) or self.current_health < 35.0:
                risk_category = RISK_CATEGORIES["CRITICAL"]
            elif risk_score >= 0.55 or any("WARNING" in b for b in breached_thresholds) or self.current_health < 60.0:
                risk_category = RISK_CATEGORIES["ELEVATED"]
            elif risk_score >= 0.30 or max_z >= 2.0 or self.current_health < 80.0:
                risk_category = RISK_CATEGORIES["MODERATE"]
            else:
                risk_category = RISK_CATEGORIES["LOW"]

        # ---------------------------------------------------------------------
        # Alert Management with Persistence & Hysteresis
        # ---------------------------------------------------------------------
        self._update_alert_state(
            timestamp=timestamp,
            is_anomaly=is_anomaly,
            risk_category=risk_category,
            breached_thresholds=breached_thresholds,
            z_scores=z_scores,
            sensor_values=sensor_values
        )

        return AnomalyInferenceResult(
            timestamp=timestamp,
            is_anomaly=is_anomaly,
            raw_anomaly_score=round(raw_score, 4),
            normalized_anomaly_score=round(norm_anomaly_score, 3),
            max_z_score=round(max_z, 2),
            feature_z_scores={k: round(v, 2) for k, v in z_scores.items()},
            feature_deviations={k: round(v, 3) for k, v in deviations.items()},
            feature_values={k: round(v, 3) for k, v in sensor_values.items()},
            health_score=round(self.current_health, 1),
            risk_category=risk_category,
            risk_score=round(risk_score, 3),
            breached_thresholds=breached_thresholds,
            model_backend=backend_label,
        )

    def _update_alert_state(
        self,
        timestamp: datetime,
        is_anomaly: bool,
        risk_category: str,
        breached_thresholds: List[str],
        z_scores: Dict[str, float],
        sensor_values: Dict[str, float]
    ):
        """Maintains stateful alert hysteresis and handles creation/resolution."""
        time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

        # Determine triggering sensors (Z > 2.0 or threshold breach)
        triggering = [
            sensor for sensor, z in z_scores.items()
            if z >= 2.0 or any(sensor in b for b in breached_thresholds)
        ]

        if is_anomaly:
            if self.consecutive_anomalies >= ALERT_PERSISTENCE_TICKS:
                alert_level = "CRITICAL" if risk_category == RISK_CATEGORIES["CRITICAL"] else "WARNING"
                
                # Check if we should create a new alert or escalate existing
                if self.active_alert is None or self.active_alert.status == "RESOLVED":
                    self._alert_counter += 1
                    alert_id = f"ALT-{datetime.now().strftime('%Y%m%d')}-{self._alert_counter:04d}"
                    reason = self._construct_alert_reason(triggering, breached_thresholds, z_scores)
                    
                    self.active_alert = AlertRecord(
                        alert_id=alert_id,
                        timestamp=time_str,
                        level=alert_level,
                        triggering_sensors=triggering,
                        reason=reason,
                        health_score=round(self.current_health, 1),
                        status="ACTIVE",
                        details={"sensor_values": sensor_values, "z_scores": z_scores}
                    )
                    self.alert_history.append(self.active_alert)
                elif self.active_alert.level == "WARNING" and alert_level == "CRITICAL":
                    # Escalate active alert level
                    self.active_alert.level = "CRITICAL"
                    self.active_alert.reason = self._construct_alert_reason(triggering, breached_thresholds, z_scores)
        else:
            if self.consecutive_normals >= ALERT_RECOVERY_TICKS and self.active_alert is not None:
                if self.active_alert.status == "ACTIVE":
                    self.active_alert.status = "RESOLVED"
                    self.active_alert = None

    def _construct_alert_reason(
        self,
        triggering: List[str],
        breached_thresholds: List[str],
        z_scores: Dict[str, float]
    ) -> str:
        """Constructs human-readable alert justification."""
        if not triggering and not breached_thresholds:
            return "Unsupervised anomaly pattern detected across multi-sensor correlation baseline."
        
        parts = []
        if breached_thresholds:
            parts.append(f"Threshold limit breached: {', '.join(breached_thresholds)}")
        if triggering:
            z_strs = [f"{s} (Z={z_scores.get(s, 0.0):.1f})" for s in triggering]
            parts.append(f"Significant sensor deviations: {', '.join(z_strs)}")
        return "; ".join(parts)

    def acknowledge_active_alert(self):
        """Allows operator to acknowledge current alert."""
        if self.active_alert and self.active_alert.status == "ACTIVE":
            self.active_alert.status = "ACKNOWLEDGED"

    def reset_state(self):
        """Resets dynamic tracking state (health, alerts, counters)."""
        self.current_health = 100.0
        self.consecutive_anomalies = 0
        self.consecutive_normals = 0
        self.active_alert = None
        self.alert_history = []


def evaluate_model_on_dataset(
    model: PredictiveMaintenanceModel,
    df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Computes rigorous retrospective evaluation metrics across an entire scenario dataset:
      - Anomaly detection rate
      - False alarm rate under normal operation
      - Precision, Recall, F1 Score (where ground-truth labels exist)
      - Warning lead time (for Scenario B)
      - Detection latency
    """
    model.reset_state()
    predictions = []
    
    for _, row in df.iterrows():
        res = model.predict_reading(row)
        predictions.append(res)

    results_df = pd.DataFrame([asdict(p) for p in predictions])
    results_df["ground_truth_fault"] = df["fault_state"].values != "NONE"
    
    # 1. Anomaly detection rate (Sensitivity / Recall on faults)
    fault_mask = results_df["ground_truth_fault"] == True
    num_faults = int(fault_mask.sum())
    if num_faults > 0:
        true_positives = int((results_df.loc[fault_mask, "is_anomaly"] == True).sum())
        anomaly_detection_rate = true_positives / num_faults
    else:
        anomaly_detection_rate = None

    # 2. False alarm rate under known normal conditions
    normal_mask = results_df["ground_truth_fault"] == False
    num_normals = int(normal_mask.sum())
    if num_normals > 0:
        false_positives = int((results_df.loc[normal_mask, "is_anomaly"] == True).sum())
        false_alarm_rate = false_positives / num_normals
    else:
        false_alarm_rate = None

    # 3. Precision & F1
    total_flagged = int((results_df["is_anomaly"] == True).sum())
    if total_flagged > 0 and num_faults > 0:
        precision = true_positives / total_flagged
        recall = anomaly_detection_rate
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    else:
        precision = None
        recall = anomaly_detection_rate
        f1 = None

    # 4. Warning Lead Time for Bearing Degradation
    # Computed strictly from real timestamps/ticks in the dataset
    lead_time_seconds = None
    warning_tick = None
    critical_tick = None

    # Find first vibration >= 5.0 and first vibration >= 7.0
    vibs = df["vibration"].values
    warn_indices = np.where(vibs >= SENSOR_CONFIGS["vibration"].warning_threshold)[0]
    crit_indices = np.where(vibs >= SENSOR_CONFIGS["vibration"].critical_threshold)[0]
    if len(warn_indices) > 0 and len(crit_indices) > 0:
        warning_tick = int(warn_indices[0])
        critical_tick = int(crit_indices[0])
        lead_time_seconds = max(0, critical_tick - warning_tick)

    # 5. Detection latency (ticks from fault injection to first detection)
    detection_latency = None
    if num_faults > 0:
        fault_start_idx = int(np.where(fault_mask)[0][0])
        subsequent_anomalies = np.where((results_df["is_anomaly"] == True) & (results_df.index >= fault_start_idx))[0]
        if len(subsequent_anomalies) > 0:
            first_detection_idx = int(subsequent_anomalies[0])
            detection_latency = max(0, first_detection_idx - fault_start_idx)

    return {
        "total_records": len(df),
        "total_anomalies_detected": total_flagged,
        "anomaly_detection_rate": round(anomaly_detection_rate, 4) if anomaly_detection_rate is not None else "N/A",
        "false_alarm_rate": round(false_alarm_rate, 4) if false_alarm_rate is not None else "N/A",
        "precision": round(precision, 4) if precision is not None else "N/A",
        "recall": round(recall, 4) if recall is not None else "N/A",
        "f1_score": round(f1, 4) if f1 is not None else "N/A",
        "warning_lead_time_seconds": lead_time_seconds if lead_time_seconds is not None else "N/A",
        "warning_tick": warning_tick,
        "critical_tick": critical_tick,
        "detection_latency_ticks": detection_latency if detection_latency is not None else "N/A",
        "final_health_score": round(model.current_health, 1),
    }
