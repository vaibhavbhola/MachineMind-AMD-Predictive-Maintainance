"""
AMD Hardware Acceleration & Execution Provider Management for MachineMind.
Provides honest runtime hardware probing, execution provider discovery,
safe ONNX Runtime session verification, CPU fallback, and deployment architectural blueprints
for AMD Ryzen AI (NPU/XDNA), ROCm (Radeon/Instinct), and industrial edge processors.
"""

import sys
import platform
import os
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from config import MODELS_DIR

# Optional import of ONNX Runtime
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    ort = None
    HAS_ONNX = False


@dataclass
class AMDHardwareStatus:
    os_name: str
    os_release: str
    cpu_architecture: str
    processor_name: str
    python_version: str
    is_amd_hardware: bool
    has_onnx_runtime: bool
    onnx_version: str
    available_providers: List[str]
    selected_provider: str
    hardware_accelerated: bool
    verification_status: str
    inference_backend_label: str
    benchmark_latency_ms: Optional[float]
    deployment_notes: str


def probe_system_hardware() -> Dict[str, Any]:
    """Inspects the local host platform, CPU, and OS environment."""
    os_name = platform.system()
    os_release = platform.release()
    arch = platform.machine()
    proc_name = platform.processor() or "Unknown Processor"
    
    # Check if processor string contains AMD
    is_amd = "amd" in proc_name.lower() or "ryzen" in proc_name.lower() or "epyc" in proc_name.lower()

    return {
        "os_name": os_name,
        "os_release": os_release,
        "cpu_architecture": arch,
        "processor_name": proc_name,
        "python_version": platform.python_version(),
        "is_amd_hardware": is_amd
    }


def verify_onnx_execution_provider(
    preferred_providers: Optional[List[str]] = None
) -> Tuple[str, bool, str, Optional[float]]:
    """
    Attempts to initialize an active ONNX Runtime session with candidate execution providers.
    Verifies that the session actually initializes and executes inference before claiming active status.

    Returns:
        (selected_provider, is_accelerated, status_message, latency_ms)
    """
    if not HAS_ONNX or ort is None:
        return (
            "None",
            False,
            "ONNX Runtime library is not installed. Native CPU Scikit-Learn active.",
            None
        )

    available = ort.get_available_providers()
    
    # Priority list of AMD acceleration backends:
    # 1. VitisAIExecutionProvider (AMD Ryzen AI NPU / XDNA)
    # 2. ROCMExecutionProvider (AMD ROCm on Linux / Radeon GPU)
    # 3. DmlExecutionProvider (DirectML on Windows for AMD GPUs)
    # 4. CPUExecutionProvider (Universal reference fallback)
    target_providers = preferred_providers or [
        "VitisAIExecutionProvider",
        "ROCMExecutionProvider",
        "DmlExecutionProvider",
        "CPUExecutionProvider"
    ]

    # Create or use a lightweight test ONNX model for verification
    test_model_path = ensure_onnx_model_exists()

    selected_provider = "CPUExecutionProvider"
    is_accelerated = False
    status_message = "CPU Fallback active."
    latency_ms = None

    for candidate in target_providers:
        if candidate in available:
            try:
                # Attempt to initialize session with candidate provider
                session_options = ort.SessionOptions()
                session_options.log_severity_level = 3  # Error only
                session = ort.InferenceSession(
                    str(test_model_path),
                    sess_options=session_options,
                    providers=[candidate]
                )
                
                # Execute test inference vector to prove actual execution
                dummy_input = np.array([[70.0, 2.2, 5.0, 1800.0]], dtype=np.float32)
                t0 = time.perf_counter()
                for _ in range(5):
                    _ = session.run(None, {"sensor_inputs": dummy_input})
                t1 = time.perf_counter()
                latency_ms = round(((t1 - t0) / 5.0) * 1000.0, 3)

                # Verified execution
                selected_provider = candidate
                if candidate in ("VitisAIExecutionProvider", "ROCMExecutionProvider"):
                    is_accelerated = True
                    status_message = f"AMD Hardware Accelerated session verified with {candidate}."
                elif candidate == "DmlExecutionProvider":
                    is_accelerated = True
                    status_message = f"DirectML GPU session verified with {candidate}."
                else:
                    is_accelerated = False
                    status_message = "Standard CPU Execution Provider verified and operational."
                break
            except Exception as e:
                # Initialization failed, try next provider
                continue

    return selected_provider, is_accelerated, status_message, latency_ms


def ensure_onnx_model_exists() -> Path:
    """
    Creates a minimal, compliant ONNX anomaly scoring model if not already present.
    Computes normalized distance score: y = sum(((x - mu) / sigma)^2).
    """
    model_path = MODELS_DIR / "machinemind_inference.onnx"
    if model_path.exists():
        return model_path

    # Build ONNX model bytes programmatically using standard ONNX operator nodes
    try:
        import onnx
        from onnx import helper, TensorProto

        # Input: sensor_inputs [1, 4] (temp, vib, press, speed)
        sensor_inputs = helper.make_tensor_value_info('sensor_inputs', TensorProto.FLOAT, [None, 4])
        # Output: anomaly_norm_score [1, 1]
        anomaly_output = helper.make_tensor_value_info('anomaly_score', TensorProto.FLOAT, [None, 1])

        # Baseline mean and standard deviation constants
        # Means: [70.0, 2.15, 5.02, 1800.0]
        # Inverted Stds: [1/2.2, 1/0.25, 1/0.18, 1/12.0] = [0.4545, 4.0, 5.555, 0.0833]
        means_tensor = helper.make_tensor('means', TensorProto.FLOAT, [4], [70.0, 2.15, 5.02, 1800.0])
        inv_stds_tensor = helper.make_tensor('inv_stds', TensorProto.FLOAT, [4], [0.4545, 4.0, 5.555, 0.0833])
        scale_tensor = helper.make_tensor('scale_div', TensorProto.FLOAT, [1], [4.0])

        # Graph nodes: (x - mean) -> mul inv_std -> square -> reduce_sum -> div 4.0
        sub_node = helper.make_node('Sub', ['sensor_inputs', 'means'], ['centered'])
        mul_node = helper.make_node('Mul', ['centered', 'inv_stds'], ['z_scores'])
        sq_node = helper.make_node('Mul', ['z_scores', 'z_scores'], ['z_sq'])
        axes_tensor = helper.make_tensor('axes', TensorProto.INT64, [1], [1])
        sum_node = helper.make_node('ReduceSum', ['z_sq', 'axes'], ['sum_sq'], keepdims=1)
        div_node = helper.make_node('Div', ['sum_sq', 'scale_div'], ['anomaly_score'])

        graph = helper.make_graph(
            [sub_node, mul_node, sq_node, sum_node, div_node],
            'MachineMindAnomalyScorer',
            [sensor_inputs],
            [anomaly_output],
            [means_tensor, inv_stds_tensor, scale_tensor, axes_tensor]
        )
        onnx_model = helper.make_model(graph, producer_name='MachineMind')
        onnx.save(onnx_model, str(model_path))
        return model_path
    except Exception:
        # If onnx package isn't present to build the model, write a fallback byte representation or handle cleanly
        pass
    return model_path


def get_amd_hardware_status() -> AMDHardwareStatus:
    """
    Returns complete, honest hardware and execution status for the dashboard.
    """
    sys_info = probe_system_hardware()
    onnx_avail = HAS_ONNX and ort is not None
    onnx_ver = ort.__version__ if onnx_avail else "Not Installed"
    providers = ort.get_available_providers() if onnx_avail else []

    selected_prov, is_accel, verif_msg, lat_ms = verify_onnx_execution_provider()

    # Determine honest label
    if is_accel and "VitisAI" in selected_prov:
        backend_label = "Inference Backend: AMD Ryzen AI NPU (Vitis AI) [Verified]"
    elif is_accel and "ROCM" in selected_prov:
        backend_label = "Inference Backend: AMD ROCm GPU [Verified]"
    elif is_accel and "Dml" in selected_prov:
        backend_label = "Inference Backend: DirectML GPU [Verified]"
    else:
        backend_label = "Inference Backend: CPU | AMD acceleration unavailable in this host environment"

    notes = (
        "Deployment Blueprint for AMD Hardware:\n"
        "1. AMD Ryzen AI (XDNA NPU): Target Windows 11 with AMD NPU driver and onnxruntime-vitisai package. "
        "Quantize feature autoencoder to INT8 for zero-CPU continuous edge telemetry scoring.\n"
        "2. AMD ROCm on Industrial Linux: Deploy on AMD Radeon PRO or Instinct accelerators using ROCMExecutionProvider "
        "for high-throughput multi-machine factory cluster monitoring.\n"
        "3. AMD EPYC / Embedded: Run multi-threaded Zen 4/5 CPU inference with AVX-512 VNNI acceleration."
    )

    return AMDHardwareStatus(
        os_name=sys_info["os_name"],
        os_release=sys_info["os_release"],
        cpu_architecture=sys_info["cpu_architecture"],
        processor_name=sys_info["processor_name"],
        python_version=sys_info["python_version"],
        is_amd_hardware=sys_info["is_amd_hardware"],
        has_onnx_runtime=onnx_avail,
        onnx_version=onnx_ver,
        available_providers=providers,
        selected_provider=selected_prov,
        hardware_accelerated=is_accel,
        verification_status=verif_msg,
        inference_backend_label=backend_label,
        benchmark_latency_ms=lat_ms,
        deployment_notes=notes
    )
