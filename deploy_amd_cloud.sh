#!/usr/bin/env bash
# ==============================================================================
# MachineMind: Deployment Script for AMD Developer Cloud (ADC) & ROCm Instances
# AMD Developer Hackathon: ACT III - Intelligent Industry Track
# ==============================================================================

set -e

echo "=== [1/5] Checking System Environment ==="
uname -a
if command -v rocminfo &> /dev/null; then
    echo "[INFO] AMD ROCm runtime detected:"
    rocminfo | grep -E "Marketing Name|Device Type" || true
else
    echo "[INFO] rocminfo not found. Proceeding with CPU/NPU execution provider fallback."
fi

echo "=== [2/5] Creating Python Virtual Environment ==="
python3 -m venv venv
source venv/bin/activate

echo "=== [3/5] Installing Dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

# Optional: Install AMD ROCm ONNX Runtime if on a supported ROCm VM
if [ -n "$ENABLE_ROCM" ]; then
    echo "[INFO] Installing onnxruntime-rocm for AMD GPU acceleration..."
    pip install --extra-index-url https://download.onnxruntime.ai/rocm/ onnxruntime-rocm || true
fi

echo "=== [4/5] Executing Automated Unit Tests ==="
python -m pytest -v

echo "=== [5/5] Launching MachineMind Dashboard ==="
echo "Access MachineMind locally at: http://localhost:8501"
echo "If deployed on a remote AMD Cloud VM, use SSH port forwarding:"
echo "  ssh -L 8501:localhost:8501 <user>@<amd-cloud-ip>"
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
