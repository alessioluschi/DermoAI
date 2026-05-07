#!/bin/bash
# DermoAI Pipeline — Environment Setup (Mac/Linux)
# Run once: bash setup_env.sh

set -e

echo "============================================================"
echo "DermoAI Pipeline — Environment Setup"
echo "============================================================"

# Create virtual environment
python -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip

echo "[1/4] Installing PyTorch with CUDA 12.4 support (RTX 3080)..."
pip install "torch>=2.6.0" torchvision --index-url https://download.pytorch.org/whl/cu124

echo "[2/4] Installing bitsandbytes with Ampere support..."
pip install "bitsandbytes>=0.45.0"

echo "[3/4] Installing remaining dependencies..."
pip install -r requirements.txt

echo "[4/4] Verifying GPU..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    cap = torch.cuda.get_device_capability(0)
    print(f'  Compute: {cap[0]}.{cap[1]}')
else:
    print('  CUDA not available — CPU fallback mode')
"

echo ""
echo "============================================================"
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Place your .pth model in models/"
echo "  2. source venv/bin/activate  (activate before each session)"
echo "  3. python -m src.pipeline --build-rag-kb  (one-time, ~10-20 min)"
echo "  4. python -m src.pipeline --vram-check"
echo "  5. python -m src.pipeline --image data/test_images/example.jpg"
echo "============================================================"
