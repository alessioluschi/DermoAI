@echo off
:: DermoAI Pipeline — Environment Setup (Windows)
:: Run once: setup_env.bat

echo ============================================================
echo DermoAI Pipeline - Environment Setup
echo ============================================================

:: Create virtual environment
python -m venv venv
call venv\Scripts\activate

python -m pip install --upgrade pip

echo [1/4] Installing PyTorch with CUDA 12.4 support (RTX 3080)...
pip install "torch>=2.6.0" torchvision --index-url https://download.pytorch.org/whl/cu124

echo [2/4] Installing bitsandbytes with Ampere support...
pip install "bitsandbytes>=0.45.0"

echo [3/4] Installing remaining dependencies...
pip install -r requirements.txt

echo [4/4] Verifying GPU...
python -c "import torch; name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'; vram=torch.cuda.get_device_properties(0).total_memory/1e9 if torch.cuda.is_available() else 0; print(f'  GPU: {name}'); print(f'  VRAM: {vram:.1f} GB')"

echo.
echo ============================================================
echo Setup complete!
echo.
echo Next steps:
echo   1. Place your .pth model in models\
echo   2. call venv\Scripts\activate  (before each session)
echo   3. python -m src.pipeline --build-rag-kb  (one-time)
echo   4. python -m src.pipeline --vram-check
echo   5. python -m src.pipeline --image data\test_images\example.jpg
echo ============================================================
pause
