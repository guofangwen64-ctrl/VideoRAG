#!/usr/bin/env bash
# Run inside the isolated .venv-vllm environment on hosts whose driver supports CUDA 12.x.
set -euo pipefail

python -m pip uninstall -y vllm torch torchvision torchaudio
python -m pip install --upgrade pip
# vLLM 0.6.6.post1 ships CUDA-12.1 binaries and supports Qwen2.5-VL.
# The extra index makes pip resolve torch==2.5.1 to its +cu121 wheel.
python -m pip install --upgrade --force-reinstall \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  'vllm==0.6.6.post1'

python - <<'PY'
import torch

print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not (torch.version.cuda or "").startswith("12."):
    raise SystemExit("Expected a CUDA 12.x PyTorch build")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; check nvidia-smi and the driver")
PY
