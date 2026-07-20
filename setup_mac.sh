#!/usr/bin/env bash
# Idempotent setup for the transcriber pipeline on Apple Silicon (M-series) macOS.
set -euo pipefail

echo "==> 1/5 ffmpeg"
if ! command -v ffmpeg >/dev/null; then
  brew install ffmpeg
else
  echo "ffmpeg is already installed."
fi

echo "==> 2/5 Python venv"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip
# TODO: mlx-whisper/pyannote/torch release cadence can lag behind the newest
# CPython; if this install fails, retry with `python3.12 -m venv .venv` instead.
pip install -r requirements.txt

echo "==> 3/5 Ollama"
if ! command -v ollama >/dev/null; then
  echo "Ollama not found. Install it: https://ollama.com/download"
  exit 1
fi
ollama pull qwen2.5:14b

echo "==> 4/5 HuggingFace token for pyannote (one-time setup)"
if [ ! -f .env ] || ! grep -q '^HF_TOKEN=' .env; then
  read -rp "Paste your HuggingFace access token (huggingface.co/settings/tokens): " HF_TOKEN
  echo "HF_TOKEN=${HF_TOKEN}" >> .env
fi
echo "Don't forget to accept the model terms at:"
echo "  https://huggingface.co/pyannote/speaker-diarization-3.1"

echo "==> 5/5 Warming up models (downloads on first run)"
python -m transcriber --warmup

echo "Done. Check with: python -m transcriber --dry-run --input-folder ./audio"
