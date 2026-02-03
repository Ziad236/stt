#!/bin/bash
set -e

echo "🚀 Starting STT service..."

# Enforce CPU mode (safe default)
export FORCE_CPU=1
export CUDA_VISIBLE_DEVICES=""

# Start FastAPI / Uvicorn in foreground
exec /app/venv/bin/python -u -m uvicorn stt_api:app \
  --host 0.0.0.0 \
  --port 8001
