#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/ridges

if [ ! -d ".venv" ]; then
  echo "Python virtual environment (.venv) not found. Please create it first:"
  echo "  uv venv --python 3.13"
  echo "  source .venv/bin/activate"
  echo "  uv pip install ."
  exit 1
fi

if [ ! -f "inference_gateway/.env" ]; then
  echo "inference_gateway/.env not found. Please configure it before running the gateway."
  exit 1
fi

echo "Activating virtual environment and starting inference gateway..."
source .venv/bin/activate

# Resolve uv binary for the current user
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
if [ -z "$UV_BIN" ] && [ -x "$HOME/.local/bin/uv" ]; then
  UV_BIN="$HOME/.local/bin/uv"
fi

if [ -z "$UV_BIN" ]; then
  echo "uv is not available on PATH. Please ensure it is installed for this user."
  exit 1
fi

"$UV_BIN" run -m inference_gateway.main

