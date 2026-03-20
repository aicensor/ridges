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

if [ ! -f "top1-krav40.py" ]; then
  echo "Agent file top1-krav40.py not found in /home/ubuntu/ridges."
  exit 1
fi

echo "Activating virtual environment..."
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

# Detect host IP for inference URL (first IP from hostname -I)
HOST_IP="${HOST_IP:-$(hostname -I | awk '{print $1}')}"
INFERENCE_URL="http://${HOST_IP}:1234"

echo "Using inference URL: ${INFERENCE_URL}"
echo "Running agent test for problem: rest-api-js"

"$UV_BIN" run python test_agent.py \
  --inference-url "${INFERENCE_URL}" \
  --agent-path ./top1-krav40.py \
  --session-id s1\
  test-problem rest-api-js

