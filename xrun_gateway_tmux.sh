#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/ridges

# Attach to existing session or create a new one running the gateway
tmux attach -t inference-gateway || tmux new -s inference-gateway "./run_gateway.sh"

