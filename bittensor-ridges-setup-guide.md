# Bittensor & Ridges AI (Subnet 62) — Setup Guide

> A step-by-step guide for installing Bittensor, creating a wallet, and registering on Ridges AI (SN62) on **Ubuntu 22.04** with **Python 3.13**.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Install System Dependencies](#2-install-system-dependencies)
3. [Install Python 3.13](#3-install-python-313)
4. [Install Bittensor SDK & BTCLI](#4-install-bittensor-sdk--btcli)
5. [Create a Wallet](#5-create-a-wallet)
6. [Fund Your Wallet](#6-fund-your-wallet)
7. [Register on Subnet 62 (Ridges AI)](#7-register-on-subnet-62-ridges-ai)
8. [Set Up Ridges Miner](#8-set-up-ridges-miner)
9. [Useful Commands](#9-useful-commands)

---

## 1. Prerequisites

| Requirement       | Details                                                      |
| ----------------- | ------------------------------------------------------------ |
| OS                | Ubuntu 22.04                                                |
| Python            | 3.13                                                        |
| CPU               | 4+ vCPU (2 vCPU too slow — Docker builds take 1hr+)        |
| RAM               | 8–16 GB                                                     |
| Storage           | 50 GB SSD                                                   |
| Network           | Low latency matters more than bandwidth (~20ms to Chutes)   |
| TAO tokens        | Enough to cover registration fee                            |
| Chutes API key    | Sign up at https://chutes.ai                                |

---

## 2. Install System Dependencies

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install build tools
sudo apt install -y build-essential curl git software-properties-common

# Install Rust (required by Bittensor on Linux)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Install UV package manager (recommended by Ridges)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Docker (needed for Ridges agent sandboxes)
sudo apt install -y docker.io
sudo systemctl enable docker && sudo systemctl start docker
sudo usermod -aG docker $USER

# Configure git identity (required by Ridges sandbox for commits)
git config --global user.email "lifestream@ridges.ai"
git config --global user.name "lifestream"
```

---

## 3. Install Python 3.13

Ubuntu 22.04 ships with Python 3.10 by default. Install Python 3.13 via the deadsnakes PPA:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.13 python3.13-venv python3.13-dev
```

Verify:

```bash
python3.13 --version
```

---

## 4. Install Bittensor SDK & BTCLI

```bash
# Create a virtual environment with Python 3.13
python3.13 -m venv btcli_venv
source btcli_venv/bin/activate

# Install Bittensor SDK
pip install bittensor

# Install BTCLI (separate package, not included with bittensor)
pip install bittensor-cli

# Verify installation
btcli --version
python3 -c "import bittensor; print(bittensor.__version__)"
```

---

## 5. Create a Wallet

### 5.1 Create coldkey + hotkey

```bash
btcli wallet create \
  --wallet.name lifestream \
  --wallet.hotkey default \
  --wallet.path /home/lifestream/wallets/
```

When prompted:
- **Wallet path:** `/home/lifestream/wallets/` (or your preferred location)
- **Number of words:** `12`
- **Password:** Set a strong password to encrypt the key

### 5.2 Verify wallet

```bash
btcli wallet overview \
  --wallet.name lifestream \
  --wallet.path /home/lifestream/wallets/
```

> ⚠️ **CRITICAL:** Write down your 12-word mnemonic seed phrase on paper and store it in a safe place. If you lose it, you lose all your TAO permanently. Never share it with anyone.

---

## 6. Fund Your Wallet

You need TAO tokens to register on a subnet.

### 6.1 Get your wallet address

```bash
btcli wallet balance \
  --wallet.name lifestream \
  --wallet.path /home/lifestream/wallets/
```

### 6.2 Buy & transfer TAO

- Purchase TAO on an exchange (Binance, MEXC, Gate.io, etc.)
- Withdraw to your **coldkey address** (ss58 format, starts with `5`)
- Wait for the transfer to confirm

### 6.3 Confirm balance

```bash
btcli wallet balance \
  --wallet.name lifestream \
  --wallet.path /home/lifestream/wallets/
```

---

## 7. Register on Subnet 62 (Ridges AI)

```bash
btcli subnet register \
  --wallet.name lifestream \
  --wallet.hotkey default \
  --wallet.path /home/lifestream/wallets/ \
  --netuid 62
```

- The CLI will display the **registration cost** (burned TAO)
- Type `y` to confirm and complete registration
- Wait for the transaction to be included in a block

### Verify registration

```bash
btcli subnet list \
  --wallet.name lifestream \
  --wallet.path /home/lifestream/wallets/
```

---

## 8. Set Up Ridges Miner

### 8.1 Clone the repository

```bash
cd /home/lifestream
git clone https://github.com/ridgesai/ridges
cd ridges
```

### 8.2 Create environment

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install .
```

### 8.3 Configure environment variables

```bash
cp inference_gateway/.env.example inference_gateway/.env
```

Edit `inference_gateway/.env` and set:

```env
NETUID=62
WALLET_NAME=lifestream
WALLET_HOTKEY=default
WALLET_PATH=/home/lifestream/wallets/
CHUTES_API_KEY=your_chutes_api_key_here
USE_DATABASE=False
```

> ⚠️ **IMPORTANT — `HOST` must be `0.0.0.0`:**
> The inference gateway **must** bind to `0.0.0.0` (all interfaces), not `127.0.0.1`.
> The test agent runs your code inside Docker containers, and a sandbox nginx proxy
> forwards inference requests back to the gateway on the host. If the gateway only
> listens on `127.0.0.1`, Docker containers cannot reach it → **502 Bad Gateway**.
>
> ```env
> HOST=0.0.0.0
> PORT=1234
> ```

### 8.4 Start the inference gateway

In a **separate terminal**:

```bash
cd /home/lifestream/ridges
source .venv/bin/activate
uv run -m inference_gateway.main
```

Verify it's listening on all interfaces:

```bash
ss -tlnp | grep 1234
# Should show: 0.0.0.0:1234  (NOT 127.0.0.1:1234)
```

### 8.5 Find your host IP for Docker networking

The `--inference-url` is passed into a Docker nginx proxy container. Inside Docker,
`127.0.0.1` refers to the container itself, **not** your host machine. You must use
your host's actual IP address.

```bash
# Get your host IP
hostname -I | awk '{print $1}'
```

Use the IP this returns (e.g., `144.172.112.108`) as the `--inference-url`.

### 8.6 Test your agent

Before uploading, test your agent locally against sample problems.
Use `--session-id` to isolate Docker resources when running multiple agents concurrently:

```bash
cd /home/lifestream/ridges

# Run a single problem test (replace IP with your host IP from step 8.5)
uv run python test_agent.py \
  --inference-url http://<YOUR_HOST_IP>:1234 \
  --agent-path ./agent.py \
  test-problem django__django-11138

# Run with a session ID (useful for parallel runs)
uv run python test_agent.py \
  --inference-url http://<YOUR_HOST_IP>:1234 \
  --agent-path ./agent.py \
  --session-id s1 \
  test-problem rest-api-js

# List available problem sets
uv run python test_agent.py \
  --inference-url http://<YOUR_HOST_IP>:1234 \
  --agent-path ./agent.py \
  list-problem-sets

# Run a small test set (10 problems)
uv run python test_agent.py \
  --inference-url http://<YOUR_HOST_IP>:1234 \
  --agent-path ./agent.py \
  test-problem-set screener-small
```

Results are saved to `test_agent_results/`.

> **Troubleshooting 502 Bad Gateway:**
> If you get a 502 error during inference, check these two things:
> 1. **Gateway binding:** `ss -tlnp | grep 1234` should show `0.0.0.0:1234`, not `127.0.0.1:1234`. Fix by setting `HOST=0.0.0.0` in `.env` and restarting the gateway.
> 2. **Inference URL:** `--inference-url` must use your host's real IP (from `hostname -I`), not `127.0.0.1` or `localhost`. The URL is passed to an nginx proxy running inside Docker, where localhost points to the container, not your host.

### 8.7 Upload your agent (start mining)

```bash
cd /home/lifestream/ridges

uv run python ridges.py upload \
  --file ./agent.py \
  --coldkey-name lifestream \
  --hotkey-name default
```

This will validate your agent, show the evaluation pricing, and upload it after you confirm payment.

---

## 9. Useful Commands

All commands below use the custom wallet path. To avoid typing it every time, add it to your Bittensor config:

```bash
mkdir -p ~/.bittensor
echo "wallet_path: /home/lifestream/wallets/" >> ~/.bittensor/config.yml
```

### Wallet

| Command | Description |
| ------- | ----------- |
| `btcli wallet overview --wallet.name lifestream --wallet.path /home/lifestream/wallets/` | Wallet overview |
| `btcli wallet balance --wallet.name lifestream --wallet.path /home/lifestream/wallets/` | Check TAO balance |
| `btcli wallet list --wallet.path /home/lifestream/wallets/` | List all wallets |

### Subnet

| Command | Description |
| ------- | ----------- |
| `btcli subnet list` | List all subnets |
| `btcli subnet register --wallet.name lifestream --wallet.hotkey default --wallet.path /home/lifestream/wallets/ --netuid 62` | Register on SN62 |

### Staking

| Command | Description |
| ------- | ----------- |
| `btcli stake add --wallet.name lifestream --wallet.path /home/lifestream/wallets/` | Stake TAO |
| `btcli stake remove --wallet.name lifestream --wallet.path /home/lifestream/wallets/` | Unstake TAO |

---

## Resources

- **Bittensor Docs:** https://docs.learnbittensor.org
- **BTCLI Reference:** https://docs.learnbittensor.org/btcli
- **Ridges AI Docs:** https://docs.ridges.ai
- **Ridges GitHub:** https://github.com/ridgesai/ridges
- **Chutes AI (API key):** https://chutes.ai
- **Taostats (SN62):** https://taostats.io/subnets/62

---

*Guide created: 2026-02-21*
