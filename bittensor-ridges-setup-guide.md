# Bittensor & Ridges AI (Subnet 62) — Setup Guide

> A step-by-step guide for installing Bittensor, wiring in an existing wallet, and registering on Ridges AI (SN62) on **Ubuntu 24.04** with **Python 3.13**.

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
| OS                | Ubuntu 24.04 (Noble)                                       |
| Python            | 3.13                                                        |
| CPU               | 4+ vCPU (2 vCPU too slow — Docker builds take 1hr+)        |
| RAM               | 8–16 GB                                                     |
| Storage           | 50 GB SSD                                                   |
| Network           | Low latency matters more than bandwidth (~20ms to Chutes)   |
| TAO tokens        | Enough to cover registration fee                            |
| Chutes API key    | Sign up at https://chutes.ai                                |

---

## 2. Install System Dependencies

You can do this manually, or use the **one‑click deployer** in [Section 10](#10-one-click-deployer-optional).

Manual steps:

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install build tools
sudo apt install -y build-essential curl git software-properties-common ca-certificates gnupg

# Install Rust (required by Bittensor on Linux)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Install UV package manager (recommended by Ridges)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Docker Engine from Docker's official apt repo (recommended)
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu noble stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable docker && sudo systemctl start docker
sudo usermod -aG docker $USER

# Configure git identity (required by Ridges sandbox for commits)
git config --global user.email "lifestream@ridges.ai"
git config --global user.name "lifestream"

# Mark the ridges repo as safe for git (avoids “dubious ownership” errors)
git config --global --add safe.directory /home/ubuntu/ridges
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

## 5. Create or Import a Wallet

### 5.1 Use an existing wallet (recommended if you already have TAO)

Copy your existing `wallets` directory onto the machine, for example to:

```bash
/home/ubuntu/wallets
```

Then verify it:

```bash
btcli wallet overview \
  --wallet.name lifestream \
  --wallet.path /home/ubuntu/wallets/
```

> ⚠️ **CRITICAL:** Treat your wallet directory and mnemonic phrase as highly sensitive.  
> Never paste your seed into this server or share it with anyone.

### 5.2 (Optional) Create a new wallet with BTCLI

Only do this if you don't already have a funded wallet:

```bash
btcli wallet create \
  --wallet.name lifestream \
  --wallet.hotkey default \
  --wallet.path /home/ubuntu/wallets/
```

---

## 6. Fund Your Wallet

You need TAO tokens to register on a subnet.

### 6.1 Get your wallet address

```bash
btcli wallet balance \
  --wallet.name lifestream \
  --wallet.path /home/ubuntu/wallets/
```

### 6.2 Buy & transfer TAO

- Purchase TAO on an exchange (Binance, MEXC, Gate.io, etc.)
- Withdraw to your **coldkey address** (ss58 format, starts with `5`)
- Wait for the transfer to confirm

### 6.3 Confirm balance

```bash
btcli wallet balance \
  --wallet.name lifestream \
  --wallet.path /home/ubuntu/wallets/
```

---

## 7. Register on Subnet 62 (Ridges AI)

```bash
btcli subnet register \
  --wallet.name lifestream \
  --wallet.hotkey default \
  --wallet.path /home/ubuntu/wallets/ \
  --netuid 62
```

- The CLI will display the **registration cost** (burned TAO)
- Type `y` to confirm and complete registration
- Wait for the transaction to be included in a block

### Verify registration

```bash
btcli subnet list \
  --wallet.name lifestream \
  --wallet.path /home/ubuntu/wallets/
```

---

## 8. Set Up Ridges Miner

### 8.1 Clone the repository

```bash
cd /home/ubuntu
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
WALLET_PATH=/home/ubuntu/wallets/
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
cd /home/ubuntu/ridges
chmod +x run_gateway.sh run_gateway_tmux.sh

# Simple foreground run
./run_gateway.sh

# Or: keep it running inside tmux
./run_gateway_tmux.sh
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

# Run a single problem test with your agent (example: rest-api-js)
./run_agent_test_rest_api_js.sh

# Run a problem set with a session ID (example: problems-0220 with session s10)
./run_agent_test_problems_0220_s10.sh

# Or run test_agent.py directly for custom runs, e.g.:
uv run python test_agent.py \
  --inference-url http://<YOUR_HOST_IP>:1234 \
  --agent-path ./top1-krav40.py \
  list-problem-sets
```

Results are saved to `test_agent_results/`.

> **Troubleshooting 502 Bad Gateway:**
> If you get a 502 error during inference, check these two things:
> 1. **Gateway binding:** `ss -tlnp | grep 1234` should show `0.0.0.0:1234`, not `127.0.0.1:1234`. Fix by setting `HOST=0.0.0.0` in `.env` and restarting the gateway.
> 2. **Inference URL:** `--inference-url` must use your host's real IP (from `hostname -I`), not `127.0.0.1` or `localhost`. The URL is passed to an nginx proxy running inside Docker, where localhost points to the container, not your host.

### 8.7 Upload your agent (start mining)

> **API URL Note:** The default API URL `https://platform-v2.ridges.ai` may be
> blocked by Cloudflare from certain IPs. If you get 403 errors, update
> `DEFAULT_API_BASE_URL` in `ridges.py` to `https://frontend-platform-api.ridges.ai`
> or use the `--url` flag:

```bash
cd /home/ubuntu/ridges

# Option A: Use the --url flag
uv run python ridges.py --url https://frontend-platform-api.ridges.ai upload \
  --file ./agent.py \
  --coldkey-name lifestream \
  --hotkey-name default

# Option B: Use the default (after editing ridges.py)
uv run python ridges.py upload \
  --file ./agent.py \
  --coldkey-name lifestream \
  --hotkey-name default
```

This will:
1. Validate your agent
2. Show the evaluation pricing (~0.23 TAO / ~$42 USD)
3. Upload it after you confirm payment

> **Balance requirement:** Your coldkey must have enough TAO to cover the eval fee.
> Check with: `btcli wallet balance --wallet.name lifestream --wallet.path /home/ubuntu/wallets/`

---

## 9. Useful Commands

All commands below use the custom wallet path. To avoid typing it every time, add it to your Bittensor config:

```bash
mkdir -p ~/.bittensor
echo "wallet_path: /home/ubuntu/wallets/" >> ~/.bittensor/config.yml
```

### Wallet

| Command | Description |
| ------- | ----------- |
| `btcli wallet overview --wallet.name lifestream --wallet.path /home/ubuntu/wallets/` | Wallet overview |
| `btcli wallet balance --wallet.name lifestream --wallet.path /home/ubuntu/wallets/` | Check TAO balance |
| `btcli wallet list --wallet.path /home/ubuntu/wallets/` | List all wallets |

### Subnet

| Command | Description |
| ------- | ----------- |
| `btcli subnet list` | List all subnets |
| `btcli subnet register --wallet.name lifestream --wallet.hotkey default --wallet.path /home/lifestream/wallets/ --netuid 62` | Register on SN62 |

### Staking

| Command | Description |
| ------- | ----------- |
| `btcli stake add --wallet.name lifestream --wallet.path /home/ubuntu/wallets/` | Stake TAO |
| `btcli stake remove --wallet.name lifestream --wallet.path /home/ubuntu/wallets/` | Unstake TAO |

---

## 10. One‑Click Deployer (Optional)

If you're setting up a fresh Ubuntu 24.04 server, you can use a single script to:

- Install system dependencies (Rust, uv, Docker, Python 3.13)
- Clone the `ridges` repo
- Create the `.venv` and install dependencies
- Configure `inference_gateway/.env`
- Create the helper scripts:
  - `run_gateway.sh`
  - `run_gateway_tmux.sh`
  - `run_agent_test_rest_api_js.sh`
  - `run_agent_test_problems_0220_s10.sh`

### 10.1 Usage

On a fresh machine:

```bash
cd /home/ubuntu
curl -O https://raw.githubusercontent.com/your-user-or-org/your-repo/main/ridges-deploy.sh
chmod +x ridges-deploy.sh
./ridges-deploy.sh
```

After it completes:

```bash
cd /home/ubuntu/ridges
./run_gateway_tmux.sh                         # start gateway in tmux
./run_agent_test_rest_api_js.sh              # quick smoke test
./run_agent_test_problems_0220_s10.sh        # run the problems-0220 problem set
```

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
