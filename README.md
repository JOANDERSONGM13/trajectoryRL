# TrajectoryRL

> **Bittensor Subnet 11** — Optimize AI agent policies through decentralized competition

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Bittensor](https://img.shields.io/badge/bittensor-7.0+-green.svg)](https://github.com/opentensor/bittensor)

TrajectoryRL is a Bittensor subnet where miners compete to optimize AI agent policies for real-world tasks. Validators evaluate policy packs using deterministic scenarios, rewarding agents that are **safe**, **efficient**, and **reliable**.

## Overview

```
┌─────────────────────────────────────────────────────┐
│              TRAJECTORYRL SUBNET (SN11)              │
│                                                      │
│  MINERS                        VALIDATORS            │
│  ┌──────────────┐              ┌──────────────┐    │
│  │ Submit       │  PackRequest │ Query miners │    │
│  │ AGENTS.md    │─────────────▶│              │    │
│  │ policy packs │              │ Evaluate via │    │
│  │              │◀─────────────│ ClawBench    │    │
│  └──────────────┘  PackResponse└──────────────┘    │
│        │                              │             │
│        │  Optimized policies          │ Set weights │
│        ▼                              ▼             │
│  ┌──────────────────────────────────────────────┐  │
│  │         BITTENSOR BLOCKCHAIN                 │  │
│  │  TAO rewards based on performance            │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### What Makes TrajectoryRL Unique

- **Deterministic Evaluation** — Uses [ClawBench](https://github.com/trajectoryRL/clawbench) scenarios with fixed fixtures (no LLM-as-judge randomness)
- **Content-Addressed Packs** — Miners submit policy bundles identified by SHA256 hash
- **Real-World Tasks** — Scenarios test email triage, calendar conflicts, task delegation, incident response
- **Safety-First Scoring** — Critical safety violations = immediate score of 0

### Anti-Copy Incentive Mechanism

TrajectoryRL employs three layers of protection against copy-paste attacks:

1. **GitHub-Based Submission**
   - Miners publish packs to public repositories FIRST
   - Git commit timestamps provide cryptographic proof of originality
   - Validators independently clone repos and verify hashes

2. **Winner-Take-All**
   - Only the best miner receives rewards (100% of epoch emissions)
   - No participation rewards for mediocre submissions
   - Forces miners to innovate, not copy

3. **First-Mover Advantage**
   - Later submissions must beat `first_best_score + 0.05` to win
   - Protects early innovators from marginal improvements
   - Prevents "copy then tweak" strategies

## Quick Start

### For Validators

```bash
# 1. Clone repo
git clone https://github.com/trajectoryRL/trajectoryrl.git
cd trajectoryrl

# 2. Install
pip install -e .

# 3. Configure
cp .env.example .env
# Edit: add ANTHROPIC_API_KEY, WALLET_NAME

# 4. Run validator
python neurons/validator.py
```

### For Miners

```bash
# 1. Install
pip install -e .

# 2. Create policy pack
# See: docs/creating_packs.md

# 3. Publish to GitHub
git init my_pack_repo
cd my_pack_repo
cp ../my_pack.json pack.json
git add pack.json
git commit -m "Initial pack submission"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main

# 4. Run miner (submits git hash + repo URL)
python neurons/miner.py \
  --pack.path ./pack.json \
  --pack.repo https://github.com/YOUR_USERNAME/YOUR_REPO \
  --pack.commit $(git rev-parse HEAD)
```

## Project Structure

```
trajectoryrl/
├── trajectoryrl/              # Main package
│   ├── protocol/              # Bittensor synapses
│   │   └── synapse.py         # PackRequest/PackResponse
│   ├── base/                  # Core classes
│   │   ├── miner.py           # TrajectoryMiner (TODO)
│   │   └── validator.py       # TrajectoryValidator
│   ├── utils/                 # Shared utilities
│   │   ├── config.py          # Configuration
│   │   ├── clawbench.py       # ClawBench integration
│   │   └── opp_schema.py      # OPP v1 validation
│   └── scoring/               # Scoring logic
│       └── __init__.py        # Score aggregation
│
├── neurons/                   # Entry points
│   ├── validator.py           # Validator node
│   └── miner.py               # Miner node
│
├── docker/                    # Docker deployment
│   ├── Dockerfile.validator
│   └── docker-compose.yml     # Includes ClawBench
│
├── tests/                     # Test suite
├── docs/                      # Documentation
├── pyproject.toml             # Package definition
└── README.md                  # This file
```

## How It Works

### For Miners

1. **Create a policy pack** (OpenClaw Policy Pack - OPP v1 format):
   ```json
   {
     "schema_version": 1,
     "files": {
       "AGENTS.md": "# Rules for the agent...",
       "SOUL.md": "# Tone and boundaries..."
     },
     "tool_policy": {
       "allow": ["exec", "slack", "memory_search"],
       "deny": ["group:runtime"]
     },
     "metadata": {
       "pack_name": "efficient_safe_ops",
       "pack_version": "1.0.0"
     }
   }
   ```

2. **Publish to public GitHub repository**
   - Commit your pack as `pack.json` in your repo
   - Note the git commit hash (40-char SHA)

3. **Submit via Bittensor Axon** (responds to `PackRequest`):
   ```python
   PackResponse(
       pack_hash="sha256_of_pack_content",
       git_commit_hash="abc123...",  # 40-char git SHA
       repo_url="https://github.com/your_name/your_repo",
       metadata={"pack_name": "efficient_safe_ops", "pack_version": "1.0.0"}
   )
   ```

4. **Validators verify and evaluate** your pack via GitHub
5. **Earn TAO** if you achieve the best score (winner-take-all)

### For Validators

1. **Query miners** for policy packs (receive `pack_hash`, `git_commit_hash`, `repo_url`)
2. **Verify GitHub submission**:
   - Clone/update miner's public repository
   - Verify commit exists and extract pack
   - Verify commit timestamp < on-chain submission time
   - Verify `sha256(pack_json) == pack_hash`
3. **Run ClawBench scenarios**:
   - `client_escalation` — P0 incident response
   - `morning_brief` — Daily operating picture
   - `inbox_to_action` — Email batch processing
   - `team_standup` — Sprint status synthesis
4. **Score results**:
   ```
   score = success - ρ*variance
   ```
   (ClawBench already includes safety and efficiency in scenario scoring)
5. **Select winner** (winner-take-all with first-mover advantage):
   - Best score wins IF it beats `first_best_score + δ` (δ = 0.05)
   - Protects early high-quality submissions from copy-paste attacks
6. **Set on-chain weights** (winner = 1.0, others = 0.0)

## Scoring

### Performance Evaluation

ClawBench scenarios evaluate packs on multiple dimensions:

- **Success** — Task completion (did it work?)
- **Safety** — No forbidden actions, confirmations required
- **Efficiency** — Minimize tool calls and tokens
- **Reliability** — Low variance across seeds (ρ = 0.1)

**Example scoring**:
```python
# Scenario: client_escalation (seed=0)
clawbench_score = 0.92  # From ClawBench regex scoring
variance_penalty = 0.1 * 0.02  # ρ * variance across seeds

final_score = 0.92 - 0.002 = 0.918
```

### Winner-Take-All Mechanism

TrajectoryRL uses **winner-take-all** reward distribution with **first-mover advantage**:

1. **Best score wins** — Miner with highest score receives 100% of rewards (weight = 1.0)
2. **First-mover protection** — Later submissions must beat `first_best_score + δ` (δ = 0.05)
3. **All others receive zero** — No participation rewards

**Example**:
```python
# Epoch 1: Miner A submits (score=0.85, timestamp=100)
weights = {A: 1.0}  # A wins

# Epoch 2: Miner B copies A's pack (score=0.85, timestamp=200)
# Required: 0.85 + 0.05 = 0.90
# B's score (0.85) < 0.90 → A retains win
weights = {A: 1.0, B: 0.0}

# Epoch 3: Miner C improves (score=0.91, timestamp=300)
# C's score (0.91) > 0.90 → C wins
weights = {A: 0.0, B: 0.0, C: 1.0}
```

This mechanism prevents copy-paste attacks by requiring meaningful improvement (5%) to overtake early submissions.

## ClawBench Integration

TrajectoryRL uses [ClawBench](https://github.com/trajectoryRL/clawbench) for deterministic evaluation:

- **Fixed fixtures** — Same inbox, calendar, tasks for every evaluation
- **Regex scoring** — No LLM judge, fully reproducible
- **Mock tools** — No real API calls, sandboxed execution

Validators must have ClawBench cloned as a sibling directory:
```bash
git clone https://github.com/trajectoryRL/clawbench.git ../clawbench
```

## Development

### Installation

```bash
# Clone with submodules
git clone https://github.com/trajectoryRL/trajectoryrl.git
cd trajectoryrl

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
# Format code
black trajectoryrl/ neurons/

# Lint
ruff check trajectoryrl/ neurons/
```

### Local Testing

```bash
# 1. Start local Bittensor subtensor
docker run -p 9944:9944 opentensor/subtensor:latest

# 2. Create test wallets
btcli wallet new_coldkey --wallet.name test_miner --no_password
btcli wallet new_coldkey --wallet.name test_validator --no_password

# 3. Run locally
NETWORK=local python neurons/miner.py --wallet.name test_miner
NETWORK=local python neurons/validator.py --wallet.name test_validator
```

## Documentation

- **[Validator Guide](docs/validator_guide.md)** — Setup and operation
- **[Miner Guide](docs/miner_guide.md)** — Creating and optimizing packs
- **[OPP Spec](docs/opp_spec.md)** — Policy pack format
- **[Scoring](docs/scoring.md)** — How rewards are calculated
- **[Architecture](docs/architecture.md)** — Technical deep-dive

## Roadmap

- [x] ~~Validator implementation~~ (v0.1.0)
- [ ] Miner implementation (v0.2.0)
- [ ] Example policy packs
- [ ] Pack optimizer tools
- [ ] Multi-seed variance testing
- [ ] Prometheus metrics
- [ ] Web dashboard

## Community

- **GitHub**: https://github.com/trajectoryRL/trajectoryRL
- **Website**: https://trajrl.com

## License

MIT © 2026 TrajectoryRL Team

---

**Built on [Bittensor](https://bittensor.com)** | **Powered by [ClawBench](https://github.com/trajectoryRL/clawbench)**
