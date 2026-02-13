# Monorepo Migration Complete! 🎉

**Date**: 2026-02-12
**Location**: `/data2/trajectory_rl/trajectoryRL/`

## ✅ What Was Created

The TrajectoryRL monorepo is now set up following **standard Bittensor subnet patterns**:

```
trajectoryRL/                      # Single repo for everything
├── trajectoryrl/                  # Main Python package
│   ├── __init__.py               # Package exports
│   ├── protocol/                  # Bittensor synapses
│   │   ├── __init__.py
│   │   └── synapse.py            # PackRequest/PackResponse
│   ├── base/                      # Core classes
│   │   ├── __init__.py
│   │   └── validator.py          # TrajectoryValidator
│   ├── utils/                     # Shared utilities
│   │   ├── __init__.py
│   │   ├── config.py             # Configuration
│   │   ├── clawbench.py          # ClawBench integration
│   │   └── opp_schema.py         # OPP v1 validation
│   └── scoring/                   # Scoring logic
│       └── __init__.py
│
├── neurons/                       # Entry points
│   ├── validator.py              # ✅ python neurons/validator.py
│   └── miner.py                  # ❌ TODO: implement
│
├── docker/                        # Docker deployment
│   ├── Dockerfile.validator
│   └── docker-compose.yml        # Includes ClawBench
│
├── tests/                         # Test suite
├── scripts/                       # Helper scripts
│   └── setup.sh
├── pyproject.toml                # Package definition
├── requirements.txt              # Dependencies
├── .env.example                  # Config template
├── README.md                     # Main docs
└── SETUP.md                      # This guide
```

## Why Monorepo?

✅ **Single source of truth** — No version sync issues
✅ **Easy imports** — `from trajectoryrl.protocol import PackRequest`
✅ **Standard pattern** — Matches Bittensor community conventions
✅ **Simpler deployment** — One `git clone`, one `pip install`

## What Changed

### Before (Scattered Structure)

```
/data2/trajectory_rl/
├── shared/trajectoryrl_protocol/   # ❌ Awkward "shared" location
├── validator/                       # ❌ Separate top-level
└── miner/                          # ❌ Separate top-level
```

### After (Monorepo)

```
/data2/trajectory_rl/trajectoryRL/
├── trajectoryrl/                   # ✅ Single package
│   ├── protocol/                  # ✅ Shared synapses
│   ├── base/                      # ✅ Core classes
│   └── utils/                     # ✅ Shared utilities
└── neurons/                        # ✅ Entry points
```

## How to Use

### Installation

```bash
cd /data2/trajectory_rl/trajectoryRL

# Quick setup
./scripts/setup.sh

# Or manual:
pip install -e .
cp .env.example .env
# Edit .env with your keys
```

### Running the Validator

```bash
# Option A: Direct
python neurons/validator.py

# Option B: Docker
cd docker
docker compose up --build
```

### Import Anywhere

```python
# From any Python script:
from trajectoryrl.protocol import PackRequest, PackResponse
from trajectoryrl.utils import validate_opp_schema
from trajectoryrl.base import TrajectoryValidator
```

## Next Steps

### 1. Fix ClawBench Integration (Critical)

The validator expects scored output from run_episode.py:

```python
# TODO in clawbench/scripts/run_episode.py
def score_episode(result: dict, scenario: dict) -> dict:
    from clawbench.scoring import score_scenario
    score_result = score_scenario(...)
    return {
        "score": score_result.normalized_score,
        "success": score_result.passed,
        "rubric": score_result.details
    }
```

### 2. Implement Miner

Create `trajectoryrl/base/miner.py`:

```python
class TrajectoryMiner:
    """Serves policy packs to validators."""

    def __init__(self, config: MinerConfig):
        self.wallet = bt.wallet(config=config)
        self.axon = bt.axon(wallet=self.wallet)
        self.pack = self.load_pack(config.pack_path)

    def forward(self, synapse: PackRequest) -> PackResponse:
        """Return policy pack to validator."""
        return PackResponse(
            pack_hash=self.pack_hash,
            pack_b64=self.pack_b64
        )
```

### 3. Create Example Packs

```bash
mkdir -p examples/packs
cat > examples/packs/baseline.json << 'EOF'
{
  "schema_version": 1,
  "files": {
    "AGENTS.md": "# Rules\n1. Be safe\n2. Be efficient",
    "SOUL.md": "# Tone\nConcise and helpful."
  },
  "tool_policy": {
    "allow": ["exec", "slack"],
    "deny": ["group:runtime"]
  },
  "metadata": {
    "pack_name": "baseline",
    "pack_version": "1.0.0"
  }
}
EOF
```

### 4. Write Tests

```bash
# Create tests/unit/test_protocol.py
pytest tests/
```

### 5. Test End-to-End

```bash
# 1. Start local subtensor
docker run -p 9944:9944 opentensor/subtensor:latest

# 2. Run miner (once implemented)
NETWORK=local python neurons/miner.py

# 3. Run validator
NETWORK=local python neurons/validator.py
```

## File Organization

### `trajectoryrl/` Package Structure

| Directory | Purpose | Examples |
|-----------|---------|----------|
| `protocol/` | Bittensor synapses | PackRequest, PackResponse |
| `base/` | Core miner/validator classes | TrajectoryValidator, TrajectoryMiner |
| `utils/` | Shared utilities | Config, ClawBench, OPP schema |
| `scoring/` | Score aggregation | TrajectoryScorer |

### Top-Level Structure

| Directory | Purpose |
|-----------|---------|
| `neurons/` | Entry points (`python neurons/validator.py`) |
| `docker/` | Docker deployment |
| `tests/` | Test suite |
| `scripts/` | Helper scripts |

## Dependencies

Managed by `pyproject.toml`:

```bash
# Install package
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"
```

## Documentation

- **[README.md](README.md)** — Main documentation
- **[SETUP.md](SETUP.md)** — Detailed setup guide
- **[/data2/trajectory_rl/IMPLEMENTATION_SUMMARY.md](../IMPLEMENTATION_SUMMARY.md)** — Implementation notes

## Old Code Location

The original scattered code is still at:
- `/data2/trajectory_rl/shared/` — Can be deleted
- `/data2/trajectory_rl/validator/` — Can be deleted
- `/data2/trajectory_rl/miner/` — Empty, can be deleted

**Keep only**:
- `/data2/trajectory_rl/trajectoryRL/` ← **This monorepo**
- `/data2/trajectory_rl/clawbench/` ← **External dependency**

## Questions?

1. Check [SETUP.md](SETUP.md) for detailed guides
2. Read the code — it's well-documented
3. See `/data2/trajectory_rl/internal_doc/miner_validator_design.md` for architecture

---

**Status**: ✅ Validator implemented, 🚧 Miner TODO, 📝 Docs complete
