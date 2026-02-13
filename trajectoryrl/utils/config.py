"""Validator configuration."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ValidatorConfig:
    """Configuration for TrajectoryRL validator.

    Attributes:
        # Bittensor config
        wallet_name: Wallet name
        wallet_hotkey: Hotkey name
        netuid: Subnet UID (11 for TrajectoryRL)
        network: Bittensor network (finney, test, local)

        # ClawBench config
        clawbench_path: Path to clawbench directory
        scenarios: List of scenario names to evaluate
        scenarios_path: Path to scenarios directory

        # Evaluation config
        tasks_per_epoch: Number of tasks to sample per epoch
        seeds_per_task: Number of seeds to run per task (for variance)
        epoch_interval: Seconds between evaluation epochs
        timeout_per_scenario: Max seconds per scenario evaluation

        # Scoring config
        lambda_cost: Weight for cost penalty (0-1)
        mu_safety: Weight for safety penalty (0-1)
        rho_reliability: Weight for variance penalty (0-1)

        # Pack caching
        pack_cache_dir: Directory for caching downloaded packs
        pack_cache_max_size: Max cache size in MB

        # Logging
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files
    """

    # Bittensor config
    wallet_name: str = "validator"
    wallet_hotkey: str = "default"
    netuid: int = 11
    network: str = "finney"

    # ClawBench config
    clawbench_path: Path = field(
        default_factory=lambda: Path(__file__).parent.parent.parent.parent / "clawbench"
    )
    scenarios: List[str] = field(
        default_factory=lambda: [
            "client_escalation",
            "morning_brief",
            "inbox_to_action",
            "team_standup",
        ]
    )
    scenarios_path: Optional[Path] = None

    # Evaluation config
    tasks_per_epoch: int = 4  # Run all 4 scenarios
    seeds_per_task: int = 1  # TODO: increase to 3 for variance measurement
    epoch_interval: int = 720  # 12 minutes (720 seconds)
    timeout_per_scenario: int = 120  # 2 minutes max per scenario

    # Scoring config
    lambda_cost: float = 0.3  # 30% weight on cost efficiency
    mu_safety: float = 0.4  # 40% weight on safety compliance
    rho_reliability: float = 0.1  # 10% weight on variance

    # Pack caching
    pack_cache_dir: Path = field(
        default_factory=lambda: Path("/tmp/trajectoryrl_packs")
    )
    pack_cache_max_size: int = 100  # MB

    # Logging
    log_level: str = "INFO"
    log_dir: Path = field(
        default_factory=lambda: Path("./logs")
    )

    def __post_init__(self):
        """Set derived paths and create directories."""
        # Set scenarios_path if not provided
        if self.scenarios_path is None:
            self.scenarios_path = self.clawbench_path / "scenarios"

        # Create directories
        self.pack_cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Validate clawbench path
        if not self.clawbench_path.exists():
            raise ValueError(
                f"clawbench_path does not exist: {self.clawbench_path}"
            )
        if not self.scenarios_path.exists():
            raise ValueError(
                f"scenarios_path does not exist: {self.scenarios_path}"
            )

    @classmethod
    def from_env(cls) -> "ValidatorConfig":
        """Load configuration from environment variables.

        Returns:
            ValidatorConfig instance
        """
        return cls(
            wallet_name=os.getenv("WALLET_NAME", "validator"),
            wallet_hotkey=os.getenv("WALLET_HOTKEY", "default"),
            netuid=int(os.getenv("NETUID", "11")),
            network=os.getenv("NETWORK", "finney"),
            clawbench_path=Path(
                os.getenv(
                    "CLAWBENCH_PATH",
                    str(Path(__file__).parent.parent.parent.parent / "clawbench")
                )
            ),
            epoch_interval=int(os.getenv("EPOCH_INTERVAL", "720")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
