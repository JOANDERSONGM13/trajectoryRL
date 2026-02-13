"""TrajectoryRL Validator - Main validator implementation."""

import asyncio
import hashlib
import json
import logging
import sys
import time
import yaml
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import bittensor as bt
import numpy as np

from ..protocol.synapse import PackRequest, PackResponse
from ..utils.opp_schema import validate_opp_schema
from ..utils.config import ValidatorConfig
from ..utils.clawbench import ClawBenchHarness, EvaluationResult
from ..scoring import TrajectoryScorer, AggregatedScore
from ..utils.github import GitHubVerifier

logger = logging.getLogger(__name__)


class TrajectoryValidator:
    """TrajectoryRL validator that evaluates policy packs using ClawBench.

    The validator:
    1. Queries miners for policy packs (via Bittensor synapses)
    2. Verifies pack hashes and caches them
    3. Runs ClawBench scenarios against each pack
    4. Scores results and sets on-chain weights

    Example:
        >>> config = ValidatorConfig.from_env()
        >>> validator = TrajectoryValidator(config)
        >>> await validator.run()
    """

    def __init__(self, config: ValidatorConfig):
        """Initialize validator.

        Args:
            config: Validator configuration
        """
        self.config = config

        # Setup logging
        self._setup_logging()

        logger.info("=" * 60)
        logger.info("TrajectoryRL Validator v0.1.0")
        logger.info("=" * 60)

        # Initialize Bittensor components
        logger.info("Initializing Bittensor components...")
        self.wallet = bt.wallet(
            name=config.wallet_name,
            hotkey=config.wallet_hotkey
        )
        self.subtensor = bt.subtensor(network=config.network)
        self.metagraph = self.subtensor.metagraph(config.netuid)
        self.dendrite = bt.dendrite(wallet=self.wallet)

        logger.info(f"Wallet: {self.wallet}")
        logger.info(f"Network: {config.network}")
        logger.info(f"Netuid: {config.netuid}")

        # Initialize ClawBench harness
        logger.info("Initializing ClawBench harness...")
        self.harness = ClawBenchHarness(
            clawbench_path=config.clawbench_path,
            timeout=config.timeout_per_scenario
        )

        # Initialize scorer
        self.scorer = TrajectoryScorer(
            lambda_cost=config.lambda_cost,
            mu_safety=config.mu_safety,
            rho_reliability=config.rho_reliability
        )

        # Initialize GitHub verifier
        logger.info("Initializing GitHub verifier...")
        self.github_verifier = GitHubVerifier(
            cache_dir=config.log_dir / "git_cache"
        )

        # Pack cache (content-addressed)
        self.pack_cache: Dict[str, dict] = {}

        # Score history for tracking
        self.score_history: Dict[int, List[float]] = defaultdict(list)

        # First-mover tracking: {miner_uid: (first_score, first_timestamp)}
        # Tracks the FIRST submission from each miner that achieved a given score level
        self.first_mover_data: Dict[int, Tuple[float, float]] = {}

        # Load scenarios
        self.scenarios = self._load_scenarios()
        logger.info(f"Loaded {len(self.scenarios)} scenarios: {list(self.scenarios.keys())}")

        logger.info("Validator initialization complete!")

    def _setup_logging(self):
        """Setup logging configuration."""
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    self.config.log_dir / f"validator_{int(time.time())}.log"
                )
            ]
        )

    def _load_scenarios(self) -> Dict[str, dict]:
        """Load scenario configurations.

        Returns:
            Dict of scenario_name -> scenario config
        """
        scenarios = {}
        for scenario_name in self.config.scenarios:
            scenario_path = self.config.scenarios_path / f"{scenario_name}.yaml"
            if not scenario_path.exists():
                logger.warning(f"Scenario not found: {scenario_path}")
                continue

            with open(scenario_path) as f:
                scenario = yaml.safe_load(f)
                scenarios[scenario_name] = scenario
                logger.debug(f"Loaded scenario: {scenario_name}")

        if not scenarios:
            raise ValueError("No scenarios loaded!")

        return scenarios

    async def run(self):
        """Main validator loop."""
        logger.info("Starting validator main loop...")
        logger.info(f"Epoch interval: {self.config.epoch_interval}s")

        epoch = 0
        while True:
            try:
                epoch += 1
                logger.info("=" * 60)
                logger.info(f"Epoch {epoch} starting")
                logger.info("=" * 60)

                await self.run_epoch()

                logger.info(f"Epoch {epoch} complete. Sleeping {self.config.epoch_interval}s...")
                await asyncio.sleep(self.config.epoch_interval)

            except KeyboardInterrupt:
                logger.info("Received interrupt, shutting down...")
                break
            except Exception as e:
                logger.error(f"Epoch failed: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait 1 min before retry

    async def run_epoch(self):
        """Run one evaluation epoch."""
        # 1. Sync metagraph
        logger.info("Syncing metagraph...")
        self.metagraph.sync(subtensor=self.subtensor)

        # 2. Get active miners
        miners = self._get_active_miners()
        logger.info(f"Found {len(miners)} active miners")

        if not miners:
            logger.warning("No active miners found!")
            return

        # 3. Evaluate each miner
        scores = {}
        for miner_uid in miners:
            score = await self._evaluate_miner(miner_uid)
            scores[miner_uid] = score

        # 4. Set weights
        if scores:
            await self._set_weights(scores)
        else:
            logger.warning("No scores to set!")

    def _get_active_miners(self) -> List[int]:
        """Get list of active miner UIDs.

        Returns:
            List of miner UIDs
        """
        # Get all UIDs with non-zero stake
        miners = []
        for uid in range(len(self.metagraph.S)):
            # Skip if UID is a validator (high stake)
            if self.metagraph.S[uid] > 1000:  # Arbitrary threshold
                continue

            # Skip if UID is not registered
            if self.metagraph.axons[uid].ip == "0.0.0.0":
                continue

            miners.append(uid)

        return miners

    async def _evaluate_miner(self, miner_uid: int) -> float:
        """Evaluate a single miner.

        Args:
            miner_uid: Miner UID

        Returns:
            Final score [0, 1]
        """
        logger.info(f"Evaluating miner {miner_uid}...")

        # Step 1: Fetch pack
        pack_response = await self._fetch_pack(miner_uid)
        if pack_response is None:
            logger.warning(f"Miner {miner_uid}: Failed to fetch pack")
            return 0.0

        # Step 2: Verify GitHub submission
        if not pack_response.git_commit_hash or not pack_response.repo_url:
            logger.warning(
                f"Miner {miner_uid}: Missing git_commit_hash or repo_url"
            )
            return 0.0

        verification = await self.github_verifier.verify_submission(
            repo_url=pack_response.repo_url,
            git_commit_hash=pack_response.git_commit_hash,
            pack_hash=pack_response.pack_hash,
            on_chain_submission_time=time.time()  # TODO: Get actual on-chain time
        )

        if not verification.valid:
            logger.warning(
                f"Miner {miner_uid}: GitHub verification failed: {verification.error}"
            )
            return 0.0

        pack = verification.pack_content
        commit_timestamp = verification.commit_timestamp
        pack_hash = pack_response.pack_hash[:8]
        logger.info(
            f"Miner {miner_uid}: Got pack {pack_hash} "
            f"(commit: {pack_response.git_commit_hash[:8]}, "
            f"timestamp: {commit_timestamp})"
        )

        # Step 3: Static lint
        lint_result = validate_opp_schema(pack)
        if not lint_result.passed:
            logger.warning(
                f"Miner {miner_uid}: Pack lint failed: {lint_result.issues}"
            )
            return 0.0

        # Step 4: Run scenarios
        results = []
        for scenario_name in self.scenarios.keys():
            for seed in range(self.config.seeds_per_task):
                try:
                    result = await self.harness.evaluate_pack(
                        pack=pack,
                        scenario_name=scenario_name,
                        seed=seed
                    )
                    results.append(result)

                    logger.info(
                        f"Miner {miner_uid}: {scenario_name} (seed={seed}) -> "
                        f"score={result.score:.3f}, success={result.success}"
                    )

                except Exception as e:
                    logger.error(
                        f"Miner {miner_uid}: {scenario_name} failed: {e}",
                        exc_info=True
                    )

        # Step 5: Aggregate scores
        if not results:
            logger.warning(f"Miner {miner_uid}: No results!")
            return 0.0

        aggregated = self.scorer.aggregate_scores(results)
        final_score = self.scorer.compute_final_score(aggregated)

        logger.info(
            f"Miner {miner_uid}: Final score = {final_score:.3f} "
            f"(mean={aggregated.mean_score:.3f}, var={aggregated.variance:.3f})"
        )

        # Track history
        self.score_history[miner_uid].append(final_score)

        # Update first-mover tracking
        # If this miner doesn't have first-mover data, or improved their score, update it
        if miner_uid not in self.first_mover_data:
            self.first_mover_data[miner_uid] = (final_score, commit_timestamp)
            logger.info(
                f"Miner {miner_uid}: First submission recorded "
                f"(score={final_score:.3f}, timestamp={commit_timestamp})"
            )
        elif final_score > self.first_mover_data[miner_uid][0]:
            # Miner improved their score, update with new timestamp
            self.first_mover_data[miner_uid] = (final_score, commit_timestamp)
            logger.info(
                f"Miner {miner_uid}: Score improved "
                f"(new={final_score:.3f}, old={self.first_mover_data[miner_uid][0]:.3f})"
            )

        return final_score

    async def _fetch_pack(self, miner_uid: int) -> Optional[PackResponse]:
        """Fetch pack from miner.

        Args:
            miner_uid: Miner UID

        Returns:
            PackResponse or None if failed
        """
        request = PackRequest(
            suite_id="clawbench_v1",
            schema_version=1,
            max_bytes=65536,
            want_pointer_ok=True
        )

        try:
            # Query miner via dendrite
            response = await self.dendrite.forward(
                axons=[self.metagraph.axons[miner_uid]],
                synapse=request,
                timeout=10.0
            )

            # dendrite.forward returns list of responses
            if not response or len(response) == 0:
                return None

            return response[0]

        except Exception as e:
            logger.warning(f"Failed to fetch pack from miner {miner_uid}: {e}")
            return None

    async def _set_weights(self, scores: Dict[int, float]):
        """Set on-chain weights based on scores using winner-take-all.

        Args:
            scores: Dict of miner_uid -> score [0, 1]
        """
        logger.info(f"Setting weights for {len(scores)} miners...")

        # Select winner with first-mover advantage
        weights_dict = self.scorer.select_winner(
            scores=scores,
            first_mover_data=self.first_mover_data,
            delta=self.config.delta_threshold if hasattr(self.config, 'delta_threshold') else 0.05
        )

        # Prepare for Bittensor API
        uids = list(weights_dict.keys())
        weights = [weights_dict[uid] for uid in uids]

        # Find winner
        winner_uid = max(weights_dict.keys(), key=lambda uid: weights_dict[uid])

        logger.info("=" * 60)
        logger.info("WINNER-TAKE-ALL RESULTS")
        logger.info("=" * 60)
        logger.info(f"🏆 Winner: Miner {winner_uid} (score={scores[winner_uid]:.3f})")
        logger.info("-" * 60)
        logger.info("All miners:")
        for uid, weight in sorted(
            weights_dict.items(),
            key=lambda x: scores[x[0]],
            reverse=True
        ):
            marker = "🏆 WINNER" if weight == 1.0 else ""
            logger.info(f"  Miner {uid}: weight={weight:.4f}, score={scores[uid]:.3f} {marker}")

        # Set weights on chain
        try:
            success = self.subtensor.set_weights(
                netuid=self.config.netuid,
                wallet=self.wallet,
                uids=uids,
                weights=weights,
                wait_for_inclusion=True,
                wait_for_finalization=False
            )

            if success:
                logger.info("✓ Weights set successfully!")
            else:
                logger.error("✗ Failed to set weights")

        except Exception as e:
            logger.error(f"Error setting weights: {e}", exc_info=True)


async def main():
    """Entry point for validator."""
    config = ValidatorConfig.from_env()
    validator = TrajectoryValidator(config)
    await validator.run()


if __name__ == "__main__":
    asyncio.run(main())
