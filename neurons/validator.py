#!/usr/bin/env python3
"""TrajectoryRL Validator - Entry point.

Evaluates policy packs from miners using ClawBench scenarios.

Usage:
    python neurons/validator.py --wallet.name validator --wallet.hotkey default
    python neurons/validator.py --netuid 11 --network finney
"""

import asyncio
import logging

from trajectoryrl.base.validator import TrajectoryValidator
from trajectoryrl.utils.config import ValidatorConfig

logger = logging.getLogger(__name__)


async def main():
    """Entry point for TrajectoryRL validator."""
    try:
        # Load configuration from environment
        config = ValidatorConfig.from_env()

        logger.info("Starting TrajectoryRL Validator...")

        # Create and run validator
        validator = TrajectoryValidator(config)
        await validator.run()

    except KeyboardInterrupt:
        logger.info("Validator stopped by user")
    except Exception as e:
        logger.error(f"Validator failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
