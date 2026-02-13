"""Scoring functions for policy pack evaluation."""

import logging
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

from .harness import EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class AggregatedScore:
    """Aggregated score across multiple evaluations.

    Attributes:
        mean_score: Average score across scenarios/seeds
        variance: Variance in scores (for reliability penalty)
        success_rate: Fraction of successful evaluations
        total_evaluations: Number of evaluations run
        scenario_scores: Dict of scenario_name -> score
    """
    mean_score: float
    variance: float
    success_rate: float
    total_evaluations: int
    scenario_scores: Dict[str, float]

    @property
    def final_score(self) -> float:
        """Compute final score with reliability penalty.

        Returns:
            Final score in [0, 1]
        """
        return max(0.0, min(1.0, self.mean_score))


class TrajectoryScorer:
    """Scores policy packs based on ClawBench scenario results."""

    def __init__(
        self,
        lambda_cost: float = 0.3,
        mu_safety: float = 0.4,
        rho_reliability: float = 0.1
    ):
        """Initialize scorer.

        Args:
            lambda_cost: Weight for cost penalty (unused in ClawBench)
            mu_safety: Weight for safety penalty (already in ClawBench scoring)
            rho_reliability: Weight for variance penalty
        """
        self.lambda_cost = lambda_cost
        self.mu_safety = mu_safety
        self.rho_reliability = rho_reliability

        logger.info(
            f"Scorer initialized: λ={lambda_cost}, μ={mu_safety}, ρ={rho_reliability}"
        )

    def aggregate_scores(
        self,
        results: List[EvaluationResult]
    ) -> AggregatedScore:
        """Aggregate multiple evaluation results.

        Args:
            results: List of EvaluationResult from multiple scenarios/seeds

        Returns:
            AggregatedScore with mean, variance, success rate
        """
        if not results:
            return AggregatedScore(
                mean_score=0.0,
                variance=0.0,
                success_rate=0.0,
                total_evaluations=0,
                scenario_scores={}
            )

        # Extract scores
        scores = [r.score for r in results]
        successes = [r.success for r in results]

        # Compute statistics
        mean_score = float(np.mean(scores))
        variance = float(np.var(scores)) if len(scores) > 1 else 0.0
        success_rate = float(np.mean(successes))

        # Group by scenario
        scenario_scores = {}
        for result in results:
            if result.scenario_name not in scenario_scores:
                scenario_scores[result.scenario_name] = []
            scenario_scores[result.scenario_name].append(result.score)

        # Average within scenarios
        scenario_scores = {
            name: float(np.mean(scores))
            for name, scores in scenario_scores.items()
        }

        logger.info(
            f"Aggregated {len(results)} results: "
            f"mean={mean_score:.3f}, var={variance:.3f}, success={success_rate:.1%}"
        )

        return AggregatedScore(
            mean_score=mean_score,
            variance=variance,
            success_rate=success_rate,
            total_evaluations=len(results),
            scenario_scores=scenario_scores
        )

    def compute_final_score(
        self,
        aggregated: AggregatedScore
    ) -> float:
        """Compute final score with reliability penalty.

        Args:
            aggregated: AggregatedScore from multiple evaluations

        Returns:
            Final score in [0, 1]
        """
        # Apply variance penalty
        reliability_penalty = self.rho_reliability * aggregated.variance
        final = aggregated.mean_score - reliability_penalty

        # Clamp to [0, 1]
        final = max(0.0, min(1.0, final))

        logger.debug(
            f"Final score: {final:.3f} = "
            f"{aggregated.mean_score:.3f} - {reliability_penalty:.3f}"
        )

        return final

    def normalize_scores_to_weights(
        self,
        scores: Dict[int, float],
        temperature: float = 0.1
    ) -> Dict[int, float]:
        """Normalize miner scores to weights using softmax.

        Args:
            scores: Dict of miner_uid -> score [0, 1]
            temperature: Softmax temperature (lower = more concentrated)

        Returns:
            Dict of miner_uid -> weight (sums to 1.0)
        """
        if not scores:
            return {}

        uids = list(scores.keys())
        raw_scores = np.array([scores[uid] for uid in uids])

        # Softmax normalization
        exp_scores = np.exp(raw_scores / temperature)
        weights = exp_scores / exp_scores.sum()

        result = {uid: float(w) for uid, w in zip(uids, weights)}

        logger.info(
            f"Normalized {len(scores)} scores to weights "
            f"(temp={temperature}, spread={weights.std():.3f})"
        )

        return result
