"""Tests for TrajectoryRL validator components.

Tests the scoring, ClawBench harness, OPP schema validation,
and config without requiring a live Bittensor network.
"""

import asyncio
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock bittensor before importing any trajectoryrl modules
# ---------------------------------------------------------------------------
_mock_bt = MagicMock()

# bt.Synapse must be a real class so PackRequest/PackResponse can inherit
class _MockSynapse:
    pass

_mock_bt.Synapse = _MockSynapse
sys.modules["bittensor"] = _mock_bt

# Now safe to import
from trajectoryrl.utils.clawbench import ClawBenchHarness, EvaluationResult
from trajectoryrl.utils.opp_schema import validate_opp_schema, ValidationResult
from trajectoryrl.scoring import TrajectoryScorer, AggregatedScore
from trajectoryrl.utils.github import GitHubVerifier, GitVerificationResult


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # /data2/trajectory_rl
CLAWBENCH_PATH = REPO_ROOT / "clawbench"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scorer():
    return TrajectoryScorer(lambda_cost=0.3, mu_safety=0.4, rho_reliability=0.1)


@pytest.fixture
def harness():
    return ClawBenchHarness(clawbench_path=CLAWBENCH_PATH, timeout=120)


@pytest.fixture
def valid_pack():
    return {
        "schema_version": 1,
        "files": {
            "AGENTS.md": "# Agent rules\nBe safe and efficient.",
            "SOUL.md": "# Tone\nProfessional and concise.",
        },
        "tool_policy": {
            "allow": ["exec", "slack", "memory_search"],
            "deny": ["group:runtime"],
        },
        "metadata": {
            "pack_name": "test_pack",
            "pack_version": "1.0.0",
            "target_suite": "clawbench_v1",
        },
    }


@pytest.fixture
def sample_results():
    """Sample EvaluationResults for scoring tests."""
    return [
        EvaluationResult(
            scenario_name="client_escalation",
            score=0.92,
            success=True,
            tool_calls=10,
            response="Escalation summary...",
            rubric={"by_category": {"safety": {"score": 1.0}}},
        ),
        EvaluationResult(
            scenario_name="morning_brief",
            score=0.85,
            success=True,
            tool_calls=8,
            response="Daily brief...",
            rubric={"by_category": {"safety": {"score": 1.0}}},
        ),
        EvaluationResult(
            scenario_name="inbox_to_action",
            score=0.78,
            success=True,
            tool_calls=15,
            response="Action queue...",
            rubric={"by_category": {"safety": {"score": 0.9}}},
        ),
        EvaluationResult(
            scenario_name="team_standup",
            score=0.88,
            success=True,
            tool_calls=7,
            response="Standup prep...",
            rubric={"by_category": {"safety": {"score": 1.0}}},
        ),
    ]


# ===================================================================
# OPP Schema Validation Tests
# ===================================================================

class TestOPPSchemaValidation:

    def test_valid_pack_passes(self, valid_pack):
        result = validate_opp_schema(valid_pack)
        assert result.passed, f"Valid pack should pass, got issues: {result.issues}"

    def test_missing_schema_version(self, valid_pack):
        del valid_pack["schema_version"]
        result = validate_opp_schema(valid_pack)
        assert not result.passed
        assert any("schema_version" in i for i in result.issues)

    def test_wrong_schema_version(self, valid_pack):
        valid_pack["schema_version"] = 2
        result = validate_opp_schema(valid_pack)
        assert not result.passed
        assert any("schema_version" in i for i in result.issues)

    def test_missing_files(self, valid_pack):
        del valid_pack["files"]
        result = validate_opp_schema(valid_pack)
        assert not result.passed
        assert any("files" in i for i in result.issues)

    def test_missing_agents_md(self, valid_pack):
        del valid_pack["files"]["AGENTS.md"]
        result = validate_opp_schema(valid_pack)
        assert not result.passed
        assert any("AGENTS.md" in i for i in result.issues)

    def test_missing_tool_policy(self, valid_pack):
        del valid_pack["tool_policy"]
        result = validate_opp_schema(valid_pack)
        assert not result.passed
        assert any("tool_policy" in i for i in result.issues)

    def test_missing_metadata(self, valid_pack):
        del valid_pack["metadata"]
        result = validate_opp_schema(valid_pack)
        assert not result.passed
        assert any("metadata" in i for i in result.issues)

    def test_invalid_file_content_type(self, valid_pack):
        valid_pack["files"]["AGENTS.md"] = 123
        result = validate_opp_schema(valid_pack)
        assert not result.passed
        assert any("string" in i for i in result.issues)

    def test_legacy_allowed_denied_rejected(self, valid_pack):
        valid_pack["tool_policy"]["allowed"] = ["exec"]
        result = validate_opp_schema(valid_pack)
        assert not result.passed
        assert any("allow" in i and "deny" in i for i in result.issues)

    def test_invalid_semver(self, valid_pack):
        valid_pack["metadata"]["pack_version"] = "1.0"
        result = validate_opp_schema(valid_pack)
        assert not result.passed
        assert any("semver" in i for i in result.issues)

    def test_oversized_pack(self, valid_pack):
        valid_pack["files"]["AGENTS.md"] = "x" * 200_000
        result = validate_opp_schema(valid_pack)
        assert not result.passed
        assert any("too large" in i.lower() or "100KB" in i for i in result.issues)

    def test_validation_result_bool(self):
        assert bool(ValidationResult(passed=True, issues=[])) is True
        assert bool(ValidationResult(passed=False, issues=["error"])) is False


# ===================================================================
# TrajectoryScorer Tests
# ===================================================================

class TestTrajectoryScorer:

    def test_aggregate_empty_results(self, scorer):
        agg = scorer.aggregate_scores([])
        assert agg.mean_score == 0.0
        assert agg.variance == 0.0
        assert agg.success_rate == 0.0
        assert agg.total_evaluations == 0

    def test_aggregate_single_result(self, scorer):
        results = [
            EvaluationResult(
                scenario_name="test",
                score=0.85,
                success=True,
                tool_calls=5,
                response="ok",
                rubric={},
            )
        ]
        agg = scorer.aggregate_scores(results)
        assert agg.mean_score == 0.85
        assert agg.variance == 0.0  # single result = zero variance
        assert agg.success_rate == 1.0
        assert agg.total_evaluations == 1
        assert agg.scenario_scores == {"test": 0.85}

    def test_aggregate_multiple_results(self, scorer, sample_results):
        agg = scorer.aggregate_scores(sample_results)

        expected_mean = (0.92 + 0.85 + 0.78 + 0.88) / 4
        assert abs(agg.mean_score - expected_mean) < 1e-6
        assert agg.variance > 0  # different scores = non-zero variance
        assert agg.success_rate == 1.0
        assert agg.total_evaluations == 4
        assert len(agg.scenario_scores) == 4

    def test_aggregate_with_failures(self, scorer):
        results = [
            EvaluationResult("a", score=0.9, success=True, tool_calls=5, response="", rubric={}),
            EvaluationResult("b", score=0.0, success=False, tool_calls=0, response="", rubric={}),
        ]
        agg = scorer.aggregate_scores(results)
        assert agg.success_rate == 0.5
        assert agg.mean_score == 0.45

    def test_compute_final_score_no_variance(self, scorer):
        agg = AggregatedScore(
            mean_score=0.9,
            variance=0.0,
            success_rate=1.0,
            total_evaluations=4,
            scenario_scores={"a": 0.9},
        )
        final = scorer.compute_final_score(agg)
        assert final == 0.9  # No penalty

    def test_compute_final_score_with_variance(self, scorer):
        agg = AggregatedScore(
            mean_score=0.9,
            variance=0.1,
            success_rate=1.0,
            total_evaluations=4,
            scenario_scores={"a": 0.9},
        )
        # final = 0.9 - 0.1 * 0.1 = 0.89
        final = scorer.compute_final_score(agg)
        assert abs(final - 0.89) < 1e-6

    def test_compute_final_score_clamped(self, scorer):
        agg = AggregatedScore(
            mean_score=0.01,
            variance=1.0,
            success_rate=0.0,
            total_evaluations=1,
            scenario_scores={},
        )
        final = scorer.compute_final_score(agg)
        assert final == 0.0  # Clamped to 0

    def test_select_winner_basic(self, scorer):
        scores = {0: 0.85, 1: 0.72, 2: 0.91}
        weights = scorer.select_winner(scores, first_mover_data={}, delta=0.05)

        assert weights[2] == 1.0  # uid 2 has highest score
        assert weights[0] == 0.0
        assert weights[1] == 0.0

    def test_select_winner_empty(self, scorer):
        assert scorer.select_winner({}, {}, delta=0.05) == {}

    def test_select_winner_single_miner(self, scorer):
        scores = {42: 0.8}
        weights = scorer.select_winner(scores, first_mover_data={}, delta=0.05)
        assert weights[42] == 1.0

    def test_select_winner_first_mover_protection(self, scorer):
        """Early miner A (score=0.85) should be protected against B (score=0.88)
        because B doesn't beat A + delta (0.85 + 0.05 = 0.90)."""
        scores = {0: 0.85, 1: 0.88}
        first_mover_data = {
            0: (0.85, 100.0),  # A submitted first
            1: (0.88, 200.0),  # B submitted later
        }
        weights = scorer.select_winner(scores, first_mover_data, delta=0.05)

        # A should win due to first-mover protection
        assert weights[0] == 1.0
        assert weights[1] == 0.0

    def test_select_winner_first_mover_beaten(self, scorer):
        """New miner B (score=0.91) beats A + delta (0.85 + 0.05 = 0.90)."""
        scores = {0: 0.85, 1: 0.91}
        first_mover_data = {
            0: (0.85, 100.0),
            1: (0.91, 200.0),
        }
        weights = scorer.select_winner(scores, first_mover_data, delta=0.05)

        # B wins because 0.91 > 0.90
        assert weights[1] == 1.0
        assert weights[0] == 0.0

    def test_full_scoring_pipeline(self, scorer, sample_results):
        """End-to-end: results -> aggregate -> final score."""
        agg = scorer.aggregate_scores(sample_results)
        final = scorer.compute_final_score(agg)

        expected_mean = (0.92 + 0.85 + 0.78 + 0.88) / 4  # 0.8575
        expected_penalty = 0.1 * agg.variance
        expected_final = max(0.0, min(1.0, expected_mean - expected_penalty))

        assert abs(final - expected_final) < 1e-6
        assert 0.0 <= final <= 1.0


# ===================================================================
# ClawBenchHarness Tests
# ===================================================================

class TestClawBenchHarness:

    def test_init_validates_paths(self):
        with pytest.raises(ValueError, match="scripts not found"):
            ClawBenchHarness(clawbench_path=Path("/nonexistent"))

    def test_init_with_valid_path(self, harness):
        assert harness.clawbench_path == CLAWBENCH_PATH
        assert harness.timeout == 120

    def test_parse_episode_output_clean_json(self, harness):
        output = '{"score": 0.92, "success": true, "tool_calls": 8, "response": "test", "rubric": {}}'
        result = harness._parse_episode_output(output)
        assert result["score"] == 0.92
        assert result["success"] is True
        assert result["tool_calls"] == 8

    def test_parse_episode_output_json_after_logs(self, harness):
        output = (
            "Some log line\n"
            "Another log line\n"
            '{"score": 0.85, "success": false, "tool_calls": 3, "response": "", "rubric": {}}'
        )
        result = harness._parse_episode_output(output)
        assert result["score"] == 0.85
        assert result["success"] is False

    def test_parse_episode_output_no_json(self, harness):
        result = harness._parse_episode_output("no json here at all")
        assert result["score"] == 0.0
        assert "error" in result

    def test_parse_episode_output_invalid_json(self, harness):
        result = harness._parse_episode_output("{invalid json}")
        assert result["score"] == 0.0
        assert "error" in result

    def test_parse_episode_output_empty(self, harness):
        result = harness._parse_episode_output("")
        assert result["score"] == 0.0
        assert "error" in result

    def test_parse_episode_output_full_rubric(self, harness):
        """Test parsing a realistic scored output with full rubric."""
        data = {
            "score": 0.75,
            "success": False,
            "tool_calls": 5,
            "response": "Here is the summary...",
            "rubric": {
                "score": 0.75,
                "points_earned": 30,
                "points_possible": 40,
                "passed": 10,
                "failed": 5,
                "total_checks": 15,
                "by_category": {
                    "safety": {"earned": 12, "possible": 12, "score": 1.0},
                    "correctness": {"earned": 10, "possible": 15, "score": 0.667},
                },
            },
        }
        output = json.dumps(data)
        result = harness._parse_episode_output(output)
        assert result["score"] == 0.75
        assert result["rubric"]["by_category"]["safety"]["score"] == 1.0

    def test_compute_hash(self, harness, valid_pack):
        h = harness._compute_hash(valid_pack)
        expected = hashlib.sha256(
            json.dumps(valid_pack, sort_keys=True).encode()
        ).hexdigest()
        assert h == expected

    def test_compute_hash_deterministic(self, harness, valid_pack):
        h1 = harness._compute_hash(valid_pack)
        h2 = harness._compute_hash(valid_pack)
        assert h1 == h2

    def test_apply_pack_to_workspace(self, harness, valid_pack):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            harness._apply_pack_to_workspace(valid_pack, workspace)

            agents_file = workspace / "AGENTS.md"
            soul_file = workspace / "SOUL.md"

            assert agents_file.exists()
            assert soul_file.exists()
            assert agents_file.read_text() == valid_pack["files"]["AGENTS.md"]
            assert soul_file.read_text() == valid_pack["files"]["SOUL.md"]

    def test_apply_pack_empty_files(self, harness):
        pack = {"files": {}}
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            harness._apply_pack_to_workspace(pack, workspace)
            assert workspace.exists()
            # No files written
            assert list(workspace.iterdir()) == []

    def test_evaluate_pack_missing_scenario(self, harness, valid_pack):
        result = asyncio.get_event_loop().run_until_complete(
            harness.evaluate_pack(
                pack=valid_pack,
                scenario_name="nonexistent_scenario",
                seed=0,
            )
        )
        assert result.score == 0.0
        assert result.error is not None
        assert "not found" in result.error.lower()


# ===================================================================
# EvaluationResult Tests
# ===================================================================

class TestEvaluationResult:

    def test_creation(self):
        r = EvaluationResult(
            scenario_name="test",
            score=0.85,
            success=True,
            tool_calls=10,
            response="hello",
            rubric={"key": "value"},
        )
        assert r.scenario_name == "test"
        assert r.score == 0.85
        assert r.error is None

    def test_error_result(self):
        r = EvaluationResult(
            scenario_name="test",
            score=0.0,
            success=False,
            tool_calls=0,
            response="",
            rubric={},
            error="Timeout after 120s",
        )
        assert r.error == "Timeout after 120s"
        assert r.score == 0.0


# ===================================================================
# GitHubVerifier Tests
# ===================================================================

class TestGitHubVerifier:

    def test_init_creates_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir) / "git_cache"
            verifier = GitHubVerifier(cache_dir=cache)
            assert cache.exists()

    def test_verify_commit_exists_invalid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            verifier = GitHubVerifier(cache_dir=Path(tmpdir))
            # Non-git directory
            result = asyncio.get_event_loop().run_until_complete(
                verifier._verify_commit_exists(
                    Path(tmpdir), "abc123" * 7  # fake hash
                )
            )
            assert result is False

    def test_get_commit_timestamp_invalid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            verifier = GitHubVerifier(cache_dir=Path(tmpdir))
            result = asyncio.get_event_loop().run_until_complete(
                verifier._get_commit_timestamp(
                    Path(tmpdir), "abc123" * 7
                )
            )
            assert result is None


# ===================================================================
# Config Tests
# ===================================================================

class TestValidatorConfig:

    def test_default_scenarios(self):
        """Test that default scenarios list is correct."""
        from trajectoryrl.utils.config import ValidatorConfig

        # Can't fully instantiate (git check), but can inspect defaults
        defaults = ValidatorConfig.__dataclass_fields__
        assert "scenarios" in defaults
        # Check the default factory produces expected list
        scenarios = defaults["scenarios"].default_factory()
        assert "client_escalation" in scenarios
        assert "morning_brief" in scenarios
        assert "inbox_to_action" in scenarios
        assert "team_standup" in scenarios

    def test_default_scoring_params(self):
        from trajectoryrl.utils.config import ValidatorConfig
        defaults = ValidatorConfig.__dataclass_fields__
        assert defaults["rho_reliability"].default == 0.1
        assert defaults["delta_threshold"].default == 0.05
        assert defaults["seeds_per_task"].default == 1
        assert defaults["epoch_interval"].default == 720


# ===================================================================
# Integration: ClawBench Scoring → Validator Scoring Pipeline
# ===================================================================

class TestScoringIntegration:
    """Tests the full data flow from ClawBench output to validator weights."""

    def test_json_output_to_evaluation_result(self, harness):
        """ClawBench --json output → _parse_episode_output → EvaluationResult."""
        json_output = json.dumps({
            "score": 0.88,
            "success": True,
            "tool_calls": 11,
            "response": "Summary of actions taken...",
            "rubric": {
                "score": 0.88,
                "points_earned": 35,
                "points_possible": 40,
                "by_category": {"safety": {"score": 1.0}},
            },
        })

        parsed = harness._parse_episode_output(json_output)
        result = EvaluationResult(
            scenario_name="client_escalation",
            score=parsed["score"],
            success=parsed["success"],
            tool_calls=parsed["tool_calls"],
            response=parsed["response"],
            rubric=parsed["rubric"],
        )

        assert result.score == 0.88
        assert result.success is True
        assert result.tool_calls == 11

    def test_full_pipeline_json_to_weights(self, harness, scorer):
        """Full: 4 scenario JSON outputs → aggregate → final score → winner."""
        scenario_outputs = [
            ("client_escalation", 0.92),
            ("morning_brief", 0.85),
            ("inbox_to_action", 0.78),
            ("team_standup", 0.88),
        ]

        results = []
        for scenario, score in scenario_outputs:
            json_output = json.dumps({
                "score": score,
                "success": score > 0.5,
                "tool_calls": 10,
                "response": f"{scenario} done",
                "rubric": {"score": score},
            })
            parsed = harness._parse_episode_output(json_output)
            results.append(EvaluationResult(
                scenario_name=scenario,
                score=parsed["score"],
                success=parsed["success"],
                tool_calls=parsed["tool_calls"],
                response=parsed["response"],
                rubric=parsed["rubric"],
            ))

        # Aggregate and score
        agg = scorer.aggregate_scores(results)
        final = scorer.compute_final_score(agg)

        # Two miners compete
        miner_scores = {0: final, 1: final * 0.8}
        weights = scorer.select_winner(miner_scores, first_mover_data={}, delta=0.05)

        assert weights[0] == 1.0  # Higher score wins
        assert weights[1] == 0.0
        assert 0.0 < final <= 1.0

    def test_clawbench_scoring_roundtrip(self):
        """Test that clawbench scoring.py output matches what harness expects."""
        sys.path.insert(0, str(CLAWBENCH_PATH))
        from clawbench.scoring import score_episode
        import yaml

        with open(CLAWBENCH_PATH / "scenarios" / "client_escalation.yaml") as f:
            scenario = yaml.safe_load(f)

        # Simulate a good result
        tool_calls = [{"tool": "exec"}] * 5 + [{"tool": "slack"}] * 3
        from collections import Counter
        tool_counts = dict(Counter(tc["tool"] for tc in tool_calls))

        scorable = {
            "response": (
                "## P0 Status: Data Export Incident\n\n"
                "Root cause: cursor reset regression in v2.14.5.\n"
                "Fix: PR #356 ready, staging validated. ETA: deploy by 1pm today.\n"
                "Affected: Zenith Financial, GlobalTech, Meridian Health.\n"
                "Calendar conflict: 2pm interview overlaps with Acme call.\n"
                "SOC 2 findings noted — defer until P0 resolved.\n\n"
                "Recommended action plan:\n"
                "1. Approve Marcus's hotfix deploy\n"
                "2. Draft reply to Dana Reeves for your approval\n"
            ),
            "tool_calls_raw": tool_calls,
            "tool_calls_by_type": tool_counts,
            "tool_calls_total": len(tool_calls),
        }

        score_result = score_episode(scorable, scenario["scoring"])

        # Build the JSON output that run_episode.py --json would produce
        output = {
            "score": score_result["score"],
            "success": score_result.get("failed", 1) == 0,
            "tool_calls": len(tool_calls),
            "response": scorable["response"],
            "rubric": score_result,
        }

        # Parse it like the validator harness would
        harness = ClawBenchHarness(clawbench_path=CLAWBENCH_PATH)
        parsed = harness._parse_episode_output(json.dumps(output))

        assert parsed["score"] == score_result["score"]
        assert "by_category" in parsed["rubric"]
        assert parsed["tool_calls"] == 8

        # Feed into EvaluationResult
        eval_result = EvaluationResult(
            scenario_name="client_escalation",
            score=parsed["score"],
            success=parsed["success"],
            tool_calls=parsed["tool_calls"],
            response=parsed["response"],
            rubric=parsed["rubric"],
        )

        # Score should be high for this good result
        assert eval_result.score >= 0.7, f"Expected good score, got {eval_result.score}"
