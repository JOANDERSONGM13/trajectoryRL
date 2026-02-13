"""ClawBench harness integration for evaluating policy packs."""

import asyncio
import json
import logging
import shutil
import subprocess
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result from running a scenario evaluation.

    Attributes:
        scenario_name: Name of the scenario
        score: Normalized score [0, 1]
        success: Whether the scenario passed
        tool_calls: Number of tool calls made
        response: Agent's final response
        rubric: Detailed scoring rubric results
        error: Error message if evaluation failed
    """
    scenario_name: str
    score: float
    success: bool
    tool_calls: int
    response: str
    rubric: Dict[str, Any]
    error: Optional[str] = None


class ClawBenchHarness:
    """Integrates with ClawBench for policy pack evaluation."""

    def __init__(self, clawbench_path: Path, timeout: int = 120):
        """Initialize harness.

        Args:
            clawbench_path: Path to clawbench directory
            timeout: Timeout in seconds for each scenario
        """
        self.clawbench_path = clawbench_path
        self.timeout = timeout
        self.scripts_path = clawbench_path / "scripts"
        self.scenarios_path = clawbench_path / "scenarios"
        self.fixtures_path = clawbench_path / "fixtures"

        # Validate paths
        if not self.scripts_path.exists():
            raise ValueError(f"ClawBench scripts not found: {self.scripts_path}")
        if not self.scenarios_path.exists():
            raise ValueError(f"ClawBench scenarios not found: {self.scenarios_path}")

    async def evaluate_pack(
        self,
        pack: dict,
        scenario_name: str,
        seed: int = 0
    ) -> EvaluationResult:
        """Evaluate a policy pack on a scenario.

        Args:
            pack: Policy pack dictionary (OPP v1 format)
            scenario_name: Name of scenario to run (e.g., "client_escalation")
            seed: Random seed for reproducibility

        Returns:
            EvaluationResult with score and details
        """
        logger.info(
            f"Evaluating pack on scenario={scenario_name}, seed={seed}, "
            f"pack_hash={self._compute_hash(pack)[:8]}"
        )

        # Create temporary workspace
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            try:
                # Apply pack to workspace
                self._apply_pack_to_workspace(pack, workspace)

                # Run scenario
                result = await self._run_scenario(
                    scenario_name=scenario_name,
                    workspace=workspace,
                    seed=seed
                )

                return result

            except Exception as e:
                logger.error(f"Evaluation failed: {e}", exc_info=True)
                return EvaluationResult(
                    scenario_name=scenario_name,
                    score=0.0,
                    success=False,
                    tool_calls=0,
                    response="",
                    rubric={},
                    error=str(e)
                )

    def _apply_pack_to_workspace(self, pack: dict, workspace: Path) -> None:
        """Write pack files to workspace directory.

        Args:
            pack: Policy pack dictionary
            workspace: Workspace directory path
        """
        # Create workspace directory
        workspace.mkdir(parents=True, exist_ok=True)

        # Write files from pack
        files = pack.get("files", {})
        for filename, content in files.items():
            file_path = workspace / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            logger.debug(f"Wrote {filename} ({len(content)} chars)")

        logger.info(f"Applied pack to workspace: {workspace}")

    async def _run_scenario(
        self,
        scenario_name: str,
        workspace: Path,
        seed: int
    ) -> EvaluationResult:
        """Run a ClawBench scenario.

        Args:
            scenario_name: Scenario name
            workspace: Workspace directory with pack files
            seed: Random seed

        Returns:
            EvaluationResult
        """
        # Load scenario config
        scenario_path = self.scenarios_path / f"{scenario_name}.yaml"
        if not scenario_path.exists():
            raise ValueError(f"Scenario not found: {scenario_path}")

        with open(scenario_path) as f:
            scenario = yaml.safe_load(f)

        # Run episode via run_episode.py
        run_script = self.scripts_path / "run_episode.py"
        if not run_script.exists():
            raise ValueError(f"run_episode.py not found: {run_script}")

        cmd = [
            "python",
            str(run_script),
            "--scenario", scenario_name,
            "--workspace", str(workspace),
            "--json",  # Output JSON for parsing
            "--wait",  # Wait for services to be ready
        ]

        logger.debug(f"Running command: {' '.join(cmd)}")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.clawbench_path)
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout
            )

            # Parse JSON output
            output = stdout.decode()
            result_data = self._parse_episode_output(output)

            # Extract scoring results
            score = result_data.get("score", 0.0)
            success = result_data.get("success", False)
            tool_calls = result_data.get("tool_calls", 0)
            response = result_data.get("response", "")
            rubric = result_data.get("rubric", {})

            return EvaluationResult(
                scenario_name=scenario_name,
                score=score,
                success=success,
                tool_calls=tool_calls,
                response=response,
                rubric=rubric
            )

        except asyncio.TimeoutError:
            logger.error(f"Scenario timeout: {scenario_name}")
            return EvaluationResult(
                scenario_name=scenario_name,
                score=0.0,
                success=False,
                tool_calls=0,
                response="",
                rubric={},
                error=f"Timeout after {self.timeout}s"
            )

    def _parse_episode_output(self, output: str) -> Dict[str, Any]:
        """Parse run_episode.py JSON output.

        Args:
            output: Raw stdout from run_episode.py

        Returns:
            Parsed result dictionary
        """
        # run_episode.py outputs JSON when --json flag is used
        # Format: {"score": 0.9, "success": true, "tool_calls": 12, ...}
        try:
            # Find JSON in output (may have logging before it)
            lines = output.strip().split("\n")
            for line in reversed(lines):
                if line.strip().startswith("{"):
                    return json.loads(line)

            raise ValueError("No JSON found in output")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON output: {e}")
            logger.debug(f"Output was: {output}")
            return {
                "score": 0.0,
                "success": False,
                "tool_calls": 0,
                "response": "",
                "rubric": {},
                "error": f"JSON parse error: {e}"
            }

    def _compute_hash(self, pack: dict) -> str:
        """Compute SHA256 hash of pack.

        Args:
            pack: Policy pack dict

        Returns:
            Hex digest
        """
        import hashlib
        content = json.dumps(pack, sort_keys=True).encode()
        return hashlib.sha256(content).hexdigest()
