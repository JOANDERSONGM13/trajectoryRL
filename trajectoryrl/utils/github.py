"""GitHub repository verification utilities."""

import json
import logging
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class GitVerificationResult:
    """Result of Git repository verification.

    Attributes:
        valid: Whether verification passed
        commit_timestamp: Unix timestamp of commit (if valid)
        pack_content: Parsed pack dict (if valid)
        error: Error message (if invalid)
    """
    valid: bool
    commit_timestamp: Optional[float] = None
    pack_content: Optional[dict] = None
    error: Optional[str] = None


class GitHubVerifier:
    """Verifies policy pack submissions from GitHub repositories."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize verifier.

        Args:
            cache_dir: Directory for caching cloned repos (default: temp dir)
        """
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "trajectoryrl_git_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"GitHubVerifier initialized with cache: {self.cache_dir}")

    async def verify_submission(
        self,
        repo_url: str,
        git_commit_hash: str,
        pack_hash: str,
        on_chain_submission_time: float
    ) -> GitVerificationResult:
        """Verify a GitHub-based pack submission.

        Args:
            repo_url: Public GitHub repository URL
            git_commit_hash: Git commit SHA (40-char hex)
            pack_hash: Expected SHA256 hash of pack content
            on_chain_submission_time: Unix timestamp of on-chain submission

        Returns:
            GitVerificationResult with validation outcome
        """
        logger.info(f"Verifying submission: {repo_url}@{git_commit_hash[:8]}")

        # Step 1: Clone or update repo
        repo_path = await self._clone_or_update_repo(repo_url)
        if repo_path is None:
            return GitVerificationResult(
                valid=False,
                error="Failed to clone repository"
            )

        # Step 2: Verify commit exists
        commit_exists = await self._verify_commit_exists(repo_path, git_commit_hash)
        if not commit_exists:
            return GitVerificationResult(
                valid=False,
                error=f"Commit {git_commit_hash[:8]} not found in repository"
            )

        # Step 3: Get commit timestamp
        commit_timestamp = await self._get_commit_timestamp(repo_path, git_commit_hash)
        if commit_timestamp is None:
            return GitVerificationResult(
                valid=False,
                error="Failed to get commit timestamp"
            )

        # Step 4: Verify timestamp is before on-chain submission
        if commit_timestamp > on_chain_submission_time:
            return GitVerificationResult(
                valid=False,
                error=f"Commit timestamp ({commit_timestamp}) is after on-chain submission ({on_chain_submission_time})"
            )

        logger.info(
            f"Commit timestamp check passed: {commit_timestamp} < {on_chain_submission_time}"
        )

        # Step 5: Extract pack from commit
        pack_content = await self._extract_pack_from_commit(repo_path, git_commit_hash)
        if pack_content is None:
            return GitVerificationResult(
                valid=False,
                error="Failed to extract pack from commit"
            )

        # Step 6: Verify pack hash
        import hashlib
        computed_hash = hashlib.sha256(
            json.dumps(pack_content, sort_keys=True).encode()
        ).hexdigest()

        if computed_hash != pack_hash:
            return GitVerificationResult(
                valid=False,
                error=f"Pack hash mismatch: expected {pack_hash[:8]}, got {computed_hash[:8]}"
            )

        logger.info(f"✓ Verification passed for {git_commit_hash[:8]}")

        return GitVerificationResult(
            valid=True,
            commit_timestamp=commit_timestamp,
            pack_content=pack_content
        )

    async def _clone_or_update_repo(self, repo_url: str) -> Optional[Path]:
        """Clone repository or update if already cached.

        Args:
            repo_url: GitHub repository URL

        Returns:
            Path to cloned repo, or None if failed
        """
        # Create safe directory name from repo URL
        repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        repo_path = self.cache_dir / repo_name

        try:
            if repo_path.exists():
                # Update existing repo
                logger.debug(f"Updating cached repo: {repo_path}")
                result = subprocess.run(
                    ["git", "-C", str(repo_path), "fetch", "--all"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode != 0:
                    logger.warning(f"Failed to update repo: {result.stderr}")
                    # Try to clone fresh
                    subprocess.run(["rm", "-rf", str(repo_path)], check=True)
                    return await self._clone_or_update_repo(repo_url)
            else:
                # Clone fresh
                logger.debug(f"Cloning repo to: {repo_path}")
                result = subprocess.run(
                    ["git", "clone", "--quiet", repo_url, str(repo_path)],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if result.returncode != 0:
                    logger.error(f"Failed to clone repo: {result.stderr}")
                    return None

            return repo_path

        except subprocess.TimeoutExpired:
            logger.error("Git operation timed out")
            return None
        except Exception as e:
            logger.error(f"Error cloning/updating repo: {e}")
            return None

    async def _verify_commit_exists(self, repo_path: Path, commit_hash: str) -> bool:
        """Verify that commit exists in repository.

        Args:
            repo_path: Path to local git repository
            commit_hash: Git commit SHA

        Returns:
            True if commit exists, False otherwise
        """
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_path), "cat-file", "-e", commit_hash],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Error verifying commit: {e}")
            return False

    async def _get_commit_timestamp(self, repo_path: Path, commit_hash: str) -> Optional[float]:
        """Get Unix timestamp of commit.

        Args:
            repo_path: Path to local git repository
            commit_hash: Git commit SHA

        Returns:
            Unix timestamp, or None if failed
        """
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_path), "show", "-s", "--format=%ct", commit_hash],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                return None

            return float(result.stdout.strip())

        except Exception as e:
            logger.error(f"Error getting commit timestamp: {e}")
            return None

    async def _extract_pack_from_commit(
        self,
        repo_path: Path,
        commit_hash: str
    ) -> Optional[dict]:
        """Extract policy pack from git commit.

        Looks for pack.json or constructs pack from AGENTS.md, SOUL.md, etc.

        Args:
            repo_path: Path to local git repository
            commit_hash: Git commit SHA

        Returns:
            Parsed pack dict, or None if failed
        """
        try:
            # First try to find pack.json
            result = subprocess.run(
                ["git", "-C", str(repo_path), "show", f"{commit_hash}:pack.json"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                # Found pack.json
                pack = json.loads(result.stdout)
                logger.debug("Extracted pack from pack.json")
                return pack

            # Otherwise, construct pack from individual files
            logger.debug("pack.json not found, constructing from files")
            pack = {
                "schema_version": 1,
                "files": {},
                "tool_policy": {},
                "metadata": {}
            }

            # Extract AGENTS.md
            result = subprocess.run(
                ["git", "-C", str(repo_path), "show", f"{commit_hash}:AGENTS.md"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                pack["files"]["AGENTS.md"] = result.stdout

            # Extract SOUL.md (optional)
            result = subprocess.run(
                ["git", "-C", str(repo_path), "show", f"{commit_hash}:SOUL.md"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                pack["files"]["SOUL.md"] = result.stdout

            # Extract tool_policy.json (optional)
            result = subprocess.run(
                ["git", "-C", str(repo_path), "show", f"{commit_hash}:tool_policy.json"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                pack["tool_policy"] = json.loads(result.stdout)

            if not pack["files"]:
                logger.warning("No pack files found in commit")
                return None

            return pack

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse pack JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Error extracting pack from commit: {e}")
            return None
