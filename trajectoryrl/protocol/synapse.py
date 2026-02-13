"""Synapse definitions for TrajectoryRL subnet communication.

Based on Bittensor synapse patterns:
- https://docs.learnbittensor.org/learn/neurons
- https://docs.learnbittensor.org/tutorials/ocr-subnet-tutorial
"""

import hashlib
import json
from typing import Any, Dict, Optional

import bittensor as bt
from pydantic import Field, field_validator


class PackRequest(bt.Synapse):
    """Validator requests the miner's current policy pack.

    Attributes:
        suite_id: Target task suite (e.g., "clawbench_v1")
        schema_version: OPP schema version (currently 1)
        max_bytes: Maximum inline payload size
        want_pointer_ok: If True, miner can return URL instead of inline
    """

    suite_id: str = Field(
        default="clawbench_v1",
        description="Target task suite identifier"
    )
    schema_version: int = Field(
        default=1,
        description="OpenClaw Policy Pack schema version"
    )
    max_bytes: int = Field(
        default=65536,  # 64KB
        description="Maximum size for inline pack_b64 payload"
    )
    want_pointer_ok: bool = Field(
        default=True,
        description="Whether validator accepts pack_url instead of inline"
    )


class PackResponse(bt.Synapse):
    """Miner returns their policy pack (GitHub-based submission).

    Attributes:
        pack_hash: SHA256 hash of the pack JSON (hex digest)
        git_commit_hash: Git commit SHA from public repository
        repo_url: Public GitHub repository URL
        metadata: Declared pack metadata (not verified by validator)
    """

    pack_hash: str = Field(
        default="",
        description="SHA256 hash of pack content (hex digest)"
    )
    git_commit_hash: str = Field(
        default="",
        description="Git commit SHA from public repository (40-char hex)"
    )
    repo_url: str = Field(
        default="",
        description="Public GitHub repository URL (e.g., https://github.com/user/repo)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Pack metadata (author, version, target_suite, etc.)"
    )

    @field_validator("pack_hash")
    @classmethod
    def validate_pack_hash(cls, v: str) -> str:
        """Ensure pack_hash is a valid SHA256 hex digest."""
        if v and len(v) != 64:
            raise ValueError(f"pack_hash must be 64 hex chars, got {len(v)}")
        if v and not all(c in "0123456789abcdef" for c in v.lower()):
            raise ValueError("pack_hash must be lowercase hex")
        return v.lower()

    @field_validator("git_commit_hash")
    @classmethod
    def validate_git_commit_hash(cls, v: str) -> str:
        """Ensure git_commit_hash is a valid git SHA (40-char hex)."""
        if v and len(v) != 40:
            raise ValueError(f"git_commit_hash must be 40 hex chars, got {len(v)}")
        if v and not all(c in "0123456789abcdef" for c in v.lower()):
            raise ValueError("git_commit_hash must be lowercase hex")
        return v.lower()

    @field_validator("repo_url")
    @classmethod
    def validate_repo_url(cls, v: str) -> str:
        """Ensure repo_url is a valid GitHub URL."""
        if v and not v.startswith(("https://github.com/", "http://github.com/")):
            raise ValueError("repo_url must be a GitHub URL")
        return v

    @staticmethod
    def compute_pack_hash(pack: dict) -> str:
        """Compute SHA256 hash from pack dict.

        Args:
            pack: Policy pack dictionary

        Returns:
            Hex digest of SHA256 hash
        """
        content = json.dumps(pack, sort_keys=True).encode()
        return hashlib.sha256(content).hexdigest()
