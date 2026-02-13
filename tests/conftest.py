"""Pytest fixtures."""
import pytest
@pytest.fixture
def sample_pack():
    return {"schema_version": 1, "files": {"AGENTS.md": "# Test"}, "tool_policy": {"allow": ["exec"]}, "metadata": {"pack_name": "test"}}
