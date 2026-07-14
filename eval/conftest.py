"""Configure flat-module imports and fail-closed paid-test credentials."""

import os
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_PAID_MARKER_CREDENTIALS = {
    "openai": "OPENAI_API_KEY",
    "rag_test": "ANTHROPIC_API_KEY",
}


def _missing_paid_credentials(
    items: Iterable[pytest.Item], environment: Mapping[str, str]
) -> list[str]:
    """Return missing credential names required by the selected paid nodes."""

    required = {
        credential
        for item in items
        for marker, credential in _PAID_MARKER_CREDENTIALS.items()
        if item.get_closest_marker(marker) is not None
    }
    return sorted(credential for credential in required if not environment.get(credential))


def pytest_collection_finish(session: pytest.Session) -> None:
    """Fail selected paid tiers before fixtures when credentials are absent."""

    if session.config.option.collectonly:
        return

    missing = _missing_paid_credentials(session.items, os.environ)
    if missing:
        raise pytest.UsageError(
            "Explicitly selected paid tests require non-empty environment variables: "
            + ", ".join(missing)
        )
