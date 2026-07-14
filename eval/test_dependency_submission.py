from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import quote

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_SCRIPT = REPOSITORY_ROOT / ".github" / "scripts" / "build_dependency_snapshot.py"
FIXED_SHA = "0123456789abcdef0123456789abcdef01234567"
FIXED_SCANNED = "2026-07-14T00:00:00Z"
HASH = "a" * 64


def load_snapshot_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_dependency_snapshot", SNAPSHOT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SNAPSHOT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


snapshot_module = load_snapshot_module()


def repository_snapshot() -> dict[str, Any]:
    return snapshot_module.build_snapshot(
        REPOSITORY_ROOT / "pyproject.toml",
        REPOSITORY_ROOT / "pylock.toml",
        sha=FIXED_SHA,
        ref="refs/heads/main",
        job_id="test-job",
        scanned=FIXED_SCANNED,
    )


def write_toml_fixture(tmp_path: Path, *, project: str, lock: str) -> tuple[Path, Path]:
    pyproject_path = tmp_path / "pyproject.toml"
    pylock_path = tmp_path / "pylock.toml"
    pyproject_path.write_text(project, encoding="utf-8")
    pylock_path.write_text(lock, encoding="utf-8")
    return pyproject_path, pylock_path


def minimal_lock(package_block: str) -> str:
    return f"""lock-version = "1.0"
requires-python = "==3.14.*"

[[packages]]
name = "example"
version = "1.0.0"
index = "https://pypi.org/simple"
{package_block}
"""


def test_snapshot_exposes_every_exact_locked_package() -> None:
    snapshot = repository_snapshot()
    resolved = snapshot["manifests"]["pylock.toml"]["resolved"]
    with (REPOSITORY_ROOT / "pylock.toml").open("rb") as handle:
        packages = tomllib.load(handle)["packages"]

    expected_purls = {
        "pkg:pypi/"
        f"{quote(snapshot_module.normalize_name(package['name']), safe='')}@"
        f"{quote(package['version'], safe='')}"
        for package in packages
    }
    assert len(resolved) == len(packages) == 55
    assert {dependency["package_url"] for dependency in resolved.values()} == expected_purls
    assert sum(node["relationship"] == "direct" for node in resolved.values()) == 10
    assert sum(node["relationship"] == "indirect" for node in resolved.values()) == 45


def test_snapshot_scopes_only_declared_roots_and_preserves_markers() -> None:
    resolved = repository_snapshot()["manifests"]["pylock.toml"]["resolved"]

    runtime = {name for name, node in resolved.items() if node.get("scope") == "runtime"}
    development = {name for name, node in resolved.items() if node.get("scope") == "development"}
    assert runtime == {"numpy", "openai", "pydantic"}
    assert development == {
        "mypy",
        "pre-commit",
        "pre-commit-hooks",
        "pytest",
        "pytest-cov",
        "ragaliq",
        "ruff",
    }
    assert all(
        "scope" not in node for node in resolved.values() if node["relationship"] == "indirect"
    )
    assert all("dependencies" not in node for node in resolved.values())
    assert resolved["colorama"]["metadata"] == {"marker": "sys_platform == 'win32'"}


def test_snapshot_is_deterministic_for_fixed_envelope() -> None:
    first = json.dumps(repository_snapshot(), indent=2, sort_keys=True)
    second = json.dumps(repository_snapshot(), indent=2, sort_keys=True)
    assert first == second


def test_snapshot_rejects_an_unhashed_artifact(tmp_path: Path) -> None:
    pyproject_path, pylock_path = write_toml_fixture(
        tmp_path,
        project="""[project]
dependencies = ["example>=1"]

[dependency-groups]
dev = []
""",
        lock=minimal_lock(
            'wheels = [{ url = "https://files.pythonhosted.org/example.whl", hashes = {} }]'
        ),
    )

    with pytest.raises(snapshot_module.SnapshotError, match="without a SHA-256 hash"):
        snapshot_module.build_snapshot(
            pyproject_path,
            pylock_path,
            sha=FIXED_SHA,
            ref="refs/heads/main",
            job_id="test-job",
            scanned=FIXED_SCANNED,
        )


def test_snapshot_rejects_a_direct_requirement_missing_from_lock(tmp_path: Path) -> None:
    pyproject_path, pylock_path = write_toml_fixture(
        tmp_path,
        project="""[project]
dependencies = ["missing>=1"]

[dependency-groups]
dev = []
""",
        lock=minimal_lock(
            f'wheels = [{{ url = "https://files.pythonhosted.org/example.whl", hashes = {{ sha256 = "{HASH}" }} }}]'
        ),
    )

    with pytest.raises(snapshot_module.SnapshotError, match="missing from pylock.toml: missing"):
        snapshot_module.build_snapshot(
            pyproject_path,
            pylock_path,
            sha=FIXED_SHA,
            ref="refs/heads/main",
            job_id="test-job",
            scanned=FIXED_SCANNED,
        )


def test_snapshot_refuses_to_ignore_future_lock_edges(tmp_path: Path) -> None:
    pyproject_path, pylock_path = write_toml_fixture(
        tmp_path,
        project="""[project]
dependencies = ["example>=1"]

[dependency-groups]
dev = []
""",
        lock=minimal_lock(
            f'''dependencies = []
wheels = [{{ url = "https://files.pythonhosted.org/example.whl", hashes = {{ sha256 = "{HASH}" }} }}]'''
        ),
    )

    with pytest.raises(snapshot_module.SnapshotError, match="contains dependency edges"):
        snapshot_module.build_snapshot(
            pyproject_path,
            pylock_path,
            sha=FIXED_SHA,
            ref="refs/heads/main",
            job_id="test-job",
            scanned=FIXED_SCANNED,
        )
