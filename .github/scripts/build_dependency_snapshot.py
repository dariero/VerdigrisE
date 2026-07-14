"""Build a GitHub dependency snapshot from VerdigrisE's committed lock contract."""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

LOCK_VERSION = "1.0"
PYTHON_REQUIREMENT = "==3.14.*"
PUBLIC_INDEX = "https://pypi.org/simple"
PUBLIC_ARTIFACT_HOST = "files.pythonhosted.org"
DETECTOR_NAME = "verdigrise-pylock"
DETECTOR_VERSION = "1.0.0"
DETECTOR_URL = "https://github.com/dariero/VerdigrisE"
JOB_CORRELATOR = "dependency-submission / submit-pylock"

_NAME_SEPARATOR = re.compile(r"[-_.]+")
_REQUIREMENT_NAME = re.compile(r"\A\s*([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)")
_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
_GIT_SHA = re.compile(r"\A[0-9a-fA-F]{40}\Z")


class SnapshotError(ValueError):
    """Raised when committed dependency data cannot be represented exactly."""


def normalize_name(name: str) -> str:
    """Return the canonical PyPI project name used in package URLs."""

    return _NAME_SEPARATOR.sub("-", name).lower()


def requirement_name(requirement: str) -> str:
    """Extract and validate the distribution name from a PEP 508 requirement."""

    match = _REQUIREMENT_NAME.match(requirement)
    if match is None:
        raise SnapshotError(f"unsupported requirement: {requirement!r}")

    tail = requirement[match.end() :].lstrip()
    if tail and tail[0] not in "[<>=!~;":
        raise SnapshotError(f"unsupported requirement: {requirement!r}")
    return normalize_name(match.group(1))


def load_toml(path: Path) -> dict[str, Any]:
    """Load one TOML document and preserve a useful path in parse failures."""

    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise SnapshotError(f"cannot read {path}: {error}") from error
    if not isinstance(data, dict):
        raise SnapshotError(f"{path} must contain a TOML table")
    return data


def direct_scopes(pyproject: dict[str, Any]) -> dict[str, str]:
    """Map declared project roots to GitHub's runtime/development scopes."""

    project = pyproject.get("project")
    if not isinstance(project, dict):
        raise SnapshotError("pyproject.toml must contain [project]")

    runtime_requirements = project.get("dependencies")
    if not isinstance(runtime_requirements, list) or not all(
        isinstance(item, str) for item in runtime_requirements
    ):
        raise SnapshotError("project.dependencies must be a list of requirement strings")

    groups = pyproject.get("dependency-groups")
    if not isinstance(groups, dict):
        raise SnapshotError("pyproject.toml must contain [dependency-groups]")

    scopes = {requirement_name(item): "runtime" for item in runtime_requirements}
    for group_name, requirements in groups.items():
        if not isinstance(requirements, list) or not all(
            isinstance(item, str) for item in requirements
        ):
            raise SnapshotError(
                f"dependency group {group_name!r} must contain only requirement strings"
            )
        for requirement in requirements:
            scopes.setdefault(requirement_name(requirement), "development")
    return scopes


def validate_artifact(package_name: str, artifact: object) -> None:
    """Require every install artifact to be a hash-bearing public PyPI file."""

    if not isinstance(artifact, dict):
        raise SnapshotError(f"{package_name} contains a malformed artifact")

    url = artifact.get("url")
    if not isinstance(url, str):
        raise SnapshotError(f"{package_name} contains an artifact without a URL")
    parsed_url = urlparse(url)
    if parsed_url.scheme != "https" or parsed_url.hostname != PUBLIC_ARTIFACT_HOST:
        raise SnapshotError(f"{package_name} contains a non-public artifact URL: {url}")

    hashes = artifact.get("hashes")
    sha256 = hashes.get("sha256") if isinstance(hashes, dict) else None
    if not isinstance(sha256, str) or _SHA256.fullmatch(sha256) is None:
        raise SnapshotError(f"{package_name} contains an artifact without a SHA-256 hash")


def locked_dependencies(pylock: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return exact GitHub dependency nodes from the PEP 751 package inventory."""

    if pylock.get("lock-version") != LOCK_VERSION:
        raise SnapshotError(f"pylock.toml must use lock-version {LOCK_VERSION!r}")
    if pylock.get("requires-python") != PYTHON_REQUIREMENT:
        raise SnapshotError(f"pylock.toml must require Python {PYTHON_REQUIREMENT!r}")

    packages = pylock.get("packages")
    if not isinstance(packages, list) or not packages:
        raise SnapshotError("pylock.toml must contain at least one package")

    resolved: dict[str, dict[str, Any]] = {}
    for package in packages:
        if not isinstance(package, dict):
            raise SnapshotError("pylock.toml contains a malformed package")

        raw_name = package.get("name")
        version = package.get("version")
        if not isinstance(raw_name, str) or not raw_name:
            raise SnapshotError("pylock.toml contains a package without a name")
        if not isinstance(version, str) or not version:
            raise SnapshotError(f"{raw_name} does not have an exact version")

        name = normalize_name(raw_name)
        if name in resolved:
            raise SnapshotError(f"pylock.toml contains duplicate package {name!r}")
        if package.get("index") != PUBLIC_INDEX:
            raise SnapshotError(f"{name} is not locked from public PyPI")
        if "dependencies" in package:
            raise SnapshotError(
                f"{name} contains dependency edges that detector {DETECTOR_VERSION} cannot submit"
            )
        if any(source in package for source in ("archive", "directory", "vcs")):
            raise SnapshotError(f"{name} uses an unsupported non-index source")

        artifacts: list[object] = []
        if "sdist" in package:
            artifacts.append(package["sdist"])
        wheels = package.get("wheels", [])
        if not isinstance(wheels, list):
            raise SnapshotError(f"{name} contains a malformed wheel list")
        artifacts.extend(wheels)
        if not artifacts:
            raise SnapshotError(f"{name} does not contain a hash-bearing artifact")
        for artifact in artifacts:
            validate_artifact(name, artifact)

        dependency: dict[str, Any] = {
            "package_url": f"pkg:pypi/{quote(name, safe='')}@{quote(version, safe='')}",
        }
        marker = package.get("marker")
        if marker is not None:
            if not isinstance(marker, str) or not marker:
                raise SnapshotError(f"{name} contains a malformed environment marker")
            dependency["metadata"] = {"marker": marker}
        resolved[name] = dependency

    return dict(sorted(resolved.items()))


def build_snapshot(
    pyproject_path: Path,
    pylock_path: Path,
    *,
    sha: str,
    ref: str,
    job_id: str,
    scanned: str,
    job_url: str | None = None,
) -> dict[str, Any]:
    """Build one dependency-submission API payload from committed repository files."""

    if _GIT_SHA.fullmatch(sha) is None:
        raise SnapshotError("sha must be a full 40-character Git commit")
    if not ref.startswith("refs/"):
        raise SnapshotError("ref must be a fully qualified Git reference")
    if not job_id:
        raise SnapshotError("job id must not be empty")

    scopes = direct_scopes(load_toml(pyproject_path))
    resolved = locked_dependencies(load_toml(pylock_path))
    missing_roots = sorted(scopes.keys() - resolved.keys())
    if missing_roots:
        raise SnapshotError(
            "direct requirements missing from pylock.toml: " + ", ".join(missing_roots)
        )

    for name, dependency in resolved.items():
        if name in scopes:
            dependency["relationship"] = "direct"
            dependency["scope"] = scopes[name]
        else:
            dependency["relationship"] = "indirect"

    job: dict[str, str] = {"correlator": JOB_CORRELATOR, "id": job_id}
    if job_url:
        job["html_url"] = job_url

    return {
        "version": 0,
        "sha": sha.lower(),
        "ref": ref,
        "job": job,
        "detector": {
            "name": DETECTOR_NAME,
            "version": DETECTOR_VERSION,
            "url": DETECTOR_URL,
        },
        "scanned": scanned,
        "manifests": {
            "pylock.toml": {
                "name": "Python 3.14 exact environment",
                "file": {"source_location": "pylock.toml"},
                "metadata": {
                    "direct_requirements": "pyproject.toml",
                    "locked_packages": len(resolved),
                    "requires_python": PYTHON_REQUIREMENT,
                },
                "resolved": resolved,
            }
        },
    }


def parse_args() -> argparse.Namespace:
    """Parse the repository-specific command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--pylock", type=Path, default=Path("pylock.toml"))
    parser.add_argument("--sha", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--job-url")
    parser.add_argument("--scanned")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Write a deterministic snapshot document for GitHub submission."""

    args = parse_args()
    scanned = args.scanned or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    snapshot = build_snapshot(
        args.pyproject,
        args.pylock,
        sha=args.sha,
        ref=args.ref,
        job_id=args.job_id,
        scanned=scanned,
        job_url=args.job_url,
    )
    args.output.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
