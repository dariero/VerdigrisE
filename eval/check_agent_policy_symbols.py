"""Fail when agent-policy prose names a Python symbol that no tracked module binds.

Ruff, mypy, the free suite, and CODEOWNERS review all read source. None of them
reads prose against source, so a procedure file can name a constant that does not
exist and stay green forever. That is what happened to `_QUESTION_VECTORS`.

Scope is deliberately narrow, and the narrowness is the point: a check that fires
on prose produces false positives, and a gate that cries wolf gets disabled. This
one only inspects inline code spans in agent-policy Markdown whose shape is
unambiguously a Python symbol in this repository's house style, and it treats a
token as known if any tracked module binds it, quotes it as a string literal, or
Python itself provides it as a builtin.

What this does NOT cover, stated plainly so nobody mistakes it for more:

- Lowercase single-word spans such as `addopts`, `openai`, or `main`. In this
  repository those are overwhelmingly TOML keys, pytest markers, and prose, so
  checking them would be almost entirely false positives.
- Dotted or called forms such as `NumpyVectorIndex.corpus_sha256(...)`; only the
  bare identifier shapes below are extracted.
- Fenced code blocks, which hold shell commands rather than symbol references.
- `README.md` and `SECURITY.md`, which are not agent policy.
- Whether a named symbol is the *correct* one. A reference to a real symbol that
  is nonetheless the wrong symbol for the sentence still passes.
"""

from __future__ import annotations

import ast
import builtins
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Backticked spans whose shape reads as a Python symbol but which name something
# outside this repository. Every entry is a deliberate exemption, not a default:
# an unlisted external name fails the check and has to be added here on purpose,
# which keeps this list an auditable ledger rather than a silent escape hatch.
EXTERNAL_NAMES = frozenset(
    {
        "CodeQL",  # GitHub's analysis product, named in the ship procedure.
    }
)

# Inline code spans only. A fenced block holds shell, not symbol references.
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_FENCE = re.compile(r"^\s*```")

# The three shapes that are unambiguously a Python symbol in this repository's
# prose. Bare lowercase words are excluded on purpose; see the module docstring.
_SYMBOL_SHAPES = (
    re.compile(r"^[A-Z][a-zA-Z0-9]*[a-z][a-zA-Z0-9]*$"),  # PascalCase, e.g. RagRecord
    re.compile(r"^_?[A-Z][A-Z0-9]*(_[A-Z0-9]+)+$"),  # SCREAMING_SNAKE, e.g. TOP_K
    re.compile(r"^_[A-Za-z][A-Za-z0-9_]*$"),  # leading underscore, e.g. _CORPUS_VECTORS
)


def tracked_files() -> list[str]:
    """Return every path git tracks, so the check follows the repository, not a glob."""

    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.split()


def agent_policy_files(tracked: list[str] | None = None) -> list[str]:
    """Return the tracked Markdown files that carry agent policy."""

    paths = tracked_files() if tracked is None else tracked
    return sorted(
        path
        for path in paths
        if path == "AGENTS.md" or (path.startswith(".agents/") and path.endswith(".md"))
    )


def known_symbols(tracked: list[str] | None = None) -> set[str]:
    """Return every name a tracked module binds, quotes, or inherits from builtins.

    String literals count because policy prose legitimately names configured
    values, not just identifiers: `INSUFFICIENT_CONTEXT` is the value of
    ABSTENTION_PHRASE rather than a symbol, and it is a real reference.
    """

    paths = tracked_files() if tracked is None else tracked
    known: set[str] = set(dir(builtins))
    for path in (path for path in paths if path.endswith(".py")):
        tree = ast.parse((PROJECT_ROOT / path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                known.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                known.add(node.id)
            elif isinstance(node, ast.arg):
                known.add(node.arg)
            elif isinstance(node, ast.Attribute):
                known.add(node.attr)
            elif isinstance(node, ast.Import | ast.ImportFrom):
                for alias in node.names:
                    known.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                known.add(node.value)
    return known


def violations() -> list[tuple[str, int, str]]:
    """Return every agent-policy reference to a symbol nothing tracked provides."""

    tracked = tracked_files()
    known = known_symbols(tracked) | EXTERNAL_NAMES
    found: list[tuple[str, int, str]] = []
    for path in agent_policy_files(tracked):
        inside_fence = False
        lines = (PROJECT_ROOT / path).read_text(encoding="utf-8").splitlines()
        for lineno, line in enumerate(lines, start=1):
            if _FENCE.match(line):
                inside_fence = not inside_fence
                continue
            if inside_fence:
                continue
            for token in _INLINE_CODE.findall(line):
                if not any(shape.match(token) for shape in _SYMBOL_SHAPES):
                    continue
                if token not in known:
                    found.append((path, lineno, token))
    return found


def main() -> int:
    found = violations()
    for path, lineno, token in found:
        print(f"{path}:{lineno}: {token} is not bound by any tracked module")
    if found:
        print(
            f"\n{len(found)} agent-policy symbol reference(s) resolve to nothing. "
            "Correct the reference, or add a deliberate entry to EXTERNAL_NAMES "
            "in eval/check_agent_policy_symbols.py if the name is external.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
