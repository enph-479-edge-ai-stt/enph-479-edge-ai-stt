"""Regression tests for the launcher notebooks in notebooks/.

These are static/structural checks, not executions: the notebooks clone the
repo and download LibriSpeech, so running them in CI is not feasible. We guard
the things that actually break on edits — valid JSON, code cells that parse,
and (the bug this suite was born from) clone URLs that drift away from the
repo's real origin. Uses stdlib json only, so it adds no dependency.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]  # the training/ project root
NOTEBOOK_DIR = PROJECT_DIR / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))

# A github .git URL — i.e. a clone target, not an issue/browse link.
CLONE_URL_RE = re.compile(r"github\.com[/:]([\w.-]+)/([\w.-]+?)\.git")


def _load(nb_path: Path) -> dict:
    return json.loads(nb_path.read_text(encoding="utf-8"))


def _cells(nb: dict):
    """Yield (cell_dict, joined_source) for every cell."""
    for cell in nb.get("cells", []):
        source = cell.get("source", "")
        yield cell, "".join(source) if isinstance(source, list) else source


def _origin_slug() -> tuple[str, str] | None:
    """(owner, repo) for the git origin remote, lowercased, or None if unknown."""
    try:
        url = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    m = CLONE_URL_RE.search(url) or re.search(r"github\.com[/:]([\w.-]+)/([\w.-]+)", url)
    return (m.group(1).lower(), m.group(2).lower()) if m else None


def test_notebooks_are_discovered():
    """Guard against the glob silently matching nothing (moved/renamed dir)."""
    assert NOTEBOOKS, f"no notebooks found under {NOTEBOOK_DIR}"


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid_notebook_json(nb_path: Path):
    nb = _load(nb_path)
    assert isinstance(nb.get("cells"), list), "missing cells list"
    assert "nbformat" in nb, "missing nbformat key"


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_code_cells_parse(nb_path: Path):
    """Every plain-Python code cell must compile (catches syntax regressions).

    Cells using IPython magics or shell escapes (``%%bash``, ``!cmd``) are not
    valid standalone Python, so they are skipped.
    """
    nb = _load(nb_path)
    for i, (cell, source) in enumerate(_cells(nb)):
        if cell.get("cell_type") != "code":
            continue
        if any(line.lstrip().startswith(("%", "!")) for line in source.splitlines()):
            continue
        compile(source, f"{nb_path.name}[cell {i}]", "exec")


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_clone_urls_match_origin(nb_path: Path):
    """Any github clone URL in a code cell must point at the repo's origin."""
    origin = _origin_slug()
    if origin is None:
        pytest.skip("git origin not available")
    nb = _load(nb_path)
    for cell, source in _cells(nb):
        if cell.get("cell_type") != "code":
            continue
        for owner, repo in CLONE_URL_RE.findall(source):
            assert (owner.lower(), repo.lower()) == origin, (
                f"{nb_path.name} clones github.com/{owner}/{repo}, "
                f"but origin is github.com/{origin[0]}/{origin[1]}"
            )
