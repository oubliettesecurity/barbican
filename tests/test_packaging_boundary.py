"""The published artifacts must contain only the ``barbican`` package.

``oubliette-barbican`` is a public PyPI package (import name ``barbican``). The
repo also holds things that have no place in it: ``tests/``, ``examples/``,
CI config and any local ``.env`` or key material. The offensive twin
(SPECTRE) never lives in this repo; ``spectre`` in any artifact path fails.

This is an allowlist check: anything outside the expected layout fails.
This module deliberately imports nothing from ``barbican`` so the publish
workflow can run it with ``--noconftest`` in a venv that has only build
tooling installed. The package is built with hatchling, so the config check
reads ``[tool.hatch.build.targets]``.

The artifact tests skip when ``dist/`` is empty so they never block a plain
test run. Set ``BARBICAN_REQUIRE_DIST=1`` (the publish workflow and the CI build job do) to
make a missing build a hard failure instead of a silent skip.
"""

from __future__ import annotations

import os
import re
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "barbican"
# Normalized distribution name, as used in the wheel dist-info dir.
DIST = "oubliette_barbican"

# Path fragments that must never appear anywhere in a built artifact.
FORBIDDEN_FRAGMENTS = (
    "/tests/",
    "/examples/",
    "/spectre",
    "/.git/",
    "/.github/",
    "/node_modules/",
)
FORBIDDEN_BASENAMES = re.compile(
    r"(^|/)(\.env(\..*)?|.*\.pem|.*\.key|id_rsa.*|.*\.sqlite3?|.*\.db)$"
)
SDIST_ALLOWED_TOP = {"README.md", "pyproject.toml", "PKG-INFO", ".gitignore"}


def _dist_artifacts() -> tuple[list[Path], list[Path]]:
    dist = REPO_ROOT / "dist"
    return sorted(dist.glob("*.whl")), sorted(dist.glob("*.tar.gz"))


def _require_or_skip(artifacts: list[Path], kind: str) -> None:
    if artifacts:
        return
    if os.getenv("BARBICAN_REQUIRE_DIST") == "1":
        pytest.fail(f"BARBICAN_REQUIRE_DIST=1 but no {kind} found in dist/")
    pytest.skip(f"no {kind} in dist/ -- nothing to inspect")


def _generic_offenders(names: list[str]) -> list[str]:
    bad = []
    for n in names:
        probe = "/" + n
        if any(frag in probe for frag in FORBIDDEN_FRAGMENTS) or FORBIDDEN_BASENAMES.search(n):
            bad.append(n)
    return bad


def test_hatch_config_only_ships_the_package() -> None:
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        targets = tomllib.load(fh)["tool"]["hatch"]["build"]["targets"]
    assert targets["wheel"]["packages"] == [f"src/{PACKAGE}"], targets["wheel"]
    for src, dst in targets["wheel"].get("force-include", {}).items():
        assert src.startswith(f"src/{PACKAGE}/"), f"force-include source outside package: {src}"
        assert dst.startswith(f"{PACKAGE}/"), f"force-include target outside package: {dst}"
    assert targets["sdist"]["only-include"] == [f"src/{PACKAGE}"], targets["sdist"]


def test_wheel_contains_only_the_package() -> None:
    wheels, _ = _dist_artifacts()
    _require_or_skip(wheels, "wheel")
    offenders: dict[str, list[str]] = {}
    dist_info = re.compile(rf"^{DIST}-[^/]+\.dist-info/")
    for whl in wheels:
        with zipfile.ZipFile(whl) as zf:
            names = zf.namelist()
        outside = [n for n in names if not (n.startswith(PACKAGE + "/") or dist_info.match(n))]
        bad = sorted(set(outside + _generic_offenders(names)))
        assert any(n.startswith(PACKAGE + "/") for n in names), f"{whl.name} has no package files"
        if bad:
            offenders[whl.name] = bad
    assert not offenders, f"wheel contains files outside {PACKAGE}/: {offenders}"


def test_sdist_contains_only_expected_files() -> None:
    _, sdists = _dist_artifacts()
    _require_or_skip(sdists, "sdist")
    offenders: dict[str, list[str]] = {}
    for sd in sdists:
        with tarfile.open(sd) as tf:
            members = [m.name for m in tf.getmembers() if m.isfile()]
        bad = []
        for name in members:
            _, _, rel = name.partition("/")  # strip "<name>-<version>/"
            ok = rel in SDIST_ALLOWED_TOP or rel.startswith(f"src/{PACKAGE}/")
            if not ok:
                bad.append(name)
        bad = sorted(set(bad + _generic_offenders(members)))
        if bad:
            offenders[sd.name] = bad
    assert not offenders, f"sdist contains unexpected files: {offenders}"
