"""``barbican.__version__`` must match ``[project].version`` in pyproject.toml.

Published wheels advertise the pyproject version; code that reports its own
version reads ``barbican.__version__``. Before this test, the package had no
``__version__`` at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import barbican

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    tomllib = None  # type: ignore[assignment]

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


@pytest.mark.skipif(tomllib is None, reason="tomllib requires Python 3.11+")
def test_runtime_version_matches_pyproject() -> None:
    with PYPROJECT.open("rb") as fh:
        declared = tomllib.load(fh)["project"]["version"]
    assert isinstance(declared, str) and declared, declared
    assert barbican.__version__ == declared, (
        f"version drift: barbican.__version__={barbican.__version__!r} "
        f"!= pyproject.toml project.version={declared!r}"
    )
