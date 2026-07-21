import subprocess
import sys

import barbican


def test_public_surface_excludes_corpus():
    # eval-only corpus generator must NOT be part of the public package surface
    assert not hasattr(barbican, "generate_synthetic")
    assert not hasattr(barbican, "CampaignSpec")
    assert hasattr(barbican, "BaselineDetector") and hasattr(barbican, "run_baseline")


def test_public_import_is_standalone_no_oubliette_shield():
    # Crown-jewel boundary invariant: importing the standalone barbican package
    # must NEVER load oubliette_shield (and therefore never the private SPECTRE
    # framework). Run in a clean subprocess so no other test's prior import can
    # mask a violation.
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import barbican\n"
            "import sys\n"
            "print('oubliette_shield' in sys.modules)\n",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False", (
        f"barbican import pulled in oubliette_shield: "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
