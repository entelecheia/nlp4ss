"""The locked course stack keeps the spaCy CLI working (the labs run `python -m spacy download`)."""

import subprocess
import sys


def test_spacy_cli_starts() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "spacy", "--help"], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
