"""
test cli module
"""

import subprocess

from nlp4ss._version import __version__


def test_cli_version() -> None:
    """The installed entry point reports the package version."""
    proc = subprocess.run(["nlp4ss", "--version"], capture_output=True, text=True, check=False)
    assert proc.returncode == 0
    assert __version__ in proc.stdout
