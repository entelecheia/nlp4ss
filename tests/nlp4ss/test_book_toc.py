"""Every page a book TOC lists must exist (#319)."""

import re
from pathlib import Path

import pytest

BOOK = Path(__file__).parents[2] / "book"


@pytest.mark.parametrize("lang", ["en", "ko"])
def test_toc_entries_exist(lang: str) -> None:
    root = BOOK / lang
    entries = re.findall(
        r"^\s*(?:-\s*)?(?:file|root):\s*(\S+)", (root / "_toc.yml").read_text(), re.M
    )
    assert entries
    missing = [
        e for e in entries if not any((root / f"{e}{ext}").is_file() for ext in (".md", ".ipynb"))
    ]
    assert missing == []
