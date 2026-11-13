"""The Python examples of the README and of the documentation run, and print what their comments say.

Every ``python`` code block of a page is executed in order, in one namespace per page, from the
repository root. A line ending in ``# -> text`` states what that block prints, line by line.
"""
import contextlib
import io
import re
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
PAGES = [p for p in PAGES if "_build" not in p.parts and "_generated" not in p.parts]
BLOCK = re.compile(r"```python\n(.*?)```", re.S)
EXPECTED = re.compile(r"#\s*->\s*(.+)$")


def _blocks(page):
    return BLOCK.findall(page.read_text(encoding="utf-8"))


@pytest.mark.parametrize("page", [p for p in PAGES if _blocks(p)], ids=lambda p: str(p.relative_to(ROOT)))
def test_examples_run_and_print_what_they_say(page, monkeypatch):
    monkeypatch.chdir(ROOT)
    namespace = {"__name__": "__docs__"}
    for code in _blocks(page):
        expected = [m.group(1).strip() for line in code.splitlines() if (m := EXPECTED.search(line))]
        out = io.StringIO()
        with contextlib.redirect_stdout(out), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exec(compile(code, str(page), "exec"), namespace)
        assert out.getvalue().splitlines() == expected, code


def test_readme_has_examples():
    assert _blocks(ROOT / "README.md"), "no python examples found in the README"


def test_pages_with_examples_are_found():
    assert len([p for p in PAGES if _blocks(p)]) >= 8, [str(p) for p in PAGES]
