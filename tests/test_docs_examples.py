"""The Python examples of the README run, in order, and print what their comments say."""
import contextlib
import io
import re
import warnings
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"
BLOCK = re.compile(r"```python\n(.*?)```", re.S)
EXPECTED = re.compile(r"#\s*->\s*(.+)$")


def test_readme_examples_run_and_print_what_they_say():
    namespace = {}
    blocks = BLOCK.findall(README.read_text(encoding="utf-8"))
    assert blocks, "no python examples found in the README"
    for code in blocks:
        expected = [m.group(1).strip() for line in code.splitlines() if (m := EXPECTED.search(line))]
        out = io.StringIO()
        with contextlib.redirect_stdout(out), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exec(compile(code, str(README), "exec"), namespace)
        assert out.getvalue().splitlines() == expected, code
