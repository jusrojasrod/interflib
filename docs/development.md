# Development

```bash
pip install -e ".[dev,tools,docs]"
pytest                                  # full suite
ruff check src tests tools              # lint rules in pyproject.toml
python tools/make_readme_figures.py     # regenerate docs/images/
```

## Tests

Every method has tests in `tests/`. Where an analytic solution or a published reference
value exists, the test checks against it: the reference table of the NIST air-index
calculator, the four-wavelength measurement of Liu et al. (2022), simulated scenes with a
known truth. `tests/test_docs_examples.py` runs the Python examples of the README and of
every page of the user guide, and checks that they print what their `# ->` comments say.
A change that alters a printed value must update the page in the same commit.

## Notebooks

The notebooks are documentation of the methods: when a change alters what a notebook
explains, update the notebook in the same pass and run it again. They are committed without
outputs (the `nbstripout` filter in `.gitattributes`), so install the filter once per
clone:

```bash
nbstripout --install
```

A notebook that matters is named `YYYYMMDD-topic.ipynb`; the dated notebooks are executed
when this site is built, and a notebook that fails stops the build. Nothing that matters
lives only in a notebook: once a method works, it moves into the package with its tests.

## This documentation

The site is built with Sphinx from `docs/`. The API reference comes from the docstrings
(NumPy style), the guide pages are Markdown (MyST), and the notebooks are executed by
myst-nb.

```bash
sphinx-build -W --keep-going -b html docs docs/_build/html                  # full build, notebooks executed
NB_EXECUTION_MODE=off sphinx-build -b html docs docs/_build/html            # quick build, notebooks not run
```

Open `docs/_build/html/index.html`. Warnings are errors in the continuous integration,
which builds the site on every push and pull request and publishes it from `main` to
<https://jusrojasrod.github.io/interflib/>.

## Conventions for the code

- Docstrings in English, NumPy style. A step, algorithm or value that rests on a source
  cites that source directly in the docstring.
- Physical constants go in `interflib.constants`, each with its source in a comment.
- The library takes and returns arrays and dataclasses; it does not print or write files.
- Commit messages are `<scope>: <description>`, with the scope one of `code`, `data`,
  `meta`.
