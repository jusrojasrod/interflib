"""Sphinx configuration of the interflib documentation.

Build from the repository root with ``sphinx-build -b html docs docs/_build/html``. The
dated notebooks are copied into ``docs/notebooks/_generated/`` and executed during the
build; set ``NB_EXECUTION_MODE=off`` for a quick build without running them.
"""
import os
import shutil
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))   # the scripts of tools/, for the command-line reference

import interflib  # noqa: E402

# -- Project ---------------------------------------------------------------------------
project = "interflib"
author = "Juan Sebastián Rojas Rodríguez"
copyright = "2026, Instituto Nacional de Metrología de Colombia (INM)"
version = release = interflib.__version__

# -- General ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "myst_nb",
    "sphinxarg.ext",
    "sphinx_copybutton",
]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints", "jupyter_execute"]
default_role = "code"

# -- API reference ---------------------------------------------------------------------
autosummary_generate = True
autodoc_default_options = {"members": True, "show-inheritance": True}
autodoc_member_order = "bysource"
autodoc_typehints = "none"            # the docstrings give the types and units
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_preprocess_types = True      # "array_like, optional" and {"a", "b"} as types, not references
napoleon_type_aliases = {
    "array_like": ":term:`array_like <numpy:array_like>`",
    "sequence": ":term:`sequence <python:sequence>`",
}
nitpick_ignore = [("py:class", "optional")]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- Markdown and notebooks ------------------------------------------------------------
myst_enable_extensions = ["dollarmath", "colon_fence", "deflist"]
myst_heading_anchors = 3
nb_execution_mode = os.environ.get("NB_EXECUTION_MODE", "force")
nb_execution_timeout = 900
nb_execution_raise_on_error = True
nb_merge_streams = True

NOTEBOOKS = ROOT / "notebooks"
GENERATED = DOCS / "notebooks" / "_generated"
GENERATED.mkdir(parents=True, exist_ok=True)
for notebook in sorted(NOTEBOOKS.glob("2026*.ipynb")):   # the dated notebooks; 00-08 are the legacy prototype
    target = GENERATED / notebook.name
    if not target.exists() or target.read_bytes() != notebook.read_bytes():
        shutil.copy2(notebook, target)

# -- HTML ------------------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = f"interflib {release}"
html_static_path = []
html_theme_options = {
    "github_url": "https://github.com/jusrojasrod/interflib",
    "navbar_align": "left",
    "show_toc_level": 2,
    "secondary_sidebar_items": ["page-toc"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
}
html_show_sourcelink = False
