"""Sphinx configuration for stamm-sdk.

Build locally with:
    pip install -e ".[docs]"
    sphinx-build -b html docs docs/_build/html

Autodoc imports the real package to pull docstrings -- that works without
the [learning] extra installed because torch/tensorflow/sklearn/joblib are
all imported lazily inside the functions that need them (see
learning/federated_client.py), never at module level.
"""

from __future__ import annotations

import os
import sys

# So autodoc can `import stamm_sdk` without the package being pip-installed.
sys.path.insert(0, os.path.abspath("../src"))

project = "stamm-sdk"
copyright = "2026, Santiago"
author = "Santiago"

try:
    from importlib.metadata import version as _pkg_version

    release = _pkg_version("stamm-sdk")
except Exception:
    release = "0.0.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",       # tolerates Google/NumPy-style sections in docstrings that have them
    "sphinx.ext.viewcode",       # adds "[source]" links next to each documented object
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",  # renders `: int | None` etc. from real type hints instead of requiring :type:/:rtype:
    "myst_parser",               # .md support, so SDK_GUIDE.md/MODEL_CATALOG.md need no rewriting
    "sphinx_design",             # {tab-set}/{tab-item} -- Manual/Automatic tabs in TRAINING_EXAMPLES.md
]

# Both suffixes so index.md (MyST) and any future .rst file both work.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = ["fieldlist", "deflist"]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"

napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "stamm-sdk"
