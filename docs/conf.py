import shutil
from pathlib import Path

here = Path(__file__).parent
shutil.copytree(
    here.parent / "notebooks", here / "notebooks", dirs_exist_ok=True
)

project = "biosspheres"
author = "Isabel A. Martínez-Ávila, Carlos Jerez-Hanckes, Paul Escapil-Inchauspé, Tobias Gebäck"
copyright = "2023-2026, the biosspheres developers"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "nbsphinx",
]

autosummary_generate = True
autodoc_default_options = {"members": True}
napoleon_google_docstring = False
napoleon_numpy_docstring = True

nbsphinx_execute = "always"
nbsphinx_timeout = 1200
nbsphinx_allow_errors = False

exclude_patterns = [
    "_build",
    "notebooks/onesphere_ode_kavian.ipynb",
    "**.ipynb_checkpoints",
]

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/iamartineza/biosspheres",
}
