import os
import sys


# from hivegraph import __version__


sys.path.insert(0, os.path.abspath("../"))

project = "hivegraph"
copyright = "2023, Saurav Maheshkar"
author = "Saurav Maheshkar"
release = "0.0.2"

extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pytorch": ("https://pytorch.org/docs/stable/", None),
    "ml_collections": ("https://ml-collections.readthedocs.io/en/latest/", None),
    "pytorch_geometric": ("https://pytorch-geometric.readthedocs.io/en/latest/", None),
}

myst_update_mathjax = False
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "inherited-members": False,
}
autosummary_generate = True

html_theme = "sphinx_rtd_theme"
html_favicon = "media/favicon.ico"
