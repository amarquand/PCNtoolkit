# Configuration file for the Sphinx documentation builder
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# Project information
project = "PCNToolkit"
copyright = "2025, Andre Marquand"
author = "Andre Marquand"
release = "1.0.0"

# Extensions
extensions = [
    "autoapi.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
]

# AutoAPI settings
autoapi_dirs = ["../pcntoolkit"]  # Directory to scan
autoapi_options = [
    "members",  # Include class/module members
    "undoc-members",  # Include items without docstrings
    "show-inheritance",  # Show base classes
    "show-module-summary",  # Include module docstring summaries
    "special-members",  # Include special methods (__init__, etc.)
]
autoapi_python_class_content = "both"  # Include both class and __init__ docstrings
autoapi_member_order = "groupwise"  # Group members by type (methods, attributes, etc.)
autoapi_add_toctree_entry = True  # Add to table of contents
autoapi_template_dir = "_templates/autoapi"  # Custom templates location
autoapi_keep_files = True  # Keep generated RST files for debugging

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# Theme settings
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": True,
    "sticky_navigation": True,
    "titles_only": False,
    "display_version": True,
    "prev_next_buttons_location": "both",
}

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# General settings
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
add_module_names = False
nitpicky = True

# AutoDoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "imported-members": True,
}
