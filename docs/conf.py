# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import time

from pkg_resources import get_distribution

release = get_distribution("myproject").version
# for example take major/minor
version = ".".join(release.split(".")[:2])


# -- Project information -----------------------------------------------------

description = """A swiss army knife for seismic data that provides tools for segy
    segy manipulation as well as an interface to using segy with xarray.
"""

project = "segysak"
copyright = u"2020-{}, The segysak Developers.".format(time.strftime("%Y"))
author = "segysak developers"

# The full version, including alpha/beta/rc tags
version = __version__
release = __version__
today_fmt = "%d %B %Y"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_rtd_theme",
]

autosummary_generate = True
add_module_names = True

source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Pygments syntax highlighting
pygments_style = "friendly"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# rtd theme options
html_theme_options = {
    #     'canonical_url': '',
    #     'analytics_id': 'UA-XXXXXXX-1',  #  Provided by Google in your dashboard
    #     'logo_only': False,
    "display_version": True,
    #     'prev_next_buttons_location': 'bottom',
    "style_external_links": True,
    #     'vcs_pageview_mode': '',
    #     'style_nav_header_background': 'white',
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
    "prev_next_buttons_location": "both",
}

html_static_path = ["_static"]
html_logo = "figures/logo.png"

github_url = "https://github.com/trhallam/segysak/"

htmlhelp_basename = "segysakdoc"


def setup(app):
    app.add_stylesheet("style.css")


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
