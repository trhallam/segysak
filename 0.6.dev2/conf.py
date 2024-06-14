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
import os
import pathlib
import shutil
from sys import path

# import sys
# sys.path.insert(0, os.path.abspath('.'))

import time
import pathlib
from shutil import copy

from pkg_resources import get_distribution

release = get_distribution("segysak").version
# for example take major/minor
version = ".".join(release.split(".")[:2])

print(release, version)


# -- Copy Extra Files
docs = pathlib.Path(__file__).parent.absolute()
(docs / "_temp").mkdir(exist_ok=True)
shutil.copy(docs / "../contributing.rst", docs / "_temp/contributing.rst")
# sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------

description = """A swiss army knife for seismic data that provides tools for segy
    segy manipulation as well as an interface to using segy with xarray.
"""

project = "segysak"
copyright = "2020-{}, The segysak Developers.".format(time.strftime("%Y"))
author = "segysak developers"

# The full version, including alpha/beta/rc tags
version = version
release = release
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
    "sphinx.ext.intersphinx",
    "sphinxcontrib.programoutput",
    "sphinx.ext.mathjax",
    "nbsphinx",
    "sphinx_copybutton",
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
    "navigation_with_keys": True,
}

html_static_path = ["_static"]
html_logo = "figures/logo.png"

github_url = "https://github.com/trhallam/segysak/"

htmlhelp_basename = "segysakdoc"

nbsphinx_prolog = """
{% set docname = "./" + env.docname[8:] %}

.. note::

   Get this example as a Jupyter Notebook :download:`ipynb <{{ docname }}.ipynb>`
"""

# build examples - comment this out if you have pre-built examples and don't want
# to rebuild for all docs
build_examples = True

if build_examples:
    # need to copy notebooks into main tree
    print("Copy examples into docs/examples")
    top_level_examples = pathlib.Path(".").absolute().parent / "examples"
    print("Top_Level", top_level_examples)
    examples_dir = pathlib.Path("examples")
    print("Examples Dir", examples_dir.absolute())
    shutil.rmtree(examples_dir, ignore_errors=True)
    examples_dir.mkdir(exist_ok=True)

    # rtds option
    rtds_action_github_repo = "trhallam/segysak"
    rtds_action_artifact_prefix = "example-notebooks-"
    rtds_action_path = str(examples_dir)

    try:
        rtds_action_github_token = os.environ["GITHUB_TOKEN"]
        nbsphinx_execute = "never"
        extensions.append("rtds_action")
        print("RDTS_ACTION found using remote pre-built examples.")
    except KeyError:
        print("RTDS_ACTION token missing building notebooks locally")
        import jupytext

        for example in top_level_examples.glob("*.py"):
            output = examples_dir / example.with_suffix(".ipynb").name
            ntbk = jupytext.read(example, fmt="py")
            jupytext.write(ntbk, output)

        shutil.copytree(
            top_level_examples / "data",
            examples_dir / "data",
        )
else:
    nbsphinx_execute = "never"


def setup(app):
    app.add_css_file("style.css")


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

intersphinx_mapping = {
    "https://docs.python.org/3/": None,
    "https://numpy.org/doc/stable/": None,
    "https://docs.scipy.org/doc/scipy/reference/": None,
    "http://xarray.pydata.org/en/stable/": None,
    "https://pandas.pydata.org/docs/": None,
}
