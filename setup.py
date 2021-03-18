#!/usr/bin/python
# -*- coding: utf8 -*-+
import re
from setuptools import setup
from setuptools import find_packages

# from segysak import __version__

# Get README and remove badges.
readme = open("README.rst").read()

install_requires = [
    "numpy",
    "pandas",
    "segyio",
    "xarray",
    "dask",
    "distributed",
    "tqdm",
    "scipy",
    "click",
    "h5netcdf",
    "setuptools_scm",
    "attrdict",
]

notebook_deps = [
    "ipython",
    "ipykernel>=4.8.0",  # because https://github.com/ipython/ipykernel/issues/274 and https://github.com/ipython/ipykernel/issues/263
    "jupyter_client>=5.2.2",  # because https://github.com/jupyter/jupyter_client/pull/314
    "ipywidgets",
    "matplotlib",
    "pyvista",
]

testing_deps = ["flake8", "pytest", "hypothesis", "affine", "pytest_cases"]

extras_require = {
    "notebook": notebook_deps,
    "testing": testing_deps,
    "docs": [
        "sphinx",
        "sphinx_rtd_theme",
        "sphinxcontrib-programoutput",
        "nbsphinx >= 0.7",
        "pandoc",
        "nbconvert!=5.4",
        "nbformat",
        "sphinx-copybutton",
        "rtds-action",
        "jupytext",
    ]
    + notebook_deps,
}


setup(
    name="segysak",
    # version="0.2.3a",
    description="SEG-Y Seismic Data Inspection and Manipulation Tools using Xarray",
    long_description=readme,
    long_description_content_type="text/x-rst",
    author="SEGY-SAK Developers",
    author_email="segysak@gamil.com",
    url="https://github.com/trhallam/segysak",
    license="GPL3",
    install_requires=install_requires,
    tests_require=testing_deps,
    extras_require=extras_require,
    packages=find_packages(),
    # add command line scripts here
    entry_points={"console_scripts": ["segysak=segysak._cli:cli"]},
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
    ],
    setup_requires=[
        "pytest-runner",
        "setuptools_scm",
    ],
    use_scm_version={
        "root": ".",
        "relative_to": __file__,
        "local_scheme": "no-local-version",
        "write_to": "segysak/version.py",
    },
    python_requires=">=3.6",
)
