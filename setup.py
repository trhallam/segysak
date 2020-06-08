#!/usr/bin/python
# -*- coding: utf8 -*-+
import re
from setuptools import setup
from setuptools import find_packages

# from segysak import __version__

# Get README and remove badges.
readme = open("README.rst").read()

setup(
    name="segysak",
    # version=__version__,
    description="SEG-Y Seismic Data Inspection and Manipulation Tools using Xarray",
    long_description=readme,
    author="SEGY-SAK Developers",
    author_email="segysak@gamil.com",
    url="https://github.com/trhallam/segysak",
    license="GPL3",
    install_requires=[
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
    ],
    extras_require={
        "docs": ["sphinx", "sphinx_rtd_theme"],
        "test": ["pytest", "hypothesis"],
    },
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
    setup_requires=["setuptools_scm"],
    use_scm_version={
        "root": ".",
        "relative_to": __file__,
        # "local_scheme": "node-and-timestamp",
        "write_to": "segysak/version.py",
    },
    python_requires=">=3.6",
)
