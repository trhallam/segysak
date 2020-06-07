#!/usr/bin/python
# -*- coding: utf8 -*-+

from setuptools import setup
from setuptools import find_packages

long_description = """
The SEG-Y File Swiss Army Knife (SEGY-SAK) is a package developed for
manipulating and transform SEG-Y Seismic Data using Xarray.
"""

setup(
    name="segysak",
    version="0.2",
    description="SEG-Y Seismic Data Inspection and Manipulation Tools",
    long_description=long_description,
    author="Tony Hallam",
    author_email="arh5@hw.ac.uk",
    url="https://github.com/trhallam/segysak",
    license="GPL3",
    install_requires=[
        "numpy",
        "pandas",
        "segyio",
        "xarray",
        "dask",
        "tqdm",
        "scipy",
        "h5netcdf",
    ],
    extras_require={"docs": ["sphinx", "sphinx_rtd_theme"]},
    packages=find_packages(),
    # add command line scripts here
    entry_points={"console_scripts": ["segysak=segysak._cli:main"]},
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.6",
)
