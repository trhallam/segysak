# Installation

*SEGY-SAK* can be installed by using `pip` from PyPi and from source.

## Python Package Index via ``pip``

From the command line run the ``pip`` package manager

```shell
python -m pip install segysak
```

## Install from source

Clone the SEGY-SAK Github repository and in the top level directory run

```shell
python -m pip install .
```

To run the tests install the test dependencies and run `pytest`

```shell
python -m pip install .[test]
pytest
```
