# Installation

*SEGY-SAK* can be installed by using pip or python setuptools, we also provide an ``environment.yml`` for use
with conda and is available through [Github](https://github.com/trhallam/segysak).


## Python Package Index via ``pip``

From the command line run the ``pip`` package manager

```shell
pip install segysak
```   

## Install from source

Clone the SEGY-SAK Github repository and in the top level directory run setuptools via

```shell
python -m pip install .
```
   
And to run the tests install the test dependencies and run `pytest`

```shell
python -m pip install .[test]
pytest
```
