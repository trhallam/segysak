### SEGY-SAK and  OpenZGY Support for distribution

SEGY-SAK distributes the OpenZGY as part of it's PyPI packages. Installing SEGY-SAK
with `pip` will activate the `segysak.openzgy` module for reading and writing ZGY
data. This does not make the Open-ZGY package available within the installed Python
environment.

If you require ZFP compression support. The `zfpy` package should be installed
manually by the user.

### SEGY-SAK and  OpenZGY Support for development

Open ZGY support is experimental and requires the installation of the Open-Zgy
Python library from the Open Group. Currently this is not available via pip so
manual installation is required.

Please follow the [instructions](https://community.opengroup.org/osdu/platform/domain-data-mgmt-services/seismic/open-zgy/-/tree/master/python)
for installation.

Alternatively, if the module is not installed manually we make a version available
through `segysak` as a submodule.

To check out `segysak` and `openzgy` in a ready to go format use the following commands when cloning the repository.
```bash
git clone --recurse-submodules <repo.git>
# checkout to the desired branch
git checkout zgy_integration
# update the submodules
git submodule init
git submodule update
```

## Usage

The openzgy library can be accessed by import from the openzgy module.

```
from segysak.openzgy import zgy_loader

seisnc = zgy_loader('azgyfile.zgy')
```

## Limitations

Currently selection and indexing methods are limited with OpenZGY and Xarray.
The easiest approach is to load the full volume into memory and then output to
a NetCDF dataset for better memory management.
