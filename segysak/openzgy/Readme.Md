### SEGYSAK and  OpenZGY Support

Open ZGY support is experimental and requires the installation of the Open-Zgy
Python library from the Open Group. Currently this is not available via pip so
manual installation is required.

Please follow the [instructions](https://community.opengroup.org/osdu/platform/domain-data-mgmt-services/seismic/open-zgy/-/tree/master/python)
for installation.

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