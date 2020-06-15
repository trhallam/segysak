
from .version import version as __version__
from ._accessor import SeisIO, SeisGeom, open_seisnc

from ._seismic_dataset import (
    create_seismic_dataset,
    create3d_dataset,
    create2d_dataset,
)
