from ._version import version as __version__

from ._seismic_dataset import (
    create_seismic_dataset,
    create3d_dataset,
    create2d_dataset,
)

from ._keyfield import (
    HorDimKeyField,
    VerDimKeyField,
    DimKeyField,
    CoordKeyField,
    VariableKeyField,
    AttrKeyField,
    VerticalKeyDim,
)

# to be depreciated
from ._accessor import open_seisnc
