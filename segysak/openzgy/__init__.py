import warnings

warnings.warn(
    "ZGY support is experimental and depedenent upon the upstream implementation by OSDU",
)

from ._openzgy import zgy_loader, zgy_writer
