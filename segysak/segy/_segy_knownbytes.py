"""Known byte positions for different SEGY file standards.
"""

from dataclasses import dataclass, field
from .._core import MetaDataClass
from .._keyfield import CoordKeyField


def standard_3d():
    return {
        CoordKeyField.iline: 189,
        CoordKeyField.xline: 193,
        CoordKeyField.cdp_x: 181,
        CoordKeyField.cdp_y: 185,
    }


def standard_3d_gath():
    return {
        CoordKeyField.iline: 189,
        CoordKeyField.xline: 193,
        CoordKeyField.cdp_x: 181,
        CoordKeyField.cdp_y: 185,
        CoordKeyField.offset: 37,
    }


def standard_2d():
    return {
        CoordKeyField.cdp: 21,
        CoordKeyField.cdp_x: 181,
        CoordKeyField.cdp_y: 185,
    }


def standard_2d_gath():
    return {
        CoordKeyField.cdp: 21,
        CoordKeyField.cdp_x: 181,
        CoordKeyField.cdp_y: 185,
        CoordKeyField.offset: 37,
    }


def petrel_3d():
    return {
        CoordKeyField.iline: 5,
        CoordKeyField.xline: 21,
        CoordKeyField.cdp_x: 73,
        CoordKeyField.cdp_y: 77,
    }


def petrel_2d():
    return {
        CoordKeyField.cdp: 21,
        CoordKeyField.cdp_x: 73,
        CoordKeyField.cdp_y: 77,
    }


@dataclass(frozen=True)
class _KnownBytes(MetaDataClass):
    standard_3d: dict = field(default_factory=standard_3d)
    standard_3d_gath: dict = field(default_factory=standard_3d_gath)
    standard_2d: dict = field(default_factory=standard_2d)
    standard_2d_gath: dict = field(default_factory=standard_2d_gath)
    petrel_3d: dict = field(default_factory=petrel_3d)
    petrel_2d: dict = field(default_factory=petrel_2d)


KNOWN_BYTES = _KnownBytes()
OUTPUT_BYTES = _KnownBytes()
