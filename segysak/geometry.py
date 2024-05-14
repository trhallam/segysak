"""Functions and utilities related to SEG-Y/Seismic Geometry"""

from typing import Tuple, Callable
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning


def plane(xy: Tuple[float, float], a: float, b: float, c: float) -> float:
    """Function of a plane for linear fitting using curve_fit"""
    x, y = xy
    return a * x + b * y + c


def fit_plane(x: np.array, y: np.array, z: np.array, p0=(0, 0, 0)) -> Callable:
    """Calculate the plane function coefficients for input data and return a partial plane function."""

    fit, _ = curve_fit(plane, (x, y), z, p0=p0)
    func = lambda xy: plane(xy, *fit)
    return func
