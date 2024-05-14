"""Functions and utilities related to SEG-Y/Seismic Geometry"""

from typing import Tuple, Callable
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
from matplotlib.transforms import Affine2D


def plane(xy: Tuple[float, float], a: float, b: float, c: float) -> float:
    """Function of a plane for linear fitting using curve_fit"""
    x, y = xy
    return a * x + b * y + c


def fit_plane(x: np.array, y: np.array, z: np.array, p0=(1.0, 1.0, 1.0)) -> Callable:
    """Calculate the plane function coefficients for input data and return a partial plane function."""

    fit, _ = curve_fit(plane, (x, y), z, p0=p0)
    func = lambda xy: plane(xy, *fit)
    return func


def lsq_affine_transform(
    x: np.array,
    y: np.array,
    zero_small_values: bool = True,
    estimate_error: bool = False,
) -> Tuple[Affine2D, Tuple[float, float]]:
    """Calculate the Affine transform from the least squared solver.

    Note, this is not an exact solution as there can be numeric error, but it is more robust than exact methods.

    Args:
        x: The input coordinates as pairs [M, 2]. E.g. [[iline, xline], ...]
        y: The output coordinates as pairs [M, 2]. E.g. [[cdp_x, cdp_y], ...]
        zero_small_values: Set small values in the LSQ solution to zero.
        estimate_error: Optionally use the transform to return a tuple of (mean, max) error for estimated transform.

    Returns:
        transform, error: Returns the matplotlib Affine2D object and optionally an error estimate tuple.
    """
    # https://stackoverflow.com/questions/20546182/how-to-perform-coordinates-affine-transformation-using-python-part-2
    n_points = x.shape[0]

    # Pad the data with ones, so that our transformation can do translations too

    pad = lambda x: np.hstack([x, np.ones((n_points, 1))])
    X = pad(x)
    Y = pad(y)

    # Solve the least squares problem X * A = Y
    # to find our transformation matrix A
    A, res, rank, s = np.linalg.lstsq(X, Y, rcond=None)

    # set really small values to zero
    if zero_small_values:
        A[np.abs(A) < 1e-10] = 0

    transform = Affine2D.from_values(*A[:, :2].ravel())

    if estimate_error:
        error = transform.transform(x) - y
        return transform, (error.mean(), error.max())
    else:
        return transform, None


def orthogonal_point_affine_transform(
    x: Tuple, y: np.array, estimate_error: bool = False
) -> Tuple[Affine2D, Tuple[float, float]]:
    """Calculate an affine transform using orthogonal points. This assumes an orthogonal survey. If you have a
    skewed goemetry, use `least_squares_affine_transform`.

        ^ (2, 2)
        |
        |
        |
        |
        |_
        |_|____________>
       (0, 0)         (1, 1)

    Args:
        x: The input coordinates as pairs [3, 2]. E.g. [[iline, xline], ...]
        y: The output coordinates as pairs [3, 2]. E.g. [[cdp_x, cdp_y], ...]
        estimate_error: Optionally use the transform to return a tuple of (mean, max) error for estimated transform.

    Returns:
        transform, error: Returns the matplotlib Affine2D object and optionally an error estimate tuple.
    """
    # direct solve for affine transform via equation substitution
    # https://cdn.sstatic.net/Sites/math/img/site-background-image.png?v=09a720444763
    # ints for iline xline will often overflow
    (x0, y0), (x1, y1), (x2, y2) = x
    (x0p, y0p), (x1p, y1p), (x2p, y2p) = y

    a = (x1p * y0 - x2p * y0 - x0p * y1 + x2p * y1 + x0p * y2 - x1p * y2) / (
        x1 * y0 - x2 * y0 - x0 * y1 + x2 * y1 + x0 * y2 - x1 * y2
    )
    c = (x1p * x0 - x2p * x0 - x0p * x1 + x2p * x1 + x0p * x2 - x1p * x2) / (
        -x1 * y0 + x2 * y0 + x0 * y1 - x2 * y1 - x0 * y2 + x1 * y2
    )
    b = (y1p * y0 - y2p * y0 - y0p * y1 + y2p * y1 + y0p * y2 - y1p * y2) / (
        x1 * y0 - x2 * y0 - x0 * y1 + x2 * y1 + x0 * y2 - x1 * y2
    )
    d = (y1p * x0 - y2p * x0 - y0p * x1 + y2p * x1 + y0p * x2 - y1p * x2) / (
        -x1 * y0 + x2 * y0 + x0 * y1 - x2 * y1 - x0 * y2 + x1 * y2
    )
    e = (
        x2p * x1 * y0
        - x1p * x2 * y0
        - x2p * x0 * y1
        + x0p * x2 * y1
        + x1p * x0 * y2
        - x0p * x1 * y2
    ) / (x1 * y0 - x2 * y0 - x0 * y1 + x2 * y1 + x0 * y2 - x1 * y2)
    f = (
        y2p * x1 * y0
        - y1p * x2 * y0
        - y2p * x0 * y1
        + y0p * x2 * y1
        + y1p * x0 * y2
        - y0p * x1 * y2
    ) / (x1 * y0 - x2 * y0 - x0 * y1 + x2 * y1 + x0 * y2 - x1 * y2)
    values = (v if ~np.isnan(v) else 0.0 for v in (a, b, c, d, e, f))
    transform = Affine2D.from_values(*values)

    if estimate_error:
        error = transform.transform(x) - y
        return transform, (error.mean(), error.max())
    else:
        return transform, None
