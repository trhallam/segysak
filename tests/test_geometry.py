import pytest
import numpy as np

from segysak.geometry import (
    plane,
    fit_plane,
    lsq_affine_transform,
    orthogonal_point_affine_transform,
    get_uniform_spacing,
)


def test_lsq_affine_transform(geometry_dataset):

    df = geometry_dataset.segysak.coordinate_df()

    X = df[["iline", "xline"]].values
    Y = df[["cdp_x", "cdp_y"]].values

    f_transform, f_error = lsq_affine_transform(
        X,
        Y,
        estimate_error=True,
    )

    assert f_error[0] <= 1e-5
    assert f_error[1] <= 1
    assert np.allclose(Y, f_transform.transform(X))

    r_transform, r_error = lsq_affine_transform(
        Y,
        X,
        estimate_error=True,
    )

    assert r_error[0] <= 1e-5
    assert r_error[1] <= 1
    assert np.allclose(X, r_transform.transform(Y))

    assert np.allclose(f_transform.inverted().get_matrix(), r_transform.get_matrix())


def test_op_affine_transform(geometry_dataset):

    selection = ((0, 0), (0, -1), (-1, 0))

    X = []
    Y = []

    for sel in selection:
        sub_ds = geometry_dataset.isel({"iline": sel[0], "xline": sel[1]})
        X.append((sub_ds.iline.item(), sub_ds.xline.item()))
        Y.append((sub_ds.cdp_x.item(), sub_ds.cdp_y.item()))

    f_transform, f_error = orthogonal_point_affine_transform(
        X,
        Y,
        estimate_error=True,
    )

    assert f_error[0] <= 1e-5
    assert f_error[1] <= 1
    assert np.allclose(Y, f_transform.transform(X))

    r_transform, r_error = orthogonal_point_affine_transform(
        Y,
        X,
        estimate_error=True,
    )

    assert r_error[0] <= 1e-5
    assert r_error[1] <= 1
    assert np.allclose(X, r_transform.transform(Y))

    assert np.allclose(f_transform.inverted().get_matrix(), r_transform.get_matrix())


@pytest.mark.parametrize(
    "points,bin_spacing_hint,expectedn",
    (
        ([[0, 0], [0, 100]], 10, 11),
        ([[0, 0], [10, 100]], 10, 11),
        ([[0, 0], [0, 100], [100, 100]], 20, 11),
    ),
)
def test_get_uniform_spacing(points, bin_spacing_hint, expectedn):
    uniform, _ = get_uniform_spacing(points, bin_spacing_hint=bin_spacing_hint)
    assert uniform.shape[0] == expectedn
