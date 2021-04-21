import pytest

import sys

import numpy as np
from scipy import interpolate
import xarray as xr

from hypothesis import given, assume, settings
from hypothesis.strategies import integers, text, floats, tuples, sampled_from
from hypothesis.extra.numpy import arrays

from affine import Affine

if sys.version_info >= (3, 8):
    from subsurface import StructuredData

from segysak import create3d_dataset
from segysak._keyfield import DimensionKeyField

from segysak._accessor import coordinate_df


class TestSeisIO:
    def test_to_subsurface_3d(self, empty3d):
        if sys.version_info >= (3, 8):
            assert isinstance(empty3d.seisio.to_subsurface(), StructuredData)
        else:
            print("Python <=3.7 not support by subsurface")
            assert True

    def test_to_subsurface_3dgath(self, empty3d_gath):
        if sys.version_info >= (3, 8):
            with pytest.raises(NotImplementedError):
                empty3d_gath.seisio.to_subsurface()
        else:
            print("Python <=3.7 not support by subsurface")
            assert True

    def test_to_subsurface_2dgath(self, empty2d_gath):
        if sys.version_info >= (3, 8):
            with pytest.raises(NotImplementedError):
                empty2d_gath.seisio.to_subsurface()
        else:
            print("Python <=3.7 not support by subsurface")
            assert True


@pytest.mark.parametrize(
    "coord,extras,linear_fillna,expected_shape",
    [
        (True, None, True, (414, 4)),
        (False, None, True, (414, 4)),
        (False, None, False, (414, 4)),
        (
            True,
            [
                "extra",
            ],
            True,
            (414, 5),
        ),
    ],
)
def test_coordinate_df(f3_dataset, coord, extras, linear_fillna, expected_shape):
    print(f3_dataset)
    df = coordinate_df(
        f3_dataset, coord=coord, extras=extras, linear_fillna=linear_fillna
    )
    assert df.shape == expected_shape


def test_humanbytes(f3_dataset):
    assert isinstance(f3_dataset.seis.humanbytes, str)


def test_get_measurement_system(f3_dataset):
    assert f3_dataset.seis.get_measurement_system() == "m"
    local = f3_dataset.copy(deep=True)
    local.attrs = {}
    assert local.seis.get_measurement_system() is None


def test_surface_from_points_nparray(f3_dataset, f3_horizon_exact):
    interpolated = f3_dataset.seis.surface_from_points(
        f3_horizon_exact[["cdp_x", "cdp_y"]].values,
        f3_horizon_exact["horizon"].values,
    )
    assert interpolated.data.mean() == 50.0


def test_surface_from_points_df(f3_dataset, f3_horizon_shift):
    interpolated = f3_dataset.seis.surface_from_points(
        f3_horizon_shift, "horizon", right=("cdp_x", "cdp_y")
    )
    assert interpolated.horizon.mean() == 50.0


def test_subsample_dims(f3_dataset):
    ss = f3_dataset.seis.subsample_dims(twt=2, xline=1)
    assert ss["twt"].size == (f3_dataset.twt.values.size * 2 ** 2 - 3)
    assert ss["xline"].size == (f3_dataset.xline.values.size * 2 - 1)


def test_fill_cdpna(f3_dataset):
    ds = f3_dataset.copy(deep=True)
    ds.cdp_x[:, 5:10] = np.nan
    ds.cdp_y[:, 5:10] = np.nan
    ds.seis.fill_cdpna()
    assert np.allclose(f3_dataset.cdp_x.values, ds.cdp_x.values, atol=0.01)
    assert np.allclose(f3_dataset.cdp_y.values, ds.cdp_y.values, atol=0.01)


def test_get_affine_transform(geometry_dataset):
    at = geometry_dataset.seis.get_affine_transform()
    df = geometry_dataset.drop_vars("data").to_dataframe().reset_index()
    assert np.allclose(
        at.transform(df[["iline", "xline"]].values),
        df[["cdp_x", "cdp_y"]].values,
        atol=1,
    )


def test_calc_corner_points(f3_dataset):
    f3_dataset.seis.calc_corner_points()
    print(f3_dataset.attrs["corner_points"])
    print(f3_dataset.attrs["corner_points_xy"])

    assert f3_dataset.attrs["corner_points"] == (
        (111, 875),
        (111, 892),
        (133, 892),
        (133, 875),
    )
    assert np.allclose(
        np.array(f3_dataset.attrs["corner_points_xy"]),
        np.array(
            (
                (620197.2, 6074233.0),
                (620622.1, 6074245.0),
                (620606.7, 6074794.5),
                (620181.94, 6074782.5),
            )
        ),
    )


def test_calc_corner_points2d(volve_2d_dataset):
    volve_2d_dataset.seis.calc_corner_points()
    assert volve_2d_dataset.attrs["corner_points"] == (1, 202)
    assert np.allclose(
        np.array(volve_2d_dataset.attrs["corner_points_xy"]),
        np.array(
            (
                (436482.16, 6477774.5),
                (436482.16, 6477774.5),
                (434044.3, 6477777.0),
                (434044.3, 6477777.0),
            )
        ),
    )


def test_get_dead_trace_map(f3_withdead_dataset):
    dead = f3_withdead_dataset.seis.get_dead_trace_map(zeros_as_nan=True)
    assert isinstance(dead, xr.DataArray)
    assert dead.sum().values == 175

    dead = f3_withdead_dataset.seis.get_dead_trace_map(
        scan=[30, 31, 32], zeros_as_nan=True
    )
    assert isinstance(dead, xr.DataArray)
    assert dead.sum().values == 175


class TestIsFunctions:
    def test_is_2d(self, empty2d):
        assert empty2d.seis.is_2d()
        assert not empty2d.seis.is_3d()
        assert not empty2d.seis.is_3dgath()
        assert not empty2d.seis.is_2dgath()

    def test_is_3d(self, empty3d):
        assert empty3d.seis.is_3d()
        assert not empty3d.seis.is_2d()
        assert not empty3d.seis.is_3dgath()
        assert not empty3d.seis.is_2dgath()

    def test_is_2d_gath(self, empty2d_gath):
        assert empty2d_gath.seis.is_2dgath()
        assert not empty2d_gath.seis.is_3d()
        assert not empty2d_gath.seis.is_3dgath()
        assert not empty2d_gath.seis.is_2d()

    def test_is_3d_gath(self, empty3d_gath):
        assert empty3d_gath.seis.is_3dgath()
        assert not empty3d_gath.seis.is_3d()
        assert not empty3d_gath.seis.is_2d()
        assert not empty3d_gath.seis.is_2dgath()

    def test_is_twt(self, empty3d_twt):
        assert empty3d_twt.seis.is_twt()
        assert not empty3d_twt.seis.is_depth()

    def test_is_depth(self, empty3d_depth):
        assert empty3d_depth.seis.is_depth()
        assert not empty3d_depth.seis.is_twt()

    def test_is_empty2d(self, empty2d):
        assert empty2d.seis.is_empty()

    def test_is_empty3d(self, empty3d):
        assert empty3d.seis.is_empty()

    def test_is_empty3d_gath(self, empty3d_gath):
        assert empty3d_gath.seis.is_empty()

    def test_is_empty2d_gath(self, empty2d_gath):
        assert empty2d_gath.seis.is_empty()

    def test_is_empty_zeros3d(self, zeros3d):
        assert not zeros3d.seis.is_empty()

    def test_zeros_like(self, zeros3d):
        assert "data" in zeros3d.variables
        assert np.all(zeros3d["data"] == 0.0)
        assert zeros3d["data"].sum() == 0.0


class TestCreate_xysel:
    """
    Test using xysel accessor
    """

    @given(
        tuples(integers(3, 50), integers(3, 50), integers(3, 10)),
        tuples(floats(-1000, 1000), floats(-1000, 1000)),
        floats(0.1, 10000),
        tuples(floats(0, 45), floats(0, 45)),
        floats(-180, 180),
        integers(1, 20),
    )
    @settings(deadline=None, max_examples=25)
    def test_full_stack_dataset_xysel(self, d, translate, scale, shear, rotate, samp):
        dataset = create3d_dataset(dims=d)
        xlines_, ilines_ = np.meshgrid(dataset.xline, dataset.iline)
        ix_pairs = np.dstack([ilines_, xlines_])
        tr = Affine.translation(*translate)
        sc = Affine.scale(scale)
        sh = Affine.shear(*shear)
        rt = Affine.rotation(rotate)
        trsfm = lambda x: tr * sc * sh * rt * x
        ix_pairs = np.apply_along_axis(trsfm, 1, ix_pairs.reshape(-1, 2)).reshape(
            ix_pairs.shape
        )
        dataset["cdp_x"] = (DimensionKeyField.cdp_3d, ix_pairs[:, :, 0])
        dataset["cdp_y"] = (DimensionKeyField.cdp_3d, ix_pairs[:, :, 1])

        dataset["data"] = (DimensionKeyField.threed_twt, np.random.rand(*d))

        test_points = np.dstack(
            [
                np.random.random(samp) * (d[0] - 0.1) + 0.1,
                np.random.random(samp) * (d[1] - 0.1) + 0.1,
            ]
        )[0]
        # make sure at least one point is in the box
        test_points[0, :] = d[0] / 2, d[1] / 2

        xys = np.apply_along_axis(trsfm, 1, test_points)

        res = dataset.seis.xysel(xys[:, 0], xys[:, 1], method="linear")
        assert isinstance(res, xr.Dataset)
        res = dataset.seis.xysel(xys[:, 0], xys[:, 1], method="nearest")
        assert isinstance(res, xr.Dataset)

    @given(
        integers(15, 60),
        floats(0, 15),
        floats(1, 15),
        tuples(floats(-1000, 1000), floats(-1000, 1000)),
        floats(0.1, 10000),
        tuples(floats(0, 45), floats(0, 45)),
        floats(-180, 180),
        integers(5, 10),
    )
    @settings(deadline=None, max_examples=25, print_blob=True)
    def test_angle_stack_dataset_xysel(
        self, o, f, s, translate, scale, shear, rotate, samp
    ):
        dims = (50, 30, 5, o)
        dataset = create3d_dataset(dims, first_offset=f, offset_step=s)
        xlines_, ilines_ = np.meshgrid(dataset.xline, dataset.iline)
        ix_pairs = np.dstack([ilines_, xlines_])
        tr = Affine.translation(*translate)
        sc = Affine.scale(scale)
        sh = Affine.shear(*shear)
        rt = Affine.rotation(rotate)
        trsfm = lambda x: tr * sc * sh * rt * x
        ix_pairs = np.apply_along_axis(trsfm, 1, ix_pairs.reshape(-1, 2)).reshape(
            ix_pairs.shape
        )
        dataset["cdp_x"] = (DimensionKeyField.cdp_3d, ix_pairs[:, :, 0])
        dataset["cdp_y"] = (DimensionKeyField.cdp_3d, ix_pairs[:, :, 1])

        if dataset.seis.is_depth():
            dataset["data"] = (DimensionKeyField.threed_ps_depth, np.random.rand(*dims))
        else:
            dataset["data"] = (DimensionKeyField.threed_ps_twt, np.random.rand(*dims))

        test_points = np.dstack(
            [
                np.random.random(samp) * (dims[0] - 0.1) + 0.1,
                np.random.random(samp) * (dims[1] - 0.1) + 0.1,
            ]
        )[0]
        # make sure at least one point is in the box
        test_points[0, :] = dims[0] / 2, dims[1] / 2

        xys = np.apply_along_axis(trsfm, 1, test_points)

        res = dataset.seis.xysel(xys[:, 0], xys[:, 1], method="linear")
        assert isinstance(res, xr.Dataset)
        res = dataset.seis.xysel(xys[:, 0], xys[:, 1], method="nearest")
        assert isinstance(res, xr.Dataset)
