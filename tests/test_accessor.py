import warnings

from multiprocessing.sharedctypes import Value
import pytest

import sys

import numpy as np
from scipy import interpolate
import xarray as xr
from matplotlib.transforms import Affine2D

from hypothesis import given, assume, settings
from hypothesis.strategies import (
    integers,
    builds,
    floats,
    tuples,
    sampled_from,
    composite,
    randoms,
)
from hypothesis.extra.numpy import arrays

from affine import Affine

# if sys.version_info >= (3, 8):
#     from subsurface import StructuredData

from segysak import create3d_dataset
from segysak._keyfield import DimensionKeyField

from segysak._accessor import coordinate_df


@pytest.fixture(
    params=[
        (xr.Dataset(),),
        (xr.DataArray(),),
    ],
    ids=["dset", "darray"],
)
def xrd(request):
    (ds,) = request.param
    return ds


@pytest.fixture(
    params=[
        (
            "test-data-segyio/f3.sgy",
            {"iline": 189, "xline": 193},
            {"cdp_x": 181, "cdp_y": 185},
        ),
        (
            "test-data-segysak/f3-withdead.sgy",
            {"iline": 189, "xline": 193},
            {"cdp_x": 181, "cdp_y": 185},
        ),
    ],
    ids=["f3", "f3dead"],
)
def segy_datasets(request, tests_path):
    segy_file, dim_bytes, extra_bytes = request.param
    ds = xr.open_dataset(
        tests_path / segy_file,
        dim_byte_fields=dim_bytes,
        extra_byte_fields=extra_bytes,
    )
    return ds


@pytest.fixture()
def segy_datasets_nocoords(segy_datasets):
    return segy_datasets.drop_vars(["cdp_x", "cdp_y"])


@pytest.fixture()
def segy_datasets_altcoords(segy_datasets):
    new_labels = {"cdp_x": "cdpx", "cdp_y": "cdpy"}
    ds = segy_datasets.rename_vars(new_labels)
    ds.segysak.set_coords(new_labels)
    return ds


def test_get_blank_segysak_attrs(xrd):
    attrs = xrd.segysak.attrs
    assert attrs == dict()


def test_segysak_store_attributes(xrd):
    attrs = dict(a=1, b="text", c=2.0, d=[1, 2, 3], e=True, f=False)
    xrd.segysak.store_attributes(**attrs)

    assert attrs == xrd.segysak.attrs


def test_segysak_set_attributes(xrd):
    attrs = dict(a=1, b="text", c=2.0, d=[1, 2, 3], e=True, f=False)
    for key, value in attrs.items():
        xrd.segysak[key] = value

    assert attrs == xrd.segysak.attrs


def test_segysak_get_attributes(xrd):
    attrs = dict(a=1, b="text", c=2.0, d=[1, 2, 3], e=True, f=False)
    for key, value in attrs.items():
        xrd.segysak[key] = value

    for key, value in attrs.items():
        assert value == xrd.segysak[key]


def test_segysak_humanbytes(xrd):
    assert isinstance(xrd.segysak.humanbytes, str)


def test_segysak_infer_dimensions(segy_datasets):
    dims = segy_datasets.segysak.infer_dimensions()
    dims_da = segy_datasets.data.segysak.infer_dimensions()

    assert dims == {
        "iline": "iline",
        "xline": "xline",
    }
    assert dims_da == {
        "iline": "iline",
        "xline": "xline",
    }


def test_segysak_get_dimensions(segy_datasets):
    dims = segy_datasets.segysak.get_dimensions()
    dims_da = segy_datasets.data.segysak.get_dimensions()

    assert dims == {
        "iline": "iline",
        "xline": "xline",
    }
    assert dims_da == {
        "iline": "iline",
        "xline": "xline",
    }

    alt_dims = {"iline": "inlines", "xline": "crosslines"}
    segy_datasets = segy_datasets.rename_dims(alt_dims)
    segy_datasets.segysak.set_dimensions(alt_dims)
    segy_datasets.data.segysak.set_dimensions(alt_dims)

    dims = segy_datasets.segysak.get_dimensions()
    dims_da = segy_datasets.data.segysak.get_dimensions()
    assert dims == alt_dims
    assert dims_da == alt_dims


def test_scale_coords(segy_datasets):
    ds = segy_datasets
    ds.segysak.scale_coords()
    assert ds.segysak["coord_scaled"]


def test_scale_altcoords(segy_datasets_altcoords):
    ds = segy_datasets_altcoords
    ds.segysak.scale_coords()
    assert ds.segysak["coord_scaled"]


def test_scale_nocoords(segy_datasets_nocoords):
    ds = segy_datasets_nocoords
    with pytest.raises(KeyError):
        ds.segysak.scale_coords()


def test_calc_corner_points_3dps():
    dims = (5, 3, 15, 4)
    dataset, trsfm = create_xy_dataset(
        dims,
        (4044.32, 2321.32),
        1500.4,
        (0.45, 0.9),
        20,
        first_offset=1,
        offset_step=10,
    )
    cp = dataset.isel(offset=0).segysak.calc_corner_points()
    assert len(cp) == 5


def test_calc_corner_points_3d(segy_datasets):
    ds = segy_datasets
    cp = ds.segysak.calc_corner_points()
    assert len(cp) == 5


def test_calc_corner_points_2d(volve_2d_dataset):
    cp = volve_2d_dataset.segysak.calc_corner_points()
    assert len(cp) == 2
    assert np.allclose(
        np.array(cp),
        np.array(
            [
                (436482.16, 6477774.5),
                (434044.3, 6477777.0),
            ]
        ),
        rtol=0.1,
    )


def test_segysak_coordinate_df(f3_dataset):
    cdf = f3_dataset.segysak.coordinate_df(three_d_only=True)
    pass


def test_segysak_fill_cdpna(f3_dataset):
    ds = f3_dataset.copy(deep=True)
    ds.cdp_x[:, 5:10] = np.nan
    ds.cdp_y[:, 5:10] = np.nan
    ds.segysak.fill_cdpna()
    assert np.allclose(f3_dataset.cdp_x.values, ds.cdp_x.values, atol=0.01)
    assert np.allclose(f3_dataset.cdp_y.values, ds.cdp_y.values, atol=0.01)


def test_segysak_get_affine_t(f3_dataset):
    assert f3_dataset.segysak.attrs.get("affine") is None

    t = f3_dataset.segysak.get_affine_transform()

    # check reuse
    t2 = f3_dataset.segysak.get_affine_transform()

    assert np.allclose(t2.get_matrix(), t.get_matrix())

    sub_sel = f3_dataset.isel(iline=[2]).segysak.coordinate_df()
    iprime = t.transform(sub_sel[["iline", "xline"]].values)
    assert np.allclose(iprime, sub_sel[["cdp_x", "cdp_y"]].values)


@pytest.mark.parametrize(
    "linear_fillna,expected_shape",
    [
        (True, (414, 4)),
        (True, (414, 4)),
        (False, (414, 4)),
        (
            True,
            (414, 4),
        ),
    ],
)
def test_coordinate_df(f3_dataset, linear_fillna, expected_shape):
    print(f3_dataset)
    df = coordinate_df(f3_dataset, linear_fillna=linear_fillna)
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
    assert ss["twt"].size == (f3_dataset.twt.values.size * 2**2 - 3)
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
    print(at.transform(df[["iline", "xline"]]))
    print(df[["cdp_x", "cdp_y"]])
    assert np.allclose(
        at.transform(df[["iline", "xline"]].values),
        df[["cdp_x", "cdp_y"]].values,
        atol=1,
    )


@composite
def affine2ds(draw):
    def builder(t, sk, sc, r):
        tfm = Affine2D()
        tfm.translate(*t).rotate(r).skew(*sk).scale(*sc)
        return tfm

    fkws = dict(allow_nan=False, allow_infinity=False, min_value=-10e2, max_value=10e2)
    t = tuples(floats(**fkws), floats(**fkws))
    sk = tuples(floats(**fkws), floats(**fkws))
    sc = tuples(floats(**fkws), floats(**fkws))
    r = floats(**fkws)
    return draw(builds(builder, t, sk, sc, r))


@given(affine2ds())
@settings(max_examples=100)
def test_get_affine_transform2(tfm):
    # build a test data set
    a = create3d_dataset(
        (10, 10, 10),
    )

    lh_x = np.full((10, 10), np.nan)
    lh_y = np.full((10, 10), np.nan)
    for ci, cx in zip((0, 0, 9, 9), (0, 9, 0, 9)):
        lh_x[ci, cx], lh_y[ci, cx] = tfm.transform([ci, cx])

    a["cdp_x"] = (("iline", "xline"), lh_x)
    a["cdp_y"] = (("iline", "xline"), lh_y)

    try:
        a.seis.fill_cdpna()
        a.seis.calc_corner_points()
        tf = a.seis.get_affine_transform()

        df = a.to_dataframe().reset_index()
        check = tf.transform(np.array([df.iline, df.xline]).T)
        assert np.allclose(df[["cdp_x", "cdp_y"]].values, check, atol=10e-3)
    except ValueError:
        pass


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
        tuples(integers(3, 10), integers(3, 10), integers(3, 10)),
        tuples(floats(-1000, 1000), floats(-1000, 1000)),
        floats(0.1, 10000),
        tuples(floats(0, 45), floats(0, 45)),
        floats(-180, 180),
        integers(1, 20),
    )
    @settings(deadline=None, max_examples=5)
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
    @settings(deadline=None, max_examples=2, print_blob=True)
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


def create_xy_dataset(dims, translate, scale, shear, rotate, **dim_kwargs):
    dataset = create3d_dataset(dims=dims, **dim_kwargs)
    il_grid, xl_grid = xr.broadcast(dataset.iline, dataset.xline)
    dataset["empty"] = xr.zeros_like(il_grid)
    df = dataset.drop_dims("twt").to_dataframe()

    transform = (
        Affine2D().translate(*translate).scale(scale).skew(*shear).rotate(rotate)
    )
    df[["cdp_x", "cdp_y"]] = transform.transform(df.reset_index()[["iline", "xline"]])
    dataset = df.to_xarray().drop_vars("empty")
    return dataset, transform


class TestCreate_xysel_segysak:
    """
    Test using segysak.xysel accessor
    """

    @given(
        tuples(integers(3, 10), integers(3, 10), integers(3, 10)),
        tuples(floats(-1000, 1000), floats(-1000, 1000)),
        floats(0.1, 10000),
        tuples(floats(0, 45), floats(0, 45)),
        floats(-180, 180),
        integers(1, 20),
    )
    @settings(deadline=None, max_examples=5)
    def test_full_stack_dataset_xysel(self, d, translate, scale, shear, rotate, samp):
        dataset, trsfm = create_xy_dataset(d, translate, scale, shear, rotate)
        dataset["data"] = (DimensionKeyField.threed_twt, np.random.rand(*d))

        test_points = np.dstack(
            [
                np.random.random(samp) * (d[0] - 0.1) + 0.1,
                np.random.random(samp) * (d[1] - 0.1) + 0.1,
            ]
        )[0]
        # make sure at least one point is in the box
        test_points[0, :] = d[0] / 2, d[1] / 2

        xys = trsfm.transform(test_points)

        res = dataset.segysak.xysel(xys, method="linear")
        assert isinstance(res, xr.Dataset)
        res = dataset.segysak.xysel(xys, method="nearest")
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
    @settings(deadline=None, max_examples=2, print_blob=True)
    def test_angle_stack_dataset_xysel(
        self, o, f, s, translate, scale, shear, rotate, samp
    ):
        dims = (50, 30, 5, o)
        dataset, trsfm = create_xy_dataset(
            dims, translate, scale, shear, rotate, first_offset=f, offset_step=s
        )
        dataset["data"] = (("iline", "xline", "twt", "offset"), np.random.rand(*dims))

        test_points = np.dstack(
            [
                np.random.random(samp) * (dims[0] - 0.1) + 0.1,
                np.random.random(samp) * (dims[1] - 0.1) + 0.1,
            ]
        )[0]
        # make sure at least one point is in the box
        test_points[0, :] = dims[0] / 2, dims[1] / 2

        xys = trsfm.transform(test_points)

        res = dataset.segysak.xysel(xys, method="linear")
        assert isinstance(res, xr.Dataset)
        res = dataset.segysak.xysel(xys, method="nearest")
        assert isinstance(res, xr.Dataset)
