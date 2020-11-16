import pytest
import numpy as np
import xarray as xr

from hypothesis import given, assume, settings
from hypothesis.strategies import integers, text, floats, tuples, sampled_from
from hypothesis.extra.numpy import arrays

from affine import Affine

from segysak import create3d_dataset
from segysak._keyfield import DimensionKeyField


def TestIsFunctions():
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
