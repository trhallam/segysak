import numpy as np

import pytest

from hypothesis import given, assume, settings
from hypothesis.strategies import integers, text, floats, tuples, sampled_from
from hypothesis.extra.numpy import arrays

from segysak._seismic_dataset import (
    _check_vert_units,
    _check_input,
    _dataset_coordinate_helper,
)
from segysak._seismic_dataset import (
    create_seismic_dataset,
    create2d_dataset,
    create3d_dataset,
)

from segysak._keyfield import VerticalUnits, VerticalKeyField


class TestCheckers:
    """
    Test data checking utilities
    """

    def test_check_input_does_nothing_to_None(self):
        assert _check_input(None) is None

    @given(integers(0, 100000))
    def test_check_input_turns_int_into_array(self, i):
        assert len(_check_input(i)) == i

    def test_check_input_converts_to_array(self):
        assert isinstance(_check_input([0, 0]), np.ndarray)

    def test_check_input_raises_error_at_multidimensions(self):
        with pytest.raises(ValueError):
            _check_input([[0, 0]])

    @pytest.mark.parametrize("u", list(VerticalUnits))
    def test_vertical_units_pass_checking(self, u):
        assert _check_vert_units(u) == u

    @given(text())
    def test_illegal_vertical_units_raise_errors(self, t):
        assume(t not in list(VerticalUnits))
        with pytest.raises(ValueError):
            _check_vert_units(t)

    @pytest.mark.parametrize("p", list(VerticalKeyField.values()))
    def test_domains_pass_checking(self, p):
        _, domain = _dataset_coordinate_helper(None, p)
        assert p == domain

    @given(text())
    def test_illegal_domain_raise_errors(self, t):
        assume(t not in list(VerticalKeyField.values()))
        with pytest.raises(ValueError):
            _dataset_coordinate_helper(None, t)


class TestCreateSeismicDataset:
    """
    Test creating a seismic dataset with various dimensions and sizes
    """

    @given(integers(0, 10000), integers(0, 100000))
    def test_create_2D_seismic_dataset_with_integers(self, s, t):
        dataset = create_seismic_dataset(
            twt=s, depth=None, cdp=t, iline=None, xline=None, offset=None
        )
        assert len(dataset.dims) == 2
        assert dataset.dims["twt"] == s

    @given(integers(0, 45))
    @settings(max_examples=10)
    def test_create_2D_seismic_dataset_with_offsets(self, o):
        dataset = create_seismic_dataset(
            twt=100, depth=None, cdp=1000, iline=None, xline=None, offset=o
        )
        assert len(dataset.dims) == 3

    @given(integers(0, 10000), integers(0, 100000), integers(0, 100000))
    def test_create_3D_seismic_dataset_with_integers(self, s, i, x):
        dataset = create_seismic_dataset(
            twt=None, depth=s, cdp=None, iline=i, xline=x, offset=None
        )
        assert len(dataset.dims) == 3
        assert dataset.dims["depth"] == s

    @given(
        arrays(float, shape=integers(0, 10000), elements=floats(-1000, 1000)),
        integers(0, 10000),
    )
    def test_create_2D_seismic_dataset_with_arrays(self, a, t):
        dataset = create_seismic_dataset(
            twt=a, depth=None, cdp=t, iline=None, xline=None, offset=None
        )
        assert len(dataset.dims) == 2
        assert dataset.dims["twt"] == len(a)

    @given(integers(0, 100))
    @settings(max_examples=10)
    def test_create_2D_seismic_dataset_with_multiple_dimensions(self, d):
        dims = {str(i): i for i in range(d)}
        dataset = create_seismic_dataset(
            twt=100, depth=None, cdp=1000, iline=None, xline=None, offset=None, **dims
        )
        assert len(dataset.dims) == d + 2

    def test_mutally_not_allowed_arguments(self):
        with pytest.raises(ValueError):
            ds = create_seismic_dataset(cdp=100, iline=100, xline=100)

        with pytest.raises(ValueError):
            ds = create_seismic_dataset(cdp=100, iline=100)

        with pytest.raises(ValueError):
            ds = create_seismic_dataset(cdp=100, xline=100)

    def test_mutually_required_arguments(self):

        with pytest.raises(ValueError):
            ds = create_seismic_dataset(iline=100)

        with pytest.raises(ValueError):
            ds = create_seismic_dataset(xline=100)


class TestCreate2DDataset:
    """
    Test creating 2D datasets with various shapes
    """

    @given(integers(1, 10000), integers(0, 100), integers(1, 100))
    def test_create_2D_dataset_custom_sampling(self, s, f, r):
        dataset = create2d_dataset(
            dims=(100, s), first_sample=f, sample_rate=r, vert_domain="TWT"
        )
        assert dataset.twt.data.max() == f + s * r - r

    @given(integers(1, 10000), integers(0, 100), integers(1, 100))
    def test_create_2D_dataset_custom_cdp(self, t, f, s):
        dataset = create2d_dataset(dims=(t, 100), first_cdp=f, cdp_step=s)
        assert dataset.cdp.data.max() == f + s * t - s

    @given(integers(1, 10000), integers(0, 100), integers(1, 100), integers(0, 50))
    def test_create_2D_dataset_wfirstoffset(self, s, f, r, o):
        dataset = create2d_dataset(
            dims=(100, s, 5),
            first_cdp=f,
            cdp_step=s,
            sample_rate=r,
            first_offset=o,
            offset_step=10,
        )
        assert dataset.offset.data.max() == 4 * 10 + o


class TestCreate3DDataset:
    """
    Test creating 3D datasets with various shapes
    """

    @given(
        tuples(integers(1, 10000), integers(1, 10000), integers(0, 1000)),
        sampled_from(list(VerticalUnits)),
    )
    def test_create_full_stack_dataset(self, d, u):
        dataset = create3d_dataset(dims=d, vert_units=u)
        assert dataset.d3_domain == "TWT"
        assert dataset.measurement_system == u

    @given(
        integers(15, 60),
        floats(0, 15),
        floats(1, 15),
        sampled_from(["TWT", "twt", "DEPTH", "depth"]),
    )
    def test_create_angle_stack_dataset(self, o, f, s, d):
        dataset = create3d_dataset(
            (1000, 1000, 100, o), first_offset=f, offset_step=s, vert_domain=d
        )
        assert dataset.d3_domain == d.upper()
        assert len(dataset.dims) == 4
