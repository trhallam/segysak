import pytest
import pathlib

# import requests
# import zipfile

import numpy as np

import segyio

from segysak.segy import put_segy_texthead, create_default_texthead
from segysak import create2d_dataset, create3d_dataset

TEST_SEGY_SIZE = 10
TEST_SEGY_REG = "test_reg.segy"
TEST_SEGY_SKEW = "test_skew.segy"

TEST_DATA_PATH = pathlib.Path(__file__).parent.absolute()
TEST_DATA_SEGYIO = TEST_DATA_PATH / "test-data-segyio"
TEST_DATA_SEGYSAK = TEST_DATA_PATH / "test-data-segysak"
TEST_DATA_VOLVE = TEST_DATA_PATH / "test-data-volve"

# fname, dim_byte_fiekds, extra_byte_fields, segyio_kwargs
TEST_FILES_SEGYIO_3D = [
    (TEST_DATA_SEGYIO / "acute-small.sgy", {"iline": 189, "xline": 193}, None, {}),
    (
        TEST_DATA_SEGYIO / "f3-lsb.sgy",
        {"iline": 189, "xline": 193},
        None,
        {"endian": "little"},
    ),
    (
        TEST_DATA_SEGYIO / "f3.sgy",
        {"iline": 189, "xline": 193},
        {"cdp_x": 181, "cdp_y": 185},
        {},
    ),
    (TEST_DATA_SEGYIO / "inv-acute-small.sgy", {"iline": 189, "xline": 193}, None, {}),
    (TEST_DATA_SEGYIO / "left-small.sgy", {"iline": 189, "xline": 193}, None, {}),
    (TEST_DATA_SEGYIO / "normal-small.sgy", {"iline": 189, "xline": 193}, None, {}),
    (TEST_DATA_SEGYIO / "obtuse-small.sgy", {"iline": 189, "xline": 193}, None, {}),
    (TEST_DATA_SEGYIO / "reflex-small.sgy", {"iline": 189, "xline": 193}, None, {}),
    (TEST_DATA_SEGYIO / "right-small.sgy", {"iline": 189, "xline": 193}, None, {}),
    (TEST_DATA_SEGYIO / "small.sgy", {"iline": 189, "xline": 193}, None, {}),
    (
        TEST_DATA_SEGYIO / "left-small.sgy",
        {"iline": 189, "xline": 193},
        {"cdp_x": 181, "cdp_y": 185},
        {},
    ),
    (
        TEST_DATA_SEGYIO / "small-lsb.sgy",
        {"iline": 189, "xline": 193},
        None,
        {"endian": "little"},
    ),
    (TEST_DATA_SEGYIO / "straight-small.sgy", {"iline": 189, "xline": 193}, None, {}),
    (TEST_DATA_SEGYIO / "text-embed-null.sgy", {"iline": 189, "xline": 193}, None, {}),
]

# fname, dim_byte_fiekds, extra_byte_fields, segyio_kwargs
TEST_FILES_SEGYIO_3DG = [
    (
        TEST_DATA_SEGYIO / "small-ps-dec-il-inc-xl-off.sgy",
        {"iline": 189, "xline": 193, "offset": 37},
        None,
        {},
    ),
    (
        TEST_DATA_SEGYIO / "small-ps-dec-il-off-inc-xl.sgy",
        {"iline": 189, "xline": 193, "offset": 37},
        None,
        {},
    ),
    (
        TEST_DATA_SEGYIO / "small-ps-dec-il-xl-inc-off.sgy",
        {"iline": 189, "xline": 193, "offset": 37},
        None,
        {},
    ),
    (
        TEST_DATA_SEGYIO / "small-ps-dec-il-xl-off.sgy",
        {"iline": 189, "xline": 193, "offset": 37},
        None,
        {},
    ),
    (
        TEST_DATA_SEGYIO / "small-ps-dec-off-inc-il-xl.sgy",
        {"iline": 189, "xline": 193, "offset": 37},
        None,
        {},
    ),
    (
        TEST_DATA_SEGYIO / "small-ps-dec-xl-inc-il-off.sgy",
        {"iline": 189, "xline": 193, "offset": 37},
        None,
        {},
    ),
    (
        TEST_DATA_SEGYIO / "small-ps-dec-xl-off-inc-il.sgy",
        {"iline": 189, "xline": 193, "offset": 37},
        None,
        {},
    ),
    (
        TEST_DATA_SEGYIO / "small-ps.sgy",
        {"iline": 189, "xline": 193, "offset": 37},
        None,
        {},
    ),
]

# fname, dim_byte_fiekds, extra_byte_fields, segyio_kwargs
TEST_FILES_SEGYIO_2D = []

# fname, dim_byte_fiekds, extra_byte_fields, segyio_kwargs
TEST_FILES_SEGYIO_2DG = []

# fname, dim_byte_fiekds, extra_byte_fields, segyio_kwargs
TEST_FILES_SEGYIO_BLANKHEAD = [
    (TEST_DATA_SEGYIO / "1x1.sgy", {}, None, {}),
    (TEST_DATA_SEGYIO / "1xN.sgy", {}, None, {}),
    (TEST_DATA_SEGYIO / "Mx1.sgy", {}, None, {}),
    (TEST_DATA_SEGYIO / "shot-gather.sgy", {}, None, {}),
]

TEST_FILES_SEGYIO_ALL = (
    TEST_FILES_SEGYIO_BLANKHEAD + TEST_FILES_SEGYIO_3D + TEST_FILES_SEGYIO_3DG
)

# fname, dim_byte_fiekds, extra_byte_fields, segyio_kwargs
TEST_FILES_VOLVE_2D = [
    (
        TEST_DATA_VOLVE / "volve10r12-full-z-il10117sub.sgy",
        {"cdp": 21},
        {"cdp_x": 73, "cdp_y": 77},
        {},
    ),
]


def create_temp_segy(n, test_file, skew=False):
    data = np.zeros((n, n, n))
    spec = segyio.spec()
    # to create a file from nothing, we need to tell segyio about the structure of
    # the file, i.e. its inline numbers, crossline numbers, etc. You can also add
    # more structural information, but offsets etc. have sensible defautls. This is
    # the absolute minimal specification for a N-by-M volume
    spec.sorting = 2
    spec.format = 1
    spec.iline = 189
    spec.xline = 193
    spec.samples = range(n)
    if skew:
        spec.ilines = range(n + n - 1)
    else:
        spec.ilines = range(n)
    spec.xlines = range(n)

    xl_val = range(n + 1, n + n + 1, 1)
    il_val = range(1, n + 1, 1)

    cdp_x, cdp_y = np.meshgrid(il_val, xl_val)

    if skew:
        for i, x in enumerate(cdp_x):
            cdp_x[i, :] = x + i

    with segyio.create(test_file, spec) as segyf:
        for i, (t, il, xl) in enumerate(
            zip(data.reshape(-1, n), cdp_x.ravel(), cdp_y.ravel())
        ):
            segyf.header[i] = {
                segyio.su.offset: 1,
                189: il,
                193: xl,
                segyio.su.cdpx: il * 1000,
                segyio.su.cdpy: xl * 1000,
                segyio.su.ns: n,
            }
            segyf.trace[i] = t
        segyf.bin.update(
            tsort=segyio.TraceSortingFormat.INLINE_SORTING,
            hdt=1000,
            mfeet=1,
            jobid=1,
            lino=1,
            reno=1,
            ntrpr=n * n,
            nart=n * n,
            fold=1,
        )

    if not skew:
        put_segy_texthead(test_file, create_default_texthead())
    else:
        put_segy_texthead(
            test_file, create_default_texthead(override={7: "Is Skewed Test"})
        )


@pytest.fixture(
    scope="module",
    params=[
        (TEST_SEGY_SIZE, TEST_SEGY_REG, False),
        (TEST_SEGY_SIZE, TEST_SEGY_SKEW, True),
    ],
    ids=["reg", "skewed"],
)
def temp_segy(temp_dir, request):
    n, file, skew = request.param
    create_temp_segy(n, str(temp_dir / file), skew)
    return temp_dir / file


@pytest.fixture(
    scope="session",
    params=TEST_FILES_SEGYIO_ALL,
    ids=[file[0].stem for file in TEST_FILES_SEGYIO_ALL],
)
def segyio_all_test_files(request):
    return request.param


@pytest.fixture(
    scope="session",
    params=TEST_FILES_SEGYIO_3D,
    ids=[file[0].stem for file in TEST_FILES_SEGYIO_3D],
)
def segyio3d_test_files(request):
    return request.param


@pytest.fixture(
    scope="session",
    params=TEST_FILES_SEGYIO_3DG,
    ids=[file[0].stem for file in TEST_FILES_SEGYIO_3DG],
)
def segyio3dps_test_files(request):
    return request.param


@pytest.fixture(
    scope="session",
    params=TEST_FILES_SEGYIO_BLANKHEAD,
    ids=[file[0].stem for file in TEST_FILES_SEGYIO_BLANKHEAD],
)
def segyio_nohead_test_files(request):
    return request.param


@pytest.fixture(
    scope="session",
)
def segy_with_nonascii_text():
    return TEST_DATA_SEGYSAK / "f3-ebcidctest.sgy"


@pytest.fixture(
    scope="session",
    params=TEST_FILES_VOLVE_2D,
    ids=[file[0].stem for file in TEST_FILES_VOLVE_2D],
)
def volve_segy2d(request):
    return request.param
