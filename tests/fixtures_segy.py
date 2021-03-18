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

TEST_DATA_SEGYIO = pathlib.Path(__file__).parent.absolute() / "test-data-segyio"
TEST_DATA_SEGYIO = pathlib.Path(TEST_DATA_SEGYIO)

TEST_FILES_SEGYIO_3D = [
    ("acute-small.sgy", {"iline": 189, "xline": 193}),
    ("f3-lsb.sgy", {"endian": "little", "iline": 189, "xline": 193}),
    ("f3.sgy", {"iline": 189, "xline": 193, "cdpx": 181, "cdpy": 185}),
    ("inv-acute-small.sgy", {"iline": 189, "xline": 193}),
    ("left-small.sgy", {"iline": 189, "xline": 193}),
    ("normal-small.sgy", {"iline": 189, "xline": 193}),
    ("obtuse-small.sgy", {"iline": 189, "xline": 193}),
    ("reflex-small.sgy", {"iline": 189, "xline": 193}),
    ("right-small.sgy", {"iline": 189, "xline": 193}),
    ("small.sgy", {"iline": 189, "xline": 193}),
    ("left-small.sgy", {"iline": 189, "xline": 193, "cdpx": 181, "cdpy": 185}),
    ("small-lsb.sgy", {"iline": 189, "xline": 193, "endian": "little"}),
    ("straight-small.sgy", {"iline": 189, "xline": 193}),
    ("text-embed-null.sgy", {"iline": 189, "xline": 193}),
]

TEST_FILES_SEGYIO_3DG = [
    ("small-ps-dec-off-inc-il-xl.sgy", {"iline": 189, "xline": 193, "offset": 37}),
    ("small-ps-dec-xl-inc-il-off.sgy", {"iline": 189, "xline": 193, "offset": 37}),
    ("small-ps.sgy", {"iline": 189, "xline": 193, "offset": 37}),
    ("small-ps-dec-il-xl-off.sgy", {"iline": 189, "xline": 193, "offset": 37}),
    ("small-ps.sgy", {"iline": 189, "xline": 193, "offset": 37}),
]

TEST_FILES_SEGYIO_2D = [
    None,
]

TEST_FILES_SEGYIO_2DG = [
    None,
]

TEST_FILES_SEGYIO_BLANKHEAD = [
    ("1x1.sgy", {}),
    ("1xN.sgy", {}),
    ("Mx1.sgy", {}),
    ("shot-gather.sgy", {}),
]

TEST_FILES_SEGYIO_ALL = (
    TEST_FILES_SEGYIO_BLANKHEAD + TEST_FILES_SEGYIO_3D + TEST_FILES_SEGYIO_3DG
)


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

    cdpx, cdpy = np.meshgrid(il_val, xl_val)

    if skew:
        for i, x in enumerate(cdpx):
            cdpx[i, :] = x + i

    with segyio.create(test_file, spec) as segyf:
        for i, (t, il, xl) in enumerate(
            zip(data.reshape(-1, n), cdpx.ravel(), cdpy.ravel())
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
    scope="session",
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
    params=[
        (TEST_DATA_SEGYIO / file, segyio_kwargs)
        for file, segyio_kwargs in TEST_FILES_SEGYIO_ALL
    ],
    ids=[file for file, _ in TEST_FILES_SEGYIO_ALL],
)
def segyio_all_test_files(request):
    file, segyio_kwargs = request.param
    return (file, segyio_kwargs)


@pytest.fixture(
    scope="session",
    params=[
        (TEST_DATA_SEGYIO / file, segyio_kwargs)
        for file, segyio_kwargs in TEST_FILES_SEGYIO_3D
    ],
    ids=[file for file, _ in TEST_FILES_SEGYIO_3D],
)
def segyio3d_test_files(request):
    file, segyio_kwargs = request.param
    return (file, segyio_kwargs)


@pytest.fixture(
    scope="session",
    params=[
        (TEST_DATA_SEGYIO / file, segyio_kwargs)
        for file, segyio_kwargs in TEST_FILES_SEGYIO_3DG
    ],
    ids=[file for file, _ in TEST_FILES_SEGYIO_3DG],
)
def segyio3dps_test_files(request):
    file, segyio_kwargs = request.param
    return (file, segyio_kwargs)


@pytest.fixture(
    scope="session",
    params=[
        (TEST_DATA_SEGYIO / file, segyio_kwargs)
        for file, segyio_kwargs in TEST_FILES_SEGYIO_BLANKHEAD
    ],
    ids=[file for file, _ in TEST_FILES_SEGYIO_BLANKHEAD],
)
def segyio_nohead_test_files(request):
    file, segyio_kwargs = request.param
    return (file, segyio_kwargs)
