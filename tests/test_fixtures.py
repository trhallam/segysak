import pytest
import pathlib

# import requests
# import zipfile

import numpy as np

import segyio

from segysak.segy import put_segy_texthead, get_segy_texthead, create_default_texthead
from segysak import create2d_dataset, create3d_dataset, create_seismic_dataset

TEMP_TEST_DATA_DIR = "test_data_temp"

TEST_SEGY_SIZE = 10
TEST_SEGY_REG = "test_reg.segy"
TEST_SEGY_SKEW = "test_skew.segy"

TEST_DATA_SEGYIO = "tests/test-data-segyio"
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


@pytest.fixture(scope="session")
def temp_dir(tmpdir_factory):
    tdir = tmpdir_factory.mktemp(TEMP_TEST_DATA_DIR)
    return pathlib.Path(str(tdir))


# def copy_segyio_tests(segyio_zip=SEGYIO_ZIP):
#     # download a copy of segyio as zip to tempdir and extract tests
#     segyio_tests = pathlib.Path("test-data-segyio")
#     segyio_tests.mkdir(exist_ok=True)
#     segyio_zip = segyio_tests / "segyio.zip"
#     if segyio_zip.exists():
#         print("Using cached segyio data")
#     else:
#         print("Downloading segyio test data")
#         r = requests.get(segyio_zip, allow_redirects=True)
#         open(segyio_zip, "wb").write(r.content)

#     try:
#         segyio_Zip = zipfile.ZipFile(segyio_zip)
#         files = [sgy for sgy in segyio_Zip.namelist() if sgy[-4:] == ".sgy"]
#         for file in files:
#             segyio_Zip.extract(file, segyio_tests / file)
#     except IOError:
#         print("Error with segyio zip file.")


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


@pytest.fixture(params=[(10, 11, "TWT"), (10, 11, "DEPTH")])
def empty2d(request):
    cdpn, n, dom = request.param
    return create2d_dataset((cdpn, n), sample_rate=1, first_cdp=1, vert_domain=dom)


@pytest.fixture(params=[(10, 11, 5, "TWT"), (10, 11, 5, "DEPTH")])
def empty2d_gath(request):
    cdpn, n, off, dom = request.param
    return create2d_dataset(
        (cdpn, n, off),
        sample_rate=1,
        first_cdp=1,
        vert_domain=dom,
        first_offset=2,
        offset_step=2,
    )


@pytest.fixture(params=[(10, 11, 12, "TWT"), (10, 11, 12, "DEPTH")])
def empty3d(request):
    iln, xln, n, dom = request.param
    return create3d_dataset(
        (iln, xln, n), sample_rate=1, first_iline=1, first_xline=1, vert_domain=dom
    )


@pytest.fixture(params=[(10, 11, 12, 5, "TWT"), (10, 11, 12, 5, "DEPTH")])
def empty3d_gath(request):
    iln, xln, n, off, dom = request.param
    return create3d_dataset(
        (iln, xln, n, off),
        sample_rate=1,
        first_iline=1,
        first_xline=1,
        vert_domain=dom,
        first_offset=2,
        offset_step=2,
    )


@pytest.fixture(params=[(10, 11, 12, "DEPTH"), (12, 5, 15, "DEPTH")])
def empty3d_depth(request):
    iln, xln, n, dom = request.param
    return create3d_dataset(
        (iln, xln, n), sample_rate=1, first_iline=1, first_xline=1, vert_domain=dom
    )


@pytest.fixture(params=[(10, 11, 12, "TWT"), (12, 5, 15, "TWT")])
def empty3d_twt(request):
    iln, xln, n, dom = request.param
    return create3d_dataset(
        (iln, xln, n), sample_rate=1, first_iline=1, first_xline=1, vert_domain=dom
    )


@pytest.fixture(params=[(10, 11, 12, "DEPTH"), (12, 5, 15, "TWT")])
def zeros3d(request):
    iln, xln, n, dom = request.param
    seisnc = create3d_dataset(
        (iln, xln, n), sample_rate=1, first_iline=1, first_xline=1, vert_domain=dom
    )
    seisnc = seisnc.seis.zeros_like()
    return seisnc
