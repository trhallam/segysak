import pytest

from segysak import create2d_dataset, create3d_dataset, create_seismic_dataset


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
