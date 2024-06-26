import pathlib
import pytest
from segysak.segy import segy_loader

TESTS_PATH = pathlib.Path(__file__).parent.absolute()
TEST_DATA_SEGYIO = TESTS_PATH / "test-data-segyio"
TEST_DATA_SEGYSAK = TESTS_PATH / "test-data-segysak"
TEST_DATA_VOLVE = TESTS_PATH / "test-data-volve"


@pytest.fixture(scope="session")
def f3_dataset():
    ds = segy_loader(
        TEST_DATA_SEGYIO / "f3.sgy",
        **{"iline": 189, "xline": 193, "cdp_x": 181, "cdp_y": 185},
    )

    ds["extra"] = ds.cdp_x * 0.0
    return ds


@pytest.fixture(scope="session")
def f3_horizon_exact(f3_dataset):
    df = f3_dataset.drop_vars("data").to_dataframe().reset_index()
    df["horizon"] = 50.0
    return df


@pytest.fixture(scope="session")
def f3_horizon_shift(f3_dataset):
    df = f3_dataset.drop_vars("data").to_dataframe().reset_index()
    df["horizon"] = 50.0
    df["cdp_x"] = df["cdp_x"] + 3.5
    df["cdp_y"] = df["cdp_y"] + 3.5
    return df


@pytest.fixture(scope="session")
def geometry_dataset():
    ds = segy_loader(TEST_DATA_SEGYSAK / "test-geometry.sgy")
    return ds


@pytest.fixture(scope="session")
def f3_withdead_dataset():
    ds = segy_loader(
        TEST_DATA_SEGYSAK / "f3-withdead.sgy",
        **{"iline": 189, "xline": 193, "cdp_x": 181, "cdp_y": 185},
    )
    return ds


@pytest.fixture(scope="session")
def volve_2d_dataset():
    ds = segy_loader(
        TEST_DATA_VOLVE / "volve10r12-full-z-il10117sub.sgy",
        cdp=21,
        cdp_x=73,
        cdp_y=77,
        vert_domain="DEPTH",
    )
    return ds
