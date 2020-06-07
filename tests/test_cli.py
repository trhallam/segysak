import sys
import pytest

from segysak._cli import cli, NAME

from tests.test_fixtures import temp_dir, temp_segy, TEMP_TEST_DATA_DIR


@pytest.mark.parametrize("help_arg", ["-h", "--help"])
def test_help(help_arg):
    sys.argv = ["", help_arg]
    with pytest.raises(SystemExit):
        cli()


def test_version():
    sys.argv = ["", "-v"]
    with pytest.raises(SystemExit):
        cli()


def test_no_input_file():
    sys.argv = [""]
    with pytest.raises(SystemExit):
        cli()


@pytest.mark.parametrize("ebc", ["ebcidc"])
def test_dump_ebcidc(temp_dir, temp_segy, ebc):
    # test_file = temp_dir / "ebcidc_dump.txt"
    sys.argv = ["", ebc, str(temp_segy)]
    with pytest.raises(SystemExit):
        cli()
