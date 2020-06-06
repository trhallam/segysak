import sys
import pytest

from segysak._cli import main, NAME

from test_fixtures import temp_dir, temp_segy, TEMP_TEST_DATA_DIR


@pytest.mark.parametrize("help_arg", ["-h", "--help"])
def test_help(help_arg):
    sys.argv = ["", help_arg]
    with pytest.raises(SystemExit):
        main()


def test_vertsion():
    sys.argv = ["", "-V"]
    with pytest.raises(SystemExit):
        main()


def test_no_input_file():
    sys.argv = [""]
    with pytest.raises(SystemExit):
        main()


def test_logging_no_spec():
    sys.argv = ["", "-L"]
    with pytest.raises(SystemExit):
        main()


@pytest.mark.parametrize("ebc", ["-e", "--ebcidc"])
def test_dump_ebcidc(temp_dir, temp_segy, ebc):
    # test_file = temp_dir / "ebcidc_dump.txt"
    sys.argv = ["", str(temp_segy), ebc]
    main()
