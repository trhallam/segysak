import sys
import pytest
from click.testing import CliRunner

from segysak._cli import cli, NAME

from .conftest import TEMP_TEST_DATA_DIR


@pytest.mark.parametrize("help_arg", ["-h", "--help"])
def test_help(help_arg):
    sys.argv = ["", help_arg]
    with pytest.raises(SystemExit):
        cli()


def test_version():
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert NAME in result.output


def test_no_input_file():
    sys.argv = [""]
    with pytest.raises(SystemExit):
        cli()


@pytest.mark.parametrize("cmd", ["scan", "ebcidc", "convert", "scrape"])
def test_all_subcommands(temp_segy, cmd):
    runner = CliRunner()
    result = runner.invoke(cli, [cmd, str(temp_segy)])
    print(dir(result))
    print(result.stdout)
    print(result.output)
    print(result.exception)

    assert result.exit_code == 0
