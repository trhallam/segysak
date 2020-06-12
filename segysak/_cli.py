"""Command line script for interacting with SEGY data.
"""

import os
import logging
import pathlib
import click

from segysak.version import version as VERSION
from segysak.segy import segy_loader, ncdf2segy, segy_header_scan, get_segy_texthead
from segysak.tools import fix_bad_chars

# configuration setup
NAME = "segysak"
LOGGER = logging.getLogger(NAME)


def check_file(input_files):

    if input_files is None:
        LOGGER.error("Require input file/s")
        raise SystemExit

    checked_files = list()

    expanded_input = input_files.copy()

    for ifile in input_files:
        if "*" in ifile:
            ifile_path = pathlib.Path(ifile)
            parent = ifile_path.absolute().parent
            expanded_input += list(parent.glob(ifile_path.name))
            expanded_input.remove(ifile)

    for ifile in expanded_input:
        ifile = pathlib.Path(ifile)
        if ifile.exists():
            checked_files.append(ifile)
        else:
            LOGGER.error("Cannot find input {segyfile}")
            raise SystemExit

    return checked_files


def _action_ebcidc_out(arg, input_file):
    try:
        ebcidc = get_segy_texthead(input_file)
    except IOError:
        LOGGER.error("Input SEGY file was not found - check name and path")
        raise SystemExit

    if arg is None:
        print(ebcidc)
    else:
        ebcidc = fix_bad_chars(ebcidc)
        with open(arg, "w") as f:
            f.write(ebcidc)


def _action_ebcidc_in(input_ebcidc_file, input_file):
    pass


def guess_file_type(file):
    """
    Guess the file type. Currently guessing is only based on file extension.
    """
    _, file_extension = os.path.splitext(file)
    if any([ext in file_extension.upper() for ext in ["SEGY", "SGY"]]):
        return "SEGY"
    elif "SEISNC" in file_extension.upper():
        return "NETCDF"
    else:
        return None


@click.group(invoke_without_command=True, no_args_is_help=True)
@click.option(
    "--version",
    "-v",
    is_flag=True,
    help="Print application version name",
    default=False,
)
def cli(version):
    LOGGER.info(f"segysak v{VERSION}")
    if version:
        click.echo(f"{NAME} {VERSION}")
        pass

    print(locals())


@cli.command(help="Print SEGY EBCIDC header")
@click.argument("filename", type=click.Path(exists=True))
def ebcidc(filename):
    input_file = pathlib.Path(filename)
    click.echo(get_segy_texthead(input_file))


@cli.command(help="Scan trace headers and print value ranges")
@click.option(
    "--max-traces", "-m", type=int, default=1000, help="Number of traces to scan"
)
@click.argument("filename", type=click.Path(exists=True))
def scan(max_traces, filename):
    input_file = pathlib.Path(filename)
    hscan, nscan = segy_header_scan(input_file, max_traces_scan=max_traces)

    click.echo(f"Traces scanned: {nscan}")
    click.echo(
        "{:>40s} {:>8s} {:>10s} {:>10s}".format("Item", "Byte Loc", "Min", "Max")
    )
    for key, item in hscan.items():
        click.echo(
            "{:>40s} {:>8d} {:>10.0f} {:>10.0f}".format(key, item[0], item[1], item[2])
        )


@cli.command(
    help="Convert file between SEGY and NETCDF (direction is guessed or can be made explicit with the --output-type option)"
)
@click.argument("input-file", type=click.Path(exists=True))
@click.option("--output-file", "-o", type=str, help="Output file name", default=None)
@click.option("--iline", "-il", type=int, default=189, help="Inline byte location")
@click.option("--xline", "-xl", type=int, default=193, help="Crossline byte location")
@click.option(
    "--crop",
    type=int,
    nargs=4,
    default=None,
    help="Crop the input volume providing 4 parameters: minil maxil minxl maxxl",
)
@click.option(
    "--output-type",
    type=click.Choice(["SEGY", "NETCDF"], case_sensitive=False),
    help="Explicitly state the desired output file type by chosing one of the options",
    default=None,
)
def convert(output_file, input_file, iline, xline, crop, output_type):
    input_file = pathlib.Path(input_file)
    if output_type is None and output_file is not None:
        output_type = guess_file_type(output_file)
    elif output_type is None and output_file is None:
        """Because currently only one conversion exists we can guess the output from the input"""
        input_type = guess_file_type(input_file)
        if input_type:
            output_type = "SEGY" if input_type == "NETCDF" else "NETCDF"

    if output_type is None:
        click.echo(
            "Output type not recognised! Please provide the desired output file type explicitly using the --output-type option"
        )
        raise SystemExit

    click.echo(f"Converting file {input_file.name} to {output_type}")

    if len(crop) == 0:
        crop = None

    if output_type == "NETCDF":
        if output_file is None:
            output_file = input_file.stem + ".SEISNC"
        _ = segy_loader(
            input_file, ncfile=output_file, iline=iline, xline=xline, ix_crop=crop
        )
        click.echo(f"Converted file saved as {output_file}")
        LOGGER.info(f"NetCDF output written to {output_file}")
    elif output_type == "SEGY":
        if output_file is None:
            output_file = input_file.stem + ".segy"
        ncdf2segy(input_file, output_file, iline=iline, xline=xline)
        click.echo(f"Converted file saved as {output_file}")
        LOGGER.info(f"SEGY output written to {output_file}")
    else:
        click.echo(f"Conversion to output-type {output_type} is not implemented yet")
        raise SystemExit


if __name__ == "__main__":
    cli()
