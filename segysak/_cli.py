"""Command line script for interacting with SEG-Y data.
"""

from typing import Tuple

import os
import sys
import pathlib
import click
import re

try:
    from ._version import version as VERSION
except ImportError:
    VERSION = None

if VERSION is None:
    try:
        from setuptools_scm import get_version

        VERSION = get_version(root="..", relative_to=__file__)
    except LookupError:
        VERSION = r"¯\_(ツ)_/¯"

from segysak.segy import (
    segy_converter,
    segy_writer,
    segy_header_scan,
    segy_header_scrape,
    get_segy_texthead,
    put_segy_texthead,
)
from segysak.tools import fix_bad_chars
from segysak.progress import Progress

# configuration setup
NAME = "segysak"


def _action_ebcidc_out(arg, input_file):
    try:
        ebcidc = get_segy_texthead(input_file)
    except IOError:
        click.secho(
            "Input SEG-Y file was not found - check name and path", fg="red", color=True
        )
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
    """
    The SEG-Y Swiss Army Knife (SEGY-SAK) is a tool for managing segy data.
    It can read and dump ebcidc headers, scan trace headers, convert SEG-Y to SEISNC and vice versa.
    """
    if version:
        click.echo(f"{NAME} {VERSION}", err=True)
    else:
        click.echo(f"segysak v{VERSION}", err=True)


# s plit a header line into line counter and text
ebcidc_line_re = re.compile("^(C[\\s|\\d]\\d)?(.+)$")


def get_ebcidc(input_file: os.PathLike, name: bool, colour: bool, new_line: bool):
    if name:
        click.secho(f"{input_file}:", color=colour, fg="green")
    text = get_segy_texthead(input_file)
    for line in text.split("\n"):
        c, txt = ebcidc_line_re.findall(line)[0]
        click.secho(c, color=colour, nl=False, fg="yellow")
        click.secho(txt, color=colour, nl=True)
    click.echo(nl=True)
    if name and new_line:
        click.echo("")


def set_ebcidc(input_file: os.PathLike, txt: str):
    put_segy_texthead(input_file, txt, line_counter=False)


@cli.command(help="Print SEG-Y EBCIDC header")
@click.argument(
    "files", metavar="FILE ...", type=click.Path(exists=True), nargs=-1, required=True
)
@click.option(
    "-n",
    "--new-line",
    is_flag=True,
    show_default=True,
    default=False,
    help="Print a blank line between consecutive header output.",
)
@click.option(
    "--name",
    is_flag=True,
    show_default=True,
    default=False,
    help="Print the file name before the text header output.",
)
@click.option(
    "--no-colour",
    is_flag=True,
    show_default=True,
    default=False,
    help="Decolourise output.",
)
@click.option(
    "-s",
    "--set",
    "set_txt",
    metavar="[TXTFILE]",
    type=click.File("r"),
    nargs=1,
    required=False,
    help="Set the output files to have text from specified file or stdin.",
)
def ebcidc(files, new_line, name, no_colour, set_txt):
    for f in files:
        input_file = pathlib.Path(f)
        if set_txt:
            txt = set_txt.readlines()
            set_ebcidc(input_file, txt)
        else:
            get_ebcidc(input_file, name, (not no_colour), new_line)


@cli.command(help="Scan trace headers and print value ranges")
@click.option(
    "--max-traces", "-m", type=click.INT, default=1000, help="Number of traces to scan"
)
@click.argument("filename", type=click.Path(exists=True))
def scan(max_traces, filename):
    input_file = pathlib.Path(filename)
    hscan = segy_header_scan(input_file, max_traces_scan=max_traces)
    click.echo(f"Traces scanned: {hscan.nscan}")
    import pandas as pd

    pd.set_option("display.max_rows", hscan.shape[0])
    click.echo(hscan[["byte_loc", "min", "max", "mean"]])

    return 0


@cli.command()
@click.option(
    "--ebcidc", "-e", is_flag=True, default=False, help="Output the text header"
)
@click.option(
    "--trace-headers",
    "-h",
    is_flag=True,
    default=False,
    help="Output the trace headers to csv",
)
@click.argument("filename", nargs=-1, type=click.Path(exists=True))
def scrape(filename, ebcidc=False, trace_headers=False):
    """Scrape the file meta information and output it to text file.

    If no options are specified both will be output. The output file will be
    <filename>.txt for the EBCIDC and <filename>.csv for
    trace headers.

    The trace headers can be read back into Python using
    pandas.read_csv(<filename>.csv, index_col=0)
    """
    with Progress(desc="File", total=len(filename)) as pbar:
        for file in filename:
            file = pathlib.Path(file)
            ebcidc_name = file.with_suffix(".txt")
            header_name = file.with_suffix(".csv")

            if ebcidc == False and trace_headers == False:
                ebcidc = True
                trace_headers = True

            if ebcidc:
                txt = get_segy_texthead(file)
                with open(ebcidc_name, "w") as txtfile:
                    txtfile.writelines(txt)

            if trace_headers:
                head_df = segy_header_scrape(file)
                head_df.to_csv(header_name)
            pbar.update(1)

    return 0


@cli.command(
    help="Convert file between SEG-Y and NETCDF (direction is guessed or can be made explicit with the --output-type option)"
)
@click.argument(
    "input-files",
    type=click.Path(exists=True),
    nargs=-1,
)
@click.option(
    "--output-file", "-o", type=click.STRING, help="Output file name", default=None
)
@click.option(
    "--iline", "-il", type=click.INT, default=189, help="Inline byte location"
)
@click.option(
    "--xline", "-xl", type=click.INT, default=193, help="Crossline byte location"
)
@click.option("--cdp-x", "-x", type=click.INT, default=181, help="CDP X byte location")
@click.option("--cdp-y", "-y", type=click.INT, default=185, help="CDP Y byte location")
@click.option(
    "--crop",
    type=click.INT,
    nargs=4,
    default=None,
    help="Crop the input volume providing 4 parameters: minil maxil minxl maxxl",
)
@click.option(
    "--output-type",
    type=click.Choice(["SEG-Y", "NETCDF"], case_sensitive=False),
    default=None,
    help="Explicitly state the desired output file type by choosing one of the options",
)
@click.option(
    "--dimension",
    "-d",
    type=click.STRING,
    default=None,
    help="Data dimension (domain) to write out, will default to TWT or DEPTH. Only used for writing to SEG-Y.",
)
def convert(
    output_file, input_files, iline, xline, cdp_x, cdp_y, crop, output_type, dimension
):

    if len(input_files) > 1 and output_file is not None:
        raise ValueError(
            "The output file option should not be used with multiple input files."
        )

    for input_file in input_files:
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

        if crop is None:
            crop_loc = None
        elif isinstance(crop, (list, tuple)) and len(crop) == 0:
            crop_loc = None
        else:
            crop_loc = crop

        if output_type == "NETCDF":
            if output_file is None:
                output_file_loc = input_file.stem + ".SEISNC"
            else:
                output_file_loc = output_file

            segy_converter(
                input_file,
                ncfile=output_file_loc,
                iline=iline,
                xline=xline,
                ix_crop=crop_loc,
                cdp_x=cdp_x,
                cdp_y=cdp_y,
            )
            click.echo(f"Converted file saved as {output_file_loc}")
            click.echo(f"NetCDF output written to {output_file_loc}")
        elif output_type == "SEG-Y":
            if output_file is None:
                output_file_loc = input_file.stem + ".segy"
            else:
                output_file_loc = output_file

            cdp_x = cdp_x
            cdp_y = cdp_y
            vars = locals()

            trace_header_map = {
                key: vars[key]
                for key in ["iline", "xline", "cdp_x", "cdp_y"]
                if vars[key] is not None
            }
            segy_writer(
                input_file,
                output_file_loc,
                trace_header_map=trace_header_map,
                dimension=dimension,
            )
            click.echo(f"Converted file saved as {output_file_loc}")
            click.echo(f"SEG-Y output written to {output_file_loc}")
        else:
            click.echo(
                f"Conversion to output-type {output_type} is not implemented yet"
            )
            raise SystemExit

    return 0


if __name__ == "__main__":
    cli()
