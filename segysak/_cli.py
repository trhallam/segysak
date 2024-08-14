"""Command line script for interacting with SEG-Y data.
"""

from typing import Union, List, Callable, Dict, Any, Tuple
from functools import wraps
from dataclasses import dataclass
import os
import sys
import pathlib
import click
import re
import xarray as xr
import pandas as pd
from loguru import logger

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


@dataclass
class Pipeline:
    input_file: Union[os.PathLike, None] = None
    output_file: Union[os.PathLike, None] = None
    end: bool = False  # the pipeline should be terminated
    ds: Union[xr.Dataset, None] = None  # the loaded dataset
    dimension: Union[Tuple[Tuple[str, int]], None] = (
        None  # the dimension name-byte pairs
    )
    variable: Union[Tuple[Tuple[str, int]], None] = (
        None  # the header variable name-byte pairs
    )

    @property
    def dims(self) -> Dict[str, int]:
        dd = {}
        if self.dimension:
            dd.update({d: b for d, b in self.dimension})
        return dd

    @property
    def vars(self) -> Dict[str, int]:
        vd = {}
        if self.variable:
            vd.update({v: b for v, b in self.variable})
        return vd

    def debug_ds(self):
        logger.debug(f"pipeline.ds:\n--------------\n{self.ds}\n--------------")


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


@click.group(no_args_is_help=True, chain=True)
@click.option(
    "--version",
    "-v",
    is_flag=True,
    help="Print application version name",
    default=False,
)
@click.option(
    "--debug-level", default="INFO", type=click.Choice(["ERROR", "INFO", "DEBUG"])
)
@click.argument(
    "file", metavar="FILE", type=click.Path(exists=True), nargs=1, required=True
)
def cli(version: str, debug_level: str, file: str):
    """
    The SEG-Y Swiss Army Knife (SEGY-SAK) is a tool for managing segy data.
    It can read and dump ebcidc headers, scan trace headers, convert SEG-Y to SEISNC and vice versa.
    """
    if version:
        click.echo(f"{NAME} {VERSION}", err=True)
        raise SystemExit
    # elif debug_level == "DEBUG":
    #     click.echo(f"segysak v{VERSION}", err=True)

    logger.remove()
    logger.add(
        sys.stderr,
        format="<d>segysak:{function:<10}</d> <level>{message}</level>",
        colorize=True,
        level=debug_level,
    )
    logger.debug(f"segysak v{VERSION}")
    pass


@cli.result_callback(replace=True)
def pipeline(
    processors: List[Callable], file: str, *args: Any, **kwargs: Dict[Any, Any]
):
    pipe = Pipeline(input_file=pathlib.Path(file))

    logger.debug("Begin process pipeline")

    for processor in processors:
        pipe = processor(pipe)
        if pipe.end:
            # pipeline ends
            logger.debug("End process pipeline")
            raise SystemExit()


def processor(f: Callable) -> Callable:
    """Helper decorator to rewrite a function so that it returns another
    function from it but also accepts the pipeline as first argument.
    """

    @wraps(f)
    def new_func(*args: Any, **kwargs: Dict[Any, Any]):
        def processor(pipeline: Pipeline):
            return f(pipeline, *args, **kwargs)

        return processor

    return new_func


def get_ebcidc(input_file: os.PathLike, name: bool, colour: bool, new_line: bool):
    ebcidc_line_re = re.compile("^(C[\\s|\\d]\\d)?(.+)$")
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


@cli.command()
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
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path
    ),
    nargs=1,
    required=False,
    help="Set the output files to have text from specified file or stdin.",
)
@click.pass_context
@processor
def ebcidc(pipeline, ctx, new_line, name, no_colour, set_txt):
    """Print SEG-Y EBCIDC header [chainable DS | FILE -> End]"""
    logger.debug(f"PARAM: {ctx.params}")

    if set_txt:
        with open(set_txt) as otxt:
            txt = otxt.readlines()
        set_ebcidc(pipeline.input_file, txt)
    else:
        get_ebcidc(pipeline.input_file, name, (not no_colour), new_line)

    pipeline.end = True
    return pipeline


@cli.command()
@click.option(
    "--max-traces", "-m", type=click.INT, default=1000, help="Number of traces to scan"
)
@click.pass_context
@processor
def scan(pipeline: Pipeline, ctx: click.Context, max_traces: int) -> Pipeline:
    "Scan trace headers and print a summary of value ranges [chainable DS | FILE -> End]"
    logger.debug(f"PARAM: {ctx.params}")
    hscan = segy_header_scan(pipeline.input_file, max_traces_scan=max_traces)
    logger.debug(f"Traces scanned: {hscan.nscan}")

    pd.set_option("display.max_rows", hscan.shape[0])
    click.echo(hscan[["byte_loc", "min", "max", "mean"]])

    pipeline.end = True
    return pipeline


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
@click.pass_context
@processor
def scrape(
    pipeline: Pipeline,
    ctx: click.Context,
    ebcidc: bool = False,
    trace_headers: bool = False,
) -> Pipeline:
    """Scrape the file meta information and output it to text file. [chainable DS | FILE -> FILE]

    If no options are specified both will be output. The output file will be
    <filename>.txt for the EBCIDC and <filename>.csv for
    trace headers.

    The trace headers can be read back into Python using
    pandas.read_csv(<filename>.csv, index_col=0)
    """
    logger.debug(f"PARAM: {ctx.params}")
    ebcidc_name = pipeline.input_file.with_suffix(".txt")
    header_name = pipeline.input_file.with_suffix(".csv")

    if ebcidc == False and trace_headers == False:
        ebcidc = True
        trace_headers = True

    if ebcidc:
        txt = get_segy_texthead(pipeline.input_file)
        with open(ebcidc_name, "w") as txtfile:
            txtfile.writelines(txt)
        logger.debug(f"Wrote ebcidc: {ebcidc_name}")

    if trace_headers:
        head_df = segy_header_scrape(pipeline.input_file)
        head_df.to_csv(header_name)
        logger.debug(f"Wrote headers: {header_name}")

    pipeline.end = True
    return pipeline


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


@cli.command("sgy")
@click.option(
    "--dimension",
    "-d",
    metavar="NAME BYTE",
    nargs=2,
    type=(click.STRING, click.INT),
    default=None,
    help="Data dimension NAME and trace header BYTE.",
    multiple=True,
)
@click.option(
    "--variable",
    "-v",
    metavar="NAME BYTE",
    nargs=2,
    type=(click.STRING, click.INT),
    default=None,
    help="Data header variable NAME and trace header BYTE.",
    multiple=True,
)
@click.option(
    "-o",
    "--output",
    metavar="FILE",
    nargs=1,
    type=click.Path(exists=False, path_type=pathlib.Path),
    help="Output file path",
)
@click.pass_context
@processor
def sgy(
    pipeline: Pipeline,
    ctx: click.Context,
    dimension: Tuple[Tuple[str, int]],
    variable: Tuple[Tuple[str, int]],
    output: Union[pathlib.Path, None],
) -> Pipeline:
    """Load or export a SEG-Y file [chainable [DS -> FILE | FILE -> DS]]"""
    logger.debug(f"PARAM: {ctx.params}")

    if dimension:
        pipeline.dimension = dimension

    if variable:
        pipeline.variable = variable

    if output is not None:
        try:
            assert pipeline.ds is not None, "SEGY output requires volume input"
            assert (
                pipeline.dims is not None
            ), "SEGY output requires dimensions, set: --dimension"
        except AssertionError:
            logger.exception("Poorly formed pipeline for SEG-Y output")
            raise SystemExit

        pipeline.ds.seisio.to_segy(
            output, trace_header_map=pipeline.vars, **pipeline.dims
        )
        logger.debug(f"Wrote SGY: {output}")
        pipeline.end = True
    else:
        # load the file
        pipeline.ds = xr.open_dataset(
            pipeline.input_file,
            dim_byte_fields=pipeline.dims,
            extra_byte_fields=pipeline.vars,
        )
        pipeline.debug_ds()

    return pipeline


@cli.command("crop")
@click.option(
    "-c",
    "--crop",
    "crops",
    metavar="DIM MIN MAX ...",
    nargs=3,
    type=(click.STRING, click.FLOAT, click.FLOAT),
    help="The cropping dimension name with min and max values.",
    multiple=True,
)
@click.pass_context
@processor
def crop(
    pipeline: Pipeline, ctx: click.Context, crops: Tuple[Tuple[str, float, float]]
):
    """Crop an input volume [chainable DS -> DS]"""
    logger.debug(f"PARAM: {ctx.params}")

    if crops:
        pipeline.ds = pipeline.ds.sel(**{d: slice(mi, mx) for d, mi, mx in crops})
    pipeline.debug_ds()
    return pipeline


if __name__ == "__main__":
    cli()
