# pylint: disable=invalid-name, unused-variable
"""Functions to interact with segy data
"""

from warnings import warn

from collections import defaultdict
import importlib

import segyio
import netCDF4
import h5netcdf
import numpy as np
import pandas as pd
import xarray as xr

has_ipywidgets = importlib.find_loader("ipywidgets") is not None

if has_ipywidgets:
    from tqdm.autonotebook import tqdm
else:
    from tqdm import tqdm

from ._keyfield import CoordKeyField, AttrKeyField, VariableKeyField, DimensionKeyField
from ._seismic_dataset import create_seismic_dataset, create3d_dataset
from segysak.seisnc import create_empty_seisnc, set_seisnc_dims
from segysak.tools import check_crop, check_zcrop, _get_userid, _get_datetime

_SEGY_MEASUREMENT_SYSTEM = defaultdict(lambda: 0)
_SEGY_MEASUREMENT_SYSTEM[1] = "m"
_SEGY_MEASUREMENT_SYSTEM[2] = "ft"
_ISEGY_MEASUREMENT_SYSTEM = defaultdict(lambda: 0, m=1, ft=2)


class SegyLoadError(Exception):
    def __init__(self, message, errors):

        # Call the base class constructor with the parameters it needs
        super().__init__(message)

        # Now for your custom code...
        self.errors = errors


def _get_tf(var):
    return str(segyio.TraceField(var))


def _bag_slices(ind, n=10):
    """Take a list of indices and create a list of bagged indices. Each bag
    will contain n indices except for the last bag which will contain the
    remainder.

    This function is designed to support bagging/chunking of data to support
    memory management or distributed processing.

    Args:
        ind (list/array-like): The input list to create bags from.
        n (int, optional): The number of indices per bag. Defaults to 10.

    Returns:
        [type]: [description]
    """
    bag = list()
    prev = 0
    for i in range(len(ind)):
        if (i + 1) % n == 0:
            bag.append(slice(prev, i + 1, 1))
            prev = i + 1
    if prev != i + 1:
        bag.append(slice(prev, i + 1, 1))
    return bag


def get_segy_texthead(segyfile, ext_headers=False, **segyio_kwargs):
    """Return the ebcidc

    Args:
        segyfile (str): Segy File Path
        ext_headers (bool): Return EBCIDC and extended headers in list.
            Defaults to False

    Returns:
        str: Returns the EBCIDC text as a formatted paragraph.
    """

    segyio_kwargs["ignore_geometry"] = True
    with segyio.open(segyfile, "r", **segyio_kwargs) as segyf:
        text = segyf.text[0].decode("ascii", "replace")
        text = text.replace("�Cro", "    ")  # petrel weird export
        text = segyio.tools.wrap(text)
        if segyf.ext_headers and ext_headers:
            text2 = segyf.text[1].decode("ascii", "replace")
            return [text, text2]
        else:
            return text


def put_segy_texthead(segyfile, ebcidc, ext_headers=False, **segyio_kwargs):

    header = ""
    if isinstance(ebcidc, dict):
        for key in ebcidc:
            if not isinstance(key, int):
                warn(
                    "ebcidc dict contains not integer keys that will be ignored",
                    UserWarning,
                )
        for line in range(1, 41):
            try:
                test = ebcidc[line]
                if len(test) > 75:
                    warn(f"EBCIDC line {line} is too long - truncating", UserWarning)
                header = header + f"C{line:02d} " + f"{ebcidc[line][:75]:<76}"
            except KeyError:
                # line not specified in dictionary
                header = header + f"C{line:02d}" + " " * 76
        header = bytes(header, "utf8")
    elif isinstance(ebcidc, bytes):
        if len(ebcidc) > 3200:
            warn("Byte EBCIDC is too large - truncating", UserWarning)
        header = ebcidc[:3200]
    elif isinstance(ebcidc, str):
        if len(ebcidc) > 3200:
            warn("String EBCIDC is too large - truncating", UserWarning)
        header = bytes(ebcidc[:3200], "utf8")
    else:
        raise ValueError("Unknown ebcidc type")

    segyio_kwargs["ignore_geometry"] = True
    with segyio.open(segyfile, "r+", **segyio_kwargs) as segyf:
        segyf.text[0] = header


def _clean_texthead(text_dict):
    """Reduce texthead dictionary to 75 characters per line.

    The first 4 Characters of a segy EBCIDC should have the form "C01 " which
    is then follwed by 75 ascii characters.

    Input should have interger keys. Other keys will be ignored.
    Missing keys will be filled by blank lines.

    Args:
        text_dict (dict): line no and string pairs

    Returns:
        (dict): line no and string pairs ready for ebcidc input
    """
    output = dict()
    for line in range(1, 41, 1):
        try:
            line_str = text_dict[line]
            if len(line_str) > 75:
                line_str = line_str[0:75]
        except KeyError:
            line_str = ""
        output[line] = line_str
    return output


def create_default_texthead(override=None):
    """Returns a simple default textual header dictionary.

    Basic fields are auto populated and a dictionary indexing lines 1-40 can
    be passed to override keyword for adjustment. By default lines 6-34 are
    empty.

    Args:
        override (dict, optional): Overide any line . Defaults to None.

    Returns:
        (dict): Dictionary with keys 1-40 for textual header of segy file

    Example:
        >>> create_default_texthead(override={7:'Hello', 8:'World!'})
        {1: 'segysak SEGY Output',
        2: 'Data created by: username ',
        3: '',
        4: 'DATA FORMAT: SEG-Y;  DATE: 2019-06-09 15:14:00',
        5: 'DATA DESCRIPTION: SEGY format data output from segysak',
        6: '',
        7: 'Hello',
        8: 'World!',
        9: '',
        ...
        34: '',
        35: '*** BYTE LOCATION OF KEY HEADERS ***',
        36: 'CMP UTM-X 181-184, ALL COORDS X100, CMP UTM-Y 185-188',
        37: 'INLINE 189-193, XLINE 194-198, ',
        38: '',
        39: '',
        40: 'END TEXTUAL HEADER'}
    """
    user = _get_userid()
    today, time = _get_datetime()
    text_dict = {
        #      123456789012345678901234567890123456789012345678901234567890123456
        1: "segysak SEGY Output",
        2: f"Data created by: {user} ",
        4: f"DATA FORMAT: SEG-Y;  DATE: {today} {time}",
        5: "DATA DESCRIPTION: SEGY format data output from segysak",
        6: "",
        35: "*** BYTE LOCATION OF KEY HEADERS ***",
        36: "CMP UTM-X 181-184, ALL COORDS X100, CMP UTM-Y 185-188",
        37: "INLINE 189-193, XLINE 194-198, ",
        40: "END TEXTUAL HEADER",
    }
    if override is not None:
        for key, line in override.items():
            text_dict[key] = line
    return _clean_texthead(text_dict)


def _active_tracefield_segyio():
    header_keys = segyio.tracefield.keys.copy()
    # removed unused byte locations
    _ = header_keys.pop("UnassignedInt1")
    _ = header_keys.pop("UnassignedInt2")
    return header_keys


def _active_binfield_segyio():
    bin_keys = segyio.binfield.keys.copy()
    _ = bin_keys.pop("Unassigned1")
    _ = bin_keys.pop("Unassigned2")
    return bin_keys


def segy_header_scan(segyfile, max_traces_scan=1000, silent=False, **segyio_kwargs):
    """Perform a scan of the segy file headers and return ranges.

    Args:
        segyfile (string): Segy File Path
        max_traces_scan (int/str, optional): Number of traces to scan.
            For all set to 0 or 'all'. Defaults to 1000.
        silent (bool): Disable progress bar.

    Returns:
        dict: Contains keys from segyio.tracefield.keys with scanned values in
        a list [byte location, min, max]
        int: Number of scanned traces
    """
    if max_traces_scan == 0 or max_traces_scan == "all":
        max_traces_scan = None
    else:
        if not isinstance(max_traces_scan, int):
            raise ValueError("max_traces_scan must be int")

    hi = 0

    segyio_kwargs["ignore_geometry"] = True

    with segyio.open(segyfile, "r", **segyio_kwargs) as segyf:
        header_keys = _active_tracefield_segyio()
        lh = len(header_keys.keys())
        hmin = np.full(lh, np.nan)
        hmax = np.full(lh, np.nan)
        for hi, h in enumerate(
            tqdm(
                segyf.header,
                desc="Scanning Headers",
                total=max_traces_scan,
                disable=silent,
            )
        ):
            val = np.array(list(h.values()))
            hmin = np.nanmin(np.vstack((hmin, val)), axis=0)
            hmax = np.nanmax(np.vstack((hmax, val)), axis=0)
            if max_traces_scan is not None and hi + 1 >= max_traces_scan:
                break  # scan to max_traces_scan

    for i, (key, item) in enumerate(header_keys.items()):
        header_keys[key] = [item, hmin[i], hmax[i]]

    return header_keys, hi + 1


def segy_bin_scrape(segyfile, **segyio_kwargs):
    """Scrape binary header

    Args:
        segyfile (str): SEGY File path

    Returns:
        dict: Binary header
    """
    bk = _active_binfield_segyio()
    segyio_kwargs["ignore_geometry"] = True
    with segyio.open(segyfile, "r", **segyio_kwargs) as segyf:
        return {key: segyf.bin[item] for key, item in bk.items()}


def segy_header_scrape(segyfile, silent=False, **segyio_kwargs):
    """Scape all data from segy trace headers

    Args:
        segyfile (str): SEGY File path
        silent (bool): Disable progress bar.

    Returns:
        (pandas.DataFrame): Header information
    """
    header_keys = _active_tracefield_segyio()
    columns = header_keys.keys()
    lc = len(columns)
    segyio_kwargs["ignore_geometry"] = True
    with segyio.open(segyfile, "r", **segyio_kwargs) as segyf:
        ntraces = segyf.tracecount
        head_df = pd.DataFrame(
            data=np.full((ntraces, lc), np.nan, dtype=int), columns=columns
        )
        pb = tqdm(total=segyf.tracecount, desc="Scraping Headers", disable=silent)
        for hi, h in enumerate(segyf.header):
            head_df.iloc[hi, :] = np.array(list(h.values()))
            pb.update()
        pb.close()
        head_df.replace(to_replace=-2147483648, value=np.nan, inplace=True)
    return head_df


def _segy3dncdf(
    segyfile,
    ncfile,
    segyio_kwargs,
    n0,
    ns,
    head_df,
    il_head_loc,
    xl_head_loc,
    silent=False,
):

    with segyio.open(segyfile, "r", **segyio_kwargs) as segyf, h5netcdf.File(
        ncfile, "a"
    ) as seisnc:

        segyf.mmap()

        # create data variable
        seisnc_data = seisnc.create_variable(
            VariableKeyField.data.value, DimensionKeyField.threed.value, float
        )

        seisnc.flush()

        # work out fast and slow dir
        if head_df[xl_head_loc].diff().min() < 0:
            contig_dir = il_head_loc
            broken_dir = xl_head_loc
            slicer = lambda x, y: slice(x, y, ...)
        else:
            contig_dir = xl_head_loc
            broken_dir = il_head_loc
            slicer = lambda x, y: slice(y, x, ...)

        print(f"Fast direction is {broken_dir}")

        pb = tqdm(total=segyf.tracecount, desc="Converting SEGY", disable=silent)

        for contig, grp in head_df.groupby(contig_dir):
            for trc, val in grp.iterrows():
                seisnc_data[val.il_index, val.xl_index, :] = segyf.trace[trc][
                    n0 : ns + 1
                ]
                pb.update()
        pb.close()


def _segy3dxr(
    segyfile,
    ds,
    segyio_kwargs,
    n0,
    ns,
    head_df,
    il_head_loc,
    xl_head_loc,
    silent=False,
):

    with segyio.open(segyfile, "r", **segyio_kwargs) as segyf:

        segyf.mmap()

        # work out fast and slow dir
        if head_df[xl_head_loc].diff().min() < 0:
            contig_dir = il_head_loc
            broken_dir = xl_head_loc
            slicer = lambda x, y: slice(x, y, ...)
        else:
            contig_dir = xl_head_loc
            broken_dir = il_head_loc
            slicer = lambda x, y: slice(y, x, ...)

        print(f"Fast direction is {broken_dir}")

        pb = tqdm(total=segyf.tracecount, desc="Converting SEGY", disable=silent)
        shape = [ds.dims[d] for d in DimensionKeyField.threed.value]
        volume = np.zeros(shape)

        for contig, grp in head_df.groupby(contig_dir):
            for trc, val in grp.iterrows():
                volume[int(val.il_index), int(val.xl_index), :] = segyf.trace[trc][
                    n0 : ns + 1
                ]
                pb.update()
        pb.close()

    ds[VariableKeyField.data.value] = (DimensionKeyField.threed.value, volume)

    return ds


def _3dsegy_loader(
    segyfile,
    head_df,
    head_bin,
    ncfile=None,
    iline=189,
    xline=193,
    cdpx=181,
    cdpy=185,
    offset=None,
    vert_domain="TWT",
    data_type="AMP",
    crop=None,
    zcrop=None,
    silent=False,
    return_geometry=False,
    **segyio_kwargs,
):
    """Convert SEGY data to NetCDF4 File

    The output ncfile has the following structure
        Dimensions:
            vert - The vertical axis
            iline - Inline axis
            xline - Xline axis
        Variables:
            INLINE_3D - The inline numbering
            CROSSLINE_3D - The xline numbering
            CDP_X - Eastings
            CDP_Y - Northings
            CDP_TRACE - Trace Number
            data - The data volume
        Attributes:
            vert.units
            vert.data.units
            ns - Number of samples in vert
            ds - Sample rate

    Args:
        segyfile (str): Input segy file path
        ncfile (str): Output SEISNC file path.
        iline (int): Inline byte location.
        xline (int): Cross-line byte location.
        vert (str): Vertical sampling domain.
        units (str): Units of amplitude data.
        crop (list): List of minimum and maximum inline and crossline to output.
            Has the form '[min_il, max_il, min_xl, max_xl]'.
        zcrop (list): List of minimum and maximum vertical samples to output.
            Has the form '[min, max]'.
        silent (bool): Disable progress bar.

    """

    # get names of columns where stuff we want is
    il_head_loc = _get_tf(iline)
    xl_head_loc = _get_tf(xline)
    x_head_loc = _get_tf(cdpx)
    y_head_loc = _get_tf(cdpy)

    # calculate vert, inline and crossline ranges/meshgrids
    il0 = head_df[il_head_loc].min()
    iln = head_df[il_head_loc].max()
    xl0 = head_df[xl_head_loc].min()
    xln = head_df[xl_head_loc].max()

    # use diff to workout il/xl skips
    dil = np.max(head_df[il_head_loc].values[1:] - head_df[il_head_loc].values[:-1])
    dxl = np.max(head_df[xl_head_loc].values[1:] - head_df[xl_head_loc].values[:-1])

    # calculate absolute index of ilines and crosslines e.g. counting from zero for indexing into segy
    head_df["il_index"] = (head_df[il_head_loc] - il0) // dil
    head_df["xl_index"] = (head_df[xl_head_loc] - xl0) // dxl

    # get vertical sample ranges
    n0 = 0
    nsamp = head_bin["Samples"]
    ns0 = head_df.DelayRecordingTime.min()

    # first and last values
    ni = 1 + (iln - il0) // dil
    nx = 1 + (xln - xl0) // dxl

    # binary header translation
    ns = head_bin["Samples"]
    ds = head_bin["Interval"]
    msys = _SEGY_MEASUREMENT_SYSTEM[head_bin["MeasurementSystem"]]

    if zcrop is not None:
        zcrop = check_zcrop(zcrop, [0, ns])
        n0, ns = zcrop
        ns0 = ds * n0
        nsamp = ns - n0 + 1

    # create_empty_seisnc(ncfile, (ni, nx, nsamp))
    ds = create3d_dataset(
        (ni, nx, nsamp),
        first_sample=ns0,
        sample_rate=ds // 1000,
        first_iline=il0,
        iline_step=max(1, dil),
        first_xline=xl0,
        xline_step=max(1, dxl),
        vert_domain=vert_domain,
        vert_units=None,  # TODO: Fix this
    )

    # create_seismic_dataset(d1=ni, d2=nx, d3=nsamp)
    text = get_segy_texthead(segyfile, **segyio_kwargs)
    ds.attrs[AttrKeyField.text.value] = text

    try:
        cdpx = (
            head_df[[il_head_loc, xl_head_loc, x_head_loc]]
            .drop_duplicates(subset=[il_head_loc, xl_head_loc])
            .pivot(il_head_loc, xl_head_loc)
            .values
        )
        cdpy = (
            head_df[[il_head_loc, xl_head_loc, y_head_loc]]
            .drop_duplicates(subset=[il_head_loc, xl_head_loc])
            .pivot(il_head_loc, xl_head_loc)
            .values
        )
        ds[CoordKeyField.cdp_x.value] = (DimensionKeyField.cdp_3d.value, cdpx)
        ds[CoordKeyField.cdp_y.value] = (DimensionKeyField.cdp_3d.value, cdpy)

        ds = ds.set_coords([CoordKeyField.cdp_x.value, CoordKeyField.cdp_y.value])

    except ValueError as err:
        raise SegyLoadError(
            "SEGY headers could not be transformed through a pivot. Likely the inline and xline "
            "numbering is odd.",
            err,
        )

    if ncfile is not None and return_geometry == False:
        ds.seisio.to_netcdf(ncfile)
    elif return_geometry:
        # return geometry -> e.g. don't process segy traces
        return ds
    else:
        ncfile = ds

    segyio_kwargs.update(dict(ignore_geometry=True, iline=iline, xline=xline))

    # filter head_df
    if offset is None and not isinstance(ncfile, xr.Dataset):  # not prestack data
        _segy3dncdf(
            segyfile,
            ncfile,
            segyio_kwargs,
            n0,
            ns,
            head_df,
            il_head_loc,
            xl_head_loc,
            silent=silent,
        )

    # load into memory
    if offset is None and isinstance(ncfile, xr.Dataset):  # not prestack data
        ds = _segy3dxr(
            segyfile,
            ncfile,
            segyio_kwargs,
            n0,
            ns,
            head_df,
            il_head_loc,
            xl_head_loc,
            silent=silent,
        )
        return ds


def segy_loader(
    segyfile,
    ncfile=None,
    cmp=None,
    iline=None,
    xline=None,
    cdpx=None,
    cdpy=None,
    offset=None,
    vert_domain="TWT",
    data_type="AMP",
    ix_crop=None,
    cmp_crop=None,
    xy_crop=None,
    z_crop=None,
    return_geometry=False,
    silent=False,
    **segyio_kwargs,
):
    """Convert SEGY data to NetCDF4 File

    The output ncfile has the following structure
        Dimensions:
            d1 - CMP or Inline axis
            d2 - Xline axis
            d3 - The vertical axis
            d4 - Offset/Angle Axis
        Coordinates:
            iline - The inline numbering
            xline - The xline numbering
            cdp_x - Eastings
            cdp_y - Northings
            cmp - Trace Number for 2d
        Variables
            data - The data volume
        Attributes:
            TBC

    Args:
        segyfile (str): Input segy file path
        ncfile (str, optional): Output SEISNC file path. If none the loaded data will be
            returned in memory as an xarray.Dataset.
        iline (int): Inline byte location, usually 189
        xline (int): Cross-line byte location, usally 193
        vert (str): Vertical sampling domain. One of ['TWT', 'DEPTH']
        data_type (str): Data type ['AMP', 'VEL']
        cmp_crop (list, optional): List of minimum and maximum cmp values to output.
            Has the form '[min_cmp, max_cmp]'. Ignored for 3D data.
        ix_crop (list, optional): List of minimum and maximum inline and crossline to output.
            Has the form '[min_il, max_il, min_xl, max_xl]'. Ignored for 2D data.
        xy_crop (list, optional): List of minimum and maximum cdp_x and cdp_y to output.
            Has the form '[min_x, max_x, min_y, max_y]'. Ignored for 2D data.
        z_crop (list, optional): List of minimum and maximum vertical samples to output.
            Has the form '[min, max]'.
        return_geometry (bool, optional): If true returns an xarray.dataset which doesn't contain data but mirrors
            the input volume header information.
        silent (bool): Disable progress bar.
        **segyio_kwargs: Extra keyword arguments for segyio.open

    Returns:
        None: ncfile is specified.
        xarray.Dataset: ncfile keyword argument is zero or return_geometry is True
    """
    # Input sanity checks
    if cmp is not None and (iline is not None or xline is not None):
        raise ValueError("cmp cannot be defined with iline and xiline")

    if iline is None and xline is not None:
        raise ValueError("iline must be defined with xline")

    if xline is None and iline is not None:
        raise ValueError("xline must be defined with iline")

    if cdpx is None:
        cdpx = 181  # Assume standard location if misisng
    x_head_loc = _get_tf(cdpx)
    if cdpy is None:
        cdpy = 185  # Assume standard location if missing
    y_head_loc = _get_tf(cdpy)

    # Start by scraping the headers.
    head_df = segy_header_scrape(segyfile, silent=silent, **segyio_kwargs)
    head_bin = segy_bin_scrape(segyfile, **segyio_kwargs)

    # Scale Coordinates
    coord_scalar = head_df.SourceGroupScalar.median()
    coord_scalar_mult = np.power(abs(coord_scalar), np.sign(coord_scalar))
    head_df[x_head_loc] = head_df[x_head_loc].astype(float)
    head_df[y_head_loc] = head_df[y_head_loc].astype(float)
    head_df[x_head_loc] = head_df[x_head_loc] * coord_scalar_mult * 1.0
    head_df[y_head_loc] = head_df[y_head_loc] * coord_scalar_mult * 1.0

    # Cropping
    if cmp_crop and cmp is not None:  # 2d cdp cropping
        cmp_head_loc = _get_tf(cmp)
        crop_min, crop_max = check_crop(
            cmp_crop, [head_df[cmp_head_loc].min(), head_df[cmp_head_loc].max()]
        )

        head_df = head_df.query(
            "@cmp_head_loc >= @crop_min & @cmp_head_loc <= @crop_max"
        )

    if ix_crop is not None and cmp is None:  # 3d inline/xline cropping
        il_head_loc = _get_tf(iline)
        xl_head_loc = _get_tf(xline)
        il_min, il_max, xl_min, xl_max = check_crop(
            ix_crop,
            [
                head_df[il_head_loc].min(),
                head_df[il_head_loc].max(),
                head_df[xl_head_loc].min(),
                head_df[xl_head_loc].max(),
            ],
        )
        query = f"@il_head_loc >= @il_min & @il_head_loc <= @il_max & @xl_head_loc >= @xl_min and @xl_head_loc <= @xl_max"
        head_df = head_df.query(query)

    # TODO: -> Could implement some cropping with a polygon here
    if xy_crop is not None and cmp is None:
        x_min, x_max, y_min, y_max = check_crop(
            xy_crop,
            [
                head_df[x_head_loc].min(),
                head_df[x_head_loc].max(),
                head_df[y_head_loc].min(),
                head_df[y_head_loc].max(),
            ],
        )
        query = "@x_head_loc >= @x_min & x_head_loc <= @x_max & @y_head_loc >= @y_min and @y_head_loc <= @y_max"
        head_df = head_df.query(query)

    if cmp is None:  # 3d data
        ds = _3dsegy_loader(
            segyfile,
            head_df,
            head_bin,
            zcrop=z_crop,
            ncfile=ncfile,
            iline=iline,
            xline=xline,
            cdpx=cdpx,
            cdpy=cdpy,
            vert_domain=vert_domain,
            data_type=data_type,
            return_geometry=return_geometry,
            silent=silent,
            **segyio_kwargs,
        )

    if ncfile is None:
        return ds

    # if offset is not None and CMP == False:
    #     pass
    #     # prestack loader

    # if CMP == True and offset is None:
    #     pass
    #     # post stack 2d data

    # if CMP == True and offset is not None:
    #     pass
    #     # pre-stack 2d data


def ncdf2segy(
    ncfile, segyfile, CMP=False, iline=189, xline=193, il_chunks=10, silent=False
):
    """Convert etlpy siesnc format (NetCDF4) to SEGY.

    Args:
        ncfile (string): The input SEISNC file
        segyfile (string): The output SEGY file
        CMP (bool, optional): The data is 2D. Defaults to False.
        iline (int, optional): Inline byte location. Defaults to 189.
        xline (int, optional): Crossline byte location. Defaults to 193.
        il_chunks (int, optional): The size of data to work on - if you have memory
            limitations. Defaults to 10.
        silent (bool, optional): Turn off progress reporting. Defaults to False.
    """

    with xr.open_dataset(ncfile, chunks={"i": il_chunks}) as seisnc:
        ni, nj, nk = seisnc.dims["i"], seisnc.dims["j"], seisnc.dims["k"]
        z0 = int(seisnc.vert.values[0])
        msys = _ISEGY_MEASUREMENT_SYSTEM[seisnc.measurement_system]
        spec = segyio.spec()
        # to create a file from nothing, we need to tell segyio about the structure of
        # the file, i.e. its inline numbers, crossline numbers, etc. You can also add
        # more structural information, but offsets etc. have sensible defautls. This is
        # the absolute minimal specification for a N-by-M volume
        spec.sorting = 1
        spec.format = 1
        spec.iline = iline
        spec.xline = xline
        spec.samples = range(nk)
        spec.ilines = range(ni)
        spec.xlines = range(nj)

        xl_val = seisnc["xline"].values
        il_val = seisnc["iline"].values

        il_bags = _bag_slices(seisnc["iline"].values, n=il_chunks)

        with segyio.create(segyfile, spec) as segyf:
            for ilb in tqdm(il_bags, desc="WRITING CHUNK", disable=silent):
                ilbl = range(ilb.start, ilb.stop, ilb.step)
                data = seisnc.isel(i=ilbl)
                for i, il in enumerate(ilbl):
                    il0, iln = il * nj, (il + 1) * nj
                    segyf.header[il0:iln] = [
                        {
                            segyio.su.offset: 1,
                            iline: il_val[il],
                            xline: xln,
                            segyio.su.cdpx: cdpx,
                            segyio.su.cdpy: cdpy,
                            segyio.su.ns: nk,
                            segyio.su.delrt: z0,
                        }
                        for xln, cdpx, cdpy in zip(
                            xl_val, data.CDP_X.values[i, :], data.CDP_Y.values[i, :]
                        )
                    ]
                    segyf.trace[il * nj : (il + 1) * nj] = data.data[i, :, :].values
            segyf.bin.update(
                tsort=segyio.TraceSortingFormat.INLINE_SORTING,
                hdt=int(seisnc.ds * 1000),
                hns=nk,
                mfeet=msys,
                jobid=1,
                lino=1,
                reno=1,
                ntrpr=ni * nj,
                nart=ni * nj,
                fold=1,
            )
