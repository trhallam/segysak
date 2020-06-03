# pylint: disable=invalid-name
"""Functions to interact with segy data
"""

from warnings import warn

from collections import defaultdict

import segyio
import netCDF4
import h5netcdf
import numpy as np
import pandas as pd
import xarray as xr

from tqdm.autonotebook import tqdm

from ._keyfield import CoordKeyField, AttrKeyField, VariableKeyField, DimensionKeyField
from ._seismic_dataset import create_seismic_dataset, create3d_dataset
from segysak.seisnc import create_empty_seisnc, set_seisnc_dims
from segysak.tools import check_crop, check_zcrop, _get_userid, _get_datetime

_SEGY_MEASUREMENT_SYSTEM = defaultdict(lambda: 0)
_SEGY_MEASUREMENT_SYSTEM[1] = "m"
_SEGY_MEASUREMENT_SYSTEM[2] = "ft"
_ISEGY_MEASUREMENT_SYSTEM = defaultdict(lambda: 0, m=1, ft=2)


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


def get_segy_texthead(segyfile, ext_headers=False):
    """Return the ebcidc

    Args:
        segyfile (str): Segy File Path
        ext_headers (bool): Return EBCIDC and extended headers in list.
            Defaults to False

    Returns:
        str: Returns the EBCIDC text as a formatted paragraph.
    """
    with segyio.open(segyfile, "r", ignore_geometry=True) as segyf:
        text = segyf.text[0].decode("ascii", "replace")
        text = text.replace("�Cro", "    ")  # petrel weird export
        text = segyio.tools.wrap(text)
        if segyf.ext_headers and ext_headers:
            text2 = segyf.text[1].decode("ascii", "replace")
            return [text, text2]
        else:
            return text


def put_segy_texthead(segyfile, ebcidc, ext_headers=False):

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

    with segyio.open(segyfile, "r+", ignore_geometry=True) as segyf:
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


def segy_header_scan(segyfile, max_traces_scan=1000, silent=False):
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
    with segyio.open(segyfile, "r", ignore_geometry=True) as segyf:
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


def segy_bin_scrape(segyfile):
    """Scrape binary header

    Args:
        segyfile (str): SEGY File path

    Returns:
        dict: Binary header
    """
    bk = _active_binfield_segyio()
    with segyio.open(segyfile, "r", ignore_geometry=True) as segyf:
        return {key: segyf.bin[item] for key, item in bk.items()}


def segy_header_scrape(segyfile, silent=False):
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
    with segyio.open(segyfile, "r", ignore_geometry=True) as segyf:
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


def segy2ncdf(
    segyfile,
    ncfile,
    CMP=False,
    iline=189,
    xline=193,
    cdpx=181,
    cdpy=185,
    vert="TWT",
    units="AMP",
    crop=None,
    zcrop=None,
    silent=False,
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
    head_df = segy_header_scrape(segyfile)
    head_bin = segy_bin_scrape(segyfile)

    # get names of columns where stuff we want is
    il_head_loc = str(segyio.TraceField(iline))
    xl_head_loc = str(segyio.TraceField(xline))
    x_head_loc = str(segyio.TraceField(cdpx))
    y_head_loc = str(segyio.TraceField(cdpy))

    # calculate vert, inline and crossline ranges/meshgrids
    il0 = head_df[il_head_loc].min()
    iln = head_df[il_head_loc].max()
    xl0 = head_df[xl_head_loc].min()
    xln = head_df[xl_head_loc].max()
    n0 = 0
    # nsamp = head_df.TRACE_SAMPLE_COUNT.min()

    # double check because sample count might not be in trace headers
    # if nsamp == 0 and nsamp < head_bin["Samples"]:
    nsamp = head_bin["Samples"]

    ns0 = head_df.DelayRecordingTime.min()
    coord_scalar = head_df.SourceGroupScalar.median()
    coord_scalar_mult = np.power(abs(coord_scalar), np.sign(coord_scalar))
    dil = np.max(head_df[il_head_loc].values[1:] - head_df[il_head_loc].values[:-1])
    dxl = np.max(head_df[xl_head_loc].values[1:] - head_df[xl_head_loc].values[:-1])

    if crop is not None:
        crop = check_crop(crop, [il0, iln, xl0, xln])
        il0, iln, xl0, xln = crop

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
        iline_step=dil,
        first_xline=xl0,
        xline_step=dxl,
        vert_domain=vert,
        vert_units=None,  # TODO: Fix this
    )

    # create_seismic_dataset(d1=ni, d2=nx, d3=nsamp)
    text = get_segy_texthead(segyfile)
    ds.attrs[AttrKeyField.text.value] = text
    ds.seisio.to_netcdf(ncfile)

    with segyio.open(
        segyfile, "r", ignore_geometry=True, iline=iline, xline=xline
    ) as segyf, h5netcdf.File(ncfile, "a") as seisnc:

        # assign CDPXY
        query = f"{il_head_loc} >= @il0 & {il_head_loc} <= @iln & {xl_head_loc} >= @xl0 and {xl_head_loc} <= @xln"
        cdpx = (
            head_df.query(query)[[il_head_loc, xl_head_loc, x_head_loc]]
            .pivot(il_head_loc, xl_head_loc)
            .values
        )
        cdpy = (
            head_df.query(query)[[il_head_loc, xl_head_loc, y_head_loc]]
            .pivot(il_head_loc, xl_head_loc)
            .values
        )

        seisnc_cdpx = seisnc.create_variable(
            CoordKeyField.cdp_x.value,
            DimensionKeyField.cdp_3d.value,
            float,
            data=cdpx * coord_scalar_mult,
        )
        seisnc_cdpy = seisnc.create_variable(
            CoordKeyField.cdp_y.value,
            DimensionKeyField.cdp_3d.value,
            float,
            data=cdpy * coord_scalar_mult,
        )

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

        head_df["il_index"] = (head_df[il_head_loc] - il0) // dil
        head_df["xl_index"] = (head_df[xl_head_loc] - xl0) // dxl

        pb = tqdm(total=segyf.tracecount, desc="Converting SEGY", disable=silent)

        for contig, grp in head_df.groupby(contig_dir):
            for trc, val in grp.iterrows():
                seisnc_data[val.il_index, val.xl_index, :] = segyf.trace[trc][
                    n0 : ns + 1
                ]
                pb.update()
        pb.close()


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
