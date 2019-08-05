# pylint: disable=invalid-name
"""Functions to interact with segy data
"""

import os
import datetime

from collections import defaultdict

import segyio
import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

from tqdm.autonotebook import tqdm
from segysak.seisnc import create_empty_seisnc, set_seisnc_dims

_SEGY_MEASUREMENT_SYSTEM = defaultdict(lambda: 0)
_SEGY_MEASUREMENT_SYSTEM[1] = 'm'
_SEGY_MEASUREMENT_SYSTEM[2] = 'ft'
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
        if (i+1)%n == 0:
            bag.append(slice(prev, i+1, 1))
            prev = i+1
    if prev != i+1:
        bag.append(slice(prev, i+1, 1))
    return bag

def segy_texthead(segyfile, ext_headers=False):
    """Return the ebcidc

    Args:
        segyfile (str): Segy File Path
        ext_headers (bool): Return EBCIDC and extended headers in list.
            Defaults to False

    Returns:
        str: Returns the EBCIDC text as a formatted paragraph.
    """
    with segyio.open(segyfile, "r", ignore_geometry=True) as segyf:
        text = segyf.text[0].decode('ascii', 'replace')
        text = text.replace('�Cro', "    ") # petrel weird export
        text = segyio.tools.wrap(text)
        if segyf.ext_headers and ext_headers:
            text2 = segyf.text[1].decode('ascii', 'replace')
            return [text, text2]
        else:
            return text

def _get_datetime():
    """Return current date and time as formatted strings

    Returns:
        str, str: Date "YYYY-MM-DD" Time "HH:MM:SS"
    """
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")

def _clean_texthead(text_dict):
    """Reduce texthead dictionary to 80 characters

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
            if len(line_str) > 80:
                line_str = line_str[0:80]
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
        {1: 'etlpy SEGY Output',
        2: 'Data created by: username ',
        3: '',
        4: 'DATA FORMAT: SEG-Y;  DATE: 2019-06-09 15:14:00',
        5: 'DATA DESCRIPTION: SEGY format data output from etlpy',
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
    user = os.getlogin()
    today, time = _get_datetime()
    text_dict = {
        #      123456789012345678901234567890123456789012345678901234567890123456
        1   : "segysak SEGY Output",
        2   :f"Data created by: {user} ",
        4   :f"DATA FORMAT: SEG-Y;  DATE: {today} {time}",
        5   : "DATA DESCRIPTION: SEGY format data output",
        6   : "",
        35  : "*** BYTE LOCATION OF KEY HEADERS ***",
        36  : "CMP UTM-X 181-184, ALL COORDS X100, CMP UTM-Y 185-188",
        37  : "INLINE 189-193, XLINE 194-198, ",
        40  : "END TEXTUAL HEADER"
        }
    if override is not None:
        for key, line in override.items():
            text_dict[key] = line
    return _clean_texthead(text_dict)


def _active_tracefield_segyio():
    header_keys = segyio.tracefield.keys.copy()
    # removed unused byte locations
    _ = header_keys.pop('UnassignedInt1')
    _ = header_keys.pop('UnassignedInt2')
    return header_keys

def _active_binfield_segyio():
    bin_keys = segyio.binfield.keys.copy()
    _ = bin_keys.pop('Unassigned1')
    _ = bin_keys.pop('Unassigned2')
    return bin_keys

def segy_header_scan(segyfile, max_traces_scan=1000):
    """Perform a scan of the segy file headers and return ranges.

    Args:
        segyfile (string): Segy File Path
        max_traces_scan (int, optional): Number of traces to scan.
            For all set to 0 or 'all'. Defaults to 1000.

    Returns:
        dict: Contains keys from segyio.tracefield.keys with scanned values in
        a list [byte location, min, max]
    """
    if not isinstance(max_traces_scan, int):
        raise Exception('max_traces_scan must be int')
    elif max_traces_scan == 0 or max_traces_scan == 'all':
        max_traces_scan = None

    hi = 0
    with segyio.open(segyfile, 'r', ignore_geometry=True) as segyf:
        header_keys = _active_tracefield_segyio()
        lh = len(header_keys.keys())
        hmin = np.full(lh, np.nan)
        hmax = np.full(lh, np.nan)
        for hi, h in enumerate(tqdm(segyf.header, desc='Scanning Headers', total=max_traces_scan)):
            val = np.array(list(h.values()))
            hmin = np.nanmin(np.vstack((hmin, val)), axis=0)
            hmax = np.nanmax(np.vstack((hmax, val)), axis=0)
            if max_traces_scan is not None and hi >= max_traces_scan:
                break # scan to max_traces_scan

    for i, (key, item) in enumerate(header_keys.items()):
        header_keys[key] = [item, hmin[i], hmax[i]]

    return header_keys, hi+1

def segy_bin_scrape(segyfile):
    """Scrape binary header

    Args:
        segyfile (str): SEGY File path

    Returns:
        dict: Binary header
    """
    bk = _active_binfield_segyio()
    with segyio.open(segyfile, 'r', ignore_geometry=True) as segyf:
        return {key:segyf.bin[item] for key, item in bk.items()}

def segy_header_scrape(segyfile, silent=False):
    """Scape all data from segy trace headers

    Args:
        segyfile (str): SEGY File path

    Returns:
        (pandas.DataFrame): Header information
    """
    header_keys = _active_tracefield_segyio()
    columns = header_keys.keys()
    lc = len(columns)
    with segyio.open(segyfile, 'r', ignore_geometry=True) as segyf:
        ntraces = segyf.tracecount
        head_df = pd.DataFrame(data=np.full((ntraces, lc), np.nan, dtype=int), columns=columns)
        pb = tqdm(total=segyf.tracecount, desc="Scraping Headers", disable=silent)
        for hi, h in enumerate(segyf.header):
            head_df.iloc[hi, :] = np.array(list(h.values()))
            pb.update()
        pb.close()
        head_df.replace(to_replace=-2147483648, value=np.nan, inplace=True)
    return head_df

def segy2ncdf(segyfile, ncfile, CMP=False, iline=189, xline=193, cdpx=181, cdpy=185,
              vert='TWT', units='AMP', silent=False):
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
        segyfile ([type]): [description]
        ncfile
        iline
        xline
        vert
        units
    """
    head_df = segy_header_scrape(segyfile)
    head_bin = segy_bin_scrape(segyfile)
    # calculate vert, inline and crossline ranges/meshgrids
    il0 = head_df['INLINE_3D'].min()
    iln = head_df["INLINE_3D"].max()
    xl0 = head_df['CROSSLINE_3D'].min()
    xln = head_df['CROSSLINE_3D'].max()
    ns0 = head_df.TRACE_SAMPLE_COUNT.min()

    dil = np.max(
        head_df['INLINE_3D'].values[1:] - head_df['INLINE_3D'].values[:-1]
    )
    dxl = np.max(
        head_df['CROSSLINE_3D'].values[1:] - head_df['CROSSLINE_3D'].values[:-1]
    )
    # first and last values
    ni = 1 + (iln - il0)//dil
    nx = 1 + (xln - xl0)//dxl

    # binary header translation
    ns = head_bin['Samples']
    ds = head_bin['Interval']
    msys = _SEGY_MEASUREMENT_SYSTEM[head_bin['MeasurementSystem']]

    create_empty_seisnc(ncfile, (ni, nx, ns))
    set_seisnc_dims(ncfile, first_sample=ns0//1000, sample_rate=ds//1000,
                    first_iline=il0, iline_step=dil,
                    first_xline=xl0, xline_step=dxl, vert_domain=vert,
                    measurement_system=msys)

    text = segy_texthead(segyfile)

    with segyio.open(segyfile, 'r', ignore_geometry=True, iline=iline, xline=xline) as segyf, \
      netCDF4.Dataset(ncfile, "a", format="NETCDF4") as seisnc:
        seisnc.text = text

        #assign CDPXY
        seisnc['CDP_X'][:, :] = head_df['CDP_X'].values.reshape((ni, nx))
        seisnc['CDP_Y'][:, :] = head_df['CDP_Y'].values.reshape((ni, nx))

        segyf.mmap()
        # load trace
        temp_line = np.full((nx, ns), np.nan, float)
        cur_iline = head_df['INLINE_3D'][0]
        pb = tqdm(total=segyf.tracecount, desc="Converting SEGY", disable=silent)
        for n, trc in enumerate(segyf.trace):
            cur_xline = (head_df['CROSSLINE_3D'][n] - xl0)//dxl
            temp_line[cur_xline, :] = trc
            if head_df['INLINE_3D'][n] > cur_iline:
                cur_iline = head_df['INLINE_3D'][n]
                seisnc['data'][(cur_iline-il0)/dil, :, :] = temp_line
                temp_line[:, :] = np.nan
            pb.update()
        pb.close()

def ncdf2segy(ncfile, segyfile, CMP=False, iline=189, xline=193, xl_chunks=10, silent=False):
    """Convert etlpy siesnc format (NetCDF4) to SEGY.

    Args:
        ncfile (string): The input SEISNC file
        segyfile (string): The output SEGY file
        CMP (bool, optional): The data is 2D. Defaults to False.
        iline (int, optional): Inline byte location. Defaults to 189.
        xline (int, optional): Crossline byte location. Defaults to 193.
        xl_chunks (int, optional): The size of data to work on - if you have memory
            limitations. Defaults to 10.
        silent (bool, optional): Turn off progress reporting. Defaults to False.
    """

    with xr.open_dataset(ncfile, chunks={'xl':xl_chunks}) as seisnc:
        ni, nj, nk = seisnc.dims['il'], seisnc.dims['xl'], seisnc.dims['v']
        msys = _ISEGY_MEASUREMENT_SYSTEM[seisnc.measurement_system]
        spec = segyio.spec()
        # to create a file from nothing, we need to tell segyio about the structure of
        # the file, i.e. its inline numbers, crossline numbers, etc. You can also add
        # more structural information, but offsets etc. have sensible defautls. This is
        # the absolute minimal specification for a N-by-M volume
        spec.sorting = 2
        spec.format = 1
        spec.iline = iline
        spec.xline = xline
        spec.samples = range(nk)
        spec.ilines = range(ni)
        spec.xlines = range(nj)

        xl_val = seisnc.coords['xline'].values
        il_val = seisnc.coords['iline'].values

        xl_bags = _bag_slices(seisnc.coords['xline'].values, n=xl_chunks)

        with segyio.create(segyfile, spec) as segyf:
            for xlb in tqdm(xl_bags, desc="WRITING CHUNK", disable=silent):
                xlbl = range(xlb.start, xlb.stop, xlb.step)
                data = seisnc.isel(xl=xlbl)
                for x, xl in enumerate(xlbl):
                    xl0, xln = xl*ni, (xl+1)*ni
                    segyf.header[xl0:xln] = [
                        {
                            segyio.su.offset: 1,
                            iline: iln,
                            xline: xl_val[xl],
                            segyio.su.cdpx: cdpx,
                            segyio.su.cdpy: cdpy
                        }
                        for iln, cdpx, cdpy in zip(il_val,
                                                   data.CDP_X.values[:, x],
                                                   data.CDP_Y.values[:, x]
                                                   )
                        ]
                    segyf.trace[xl*ni:(xl+1)*ni] = data.data[:, x, :].values
            segyf.bin.update(
                tsort=segyio.TraceSortingFormat.INLINE_SORTING,
                hdt=int(seisnc.ds*1000),
                mfeet=msys,
                jobid=1,
                lino=1,
                reno=1,
                ntrpr=ni*nj,
                nart=ni*nj,
                fold=1
            )
