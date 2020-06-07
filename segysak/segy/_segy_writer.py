
import importlib
import xarray as xr
import segyio

try:
    has_ipywidgets = importlib.find_loader("ipywidgets") is not None
    if has_ipywidgets:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except ModuleNotFoundError:
    from tqdm import tqdm


from ._segy_globals import _ISEGY_MEASUREMENT_SYSTEM


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
