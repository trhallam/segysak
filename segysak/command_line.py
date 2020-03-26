
"""Command line script for interacting with SEGY data.
"""

import os
import sys
import time
import logging
import argparse
import pathlib

import segysak
from segysak.version import version
from segysak.segy import segy2ncdf, ncdf2segy, segy_header_scan, get_segy_texthead

# configuration setup
NAME = 'segysak'
LOGGER = logging.getLogger(NAME)

class segysakArgsParser():
    """Passable arguments to segysak

    """
    console_description = """
    The SEGY Swiss Army Knife (segysak) is a tool for managing segy data.
    It can:
      read and dump ebcidc headers
      scan trace headers
      convert segy to SEISNC
      convert SEISNC to segy
    """

    def __init__(self):
        """SEGY-SAK ArgParse Constructor"""
        self.app_name = NAME
        self.version = version
        self.parser = argparse.ArgumentParser(description = self.console_description,
                                     formatter_class = argparse.RawDescriptionHelpFormatter)
        self.parser.add_argument("-V", help=f"print application version name", action='store_true', default=False)
        #self.parser.add_argument("-S", "--silent", help='silence info and warning messages, errors will still be raised',
        #    action='store_true', default=False)
        # logging will output to
        self.parser.add_argument("-L", help=f'output logging to file, if none specified will use {NAME}_date_time.log',
            nargs='?', const='DEFAULT', default=None)
        #self.parser.add_argument("--debugging", help='activate additional debugging messages', action='store_true', default=False)
        self.parser.add_argument("files", metavar='file', type=str, nargs='+', help="Input file location and name")
        self.parser.add_argument("-e,", "--ebcidc", help='Print SEGY EBCIDC header',
            action='store_true', default=False)
        self.parser.add_argument("--scan", help="Scan trace headers and print value ranges.",
            nargs='?', const=1000, default=False, type=int)
        self.output_format_group =  self.parser.add_mutually_exclusive_group()
        self.output_format_group.add_argument("-nc", "--netCDF",
            help="Convert SEGY to netCDF File (SEISNC)",
            nargs='?', const=None, default=False, type=str)
        self.output_format_group.add_argument("-sgy", "--SEGY",
            help="Convert SEISNC to SEGY File",
            nargs='?', const=None, default=False, type=str)
        self.parser.add_argument('--iline', help="Inline byte location",
            default=189, type=int)
        self.parser.add_argument('--xline', help='Crossline byte location',
            default=193, type=int)
        self.parser.add_argument('--crop', help='Crop the input volume using a list [minil, maxil, minxl, maxxl]',
            default=None, nargs='+')

    def parse_args(self):
        args = self.parser.parse_args()
        # process default args
        if args.V:
            print(self.app_name, self.version)
            raise SystemExit

        if args.L == 'DEFAULT':
             t = time.localtime(time.time())
             args.L = f"{self.app_name}_{t.tm_year}{t.tm_mon:02d}{t.tm_mday:02d}_{t.tm_hour:02d}{t.tm_min:02d}{t.tm_sec:02d}.log"

        if args.crop:
            args.crop = [int(i) for i in args.crop]

        return args

def check_file(input_files):

    if input_files is None:
        LOGGER.error('Require input file/s')
        raise SystemExit

    checked_files = list()

    expanded_input = input_files.copy()

    for ifile in input_files:
        if '*' in ifile:
            ifile_path = pathlib.Path(ifile)
            parent = ifile_path.absolute().parent
            expanded_input += list(parent.glob(ifile_path.name))
            expanded_input.remove(ifile)

    for ifile in expanded_input:
        ifile = pathlib.Path(ifile)
        if ifile.exists():
            checked_files.append(ifile)
        else:
            LOGGER.error('Cannot find input {segyfile}')
            raise SystemExit

    return checked_files

def main():

    # how to track down warnings
    #import warnings
    #warnings.filterwarnings('error', category=UnicodeWarning)
    #warnings.filterwarnings('error', category=DeprecationWarning, module='numpy')

    #parse args
    sys_parser = segysakArgsParser()
    args = sys_parser.parse_args()

    # gaffe
    print(f'SEGY-SAK - v{version}')

    # initialise logging
    LOGGER.info(f'segysak v{version}')

    # check inputs
    checked_files = check_file(args.files)

    # Generate or Load Configuration File

    for input_file in checked_files:
        print(input_file.name)
        # Print EBCIDC header
        if args.ebcidc:
            try:
                print(get_segy_texthead(input_file))
            except IOError:
                LOGGER.error("Input SEGY file was not found - check name and path")

        if args.scan > 0:
            hscan, nscan = segy_header_scan(input_file, max_traces_scan=args.scan)
            width = 10
            print(f'Traces scanned: {nscan}')
            print("{:>40s} {:>8s} {:>10s} {:>10s}".format('Item', 'Byte Loc', 'Min', 'Max'))
            for key, item in hscan.items():
                print("{:>40s} {:>8d} {:>10.0f} {:>10.0f}".format(key, item[0], item[1], item[2]))

        iline = args.iline
        xline = args.xline

        if args.netCDF is None or args.netCDF is not False:
            if args.netCDF is None:
                outfile = input_file.stem + '.SEISNC'
            else:
                outfile = args.netCDF
            segy2ncdf(input_file, outfile, iline=iline, xline=xline, crop=args.crop)
            LOGGER.info(f"NetCDF output written to {outfile}")

        if args.SEGY is None or args.SEGY is not False:
            if args.SEGY is None:
                outfile = input_file.stem + '.segy'
            else:
                outfile = args.SEGY
            ncdf2segy(input_file, outfile, iline=iline, xline=xline)#, crop=args.crop)
            LOGGER.info(f"SEGY output written to {outfile}")

if __name__ == "__main__":
    main()