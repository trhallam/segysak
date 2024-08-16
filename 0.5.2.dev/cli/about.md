# SEGY-SAK - Command Line Interface

SEGY-SAK offers an interface on the command line to inspect SEG-Y files and to
convert between data formats. These are convenience wrappers around core `segysak`
functions to allow fast an easy interrogation of/interaction wtih SEG-Y files.

For example, it is possible to scrape the text header from a SEG-Y file

```shell
segysak ebcidc volve10r12-full-twt-sub3d.sgy
C 1 SEGY OUTPUT FROM Petrel 2017.2 Saturday, June 06 2020 10:15:00
C 2 Name: ST10010ZDC12-PZ-PSDM-KIRCH-FULL-T.MIG_FIN.POST_STACK.3D.JS-017534
ÝCroC 3
C 4 First inline: 10090  Last inline: 10150
C 5 First xline:  2150   Last xline:  2351
...
C37
C38
C39
C40 END EBCDIC
```

Standard linux redirects can be used to output the header to a file or to other
command line tools.

```shell
segysak ebcidc volve10r12-full-twt-sub3d.sgy > header.txt
segysak ebcidc volve10r12-full-twt-sub3d.sgy | less
```
From `segysak>=0.5` file conversion is conducted with lazy loading, this should
allow very large SEG-Y files to be converted to more performant file formats such
as ZGY[^1] and NetCDF4.



```shell
segysak convert 
```

A full list of sub-commands is available in the [CLI Reference](command-line-ref.md)
or individually for each sub-command individually using the `--help` flag.

```shell
segysak scan --help
```

[^1]: ZGY file operations require `pyzgy>=0.10`.
