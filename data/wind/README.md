# Wind (NOAA GFS 10 m, 1 degree)

`fetch_wind.py` downloads the newest GFS f000 cycle (UGRD/VGRD 10 m) from the NOMADS grib filter and writes:

- `gfs-wind-1deg.bin`: Int16 little-endian, u then v, each 181 rows (lat 90 to -90) x 360 cols (lon 0 to 359), value = m/s x 100. 260,640 bytes.
- `gfs-wind-1deg.json`: metadata (refTime, grid, scale).

The workflow `.github/workflows/wind.yml` runs at 04:30, 10:30, 16:30, 22:30 UTC and force-pushes the two files to the orphan branch `data`:

- https://raw.githubusercontent.com/LykosTR/yzr-net/data/wind/gfs-wind-1deg.bin
- https://raw.githubusercontent.com/LykosTR/yzr-net/data/wind/gfs-wind-1deg.json

Data: NOAA/NCEP GFS (public domain).
