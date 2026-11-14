"""Fetch latest NOAA GFS 10 m wind (1 deg) and write Int16 LE binary + JSON metadata."""
import datetime as dt, json, os, struct, sys, urllib.request
import numpy as np
import eccodes

NX, NY = 360, 181
OUT = os.path.dirname(os.path.abspath(__file__))
URL = ("https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_1p00.pl?dir=%2Fgfs.{d}%2F{h}%2Fatmos"
       "&file=gfs.t{h}z.pgrb2.1p00.f000&var_UGRD=on&var_VGRD=on&lev_10_m_above_ground=on")


def download():
    now = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    cyc = now.replace(hour=now.hour - now.hour % 6)
    for _ in range(5):
        url = URL.format(d=cyc.strftime("%Y%m%d"), h=cyc.strftime("%H"))
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "yzr-net-wind/1.0"})
            data = urllib.request.urlopen(req, timeout=60).read()
            if data[:4] == b"GRIB" and len(data) > 1000:
                print("using cycle", cyc.isoformat(), len(data), "bytes")
                return cyc, data
        except Exception as e:
            print("cycle", cyc.isoformat(), "unavailable:", e)
        cyc -= dt.timedelta(hours=6)
    sys.exit("no GFS cycle available")


def decode(path):
    fields = {}
    with open(path, "rb") as f:
        while True:
            g = eccodes.codes_grib_new_from_file(f)
            if g is None:
                break
            name = eccodes.codes_get(g, "shortName")
            nx, ny = eccodes.codes_get(g, "Ni"), eccodes.codes_get(g, "Nj")
            if (nx, ny) != (NX, NY):
                sys.exit(f"bad grid {nx}x{ny}")
            lat1 = eccodes.codes_get(g, "latitudeOfFirstGridPointInDegrees")
            lon1 = eccodes.codes_get(g, "longitudeOfFirstGridPointInDegrees")
            if lat1 != 90 or lon1 != 0:
                sys.exit(f"unexpected origin {lat1},{lon1}")
            if eccodes.codes_get(g, "jScansPositively") != 0:
                sys.exit("unexpected scan order")
            v = eccodes.codes_get_values(g).reshape(NY, NX)
            fields[name] = v
            eccodes.codes_release(g)
    for k in ("10u", "10v"):
        if k not in fields:
            sys.exit(f"missing {k}; got {list(fields)}")
    return fields["10u"], fields["10v"]


def main():
    cyc, data = download()
    tmp = os.path.join(OUT, "_tmp.grib2")
    with open(tmp, "wb") as f:
        f.write(data)
    u, v = decode(tmp)
    os.remove(tmp)
    arr = np.concatenate([u.ravel(), v.ravel()])
    if not np.isfinite(arr).all() or np.abs(arr).max() > 120:
        sys.exit("NaN or out-of-range wind values")
    q = np.round(arr * 100).astype("<i2")
    if q.size != 2 * NX * NY:
        sys.exit("wrong size")
    with open(os.path.join(OUT, "gfs-wind-1deg.bin"), "wb") as f:
        f.write(q.tobytes())
    meta = {
        "source": "NOAA GFS 1.00 deg via NOMADS grib filter (UGRD/VGRD 10 m above ground)",
        "refTime": cyc.strftime("%Y-%m-%dT%H:00:00Z"), "forecastHour": 0,
        "nx": NX, "ny": NY, "la1": 90, "lo1": 0, "dx": 1, "dy": 1,
        "scale": 0.01, "order": "u,v",
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(os.path.join(OUT, "gfs-wind-1deg.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("ok", meta["refTime"], "mean|wind|", float(np.hypot(u, v).mean()))


main()
