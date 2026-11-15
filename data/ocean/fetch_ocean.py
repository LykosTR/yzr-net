"""Fetch the newest Copernicus Marine global surface current (1/3 degree overview store of the public GLOBAL_ANALYSISFORECAST_PHY_001_024
ARCO zarr on CloudFerro) and write it for the browser: cmems-uv.json (metadata) + cmems-uv.bin.gz (Int16 LE, u then v, ascending latitude,
value = m/s x 1000, -32768 = land / no data)."""
import datetime as dt, gzip, json, math, os, sys, time, urllib.request
import numpy as np
from numcodecs import blosc

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
BASE = ("https://s3.waw3-1.cloudferro.com/mdl-arco-time-014/arco/GLOBAL_ANALYSISFORECAST_PHY_001_024/"
        "cmems_mod_glo_phy_anfc_0.083deg_PT1H-m_202406/downsampled4.zarr/")
EPOCH = dt.datetime(2022, 6, 1, tzinfo=dt.timezone.utc)
NY, NX = 511, 1080


def get(path):
    for attempt in range(3):
        try:
            req = urllib.request.Request(BASE + path, headers={"User-Agent": "yzr-net-ocean/1.0"})
            return urllib.request.urlopen(req, timeout=90).read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            time.sleep(3)
        except Exception as e:
            print("retry", attempt, e)
            time.sleep(3)
    return None


def f32(buf, n):
    a = np.frombuffer(blosc.decompress(buf), dtype="<f4")
    if a.size < n:
        sys.exit("short chunk %d < %d" % (a.size, n))
    a = a[:n].astype(np.float32).copy()
    a[~((a > -1e30) & (a < 1e30))] = np.nan
    return a


def main():
    t0 = int((dt.datetime.now(dt.timezone.utc) - EPOCH).total_seconds() // 3600) - 3
    t = u = v = None
    for k in range(49):
        tt = t0 - k
        bu, bv = get("uo/%d.0.0.0" % tt), get("vo/%d.0.0.0" % tt)
        if bu and bv:
            t, u, v = tt, f32(bu, NY * NX).reshape(NY, NX), f32(bv, NY * NX).reshape(NY, NX)
            break
    if t is None:
        sys.exit("no model hour found in the last 48 h")
    la = f32(get("latitude/0"), 2041)[:NY]
    lo = f32(get("longitude/0"), 4320)[:NX]
    flip = la[0] > la[-1]
    if flip:
        u, v, la = u[::-1], v[::-1], la[::-1]
    lat0, dlat = float(la[0]), float((la[-1] - la[0]) / (NY - 1))
    lon0, dlon = float(lo[0]), float((lo[-1] - lo[0]) / (NX - 1))
    for a in (u, v):
        a[(np.abs(a) >= 20)] = np.nan
    ok = np.isfinite(u) & np.isfinite(v)
    if ok.sum() < NY * NX * 0.3:
        sys.exit("only %d ocean cells, refusing to publish" % ok.sum())
    raw = np.concatenate([np.where(ok, np.clip(np.rint(u * 1000), -32767, 32767), -32768).astype("<i2").ravel(),
                          np.where(ok, np.clip(np.rint(v * 1000), -32767, 32767), -32768).astype("<i2").ravel()]).tobytes()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "cmems-uv.bin.gz"), "wb") as f:
        f.write(gzip.compress(raw, 9))
    with open(os.path.join(OUT, "cmems-uv.json"), "w") as f:
        json.dump({"at": int(time.time() * 1000), "t": t, "nx": NX, "ny": NY, "lat0": lat0, "dlat": dlat, "lon0": lon0, "dlon": dlon,
                   "scale": 1000}, f, separators=(",", ":"))
    print("ok t=%d flip=%s lat0=%.4f dlat=%.5f lon0=%.4f dlon=%.5f bytes=%d" % (t, flip, lat0, dlat, lon0, dlon, os.path.getsize(os.path.join(OUT, "cmems-uv.bin.gz"))))


main()
