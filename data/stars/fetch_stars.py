"""Catalogue of ~1M nearby stars (ESA Gaia DR3 TAP) for the Space Radar browser.

Selection: parallax_over_error > 10, parallax > 0.33 mas (< ~3 kpc), brightest first (phot_g_mean_mag), TOP N.
Output (data/stars/out):
  index.txt.gz   line k (0-based) = star of rank k:  "<label>\\t<G>"  label = "HIP 32349" if Hipparcos-matched else "Gaia <source_id>"
  st/<k>.json    shard k = ranks k*1000 .. k*1000+999:
                 {"at": ms, "first": k*1000, "fields": [...], "rows": [[id, ra deg, dec deg, plx mas, G, bp_rp, teff K, R Rsun, logg, L Lsun], ...]}
Teff / R / logg: GSP-Phot, R and L fall back to FLAME; missing teff is derived from BP-RP on the client."""
import csv, gzip, io, json, os, sys, time, urllib.parse, urllib.request

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
TAP = "https://gea.esac.esa.int/tap-server/tap"
UA = {"User-Agent": "yzr-net-stars/1.0"}
TOP = 1000000
MIN_ROWS = 800000
SHARD = 1000
ROW_FIELDS = ["id", "ra", "dec", "plx", "G", "bprp", "teff", "R", "logg", "L"]
QUERY = (
    "SELECT TOP %d g.source_id, g.ra, g.dec, g.parallax, g.phot_g_mean_mag, g.bp_rp, "
    "a.teff_gspphot, a.radius_gspphot, a.logg_gspphot, a.radius_flame, a.lum_flame, h.original_ext_source_id AS hip "
    "FROM gaiadr3.gaia_source AS g "
    "LEFT JOIN gaiadr3.astrophysical_parameters AS a ON a.source_id = g.source_id "
    "LEFT JOIN gaiadr3.hipparcos2_best_neighbour AS h ON h.source_id = g.source_id "
    "WHERE g.parallax_over_error > 10 AND g.parallax > 0.33 AND g.phot_g_mean_mag IS NOT NULL "
    "ORDER BY g.phot_g_mean_mag ASC" % TOP
)


def _f(x):
    return None if x in (None, "", "null") else float(x)


def _r(x, d):
    return None if x is None else round(x, d)


def parse_rows(text):
    """Gaia CSV -> list of (label, row). Pure; unit-tested."""
    out = []
    for r in csv.DictReader(io.StringIO(text)):
        ra, dec, plx, g = _f(r["ra"]), _f(r["dec"]), _f(r["parallax"]), _f(r["phot_g_mean_mag"])
        if None in (ra, dec, plx, g) or plx <= 0:
            continue
        rad = _f(r.get("radius_gspphot")) or _f(r.get("radius_flame"))
        hip = r.get("hip") or ""
        label = ("HIP " + hip.split(".")[0]) if hip else "Gaia " + r["source_id"]
        out.append((label, [int(r["source_id"]), _r(ra, 6), _r(dec, 6), _r(plx, 4), _r(g, 3), _r(_f(r["bp_rp"]), 3),
                            _r(_f(r["teff_gspphot"]), 0), _r(rad, 3), _r(_f(r["logg_gspphot"]), 2), _r(_f(r["lum_flame"]), 4)]))
    return out


def build(stars, at=0):
    idx = "\n".join("%s\t%g" % (lab, row[4]) for lab, row in stars)
    shards = {}
    for k in range(0, len(stars), SHARD):
        shards[k // SHARD] = {"at": at, "first": k, "fields": ROW_FIELDS, "rows": [row for _, row in stars[k:k + SHARD]]}
    return idx, shards


def http(url, data=None, tries=5, timeout=300):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers=UA)
            return urllib.request.urlopen(req, timeout=timeout)
        except Exception as e:
            print("retry", k, str(e)[:100], url[:100])
            time.sleep(5 + 10 * k)
    return None


def run_async():
    body = urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "PHASE": "RUN", "QUERY": QUERY}).encode()

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None
    opener = urllib.request.build_opener(NoRedirect)
    try:
        opener.open(urllib.request.Request(TAP + "/async", data=body, headers=UA), timeout=120)
        sys.exit("async submit returned no redirect")
    except urllib.error.HTTPError as e:
        job = e.headers.get("Location")
    if not job:
        sys.exit("no job url")
    for _ in range(360):
        ph = http(job + "/phase").read().decode().strip()
        print("phase", ph)
        if ph == "COMPLETED":
            return http(job + "/results/result", timeout=900).read().decode("utf-8", "replace")
        if ph in ("ERROR", "ABORTED"):
            sys.exit("job " + ph)
        time.sleep(20)
    sys.exit("job timeout")


def main():
    t0 = time.time()
    stars = parse_rows(run_async())
    if len(stars) < MIN_ROWS:
        sys.exit("only %d stars, refusing to publish" % len(stars))
    idx, shards = build(stars, int(time.time() * 1000))
    os.makedirs(os.path.join(OUT, "st"), exist_ok=True)
    with open(os.path.join(OUT, "index.txt.gz"), "wb") as f:
        f.write(gzip.compress(idx.encode("utf-8"), 9, mtime=0))
    for k, sh in shards.items():
        with open(os.path.join(OUT, "st", "%d.json" % k), "w") as f:
            json.dump(sh, f, separators=(",", ":"))
    print("ok stars=%d shards=%d index=%d bytes in %.0fs" % (len(stars), len(shards), os.path.getsize(os.path.join(OUT, "index.txt.gz")), time.time() - t0))


if __name__ == "__main__":
    main()
