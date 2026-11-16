"""Weekly catalogue of ~1M stars around the Sun for the Space Radar browser.

Sources
  * ESA Gaia DR3 (served by the AIP Gaia mirror; the ESA archive is unreachable from GitHub runners): position, parallax, G, BP-RP,
    GSP-Phot Teff / logg / radius, FLAME radius / luminosity, Hipparcos cross-match.
    Selection: parallax_over_error > 10, parallax > 0.333 mas (closer than ~3 kpc), G < 11.5, fetched in 192 HEALPix sky
    chunks (source_id ranges; a chunk the server cancels is split 4 ways), merged and cut to the TOP brightest.
  * HYG v4.1 (Hipparcos + Yale + Gliese, CC BY-SA 4.0): proper names, Bayer / Flamsteed / HD / HR / Gliese
    designations, spectral types, and the brightest stars Gaia saturates on (Sirius, Vega, ...).

Output (data/stars/out)
  index.txt.gz  line k (0-based) = star of rank k (brightest first): "<display>\\t<mag>\\t<aliases>"
                aliases = "|"-separated designations (α Ori, Alp Ori, alpha Ori, 58 Ori, HIP 27989, HD 39801, Gaia DR3 ...)
  st/<k>.json   shard k = ranks k*1000 .. k*1000+999: {"at": ms, "first": k*1000, "fields": [...], "rows": [...]}
                row = [gaia id (0 = Hipparcos-only), ra deg, dec deg (ICRS), parallax mas, mag (G; V for src 1),
                       BP-RP, Teff K, radius Rsun, logg, luminosity Lsun, spectral type, src (0 Gaia, 1 Hipparcos/HYG)]
Missing Teff / radius / luminosity are left null; the client derives them from colour and absolute magnitude.
  sky/0.bin, 1.bin, 2.bin  point cloud of every star, rank order (ranks 0..49999, 50000..249999, 250000..end), 16-byte
                little-endian records, record i == rank i: float32 x, y, z (parsec, ICRS cartesian from ra/dec/parallax),
                int16 absolute magnitude*100 (clamped +-3000; 32767 = unknown: null mag or parallax <= 0, then x=y=z=0 if parallax unusable), uint16 Teff K (row Teff, else from BP-RP, else 5772)
  sky/meta.json {"at", "n", "tiers": cumulative ends [50000, 250000, n], "rec": 16, "fields"}"""
import csv, gzip, io, json, math, os, struct, sys, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
UA = {"User-Agent": "yzr-net-stars/1.0"}
TAPS = ["https://gaia.aip.de/tap"]   # ESA archive + VizieR time out from GitHub runners (2026-11)
HYG = "https://raw.githubusercontent.com/astronexus/HYG-Database/main/hyg/CURRENT/hygdata_v41.csv"
TOP = int(os.environ.get("STARS_TOP", "1000000"))
MIN_ROWS = int(os.environ.get("STARS_MIN", "800000"))
MAX_PC = 3000.0
SHARD = 1000
ROW_FIELDS = ["id", "ra", "dec", "plx", "mag", "bprp", "teff", "R", "logg", "L", "sp", "src"]
GREEK = {"Alp": "α", "Bet": "β", "Gam": "γ", "Del": "δ", "Eps": "ε", "Zet": "ζ", "Eta": "η", "The": "θ", "Iot": "ι",
         "Kap": "κ", "Lam": "λ", "Mu": "μ", "Nu": "ν", "Xi": "ξ", "Omi": "ο", "Pi": "π", "Rho": "ρ", "Sig": "σ",
         "Tau": "τ", "Ups": "υ", "Phi": "φ", "Chi": "χ", "Psi": "ψ", "Ome": "ω"}
GREEK_LONG = {"Alp": "alpha", "Bet": "beta", "Gam": "gamma", "Del": "delta", "Eps": "epsilon", "Zet": "zeta", "The": "theta",
              "Iot": "iota", "Kap": "kappa", "Lam": "lambda", "Omi": "omicron", "Sig": "sigma", "Ups": "upsilon", "Ome": "omega"}


def _f(x):
    if x in (None, "", "null", "NaN", "nan"):
        return None
    v = float(x)
    return None if math.isnan(v) else v


def _r(x, d):
    return None if x is None else round(x, d)


# ---------------------------------------------------------------- HYG
def hyg_names(r):
    """(display, aliases) for one HYG row. Pure; unit-tested."""
    al = []
    con = r.get("con") or ""
    bay = r.get("bayer") or ""
    bayer = ""
    if bay and con:
        base, _, sub = bay.partition("-")
        g = GREEK.get(base, base)
        bayer = g + (sub if sub else "") + " " + con
        al.append(base + sub + " " + con)
        if base in GREEK_LONG:
            al.append(GREEK_LONG[base] + sub + " " + con)
        al.insert(0, bayer)
    flam = r.get("flam") or ""
    if flam and con:
        al.append(flam + " " + con)
    for key, pre in (("hip", "HIP "), ("hd", "HD "), ("hr", "HR ")):
        if r.get(key):
            al.append(pre + r[key])
    if r.get("gl"):
        al.append(r["gl"])
    disp = r.get("proper") or bayer or (flam + " " + con if flam and con else "") or \
        ("HD " + r["hd"] if r.get("hd") else "HIP " + r["hip"] if r.get("hip") else r.get("gl") or "")
    return disp, al


def bv_teff(bv):
    """Ballesteros (2012) B-V -> Teff (K)."""
    if bv is None:
        return None
    bv = max(-0.35, min(2.2, bv))
    return 4600.0 * (1.0 / (0.92 * bv + 1.7) + 1.0 / (0.92 * bv + 0.62))


def bc_v(teff):
    """Torres (2010, after Flower 1996) bolometric correction BC_V(Teff)."""
    lt = math.log10(teff)
    if lt < 3.70:
        c = [-0.190537291496456e5, 0.155144866764412e5, -0.421278819301717e4, 0.381476328422343e3]
    elif lt < 3.90:
        c = [-0.370510203809015e5, 0.385672629965804e5, -0.150651486316025e5, 0.261724637119416e4, -0.170623810323864e3]
    else:
        c = [-0.118115450538963e6, 0.137145973583929e6, -0.636233812100225e5, 0.147412923562646e5, -0.170587278406872e4, 0.788731721804990e2]
    return sum(ci * lt ** i for i, ci in enumerate(c))


def hyg_row(r):
    """HYG record -> catalogue row (src 1) or None when it has no usable distance."""
    d = _f(r.get("dist"))
    if not d or d <= 0 or d >= 100000 or d > MAX_PC:
        return None
    mag, absmag, ci = _f(r.get("mag")), _f(r.get("absmag")), _f(r.get("ci"))
    teff = bv_teff(ci)
    L = R = None
    if teff and absmag is not None:
        L = 10 ** (0.4 * (4.74 - (absmag + bc_v(teff))))
        R = math.sqrt(L) * (5772.0 / teff) ** 2
    bprp = None if ci is None else 1.3 * ci - 0.06   # rough B-V -> BP-RP, only used as a colour fallback
    return [0, _r(_f(r["ra"]) * 15.0, 6), _r(_f(r["dec"]), 6), _r(1000.0 / d, 4), _r(mag, 3), _r(bprp, 3),
            _r(teff, 0), _r(R, 3), None, _r(L, 4), (r.get("spect") or None), 1]


def load_hyg(text):
    """hip -> (display, aliases, spect, row) and the list of HYG stars without a Hipparcos number."""
    by_hip, nohip = {}, []
    for r in csv.DictReader(io.StringIO(text)):
        if r.get("id") == "0":       # the Sun
            continue
        disp, al = hyg_names(r)
        rec = (disp, al, r.get("spect") or None, hyg_row(r), _f(r.get("mag")))
        if r.get("hip"):
            by_hip[int(r["hip"])] = rec
        elif rec[3] is not None:
            nohip.append(rec)
    return by_hip, nohip


# ---------------------------------------------------------------- Gaia
def parse_gaia(records):
    """Gaia records (dicts, or CSV text) -> list of (row, hip or None). Pure; unit-tested."""
    out = []
    if isinstance(records, str):
        records = csv.DictReader(io.StringIO(records))
    for r in records:
        ra, dec, plx, g = _f(r["ra"]), _f(r["dec"]), _f(r["parallax"]), _f(r["phot_g_mean_mag"])
        if None in (ra, dec, plx, g) or plx <= 0:
            continue
        rad = _f(r.get("radius_flame")) or _f(r.get("radius_gspphot"))
        hip = _f(r.get("hip"))
        out.append(([int(r["source_id"]), _r(ra, 6), _r(dec, 6), _r(plx, 4), _r(g, 3), _r(_f(r["bp_rp"]), 3),
                     _r(_f(r["teff_gspphot"]), 0), _r(rad, 3), _r(_f(r["logg_gspphot"]), 2), _r(_f(r["lum_flame"]), 4), None, 0],
                    int(hip) if hip else None))
    return out


def merge(gaia, by_hip, nohip):
    """-> list of (display, aliases, row), brightest first."""
    stars, seen = [], set()
    for row, hip in gaia:
        al = ["Gaia DR3 %d" % row[0]]
        disp = "Gaia DR3 %d" % row[0]
        h = by_hip.get(hip) if hip else None
        if h:
            seen.add(hip)
            disp = h[0] or disp
            al = h[1] + al
            row[10] = h[2]
        elif hip:
            al.insert(0, "HIP %d" % hip)
            disp = "HIP %d" % hip
        stars.append((disp, al, row))
    for hip, h in by_hip.items():
        if hip not in seen and h[3] is not None:
            stars.append((h[0], h[1], h[3]))
    for h in nohip:
        stars.append((h[0], h[1], h[3]))
    stars.sort(key=lambda s: (s[2][4] if s[2][4] is not None else 99))
    return stars[:TOP]


def build(stars, at=0):
    lines = []
    for disp, al, row in stars:
        a = [x for x in al if x.lower() != disp.lower()]
        lines.append("%s\t%s\t%s" % (disp, "" if row[4] is None else "%g" % row[4], "|".join(a)))
    shards = {}
    for k in range(0, len(stars), SHARD):
        shards[k // SHARD] = {"at": at, "first": k, "fields": ROW_FIELDS, "rows": [s[2] for s in stars[k:k + SHARD]]}
    return "\n".join(lines), shards


# ---------------------------------------------------------------- sky (binary point cloud)
SKY_TIERS = [50000, 250000]           # rank bounds of sky/0.bin and sky/1.bin; sky/2.bin holds the rest
SKY_REC = 16
SKY_UNKNOWN = 32767


def teff_bprp(c):
    """Python port of the client's srStarTeffBpRp (Gaia BP-RP -> Teff K). Pure."""
    if c is None or not math.isfinite(c):
        return None
    c = max(-0.6, min(5.0, c))
    if c < 0.2:
        tc = [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.2]
        tt = [40000, 40000, 30000, 22000, 16000, 12500, 9700, 8495]
        for i in range(1, len(tc)):
            if c <= tc[i]:
                f = (c - tc[i - 1]) / (tc[i] - tc[i - 1])
                return math.exp(math.log(tt[i - 1]) + f * (math.log(tt[i]) - math.log(tt[i - 1])))
    if c > 2.0:
        return max(2300.0, 3550.0 - 420.0 * (c - 2.0))
    th = 0.4929 + 0.5092 * c - 0.0353 * c * c
    return 5040.0 / th


def sky_record(row):
    """One 16-byte record: f32 x,y,z (pc, ICRS), i16 absmag*100 (32767 unknown), u16 Teff K. Pure."""
    ra, dec, plx, mag, bprp, teff = row[1], row[2], row[3], row[4], row[5], row[6]
    t = teff if teff else (teff_bprp(bprp) or 5772.0)
    t = int(round(max(1500.0, min(65000.0, t))))
    if plx is None or ra is None or dec is None or plx <= 0:
        return struct.pack("<fffhH", 0.0, 0.0, 0.0, SKY_UNKNOWN, t)
    d = 1000.0 / plx
    ra, dec = math.radians(ra), math.radians(dec)
    cd = d * math.cos(dec)
    if mag is None:
        am = SKY_UNKNOWN
    else:
        am = int(round(max(-30.0, min(30.0, mag - 5.0 * math.log10(d / 10.0))) * 100))
    return struct.pack("<fffhH", cd * math.cos(ra), cd * math.sin(ra), d * math.sin(dec), am, t)


def build_sky(stars, at=0):
    """-> (tiers, meta): tiers = list of bytes (rank slices SKY_TIERS), one record per star in rank order. Pure."""
    n = len(stars)
    bounds, prev = [], 0
    for b in list(SKY_TIERS) + [n]:
        prev = max(prev, min(b, n))
        bounds.append(prev)
    tiers, lo = [], 0
    for hi in bounds:
        tiers.append(b"".join(sky_record(s[2]) for s in stars[lo:hi]))
        lo = hi
    meta = {"at": at, "n": n, "tiers": bounds, "rec": SKY_REC,
            "fields": "f32 x,y,z pc ICRS; i16 absmag*100 (32767 unknown); u16 teff K"}
    return tiers, meta


# ---------------------------------------------------------------- network
CHUNK_LEVEL = 2                       # HEALPix level of the sky chunks: 12 * 4**2 = 192 queries
MAX_SPLIT = 4                         # a chunk the server cancels (statement timeout) is split 4 ways, down to level 6
G_LIMIT = 11.5                        # ~1.3M candidates (measured: G<13 gives ~4.5M); the TOP brightest are kept after the merge
VO = "{http://www.ivoa.net/xml/VOTable/v1.3}"


def http(req, timeout=300, tries=4):
    for k in range(tries):
        try:
            return urllib.request.urlopen(req, timeout=timeout).read()
        except urllib.error.HTTPError as e:
            print("retry", k, e.code, e.read()[:160])
        except Exception as e:
            print("retry", k, str(e)[:120])
        time.sleep(5 + 15 * k)
    return None


def votable_rows(xml):
    """VOTable TABLEDATA -> list of dicts (empty cells = ''). Pure; unit-tested."""
    root = ET.fromstring(xml)
    for info in root.iter(VO + "INFO"):
        if info.attrib.get("name") == "QUERY_STATUS" and info.attrib.get("value") == "ERROR":
            raise RuntimeError("TAP error: %s" % (info.text or "")[:300])
    names = [f.attrib["name"] for f in root.iter(VO + "FIELD")]
    out = []
    for tr in root.iter(VO + "TR"):
        out.append({n: (td.text or "") for n, td in zip(names, tr.findall(VO + "TD"))})
    return out


QUERY_CHUNK = (
    "SELECT g.source_id, g.ra, g.dec, g.parallax, g.phot_g_mean_mag, g.bp_rp, "
    "a.teff_gspphot, a.radius_gspphot, a.logg_gspphot, a.radius_flame, a.lum_flame, h.original_ext_source_id AS hip "
    "FROM gaiadr3.gaia_source AS g "
    "LEFT OUTER JOIN gaiadr3.astrophysical_parameters AS a ON a.source_id = g.source_id "
    "LEFT OUTER JOIN gaiadr3.hipparcos2_best_neighbour AS h ON h.source_id = g.source_id "
    "WHERE g.source_id BETWEEN %d AND %d AND g.phot_g_mean_mag < %g AND g.parallax_over_error > 10 AND g.parallax > 0.333"
)


def range_query(lo, hi):
    return QUERY_CHUNK % (lo, hi, G_LIMIT)


def chunk_range(pix, level=CHUNK_LEVEL):
    span = 2 ** 35 * 4 ** (12 - level)
    return pix * span, (pix + 1) * span - 1


def chunk_query(pix):
    return range_query(*chunk_range(pix))


def fetch_range(base, lo, hi, depth=0):
    """One sky range; when the server keeps cancelling it, split it in 4 and recurse."""
    body = urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "votable", "QUERY": range_query(lo, hi)}).encode()
    for k in range(2 if depth < MAX_SPLIT else 5):
        b = http(urllib.request.Request(base + "/sync", data=body, headers=UA), 600, tries=1)
        if b:
            try:
                return votable_rows(b)
            except Exception as e:
                print("range", lo, "bad answer", str(e)[:120])
        time.sleep(5 + 15 * k)
    if depth >= MAX_SPLIT:
        raise RuntimeError("range %d..%d failed" % (lo, hi))
    q = (hi - lo + 1) // 4
    print("split", lo, "depth", depth + 1, flush=True)
    out = []
    for j in range(4):
        out.extend(fetch_range(base, lo + j * q, (lo + (j + 1) * q - 1) if j < 3 else hi, depth + 1))
    return out


def fetch_gaia():
    from concurrent.futures import ThreadPoolExecutor
    n = 12 * 4 ** CHUNK_LEVEL
    for base in TAPS:
        try:
            t, rows = time.time(), []
            with ThreadPoolExecutor(3) as ex:
                for pix, part in enumerate(ex.map(lambda p: fetch_range(base, *chunk_range(p)), range(n))):
                    rows.extend(part)
                    if pix % 16 == 15:
                        print("chunks %d/%d rows %d %.0fs" % (pix + 1, n, len(rows), time.time() - t), flush=True)
            print("gaia from", base, "%.0fs" % (time.time() - t), len(rows), "rows")
            return rows
        except Exception as e:
            print("TAP failed", base, str(e)[:300])
    sys.exit("no Gaia TAP service answered")


def main():
    t0 = time.time()
    hy = http(urllib.request.Request(HYG, headers=UA), 300)
    if not hy:
        sys.exit("HYG download failed")
    by_hip, nohip = load_hyg(hy.decode("utf-8", "replace"))
    gaia = parse_gaia(fetch_gaia())
    stars = merge(gaia, by_hip, nohip)
    print("gaia rows", len(gaia), "hyg", len(by_hip), "+", len(nohip), "merged", len(stars))
    if len(stars) < MIN_ROWS:
        sys.exit("only %d stars, refusing to publish" % len(stars))
    idx, shards = build(stars, int(time.time() * 1000))
    os.makedirs(os.path.join(OUT, "st"), exist_ok=True)
    with open(os.path.join(OUT, "index.txt.gz"), "wb") as f:
        f.write(gzip.compress(idx.encode("utf-8"), 9, mtime=0))
    for k, sh in shards.items():
        with open(os.path.join(OUT, "st", "%d.json" % k), "w", encoding="utf-8") as f:
            json.dump(sh, f, separators=(",", ":"), ensure_ascii=False)
    tiers, meta = build_sky(stars, int(time.time() * 1000))
    os.makedirs(os.path.join(OUT, "sky"), exist_ok=True)
    for k, b in enumerate(tiers):
        with open(os.path.join(OUT, "sky", "%d.bin" % k), "wb") as f:
            f.write(b)
    with open(os.path.join(OUT, "sky", "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, separators=(",", ":"))
    print("ok stars=%d shards=%d index=%d bytes in %.0fs" % (len(stars), len(shards), os.path.getsize(os.path.join(OUT, "index.txt.gz")), time.time() - t0))


if __name__ == "__main__":
    main()
