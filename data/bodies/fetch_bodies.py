"""Fetch live osculating orbital elements of the Solar System bodies and write them for the browser.

Sources (public, no key):
  * NASA/JPL Horizons API  -> planets, dwarf planets and every natural satellite (ELEMENTS at "now", ecliptic J2000 / ICRF,
    planets + dwarfs about the Sun, satellites about their parent planet's centre).
  * NASA/JPL SBDB Query API -> asteroids (all PHAs, all NEOs brighter than H 19 and every asteroid brighter than H 9.5).

Output (data/bodies/out):
  bodies.json    {"at": ms, "jd": epoch (TDB), "frame": ..., "bodies": [...]}  one object per Horizons body:
                 {"id", "name", "type", "parent", "a" km, "e", "i" deg, "om" deg, "w" deg, "ma" deg (mean anomaly AT jd),
                  "n" deg/s (mean motion), "per" s}
  asteroids.json {"at": ms, "fields": [...], "rows": [[...]]}  SBDB rows; "epoch" is the element epoch (JD TDB),
                 a in au, n in deg/day, ma in deg at epoch.
Client propagation (two-body Kepler): M(t) = ma + n * (t - epoch); solve E, convert to position. Refetched every hour,
so the error never grows past one refresh interval."""
import concurrent.futures as cf, datetime as dt, json, math, os, re, sys, time, urllib.parse, urllib.request

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
HZ = "https://ssd.jpl.nasa.gov/api/horizons.api?"
SBDB = "https://ssd-api.jpl.nasa.gov/sbdb_query.api?"
UA = {"User-Agent": "yzr-net-bodies/1.0"}
J2000_JD = 2451545.0

PLANETS = {199: "Mercury", 299: "Venus", 399: "Earth", 499: "Mars", 599: "Jupiter", 699: "Saturn", 799: "Uranus", 899: "Neptune"}
DWARFS = {999: "Pluto", "1;": "Ceres", "136199;": "Eris", "136108;": "Haumea", "136472;": "Makemake", "225088;": "Gonggong",
          "50000;": "Quaoar", "90377;": "Sedna", "90482;": "Orcus"}
PARENT = {3: 399, 4: 499, 5: 599, 6: 699, 7: 799, 8: 899, 9: 999}  # leading digit of a satellite id -> parent planet id


def http(url, tries=5, timeout=90):
    for k in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read().decode("utf-8", "replace")
        except Exception as e:
            print("retry", k, str(e)[:100], url[:110])
            time.sleep(2 + 3 * k)
    return None


def jd_now():
    d = dt.datetime.now(dt.timezone.utc)
    jd_utc = 2440587.5 + d.timestamp() / 86400.0
    return jd_utc + 69.184 / 86400.0  # TT-UTC (leap seconds + 32.184); TDB ~ TT to < 2 ms


def moon_ids():
    txt = json.loads(http(HZ + urllib.parse.urlencode({"format": "json", "COMMAND": "'MB'"})))["result"]
    out = []
    for line in txt.splitlines():
        m = re.match(r"\s+(\d+)\s{2,}(.+?)(\s{2,}|$)", line)
        if not m:
            continue
        i, name = int(m.group(1)), m.group(2).strip().replace("_", " ")
        if i == 635:
            continue  # Daphnis: Horizons has no ephemeris after 2018
        if i == 301 or (401 <= i < 1000 and i % 100 != 99 and i // 100 in PARENT):
            out.append((i, name, PARENT[i // 100]))
        elif 55500 <= i < 55600:
            out.append((i, name, 599))
        elif 65000 <= i < 65400:
            out.append((i, name, 699))
    return out


def elements(cmd, center, jd):
    q = urllib.parse.urlencode({
        "format": "json", "COMMAND": "'%s'" % cmd, "OBJ_DATA": "'NO'", "MAKE_EPHEM": "'YES'", "EPHEM_TYPE": "'ELEMENTS'",
        "CENTER": "'%s'" % center, "TLIST": "'%.6f'" % jd, "REF_PLANE": "'ECLIPTIC'", "REF_SYSTEM": "'ICRF'",
        "OUT_UNITS": "'KM-S'", "CSV_FORMAT": "'YES'"})
    txt = http(HZ + q)
    if not txt:
        return None
    res = json.loads(txt).get("result", "")
    a, b = res.find("$$SOE"), res.find("$$EOE")
    if a < 0 or b < 0:
        return None
    f = [x.strip() for x in res[a + 5:b].strip().split(",")]
    # JDTDB, Calendar, EC, QR, IN, OM, W, Tp, N, MA, TA, A, AD, PR
    try:
        v = dict(e=float(f[2]), i=float(f[4]), om=float(f[5]), w=float(f[6]), n=float(f[8]), ma=float(f[9]), a=float(f[11]), per=float(f[13]))
    except (ValueError, IndexError):
        return None
    if not (0 <= v["e"] < 1 and v["a"] > 0):
        return None
    return {k: float("%.10g" % x) for k, x in v.items()}


def one(job):
    (bid, name, typ, parent), center, jd = job
    el = elements(bid, center, jd)
    if isinstance(bid, str):
        bid = 20000000 + int(bid.rstrip(";"))  # SPK id of a numbered asteroid
    if el is None:
        print("skip", bid, name)
        return None
    return dict(id=bid, name=name, type=typ, parent=parent, **el)


def asteroids():
    base = {"fields": "spkid,full_name,a,e,i,om,w,ma,n,epoch,H,diameter,pha,neo", "sb-kind": "a", "full-prec": "0"}
    seen, rows = set(), []
    for cd, cls in (({"AND": ["H|RG|-5|9.5"]}, None), ({"AND": ["H|RG|-5|19"]}, "NEO"), (None, "PHA")):
        q = dict(base)
        if cd:
            q["sb-cdata"] = json.dumps(cd, separators=(",", ":"))
        if cls == "NEO":
            q["sb-group"] = "neo"
        if cls == "PHA":
            q["sb-group"] = "pha"
        txt = http(SBDB + urllib.parse.urlencode(q), timeout=180)
        if not txt:
            sys.exit("SBDB request failed")
        j = json.loads(txt)
        for r in j["data"]:
            if r[0] in seen or None in r[2:10]:
                continue
            seen.add(r[0])
            rows.append(r)
    out = []
    for r in rows:
        spk, name = r[0], " ".join(str(r[1]).split())
        num = re.match(r"(\d+) ", name)
        out.append([int(spk), name, round(float(r[2]), 6), round(float(r[3]), 6), round(float(r[4]), 4), round(float(r[5]), 4),
                    round(float(r[6]), 4), round(float(r[7]), 4), round(float(r[8]), 6), float(r[9]),
                    None if r[10] in (None, "") else round(float(r[10]), 2), None if r[11] in (None, "") else round(float(r[11]), 3),
                    1 if r[12] == "Y" else 0, 1 if r[13] == "Y" else 0])
    return out


def main():
    jd = jd_now()
    jobs = []
    for bid, nm in PLANETS.items():
        jobs.append(((bid, nm, "Planet", 10), "500@10", jd))
    for bid, nm in DWARFS.items():
        jobs.append(((bid, nm, "Dwarf planet", 10), "500@10", jd))
    moons = moon_ids()
    for bid, nm, par in moons:
        jobs.append(((bid, nm, "Moon", par), "500@%d" % par, jd))
    print("jobs", len(jobs), "moons", len(moons))
    t0 = time.time()
    res, todo = [], jobs
    for rnd, workers in enumerate((3, 1, 1)):
        with cf.ThreadPoolExecutor(workers) as ex:
            got = list(ex.map(one, todo))
        res += [r for r in got if r]
        todo = [j for j, r in zip(todo, got) if not r]
        if not todo:
            break
        time.sleep(10)
    print("missing:", [j[0][1] for j in todo])
    print("horizons ok %d/%d in %.0fs" % (len(res), len(jobs), time.time() - t0))
    core = [r for r in res if r["type"] != "Moon"]
    have = {r["id"] for r in core}
    if not set(PLANETS) <= have or len(res) < 0.9 * len(jobs):
        sys.exit("too few bodies (%d/%d), refusing to publish" % (len(res), len(jobs)))
    ast = asteroids()
    if len(ast) < 3000:
        sys.exit("only %d asteroids, refusing to publish" % len(ast))
    os.makedirs(OUT, exist_ok=True)
    now = int(time.time() * 1000)
    with open(os.path.join(OUT, "bodies.json"), "w") as f:
        json.dump({"at": now, "jd": round(jd, 6), "frame": "ecliptic J2000 (ICRF); km, deg, s; planets/dwarfs about the Sun, moons about parent",
                   "bodies": res}, f, separators=(",", ":"))
    with open(os.path.join(OUT, "asteroids.json"), "w") as f:
        json.dump({"at": now, "frame": "heliocentric ecliptic J2000; a au, angles deg, n deg/day, epoch JD TDB, diameter km",
                   "fields": ["spk", "name", "a", "e", "i", "om", "w", "ma", "n", "epoch", "H", "diameter", "pha", "neo"], "rows": ast},
                  f, separators=(",", ":"))
    print("ok bodies=%d asteroids=%d bytes=%d+%d" % (len(res), len(ast), os.path.getsize(os.path.join(OUT, "bodies.json")),
                                                    os.path.getsize(os.path.join(OUT, "asteroids.json"))))


main()
