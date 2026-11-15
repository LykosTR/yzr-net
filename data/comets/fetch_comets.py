"""Fetch comet orbits from NASA/JPL SBDB and write them for the browser.

Source (public, no key): NASA/JPL SBDB Query API, every comet (sb-kind=c), full precision.
Kept: comets within 15 au of the Sun right now (two-body estimate from the perihelion elements)
plus a short list of well-known comets wherever they are.

Output (data/comets/out/comets.json):
  {"at": ms, "jd": JD (TDB) of the fetch, "fields": [...], "rows": [[...]]}
  fields: spk, name, pdes, prefix, e, q (au), i, om, w (deg, ecliptic J2000), tp (JD TDB of perihelion),
          epoch (JD TDB), M1, K1 (total-magnitude law m = M1 + 5 log D + K1 log r), diameter (km, nucleus, or null),
          per (years, or null)
Client propagation: two-body from (q, e, tp): ellipse, parabola or hyperbola."""
import datetime as dt, json, math, os, time, urllib.parse, urllib.request

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
SBDB = "https://ssd-api.jpl.nasa.gov/sbdb_query.api?"
UA = {"User-Agent": "yzr-net-comets/1.0"}
K = 0.01720209895  # Gaussian gravitational constant (au^1.5 / day)
FIELDS = ["spkid", "full_name", "pdes", "prefix", "e", "q", "i", "om", "w", "tp", "epoch", "M1", "K1", "diameter", "per_y"]
FAMOUS = {"1P", "2P", "9P", "19P", "21P", "29P", "46P", "67P", "81P", "103P", "109P", "12P", "13P", "55P",
          "1995 O1", "1996 B2", "2020 F3", "2023 A3", "2022 E3", "2006 P1"}
R_MAX = 15.0


def http(url, tries=5, timeout=120):
    for k in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read().decode("utf-8")
        except Exception as e:
            print("retry", k, str(e)[:100])
            time.sleep(3 + 4 * k)
    raise SystemExit("SBDB unreachable")


def jd_now():
    return 2440587.5 + time.time() / 86400.0 + 69.184 / 86400.0


def helio_r(q, e, tp, jd):
    """heliocentric distance (au) at jd from perihelion elements (two-body)"""
    t = jd - tp
    if abs(e - 1.0) < 1e-6:  # parabola: Barker
        w = 3 * K * t / math.sqrt(2 * q ** 3)
        y = (w + math.sqrt(w * w + 1)) ** (1 / 3)
        s = y - 1 / y
        return q * (1 + s * s)
    a = q / abs(1 - e)
    n = K / a ** 1.5
    M = n * t
    if e < 1:
        M = math.remainder(M, 2 * math.pi)
        E = M if e < 0.8 else math.pi * (1 if M >= 0 else -1)
        for _ in range(60):
            d = (E - e * math.sin(E) - M) / (1 - e * math.cos(E))
            E -= d
            if abs(d) < 1e-12:
                break
        return a * (1 - e * math.cos(E))
    H = math.asinh(M / e)
    for _ in range(80):
        d = (e * math.sinh(H) - H - M) / (e * math.cosh(H) - 1)
        H -= d
        if abs(d) < 1e-12:
            break
    return a * (e * math.cosh(H) - 1)


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    q = urllib.parse.urlencode({"fields": ",".join(FIELDS), "sb-kind": "c", "full-prec": "1"})
    d = json.loads(http(SBDB + q))
    assert d.get("fields") == FIELDS, d.get("fields")
    jd = jd_now()
    rows = []
    for r in d["data"]:
        spk, name, pdes, prefix = r[0], (r[1] or "").strip(), r[2], r[3]
        e, qq, inc, om, w, tp, ep = (num(x) for x in r[4:11])
        if None in (e, qq, inc, om, w, tp) or qq <= 0 or e < 0:
            continue
        if prefix in ("D", "X"):
            continue  # defunct / no usable orbit
        try:
            rnow = helio_r(qq, e, tp, jd)
        except (OverflowError, ValueError, ZeroDivisionError):
            continue
        if not (rnow < R_MAX or pdes in FAMOUS):
            continue
        rows.append([int(spk), name, pdes, prefix, round(e, 10), round(qq, 10), round(inc, 8), round(om, 8), round(w, 8),
                     round(tp, 8), ep, num(r[11]), num(r[12]), num(r[13]), num(r[14])])
    rows.sort(key=lambda x: x[5])
    os.makedirs(OUT, exist_ok=True)
    out = {"at": int(time.time() * 1000), "jd": round(jd, 6),
           "fields": ["spk", "name", "pdes", "prefix", "e", "q", "i", "om", "w", "tp", "epoch", "M1", "K1", "diameter", "per"],
           "rows": rows}
    with open(os.path.join(OUT, "comets.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    print("comets:", len(rows), "of", len(d["data"]))


if __name__ == "__main__":
    main()
