"""Monthly catalogue of bright galaxies for the Space Radar browser.

Source
  * HyperLEDA (Observatoire de Lyon) SQL service, table meandata: position, morphology, D25, axis ratio, PA, total B mag
    (corrected), B-V, distance modulus (modbest), heliocentric velocity v, PGC number.
    Selection: btc < LIMIT (default 15.0, ~53k rows before the distance filter). Fetched in one query; if that fails
    the same selection is fetched in declination bands.
  * Fallback (TODO, not implemented): RC3 via VizieR VII/155 (RA/DE sexagesimal, T, type, D25, R25, PA, BT, B-VT, cz; no distance
    -> cz/H0). If HyperLEDA fails the script exits non-zero, so the workflow does not publish broken data.

Processing
  distance D (Mpc): 10^((modbest-25)/5) when modbest present (sane 0.01..1500 Mpc), else v/H0 (H0 = 70) when v > 500 km/s, else dropped.
  major diameter kpc = D25 (arcmin -> rad) * D * 1000; axis ratio b/a = 10^-logr25; T filled from the class string when blank.
  Dropped: no D25, no distance, the Milky Way, anything else without a usable distance (star clusters / nebulae such as NGC 2239).

Output (data/galaxies/out)
  gal.bin    N records, brightest first (record i == index i), 28 bytes, little-endian, struct format "<ffffhhBbBBI":
               float32 x, y, z   Mpc, ICRS cartesian (x toward RA 0 Dec 0, z toward the north pole)
               float32 diam      major axis D25 in kpc
               int16   btc100    total B magnitude * 100
               int16   pa10      position angle (deg, N through E) * 10, -1 unknown
               uint8   ba255     minor/major axis ratio * 255 (255 when unknown)
               int8    t10       morphological T * 10 (E=-50 ... Im=100), -128 unknown
               uint8   flags     bit0 bar (SB, IB), bit1 weak bar (SAB, IAB), bit2 B-V known, bit3 distance from redshift
               uint8   bve       (B-V + 0.5) * 100 clamped 0..250; 255 when unknown (client defaults by T)
               uint32  pgc       PGC number (0 unknown)
  names.json [[index, "display name", "alias|alias|..."], ...] common-named galaxies, Messier objects, and all btc < 10
  search.json ["NGC 4565", "UGC 7772", ...] one HyperLEDA name per record, aligned with the record index (display formatted; client search)
  meta.json  {"at": ISO time, "n", "fmt", "rec", "limit", "source", "h0"}
"""
import csv, io, json, math, os, re, struct, sys, time, urllib.request
from datetime import datetime, timezone

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
UA = {"User-Agent": "yzr-net-galaxies/1.0"}
LIMIT = float(os.environ.get("GAL_LIMIT", "15.0"))
MIN_ROWS = int(os.environ.get("GAL_MIN", "30000"))
H0 = 70.0
FMT = "<ffffhhBbBBI"
REC = struct.calcsize(FMT)
BASE = "http://atlas.obs-hp.fr/hyperleda/fG.cgi?n=meandata&c=o&of=1,leda,simbad&nra=l&nakd=1"
COLS = "pgc,objname,al2000,de2000,t,type,logd25,logr25,pa,btc,bve,modbest,v"

# HyperLEDA objname -> (display name, aliases)
NAMES = {
    "NGC0224": ("Andromeda Galaxy (M31)", "M31|Messier 31|Andromeda|NGC 224"),
    "NGC0221": ("M32", "M32|Messier 32|NGC 221"),
    "NGC0205": ("M110", "M110|Messier 110|NGC 205"),
    "NGC0598": ("Triangulum Galaxy (M33)", "M33|Messier 33|Triangulum|Pinwheel Galaxy|NGC 598"),
    "ESO056-115": ("Large Magellanic Cloud", "LMC|Large Magellanic Cloud|Büyük Macellan Bulutu"),
    "NGC0292": ("Small Magellanic Cloud", "SMC|Small Magellanic Cloud|Küçük Macellan Bulutu|NGC 292"),
    "NGC3031": ("Bode's Galaxy (M81)", "M81|Messier 81|Bode's Galaxy|NGC 3031"),
    "NGC3034": ("Cigar Galaxy (M82)", "M82|Messier 82|Cigar Galaxy|NGC 3034"),
    "NGC5194": ("Whirlpool Galaxy (M51)", "M51|Messier 51|Whirlpool|NGC 5194"),
    "NGC5457": ("Pinwheel Galaxy (M101)", "M101|Messier 101|Pinwheel|NGC 5457"),
    "NGC4594": ("Sombrero Galaxy (M104)", "M104|Messier 104|Sombrero|NGC 4594"),
    "NGC5128": ("Centaurus A", "Cen A|Centaurus A|NGC 5128"),
    "NGC0253": ("Sculptor Galaxy", "Sculptor Galaxy|Silver Coin|NGC 253"),
    "NGC4826": ("Black Eye Galaxy (M64)", "M64|Messier 64|Black Eye|NGC 4826"),
    "NGC5055": ("Sunflower Galaxy (M63)", "M63|Messier 63|Sunflower|NGC 5055"),
    "NGC3623": ("M65 (Leo Triplet)", "M65|Messier 65|Leo Triplet|NGC 3623"),
    "NGC3627": ("M66 (Leo Triplet)", "M66|Messier 66|Leo Triplet|NGC 3627"),
    "NGC3628": ("Hamburger Galaxy (Leo Triplet)", "Hamburger Galaxy|Leo Triplet|NGC 3628"),
    "NGC1300": ("NGC 1300", "NGC 1300|barred spiral"),
    "NGC4486": ("Virgo A (M87)", "M87|Messier 87|Virgo A|NGC 4486"),
    "NGC4565": ("Needle Galaxy", "Needle Galaxy|NGC 4565"),
    "NGC4631": ("Whale Galaxy", "Whale Galaxy|NGC 4631"),
    "NGC4038": ("Antennae Galaxies", "Antennae|NGC 4038"),
    "NGC4039": ("Antennae Galaxies (NGC 4039)", "Antennae|NGC 4039"),
    "NGC6946": ("Fireworks Galaxy", "Fireworks Galaxy|NGC 6946"),
    "NGC0891": ("NGC 891", "NGC 891"),
    "NGC1365": ("Great Barred Spiral (NGC 1365)", "Great Barred Spiral|NGC 1365"),
    "NGC7331": ("NGC 7331", "NGC 7331"),
    "NGC2403": ("NGC 2403", "NGC 2403"),
    "NGC0628": ("M74", "M74|Messier 74|Phantom Galaxy|NGC 628"),
    "NGC1068": ("M77", "M77|Messier 77|Cetus A|NGC 1068"),
    "NGC5236": ("Southern Pinwheel (M83)", "M83|Messier 83|Southern Pinwheel|NGC 5236"),
    "NGC4736": ("M94", "M94|Messier 94|NGC 4736"),
    "NGC3351": ("M95", "M95|Messier 95|NGC 3351"),
    "NGC3368": ("M96", "M96|Messier 96|NGC 3368"),
    "NGC4192": ("M98", "M98|Messier 98|NGC 4192"),
    "NGC4254": ("M99", "M99|Messier 99|NGC 4254"),
    "NGC4321": ("M100", "M100|Messier 100|NGC 4321"),
    "NGC4258": ("M106", "M106|Messier 106|NGC 4258"),
    "NGC3556": ("M108", "M108|Messier 108|NGC 3556"),
    "NGC3992": ("M109", "M109|Messier 109|NGC 3992"),
    "NGC4472": ("M49", "M49|Messier 49|NGC 4472"),
    "NGC4579": ("M58", "M58|Messier 58|NGC 4579"),
    "NGC4621": ("M59", "M59|Messier 59|NGC 4621"),
    "NGC4649": ("M60", "M60|Messier 60|NGC 4649"),
    "NGC4303": ("M61", "M61|Messier 61|NGC 4303"),
    "NGC4374": ("M84", "M84|Messier 84|NGC 4374"),
    "NGC4382": ("M85", "M85|Messier 85|NGC 4382"),
    "NGC4406": ("M86", "M86|Messier 86|NGC 4406"),
    "NGC4501": ("M88", "M88|Messier 88|NGC 4501"),
    "NGC4552": ("M89", "M89|Messier 89|NGC 4552"),
    "NGC4569": ("M90", "M90|Messier 90|NGC 4569"),
    "NGC4548": ("M91", "M91|Messier 91|NGC 4548"),
    "NGC3379": ("M105", "M105|Messier 105|NGC 3379"),
    "NGC5866": ("Spindle Galaxy (M102?)", "M102|Messier 102|Spindle Galaxy|NGC 5866"),
    "IC0342": ("IC 342", "IC 342|Hidden Galaxy"),
    "NGC2841": ("NGC 2841", "NGC 2841"),
    "NGC4244": ("Silver Needle", "Silver Needle|NGC 4244"),
}
T_CLASS = [("E", -5), ("S0-a", -1), ("S0", -2), ("Sab", 2), ("Sa", 1), ("Sbc", 4), ("Sb", 3), ("Scd", 6), ("Sc", 5), ("Sd", 7),
           ("Sm", 9), ("Im", 10), ("Irr", 10), ("I", 10)]


def http(url, timeout=120, tries=4):
    for k in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read().decode("utf-8", "replace")
        except Exception as e:
            print("retry", k, str(e)[:120], file=sys.stderr)
        time.sleep(5 + 15 * k)
    return None


def parse_csv(text):
    lines = [l for l in text.splitlines() if l and not l.startswith("#")]
    rows = list(csv.DictReader(lines))
    if not rows or "objname" not in rows[0]:
        raise ValueError("unexpected HyperLEDA reply: " + text[:200])
    return rows


def query(where):
    return parse_csv(http(BASE + "&d=" + COLS + "&sql=" + where + "&ob=btc&a=csv[,]") or "")


def fetch():
    try:
        return query("btc<%g" % LIMIT)
    except Exception as e:
        print("single query failed:", e, file=sys.stderr)
    rows = []
    for lo, hi in [(-90, -30), (-30, 0), (0, 30), (30, 90)]:
        rows += query("btc<%g%%20and%%20de2000>=%d%%20and%%20de2000<%d" % (LIMIT, lo, hi))
    return rows


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def t_from_class(c):
    c = re.sub(r"^(SAB|SB|SA|IAB|IB|IA)", lambda m: "I" if m.group(1)[0] == "I" else "S", c or "").rstrip("?")
    for k, v in T_CLASS:
        if c == k:
            return v
    return None


def display(name):
    m = re.match(r"^(NGC|IC|UGC|PGC|MCG|CGCG)(\d+)", name)
    if m and name == m.group(0):
        return "%s %d" % (m.group(1), int(m.group(2)))
    m = re.match(r"^ESO(\d+)-(\d+)$", name)
    if m:
        return "ESO %d-%d" % (int(m.group(1)), int(m.group(2)))
    return name


def search_name(g):
    n = display(g["name"].replace("_", " ")).strip()
    return n if n else ("PGC %d" % g["pgc"] if g["pgc"] else "")


def process(rows):
    out, stat = [], {"nosize": 0, "nodist": 0, "mw": 0}
    for r in rows:
        name = r["objname"].strip()
        if name.lower().replace(" ", "") in ("milkyway", "mw", "thegalaxy"):
            stat["mw"] += 1; continue
        ra, de, btc, ld = num(r["al2000"]), num(r["de2000"]), num(r["btc"]), num(r["logd25"])
        if None in (ra, de, btc) or ld is None:
            stat["nosize"] += 1; continue
        mod, v = num(r["modbest"]), num(r["v"])
        z = 0
        if mod is not None and 15 < mod < 40:
            d = 10 ** ((mod - 25) / 5)
        elif v is not None and v > 500:
            d = v / H0; z = 1
        else:
            stat["nodist"] += 1; continue
        d = min(max(d, 0.01), 1500.0)
        diam = math.radians(10 ** ld * 0.1 / 60) * d * 1000
        lr = num(r["logr25"])
        ba = 10 ** -lr if lr is not None else 1.0
        pa = num(r["pa"])
        cls = (r["type"] or "").strip()
        t = num(r["t"])
        if t is None:
            t = t_from_class(cls)
        bve = num(r["bve"])
        flags = (cls.startswith(("SB", "IB"))) | ((cls.startswith(("SAB", "IAB"))) << 1)
        if cls.startswith(("SAB", "IAB")):
            flags &= ~1
        if bve is not None:
            flags |= 4
        flags |= z << 3
        a, b = math.radians(ra * 15.0), math.radians(de)   # HyperLEDA al2000 is decimal HOURS (0..24), not degrees
        out.append(dict(name=name, pgc=int(num(r["pgc"]) or 0), btc=btc, x=d * math.cos(b) * math.cos(a), y=d * math.cos(b) * math.sin(a),
                        z=d * math.sin(b), diam=diam, pa=pa, ba=ba, t=t, flags=flags, bve=bve))
    out.sort(key=lambda g: g["btc"])
    return out, stat


def pack(g):
    t10 = -128 if g["t"] is None else max(-100, min(127, round(g["t"] * 10)))
    pa10 = -1 if g["pa"] is None else int(round(g["pa"] % 180 * 10))
    bve = 255 if g["bve"] is None else max(0, min(250, round((g["bve"] + 0.5) * 100)))
    return struct.pack(FMT, g["x"], g["y"], g["z"], g["diam"], max(-32768, min(32767, round(g["btc"] * 100))), pa10,
                       max(0, min(255, round(g["ba"] * 255))), t10, g["flags"], bve, g["pgc"] & 0xFFFFFFFF)


def main():
    rows = fetch()
    print("fetched", len(rows))
    gal, stat = process(rows)
    print("kept", len(gal), "dropped", stat)
    if len(gal) < MIN_ROWS:
        sys.exit("too few galaxies (%d < %d); not publishing" % (len(gal), MIN_ROWS))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "gal.bin"), "wb") as f:
        for g in gal:
            f.write(pack(g))
    names = []
    for i, g in enumerate(gal):
        if g["name"] in NAMES:
            disp, al = NAMES[g["name"]]
            names.append([i, disp, al])
        elif g["btc"] < 10:
            names.append([i, display(g["name"]), "|".join(filter(None, [display(g["name"]), g["name"], "PGC %d" % g["pgc"] if g["pgc"] else ""]))])
    with open(os.path.join(OUT, "names.json"), "w", encoding="utf-8") as f:
        json.dump(names, f, ensure_ascii=False, separators=(",", ":"))
    with open(os.path.join(OUT, "search.json"), "w", encoding="utf-8") as f:
        json.dump([search_name(g) for g in gal], f, ensure_ascii=False, separators=(",", ":"))
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "n": len(gal), "fmt": FMT, "rec": REC, "limit": LIMIT,
                   "source": "HyperLEDA meandata (btc<%g)" % LIMIT, "h0": H0}, f)
    print("wrote", len(gal), "records,", len(names), "names")


if __name__ == "__main__":
    main()
