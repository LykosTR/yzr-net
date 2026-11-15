"""Daily catalogue of every numbered asteroid (NASA/JPL SBDB Query API) for the browser.

Output (data/asteroids/out):
  index.txt.gz   line k (0-based) = asteroid number k+1:  "<name>\\t<H>"  (H empty when unknown, empty line = no data).
                 name is the SBDB full name without its number: "Eros", or a provisional designation in
                 parentheses: "(2011 PX10)".
  el/<k>.json    element shard k = numbers k*1000+1 .. k*1000+1000:
                 {"at": ms, "first": k*1000+1, "fields": [...], "rows": [row | null, ...]}
                 row = [a au, e, i, om, w, ma deg (at epoch), n deg/day, epoch JD TDB, H, diameter km, albedo,
                        spectral class, rotation period h, pha 0/1, neo 0/1]
Client propagation: two-body Kepler, M(t) = ma + n * (t - epoch)."""
import gzip, json, os, re, sys, time, urllib.parse, urllib.request

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
SBDB = "https://ssd-api.jpl.nasa.gov/sbdb_query.api?"
UA = {"User-Agent": "yzr-net-catalog/1.0"}
FIELDS = "spkid,full_name,a,e,i,om,w,ma,n,epoch,H,diameter,albedo,spec_B,spec_T,rot_per,pha,neo"
ROW_FIELDS = ["a", "e", "i", "om", "w", "ma", "n", "epoch", "H", "diameter", "albedo", "spec", "rot", "pha", "neo"]
SHARD = 1000
PAGE = 50000
MIN_ROWS = 850000


def http(url, tries=6, timeout=240):
    for k in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read().decode("utf-8", "replace")
        except Exception as e:
            print("retry", k, str(e)[:100], url[:110])
            time.sleep(3 + 5 * k)
    return None


def split_name(full):
    m = re.match(r"\s*(\d+)\s+(.*)$", str(full))
    num, rest = int(m.group(1)), m.group(2).strip()
    named = re.match(r"(.+?)\s+\([^()]*\)$", rest)
    if named and not rest.startswith("("):
        return num, named.group(1).strip()
    return num, rest


def _f(x):
    return None if x in (None, "") else float(x)


def _sig(x, d):
    return None if x is None else float("%.*g" % (d, x))


def _rnd(x, d):
    return None if x is None else round(x, d)


def make_row(r):
    a, e, i, om, w, ma, n, ep, H, dia, alb = (_f(v) for v in r[2:13])
    spec = r[13] or r[14] or None
    return [_sig(a, 9), _sig(e, 9), _rnd(i, 6), _rnd(om, 6), _rnd(w, 6), _rnd(ma, 6), _sig(n, 9), ep,
            _rnd(H, 2), _rnd(dia, 3), _rnd(alb, 3), spec, _rnd(_f(r[15]), 4), 1 if r[16] == "Y" else 0, 1 if r[17] == "Y" else 0]


def build(by_num, count, at=0):
    lines, shards = [], {}
    for num in range(1, count + 1):
        k, j = (num - 1) // SHARD, (num - 1) % SHARD
        sh = shards.get(k)
        if sh is None:
            sh = shards[k] = {"at": at, "first": k * SHARD + 1, "fields": ROW_FIELDS, "rows": []}
        got = by_num.get(num)
        if got is None:
            lines.append("")
            sh["rows"].append(None)
            continue
        name, row = got
        lines.append("%s\t%s" % (name, "" if row[8] is None else ("%g" % row[8])))
        sh["rows"].append(row)
    return "\n".join(lines), shards


def fetch_all():
    by_num, start, total = {}, 0, None
    while total is None or start < total:
        q = {"fields": FIELDS, "sb-kind": "a", "sb-ns": "n", "full-prec": "1", "limit": str(PAGE), "limit-from": str(start)}
        txt = http(SBDB + urllib.parse.urlencode(q))
        if not txt:
            sys.exit("SBDB page %d failed" % start)
        j = json.loads(txt)
        total = int(j.get("count", 0))
        data = j.get("data") or []
        if not data:
            break
        for r in data:
            if None in r[2:10]:
                continue
            num, name = split_name(r[1])
            by_num[num] = (name, make_row(r))
        start += len(data)
        print("page", start, "/", total)
    return by_num


def main():
    t0 = time.time()
    by_num = fetch_all()
    if len(by_num) < MIN_ROWS:
        sys.exit("only %d asteroids, refusing to publish" % len(by_num))
    count = max(by_num)
    idx, shards = build(by_num, count, int(time.time() * 1000))
    os.makedirs(os.path.join(OUT, "el"), exist_ok=True)
    with open(os.path.join(OUT, "index.txt.gz"), "wb") as f:
        f.write(gzip.compress(idx.encode("utf-8"), 9, mtime=0))
    for k, sh in shards.items():
        with open(os.path.join(OUT, "el", "%d.json" % k), "w") as f:
            json.dump(sh, f, separators=(",", ":"))
    print("ok asteroids=%d max=%d shards=%d index=%d bytes in %.0fs" % (
        len(by_num), count, len(shards), os.path.getsize(os.path.join(OUT, "index.txt.gz")), time.time() - t0))


if __name__ == "__main__":
    main()
