#!/usr/bin/env python3
"""Body colour maps for Cosmos, re-hosted from the site's original Blogger tile sets (tiles.json).
Each body's tiles are slices of one equirectangular map (north up, lon 0 at the centre); cell edges sit at
round(i * W / n), exactly as the old in-browser assembler placed them.
Outputs (out/<key>/): <key>_color_{2,4,8}k.webp (only tiers <= the source width) + meta.json.
Needs: pip install numpy pillow.   Usage: build_textures.py [raw_dir] [key ...]"""
import io, json, os, re, sys, time, urllib.request
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
TIERS = (2, 4, 8)


def fetch(url, path):
    if os.path.exists(path):
        return open(path, 'rb').read()
    full = re.sub(r'/s\d+(-[a-z0-9-]+)?/', '/s0/', url)
    err = None
    for u in (full, url, full, url):
        try:
            req = urllib.request.Request(u, headers={'User-Agent': 'yzr-net-data-feed'})
            with urllib.request.urlopen(req, timeout=300) as r:
                b = r.read()
            Image.open(io.BytesIO(b)).verify()
            open(path, 'wb').write(b)
            return b
        except Exception as e:
            err = e
            time.sleep(3)
    raise RuntimeError('tile failed: %s (%s)' % (url, err))


def assemble(key, spec, raw):
    W, H = spec['canvas']
    rows = spec['rows']
    nr, nc = len(rows), len(rows[0])
    fx = [round(c * W / nc) for c in range(nc + 1)]
    fy = [round(r * H / nr) for r in range(nr + 1)]
    can = Image.new('RGBA', (W, H))
    alpha = False
    for r in range(nr):
        for c in range(nc):
            b = fetch(rows[r][c], os.path.join(raw, '%s_%d_%d' % (key, r, c)))
            t = Image.open(io.BytesIO(b))
            t = t.convert('RGBA')
            alpha = alpha or t.getextrema()[3][0] < 255
            w, h = fx[c + 1] - fx[c], fy[r + 1] - fy[r]
            if t.size != (w, h):
                t = t.resize((w, h), Image.LANCZOS)
            can.paste(t, (fx[c], fy[r]))
    return can if alpha else can.convert('RGB')


def main():
    raw = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'raw')
    only = set(sys.argv[2:])
    os.makedirs(raw, exist_ok=True)
    man = json.load(open(os.path.join(HERE, 'tiles.json')))
    built = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    for key, spec in man.items():
        if only and key not in only:
            continue
        t0 = time.time()
        d = os.path.join(OUT, key)
        os.makedirs(d, exist_ok=True)
        if 'files' in spec:   # single maps copied as-is (e.g. the shared asteroid rock PBR set)
            files = []
            for name, url in spec['files'].items():
                im = Image.open(io.BytesIO(fetch(url, os.path.join(raw, '%s_%s' % (key, name)))))
                im = im.convert('L' if im.mode in ('L', 'I', 'I;16', 'F') else 'RGB')
                f = '%s_%s.webp' % (key, name)
                im.save(os.path.join(d, f), 'WEBP', quality=92, method=6)
                files.append(f)
            json.dump({'v': 1, 'built': built, 'files': files, 'source': 'previous Cosmos texture (see tiles.json)'},
                      open(os.path.join(d, 'meta.json'), 'w'), indent=1)
            print(key, files, flush=True)
            continue
        img = assemble(key, spec, raw)
        W, H = img.size
        files = []
        for k in TIERS:
            w = k * 1024
            if w > W and k != TIERS[0]:
                break
            w = min(w, W)
            h = max(1, round(H * w / W))
            im = img if (w, h) == img.size else img.resize((w, h), Image.LANCZOS)
            f = '%s_color_%dk.webp' % (key, k)
            im.save(os.path.join(d, f), 'WEBP', quality=90, method=6)
            files.append(f)
        json.dump({'v': 1, 'built': built, 'size': [W, H], 'tiers': [int(f.split('_')[-1][:-6]) for f in files],
                   'files': files, 'source': 'previous Cosmos texture (Blogger tiles, see tiles.json)'},
                  open(os.path.join(d, 'meta.json'), 'w'), indent=1)
        print(key, W, H, img.mode, files, round(time.time() - t0), 's', flush=True)


if __name__ == '__main__':
    main()
