#!/usr/bin/env python3
"""Moon surface maps for Cosmos, built from NASA data (NASA's Scientific Visualization Studio, CGI Moon Kit,
https://svs.gsfc.nasa.gov/4720 -- LRO LROC WAC Hapke-normalized colour mosaic + LRO LOLA elevation, public domain).

Outputs (out/), equirectangular, 0 deg longitude at the centre column, north up:
  moon_color_{2,4,8}k.webp   colour (2025 LROC edition, 16-bit sRGB source)
  moon_normal_{2,4,8}k.webp  tangent-space normal map (x = east, y = north, z = up; OpenGL convention) computed from the
                             LOLA 64 px/deg elevation at TRUE scale (no exaggeration), on the sphere R = 1737.4 km
  meta.json
Needs: pip install numpy pillow opencv-python.   Usage: build_moon.py [raw_dir]  (raw_dir: already downloaded TIFFs)"""
import json, os, sys, time, urllib.request
import numpy as np
import cv2
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
SVS = 'https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/'
COLOR, DEM = 'lroc_color_16bit_srgb_8k.tif', 'ldem_64.tif'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
R_KM = 1737.4
SIZES = {8: (8192, 4096), 4: (4096, 2048), 2: (2048, 1024)}


# svs.gsfc.nasa.gov omits its Let's Encrypt intermediate (YR2) from the TLS handshake, so a plain verify fails.
# The intermediate is fetched from its AIA URL and accepted ONLY if its SHA-256 matches this pin; it then anchors the
# chain (partial-chain verification), hostname checking stays on.
YR2_URL = 'http://yr2.i.lencr.org/'
YR2_SHA256 = '238b85a0099c65b970477d5724f1a1d475ce5058cffe4efa8733899bdb863c47'   # C=US, O=Let's Encrypt, CN=YR2 (to 2028-09-02)


def ssl_ctx():
    import hashlib, ssl
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(YR2_URL, timeout=60) as r:
            der = r.read()
        if hashlib.sha256(der).hexdigest() == YR2_SHA256:
            ctx.load_verify_locations(cadata=der)
            ctx.verify_flags |= getattr(ssl, 'VERIFY_X509_PARTIAL_CHAIN', 0x80000)
        else:
            print('YR2 pin mismatch -- default trust only', flush=True)
    except Exception as e:
        print('YR2 fetch failed (%s) -- default trust only' % e, flush=True)
    return ctx


def src(raw, name):
    p = os.path.join(raw, name)
    if not os.path.exists(p):
        print('download', name, flush=True)
        req = urllib.request.Request(SVS + name, headers={'User-Agent': 'yzr-net-data-feed'})
        with urllib.request.urlopen(req, timeout=1800, context=ssl_ctx()) as r, open(p + '.part', 'wb') as f:
            while True:
                b = r.read(1 << 22)
                if not b:
                    break
                f.write(b)
        os.replace(p + '.part', p)
    return p


def webp(arr, path, q):
    Image.fromarray(arr).save(path, 'WEBP', quality=q, method=5)
    print('wrote', os.path.basename(path), round(os.path.getsize(path) / 1e6, 2), 'MB', flush=True)


def normals(h):
    """h: elevation in km on an equirect grid (rows north->south). Returns uint8 RGB tangent-space normals."""
    H, W = h.shape
    dlon, dlat = 2 * np.pi / W, np.pi / H
    lat = np.pi / 2 - (np.arange(H) + 0.5) * dlat
    coslat = np.maximum(np.cos(lat), 0.02)[:, None]
    hx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) / (2 * R_KM * coslat * dlon)   # d h / d east
    up, dn = np.vstack([h[:1], h[:-1]]), np.vstack([h[1:], h[-1:]])
    hy = (up - dn) / (2 * R_KM * dlat)                                                    # d h / d north
    n = np.dstack([-hx, -hy, np.ones_like(h)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return np.clip(np.rint((n * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8)


def main():
    raw = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'raw')
    os.makedirs(raw, exist_ok=True); os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    c = cv2.imread(src(raw, COLOR), cv2.IMREAD_UNCHANGED)
    assert c is not None and c.ndim == 3, 'colour TIFF unreadable'
    c = c[:, :, 2::-1]                                               # BGR -> RGB
    c8 = np.clip(np.rint(c.astype(np.float32) / (65535.0 if c.dtype == np.uint16 else 255.0) * 255), 0, 255).astype(np.uint8)
    del c
    d = cv2.imread(src(raw, DEM), cv2.IMREAD_UNCHANGED)
    assert d is not None and d.ndim == 2, 'elevation TIFF unreadable'
    d = d.astype(np.float32)
    for k, (w, h) in SIZES.items():
        cc = c8 if c8.shape[1] == w else cv2.resize(c8, (w, h), interpolation=cv2.INTER_AREA)
        webp(cc, os.path.join(OUT, 'moon_color_%dk.webp' % k), 90)
        hh = cv2.resize(d, (w, h), interpolation=cv2.INTER_AREA)
        webp(normals(hh), os.path.join(OUT, 'moon_normal_%dk.webp' % k), 92)
    meta = {'v': 1, 'built': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'radius_km': R_KM,
            'projection': 'equirectangular, lon 0 at centre, north up',
            'normal': 'tangent space x=east y=north z=up (OpenGL), true scale from LOLA 64 ppd',
            'files': sorted(f for f in os.listdir(OUT) if f.endswith('.webp')),
            'source': 'NASA SVS 4720 CGI Moon Kit: LROC WAC Hapke-normalized colour (2025) + LOLA ldem_64',
            'credit': "NASA's Scientific Visualization Studio; LRO LROC (NASA/GSFC/Arizona State University); LRO LOLA (NASA/GSFC)"}
    json.dump(meta, open(os.path.join(OUT, 'meta.json'), 'w'), indent=1)
    print('done in', round(time.time() - t0), 's')


if __name__ == '__main__':
    main()
