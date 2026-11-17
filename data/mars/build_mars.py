#!/usr/bin/env python3
"""Mars surface maps for Space Radar (USGS Astrogeology, public domain):
  colour : Mars Viking Colorized Global Mosaic 925 m (MDIM 2.1 colour, 23059x11530)
  relief : Mars MGS MOLA DEM Global Mosaic 463 m (int16 metres above R = 3396.19 km)
The Viking colour is graded to the site's previous Mars look (per-channel mean / std of the band |lat| < 55 deg, measured
from the previous map: REF_MEAN / REF_STD) -- the albedo pattern, caps and detail are Viking's own.
Outputs (out/): mars_{color,normal}_{2,4,8}k.webp + meta.json -- equirectangular, lon 0 at the centre, north up;
normal map tangent space x = east, y = north, z = up (OpenGL), true scale.
Needs: pip install numpy pillow opencv-python.   Usage: build_mars.py [raw_dir]"""
import json, os, sys, time, urllib.request
import numpy as np
import cv2
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
USGS = 'https://planetarymaps.usgs.gov/mosaic/'
COLOR, DEM = 'Mars_Viking_ClrMosaic_global_925m.tif', 'Mars_MGS_MOLA_DEM_mosaic_global_463m.tif'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
R_KM = 3396.19
SIZES = {8: (8192, 4096), 4: (4096, 2048), 2: (2048, 1024)}
REF_MEAN = np.array([174.9, 87.2, 59.8], np.float32)
REF_STD = np.array([43.8, 18.9, 10.0], np.float32)
AO = 0.35                                    # strength of the relief occlusion term (see main)


def src(raw, name):
    p = os.path.join(raw, name)
    if not os.path.exists(p):
        print('download', name, flush=True)
        req = urllib.request.Request(USGS + name, headers={'User-Agent': 'yzr-net-data-feed'})
        with urllib.request.urlopen(req, timeout=1800) as r, open(p + '.part', 'wb') as f:
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
    H, W = h.shape
    dlon, dlat = 2 * np.pi / W, np.pi / H
    lat = np.pi / 2 - (np.arange(H) + 0.5) * dlat
    coslat = np.maximum(np.cos(lat), 0.02)[:, None]
    hx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) / (2 * R_KM * coslat * dlon)
    up, dn = np.vstack([h[:1], h[:-1]]), np.vstack([h[1:], h[-1:]])
    hy = (up - dn) / (2 * R_KM * dlat)
    n = np.dstack([-hx, -hy, np.ones_like(h)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return np.clip(np.rint((n * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8)


def main():
    raw = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'raw')
    os.makedirs(raw, exist_ok=True); os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    W, H = SIZES[8]
    c = cv2.imread(src(raw, COLOR), cv2.IMREAD_UNCHANGED)
    assert c is not None and c.ndim == 3, 'colour TIFF unreadable'
    c = cv2.resize(c[:, :, 2::-1], (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)   # BGR -> RGB
    band = c[int(H * 35 / 180):int(H * 145 / 180)].reshape(-1, 3)
    m, s = band.mean(0), band.std(0)
    g = (c - m) / s * REF_STD + REF_MEAN
    # ice caps (bright, unsaturated in Viking) keep their own near-white instead of the red grade
    mx, mn = c.max(axis=2), c.min(axis=2)
    sat, lum = 1 - mn / np.maximum(mx, 1), c.mean(axis=2)
    w = (np.clip((0.40 - sat) / 0.15, 0, 1) * np.clip((lum - 140) / 40, 0, 1))[..., None]
    ice = np.array([1.0, 0.985, 0.97], np.float32) * (215 + 38 * np.clip((lum - 140) / 80, 0, 1))[..., None]   # neutral (Viking caps carry a lilac cast)
    c8 = np.clip(np.rint(g * (1 - w) + ice * w), 0, 255).astype(np.uint8)
    del g, mx, mn, sat, lum, w, ice
    del c, band
    d = cv2.imread(src(raw, DEM), cv2.IMREAD_UNCHANGED)
    assert d is not None and d.ndim == 2, 'DEM TIFF unreadable'
    h = cv2.resize(d.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA) / 1000.0    # metres -> km
    del d
    # relief occlusion: canyons and crater floors sit darker, rims and ridges lighter (light-independent, like ambient
    # occlusion) -- keeps Valles Marineris & co readable at full phase, where the normal map has nothing to shade
    hp = h - cv2.GaussianBlur(h, (0, 0), 6)
    ao = np.clip(1 + AO * np.tanh(hp / (3 * float(hp.std()))), 0.55, 1.25)[..., None]
    c8 = np.clip(np.rint(c8.astype(np.float32) * ao), 0, 255).astype(np.uint8)
    del hp, ao
    for k, (w, hh) in SIZES.items():
        cc = c8 if k == 8 else cv2.resize(c8, (w, hh), interpolation=cv2.INTER_AREA)
        webp(cc, os.path.join(OUT, 'mars_color_%dk.webp' % k), 90)
        hk = h if k == 8 else cv2.resize(h, (w, hh), interpolation=cv2.INTER_AREA)
        webp(normals(hk), os.path.join(OUT, 'mars_normal_%dk.webp' % k), 92)
    meta = {'v': 1, 'built': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'radius_km': R_KM,
            'projection': 'equirectangular, lon 0 at centre, north up',
            'normal': 'tangent space x=east y=north z=up (OpenGL), true scale from MOLA 463 m',
            'colour': 'Viking colorized mosaic graded to mean %s / std %s (|lat| < 55)' % (REF_MEAN.tolist(), REF_STD.tolist()),
            'files': sorted(f for f in os.listdir(OUT) if f.endswith('.webp')),
            'source': 'USGS Astrogeology: Mars Viking Colorized Global Mosaic 925m, Mars MGS MOLA DEM Global Mosaic 463m',
            'credit': 'NASA/JPL (Viking), NASA/GSFC MOLA Science Team (MGS MOLA), USGS Astrogeology'}
    json.dump(meta, open(os.path.join(OUT, 'meta.json'), 'w'), indent=1)
    print('done in', round(time.time() - t0), 's')


if __name__ == '__main__':
    main()
