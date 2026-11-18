#!/usr/bin/env python3
"""Mars surface maps for Space Radar (USGS Astrogeology, public domain):
  colour : Mars Viking Colorized Global Mosaic 925 m (MDIM 2.1 colour, 23059x11530)
  relief : Mars MGS MOLA DEM Global Mosaic 463 m (int16 metres above R = 3396.19 km)
The Viking colour is graded to the site's previous Mars look (per-channel distribution of the band |lat| < 55 deg, measured
from the previous map: REF_Q, CDF match) -- the albedo pattern, caps and detail are Viking's own.
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
# previous site map, band |lat| < 55 deg: per-channel quantiles (3.125 % steps, 1st..31st) -- the Viking colour is
# CDF-matched to these, so the overall look stays the old one while albedo, caps and detail are Viking's
REF_Q = np.array([
    [83, 96, 107, 117, 126, 134, 141, 147, 152, 156, 161, 165, 169, 173, 177, 181, 185, 189, 193, 197, 201, 204, 208, 211, 214, 217, 220, 223, 226, 231, 239],
    [52, 57, 61, 65, 68, 71, 73, 75, 77, 79, 80, 82, 83, 84, 86, 87, 88, 90, 91, 93, 94, 96, 97, 99, 101, 103, 105, 108, 111, 117, 124],
    [45, 47, 49, 50, 51, 52, 53, 53, 54, 54, 55, 56, 56, 57, 58, 58, 59, 60, 61, 61, 62, 63, 64, 65, 66, 67, 69, 70, 73, 76, 82]], np.float32)
AO = 0.22                                   # strength of the relief occlusion term (see main)


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
    # the mosaic's first column (lon 180) carries no-data: it became a dark pole-to-pole line on the globe -> rebuild
    # every edge column darker than its inner neighbours from the wrap-around average
    for j in (0, 1, -1, -2):
        inner = c[:, 2] if j >= 0 else c[:, -3]
        if c[:, j].mean() < 0.9 * inner.mean():
            c[:, j] = (c[:, 2] + c[:, -3]) * 0.5
    band = c[int(H * 35 / 180):int(H * 145 / 180)].reshape(-1, 3)
    pct = np.linspace(100 / 32, 3100 / 32, 31)
    g = np.empty_like(c)
    for i in range(3):   # CDF match, linear extrapolation beyond the 1st / 31st quantile
        q = np.percentile(band[:, i], pct)
        g[..., i] = np.interp(c[..., i], q, REF_Q[i])
        lo = c[..., i] < q[0]; hi = c[..., i] > q[-1]
        g[..., i][lo] = REF_Q[i][0] + (c[..., i][lo] - q[0]) * (REF_Q[i][1] - REF_Q[i][0]) / max(q[1] - q[0], 1e-3)
        g[..., i][hi] = REF_Q[i][-1] + (c[..., i][hi] - q[-1]) * (REF_Q[i][-1] - REF_Q[i][-2]) / max(q[-1] - q[-2], 1e-3)
    g = g * 0.92 + np.array([0.0, -3.0, 0.0], np.float32)   # rendered next to the old map it read brighter / more orange: a touch darker, less yellow
    # Viking's dark albedo regions carry a blue/violet cast that the per-channel match turns into blue spots: Mars has
    # no blue surface -> blue is capped below green and green below red (the dark regions read dark brown, as before)
    g[..., 1] = np.minimum(g[..., 1], g[..., 0] * 0.62)
    g[..., 2] = np.minimum(g[..., 2], g[..., 1] * 0.80 + 4)
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
    hp = h - cv2.GaussianBlur(h, (0, 0), 10)
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
            'colour': 'Viking colorized mosaic, per-channel CDF-matched to the previous site map (|lat| < 55); ice caps neutral; relief occlusion %g' % AO,
            'files': sorted(f for f in os.listdir(OUT) if f.endswith('.webp')),
            'source': 'USGS Astrogeology: Mars Viking Colorized Global Mosaic 925m, Mars MGS MOLA DEM Global Mosaic 463m',
            'credit': 'NASA/JPL (Viking), NASA/GSFC MOLA Science Team (MGS MOLA), USGS Astrogeology'}
    json.dump(meta, open(os.path.join(OUT, 'meta.json'), 'w'), indent=1)
    print('done in', round(time.time() - t0), 's')


if __name__ == '__main__':
    main()
