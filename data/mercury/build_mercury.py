#!/usr/bin/env python3
"""Mercury surface maps for Cosmos, built from MESSENGER data (USGS Astrogeology, public domain):
  colour : Mercury MESSENGER MDIS Basemap MD3 Color Mosaic Global 665 m (23040x11520, 1000/750/430 nm)
  relief : Mercury MESSENGER USGS DEM Global 665 m v2 (int16 metres above R = 2439.4 km)
The MD3 colour is a strongly stretched product; Mercury's real look is a near-neutral grey, so the luminance carries the
detail and only a small share of the MD3 chroma is kept (the real tan/blue unit differences, faint). Data gaps (no
coverage, ~3 %, mostly the north polar cap) are filled from a smooth inpaint; the normal map still gives them relief.
Outputs (out/): mercury_{color,normal}_{2,4,8}k.webp + meta.json -- equirectangular, lon 0 at the centre, north up;
normal map tangent space x = east, y = north, z = up (OpenGL), true scale.
Needs: pip install numpy pillow opencv-python.   Usage: build_mercury.py [raw_dir]"""
import json, os, sys, time, urllib.request
import numpy as np
import cv2
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
USGS = 'https://planetarymaps.usgs.gov/mosaic/'
COLOR, DEM = 'Mercury_MESSENGER_MDIS_Basemap_MD3Color_Mosaic_Global_665m.tif', 'Mercury_Messenger_USGS_DEM_Global_665m_v2.tif'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
R_KM = 2439.4
SIZES = {8: (8192, 4096), 4: (4096, 2048), 2: (2048, 1024)}
MEAN, STD, CHROMA = 140.0, 30.0, 0.18      # target grey level / contrast (8-bit sRGB), share of MD3 chroma kept
TINT = np.array([1.02, 1.0, 0.965], np.float32)   # faint warm cast of Mercury's real spectrum (reddish slope)


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
    """h: elevation in km on an equirect grid (rows north->south). Returns uint8 RGB tangent-space normals."""
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


def colour(c):
    """c: uint8 RGB MD3 at full size. Returns uint8 RGB 8192x4096, gaps filled, neutralised."""
    W, H = SIZES[8]
    gap = (c.max(axis=2) == 0).astype(np.uint8)
    c8 = cv2.resize(c, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
    g8 = cv2.resize(gap * 255, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    valid = g8 < 0.02
    # gap fill: inpaint a 1k copy (fast, smooth), upsample, blend by the gap fraction (also repairs dark gap rims)
    small = cv2.resize(np.clip(c8, 0, 255).astype(np.uint8), (1024, 512), interpolation=cv2.INTER_AREA)
    smask = (cv2.resize(g8, (1024, 512), interpolation=cv2.INTER_AREA) > 0.02).astype(np.uint8)
    smask = cv2.dilate(smask, np.ones((3, 3), np.uint8))
    fill = cv2.resize(cv2.inpaint(small, smask, 6, cv2.INPAINT_TELEA), (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    a = np.clip(g8 * 1.6, 0, 1)[..., None]
    c8 = c8 * (1 - a) + fill * a
    lum = c8 @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    m, s = float(lum[valid].mean()), float(lum[valid].std())
    lt = (lum - m) / s * STD + MEAN
    chroma = (c8 - lum[..., None]) * CHROMA
    out = (lt[..., None] + chroma) * TINT
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def main():
    raw = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'raw')
    os.makedirs(raw, exist_ok=True); os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    c = cv2.imread(src(raw, COLOR), cv2.IMREAD_UNCHANGED)
    assert c is not None and c.ndim == 3, 'colour TIFF unreadable'
    c8 = colour(c[:, :, 2::-1])                                   # BGR -> RGB
    del c
    d = cv2.imread(src(raw, DEM), cv2.IMREAD_UNCHANGED)
    assert d is not None and d.ndim == 2, 'DEM TIFF unreadable'
    d = d.astype(np.float32) / 1000.0                              # metres -> km
    for k, (w, h) in SIZES.items():
        cc = c8 if k == 8 else cv2.resize(c8, (w, h), interpolation=cv2.INTER_AREA)
        webp(cc, os.path.join(OUT, 'mercury_color_%dk.webp' % k), 90)
        webp(normals(cv2.resize(d, (w, h), interpolation=cv2.INTER_AREA)), os.path.join(OUT, 'mercury_normal_%dk.webp' % k), 92)
    meta = {'v': 1, 'built': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'radius_km': R_KM,
            'projection': 'equirectangular, lon 0 at centre, north up',
            'normal': 'tangent space x=east y=north z=up (OpenGL), true scale from the MESSENGER 665 m DEM',
            'colour': 'MDIS MD3 luminance (mean %g, std %g) + %g of its chroma, faint warm tint; gaps inpainted' % (MEAN, STD, CHROMA),
            'files': sorted(f for f in os.listdir(OUT) if f.endswith('.webp')),
            'source': 'USGS Astrogeology: MESSENGER MDIS MD3 Color Mosaic Global 665m, MESSENGER USGS DEM Global 665m v2',
            'credit': 'NASA/Johns Hopkins University Applied Physics Laboratory/Carnegie Institution of Washington; USGS Astrogeology'}
    json.dump(meta, open(os.path.join(OUT, 'meta.json'), 'w'), indent=1)
    print('done in', round(time.time() - t0), 's')


if __name__ == '__main__':
    main()
