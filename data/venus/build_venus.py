#!/usr/bin/env python3
"""Venus SURFACE maps for Space Radar, built from Magellan data (USGS Astrogeology, public domain):
  surface: Venus Magellan C3-MDIR Global Mosaic 2025 m (radar backscatter, 18775x9388)
  relief : Venus Magellan Topography Global 4641 m v02 (int16 metres above R = 6051.8 km, -32768 = no data)
The surface under the clouds has no visible-light map, so the radar mosaic carries the detail and is rendered with the
site's orange surface palette (a brightness -> colour table measured from the previous map, see LUT below).
Magellan left gaps (orbit strips, polar caps, ~8 %): filled with a smooth inpaint plus fine noise of the surrounding
texture strength, so a gap reads as unresolved terrain, not as a black stripe; the normal map is flat-ish there.
Outputs (out/): venus_{color,normal}_{2,4,8}k.webp + meta.json -- equirectangular, lon 0 at the centre, north up;
normal map tangent space x = east, y = north, z = up (OpenGL), true scale.
Needs: pip install numpy pillow opencv-python.   Usage: build_venus.py [raw_dir]"""
import json, os, sys, time, urllib.request
import numpy as np
import cv2
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
USGS = 'https://planetarymaps.usgs.gov/mosaic/'
RADAR, DEM = 'Venus_Magellan_C3-MDIR_Global_Mosaic_2025m.tif', 'Venus_Magellan_Topography_Global_4641m_v02.tif'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
R_KM = 6051.8
SIZES = {8: (8192, 4096), 4: (4096, 2048), 2: (2048, 1024)}
# previous site map: luminance quantiles (0..100 % in 32 steps) and luminance -> mean RGB table
LUM_Q = [16.2, 78.7, 84.9, 89.6, 93.9, 97.8, 101.2, 104.2, 106.7, 109.2, 111.4, 113.3, 114.9, 116.8, 118.7, 120.3, 122.1,
         124.0, 125.8, 127.8, 129.8, 132.1, 134.5, 137.2, 140.3, 143.7, 147.4, 151.9, 157.5, 164.5, 174.5, 190.7, 250.8]
LUT_L = [34.5, 41.8, 49.2, 56.5, 63.8, 71.2, 78.5, 85.8, 93.2, 100.5, 107.8, 115.2, 122.5, 129.8, 137.2, 144.5, 151.8,
         159.2, 166.5, 173.8, 181.1, 188.5, 195.8, 203.1, 210.5, 217.8, 225.1, 232.5, 239.8, 247.1]
LUT = [[69, 29, 6], [80, 34, 8], [94, 41, 10], [107, 47, 13], [120, 53, 15], [132, 59, 18], [145, 65, 20], [156, 72, 22],
       [167, 78, 24], [176, 86, 27], [185, 93, 29], [193, 101, 32], [199, 108, 36], [205, 116, 39], [210, 125, 44],
       [214, 133, 49], [219, 142, 54], [223, 150, 59], [227, 159, 64], [231, 167, 70], [235, 176, 75], [240, 184, 79],
       [244, 192, 85], [248, 201, 93], [251, 209, 106], [252, 217, 127], [252, 224, 151], [253, 232, 174],
       [254, 239, 194], [254, 246, 209]]


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


def fill(v, gap, W, H, detail):
    """v float32 full res, gap bool full res -> float32 WxH with gaps filled (low-frequency inpaint + fine noise)."""
    vz = np.where(gap, 0, v).astype(np.float32)
    s = cv2.resize(vz, (W, H), interpolation=cv2.INTER_AREA)
    g = cv2.resize(gap.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    s = np.where(g < 0.999, s / np.maximum(1 - g, 1e-3), 0)          # mean of the VALID part of each output pixel
    lo_w, lo_h = 1024, 512
    small = cv2.resize(s, (lo_w, lo_h), interpolation=cv2.INTER_AREA)
    smask = (cv2.resize(g, (lo_w, lo_h), interpolation=cv2.INTER_AREA) > 0.02).astype(np.uint8)
    smask = cv2.dilate(smask, np.ones((3, 3), np.uint8))
    lo, hi = float(small[smask == 0].min()), float(small[smask == 0].max())
    s8 = np.clip(np.rint((small - lo) / (hi - lo) * 255), 0, 255).astype(np.uint8)   # low-frequency base only
    inp = cv2.inpaint(s8, smask, 8, cv2.INPAINT_TELEA).astype(np.float32) / 255 * (hi - lo) + lo
    base = cv2.resize(inp, (W, H), interpolation=cv2.INTER_CUBIC)
    if detail > 0:   # fine fractal noise so a filled gap carries texture of the surrounding strength
        rng = np.random.default_rng(7)
        nz = np.zeros((H, W), np.float32)
        for k, amp in ((64, 1.0), (128, 0.6), (256, 0.4), (512, 0.25), (1024, 0.15)):
            nz += cv2.resize(rng.standard_normal((k // 2, k)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC) * amp
        base += nz / nz.std() * detail
    a = np.clip(g * 1.4, 0, 1)
    return s * (1 - a) + base * a


def main():
    raw = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'raw')
    os.makedirs(raw, exist_ok=True); os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    W, H = SIZES[8]
    r = cv2.imread(src(raw, RADAR), cv2.IMREAD_UNCHANGED)
    assert r is not None and r.ndim == 2, 'radar TIFF unreadable'
    gap = cv2.dilate((r < 6).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0   # + the dark seam pixels at strip edges
    rf = r.astype(np.float32)
    hf = rf - cv2.GaussianBlur(rf, (0, 0), 6)
    detail = float(hf[~gap].std()) * 1.5
    del hf
    v = fill(rf, gap, W, H, detail)
    del r, rf, gap
    # histogram-match the radar to the previous map's luminance, then colour through its palette
    qs = np.percentile(v, np.linspace(0, 100, len(LUM_Q)))
    lum = np.interp(v, qs, LUM_Q).astype(np.float32)
    lut = np.array(LUT, np.float32)
    col = np.dstack([np.interp(lum, LUT_L, lut[:, i]) for i in range(3)])
    c8 = np.clip(np.rint(col), 0, 255).astype(np.uint8)
    del v, lum, col
    d = cv2.imread(src(raw, DEM), cv2.IMREAD_UNCHANGED)
    assert d is not None and d.ndim == 2, 'DEM TIFF unreadable'
    dgap = d <= -32000
    h = fill(d.astype(np.float32), dgap, W, H, 0) / 1000.0          # metres -> km, gaps smooth
    for k, (w, hh) in SIZES.items():
        cc = c8 if k == 8 else cv2.resize(c8, (w, hh), interpolation=cv2.INTER_AREA)
        webp(cc, os.path.join(OUT, 'venus_color_%dk.webp' % k), 90)
        hk = h if k == 8 else cv2.resize(h, (w, hh), interpolation=cv2.INTER_AREA)
        webp(normals(hk), os.path.join(OUT, 'venus_normal_%dk.webp' % k), 92)
    meta = {'v': 1, 'built': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'radius_km': R_KM,
            'projection': 'equirectangular, lon 0 at centre, north up',
            'normal': 'tangent space x=east y=north z=up (OpenGL), true scale from Magellan topography 4641 m',
            'colour': 'Magellan C3-MDIR radar backscatter, histogram-matched and coloured with the previous site palette; gaps inpainted',
            'files': sorted(f for f in os.listdir(OUT) if f.endswith('.webp')),
            'source': 'USGS Astrogeology: Venus Magellan C3-MDIR Global Mosaic 2025m, Venus Magellan Topography Global 4641m v02',
            'credit': 'NASA/JPL (Magellan); USGS Astrogeology'}
    json.dump(meta, open(os.path.join(OUT, 'meta.json'), 'w'), indent=1)
    print('done in', round(time.time() - t0), 's')


if __name__ == '__main__':
    main()
