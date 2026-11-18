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
DETAIL_GAIN = 1.25                         # radar fine band vs the old map's (crisper than the old map)
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


def vmed5(a, axis):
    """5-tap median along one axis (rows: axis 0, columns: axis 1), edge-clamped."""
    st = [np.roll(a, k, axis=axis) for k in (-2, -1, 0, 1, 2)]
    return np.median(np.stack(st), axis=0).astype(np.float32)


def delines(s, valid):
    """Magellan swath seams: 1-3 px rows / columns offset from their neighbours over a long run. A pixel whose
    across-line median residual stays coherent along the line for ~150 px is a seam -> replaced by that median.
    Real lineaments are neither perfectly axis-aligned nor that coherent, so they survive."""
    for ax, k in ((0, (151, 1)), (1, (1, 151))):   # horizontal seams: median across rows, coherence along x; then vertical
        med = vmed5(s, ax)
        r = np.where(valid, s - med, 0).astype(np.float32)
        rb = cv2.blur(r, k)
        thr = 0.35 * float(r[valid].std())
        seam = (np.abs(rb) > thr) & valid
        seam = cv2.dilate(seam.astype(np.uint8), np.ones((3, 3) if ax == 0 else (3, 3), np.uint8)) > 0
        s = np.where(seam, med, s)
    return s


def nblur(f, valid, k):
    """box blur of f over VALID pixels only (normalised convolution)."""
    w = valid.astype(np.float32)
    return cv2.blur(np.where(valid, f, 0).astype(np.float32), k) / np.maximum(cv2.blur(w, k), 1e-3)


def destripe(s, valid):
    """Magellan orbit swaths run north-south and differ slightly in gain: narrow vertical brightness bands. The band
    signal = what is left after removing the across-track (x) trend over 41 px, averaged along-track (y) over 401 px;
    it is subtracted. Then the same once more with the axes swapped for the rarer east-west seams."""
    for kx, ky in (((41, 1), (1, 401)), ((1, 41), (401, 1))):
        hp = s - nblur(s, valid, kx)
        band = nblur(hp, valid, ky)
        s = np.where(valid, s - band, s).astype(np.float32)
    return s


def mirror_fill(s, mask):
    """each gap pixel takes the valid pixel mirrored across its nearest gap edge (2*nearest - p): real neighbouring
    radar texture continues into the gap instead of a smooth blur; falls back to the nearest valid pixel."""
    H, W = s.shape
    inv = (mask == 0).astype(np.uint8)                       # 0 where valid -> distance to the nearest VALID pixel
    _, lab = cv2.distanceTransformWithLabels(1 - inv, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    vy, vx = np.nonzero(inv)                                 # row-major order = label order (1..n)
    gy, gx = np.nonzero(mask)
    li = lab[gy, gx] - 1
    ny, nx = vy[li], vx[li]
    ry, rx = np.clip(2 * ny - gy, 0, H - 1), (2 * nx - gx) % W
    ok = inv[ry, rx] > 0
    out = s.copy()
    out[gy, gx] = np.where(ok, s[ry, rx], s[ny, nx])
    return out


def destep(s, valid, R=48):
    """Magellan mosaic tiles carry slightly different gains: straight brightness STEPS along fixed rows / columns.
    The step across each row boundary (s[y+1]-s[y]) averaged along x over 301 px stays near 0 for real terrain and is
    coherent at a tile seam; such steps are cancelled with a linear ramp over +-R px (half on each side)."""
    k = np.concatenate([np.linspace(0.0, 0.5, R, endpoint=False)[1:], [0.5], [-0.5], -np.linspace(0.5, 0.0, R, endpoint=False)[:0:-1][::-1][::-1]])
    k = np.concatenate([np.linspace(1.0 / R, 1.0, R) * 0.5, -np.linspace(1.0, 1.0 / R, R) * 0.5]).astype(np.float32)
    for ax in (0, 1):
        B = 8                                                          # band-to-band step: survives the seam's own gap rows
        def wsum(f):                                                   # f summed over rows y-B+1..y (along ax)
            c = np.cumsum(f, axis=ax, dtype=np.float64)
            sh = np.roll(c, B, axis=ax)
            if ax == 0: sh[:B] = 0
            else: sh[:, :B] = 0
            return (c - sh).astype(np.float32)
        sv, wv = wsum(np.where(valid, s, 0)), wsum(valid.astype(np.float32))
        up = sv / np.maximum(wv, 1e-3)                                 # mean of rows y-B+1..y
        dn = np.roll(sv, -B, axis=ax) / np.maximum(np.roll(wv, -B, axis=ax), 1e-3)   # mean of rows y+1..y+B
        d = dn - up                                                    # step across the boundary y | y+1 (unique peak)
        edge = np.zeros_like(d, dtype=bool)
        if ax == 0: edge[:B] = True; edge[-B:] = True
        else: edge[:, :B] = True; edge[:, -B:] = True
        d = np.where(edge, 0, d).astype(np.float32)
        vv = (wv >= B - 0.5) & (np.roll(wv, -B, axis=ax) >= B - 0.5) & ~edge   # only fully valid windows on both sides
        D = nblur(d, vv, (301, 1) if ax == 0 else (1, 301))
        cov = cv2.blur(vv.astype(np.float32), (301, 1) if ax == 0 else (1, 301))
        D = np.where(cov > 0.6, np.clip(D, -25, 25), 0).astype(np.float32)   # a step needs >= 60 % clean support
        thr = 4.0 * float(np.median(np.abs(D[vv])))
        aD = np.abs(D)
        pk = aD >= cv2.dilate(aD, np.ones((2 * B + 1, 1) if ax == 0 else (1, 2 * B + 1), np.uint8))   # one response per seam
        pk &= aD > np.roll(aD, 1, axis=ax)                              # plateau tie (flat runs): keep the first row
        D = np.where((aD > thr) & pk, D, 0).astype(np.float32)
        C = np.zeros_like(D)                                           # seam between rows p and p+1, step D = below - above
        for t in range(-R + 1, R + 1):                                 # rows p+t: above (t <= 0) +D/2, below (t >= 1) -D/2, linear to 0 at R
            g = 0.5 * (1 - (-t) / R) if t <= 0 else -0.5 * (1 - (t - 1) / R)
            C += np.float32(g) * np.roll(D, t, axis=ax)
        s = np.where(valid, s + C, s).astype(np.float32)
    return s


def fill(v, gap, W, H, detail):
    """v float32 full res, gap bool full res -> float32 WxH: seams removed, THIN gaps (orbit strips, dropout rows)
    inpainted at full output resolution from their neighbours, WIDE gaps (wedges, polar caps) from a low-frequency
    inpaint plus fine noise of the surrounding texture strength."""
    vz = np.where(gap, 0, v).astype(np.float32)
    s = cv2.resize(vz, (W, H), interpolation=cv2.INTER_AREA)
    g = cv2.resize(gap.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    s = np.where(g < 0.999, s / np.maximum(1 - g, 1e-3), 0)          # mean of the VALID part of each output pixel
    mask = cv2.dilate((g > 0.02).astype(np.uint8), np.ones((3, 3), np.uint8))
    valid = mask == 0
    if detail > 0:
        s = delines(s, valid)
        s = destripe(s, valid)
        s = destep(s, valid)
    # wide-gap base: low-frequency inpaint (+ noise)
    lo_w, lo_h = 1024, 512
    small = cv2.resize(np.where(valid, s, 0).astype(np.float32), (lo_w, lo_h), interpolation=cv2.INTER_AREA)
    sg = cv2.resize(valid.astype(np.float32), (lo_w, lo_h), interpolation=cv2.INTER_AREA)
    small = np.where(sg > 0.05, small / np.maximum(sg, 1e-3), 0).astype(np.float32)
    smask = cv2.dilate((sg < 0.98).astype(np.uint8), np.ones((3, 3), np.uint8))
    base = cv2.resize(cv2.inpaint(small, smask, 8, cv2.INPAINT_TELEA), (W, H), interpolation=cv2.INTER_CUBIC)
    if detail > 0:
        rng = np.random.default_rng(7)
        nz = np.zeros((H, W), np.float32)
        for k, amp in ((64, 1.0), (128, 0.6), (256, 0.4), (512, 0.25), (1024, 0.15)):
            nz += cv2.resize(rng.standard_normal((k // 2, k)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC) * amp
        base += nz / nz.std() * detail
    # thin gaps: full-resolution inpaint from the neighbouring pixels
    near = cv2.inpaint(np.where(valid, s, 0).astype(np.float32), mask, 4, cv2.INPAINT_TELEA)
    if detail > 0:   # thin gaps: the neighbours' real radar texture mirrored in (a smooth inpaint read as a soft band)
        mir = mirror_fill(s, mask)
        near = mir - cv2.blur(mir, (9, 9)) + cv2.blur(near, (9, 9))
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)               # px from the nearest valid pixel
    wide = np.clip((dist - 6) / 10, 0, 1)                             # thin (<= ~6 px from data) -> neighbours, wider -> base
    filled = near * (1 - wide) + base * wide
    return np.where(valid, s, filled).astype(np.float32)


def old_colour(raw):
    """The previous site Venus colour map (18 JPEG tiles, 6x3 x 1366 px, listed in old_tiles.txt row-major from the
    north-west). It is the same Magellan mosaic rotated 180 deg (NCC 0.985 against the radar after flipping both axes),
    so it is rotated back to north up / lon 0 at the centre -- then it sits exactly on the Magellan relief."""
    urls = [u.strip() for u in open(os.path.join(HERE, 'old_tiles.txt')) if u.strip()]
    assert len(urls) == 18, 'old_tiles.txt must list 18 tiles'
    T = 1366
    can = np.zeros((3 * T, 6 * T, 3), np.uint8)
    for i, u in enumerate(urls):
        p = os.path.join(raw, 'old_%02d.jpg' % i)
        if not os.path.exists(p):
            req = urllib.request.Request(u.replace('/s1600/', '/s0/'), headers={'User-Agent': 'yzr-net-data-feed'})
            with urllib.request.urlopen(req, timeout=300) as r, open(p + '.part', 'wb') as f:
                f.write(r.read())
            os.replace(p + '.part', p)
        t = cv2.imread(p, cv2.IMREAD_COLOR)
        assert t is not None, 'tile unreadable: %d' % i
        if t.shape[:2] != (T, T):
            t = cv2.resize(t, (T, T), interpolation=cv2.INTER_AREA if t.shape[0] > T else cv2.INTER_CUBIC)
        y, x = divmod(i, 6)
        can[y * T:(y + 1) * T, x * T:(x + 1) * T] = t[:, :, ::-1]          # BGR -> RGB
    W, H = SIZES[8]
    c = cv2.resize(can, (W, H), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(c[::-1, ::-1])


def radar_detail(c8, raw):
    """Sharpen the old colour map with Magellan's own fine detail. The old map is the same mosaic but softer, with
    wavy resampling ripples; the radar is sharp but carries orbit-strip steps and N-S scan streaks. So: broad tones
    stay the old map's, the fine band (below ~15 px) comes from the radar where it has data:
      - gaps are pre-filled with the old map in radar units, so no gap edge enters the detail band;
      - the detail band is radar minus its MEDIAN (15 px): a strip-to-strip brightness step is an edge the median
        keeps, so steps stay out of the detail -- only texture goes in;
      - N-S / E-W streaks are removed from the detail band by an FFT wedge notch (+-3 deg around both axes);
      - the old map's own fine band (ripples) is replaced, feathered at gap edges."""
    H, W = c8.shape[:2]
    old = c8.astype(np.float32)
    L = old @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    r0 = cv2.imread(src(raw, RADAR), cv2.IMREAD_UNCHANGED)
    assert r0 is not None and r0.ndim == 2, 'radar TIFF unreadable'
    gapn = cv2.dilate((r0 < 6).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(np.float32)
    r = cv2.resize(r0.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    valid = cv2.resize(gapn, (W, H), interpolation=cv2.INTER_AREA) < 0.02
    del r0, gapn
    a, b = np.polyfit(L[valid][::50], r[valid][::50], 1)                 # radar ~ a * L + b
    rf = np.where(valid, r, a * L + b).astype(np.float32)
    hp = rf - cv2.medianBlur(np.clip(np.rint(rf), 0, 255).astype(np.uint8), 15).astype(np.float32)
    F = np.fft.rfft2(hp)
    ky = np.fft.fftfreq(H)[:, None] * H / W
    kx = np.fft.rfftfreq(W)[None, :]
    ang = np.degrees(np.arctan2(np.abs(ky), np.abs(kx) + 1e-12))
    m = (np.clip((ang - 3.0) / 1.5, 0, 1) * np.clip((87.0 - ang) / 1.5, 0, 1)).astype(np.float32)
    m[np.hypot(kx, ky) < 0.004] = 1
    hp = np.fft.irfft2(F * m, s=hp.shape).astype(np.float32)
    del F, m, ky, kx, ang
    ohp = L - cv2.medianBlur(np.clip(np.rint(L), 0, 255).astype(np.uint8), 15).astype(np.float32)
    k = float(ohp[valid].std() / hp[valid].std()) * DETAIL_GAIN
    w = cv2.GaussianBlur(cv2.erode(valid.astype(np.float32), np.ones((5, 5), np.uint8)), (0, 0), 2.5)
    Ln = L - w * ohp + w * hp * k
    out = old * np.clip((Ln / np.maximum(L, 1))[..., None], 0.5, 1.8)
    print('radar detail: radar = %.3f*L + %.1f, gain %.2f' % (a, b, k), flush=True)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def main():
    raw = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'raw')
    os.makedirs(raw, exist_ok=True); os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    W, H = SIZES[8]
    c8 = radar_detail(old_colour(raw), raw)   # s29: the previous site colour map (user's choice), relief stays Magellan
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
            'colour': 'previous site Venus map (Magellan-based, old_tiles.txt), rotated 180 deg to north up; fine band (<15 px) from Magellan C3-MDIR radar (median high-pass, FFT destriped)',
            'files': sorted(f for f in os.listdir(OUT) if f.endswith('.webp')),
            'source': 'colour: previous site Venus map (Magellan-based); relief: USGS Astrogeology Venus Magellan Topography Global 4641m v02',
            'credit': 'NASA/JPL (Magellan); USGS Astrogeology'}
    json.dump(meta, open(os.path.join(OUT, 'meta.json'), 'w'), indent=1)
    print('done in', round(time.time() - t0), 's')


if __name__ == '__main__':
    main()
