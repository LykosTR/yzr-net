#!/usr/bin/env python3
"""Surface maps of the large moons for Space Radar (USGS Astrogeology mission mosaics, public domain):
  io       : Galileo SSI + Voyager global mosaic, ClrMerge 1 km (real colour)
  europa   : Voyager + Galileo SSI global mosaic 500 m (grey)
  ganymede : Voyager + Galileo SSI global mosaic 1 km (grey detail) + global colour mosaic 1435 m (chroma)
  callisto : Voyager + Galileo SSI global mosaic 1 km (grey)
  titan    : Cassini ISS P19658 global mosaic 4 km (938 nm, grey)
  dione    : Cassini + Voyager global mosaic 154 m (grey)
  iapetus  : Cassini + Voyager global mosaic 783 m (grey)
No global DEM of these bodies exists on the USGS servers (only regional Galileo DTMs): the normal maps are flat
(128, 128, 255) -- the client's micro-detail layer attaches to them; relief is never derived from image brightness.
Pipeline per body (all px figures at the top tier, 8k; Titan's top tier is 4k = its source resolution):
  luminance L + no-data mask M (area-resampled from the source) -> rolled onto the previous site map's longitude when
  the two correlate (NCC >= 0.35; the client rotation was tuned to it) -> baked harsh lighting flattened (soft clamp
  to the local median where the band-pass std is > 2x the body median) -> low-resolution areas (map Q: fine / coarse
  DoG energy ratio) de-ringed (5x5 medians at 2k / 1k) -> gaps filled (wavy feathered ramp sized to the hole, TELEA
  ring inpaint) -> polar caps (|lat| > 65) de-rippled in an azimuthal grid -> multiplicative grain from 3-D noise on
  the sphere (no 1/cos lat streaks) on fills, low-Q terrain and caps -> colour (Io: own colour; Ganymede: low-passed
  colour-mosaic chroma; others: natural-colour LUT on the low-passed luminance x detail ratio) -> luminance mean of the
  previous map (tone power when brightening, gain when darkening; Io plain gain) -> no blue cast.
Outputs (out/<key>/): <key>_{color,normal}_{2,4,8}k.webp + meta.json -- equirectangular, north up, east to the right.
Needs: pip install numpy pillow opencv-python.   Usage: build_moons.py [raw_dir] [--only k1,k2] [--size W] [--out dir]"""
import json, os, sys, time, urllib.request
import numpy as np
import cv2
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
USGS = 'https://planetarymaps.usgs.gov/mosaic/'
OLD = 'https://raw.githubusercontent.com/LykosTR/yzr-net/data/textures/%s/%s_color_4k.webp'   # previous site maps
HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = {8: (8192, 4096), 4: (4096, 2048), 2: (2048, 1024)}
NCC_MIN = 0.35
NCC_LP_MIN = 0.5                            # low-pass match that still sets the orientation of an artistic previous map
GALILEO = 'NASA/JPL/DLR (Galileo SSI), NASA/JPL (Voyager), USGS Astrogeology'
CASSINI = 'NASA/JPL/Space Science Institute (Cassini ISS), NASA/JPL (Voyager), USGS Astrogeology'
# centre: longitude at the source image's centre column (0 or 180; every source is drawn east to the right);
# mode: own = the mosaic's own colour, chroma = luminance from `grey` x low-passed chroma of `color`, lut = colour LUT
# on the low-passed luminance (stops: normalised luminance -> RGB); neutral: blue capped at max(R, G) instead of G + 6
BODIES = {
    'io': dict(name='Io', radius_km=1821.46, color='Io_GalileoSSI-Voyager_Global_Mosaic_ClrMerge_1km.tif', centre=0,
               mode='own', sat=0.85, linear=True, base='own', pole_flat=75, top=8, credit=GALILEO,
               source='USGS Astrogeology: Io Galileo SSI / Voyager Global Mosaic ClrMerge 1 km'),
    'europa': dict(name='Europa', radius_km=1562.09, grey='Europa_Voyager_GalileoSSI_global_mosaic_500m.tif', centre=180,
                   mode='lut', neutral=True, top=8, credit=GALILEO,
                   lut=[(0, (120, 92, 70)), (0.3, (170, 140, 112)), (0.6, (214, 200, 182)), (1, (240, 236, 228))],
                   source='USGS Astrogeology: Europa Voyager / Galileo SSI Global Mosaic 500 m'),
    'ganymede': dict(name='Ganymede', radius_km=2632.345, grey='Ganymede_Voyager_GalileoSSI_global_mosaic_1km.tif',
                     color='Ganymede_Voyager_GalileoSSI_Global_ClrMosaic_1435m.tif', centre=180, mode='chroma', top=8,
                     credit=GALILEO, source='USGS Astrogeology: Ganymede Voyager / Galileo SSI Global Mosaic 1 km, '
                                            'Ganymede Voyager / Galileo SSI Global Color Mosaic 1435 m'),
    'callisto': dict(name='Callisto', radius_km=2409.3, grey='Callisto_Voyager_GalileoSSI_global_mosaic_1km.tif', centre=180,
                     mode='lut', top=8, credit=GALILEO,
                     lut=[(0, (52, 46, 40)), (0.3, (88, 78, 68)), (0.6, (140, 130, 118)), (1, (222, 218, 210))],
                     source='USGS Astrogeology: Callisto Voyager / Galileo SSI Global Mosaic 1 km'),
    # disabled: the ISS frame patchwork shows as seams and the previous map's polar lakes would be lost
    'titan': dict(name='Titan', disabled=True, radius_km=2575.0, grey='Titan_ISS_P19658_Mosaic_Global_4km.tif', centre=180, mode='lut',
                  top=4, credit='NASA/JPL/Space Science Institute (Cassini ISS), USGS Astrogeology',
                  lut=[(0, (52, 36, 20)), (0.4, (120, 84, 44)), (0.7, (178, 130, 70)), (1, (214, 170, 104))],
                  source='USGS Astrogeology: Titan Cassini ISS P19658 Global Mosaic 4 km (938 nm)'),
    'dione': dict(name='Dione', radius_km=563.0, grey='Dione_Cassini_Voyager_mosaic_global_154m.tif', centre=0, mode='lut',
                  neutral=True, top=8, credit=CASSINI,
                  lut=[(0, (120, 118, 114)), (0.5, (184, 182, 178)), (1, (236, 236, 234))],
                  source='USGS Astrogeology: Dione Cassini / Voyager Global Mosaic 154 m'),
    'iapetus': dict(name='Iapetus', radius_km=736.0, grey='Iapetus_Cassini_Voyager_mosaic_global_783m.tif', centre=0,
                    mode='lut', neutral=True, pole_flat=75, top=8, credit=CASSINI,
                    lut=[(0, (48, 36, 28)), (0.25, (86, 64, 48)), (0.5, (160, 148, 132)), (1, (232, 228, 220))],
                    source='USGS Astrogeology: Iapetus Cassini / Voyager Global Mosaic 783 m'),
}
# all px at the top tier (8k); scaled with the working width
MASK_COV, MASK_ERODE = 0.99, 5              # source pixel valid above this coverage; erosion drops the partial rim
RAMP, PERT = 128, 48                        # gap fill ramp, starting 0..2*PERT px inside the data (wavy, no straight edge)
GRAIN_GAP = 0.6                             # gap grain: fraction of the local relative std of nearby valid data
GRAIN_Q = 0.6                               # low-resolution / polar grain: fraction of the median relative detail std (Q > 0.8)
SIG_LUT = 6                                 # low-pass for the colour LUT / detail ratio
SIG_CHROMA = 16                             # low-pass for Ganymede's colour-mosaic chroma
SIG_FLAT, FLAT_LO, FLAT_HI, FLAT_K = 40, 1.5, 2.5, 0.3   # harsh-light flatten: median scale, weight ramp (x body median), clamp
Q_SIG, Q_LO, Q_HI = 96, 0.4, 0.9            # resolution map: smoothing, smoothstep on the fine/coarse energy ratio / reference
DET_MAX = 0.10                              # cap of the relative detail level that sets the low-resolution grain
RES_MAX, RESP_MIN = 1.5, 0.05                # old-map base: local registration limit (px @2k) and minimum correlation peak
DETAIL_W = 0.8                              # old-map base: mission detail weight x Q
CHROMA_SAT = 0.5                            # Ganymede: colour-mosaic chroma gain (its frames carry per-frame tints)
KNEE = 0.9                                  # soft highlight knee (fraction of 255)
POLE_ROWS = 6


def src(raw, name, url=None):
    p = os.path.join(raw, name)
    if not os.path.exists(p):
        print('download', name, flush=True)
        req = urllib.request.Request(url or USGS + name, headers={'User-Agent': 'yzr-net-data-feed'})
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


def tiff_map(path):
    """uncompressed 8-bit strip GeoTIFF (chunky or planar) -> read-only memmap (C, H, W)"""
    im = Image.open(path)
    t = im.tag_v2
    W, H = im.size
    spp, planar = t.get(277, 1), t.get(284, 1)
    off, cnt = np.asarray(t[273], np.int64), np.asarray(t[279], np.int64)
    assert t.get(259, 1) == 1 and all(b == 8 for b in np.atleast_1d(t.get(258))), 'TIFF not 8-bit uncompressed'
    assert np.all(off[1:] == off[:-1] + cnt[:-1]) and cnt.sum() == W * H * spp, 'TIFF strips not contiguous'
    im.close()
    if spp == 1:
        return np.memmap(path, np.uint8, 'r', offset=int(off[0]), shape=(1, H, W))
    if planar == 2:
        return np.memmap(path, np.uint8, 'r', offset=int(off[0]), shape=(spp, H, W))
    return np.memmap(path, np.uint8, 'r', offset=int(off[0]), shape=(H, W, spp)).transpose(2, 0, 1)


def area(arr, W, H, strip=64):
    """resample (C, Hin, Win) 8-bit data to (H, W), 0 in every channel = no data (excluded): exact area average when
    shrinking (streamed in row strips), bilinear when enlarging. Returns (values (C, H, W), coverage (H, W))"""
    C, Hin, Win = arr.shape
    if Hin <= H:
        x = np.asarray(arr).astype(np.float32)
        m = (x.max(0) > 0).astype(np.float32)
        cov = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
        val = np.stack([cv2.resize(x[c] * m, (W, H), interpolation=cv2.INTER_LINEAR) for c in range(C)])
        return np.where(cov > 1e-6, val / np.maximum(cov, 1e-6), 0).astype(np.float32), cov
    s = Hin / H
    val = np.zeros((C, H, W), np.float32)
    cov = np.zeros((H, W), np.float32)
    for r0 in range(0, H, strip):
        r1 = min(H, r0 + strip)
        a, b = int(np.floor(r0 * s)), min(Hin, int(np.ceil(r1 * s)))
        A = np.zeros((r1 - r0, b - a), np.float32)
        for i in range(r0, r1):
            lo, hi = i * s, (i + 1) * s
            for j in range(int(np.floor(lo)), min(int(np.ceil(hi)), Hin)):
                A[i - r0, j - a] = (min(hi, j + 1) - max(lo, j)) / s
        x = np.asarray(arr[:, a:b]).astype(np.float32)
        m = (x.max(0) > 0).astype(np.float32)
        cov[r0:r1] = A @ cv2.resize(m, (W, b - a), interpolation=cv2.INTER_AREA)
        for c in range(C):
            val[c, r0:r1] = A @ cv2.resize(x[c] * m, (W, b - a), interpolation=cv2.INTER_AREA)
    return np.where(cov > 1e-6, val / np.maximum(cov, 1e-6), 0).astype(np.float32), cov


def blur(x, s):
    """Gaussian blur that wraps around in longitude (no seam at lon +-180)"""
    p = min(int(4 * s) + 1, x.shape[1])
    y = cv2.GaussianBlur(np.concatenate([x[:, -p:], x, x[:, :p]], 1), (0, 0), s, borderType=cv2.BORDER_REFLECT)
    return y[:, p:-p]


def lblur(x, s):
    """wide blur computed on a reduced grid (sigma >= 4 px there), bilinear back up"""
    d = int(s // 4)
    if d < 3:
        return blur(x, s)
    H, W = x.shape
    y = blur(cv2.resize(x, (W // d, H // d), interpolation=cv2.INTER_AREA), s / d)
    return cv2.resize(y, (W, H), interpolation=cv2.INTER_LINEAR)


def nblur(x, m, s):
    f = lblur if s >= 12 else blur
    return f(x * m, s) / np.maximum(f(m, s), 1e-4)


def dist(mask):
    """distance (px) from every True pixel to the nearest False pixel, wrapping in longitude"""
    p = min(512, mask.shape[1] // 4)
    u = np.concatenate([mask[:, -p:], mask, mask[:, :p]], 1).astype(np.uint8)
    return cv2.distanceTransform(u, cv2.DIST_L2, 5)[:, p:-p]


def smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def luma(c):
    return c[..., 0] * 0.299 + c[..., 1] * 0.587 + c[..., 2] * 0.114


def ncc(a, b, w):
    a = a - (a * w).sum() / w.sum(); b = b - (b * w).sum() / w.sum()
    return float((a * b * w).sum() / np.sqrt((a * a * w).sum() * (b * b * w).sum() + 1e-12))


def align(L, M, O):
    """longitude shift (+ flip) of the new map that best matches the previous map: NCC of high-passed luminance over
    the valid mask at 1k (cos-lat weighted), refined at 4k. shift_px is in working px (np.roll axis 1)."""
    res = {}
    H, W = L.shape
    def prep(w, h):
        m = cv2.resize(M, (w, h), interpolation=cv2.INTER_AREA)
        l = cv2.resize(L * M, (w, h), interpolation=cv2.INTER_AREA) / np.maximum(m, 1e-4)
        o = luma(cv2.resize(O, (w, h), interpolation=cv2.INTER_AREA))
        s = w / 1024 * 2.0
        m = cv2.erode((m > 0.99).astype(np.uint8), np.ones((5, 5), np.uint8), iterations=max(1, w // 1024)).astype(np.float32)
        lat = np.pi / 2 - (np.arange(h) + 0.5) * np.pi / h
        return l - blur(l, s), o - blur(o, s), m * np.cos(lat)[:, None].astype(np.float32)
    hl, ho, wm = prep(1024, 512)
    cand = {}
    for flip in (False, True):
        a = hl[:, ::-1] if flip else hl
        mm = wm[:, ::-1] if flip else wm
        sc = [ncc(np.roll(a, s, 1), ho, np.roll(mm, s, 1)) for s in range(1024)]
        k = int(np.argmax(sc))
        cand[flip] = (sc[k], k)
        res['candidates_' + ('flip' if flip else 'noflip')] = {'best_shift_1k': k, 'ncc': round(sc[k], 4), 'ncc_shift0': round(sc[0], 4), 'ncc_roll180': round(sc[512], 4)}
        print('  align', 'flip' if flip else 'noflip', 'best shift', k, '/1024 ncc %.4f' % sc[k], '| shift0 %.4f' % sc[0], '| roll180 %.4f' % sc[512], flush=True)
    # a mirrored previous map is never copied (it would mirror the real geography): only the unflipped roll is applied
    if cand[True][0] > cand[False][0] and cand[True][0] >= NCC_MIN:
        print('  WARNING: the previous map matches better MIRRORED (ncc %.3f) -- mirror not applied' % cand[True][0], flush=True)
        res['mirrored_previous_map'] = True
    k1 = cand[False][1]
    hl, ho, wm = prep(4096, 2048)
    sc = {s: ncc(np.roll(hl, s, 1), ho, np.roll(wm, s, 1)) for s in range(4 * k1 - 6, 4 * k1 + 7)}
    s4 = max(sc, key=sc.get)
    res.update(ncc=round(sc[s4], 4), shift_px=int(round(s4 * W / 4096)) % W, flip=False)
    print('  align refined at 4k: shift %d px ncc %.4f' % (res['shift_px'], sc[s4]), flush=True)
    return res


def pushpull(x, m):
    """fill m == 0 holes of x from the surrounding valid pixels (pyramid push-pull, wrap in longitude)"""
    p = min(512, x.shape[1] // 4)
    x = np.concatenate([x[:, -p:], x, x[:, :p]], 1); m = np.concatenate([m[:, -p:], m, m[:, :p]], 1)
    xs, ms = [x * m], [m]
    while min(xs[-1].shape) > 8:
        h, w = xs[-1].shape
        xs.append(cv2.resize(xs[-1], (w // 2, h // 2), interpolation=cv2.INTER_AREA))
        ms.append(cv2.resize(ms[-1], (w // 2, h // 2), interpolation=cv2.INTER_AREA))
    f = np.where(ms[-1] > 1e-4, xs[-1] / np.maximum(ms[-1], 1e-4), (xs[-1].sum() / max(ms[-1].sum(), 1e-4)))
    for xl, ml in zip(xs[-2::-1], ms[-2::-1]):
        up = cv2.resize(f, (xl.shape[1], xl.shape[0]), interpolation=cv2.INTER_LINEAR)
        a = np.clip(ml * 1.5, 0, 1)
        f = np.where(ml > 1e-4, xl / np.maximum(ml, 1e-4), 0) * a + up * (1 - a)
    return f[:, p:-p].astype(np.float32)


_NOISE = {}


def sphere_noise(H, W, wavelengths, seed, chunk=256):
    """isotropic grain: 3-D value noise sampled at each pixel's unit-sphere position (seamless at lon +-180 and the
    poles, not stretched by 1/cos lat); octave wavelengths in px at the equator, equal weights; unit std"""
    rng = np.random.default_rng(seed)
    tab = rng.standard_normal(1 << 20).astype(np.float32)
    lat = np.radians(90 - (np.arange(H) + 0.5) * 180 / H)
    lon = (np.arange(W) + 0.5) / W * 2 * np.pi
    cl, sl = np.cos(lon)[None, :], np.sin(lon)[None, :]
    out = np.zeros((H, W), np.float32)

    def h(X, Y, Z):
        return tab[((X * 73856093) ^ (Y * 19349663) ^ (Z * 83492791)) & ((1 << 20) - 1)]
    for lam in wavelengths:
        fr = W / (2 * np.pi * lam)
        off = rng.uniform(0, 1000, 3)
        for r0 in range(0, H, chunk):
            la = lat[r0:r0 + chunk][:, None]
            c = (fr * np.cos(la)).astype(np.float64)
            pts = (c * cl + off[0], c * sl + off[1], np.broadcast_to(fr * np.sin(la) + off[2], (la.shape[0], W)))
            i = [np.floor(q).astype(np.int64) for q in pts]
            t = [(q - k) for q, k in zip(pts, i)]
            t = [(q * q * (3 - 2 * q)).astype(np.float32) for q in t]
            acc = 0
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        wgt = (t[0] if dx else 1 - t[0]) * (t[1] if dy else 1 - t[1]) * (t[2] if dz else 1 - t[2])
                        acc = acc + wgt * h(i[0] + dx, i[1] + dy, i[2] + dz)
            out[r0:r0 + chunk] += acc
    return out / (float(out.std()) + 1e-6)


def noise(H, W, f):
    """grain field, octaves 2 / 4 / 8 / 16 px (x f) at the equator; one per working size (shared by the bodies)"""
    if (H, W) not in _NOISE:
        _NOISE.clear()
        _NOISE[(H, W)] = sphere_noise(H, W, tuple(max(l * f, 1.0) for l in (2, 4, 8, 16)), 7)
    return _NOISE[(H, W)]


def polar(img, fn, n=2048, lat0=65.0, blend=(70.0, 80.0), extra=None):
    """caps |lat| > lat0 processed in an azimuthal-equidistant n x n grid (no pole pinch / longitude streaks there),
    reprojected back and blended over smoothstep(blend) in |lat|. fn(P, E, rho_deg): P = the cap, E = `extra` remapped
    the same way (or None), rho_deg = angular distance from the pole of every grid cell"""
    H, W = img.shape
    out = img.copy()
    g = (np.arange(n) + 0.5) / n * 2 - 1
    u, v = np.meshgrid(g, g)
    rho = np.sqrt(u * u + v * v) * (90 - lat0)
    xs = (np.degrees(np.arctan2(v, u)) % 360 / 360 * W - 0.5 + 4).astype(np.float32)
    src_ = np.concatenate([img[:, -4:], img, img[:, :4]], 1).astype(np.float32)
    ex_ = None if extra is None else np.concatenate([extra[:, -4:], extra, extra[:, :4]], 1).astype(np.float32)
    lat_r = 90 - (np.arange(H) + 0.5) * 180 / H
    lon = np.radians((np.arange(W) + 0.5) / W * 360)
    for north in (True, False):
        ys = ((rho if north else 180 - rho) / 180 * H - 0.5).astype(np.float32)
        P = fn(cv2.remap(src_, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT),
               None if ex_ is None else cv2.remap(ex_, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT), rho)
        rows = np.where(lat_r >= lat0)[0] if north else np.where(lat_r <= -lat0)[0]
        rr = ((90 - np.abs(lat_r[rows])) / (90 - lat0))[:, None]
        px = ((rr * np.cos(lon)[None, :] + 1) / 2 * n - 0.5).astype(np.float32)
        py = ((rr * np.sin(lon)[None, :] + 1) / 2 * n - 0.5).astype(np.float32)
        B = cv2.remap(P, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        w = smooth((np.abs(lat_r[rows]) - blend[0]) / (blend[1] - blend[0]))[:, None].astype(np.float32)
        out[rows] = img[rows] * (1 - w) + B * w
    return out


def cap(P, E, rho, pole_flat=None):
    """polar cap in the azimuthal grid: gaps re-inpainted here (an equirectangular inpaint smears into rings / hair at
    the pole), 5x5 median + soft blur, and the last 6 deg around the pole (source smear) heavily smoothed -- the sphere
    grain applied afterwards gives the texture back"""
    n = P.shape[0]
    if E is not None and E.max() > 0.5:
        hole = (E > 0.5).astype(np.uint8)
        m = n // 4                                                    # TELEA at 1/4 of the grid, smoothed deeper inside
        sm = cv2.resize(P, (m, m), interpolation=cv2.INTER_AREA)
        hs = (cv2.resize(hole.astype(np.float32), (m, m), interpolation=cv2.INTER_AREA) > 0.01).astype(np.uint8)
        inp = cv2.inpaint(np.where(hs > 0, 0, sm).astype(np.float32), hs, 8, cv2.INPAINT_TELEA)
        dd = cv2.distanceTransform(hs, cv2.DIST_L2, 5)
        a = np.exp(-dd / 16.0)
        inp = cv2.GaussianBlur(inp, (0, 0), 1.5) * a + cv2.GaussianBlur(inp, (0, 0), 10) * (1 - a)
        wf = np.clip(E, 0, 1)
        P = P * (1 - wf) + cv2.resize(inp, (n, n), interpolation=cv2.INTER_LINEAR) * wf
    P = cv2.GaussianBlur(cv2.medianBlur(P, 5), (0, 0), 2)
    k = smooth((6.0 - rho) / 3.0).astype(np.float32)                 # 1 inside 3 deg of the pole, 0 beyond 6 deg
    P = P * (1 - k) + cv2.GaussianBlur(P, (0, 0), n / 50.0) * k
    if pole_flat:                                                     # bad polar frames (blobs, limb streaks): the ring
        rf = 90.0 - pole_flat                                         # mean just outside |lat| = pole_flat, feathered
        ring = (rho >= rf) & (rho <= rf + 2)                          # (the sphere grain comes on top afterwards)
        k2 = smooth((rf + 1 - rho) / 2.0).astype(np.float32)
        P = P * (1 - k2) + float(P[ring].mean()) * k2
    return P


def upsize(x, W, H, p=32):
    """bicubic enlargement that wraps in longitude (no seam at lon +-180)"""
    k = W // x.shape[1]
    return cv2.resize(wrap_pad(x, p), (W + 2 * p * k, H), interpolation=cv2.INTER_CUBIC)[:, p * k:-p * k]


def lowres_std(hp2, m, dn=16):
    """local std of a high-pass (from its square) on a 1/dn grid, inpainted where m has no support, back at full res"""
    H, W = hp2.shape
    w, h = W // dn, H // dn
    v = nblur(cv2.resize(hp2 * m, (w, h), interpolation=cv2.INTER_AREA), cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA), 4)
    sup = blur(cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA), 4) > 0.05
    v = pushpull(v.astype(np.float32), sup.astype(np.float32))
    return cv2.resize(np.sqrt(np.maximum(v, 0)), (W, H), interpolation=cv2.INTER_LINEAR)


def wrap_pad(x, p):
    return np.concatenate([x[:, -p:], x, x[:, :p]], 1)


def local_median(L, s):
    """median over a ~2*sigma window, computed at 1/8 scale on 8-bit data (wraps in longitude), bilinear back up"""
    H, W = L.shape
    d = max(1, int(s // 5))
    small = cv2.resize(L, (W // d, H // d), interpolation=cv2.INTER_AREA)
    k = int(2 * s / d) | 1
    sc = 255.0 / max(float(small.max()), 1.0)
    p = k
    m = cv2.medianBlur(np.clip(np.rint(wrap_pad(small, p) * sc), 0, 255).astype(np.uint8), k)[:, p:-p].astype(np.float32) / sc
    return np.maximum(cv2.resize(blur(m, 1.0), (W, H), interpolation=cv2.INTER_LINEAR), 1.0)


def wrapfix(x, m, n=64):
    """the source steps at its own left / right edge: spread half the (vertically smoothed) step over n columns each side"""
    ok = (m[:, 0] & m[:, -1]).astype(np.float32)
    d = cv2.GaussianBlur(((x[:, 0] - x[:, -1]) * ok)[:, None].astype(np.float32), (0, 0), 2)[:, 0]
    r = np.linspace(0.5, 0, n, dtype=np.float32)[None, :]
    x[:, :n] -= d[:, None] * r
    x[:, -n:] += d[:, None] * r[:, ::-1]
    return x


def lut(n, stops):
    xs = np.array([s[0] for s in stops], np.float32)
    ys = np.array([s[1] for s in stops], np.float32)
    return np.dstack([np.interp(n, xs, ys[:, c]).astype(np.float32) for c in range(3)])


def build(key, cfg, raw, out_root, Wt=None):
    t0 = time.time()
    rawk = os.path.join(raw, key); os.makedirs(rawk, exist_ok=True)
    out = os.path.join(out_root, key); os.makedirs(out, exist_ok=True)
    top = cfg['top']
    W, H = (Wt, Wt // 2) if Wt else SIZES[top]
    f = W / SIZES[top][0]                                             # px figures are at the top tier
    roll0 = W // 2 if cfg['centre'] == 180 else 0                     # source centre lon 180 -> lon 0 at the centre
    print('==', key, W, 'x', H, flush=True)

    # --- luminance, mask, own colour -------------------------------------------------------------------------------
    rgb = None
    if cfg['mode'] == 'own':
        v, cov = area(tiff_map(src(rawk, cfg['color'])), W, H)
        for c in range(3):
            v[c] = wrapfix(v[c], cov > MASK_COV)
        rgb = np.roll(np.ascontiguousarray(v.transpose(1, 2, 0)), roll0, 1)
        L = luma(rgb)
    else:
        v, cov = area(tiff_map(src(rawk, cfg['grey'])), W, H)
        L = np.roll(wrapfix(v[0], cov > MASK_COV), roll0, 1)
    del v
    cov = np.roll(cov, roll0, 1)
    M = cv2.erode((cov > MASK_COV).astype(np.uint8), np.ones((MASK_ERODE, MASK_ERODE), np.uint8)).astype(np.float32)
    del cov
    print('  gap fraction %.3f' % (1 - M.mean()), flush=True)
    O = np.array(Image.open(src(rawk, key + '_old_color_4k.webp', OLD % (key, key))).convert('RGB'), np.float32)

    # --- alignment to the previous site map ------------------------------------------------------------------------
    al = align(L, M, O)
    al['applied'] = al['ncc'] >= NCC_MIN
    shift = al['shift_px'] if al['applied'] else 0
    if not al['applied']:
        # an artistic previous map (blurred, re-tinted) fails the high-pass match but may still carry the same albedo
        # layout: its longitude (which the client rotation is tuned to) is then taken from a low-pass match -- for the
        # orientation only, never for the colour
        lp = align_lowpass(L, M, O)
        al['lowpass'] = lp
        if lp['ncc'] >= NCC_LP_MIN:
            shift = lp['shift_px']
            print('  WARNING: high-pass NCC %.3f < %.2f (artistic previous map); low-pass NCC %.3f -> rolled %d px for the '
                  'orientation only' % (al['ncc'], NCC_MIN, lp['ncc'], shift), flush=True)
        else:
            print('  WARNING: best NCC %.3f < %.2f -- keeping the USGS orientation (lon 0 at the centre)' % (al['ncc'], NCC_MIN), flush=True)
    al['rolled'] = bool(shift)
    L, M = np.roll(L, shift, 1), np.roll(M, shift, 1)
    if rgb is not None:
        rgb = np.roll(rgb, shift, 1)
    al['shift_8k'] = int(round(shift / f)) if top == 8 else None
    al['centre_lon'] = round(((-shift / W * 360 + 180) % 360) - 180, 3)
    # colour rule: an aligned previous map is the colour base at every scale (its colours read real; mission detail is
    # added only where it is better resolved and locally registered); an unrelated (artistic) previous map -> LUT
    oldbase = al['applied'] and os.environ.get('MOONS_BASE_' + key.upper(), cfg.get('base', 'old')) == 'old'
    print('  colour base: %s' % ('previous site map + mission detail' if oldbase else cfg['mode']), flush=True)

    # --- colour chroma field (own / colour mosaic): colour / luminance, low-passed for Ganymede ----------------------
    Cr = None
    if oldbase:
        if rgb is not None:
            del rgb
    elif cfg['mode'] == 'own':
        Cr = rgb / np.maximum(L, 8.0)[..., None]
        Cr = 1 + cfg['sat'] * (Cr - 1)                                # luma(Cr) stays 1: saturation only
        for c in range(3):
            Cr[..., c] = np.where(M > 0, Cr[..., c], pushpull(np.ascontiguousarray(Cr[..., c]), M))
        del rgb
    elif cfg['mode'] == 'chroma':
        v, cc = area(tiff_map(src(rawk, cfg['color'])), W, H)
        cm = (np.roll(cc, roll0, 1) > MASK_COV).astype(np.float32)
        v, cm = np.roll(v, roll0 + shift, 2), np.roll(cm, shift, 1)
        Yc = luma(v.transpose(1, 2, 0))
        sh = align_aux(L, M, Yc, cm)                                  # (both in the aligned frame now)
        if sh:
            print('  colour mosaic shifted %d px onto the grey mosaic' % sh, flush=True)
            v, cm, Yc = np.roll(v, sh, 2), np.roll(cm, sh, 1), np.roll(Yc, sh, 1)
        s = SIG_CHROMA * f
        lpY = np.maximum(nblur(Yc, cm, s), 4.0)
        sup = (blur(cm, s) > 0.5).astype(np.float32)
        Cr = np.empty((H, W, 3), np.float32)
        for c in range(3):
            Cr[..., c] = pushpull(np.where(sup > 0, nblur(v[c], cm, s) / lpY, 1).astype(np.float32), sup)
        Cr = 1 + CHROMA_SAT * np.clip(Cr - 1, -0.35, 0.35)
        del v, cm, Yc, lpY, sup

    # --- baked harsh lighting: soft clamp to the local median where the band-pass std is extreme ---------------------
    Lf = np.where(M > 0, L, pushpull(L, M))
    med = local_median(Lf, SIG_FLAT * f)
    bp = (blur(Lf, 2 * f) - blur(Lf, 16 * f)) / med
    sd = np.sqrt(nblur(bp * bp, M, 64 * f))
    ms = float(np.median(sd[M > 0]))
    w6 = smooth((sd / ms - FLAT_LO) / (FLAT_HI - FLAT_LO)) * M
    r = Lf / med
    flat = med * (1 + FLAT_K * np.tanh((r - 1) / FLAT_K))
    L = Lf + w6 * (flat - Lf)
    print('  harsh-light flatten: weight mean %.3f, area > 0.5: %.3f' % (float(w6.mean()), float((w6 > 0.5).mean())), flush=True)
    del bp, sd, r, flat, med, Lf

    # --- resolution map Q: fine (DoG 1.5/6 px) over coarse (DoG 6/24 px) band energy, smoothed sigma 96, relative to
    # the body's high-resolution reference (median of the ratio over its best-resolved quarter, |lat| < 60). The plain
    # fine-band energy tracks terrain roughness (smooth plains read as low-res); the ratio tracks the image resolution.
    g6 = blur(L, 6 * max(f, 0.5))
    band = (blur(L, 1.5 * max(f, 0.5)) - g6) * M
    e = nblur(band * band, M, Q_SIG * f)
    band = (g6 - blur(L, 24 * max(f, 0.5))) * M
    e = np.sqrt(e / np.maximum(nblur(band * band, M, Q_SIG * f), 1e-3))
    del band, g6
    lat_r = 90 - (np.arange(H) + 0.5) * 180 / H
    sel = (M > 0) & (np.abs(lat_r) < 60)[:, None]
    ev = e[sel]
    ref = float(np.median(ev[ev >= np.percentile(ev, 75)]))
    del ev, sel
    Q = (smooth((e / ref - Q_LO) / (Q_HI - Q_LO)) * M).astype(np.float32)
    del e
    if oldbase:
        c8, okf = old_base(O, L, M, Q, f, lat_r)
        wfill, g, det = np.zeros_like(M), 1.0, 0.0
        colour_txt = ('previous site map (Lanczos to the top tier) at every scale, its own colours; mission detail L / G8(L) added with weight 0.8 Q where locally registered (phase correlation, residual <= %g px @2k; %.0f %% of the data area)' % (RES_MAX, 100 * okf))
    else:
        # de-ringing where Q is low: 5x5 median at 2k (twice), and at 1k below Q 0.4 (the lowest-resolution frames ring at
        # 20..60 px); the polar caps are redone in an azimuthal grid further down
        m2 = cv2.resize(M, (2048, 1024), interpolation=cv2.INTER_AREA)
        L2 = pushpull(cv2.resize(L * M, (2048, 1024), interpolation=cv2.INTER_AREA) / np.maximum(m2, 1e-4), (m2 > 0.5).astype(np.float32))
        for _ in range(2):
            L2 = cv2.medianBlur(wrap_pad(L2, 4), 5)[:, 4:-4]
        L1 = cv2.medianBlur(wrap_pad(cv2.resize(L2, (1024, 512), interpolation=cv2.INTER_AREA), 4), 5)[:, 4:-4]
        a1 = smooth(1 - Q / 0.4)
        L = L * Q + (upsize(L2, W, H) * (1 - a1) + upsize(L1, W, H) * a1) * (1 - Q)
        del L2, L1, a1, m2
        # grain level: 0.6 x the median relative local std of the well-resolved terrain (multiplicative: dark terrain keeps
        # its level, no black holes)
        hp = L - blur(L, 16 * f)
        rel = np.sqrt(nblur(hp * hp, M, 32 * f)) / np.maximum(nblur(L, M, 32 * f), 8.0)
        hiq = (Q > 0.8) & (M > 0)
        det = min(float(np.median(rel[hiq if hiq.sum() > 1000 else M > 0])), DET_MAX)
        del hp, rel, hiq
        print('  Q: ref ratio %.3f, low-Q (< 0.5) area %.3f, grain level %.4f' % (ref, float(((Q < 0.5) & (M > 0)).mean()), det), flush=True)
        N = noise(H, W, f)                                                # isotropic 3-D noise on the sphere, unit std

        # --- gaps / poles: wavy feathered fill from the surrounding ring + matched (multiplicative) grain ----------------
        if M.min() < 1:
            rng = np.random.default_rng(3)
            pert = cv2.resize(blur(rng.standard_normal((H // 32, W // 32), dtype=np.float32), 2), (W, H), interpolation=cv2.INTER_CUBIC)
            pert = np.clip(pert / (pert.std() + 1e-6), -1, 1)
            # ramp width follows the hole size (2 x its inner radius, 8..RAMP px): a speck of black shadow (0 = no data)
            # must not pull 128 px of fill into the imaged terrain around it
            rad = dist(M < 1)
            d8 = 8
            kd = int(RAMP * f * 1.5 / d8)                                 # spread each hole's radius over its ramp zone
            rs = cv2.dilate(wrap_pad(np.ascontiguousarray(rad[d8 // 2::d8, d8 // 2::d8]), kd + 1),
                            np.ones((2 * kd + 1, 2 * kd + 1), np.uint8))[:, kd + 1:-kd - 1]
            ramp = cv2.resize(blur(np.clip(2 * rs, 8 * f, RAMP * f), 2), (W, H), interpolation=cv2.INTER_LINEAR)
            del rad, rs
            w = 1 - smooth((dist(M > 0) - PERT / RAMP * ramp * (1 + pert)) / ramp)   # 1 in the gap, ramps to 0 inside the data
            del pert, ramp
            known = (w < 1e-3).astype(np.float32)
            lp = np.maximum(blur(L, 16 * f), 8.0)
            hp = (L - lp) / lp
            tgt = lowres_std(hp * hp, known)                                  # local relative std of the data
            tgt = np.minimum(tgt, float(np.percentile(tgt[known > 0], 75)))   # rough rim terrain must not set the grain
            del hp, lp
            # ring inpaint at 1k (TELEA), smoothed deeper inside large holes so it never streaks
            w1, h1 = 1024, 512
            k1 = cv2.resize(known, (w1, h1), interpolation=cv2.INTER_AREA)
            d1 = cv2.resize(L * known, (w1, h1), interpolation=cv2.INTER_AREA) / np.maximum(k1, 1e-4)
            hole = (k1 < 0.99).astype(np.uint8)
            p = 64
            inp = cv2.inpaint(wrap_pad(np.where(hole > 0, 0, d1).astype(np.float32), p), wrap_pad(hole, p), 8, cv2.INPAINT_TELEA)[:, p:-p]
            dd = dist(hole > 0)
            a = np.exp(-dd / 24.0)
            inp = blur(inp, 1.5) * a + blur(inp, 10) * (1 - a)
            F = cv2.resize(inp, (W, H), interpolation=cv2.INTER_LINEAR)
            del k1, d1, hole, inp, dd, a
            # small holes (shadow specks) and the first px of large ones: full-resolution TELEA, so they carry the local
            # texture instead of a flat 1k patch
            hole8 = (known < 1).astype(np.uint8)
            p = 32
            F8 = cv2.inpaint(wrap_pad(np.where(hole8 > 0, 0, L).astype(np.float32), p), wrap_pad(hole8, p), 3, cv2.INPAINT_TELEA)[:, p:-p]
            a = np.exp(-dist(hole8 > 0) / (16 * f))
            F = F8 * a + F * (1 - a)
            del hole8, F8, a
            F = F * (1 + np.clip(GRAIN_GAP * tgt * N, -0.6, 0.6))
            L = L * (1 - w) + F * w
            if Cr is not None and cfg['mode'] == 'own':   # Io: chroma in the gaps from the smooth surroundings
                for c in range(3):
                    sm = blur(pushpull(np.ascontiguousarray(Cr[..., c]), known), 24 * f)
                    Cr[..., c] = Cr[..., c] * (1 - w) + sm * w
            del F, tgt, known
            wfill = w
        else:
            wfill = np.zeros_like(M)

        # --- polar caps: de-ripple + soften in an azimuthal grid (no pole pinch / longitude streaks), then grain on the
        # low-Q terrain and the caps with the isotropic sphere noise (no 1/cos lat stretch)
        L = polar(L, lambda P, E, r: cap(P, E, r, cfg.get('pole_flat')), n=int(2048 * f) // 2 * 2, extra=wfill)
        if Cr is not None:                                            # chroma: no per-pixel colour streaks at the caps
            for c in range(3):
                Cr[..., c] = polar(np.ascontiguousarray(Cr[..., c]), lambda P, E, r: cap(cv2.GaussianBlur(P, (0, 0), 8), None, r,
                                   cfg.get('pole_flat')), n=int(2048 * f) // 2 * 2)
        gw = np.maximum((1 - Q) * M * (1 - wfill), smooth((np.abs(lat_r) - 70) / 10)[:, None].astype(np.float32))
        L = L * (1 + np.clip(GRAIN_Q * det * gw * N, -0.6, 0.6))
        del gw, N

        # --- colour --------------------------------------------------------------------------------------------------------
        L = np.maximum(L, 0.5)
        if cfg['mode'] == 'lut':
            Ls = blur(L, SIG_LUT * f)
            valid = (M > 0) & (wfill < 0.5)
            lo, hi = np.percentile(Ls[valid], [1, 99.5])
            n = np.clip((Ls - lo) / max(hi - lo, 1e-3), 0, 1)
            col = lut(n, cfg['lut']) * np.clip(L / np.maximum(Ls, 0.5), 0, 3)[..., None]
            del Ls, n
        else:
            col = L[..., None] * Cr
            del Cr
        del L

        # --- exposure: global luminance mean of the previous map; soft highlight knee; no blue cast -------------------------
        # a tone curve (power per channel), not a gain: a 2x brighter previous map must not blow the highlights out
        Y, YO = luma(col), luma(O)
        cs = np.clip(col[::4, ::4], 0, 255) / 255
        if cfg.get('linear'):                                             # gamma-neutral: plain gain, the knee takes the top
            g = 1.0
        lo, hi = 0.2, 5.0
        for _ in range(0 if cfg.get('linear') else 40):
            g = (lo * hi) ** 0.5
            if luma(cs ** g).mean() * 255 > YO.mean():
                lo = g
            else:
                hi = g
        if cfg.get('linear') or g > 1:                                  # darkening: a plain gain (a power > 1 would
            g = 1.0                                                       # multiply every ripple / ringing contrast)
            col *= float(YO.mean() / max(Y.mean(), 1e-3))
        else:
            col = 255 * (np.clip(col, 0, 255) / 255) ** g
        print('  luma mean new %.1f -> old %.1f (tone power %.3f)' % (Y.mean(), YO.mean(), g), flush=True)
        del Y, cs, O
        T = KNEE * 255
        col = np.where(col > T, T + (255 - T) * (1 - np.exp(-(col - T) / (255 - T))), col)
        if cfg.get('neutral'):
            col[..., 2] = np.minimum(col[..., 2], np.maximum(col[..., 0], col[..., 1]))
        else:
            col[..., 2] = np.minimum(col[..., 2], col[..., 1] + 6)
        pr = max(2, int(round(POLE_ROWS * f)))
        col[:pr] = col[pr:2 * pr + 1].reshape(-1, 3).mean(0)
        col[-pr:] = col[-2 * pr - 1:-pr].reshape(-1, 3).mean(0)
        c8 = np.clip(np.rint(col), 0, 255).astype(np.uint8)
        del col
        colour_txt = ({'own': 'the mosaic\'s own colour, saturation x%s' % cfg.get('sat'),
                       'chroma': 'grey mosaic luminance x colour-mosaic chroma low-passed sigma %d px' % SIG_CHROMA,
                       'lut': 'natural-colour LUT on the low-passed luminance (sigma %d px) x detail ratio, stops %s' % (SIG_LUT, cfg.get('lut'))}[cfg['mode']] +
                      '; harsh terminator lighting soft-clamped to the local median; low-resolution areas de-ringed; gaps filled '
                      '(wavy ramp <= %d px, ring inpaint); polar caps de-rippled in an azimuthal grid; multiplicative sphere-noise '
                      'grain (%g x local detail); luminance mean of the previous map (tone power %.3f); no blue cast' % (RAMP, GRAIN_GAP, g))
    if os.environ.get('MOONS_DEBUG'):                                # diagnostics (Q, mask, fill weight) outside the feed
        np.save(os.path.join(os.environ['MOONS_DEBUG'], key + '_q.npy'),
                cv2.resize(np.dstack([Q, M, wfill]), (2048, 1024), interpolation=cv2.INTER_AREA).astype(np.float16))

    # --- write tiers ----------------------------------------------------------------------------------------------------
    files = []
    for kk, (w, hh) in SIZES.items():
        if Wt is None and kk > top:
            continue
        if Wt and w > W:
            continue
        cc = c8 if w == W else cv2.resize(c8, (w, hh), interpolation=cv2.INTER_AREA)
        nm = '%s_color_%dk.webp' % (key, kk)
        webp(cc, os.path.join(out, nm), 90); files.append(nm)
        nm = '%s_normal_%dk.webp' % (key, kk)
        webp(np.full((hh, w, 3), (128, 128, 255), np.uint8), os.path.join(out, nm), 92); files.append(nm)
    meta = {'v': 1, 'built': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'radius_km': cfg['radius_km'],
            'projection': 'equirectangular, north up, east to the right; ' + (
                'rolled onto the previous site map, lon %g at the centre' % al['centre_lon'] if al['rolled'] else 'lon 0 at the centre'),
            'align': al, 'max_tier': '%dk' % top,
            'normal': 'flat (128,128,255): no global DEM of %s is available; relief is not derived from image brightness' % cfg['name'],
            'colour': colour_txt, 'colour_base': 'previous site map' if oldbase else 'mission mosaic (%s)' % cfg['mode'],
            'files': sorted(files),
            'source': cfg['source'], 'credit': cfg['credit']}
    json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'), indent=1)
    print('  %s done in %d s' % (key, time.time() - t0), flush=True)


def old_base(O, L, M, Q, f, lat_r):
    """previous map as the colour base (Lanczos to the working size) x mission detail L / G8(L) with weight 0.8 Q,
    only where the mosaic is locally registered to the previous map (phase correlation at 2k on 256 px tiles every
    128 px: residual shift <= RES_MAX px and a real correlation peak), feathered off the no-data edge and the caps"""
    H, W = L.shape
    p = 16
    col = cv2.resize(np.concatenate([O[:, -p:], O, O[:, :p]], 1), (W + 2 * p * W // O.shape[1], H),
                     interpolation=cv2.INTER_LANCZOS4)[:, p * W // O.shape[1]:-p * W // O.shape[1]]
    # local registration
    w2, h2, t, st = 2048, 1024, 256, 128
    m2 = cv2.resize(M, (w2, h2), interpolation=cv2.INTER_AREA)
    a = luma(cv2.resize(O, (w2, h2), interpolation=cv2.INTER_AREA))
    b = pushpull(cv2.resize(L * M, (w2, h2), interpolation=cv2.INTER_AREA) / np.maximum(m2, 1e-4), (m2 > 0.5).astype(np.float32))
    a, b = (x - blur(x, 8) for x in (a, b))
    a, b = wrap_pad(a, t), wrap_pad(b, t)
    win = cv2.createHanningWindow((t, t), cv2.CV_32F)
    acc, wsum = np.zeros((h2, w2 + 2 * t), np.float32), np.zeros((h2, w2 + 2 * t), np.float32)
    n_ok = n_all = 0
    for y0 in range(0, h2 - t + 1, st):
        for x0 in range(t // 2, w2 + t // 2 + 1, st):
            if m2[y0:y0 + t, (np.arange(x0, x0 + t) - t) % w2].mean() < 0.5:
                continue
            (dx, dy), resp = cv2.phaseCorrelate(np.ascontiguousarray(a[y0:y0 + t, x0:x0 + t]).astype(np.float64),
                                                np.ascontiguousarray(b[y0:y0 + t, x0:x0 + t]).astype(np.float64), win.astype(np.float64))
            ok = float(np.hypot(dx, dy) <= RES_MAX and resp >= RESP_MIN)
            n_ok += ok; n_all += 1
            acc[y0:y0 + t, x0:x0 + t] += win * ok
            wsum[y0:y0 + t, x0:x0 + t] += win
    acc[:, t:2 * t] += acc[:, -t:]; wsum[:, t:2 * t] += wsum[:, -t:]
    acc[:, -2 * t:-t] += acc[:, :t]; wsum[:, -2 * t:-t] += wsum[:, :t]
    okm = (acc / np.maximum(wsum, 1e-4))[:, t:-t]
    okm = cv2.resize(blur(okm, 8), (W, H), interpolation=cv2.INTER_LINEAR)
    okf = n_ok / max(n_all, 1)
    print('  local registration: %d / %d tiles within %g px' % (n_ok, n_all, RES_MAX), flush=True)
    wgt = DETAIL_W * Q * smooth((okm - 0.5) / 0.4) * smooth(dist(M > 0) / (32 * f))
    wgt *= (1 - smooth((np.abs(lat_r) - 70) / 10))[:, None].astype(np.float32)   # no equirectangular detail at the caps
    ratio = np.clip(L / np.maximum(blur(L, 8 * f), 1.0), 0.4, 2.0)
    col *= (1 + wgt * (ratio - 1))[..., None]
    return np.clip(np.rint(col), 0, 255).astype(np.uint8), okf


def align_lowpass(L, M, O):
    """longitude shift from low-passed luminance (sigma 3 px at 512 x 256, cos-lat weighted, no flip)"""
    H, W = L.shape
    m = cv2.resize(M, (512, 256), interpolation=cv2.INTER_AREA)
    l = pushpull(cv2.resize(L * M, (512, 256), interpolation=cv2.INTER_AREA) / np.maximum(m, 1e-4), (m > 0.5).astype(np.float32))
    o = luma(cv2.resize(O, (512, 256), interpolation=cv2.INTER_AREA))
    lat = np.cos(np.radians(90 - (np.arange(256) + 0.5) * 180 / 256))[:, None].astype(np.float32)
    l, o = blur(l, 3), blur(o, 3)
    sc = [ncc(np.roll(l, s, 1), o, lat * np.ones_like(o)) for s in range(512)]
    k = int(np.argmax(sc))
    return {'shift_512': k, 'ncc': round(sc[k], 4), 'shift_px': int(round(k * W / 512)) % W}


def align_aux(L, M, Y, m):
    """longitude offset (px) of a second mosaic against the grey one (NCC of high-passed luminance at 1k, +-16 px @1k)"""
    H, W = L.shape
    def hp(x, mk):
        a = cv2.resize(mk, (1024, 512), interpolation=cv2.INTER_AREA)
        b = cv2.resize(x * mk, (1024, 512), interpolation=cv2.INTER_AREA) / np.maximum(a, 1e-4)
        return b - blur(b, 2), (a > 0.99).astype(np.float32)
    a, ma = hp(L, M)
    b, mb = hp(Y, m)
    sc = {s: ncc(a, np.roll(b, s, 1), ma * np.roll(mb, s, 1)) for s in range(-16, 17)}
    s = max(sc, key=sc.get)
    print('  colour mosaic vs grey: best shift %d px @1k ncc %.3f (shift0 %.3f)' % (s, sc[s], sc[0]), flush=True)
    return int(round(s * W / 1024))


def main():
    args = sys.argv[1:]
    opt = {}
    pos = []
    i = 0
    while i < len(args):
        if args[i].startswith('--'):
            opt[args[i][2:]] = args[i + 1]; i += 2
        else:
            pos.append(args[i]); i += 1
    raw = pos[0] if pos else os.path.join(HERE, 'raw')
    out = opt.get('out', os.path.join(HERE, 'out'))
    keys = opt['only'].split(',') if 'only' in opt else [k for k in BODIES if not BODIES[k].get('disabled')]
    Wt = int(opt['size']) if 'size' in opt else None
    t0 = time.time()
    for k in keys:
        build(k, BODIES[k], raw, out, Wt)
    print('done in', round(time.time() - t0), 's')


if __name__ == '__main__':
    main()
