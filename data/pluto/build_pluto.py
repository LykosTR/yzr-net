#!/usr/bin/env python3
"""Pluto surface maps for Space Radar (USGS Astrogeology, public domain):
  image  : Pluto New Horizons Global Mosaic 300 m (LORRI + MVIC, 8-bit, 24888x12444, 0 = unimaged)
  relief : Pluto New Horizons Global DEM 300 m (int16 metres above R = 1188.3 km, -32768 = no data)
The New Horizons mosaic is grey: colour is a natural-colour LUT on its low-passed luminance x its detail ratio (one palette,
no green / blue). A resolution map Q (band-pass energy vs Sputnik / Tombaugh) marks the low-resolution approach imagery:
there the luminance is de-rippled and grained and the relief fades out. The unimaged south (polar night) takes the
previous site map's luminance (rolled onto it when the two correlate: the client rotation was tuned to it, see meta.align)
through the same LUT, with a wavy feathered ramp and grain matched to the local detail.
Outputs (out/): pluto_{color,normal}_{2,4,8}k.webp + meta.json -- equirectangular, north up, east to the right;
normal map tangent space x = east, y = north, z = up (OpenGL), true scale, flat where the DEM has no data.
Needs: pip install numpy pillow opencv-python.   Usage: build_pluto.py [raw_dir]"""
import json, os, sys, time, urllib.request
import numpy as np
import cv2
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
KEY = 'pluto'
USGS = 'https://planetarymaps.usgs.gov/mosaic/'
MOSAIC, DEM = 'Pluto_NewHorizons_Global_Mosaic_300m_Jul2017_8bit.tif', 'Pluto_NewHorizons_Global_DEM_300m_Jul2017_16bit.tif'
OLD = 'https://raw.githubusercontent.com/LykosTR/yzr-net/data/textures/pluto/pluto_color_4k.webp'   # previous site map
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
R_KM = 1188.3                               # TIFF label: A/B/C_AXIS_RADIUS 1188.3 km
SIZES = {8: (8192, 4096), 4: (4096, 2048), 2: (2048, 1024)}
AO = 0.18                                   # relief occlusion strength
FEATHER = 24                                # px at 8k: DEM gap blend ramp
RAMP, PERT = 128, 48                        # px at 8k: image gap blend ramp, starting 0..2*PERT px inside the data (the
                                            # mosaic's rim is smeared terminator imagery; a wavy start hides straight edges)
GRAIN = 0.7                                 # grain in filled areas: fraction of the missing high-frequency std
RELIEF_IMAGED_ONLY = True                    # relief only where mission imagery is shown (DEM rim beyond it is noise)
RATIO_MAX = 4.0                             # clip of the mosaic detail ratio L / lowpass L
SAT = 0.9                                   # global saturation after the LUT
LUT_T = np.array([0.0, 0.15, 0.35, 0.55, 0.75, 1.0])           # natural-colour LUT on normalised low-passed luminance
LUT_RGB = np.array([[70, 40, 30], [100, 58, 42], [160, 112, 86], [200, 168, 140], [226, 210, 190], [242, 238, 230]], np.float32)
REF_LON, REF_LAT = (150, 230), (-10, 50)   # Sputnik Planitia / Tombaugh Regio box (deg E, deg N): the resolution reference
Q_EDGE = (0.4, 0.9)                         # Q = smoothstep of (fine/coarse band ratio / its median over the reference box)
LUMA_MEAN, CONTRAST, MID_SAT = 150.0, 0.12, 1.15   # exposure (old map 136), S-curve slope gain, mid-tone saturation
KNEE = None                                 # soft-knee start for highlights (fraction of 255), None = hard clip
TOE = 0.12                                  # soft-toe start for shadows (fraction of 255)
NCC_MIN = 0.35
MASK_COV, MASK_ERODE = 0.99, 5                # mosaic pixel valid above this source coverage; erosion (px) drops the partial rim


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


def tiff_map(path, dtype):
    """uncompressed, contiguous-strip GeoTIFF -> read-only memmap (H, W) + its GDAL nodata value"""
    im = Image.open(path)
    t = im.tag_v2
    W, H = im.size
    off = np.asarray(t[273], np.int64)
    rb = W * np.dtype(dtype).itemsize
    assert t.get(259, 1) == 1 and len(off) and np.all(np.diff(off) == rb * (H // len(off))), 'TIFF not contiguous/uncompressed'
    nod = t.get(42113)
    im.close()
    return np.memmap(path, dtype=dtype, mode='r', offset=int(off[0]), shape=(H, W)), (float(nod) if nod is not None else None)


def area(srcarr, W, H, nodata=None, strip=64):
    """exact area resample of a huge 2-D array, streamed in row strips; nodata pixels excluded.
    returns (value, coverage) at (H, W), coverage = fraction of valid source area"""
    Hin, Win = srcarr.shape
    s = Hin / H
    val = np.zeros((H, W), np.float32)
    cov = np.ones((H, W), np.float32)
    for r0 in range(0, H, strip):
        r1 = min(H, r0 + strip)
        a, b = int(np.floor(r0 * s)), min(Hin, int(np.ceil(r1 * s)))
        A = np.zeros((r1 - r0, b - a), np.float32)
        for i in range(r0, r1):
            lo, hi = i * s, (i + 1) * s
            for j in range(int(np.floor(lo)), min(int(np.ceil(hi)), Hin)):
                A[i - r0, j - a] = (min(hi, j + 1) - max(lo, j)) / s
        x = np.asarray(srcarr[a:b]).astype(np.float32)
        if nodata is not None:
            m = (x != nodata).astype(np.float32)
            x *= m
            cov[r0:r1] = A @ cv2.resize(m, (W, b - a), interpolation=cv2.INTER_AREA)
        val[r0:r1] = A @ cv2.resize(x, (W, b - a), interpolation=cv2.INTER_AREA)
    return np.where(cov > 1e-6, val / np.maximum(cov, 1e-6), 0).astype(np.float32), cov


def blur(x, s):
    """Gaussian blur that wraps around in longitude (no seam at lon +-180)"""
    p = min(int(4 * s) + 1, x.shape[1])
    y = cv2.GaussianBlur(np.concatenate([x[:, -p:], x, x[:, :p]], 1), (0, 0), s, borderType=cv2.BORDER_REFLECT)
    return y[:, p:-p]


def nblur(x, m, s):
    return blur(x * m, s) / np.maximum(blur(m, s), 1e-4)


def dist(mask):
    """distance (px) from every True pixel to the nearest False pixel, wrapping in longitude"""
    p = 512
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
    """longitude shift (+ flip) of the new map that best matches the previous map: NCC of high-passed luminance
    over the valid mask at 1k (cos-lat weighted), refined at 4k. Returns dict; shift is in 8k px (np.roll axis 1)."""
    res = {}
    def prep(w, h):
        m = cv2.resize(M, (w, h), interpolation=cv2.INTER_AREA)
        l = cv2.resize(L * M, (w, h), interpolation=cv2.INTER_AREA) / np.maximum(m, 1e-4)
        o = luma(cv2.resize(O, (w, h), interpolation=cv2.INTER_AREA))
        s = w / 1024 * 2.0
        m = cv2.erode((m > 0.99).astype(np.uint8), np.ones((5, 5), np.uint8), iterations=max(1, w // 1024)).astype(np.float32)
        lat = np.pi / 2 - (np.arange(h) + 0.5) * np.pi / h
        return l - blur(l, s), o - blur(o, s), m * np.cos(lat)[:, None].astype(np.float32)
    hl, ho, wm = prep(1024, 512)
    best = (-2, 0, False)
    for flip in (False, True):
        a = hl[:, ::-1] if flip else hl
        mm = wm[:, ::-1] if flip else wm
        sc = [ncc(np.roll(a, s, 1), ho, np.roll(mm, s, 1)) for s in range(1024)]
        k = int(np.argmax(sc))
        res['candidates_' + ('flip' if flip else 'noflip')] = {'best_shift_1k': k, 'ncc': round(sc[k], 4), 'ncc_shift0': round(sc[0], 4), 'ncc_roll180': round(sc[512], 4)}
        print('align', 'flip' if flip else 'noflip', 'best shift', k, '/1024 ncc %.4f' % sc[k], '| shift0 %.4f' % sc[0], '| roll180 %.4f' % sc[512], flush=True)
        if sc[k] > best[0]:
            best = (sc[k], k, flip)
    hl, ho, wm = prep(4096, 2048)
    if best[2]:
        hl, wm = hl[:, ::-1], wm[:, ::-1]
    sc = {s: ncc(np.roll(hl, s, 1), ho, np.roll(wm, s, 1)) for s in range(4 * best[1] - 6, 4 * best[1] + 7)}
    s4 = max(sc, key=sc.get)
    res.update(ncc=round(sc[s4], 4), shift_8k=int(2 * s4) % 8192, flip=bool(best[2]))
    print('align refined at 4k: shift %d px (8k) ncc %.4f flip %s' % (res['shift_8k'], sc[s4], best[2]), flush=True)
    return res


def pushpull(x, m):
    """fill m == 0 holes of x from the surrounding valid pixels (pyramid push-pull, wrap in longitude)"""
    p = 512
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


def fractal(H, W, seed, octaves=(0.7, 1.4, 2.8, 5.6, 11.2), unit_hp=True):
    """unit-std grain (octaves 0.7..5.6 px), wrapping in longitude; scaled so its sigma-2 high-pass has unit std"""
    rng = np.random.default_rng(seed)
    n = np.zeros((H, W), np.float32)
    for s in octaves:
        g = blur(rng.standard_normal((H, W), dtype=np.float32), s)
        n += g / (g.std() + 1e-6)
    if not unit_hp:          # unit total std
        return n / (n.std() + 1e-6)
    hp = n - blur(n, 2)
    return n / (hp.std() + 1e-6)


def lowres_std(hp2, m, dn=16):
    """local std of a high-pass (from its square) on a 1/dn grid, inpainted where m has no support, back at full res"""
    H, W = hp2.shape
    w, h = W // dn, H // dn
    v = nblur(cv2.resize(hp2 * m, (w, h), interpolation=cv2.INTER_AREA), cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA), 4)
    sup = blur(cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA), 4) > 0.05
    v = pushpull(v.astype(np.float32), sup.astype(np.float32))
    return cv2.resize(np.sqrt(np.maximum(v, 0)), (W, H), interpolation=cv2.INTER_LINEAR)


def wrapfix(O, n=64):
    """the previous map steps at its lon +-180 edge: spread half the (vertically smoothed) step over n columns each side"""
    d = cv2.GaussianBlur(np.ascontiguousarray(O[:, :1] - O[:, -1:]), (0, 0), 2)
    r = np.linspace(0.5, 0, n, dtype=np.float32)[None, :, None]
    O[:, :n] -= d.reshape(-1, 1, O.shape[2]) * r
    O[:, -n:] += d.reshape(-1, 1, O.shape[2]) * r[:, ::-1]
    return O


def wrapfix_dem(h, D, n=64):
    """the DEM steps at its own lon 0/360 edge (a raw-source seam): same treatment as wrapfix, rows with data on both sides"""
    d = (h[:, 0] - h[:, -1]) * (D[:, 0] * D[:, -1])
    d = cv2.GaussianBlur(d[:, None].astype(np.float32), (0, 0), 2)[:, 0]
    r = np.linspace(0.5, 0, n, dtype=np.float32)[None, :]
    h[:, :n] -= d[:, None] * r
    h[:, -n:] += d[:, None] * r[:, ::-1]
    return h


def sphere_noise(H, W, wavelengths, seed, unit_hp=False, chunk=256):
    """isotropic grain: 3-D value noise sampled at each pixel's unit-sphere position (seamless at lon +-180 and the
    poles, not stretched by 1/cos lat); octave wavelengths in px at the equator, equal weights; unit std (or unit
    sigma-2 high-pass std)"""
    rng = np.random.default_rng(seed)
    tab = rng.standard_normal(1 << 20).astype(np.float32)
    lat = np.radians(90 - (np.arange(H) + 0.5) * 180 / H)
    lon = (np.arange(W) + 0.5) / W * 2 * np.pi
    cl, sl = np.cos(lon)[None, :], np.sin(lon)[None, :]
    out = np.zeros((H, W), np.float32)

    def h(X, Y, Z):
        return tab[((X * 73856093) ^ (Y * 19349663) ^ (Z * 83492791)) & ((1 << 20) - 1)]
    for lam in wavelengths:
        f = W / (2 * np.pi * lam)
        off = rng.uniform(0, 1000, 3)
        for r0 in range(0, H, chunk):
            la = lat[r0:r0 + chunk][:, None]
            c = (f * np.cos(la)).astype(np.float64)
            pts = (c * cl + off[0], c * sl + off[1], np.broadcast_to(f * np.sin(la) + off[2], (la.shape[0], W)))
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
    if unit_hp:
        return out / (float((out - blur(out, 2)).std()) + 1e-6)
    return out / (float(out.std()) + 1e-6)


def polar(img, fn, n=2048, lat0=65.0, blend=(70.0, 80.0)):
    """caps |lat| > lat0 processed in an azimuthal-equidistant n x n grid (no pole pinch / longitude streaks there),
    reprojected back and blended over smoothstep(blend) in |lat|"""
    H, W = img.shape
    out = img.copy()
    g = (np.arange(n) + 0.5) / n * 2 - 1
    u, v = np.meshgrid(g, g)
    rho = np.sqrt(u * u + v * v) * (90 - lat0)
    xs = (np.degrees(np.arctan2(v, u)) % 360 / 360 * W - 0.5 + 4).astype(np.float32)
    src_ = np.concatenate([img[:, -4:], img, img[:, :4]], 1).astype(np.float32)
    lat_r = 90 - (np.arange(H) + 0.5) * 180 / H
    lon = np.radians((np.arange(W) + 0.5) / W * 360)
    for north in (True, False):
        ys = ((rho if north else 180 - rho) / 180 * H - 0.5).astype(np.float32)
        P = fn(cv2.remap(src_, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))
        rows = np.where(lat_r >= lat0)[0] if north else np.where(lat_r <= -lat0)[0]
        rr = ((90 - np.abs(lat_r[rows])) / (90 - lat0))[:, None]
        px = ((rr * np.cos(lon)[None, :] + 1) / 2 * n - 0.5).astype(np.float32)
        py = ((rr * np.sin(lon)[None, :] + 1) / 2 * n - 0.5).astype(np.float32)
        B = cv2.remap(P, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        w = smooth((np.abs(lat_r[rows]) - blend[0]) / (blend[1] - blend[0]))[:, None].astype(np.float32)
        out[rows] = img[rows] * (1 - w) + B * w
    return out


def normals(h, scale=None):
    H, W = h.shape
    dlon, dlat = 2 * np.pi / W, np.pi / H
    lat = np.pi / 2 - (np.arange(H) + 0.5) * dlat
    coslat = np.maximum(np.cos(lat), 0.02)[:, None]
    hx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) / (2 * R_KM * coslat * dlon)
    up, dn = np.vstack([h[:1], h[:-1]]), np.vstack([h[1:], h[-1:]])
    hy = (up - dn) / (2 * R_KM * dlat)
    if scale is not None:   # relief strength 0..1 (0 = flat: no DEM / low-resolution imagery)
        hx, hy = hx * scale, hy * scale
    n = np.dstack([-hx, -hy, np.ones_like(h)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return np.clip(np.rint((n * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8)


def load_mosaic(raw, W, H):
    a, nod = tiff_map(src(raw, MOSAIC), np.uint8)
    L, cov = area(a, W, H, nodata=nod if nod is not None else 0)
    del a
    return L, cov


def load_dem(raw, W, H):
    a, nod = tiff_map(src(raw, DEM), np.int16)
    h, cov = area(a, W, H, nodata=nod if nod is not None else -32768)
    del a
    return h / 1000.0, cov                                            # metres -> km


def main():
    raw = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'raw')
    os.makedirs(raw, exist_ok=True); os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    W, H = SIZES[8]
    half = W // 2                                                     # source: lon 0..360 -> lon 0 at the centre
    L, cov = load_mosaic(raw, W, H)
    L, cov = np.roll(L, half, 1), np.roll(cov, half, 1)
    M = cv2.erode((cov > MASK_COV).astype(np.uint8), np.ones((MASK_ERODE, MASK_ERODE), np.uint8)).astype(np.float32)
    del cov
    print('mosaic gap fraction %.3f' % (1 - M.mean()), flush=True)
    O = wrapfix(np.array(Image.open(src(raw, KEY + '_old_color_4k.webp', OLD)).convert('RGB'), np.float32))
    al = align(L, M, O)
    al['applied'] = al['ncc'] >= NCC_MIN
    if al['applied']:
        if al['flip']:
            L, M = L[:, ::-1].copy(), M[:, ::-1].copy()
        L, M = np.roll(L, al['shift_8k'], 1), np.roll(M, al['shift_8k'], 1)
    else:
        print('WARNING: best NCC %.3f < %.2f -- keeping the USGS orientation (lon 0 at the centre)' % (al['ncc'], NCC_MIN), flush=True)
    shift, flip = (al['shift_8k'], al['flip']) if al['applied'] else (0, False)
    al['centre_lon'] = round(((-shift / W * 360 + 180) % 360) - 180, 3) if not flip else None

    # --- resolution map Q: fine (DoG 1.5/6 px) over coarse (DoG 6/24 px) band energy, smoothed sigma 96, relative to
    # Sputnik / Tombaugh. The plain fine-band energy tracks terrain roughness (smooth Sputnik read as low-res), the
    # fine/coarse ratio tracks the image's own resolution.
    g2 = blur(L, 6)
    f = (blur(L, 1.5) - g2) * M
    c = (g2 - blur(L, 24)) * M
    del g2
    e = np.sqrt(nblur(f * f, M, 96) / np.maximum(nblur(c * c, M, 96), 1e-3))
    del f, c
    c0 = al['centre_lon'] if al['applied'] and al['centre_lon'] is not None else 0.0
    x0, x1 = (int(((lo - c0 + 180) % 360) / 360 * W) for lo in REF_LON)
    y0, y1 = (int((90 - la) / 180 * H) for la in REF_LAT[::-1])
    ref = e[y0:y1, x0:x1][M[y0:y1, x0:x1] > 0]
    Q = (smooth((e / float(np.median(ref)) - Q_EDGE[0]) / (Q_EDGE[1] - Q_EDGE[0])) * M).astype(np.float32)
    del e, ref
    print('Q: mean %.3f over the data, %.3f of it < 0.5' % (Q[M > 0].mean(), (Q[M > 0] < 0.5).mean()), flush=True)
    if os.environ.get('PREVIEW_DIR'):
        Image.fromarray(np.rint(cv2.resize(Q, (2048, 1024), interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)).save(
            os.path.join(os.environ['PREVIEW_DIR'], KEY + '_Q.png'))

    # --- low-Q (approach imagery): de-ripple at 2k + fractal grain ------------------------------------------------
    m2 = cv2.resize(M, (2048, 1024), interpolation=cv2.INTER_AREA)
    l2 = pushpull(cv2.resize(L * M, (2048, 1024), interpolation=cv2.INTER_AREA) / np.maximum(m2, 1e-4), (m2 > 0.5).astype(np.float32))
    l2 = np.concatenate([l2[:, -32:], l2, l2[:, :32]], 1).astype(np.float32)      # wrap: filters must not see a border
    l2 = cv2.bilateralFilter(cv2.medianBlur(l2, 5), 9, 12, 3)
    l2 = cv2.resize(l2, (W + 256, H), interpolation=cv2.INTER_CUBIC)[:, 128:-128]   # 32 px @2k = 128 px @8k
    Lc = L * Q + l2 * (1 - Q)
    del l2, m2
    hp = L - blur(L, 16)                                              # grain: 0.6 x median relative local std of the high-res
    rel = np.sqrt(nblur(hp * hp, M, 32)) / np.maximum(nblur(L, M, 32), 8.0)   # region, multiplicative (dark terrain keeps
    sd_hi = float(np.median(rel[Q > 0.8]))                            # its level: no black holes)
    del hp, L, rel
    print('low-Q grain: relative amplitude %.3f (applied after the polar caps)' % (0.6 * sd_hi), flush=True)

    # --- gaps: old-map luminance (mapped onto the mosaic's scale, highs soft-clamped) + matched grain -------------
    img = np.ones((H, W), bool)
    Lall = Lc
    if M.min() < 1:
        rng = np.random.default_rng(3)
        pert = cv2.resize(blur(rng.standard_normal((H // 32, W // 32), dtype=np.float32), 2), (W, H), interpolation=cv2.INTER_CUBIC)
        pert = np.clip(pert / (pert.std() + 1e-6), -1, 1)
        w = 1 - smooth((dist(M > 0) - PERT * (1 + pert)) / RAMP)       # 1 in the gap, ramps to 0 inside the data
        del pert
        core = (M > 0) & (w < 1e-3)
        FY = cv2.resize(luma(O), (W, H), interpolation=cv2.INTER_CUBIC) if al['applied'] else pushpull(Lc, core.astype(np.float32))
        a, b = blur(FY, 24)[core], blur(Lc, 24)[core]
        FY = (FY - float(a.mean())) * float(b.std() / max(a.std(), 1e-3)) + float(b.mean())
        del a, b
        band = w > 0.5
        p85 = float(np.percentile(FY[band], 85))                     # old-map bright curls must not stand out
        FY = np.where(FY > p85, p85 + 0.2 * (FY - p85), FY).astype(np.float32)
        hpY = Lc - blur(Lc, 2)
        tgt = lowres_std(hpY * hpY, core.astype(np.float32))
        tgt = np.minimum(tgt, float(np.percentile(tgt[core], 75)))
        hpF = FY - blur(FY, 2)
        have = lowres_std(hpF * hpF, np.ones_like(M))
        del hpY, hpF
        FY += GRAIN * np.sqrt(np.maximum(tgt * tgt - have * have, 0)) * sphere_noise(H, W, (2, 4, 8, 16), 7, unit_hp=True)
        del tgt, have
        Lall = Lc * (1 - w) + FY * w
        if RELIEF_IMAGED_ONLY:
            img = w < 0.5
        del FY, w, band, core, Lc
    del O

    # --- polar caps: de-ripple + soften + grain in azimuthal projection (pole pinch, longitude streaks, ripple bands)
    Lall = polar(Lall, lambda P: cv2.GaussianBlur(cv2.medianBlur(P, 5), (0, 0), 2))
    # grain on low-Q terrain and the caps: isotropic 3-D noise on the sphere (no 1/cos lat stretch, no pole streaks)
    lat_r = 90 - (np.arange(H) + 0.5) * 180 / H
    gw = np.maximum((1 - Q) * M, smooth((np.abs(lat_r) - 70) / 10)[:, None].astype(np.float32))
    Lall = Lall * (1 + np.clip(0.6 * sd_hi * gw * sphere_noise(H, W, (4, 8, 16), 11), -0.6, 0.6))
    del gw

    # --- colour: natural-colour LUT on low-passed luminance x detail ratio (one palette for data and fill) ------
    Ls = blur(Lall, 6)
    lo, hi = np.percentile(Ls[M > 0], [1, 99.5])
    t = np.clip((Ls - lo) / max(hi - lo, 1e-3), 0, 1)
    ratio = np.clip(Lall / np.maximum(Ls, 1.0), 0, RATIO_MAX)
    del Lall, Ls
    col = np.empty((H, W, 3), np.float32)
    for i in range(3):
        col[..., i] = np.interp(t, LUT_T, LUT_RGB[:, i]).astype(np.float32) * ratio
    del t, ratio
    Y = luma(col)[..., None]
    col = Y + SAT * (col - Y)
    del Y
    # exposure + contrast: luma mean -> LUMA_MEAN, S-curve (slope 1 + CONTRAST at the mean), mid-tone saturation
    Y = luma(col)
    col *= LUMA_MEAN / float(Y.mean())
    Y = luma(col)
    m, A = float(Y.mean()), 110.0
    col *= ((m + A * np.tanh((1 + CONTRAST) * (Y - m) / A)) / np.maximum(Y, 1.0))[..., None]
    Y = luma(col)
    wm = (smooth((Y / 255 - 0.25) / 0.1) * smooth((0.85 - Y / 255) / 0.1))[..., None]   # luma 0.35..0.75
    col = Y[..., None] + (1 + (MID_SAT - 1) * wm) * (col - Y[..., None])
    del Y, wm
    print('colour luma mean %.1f' % float(luma(col).mean()), flush=True)
    col[..., 1] = np.minimum(col[..., 1], col[..., 0])               # no green / blue anywhere
    col[..., 2] = np.minimum(col[..., 2], col[..., 1] + 2)

    # --- relief ---------------------------------------------------------------------------------------------------
    h, dcov = load_dem(raw, W, H)
    h = wrapfix_dem(h, (dcov > 0.99).astype(np.float32))                # source columns: lon 0 | 360 meet here
    h, dcov = np.roll(h, half, 1), np.roll(dcov, half, 1)
    if al['applied']:
        if al['flip']:
            h, dcov = h[:, ::-1].copy(), dcov[:, ::-1].copy()
        h, dcov = np.roll(h, shift, 1), np.roll(dcov, shift, 1)
    D = ((dcov > 0.99) & img).astype(np.float32)                    # (both in the aligned frame)
    del img, dcov
    print('DEM gap fraction %.3f' % (1 - D.mean()), flush=True)
    h = prep_height(h, D)
    lp = nblur(h, D, 8)                                               # clamp spikes > 6 sigma of the local high-pass
    hp = (h - lp) * D
    sg = np.sqrt(nblur(hp * hp, D, 64))
    h = np.where(D > 0, lp + np.clip(hp, -6 * sg, 6 * sg), 0).astype(np.float32)
    del lp, hp, sg
    if D.min() < 1:
        d1 = cv2.resize(h * D, (1024, 512), interpolation=cv2.INTER_AREA)
        m1 = cv2.resize(D, (1024, 512), interpolation=cv2.INTER_AREA)
        d1 = np.where(m1 > 0.5, d1 / np.maximum(m1, 1e-4), 0).astype(np.float32)
        inp = cv2.inpaint(d1, (m1 <= 0.5).astype(np.uint8), 8, cv2.INPAINT_TELEA)
        inp = cv2.resize(blur(inp, 2), (W, H), interpolation=cv2.INTER_LINEAR)
        wv = 1 - smooth(dist(D > 0) / FEATHER)                       # data side: data -> smooth inpaint
        h = h * (1 - wv) + inp * wv
        del inp, d1, m1, wv
    wlp = smooth((0.55 - Q) / 0.2)                                    # low-resolution imagery: DEM low-passed (sigma 4)
    h = h * (1 - wlp) + blur(h, 4) * wlp
    del wlp
    h = polar(h, lambda P: cv2.GaussianBlur(cv2.medianBlur(P, 5), (0, 0), 2))   # polar caps smoothed in azimuthal grid
    qd = (Q * smooth(dist(D > 0) / 128)).astype(np.float32)           # relief strength: Q x DEM-valid feathered 128 px
    lat_r = 90 - (np.arange(H) + 0.5) * 180 / H
    qd *= (1 - smooth((np.abs(lat_r) - 82) / 6))[:, None].astype(np.float32)   # relief fades out over |lat| 82..88
    del Q
    hp = (h - blur(h, 10)) * qd
    sd = float(hp[D > 0].std())
    ao = np.clip(1 + AO * np.tanh(hp / (3 * sd)), 0.55, 1.25)[..., None]
    col = col * ao
    del hp, ao, D
    if KNEE is not None:
        T = KNEE * 255
        col = np.where(col > T, T + (255 - T) * (1 - np.exp(-(col - T) / (255 - T))), col)
    T = TOE * 255                                                     # soft toe: shadows never crush to black
    col = np.where(col < T, T - 0.6 * T * (1 - np.exp(np.minimum(col - T, 0) / (0.6 * T))), col)   # floor 0.4 T
    c8 = np.clip(np.rint(np.nan_to_num(col, nan=128.0)), 0, 255).astype(np.uint8)
    print('non-finite colour px', int((~np.isfinite(col)).any(2).sum()), flush=True)
    del col
    for kk, (w, hh) in SIZES.items():
        cc = c8 if kk == 8 else cv2.resize(c8, (w, hh), interpolation=cv2.INTER_AREA)
        webp(cc, os.path.join(OUT, '%s_color_%dk.webp' % (KEY, kk)), 90)
        hk = h if kk == 8 else cv2.resize(h, (w, hh), interpolation=cv2.INTER_AREA)
        fk = qd if kk == 8 else cv2.resize(qd, (w, hh), interpolation=cv2.INTER_AREA)
        webp(normals(hk, fk), os.path.join(OUT, '%s_normal_%dk.webp' % (KEY, kk)), 92)
    meta = {'v': 1, 'built': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'radius_km': R_KM,
            'projection': 'equirectangular, north up, east to the right; ' + (
                'rolled onto the previous site map, lon %g at the centre' % al['centre_lon'] if al['applied'] else 'lon 0 at the centre'),
            'align': al,
            'normal': 'tangent space x=east y=north z=up (OpenGL), true scale from the New Horizons DEM 300 m, xy scaled by the resolution map Q '
                      'x DEM coverage (feather 128 px); flat where the DEM has no data',
            'colour': 'natural-colour LUT on low-passed (sigma 6 px @8k) New Horizons luminance x detail ratio, saturation %g, B <= G + 2, G <= R; '
                      'approach imagery de-rippled + grain; unimaged south: previous map luminance (highs soft-clamped at p85, wavy ramp %d px, '
                      'grain %g) through the same LUT; relief occlusion %g' % (SAT, RAMP, GRAIN, AO),
            'files': sorted(f for f in os.listdir(OUT) if f.endswith('.webp')),
            'source': 'USGS Astrogeology: Pluto New Horizons Global Mosaic 300m (Jul 2017), Pluto New Horizons Global DEM 300m (Jul 2017)',
            'credit': 'NASA/JHUAPL/SwRI (New Horizons), USGS Astrogeology'}
    json.dump(meta, open(os.path.join(OUT, 'meta.json'), 'w'), indent=1)
    print('done in', round(time.time() - t0), 's')


def prep_height(h, D):
    return h


if __name__ == '__main__':
    main()
