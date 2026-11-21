#!/usr/bin/env python3
"""Ceres surface maps for Cosmos (USGS Astrogeology / DLR, public domain):
  image  : Ceres Dawn FC global mosaic 140 m (DLR, Feb 2016, clear filter, ISIS cube 21093x10546, 8-bit, 1 = no data /
           shadow; lon -180..180, Kait longitude system)
  relief : Ceres Dawn FC HAMO DTM DLR global 60 ppd (Oct 2016, int16 metres above R = 470 km, lon 0..360)
Colour: Ceres is near-neutral grey -- the mosaic's own albedo at every scale, tinted with the previous site map's mean
colour (half saturation) when the previous map does not correlate with the mosaic (it is an artistic map), else its
low-passed chroma x the mosaic detail; shadows and the unimaged south pole are filled from the surroundings
(push-pull) with grain matched to the local detail; Occator's faculae keep a soft-knee highlight. See meta.align.
Outputs (out/): ceres_{color,normal}_{2,4,8}k.webp + meta.json -- equirectangular, north up, east to the right;
normal map tangent space x = east, y = north, z = up (OpenGL), true scale (oblateness removed).
Needs: pip install numpy pillow opencv-python.   Usage: build_ceres.py [raw_dir]"""
import json, os, re, sys, time, urllib.request
import numpy as np
import cv2
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
KEY = 'ceres'
USGS = 'https://planetarymaps.usgs.gov/mosaic/'
MOSAIC, DEM = 'Ceres_Dawn_FC_DLR_global_59ppd_Feb2016.cub', 'Ceres_Dawn_FC_HAMO_DTM_DLR_Global_60ppd_Oct2016.tif'
OLD = 'https://raw.githubusercontent.com/LykosTR/yzr-net/data/textures/ceres/ceres_color_4k.webp'   # previous site map
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
R_KM = 470.0                                # TIFF label: A/B/C_AXIS_RADIUS 470.0 km (heights are relative to it)
SIZES = {8: (8192, 4096), 4: (4096, 2048), 2: (2048, 1024)}
AO = 0.18                                   # relief occlusion strength
FEATHER = 24                                # px at 8k: DEM gap blend ramp
RAMP, PERT = 24, 4                          # px at 8k: gap blend ramp for shadows (keep it tight)
RAMP_POLE, PERT_POLE = 128, 48              # px at 8k: wavy ramp for the big unimaged south-pole gap
POLE_LAT = (60, 75)                         # baked polar lighting flattened, blend smoothstep over |lat| (deg)
POLE_ROWS = 6                               # first / last rows replaced by the mean of the next 7 (rows collapse to a point)
GRAIN = 0.45                                # grain in filled areas: fraction of the missing high-frequency std (shadow-dense poles)
RELIEF_IMAGED_ONLY = False                   # relief only where mission imagery is shown (DEM rim beyond it is noise)
SIG_LP = 24                                 # px at 8k: chroma / large-scale albedo from the previous map below this scale
RATIO_MAX = 8.0                             # clip of the mosaic detail ratio L / lowpass L
SAT = 0.5                                   # chroma gain on the previous map's colour
KNEE = 0.9                                  # soft-knee start for highlights (fraction of 255): Occator's faculae
TOE = 0.12                                  # soft-toe start for shadows (fraction of 255)
NCC_MIN = 0.35
MASK_COV, MASK_ERODE = 0.5, 1                # shadows are speckled: a pixel is valid when most of its source area is lit (no erosion)


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


def fractal(H, W, seed):
    """unit-std grain (octaves 0.7..5.6 px), wrapping in longitude; scaled so its sigma-2 high-pass has unit std"""
    rng = np.random.default_rng(seed)
    n = np.zeros((H, W), np.float32)
    for s in (0.7, 1.4, 2.8, 5.6, 11.2):
        g = blur(rng.standard_normal((H, W), dtype=np.float32), s)
        n += g / (g.std() + 1e-6)
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


def normals(h, flat=None):
    H, W = h.shape
    dlon, dlat = 2 * np.pi / W, np.pi / H
    lat = np.pi / 2 - (np.arange(H) + 0.5) * dlat
    coslat = np.maximum(np.cos(lat), 0.02)[:, None]
    hx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) / (2 * R_KM * coslat * dlon)
    up, dn = np.vstack([h[:1], h[:-1]]), np.vstack([h[1:], h[-1:]])
    hy = (up - dn) / (2 * R_KM * dlat)
    n = np.dstack([-hx, -hy, np.ones_like(h)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    if flat is not None:   # fade to (0, 0, 1) where the DEM has no data
        n *= (1 - flat)[..., None]; n[..., 2] += flat
        n /= np.linalg.norm(n, axis=2, keepdims=True)
    return np.clip(np.rint((n * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8)


class Cube:
    """tiled 8-bit ISIS3 cube, read-only: cube[a:b] -> rows a..b-1"""
    def __init__(self, path):
        lab = open(path, 'rb').read(65536).decode('latin1')
        g = lambda k: int(re.search(r'^\s*' + k + r'\s*=\s*(\d+)', lab, re.M).group(1))
        assert 'Format      = Tile' in lab and 'UnsignedByte' in lab, 'unexpected cube layout'
        self.shape = (g('Lines'), g('Samples'))
        ts, tl = g('TileSamples'), g('TileLines')
        nr, nc = -(-self.shape[0] // tl), -(-self.shape[1] // ts)
        self.tl = tl
        self.mm = np.memmap(path, np.uint8, 'r', offset=g('StartByte') - 1, shape=(nr, nc, tl, ts))

    def __getitem__(self, sl):
        a, b = sl.start, sl.stop
        t0, t1 = a // self.tl, (b - 1) // self.tl
        blk = np.asarray(self.mm[t0:t1 + 1]).transpose(0, 2, 1, 3).reshape((t1 - t0 + 1) * self.tl, -1)
        return blk[a - t0 * self.tl:b - t0 * self.tl, :self.shape[1]]


def load_mosaic(raw, W, H):
    return area(Cube(src(raw, MOSAIC)), W, H, nodata=1)            # 1 = no data (south pole) and full shadow


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
    half = W // 2                                                     # DTM: lon 0..360 -> lon 0 at the centre (mosaic already is)
    L, cov = load_mosaic(raw, W, H)
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

    # --- polar lighting: high incidence bakes in streaks and black shadows -> flatten toward L / local mean --------
    lat = 90 - (np.arange(H) + 0.5) * 180 / H
    wl = smooth((np.abs(lat) - POLE_LAT[0]) / (POLE_LAT[1] - POLE_LAT[0]))[:, None].astype(np.float32)
    gm = float(L[M > 0].mean())
    Lf = np.where(M > 0, L / np.maximum(nblur(L, M, 40), 1.0) * gm, 0).astype(np.float32)
    m1 = cv2.resize(M, (1024, 512), interpolation=cv2.INTER_AREA)
    med = pushpull(cv2.resize(Lf, (1024, 512), interpolation=cv2.INTER_AREA) / np.maximum(m1, 1e-4), (m1 > 0.5).astype(np.float32))
    med = cv2.resize(cv2.medianBlur(med.astype(np.float32), 5), (W, H), interpolation=cv2.INTER_LINEAR)
    r = Lf / np.maximum(med, 1.0)
    Lf = med * (1 + 0.3 * np.tanh((r - 1) / 0.3))                     # soft clamp to [0.7, 1.3] x local median
    L = np.where(M > 0, L * (1 - wl) + Lf * wl, L).astype(np.float32)
    del Lf, med, r, m1, wl

    # --- colour: previous map's low-passed colour x the mosaic's detail ratio -------------------------------------
    Olp = np.dstack([blur(O[..., i], SIG_LP * O.shape[1] / W) for i in range(3)])            # at the old map's size
    Ylp = luma(Olp)[..., None]
    Olp = Ylp + SAT * (Olp - Ylp)
    del Ylp
    lpL = nblur(L, M, SIG_LP)
    ratio = np.clip(np.where(M > 0, L / np.maximum(lpL, 1.0), 1.0), 0, RATIO_MAX)
    del lpL
    col = np.empty((H, W, 3), np.float32)
    if al['applied']:
        for i in range(3):
            col[..., i] = cv2.resize(Olp[..., i], (W, H), interpolation=cv2.INTER_LINEAR) * ratio
    else:   # unrelated old map: keep the mosaic's own albedo at every scale, only the old map's mean tint
        tint = Olp.reshape(-1, 3).mean(0); tint = tint / luma(tint[None])[0]
        for i in range(3):
            col[..., i] = L * tint[i]
    del ratio, Olp, L

    # --- gaps: feathered fill (aligned previous map, else push-pull from the surroundings) + matched grain ---------
    if M.min() < 1:
        rng = np.random.default_rng(3)
        pert = cv2.resize(blur(rng.standard_normal((H // 32, W // 32), dtype=np.float32), 2), (W, H), interpolation=cv2.INTER_CUBIC)
        pert = np.clip(pert / (pert.std() + 1e-6), -1, 1)
        w = 1 - smooth((dist(M > 0) - PERT * (1 + pert)) / RAMP)       # 1 in the gap, ramps to 0 inside the data
        # the big south-pole gap (components touching the bottom row, at 1k) gets a wide wavy ramp and the ring mean
        g1 = (cv2.resize(M, (1024, 512), interpolation=cv2.INTER_AREA) < 0.5).astype(np.uint8)
        n, lab = cv2.connectedComponents(g1, connectivity=8)
        big = np.isin(lab, [k for k in np.unique(lab[-1]) if k > 0])
        G = (cv2.resize(big.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) > 0.5) & (M < 1)
        dG = dist(~G)
        w = np.maximum(w, 1 - smooth((dG - PERT_POLE * (1 + pert)) / RAMP_POLE))
        ring = (~G) & (dG < 256) & (M > 0) & (w < 1e-3)
        ringc = col[ring].mean(0) if ring.any() else col[M > 0].mean(0)
        gb = smooth(dist(G) / RAMP_POLE)[..., None]                    # inside the pole gap: push-pull -> ring mean
        del pert, g1, lab, big, dG, ring
        Y = luma(col)
        hpY = Y - blur(Y, 2)
        tgt = lowres_std(hpY * hpY, (M > 0).astype(np.float32) * (w < 1e-3))
        tgt = np.minimum(tgt, float(np.percentile(tgt[w < 1e-3], 75)))  # rough rim terrain must not set the grain
        del hpY
        F = np.empty_like(col)
        for i in range(3):
            F[..., i] = (cv2.resize(O[..., i], (W, H), interpolation=cv2.INTER_CUBIC) if al['applied']
                         else pushpull(col[..., i], (w < 1e-3).astype(np.float32)))
        F = F * (1 - gb) + ringc[None, None, :] * gb
        del gb, G
        YF = luma(F)
        hpF = YF - blur(YF, 2)
        have = lowres_std(hpF * hpF, np.ones_like(M))
        del hpF
        amp = GRAIN * np.sqrt(np.maximum(tgt * tgt - have * have, 0))
        g = np.clip(1 + amp * fractal(H, W, 7) / np.maximum(YF, 8.0), 0.6, 1.5)
        del amp, tgt, have, YF
        for i in range(3):
            col[..., i] = col[..., i] * (1 - w) + F[..., i] * g * w
        img = (w < 0.5) if RELIEF_IMAGED_ONLY else np.ones((H, W), bool)
        del F, g, w, Y
    else:
        img = np.ones((H, W), bool)

    # --- exposure: global luminance mean / std of the previous map ------------------------------------------------
    Y, YO = luma(col), luma(O)
    k = float(YO.std() / max(Y.std(), 1e-3))
    col = (col - float(Y.mean())) * k + float(YO.mean())
    print('luma mean/std new %.1f/%.1f -> old %.1f/%.1f' % (Y.mean(), Y.std(), YO.mean(), YO.std()), flush=True)
    del Y, YO, O

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
    wg = np.zeros_like(D)
    if D.min() < 1:
        d1 = cv2.resize(h * D, (1024, 512), interpolation=cv2.INTER_AREA)
        m1 = cv2.resize(D, (1024, 512), interpolation=cv2.INTER_AREA)
        d1 = np.where(m1 > 0.5, d1 / np.maximum(m1, 1e-4), 0).astype(np.float32)
        inp = cv2.inpaint(d1, (m1 <= 0.5).astype(np.uint8), 8, cv2.INPAINT_TELEA)
        inp = cv2.resize(blur(inp, 2), (W, H), interpolation=cv2.INTER_LINEAR)
        wv = 1 - smooth(dist(D > 0) / FEATHER)                       # data side: data -> smooth inpaint
        h = h * (1 - wv) + inp * wv
        wg = smooth(dist(D < 1) / FEATHER)                            # gap side: normals -> flat
        del inp, d1, m1, wv
    hp = (h - blur(h, 10)) * (1 - wg)
    sd = float(hp[D > 0].std())
    ao = np.clip(1 + AO * np.tanh(hp / (3 * sd)), 0.55, 1.25)[..., None]
    col = col * ao
    del hp, ao, D
    if KNEE is not None:
        T = KNEE * 255
        col = np.where(col > T, T + (255 - T) * (1 - np.exp(-(col - T) / (255 - T))), col)
    for a, b in ((slice(0, POLE_ROWS), slice(POLE_ROWS, 2 * POLE_ROWS + 1)), (slice(H - POLE_ROWS, H), slice(H - 2 * POLE_ROWS - 1, H - POLE_ROWS))):
        col[a] = col[b].mean((0, 1))                                  # pole rows collapse to a point on the sphere
        h[a] = h[b].mean()
    T = TOE * 255                                                     # soft toe: shadows never crush to black
    col = np.where(col < T, T - 0.6 * T * (1 - np.exp(np.minimum(col - T, 0) / (0.6 * T))), col)   # floor 0.4 T
    c8 = np.clip(np.rint(col), 0, 255).astype(np.uint8)
    del col
    for kk, (w, hh) in SIZES.items():
        cc = c8 if kk == 8 else cv2.resize(c8, (w, hh), interpolation=cv2.INTER_AREA)
        webp(cc, os.path.join(OUT, '%s_color_%dk.webp' % (KEY, kk)), 90)
        hk = h if kk == 8 else cv2.resize(h, (w, hh), interpolation=cv2.INTER_AREA)
        fk = wg if kk == 8 else cv2.resize(wg, (w, hh), interpolation=cv2.INTER_AREA)
        webp(normals(hk, fk if wg.max() > 0 else None), os.path.join(OUT, '%s_normal_%dk.webp' % (KEY, kk)), 92)
    meta = {'v': 1, 'built': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'radius_km': R_KM,
            'projection': 'equirectangular, north up, east to the right; ' + (
                'rolled onto the previous site map, lon %g at the centre' % al['centre_lon'] if al['applied'] else 'lon 0 at the centre'),
            'align': al,
            'normal': 'tangent space x=east y=north z=up (OpenGL), true scale from the Dawn HAMO DTM (DLR, 60 ppd), oblateness removed',
            'colour': ('previous site map low-passed (sigma %d px @8k) x Dawn mosaic detail' % SIG_LP if al['applied'] else
                       'Dawn mosaic albedo x the previous site map mean tint') +
                      ' (saturation %g); |lat| > %g..%g lighting flattened (L / local mean, soft clamp 0.7..1.3); shadows filled (ramp %d px), south pole '
                      '(wavy ramp %d px, ring mean) + matched grain %g; luminance mean/std of the previous map; highlight soft knee %g; pole rows '
                      'averaged; relief occlusion %g' % (SAT, POLE_LAT[0], POLE_LAT[1], RAMP, RAMP_POLE, GRAIN, KNEE, AO),
            'files': sorted(f for f in os.listdir(OUT) if f.endswith('.webp')),
            'source': 'USGS Astrogeology: Ceres Dawn FC DLR global mosaic 59ppd/140m (Feb 2016), Ceres Dawn FC HAMO DTM DLR global 60ppd (Oct 2016)',
            'credit': 'NASA/JPL-Caltech/UCLA/MPS/DLR/IDA (Dawn), USGS Astrogeology'}
    json.dump(meta, open(os.path.join(OUT, 'meta.json'), 'w'), indent=1)
    print('done in', round(time.time() - t0), 's')


def prep_height(h, D):
    """heights are above a 470 km sphere: remove Ceres' oblateness (zonal fit in sin^2 lat, degree 2, cos-lat weighted)"""
    H = h.shape[0]
    lat = np.pi / 2 - (np.arange(H) + 0.5) * np.pi / H
    cnt = D.sum(1)
    ok = cnt > 0.5 * h.shape[1]
    rm = (h * D).sum(1) / np.maximum(cnt, 1)
    x = np.sin(lat) ** 2
    A = np.vstack([np.ones_like(x), x, x * x]).T
    wt = np.sqrt(np.cos(lat))[ok]
    cf = np.linalg.lstsq(A[ok] * wt[:, None], rm[ok] * wt, rcond=None)[0]
    print('oblateness fit (km): %.2f %+.2f s2 %+.2f s4' % tuple(cf), flush=True)
    return (h - (A @ cf)[:, None]).astype(np.float32)


if __name__ == '__main__':
    main()
