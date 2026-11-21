"""Global height field (km above a best-fit triaxial ellipsoid) from a NAIF DSK shape model (type 2 plates), resampled
to an equirectangular grid (lat 90..-90 top to bottom, lon -180..180 left to right, east positive).
Needs: pip install spiceypy scipy"""
import numpy as np


def read_vertices(path, chunk=200000):
    import spiceypy as sp
    h = sp.dasopr(path)
    try:
        dla = sp.dlabfs(h)
        nv, npl = sp.dskz02(h, dla)
        out = []
        s = 1
        while s <= nv:
            v = sp.dskv02(h, dla, s, min(chunk, nv - s + 1))
            out.append(np.asarray(v, np.float64))
            s += len(v)
        return np.vstack(out)
    finally:
        sp.dascls(h)


def fit_ellipsoid(V):
    A = V * V
    sol, *_ = np.linalg.lstsq(A, np.ones(len(V)), rcond=None)
    return tuple(1.0 / np.sqrt(sol))


def dsk_height(path, W=4096, H=2048):
    from scipy.spatial import cKDTree
    V = read_vertices(path)
    print('  DSK vertices: %d' % len(V), flush=True)
    a, b, c = fit_ellipsoid(V)
    r = np.linalg.norm(V, axis=1)
    U = V / r[:, None]
    er = 1.0 / np.sqrt((U[:, 0] / a) ** 2 + (U[:, 1] / b) ** 2 + (U[:, 2] / c) ** 2)
    hv = (r - er).astype(np.float32)
    print('  ellipsoid a,b,c = %.3f %.3f %.3f km; vertex height min %.3f max %.3f std %.3f km' % (a, b, c, hv.min(), hv.max(), hv.std()), flush=True)
    del V, r, er
    tree = cKDTree(U)
    lon = np.radians((np.arange(W) + 0.5) / W * 360 - 180)
    out = np.empty((H, W), np.float32)
    rows = 64
    for r0 in range(0, H, rows):
        la = np.radians(90 - (np.arange(r0, min(H, r0 + rows)) + 0.5) * 180 / H)[:, None]
        d = np.stack([np.cos(la) * np.cos(lon), np.cos(la) * np.sin(lon), np.broadcast_to(np.sin(la), (la.shape[0], W))], -1).reshape(-1, 3)
        dist, idx = tree.query(d, k=6, workers=-1)
        w = 1.0 / np.maximum(dist, 1e-9) ** 2
        out[r0:r0 + la.shape[0]] = ((hv[idx] * w).sum(1) / w.sum(1)).reshape(la.shape[0], W)
    print('  DSK grid height min %.3f max %.3f std %.3f km' % (out.min(), out.max(), out.std()), flush=True)
    return out, (a, b, c)
