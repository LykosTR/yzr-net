#!/usr/bin/env python3
"""Earth atmosphere look-up tables for Cosmos (physically based sky, Hillaire 2020 style).

Outputs (data/atmosphere/out/):
  earth.bin  -- 256-byte float32 header (model parameters) + two RGBA float16 tables:
                transmittance T(r, mu)      TW x TH  (Bruneton 2017 parametrisation)
                multiple scattering Psi(r, mu_s)  MW x MH  (Hillaire 2020, isotropic 2nd order / (1 - f_ms))
  meta.json  -- version, build time, parameters (human readable)

Units: lengths in Earth radii (1 = 6371 km), coefficients per Earth radius. The renderer marches single scattering
per pixel with these two tables, so the sky, the limb seen from orbit, twilight and the haze over the ground all come
from one model in every map engine.

Geometry exaggeration X: every height (scale heights, ozone layer, top) is multiplied by X and every coefficient divided
by X, so optical depths (= colours) stay physical while the limb stays readable at globe scale.
"""
import json, os, struct, time
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
VERSION = 1
RE_M = 6371000.0          # 1 unit = mean Earth radius (scene unit)
X = 1.6                   # geometry exaggeration (heights x X, coefficients / X)
TURBIDITY = 2.5           # aerosol (Mie) density multiplier: the pale, hazy horizon of the reference look

# physical model (Hillaire 2020 / Bruneton 2017 Earth defaults), per metre and metres
BETA_R = np.array([5.802e-6, 13.558e-6, 33.100e-6])
H_R = 8000.0
BETA_MS = 3.996e-6 * TURBIDITY
BETA_MA = 4.400e-6 * TURBIDITY
H_M = 1200.0
G_MIE = 0.80
BETA_O = np.array([0.650e-6, 1.881e-6, 0.085e-6])
OZ_C, OZ_W = 25000.0, 15000.0   # ozone tent: 1 - |h - c| / w
TOP_M = 100000.0
ALBEDO = 0.30

TW, TH = 256, 64          # transmittance (mu, r)
MW, MH = 32, 32           # multiple scattering (mu_s, r)

# scene units
k = RE_M / X
bR, bMs, bMa, bO = BETA_R * k, BETA_MS * k, BETA_MA * k, BETA_O * k
hR, hM = H_R * X / RE_M, H_M * X / RE_M
ozC, ozW = OZ_C * X / RE_M, OZ_W * X / RE_M
RT = 1.0 + TOP_M * X / RE_M
RG = 1.0


def density(h):
    """h: altitude (units) -> (rayleigh, mie, ozone) densities"""
    h = np.maximum(h, 0.0)
    return np.exp(-h / hR), np.exp(-h / hM), np.maximum(0.0, 1.0 - np.abs(h - ozC) / ozW)


def extinction(h):
    dr, dm, do = density(h)
    return dr[..., None] * bR + (dm * (bMs + bMa))[..., None] + do[..., None] * bO


def dist_top(r, mu):
    return -r * mu + np.sqrt(np.maximum(r * r * (mu * mu - 1.0) + RT * RT, 0.0))


def dist_ground(r, mu):
    return -r * mu - np.sqrt(np.maximum(r * r * (mu * mu - 1.0) + RG * RG, 0.0))


def hits_ground(r, mu):
    return (mu < 0.0) & (r * r * (mu * mu - 1.0) + RG * RG >= 0.0)


H_ = np.sqrt(RT * RT - RG * RG)


def trans_uv_to_rmu(u, v):
    """Bruneton GetRMuFromTransmittanceTextureUv (u: mu axis, v: r axis), texel centres in [0, 1]"""
    rho = H_ * v
    r = np.sqrt(rho * rho + RG * RG)
    dmin = RT - r
    dmax = rho + H_
    d = dmin + u * (dmax - dmin)
    mu = np.where(d == 0.0, 1.0, (H_ * H_ - rho * rho - d * d) / np.maximum(2.0 * r * d, 1e-12))
    return r, np.clip(mu, -1.0, 1.0)


def build_transmittance():
    u = (np.arange(TW) + 0.5) / TW
    v = (np.arange(TH) + 0.5) / TH
    U, V = np.meshgrid(u, v)                # [TH, TW]
    r, mu = trans_uv_to_rmu(U, V)
    n = 500
    d = dist_top(r, mu)
    t = (np.arange(n) + 0.5)[None, None, :] / n * d[..., None]
    rr = np.sqrt(r[..., None] ** 2 + t * t + 2.0 * r[..., None] * mu[..., None] * t)
    ext = extinction(rr - RG)               # [TH, TW, n, 3]
    od = ext.sum(axis=2) * (d / n)[..., None]
    return np.exp(-od)                      # [TH, TW, 3]


def trans_lookup(T, r, mu):
    """bilinear lookup of the transmittance table (same mapping as the shader)"""
    rho = np.sqrt(np.maximum(r * r - RG * RG, 0.0))
    d = dist_top(r, mu)
    dmin, dmax = RT - r, rho + H_
    u = (d - dmin) / np.maximum(dmax - dmin, 1e-12)
    v = rho / H_
    x = np.clip(u * TW - 0.5, 0, TW - 1)
    y = np.clip(v * TH - 0.5, 0, TH - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, TW - 1), np.minimum(y0 + 1, TH - 1)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    return (T[y0, x0] * (1 - fx) * (1 - fy) + T[y0, x1] * fx * (1 - fy) +
            T[y1, x0] * (1 - fx) * fy + T[y1, x1] * fx * fy)


def sun_vis(r, mu_s):
    """0 when the planet hides the sun (sharp: the renderer softens it)"""
    return (~hits_ground(r, mu_s)).astype(np.float64)


def build_ms(T):
    nd, ns = 64, 40
    i = np.arange(nd) + 0.5                 # Fibonacci sphere directions
    phi = np.arccos(1.0 - 2.0 * i / nd)
    th = np.pi * (1.0 + 5 ** 0.5) * i
    dirs = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], -1)   # [nd, 3]
    mus = (np.arange(MW) + 0.5) / MW * 2.0 - 1.0
    rs = RG + (np.arange(MH) + 0.5) / MH * (RT - RG)
    out = np.zeros((MH, MW, 3))
    for j, r in enumerate(rs):
        for q, mu_s in enumerate(mus):
            sun = np.array([np.sqrt(max(1 - mu_s * mu_s, 0.0)), 0.0, mu_s])
            mu = dirs[:, 2]                                  # x = (0, 0, r): view zenith cosine = dir.z
            g = hits_ground(np.full(nd, r), mu)
            tmax = np.where(g, dist_ground(np.full(nd, r), mu), dist_top(np.full(nd, r), mu))
            t = (np.arange(ns) + 0.5)[None, :] / ns * tmax[:, None]   # [nd, ns]
            dt = (tmax / ns)[:, None]
            P = np.array([0.0, 0.0, r])[None, None, :] + dirs[:, None, :] * t[..., None]
            pr = np.linalg.norm(P, axis=-1)
            pmu_s = (P @ sun) / pr
            dr, dm, do = density(pr - RG)
            sig_s = dr[..., None] * bR + (dm * bMs)[..., None]
            ext = sig_s + (dm * bMa)[..., None] + do[..., None] * bO
            od = np.cumsum(ext * dt[..., None], axis=1) - ext * dt[..., None] * 0.5
            Tv = np.exp(-od)
            Ts = trans_lookup(T, pr, pmu_s) * sun_vis(pr, pmu_s)[..., None]
            L = (Tv * sig_s * Ts * dt[..., None]).sum(1) / (4.0 * np.pi)
            F = (Tv * sig_s * dt[..., None]).sum(1)
            # ground bounce (Lambert)
            Pg = np.array([0.0, 0.0, r])[None, :] + dirs * tmax[:, None]
            gmu = (Pg @ sun) / np.linalg.norm(Pg, axis=-1)
            Tg = np.exp(-(ext * dt[..., None]).sum(1))
            Lg = Tg * trans_lookup(T, np.full(nd, RG), gmu) * (np.maximum(gmu, 0.0) * ALBEDO / np.pi)[:, None]
            L = L + np.where(g[:, None], Lg, 0.0)
            L2, fms = L.mean(0), F.mean(0)
            out[j, q] = L2 / (1.0 - fms)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    T = build_transmittance()
    M = build_ms(T)
    hdr = np.zeros(64, np.float32)
    vals = [0.0, X, RT, hR, hM, ozC, ozW, *bR, bMs, bMa, *bO, G_MIE, TW, TH, MW, MH, ALBEDO]
    hdr[:len(vals)] = vals

    def rgba16(a):
        h, w, _ = a.shape
        o = np.ones((h, w, 4), np.float32)
        o[..., :3] = a
        return o.astype(np.float16).tobytes()
    with open(os.path.join(OUT, 'earth.bin'), 'wb') as f:
        f.write(b'SRAT')                    # magic in slot 0, then 63 float32 parameters (256-byte header)
        f.write(hdr[1:].tobytes())
        f.write(rgba16(T))
        f.write(rgba16(M))
    meta = {'version': VERSION, 'built': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'exaggeration': X, 'turbidity': TURBIDITY, 'top_km': TOP_M / 1000.0,
            'transmittance': [TW, TH], 'multiscatter': [MW, MH],
            'header': 'magic SRAT + float32[63]: X, rTop, hR, hM, ozC, ozW, bR(3), bMs, bMa, bO(3), g, TW, TH, MW, MH, albedo (units: Earth radii)'}
    with open(os.path.join(OUT, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print('T min %.4g max %.4g | MS max %s | %.1fs' % (T.min(), T.max(), M.reshape(-1, 3).max(0), time.time() - t0))


if __name__ == '__main__':
    main()
