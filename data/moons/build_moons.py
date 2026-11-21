#!/usr/bin/env python3
"""Surface maps of the large moons for Cosmos (USGS Astrogeology mission mosaics, public domain):
  io       : Galileo SSI + Voyager global mosaic, ClrMerge 1 km (real colour)
  europa   : Voyager + Galileo SSI global mosaic 500 m (grey)
  ganymede : Voyager + Galileo SSI global mosaic 1 km (grey detail) + global colour mosaic 1435 m (chroma)
  callisto : Voyager + Galileo SSI global mosaic 1 km (grey)
  titan    : Cassini ISS P19658 global mosaic 4 km (938 nm, grey)
  dione    : Cassini + Voyager global mosaic 154 m (grey)
  iapetus  : Cassini + Voyager global mosaic 783 m (grey)
  haumea   : Astra-Planetshine artist's impression (colour + bump), used as published
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
S3 = 'https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/'   # bucket behind USGS (zips, sub-folders)
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
               mode='own', sat=0.85, linear=True, base='own', pole_flat=75, top=8, slope_deg=5, credit=GALILEO,
               source='USGS Astrogeology: Io Galileo SSI / Voyager Global Mosaic ClrMerge 1 km'),
    'europa': dict(name='Europa', radius_km=1562.09, grey='Europa_Voyager_GalileoSSI_global_mosaic_500m.tif', centre=180,
                   mode='lut', neutral=True, top=8, slope_deg=4, credit=GALILEO,
                   lut=[(0, (120, 92, 70)), (0.3, (170, 140, 112)), (0.6, (214, 200, 182)), (1, (240, 236, 228))],
                   source='USGS Astrogeology: Europa Voyager / Galileo SSI Global Mosaic 500 m'),
    'ganymede': dict(name='Ganymede', radius_km=2632.345, grey='Ganymede_Voyager_GalileoSSI_global_mosaic_1km.tif',
                     color='Ganymede_Voyager_GalileoSSI_Global_ClrMosaic_1435m.tif', centre=180, mode='chroma', top=8, slope_deg=7,
                     credit=GALILEO, source='USGS Astrogeology: Ganymede Voyager / Galileo SSI Global Mosaic 1 km, '
                                            'Ganymede Voyager / Galileo SSI Global Color Mosaic 1435 m'),
    'callisto': dict(name='Callisto', radius_km=2409.3, grey='Callisto_Voyager_GalileoSSI_global_mosaic_1km.tif', centre=180,
                     mode='lut', top=8, slope_deg=8, credit=GALILEO,
                     lut=[(0, (52, 46, 40)), (0.3, (88, 78, 68)), (0.6, (140, 130, 118)), (1, (222, 218, 210))],
                     source='USGS Astrogeology: Callisto Voyager / Galileo SSI Global Mosaic 1 km'),
    # disabled: the ISS frame patchwork shows as seams and the previous map's polar lakes would be lost
    'titan': dict(name='Titan', disabled=True, radius_km=2575.0, grey='Titan_ISS_P19658_Mosaic_Global_4km.tif', centre=180, mode='lut',
                  top=4, credit='NASA/JPL/Space Science Institute (Cassini ISS), USGS Astrogeology',
                  lut=[(0, (52, 36, 20)), (0.4, (120, 84, 44)), (0.7, (178, 130, 70)), (1, (214, 170, 104))],
                  source='USGS Astrogeology: Titan Cassini ISS P19658 Global Mosaic 4 km (938 nm)'),
    'dione': dict(name='Dione', disabled=True, radius_km=563.0, grey='Dione_Cassini_Voyager_mosaic_global_154m.tif', centre=0, mode='lut',
                  neutral=True, top=8, credit=CASSINI,
                  lut=[(0, (120, 118, 114)), (0.5, (184, 182, 178)), (1, (236, 236, 234))],
                  source='USGS Astrogeology: Dione Cassini / Voyager Global Mosaic 154 m'),
    'iapetus': dict(name='Iapetus', disabled=True, radius_km=736.0, grey='Iapetus_Cassini_Voyager_mosaic_global_783m.tif', centre=0,
                    mode='lut', neutral=True, pole_flat=75, top=8, credit=CASSINI,
                    lut=[(0, (48, 36, 28)), (0.25, (86, 64, 48)), (0.5, (160, 148, 132)), (1, (232, 228, 220))],
                    source='USGS Astrogeology: Iapetus Cassini / Voyager Global Mosaic 783 m'),
    # ---- catalogue moons (no previous site map: USGS orientation, exposure from the measured geometric albedo) ----
    'mimas': dict(name='Mimas', radius_km=198.2, grey='MI_170630_DLR_basemap_degrees.tif', old=False, mean=128,
                  zip=(S3 + 'Mimas/Cassini_DLR_Mimas.zip', 'Cassini_DLR/MI_170630_DLR_basemap_degrees.tif'), centre=0,
                  mode='lut', neutral=True, top=8, own_detail=True, slope_deg=7,
                  hybrid=dict(terrain='oberon', rot=(37, 211, -64), sig_deg=1.5, relief_km=8,
                              craters=[(-1.7, 248.2, 139, 10.0, 1.5, 6.0)]),
                  credit='NASA/JPL/Space Science Institute (Cassini ISS), DLR (Roatsch et al.), USGS Astrogeology',
                  lut=[(0, (112, 110, 106)), (0.5, (180, 178, 174)), (1, (234, 233, 230))],
                  source='DLR Cassini ISS Mimas global basemap (June 2017), USGS Astrogeology mosaic archive'),
    'enceladus': dict(name='Enceladus', radius_km=252.1, grey='Enceladus_Cassini_ISS_Global_Mosaic_100m_HPF.tif', old=False,
                      mean=176, centre=180, mode='lut', top=8, credit=CASSINI,
                      dsk='cas_enceladus_ssd_spc_1024icq_v2.bds', deshade=(24, 0.3),
                      lut=[(0, (156, 162, 166)), (0.5, (212, 218, 222)), (1, (246, 249, 252))],
                      source='USGS Astrogeology: Enceladus Cassini ISS Global Mosaic 100 m (high-pass filtered)'),
    'tethys': dict(name='Tethys', radius_km=531.1, grey='Tethys_Cassini_mosaic_global_293m.tif', old=False, mean=150, centre=0,
                   mode='lut', neutral=True, top=8, slope_deg=6, credit=CASSINI,
                   despot=2.2, hl=1.35, hyb_clip=(0.72, 1.35),
                   hybrid=dict(terrain='umbriel', rot=(-48, 77, 119), sig_deg=1.5, relief_km=9,
                               craters=[(32.8, 231.1, 445, 3.0, 1.0, 0.0, dict(floor=1.0, dome=1.5))],   # rough (cratered) floor kept
                               troughs=[dict(name='Ithaca Chasma', pole=(32.8, 231.1), centre_lon=321.1, half_deg=110, taper_deg=15,
                                             depth=3.5, w=50.0, wall=14.0, shoulder=0.8, shoulder_r=60.0, shoulder_w=12.0)]),
                   lut=[(0, (128, 126, 122)), (0.5, (192, 190, 186)), (1, (240, 239, 236))],
                   source='USGS Astrogeology: Tethys Cassini Global Mosaic 293 m'),
    'rhea': dict(name='Rhea', radius_km=763.8, grey='Rhea_Cassini_Voyager_mosaic_global_417m.tif', old=False, mean=124, centre=0,
                 mode='lut', neutral=True, top=8, slope_deg=6, credit=CASSINI,
                 lut=[(0, (116, 113, 108)), (0.5, (184, 181, 176)), (1, (236, 234, 230))],
                 source='USGS Astrogeology: Rhea Cassini / Voyager Global Mosaic 417 m'),
    'charon': dict(name='Charon', radius_km=606.0, grey='Charon_NewHorizons_Global_Mosaic_300m_Jul2017_8bit.tif', old=False,
                   mean=117, centre=180, mode='lut', neutral=True, top=8, rotfill=50,
                   dem='Charon_NewHorizons_Global_DEM_300m_Jul2017_16bit.tif', cap_tint=(68, 80, (128, 100, 88)),
                   credit='NASA/JHUAPL/SwRI (New Horizons LORRI / MVIC), USGS Astrogeology',
                   lut=[(0, (72, 70, 68)), (0.5, (142, 139, 134)), (1, (212, 208, 202))],
                   source='USGS Astrogeology: Charon New Horizons LORRI / MVIC Global Mosaic 300 m (July 2017)'),
    'triton': dict(name='Triton', radius_km=1353.4, color='Triton_Voyager2_ClrMosaic_GlobalFill_600m.tif', old=False, mean=124,
                   centre=0, mode='own', sat=0.15, neutral=True, top=8, slope_deg=5, flip=True, rotfill=60, rot_sig=20, mask_smooth=0.03, rim=24, tint=(236, 214, 200),
                   credit='NASA/JPL (Voyager 2), USGS Astrogeology',
                   source='USGS Astrogeology: Triton Voyager 2 Global Color Mosaic 600 m'),
    # disabled (s35): the client keeps Phobos on the shared asteroid rock surface
    'phobos': dict(name='Phobos', disabled=True, radius_km=11.1, grey='Phobos_Viking_Mosaic_40ppd_DLRcontrol.tif', old=False, mean=66, centre=0,
                   mode='lut', neutral=True, top=4, nonormal=True, credit='NASA/JPL (Viking Orbiter), DLR, USGS Astrogeology',
                   lut=[(0, (62, 56, 52)), (0.5, (120, 110, 102)), (1, (182, 172, 162))],
                   source='USGS Astrogeology: Phobos Viking Global Mosaic 40 ppd (DLR control)'),
}
VOYU = 'NASA/JPL (Voyager 2), USGS Astrogeology (T. Becker), JPL Solar System Simulator'
ART_PACK = 'https://drive.usercontent.google.com/download?id=1rHyIo5zdNGlTeM7PeXabCYb2PC8eC0RO&export=download&confirm=t'
for _k, _i, _r, _m, _rel, _lut in (
        ('ariel', 1, 578.9, 92, 10, [(0, (84, 83, 81)), (0.5, (150, 149, 147)), (1, (214, 212, 208))]),
        ('umbriel', 2, 584.7, 66, 10, [(0, (52, 51, 50)), (0.5, (104, 103, 101)), (1, (168, 165, 162))]),
        ('titania', 3, 788.9, 78, 10, [(0, (72, 68, 64)), (0.5, (136, 132, 127)), (1, (204, 196, 188))]),
        ('oberon', 4, 761.4, 72, 12, [(0, (68, 62, 57)), (0.5, (130, 125, 119)), (1, (198, 188, 178))]),
        ('miranda', 5, 235.8, 76, 20, [(0, (72, 71, 70)), (0.5, (138, 137, 135)), (1, (206, 204, 200))])):
    _A = {'umbriel': 'Umbirel'}.get(_k, _k.capitalize())             # (the pack's own spelling)
    _x = '.Png' if _k == 'ariel' else '.png'
    # s37: hybrid = Astra-Planetshine's full-globe map (sharp, unseen half painted) x the real Voyager large-scale albedo
    BODIES[_k] = dict(name=_k.capitalize(), radius_km=_r, art=(ART_PACK, 'uranus_project_missing_data.zip'),
                      hl={'umbriel': 1.6}.get(_k),   # Wunda's ring is real (~2x) but painted pure white
                      art_color=_A + 'Colour' + _x, art_bump=_A + 'Bump' + _x, relief_km=_rel,
                      grey='ura%dvuu2.tif' % _i, url='https://maps.jpl.nasa.gov/tmaps/pix/', last_col_bad=True,
                      mean=_m, top=8, flip=True, rim=40, pole_cut=72, lut=_lut,
                      credit='Astra-Planetshine (deviantart.com/astra-planetshine, "Uranus Project Missing Data"); ' + VOYU,
                      source='Astra-Planetshine "Uranus Project Missing Data" colour + bump maps x USGS Voyager 2 airbrush mosaic '
                             '(southern hemisphere), JPL Solar System Simulator map %d/%s' % (_i, _k.capitalize()))
# s37: Gonggong / Sedna -- no spacecraft image: illustrative globes (a Uranian moon's terrain rotated + recoloured)
# colours (sRGB) from photometry vs the Sun (B-V 0.65, V-R 0.36): reflectance B/V/R -> linear (R, V, B) -> gamma 2.2
#   Gonggong B-V 1.38, V-R 0.86 (Herschel 'TNOs are Cool'): (1, .63, .32) lin -> (1, .81, .60); provinces lighter (methane frost)
#   Sedna    B-V 1.24, V-R 0.78: (1, .68, .39) lin -> (1, .84, .66); provinces a little redder
#   the three-band estimate reads paler than true colour: Gonggong's B-V equals Mars' (~1.36) -> saturation set to Mars'
#   true-colour look, Sedna a step less red; Gonggong (1, .70, .50), Sedna (1, .74, .57)
_ILL = 'Astra-Planetshine (deviantart.com/astra-planetshine, "Uranus Project Missing Data"), recoloured; colours: published TNO photometry'
BODIES['gonggong'] = dict(name='Gonggong', tno=True, terrain='umbriel', despot=2.2, rot=(63, 141, -38), seed=11, radius_km=615, top=8, mean=58,
                          contrast=1.25, relief_km=10, hl=1.35, rgb=(1.0, 0.70, 0.50), rgb2=(1.0, 0.79, 0.64), credit=_ILL,
                          source='ILLUSTRATIVE: Umbriel terrain (artist map) rotated, recoloured to Gonggong (very red, geometric albedo 0.14)')
BODIES['sedna'] = dict(name='Sedna', tno=True, terrain='ariel', rot=(-112, 37, 74), seed=23, radius_km=497, top=8, mean=92,
                       contrast=0.85, relief_km=8, hl=1.35, rgb=(1.0, 0.74, 0.57), rgb2=(1.0, 0.68, 0.50), credit=_ILL,
                       source='ILLUSTRATIVE: Ariel terrain (artist map) rotated, recoloured to Sedna (deep red, geometric albedo 0.32)')
# s46: Haumea -- Astra-Planetshine's "Haumea Artist's Impression v3" (colour + bump, equirectangular 8192 x 4096, made for a 3-axis
# ellipsoid): the artist's colour map as published (no recolouring), relief from the bump map; the client scales the sphere mesh to the ellipsoid
BODIES['haumea'] = dict(name='Haumea', direct=True, radius_km=798, relief_km=12, top=8,
                        art=('https://drive.usercontent.google.com/download?id=1FlG7PgqQdc5jiMQfe_MYd7ms-HjTQfAe&export=download&confirm=t', 'haumea_v3_1.zip'),
                        art_color='HaumeaColour.png', art_bump='HaumeaBump.png',
                        credit="Astra-Planetshine (DeviantArt), artist's impression, used with permission",
                        source="Astra-Planetshine \"Haumea Artist's Impression v3\" (deviantart.com/astra-planetshine/art/Haumea-Artist-s-Impression-v3-566021580): "
                               "colour + bump maps, ILLUSTRATIVE (Haumea has never been imaged)")
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
DEM_ROLL = 0                                # extra roll (px) of a DEM against the mosaic frame
KNEE = 0.9                                  # soft highlight knee (fraction of 255)
POLE_ROWS = 6
RAMP_ROT = 48                               # rotation fill seam (px at the top tier)
FILL_RELIEF = 0.7   # relief weight in filled / low-data areas (their detail is plausible terrain)
USM_A = 0.35       # luminance unsharp amount (sigma 1 px at the top tier)
SIG_ROT = 48                               # rotation fill: detail band below this scale (px at the top tier)


def fetch(raw, cfg, field):
    """source file named cfg[field]: from cfg['zip'] = (url, member) when given, else cfg['url'] + name, else USGS"""
    name = cfg[field]
    p = os.path.join(raw, name)
    if os.path.exists(p):
        return p
    if cfg.get('zip'):
        import zipfile
        z = src(raw, os.path.basename(cfg['zip'][0]), cfg['zip'][0])
        with zipfile.ZipFile(z) as zf, zf.open(cfg['zip'][1]) as r, open(p + '.part', 'wb') as f:
            while True:
                b = r.read(1 << 22)
                if not b:
                    break
                f.write(b)
        os.replace(p + '.part', p)
        return p
    return src(raw, name, cfg['url'] + name if cfg.get('url') else None)


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
    if t.get(259, 1) != 1 and W * H * spp <= 64 << 20:              # small compressed TIFF (JPL Uranian maps): decode fully
        a = np.array(im.convert('RGB' if spp >= 3 else 'L'))
        im.close()
        a[:, -1] = a[:, -2]                                           # JPL maps: the last column is a dark seam
        return a[None] if a.ndim == 2 else a.transpose(2, 0, 1)
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


def iso_blur(x, s, lat_r, lat0=50.0):
    """Gaussian blur that stays isotropic on the sphere: equirectangular blur at low latitudes, the caps blurred in an
    azimuthal grid (a lat/lon blur averages along the parallels near a pole and leaves a star pattern there)"""
    H, W = x.shape
    n = int(2048 * W / 8192) // 2 * 2 or 512
    n = max(n, 512)
    sg = s * (360.0 / W) / (2 * (90 - lat0) / n)                     # sigma in azimuthal-grid px
    P = polar(x, lambda P, E, r: cv2.GaussianBlur(P, (0, 0), sg), n=n, lat0=lat0, blend=(lat0 + 3, lat0 + 13))
    w = smooth((np.abs(lat_r) - (lat0 + 3)) / 10.0)[:, None].astype(np.float32)
    return blur(x, s) * (1 - w) + P * w


def lut(n, stops):
    xs = np.array([s[0] for s in stops], np.float32)
    ys = np.array([s[1] for s in stops], np.float32)
    return np.dstack([np.interp(n, xs, ys[:, c]).astype(np.float32) for c in range(3)])


def tiff_map16(path):
    """uncompressed, contiguous-strip 16-bit signed GeoTIFF -> read-only memmap (H, W) + its GDAL nodata value"""
    im = Image.open(path)
    t = im.tag_v2
    W, H = im.size
    off = np.asarray(t[273], np.int64)
    assert t.get(259, 1) == 1 and len(off) and np.all(np.diff(off) == W * 2 * (H // len(off))), 'DEM TIFF not contiguous/uncompressed'
    nod = t.get(42113)
    im.close()
    return np.memmap(path, dtype=np.int16, mode='r', offset=int(off[0]), shape=(H, W)), (float(nod) if nod is not None else None)


def area_dem(a, W, H, nodata, strip=64):
    """exact area resample of a huge 2-D int16 array (row strips), nodata excluded; returns (value, coverage) at (H, W)"""
    Hin = a.shape[0]
    s = Hin / H
    val = np.zeros((H, W), np.float32)
    cov = np.zeros((H, W), np.float32)
    for r0 in range(0, H, strip):
        r1 = min(H, r0 + strip)
        a0, b0 = int(np.floor(r0 * s)), min(Hin, int(np.ceil(r1 * s)))
        A = np.zeros((r1 - r0, b0 - a0), np.float32)
        for i in range(r0, r1):
            lo, hi = i * s, (i + 1) * s
            for j in range(int(np.floor(lo)), min(int(np.ceil(hi)), Hin)):
                A[i - r0, j - a0] = (min(hi, j + 1) - max(lo, j)) / s
        x = np.asarray(a[a0:b0]).astype(np.float32)
        m = (x != nodata).astype(np.float32) if nodata is not None else np.ones_like(x)
        cov[r0:r1] = A @ cv2.resize(m, (W, b0 - a0), interpolation=cv2.INTER_AREA)
        val[r0:r1] = A @ cv2.resize(x * m, (W, b0 - a0), interpolation=cv2.INTER_AREA)
    return np.where(cov > 1e-6, val / np.maximum(cov, 1e-6), 0).astype(np.float32), cov


def height_normals(h, R, scale=None):
    """tangent-space normals (float, H x W x 3; x east, y north, z up) of a height field h (km) on a sphere of radius R km;
    scale = optional per-pixel strength of the xy slope (0 = flat)"""
    H, W = h.shape
    dlon, dlat = 2 * np.pi / W, np.pi / H
    lat = np.pi / 2 - (np.arange(H) + 0.5) * dlat
    coslat = np.maximum(np.cos(lat), 0.02)[:, None]
    hx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) / (2 * R * coslat * dlon)
    up, dn = np.vstack([h[:1], h[:-1]]), np.vstack([h[1:], h[-1:]])
    hy = (up - dn) / (2 * R * dlat)
    if scale is not None:
        hx, hy = hx * scale, hy * scale
    n = np.dstack([-hx, -hy, np.ones_like(h)]).astype(np.float32)
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return n


def _renorm(n):
    return n / np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-6)


def n_to_8(n):
    return np.clip(np.rint((n * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8)


def image_gradient(hp, R):
    """(dh/dx_east, dh/dy_north) of a height field on the sphere, per unit of hp (km / km); cos(lat) clamped at 85 deg"""
    H, W = hp.shape
    lat = np.radians(90 - (np.arange(H) + 0.5) * 180 / H)
    cl = np.maximum(np.cos(lat), np.cos(np.radians(85)))[:, None].astype(np.float32)
    hx = (np.roll(hp, -1, 1) - np.roll(hp, 1, 1)) / (2 * 2 * np.pi * R * cl / W)
    up, dn = np.vstack([hp[:1], hp[:-1]]), np.vstack([hp[1:], hp[-1:]])
    hy = (up - dn) / (2 * np.pi * R / H)
    return hx.astype(np.float32), hy.astype(np.float32)


def image_relief(c8, Wt, D, R, slope_deg, f, lat_r):
    """no DEM: band-pass luminance (6 px at 8k) x weight Wt as a height (bright = high), scaled so the RMS slope angle over D > 0.5,
    |lat| < 60 equals slope_deg; returns (normals float H x W x 3, k km per luminance unit, measured RMS slope deg)"""
    L = luma(c8.astype(np.float32))
    hp = (L - iso_blur(L, 6 * f, lat_r)) + 0.6 * (L - iso_blur(L, 2 * f, lat_r))   # s48: two scales, sharper relief
    hp = iso_blur(hp, 0.5 * f, lat_r)
    c_ = 2.5 * float(np.sqrt(np.mean(hp[::5, ::5] ** 2))) + 1e-6
    hp = (c_ * np.tanh(hp / c_)).astype(np.float32)                                  # albedo edges must not become cliffs
    hp = (hp * Wt).astype(np.float32)
    hx, hy = image_gradient(hp, R)
    g = np.sqrt(hx * hx + hy * hy)
    sel = (D > 0.5) & (np.abs(lat_r) < 60)[:, None]
    gs = g[sel][::7].astype(np.float64)
    assert gs.size > 1000, 'too few sharp pixels for the slope calibration'
    rms = lambda k: float(np.degrees(np.sqrt(np.mean(np.arctan(k * gs) ** 2))))
    lo, hi = 1e-6, 1e3
    for _ in range(60):
        k = (lo * hi) ** 0.5
        if rms(k) < slope_deg:
            lo = k
        else:
            hi = k
    meas = rms(k)
    pf = (1 - smooth((np.abs(lat_r) - 80) / 8.0))[:, None].astype(np.float32)
    n = np.dstack([-hx * k * pf, -hy * k * pf, np.ones_like(hx)]).astype(np.float32)
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return n, k, meas


def own_detail(col, Q, f, lat_r):
    """low-resolution areas get band-pass luminance detail of the body's OWN sharp terrain through 5 sphere rotations
    (fixed seed 47); per pixel the rotated copy whose rotated Q is highest; adds (1 - Q) * Qrot * D to the luminance"""
    L = luma(col)
    D = (L - iso_blur(L, 4 * f, lat_r)).astype(np.float32)
    sel_ = Q > 0.8
    c_ = 2.0 * float(np.sqrt(np.mean(D[sel_] ** 2) if sel_.sum() > 1000 else np.mean(D ** 2))) + 1e-6
    D = (c_ * np.tanh(D / c_)).astype(np.float32)                                     # s48: no high-contrast feature clones
    rng = np.random.default_rng(47)
    bq = np.zeros_like(Q)
    bd = np.zeros_like(Q)
    for i in range(5):
        ax = rng.standard_normal(3); ax /= np.linalg.norm(ax)
        t = np.radians(rng.uniform(40, 140))
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        Rm = np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)
        qr = sphere_rotate(Q, Rm)
        dr = sphere_rotate(D, Rm)
        better = qr > bq
        bq = np.where(better, qr, bq)
        bd = np.where(better, dr, bd)
        print('  own-detail rotation %d: axis %s, %d deg' % (i, np.round(ax, 2), round(np.degrees(t))), flush=True)
    add = (1 - Q) * np.clip(bq, 0, 1) * bd
    col *= ((L + add) / np.maximum(L, 1.0))[..., None]
    print('  own detail: mean |add| %.2f, weight (1-Q)*Qrot mean %.3f' % (float(np.abs(add).mean()), float(((1 - Q) * np.clip(bq, 0, 1)).mean())), flush=True)
    return col


DSK_URL = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/co-s_j_e_v-spice-6-v1.0/cosp_1000/data/dsk/%s'
import tempfile
SCRATCH = os.environ.get('MOONS_SCRATCH') or os.path.join(tempfile.gettempdir(), 'moons_build')   # check images + hybrid temp build
os.makedirs(SCRATCH, exist_ok=True)


def dsk_aligned_height(path, c8, W, H, f, lat_r, key):
    """NAIF DSK height (km) on the build grid, aligned (longitude mirror + roll) to the colour mosaic by the correlation of
    the band-pass gradient magnitudes; searches the roll within +-64 px only (the source centre is set per body), warns when NCC < 0.03; writes SCRATCH/enc_align.png for a visual check"""
    from dsk_relief import dsk_height
    h0, _ = dsk_height(path, 4096, 2048)
    h = h0 if W == 4096 else cv2.resize(h0, (W, H), interpolation=cv2.INTER_CUBIC)
    del h0
    L = luma(c8.astype(np.float32))
    gm = lambda x: np.hypot(*np.gradient(x - iso_blur(x, 8.0, lat_r)))
    GL, Gh = gm(L), gm(h)
    sel = np.abs(lat_r) < 60
    def rn(G):
        G = G[sel]; G = G - G.mean(1, keepdims=True)
        return G / np.maximum(G.std(1, keepdims=True), 1e-9)
    A = rn(GL)
    B = rn(Gh)
    cc = np.fft.irfft((np.conj(np.fft.rfft(B, axis=1)) * np.fft.rfft(A, axis=1)).sum(0), n=W)   # shift s: B rolled by s matches A
    cand = np.arange(-64, 65)
    s = int(cand[np.argmax(cc[cand % W])])
    nc = float(np.mean(A * np.roll(B, s, 1)))
    nc0 = float(np.mean(A * B))
    print('  DSK orientation: roll %d px (window -64..64), NCC %.4f (no roll %.4f)' % (s, nc, nc0), flush=True)
    if nc < 0.03:
        print('  WARNING: DSK orientation NCC %.4f < 0.03' % nc, flush=True)
    h = np.ascontiguousarray(np.roll(h, s, 1)).astype(np.float32)
    # visual check: hillshade (sun NW, 45 deg) blended 50/50 with the mosaic
    w2, h2 = 2048, 1024
    hs = cv2.resize(h, (w2, h2), interpolation=cv2.INTER_AREA)
    n = height_normals(hs, 252.1)
    l = np.array([-0.5, 0.5, 0.7071], np.float32)
    sh = np.clip(n @ l, 0, 1)
    g = cv2.resize(L, (w2, h2), interpolation=cv2.INTER_AREA)
    g = np.clip(g / max(float(np.percentile(g, 99.5)), 1.0), 0, 1)
    out = (0.5 * g + 0.5 * sh / max(float(np.percentile(sh, 99.5)), 1e-3))
    cv2.imwrite(os.path.join(SCRATCH, 'enc_align.png'), np.clip(out * 255, 0, 255).astype(np.uint8))
    return h


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
        v, cov = area(tiff_map(fetch(rawk, cfg, 'color')), W, H)
        for c in range(3):
            v[c] = wrapfix(v[c], cov > MASK_COV)
        rgb = np.roll(np.ascontiguousarray(v.transpose(1, 2, 0)), roll0, 1)
        L = luma(rgb)
    else:
        v, cov = area(tiff_map(fetch(rawk, cfg, 'grey')), W, H)
        L = np.roll(wrapfix(v[0], cov > MASK_COV), roll0, 1)
    del v
    cov = np.roll(cov, roll0, 1)
    M = cv2.erode((cov > MASK_COV).astype(np.uint8), np.ones((MASK_ERODE, MASK_ERODE), np.uint8)).astype(np.float32)
    del cov
    print('  gap fraction %.3f' % (1 - M.mean()), flush=True)
    if cfg.get('flip'):
        # retrograde orbit (Triton, the Uranian moons): the client's mesh +Y is the orbit normal = the IAU SOUTH pole. The
        # map is turned 180 deg about the sub-planet axis (lat -> -lat, lon -> -lon), so lon 0 still faces the planet
        L, M = np.ascontiguousarray(L[::-1, ::-1]), np.ascontiguousarray(M[::-1, ::-1])
        if rgb is not None:
            rgb = np.ascontiguousarray(rgb[::-1, ::-1])
    rot_txt = ''
    colour_extra = ''
    Mdat = None
    if cfg.get('pole_cut'):                                           # radially smeared polar source (airbrush maps): refilled
        latc = 90 - (np.arange(H) + 0.5) * 180 / H
        M = (M * (np.abs(latc) <= cfg['pole_cut'])[:, None]).astype(np.float32)
    if cfg.get('rim'):                                                # dark terminator fringe along the imaged area: dropped
        M = (M * (dist(M > 0) > cfg['rim'] * f)).astype(np.float32)
    if cfg.get('mask_smooth'):
        M = smooth_mask(L, M, cfg['mask_smooth'], f)
    if cfg.get('rotfill'):
        Mdat = M
        L, M, rgb, rot_txt = rot_fill(L, M, rgb, cfg['rotfill'], f, cfg.get('rot_sig', SIG_ROT))
    if cfg.get('old', True):
        O = np.array(Image.open(src(rawk, key + '_old_color_4k.webp', OLD % (key, key))).convert('RGB'), np.float32)
    else:
        O = None

    # --- alignment to the previous site map ------------------------------------------------------------------------
    if O is None:
        al = {'applied': False, 'ncc': None, 'note': 'no previous site map: USGS orientation (lon 0 at the centre)'}
        shift = 0
    else:
        al = align(L, M, O)
        al['applied'] = al['ncc'] >= NCC_MIN
        shift = al['shift_px'] if al['applied'] else 0
    if O is not None and not al['applied']:
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
        v, cc = area(tiff_map(fetch(rawk, cfg, 'color')), W, H)
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
    if Mdat is not None:                                              # baked lighting exists only in the imaged area
        w6 = w6 * Mdat
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
    if Mdat is not None:                                              # rotation-filled area: already clean detail; the lat/lon
        Q = np.maximum(Q, blur(1 - Mdat, 4 * f + 0.5)).astype(np.float32)   # de-ringing medians would smear it radially at the poles
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
        if not cfg.get('own_detail', True):                           # s48: own-terrain detail replaces the noise grain
            L = L * (1 + np.clip(GRAIN_Q * det * gw * N, -0.6, 0.6))
        del gw, N

        # --- colour --------------------------------------------------------------------------------------------------------
        L = np.maximum(L, 0.5)
        if cfg['mode'] == 'lut':
            Ls = iso_blur(L, SIG_LUT * f, lat_r)
            valid = (M > 0) & (wfill < 0.5)
            lo, hi = np.percentile(Ls[valid], [1, 99.5])
            n = np.clip((Ls - lo) / max(hi - lo, 1e-3), 0, 1)
            col = lut(n, cfg['lut']) * np.clip(L / np.maximum(Ls, 0.5), 0, 3)[..., None]
            del Ls, n
        else:
            col = L[..., None] * Cr
            del Cr
        del L
        if cfg.get('tint'):                                              # measured overall colour, luminance kept
            tg = np.array(cfg['tint'], np.float32); col *= tg / float(luma(tg[None])[0])
        if cfg.get('cap_tint'):                                          # known polar colour (Charon: Mordor Macula), luminance kept
            la0, la1, tc = cfg['cap_tint']
            tc = np.array(tc, np.float32); tc /= float(luma(tc[None])[0])
            wc = smooth((lat_r - la0) / (la1 - la0))[:, None, None].astype(np.float32)
            col *= 1 + wc * (tc - 1)
        if cfg.get('own_detail', True):                                      # low-Q areas: detail of the body's own sharp terrain
            col = own_detail(col, Q, f, lat_r)
            colour_extra = ('; low-resolution areas carry band-pass detail of %s\'s own sharp terrain through 5 sphere rotations, '
                            'weight (1-Q)' % cfg['name'])

        # --- exposure: global luminance mean of the previous map; soft highlight knee; no blue cast -------------------------
        # a tone curve (power per channel), not a gain: a 2x brighter previous map must not blow the highlights out
        Y = luma(col)
        YOm = float(luma(O).mean()) if O is not None else float(cfg['mean'])   # no previous map: target from the measured albedo
        cs = np.clip(col[::4, ::4], 0, 255) / 255
        if cfg.get('linear'):                                             # gamma-neutral: plain gain, the knee takes the top
            g = 1.0
        lo, hi = 0.2, 5.0
        for _ in range(0 if cfg.get('linear') else 40):
            g = (lo * hi) ** 0.5
            if luma(cs ** g).mean() * 255 > YOm:
                lo = g
            else:
                hi = g
        if cfg.get('linear') or g > 1:                                  # darkening: a plain gain (a power > 1 would
            g = 1.0                                                       # multiply every ripple / ringing contrast)
            col *= float(YOm / max(Y.mean(), 1e-3))
        else:
            col = 255 * (np.clip(col, 0, 255) / 255) ** g
        print('  luma mean new %.1f -> target %.1f (tone power %.3f)' % (Y.mean(), YOm, g), flush=True)
        del Y, cs, O
        if cfg.get('deshade'):                                           # baked mosaic shadows would double the real relief shading
            s_, a_ = cfg['deshade']
            Yu = luma(col); Yl = iso_blur(Yu, s_ * f, lat_r)
            col *= np.clip((Yl + a_ * (Yu - Yl)) / np.maximum(Yu, 1.0), 0.3, 3.0)[..., None]
            del Yu, Yl
        else:
            Yu = luma(col); Yb = iso_blur(Yu, 1.0 * f, lat_r)
            col *= np.clip((Yu + USM_A * (Yu - Yb)) / np.maximum(Yu, 1.0), 0.6, 1.6)[..., None]   # s48: sharper colour
            del Yu, Yb
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
                      'grain (%g x local detail); luminance mean %s (tone power %.3f); no blue cast' % (RAMP, GRAIN_GAP,
                      'of the previous map' if cfg.get('old', True) else '%g (from the measured geometric albedo)' % cfg['mean'], g) + rot_txt + colour_extra)

    # --- relief: Charon = New Horizons DEM; image-derived for the others with slope_deg; the rest stays flat -----------------
    nrm8 = None                                                       # top-tier normal map (uint8) or per-tier callable
    nrm_fn = None
    nrm_txt = None
    if cfg.get('dsk'):
        hd = dsk_aligned_height(src(rawk, cfg['dsk'], DSK_URL % cfg['dsk']), c8, W, H, f, lat_r, key)
        pwf = lambda hh: (1 - smooth((np.abs(90 - (np.arange(hh) + 0.5) * 180 / hh) - 86) / 3.0))[:, None].astype(np.float32)
        nrm_fn = lambda w, hh: n_to_8(height_normals(hd if w == W else cv2.resize(hd, (w, hh), interpolation=cv2.INTER_AREA), cfg['radius_km'],
                                                     np.broadcast_to(pwf(hh), (hh, w))))
        nrm_txt = ('tangent space x=east y=north z=up, TRUE SCALE from the NAIF/JPL Cassini SPC global shape model (Park et al. 2023, '
                   'ICQ 1024, ~0.4 km), ellipsoid removed')
    elif cfg.get('dem') and not cfg.get('nonormal'):
        dn_, dnod = tiff_map16(src(rawk, cfg['dem'], S3 + cfg['dem']))
        hd, dcov = area_dem(dn_, W, H, dnod if dnod is not None else -32768)
        del dn_
        hd = hd / 1000.0                                              # metres -> km
        hd, dcov = np.roll(hd, roll0 + DEM_ROLL, 1), np.roll(dcov, roll0 + DEM_ROLL, 1)   # same roll as the mosaic
        D = cv2.erode((dcov > MASK_COV).astype(np.uint8), np.ones((MASK_ERODE, MASK_ERODE), np.uint8)).astype(np.float32)
        del dcov
        print('  DEM coverage %.3f of the map' % float(D.mean()), flush=True)
        wd = smooth(dist(D > 0) / (16 * f))                           # 0 at the DEM mask edge, 1 beyond 16 px (no cliff)
        hd = (np.where(D > 0, hd, 0) * wd).astype(np.float32)
        del D, wd
        qd = (Q * (1 - smooth((np.abs(lat_r) - 82) / 6))[:, None]).astype(np.float32)
        nrm_fn = lambda w, hh: n_to_8(height_normals(hd if w == W else cv2.resize(hd, (w, hh), interpolation=cv2.INTER_AREA), cfg['radius_km'],
                                                     qd if w == W else cv2.resize(qd, (w, hh), interpolation=cv2.INTER_AREA)))
        nrm_txt = 'tangent space x=east y=north z=up, true scale from the New Horizons DEM 300 m (Jul 2017), flat outside DEM coverage'
        print('  DEM relief: min %.2f max %.2f km' % (float(hd.min()), float(hd.max())), flush=True)
    elif cfg.get('slope_deg') and not cfg.get('nonormal'):
        Dm = (Mdat if Mdat is not None else M).astype(np.float32)        # real imaged pixels only (fills / rotation fill = 0)
        ed_ = Dm * smooth(dist(Dm > 0.5) / (16 * f))
        Wr = (np.maximum(Q, 0.5) * (ed_ + FILL_RELIEF * (1 - ed_))).astype(np.float32)   # s48: filled areas get relief too
        del ed_
        nn, kk_, meas = image_relief(c8, Wr, Dm, cfg['radius_km'], cfg['slope_deg'], f, lat_r)
        del Wr, Dm
        print('  image-derived relief: k %.5f km per luminance unit, RMS slope %.3f deg (target %g)' % (kk_, meas, cfg['slope_deg']), flush=True)
        nrm_fn = lambda w, hh: n_to_8(nn if w == W else _renorm(cv2.resize(nn, (w, hh), interpolation=cv2.INTER_AREA)))
        nrm_txt = ('tangent space x=east y=north z=up, IMAGE-DERIVED (no global DEM): band-pass luminance (6 px at 8k) as height, bright = high, '
                   'RMS slope %g deg where the mosaic is sharp (x Q), x %g in filled areas' % (cfg['slope_deg'], FILL_RELIEF))
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
        if cfg.get('nonormal'):                                     # shape-model body: relief is in the mesh
            continue
        nm = '%s_normal_%dk.webp' % (key, kk)
        webp(nrm_fn(w, hh) if nrm_fn else np.full((hh, w, 3), (128, 128, 255), np.uint8), os.path.join(out, nm), 92); files.append(nm)
    meta = {'v': 1, 'built': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'radius_km': cfg['radius_km'],
            'projection': 'equirectangular, north up, east to the right; ' + ('ROTATED 180 deg about the sub-planet axis (south up: the client mesh +Y is the IAU south pole of this retrograde moon); ' if cfg.get('flip') else '') + (
                'rolled onto the previous site map, lon %g at the centre' % al['centre_lon'] if al['rolled'] else 'lon 0 at the centre'),
            'align': al, 'max_tier': '%dk' % top,
            'normal': (nrm_txt if nrm_txt else 'none: shape-model body, the relief is in the client mesh' if cfg.get('nonormal') else
                       'flat (128,128,255): relief is not derived from image brightness' + ('' if cfg.get('old', True) is False else '; no global DEM of %s is available' % cfg['name'])),
            'colour': colour_txt, 'colour_base': 'previous site map' if oldbase else 'mission mosaic (%s)' % cfg['mode'],
            'files': sorted(files),
            'source': cfg['source'], 'credit': cfg['credit']}
    json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'), indent=1)
    print('  %s done in %d s' % (key, time.time() - t0), flush=True)


ART_SIG = 3.0                               # art hybrid: low-pass (deg) of the Voyager / art luminance ratio
ART_W, ART_CLAMP = 0.8, (0.7, 1.4)          # art hybrid: weight of the Voyager large-scale structure, ratio clamp
ART_SAT = 0.12                              # art hybrid: saturation kept from the artist's colour (Uranian moons are near-grey)
ART_USM = (1.2, 0.55)                       # art hybrid: unsharp mask (sigma px at the top tier, amount) on the luminance


def art_src(raw, cfg, member):
    """member of the artist's pack (cfg['art'] = (zip url, local zip name)), extracted once into raw/"""
    p = os.path.join(raw, os.path.basename(member))
    if not os.path.exists(p):
        import zipfile
        z = src(os.path.dirname(raw), cfg['art'][1], cfg['art'][0])   # one pack for all bodies, in the raw root
        with zipfile.ZipFile(z) as zf, zf.open(member) as r, open(p + '.part', 'wb') as f:
            while True:
                b = r.read(1 << 22)
                if not b:
                    break
                f.write(b)
        os.replace(p + '.part', p)
    return p


def build_art(key, cfg, raw, out_root, Wt=None):
    """artist's full-globe map (sharp detail, plausible unseen hemisphere) x the real Voyager large-scale albedo where
    imaged; relief = the artist's bump map, true scale (cfg['relief_km'] = height span of the bump range)"""
    t0 = time.time()
    rawk = os.path.join(raw, key); os.makedirs(rawk, exist_ok=True)
    out = os.path.join(out_root, key); os.makedirs(out, exist_ok=True)
    top = cfg['top']
    W, H = (Wt, Wt // 2) if Wt else SIZES[top]
    f = W / SIZES[top][0]
    print('== art', key, W, 'x', H, flush=True)
    # orientation: the artist's maps are the Voyager/JPL map turned 180 deg (NCC peak at 0 shift for all five); the
    # JPL map goes through the s34 chain (centre 180 roll, flip) -> the artist's map needs only the half-width roll
    A = np.asarray(Image.open(art_src(rawk, cfg, cfg['art_color'])).convert('RGB'))
    up = W / A.shape[1]                                               # > 1: the colour source is coarser than the tier
    A = cv2.resize(A, (W, H), interpolation=cv2.INTER_AREA if A.shape[1] >= W else cv2.INTER_LANCZOS4).astype(np.float32)
    A = np.roll(A, W // 2, 1)
    La = np.maximum(luma(A), 1.0)
    lat_r = 90 - (np.arange(H) + 0.5) * 180 / H
    # --- real large-scale structure: low-passed Voyager / art luminance ratio over the imaged area ---------------------
    v, cov = area(tiff_map(fetch(rawk, cfg, 'grey')), W, H)
    G = v[0]; del v
    if cfg.get('last_col_bad'):
        cov[:, -max(1, int(W / 1440 + 0.5)) * 2:] = 0
    G, cov = np.roll(G, W // 2, 1), np.roll(cov, W // 2, 1)
    if cfg.get('flip'):
        G, cov = np.ascontiguousarray(G[::-1, ::-1]), np.ascontiguousarray(cov[::-1, ::-1])
    M = cv2.erode(((cov > MASK_COV) & (G > 8)).astype(np.uint8), np.ones((MASK_ERODE, MASK_ERODE), np.uint8)).astype(np.float32)
    M = (M * (dist(M > 0) > cfg.get('rim', 40) * f)).astype(np.float32)
    M *= (np.abs(lat_r) <= cfg.get('pole_cut', 72))[:, None]
    s = ART_SIG / 360 * W
    R = nblur(G, M, s) / np.maximum(nblur(La, M, s), 1.0)
    R /= float(np.median(R[M > 0]))
    R = np.clip(R, *ART_CLAMP)
    wR = ART_W * blur(M, s) * smooth((60 - np.abs(lat_r)) / 8)[:, None]   # fades out before the airbrush polar smear
    R = 1 + wR * (R - 1)
    print('  voyager ratio: imaged %.3f, ratio p5/p95 %.3f / %.3f' % (float(M.mean()), *np.percentile(R[M > 0], [5, 95])), flush=True)
    del G, cov
    # --- colour: artist's chroma desaturated, measured tint, luminance mean from the albedo ---------------------------
    L = La * R
    # sharpen: unsharp mask on the luminance (the painted bump is NOT used here: its 8-bit terraces became contour lines)
    L = L + ART_USM[1] * (L - blur(L, ART_USM[0] * max(up, 1.0)))
    if cfg.get('hl'):                                                 # bright spots (painted / copied terrain): soft cap
        L = hl_cap(L, cfg['hl'], W)
    L = np.maximum(L, 0.5)
    Cr = 1 + ART_SAT * (A / La[..., None] - 1)
    del A, R
    col = L[..., None] * Cr
    tg = np.array(lut(np.array([[0.5]], np.float32), cfg['lut'])[0, 0], np.float32)
    col *= tg / float(luma(tg[None])[0])
    col *= float(cfg['mean']) / max(float(luma(col).mean()), 1e-3)
    T = KNEE * 255
    col = np.where(col > T, T + (255 - T) * (1 - np.exp(-(col - T) / (255 - T))), col)
    col[..., 2] = np.minimum(col[..., 2], np.maximum(col[..., 0], col[..., 1]))
    c8 = np.clip(np.rint(col), 0, 255).astype(np.uint8)
    del col, L, Cr, La
    # --- relief: artist's bump, true scale --------------------------------------------------------------------------------
    B = np.asarray(Image.open(art_src(rawk, cfg, cfg['art_bump'])).convert('L'))
    B = cv2.resize(B, (W, H), interpolation=cv2.INTER_AREA if B.shape[1] >= W else cv2.INTER_CUBIC).astype(np.float32)
    B = blur(np.roll(B, W // 2, 1), 1.6 * W / 8192)                     # 8-bit terraces (contour steps) of the painted bump
    lo, hi = np.percentile(B, [0.5, 99.5])
    hgt = (B - lo) / max(hi - lo, 1.0) * cfg['relief_km']
    del B
    Rk = cfg['radius_km']
    dl, dp = 2 * np.pi / W, np.pi / H
    cl = np.maximum(np.cos(np.radians(lat_r)), 0.02)[:, None]
    hx = (np.roll(hgt, -1, 1) - np.roll(hgt, 1, 1)) / (2 * Rk * cl * dl)
    up, dn = np.vstack([hgt[:1], hgt[:-1]]), np.vstack([hgt[1:], hgt[-1:]])
    hy = (up - dn) / (2 * Rk * dp)
    del hgt, up, dn
    pw = smooth((np.abs(lat_r) - 84) / 5)[:, None]                    # last deg: equirect gradients are unreliable
    hx, hy = hx * (1 - pw), hy * (1 - pw)
    n = np.dstack([-hx, -hy, np.ones_like(hx)]).astype(np.float32)
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    n8 = np.clip(np.rint((n * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8)
    print('  relief: slope rms %.3f' % float(np.sqrt((hx * hx + hy * hy).mean())), flush=True)
    del n, hx, hy
    files = []
    for kk, (w, hh) in SIZES.items():
        if (Wt is None and kk > top) or (Wt and w > W):
            continue
        cc = c8 if w == W else cv2.resize(c8, (w, hh), interpolation=cv2.INTER_AREA)
        nm = '%s_color_%dk.webp' % (key, kk); webp(cc, os.path.join(out, nm), 90); files.append(nm)
        nn = n8 if w == W else cv2.resize(n8, (w, hh), interpolation=cv2.INTER_AREA)
        nm = '%s_normal_%dk.webp' % (key, kk); webp(nn, os.path.join(out, nm), 92); files.append(nm)
    meta = {'v': 1, 'built': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'radius_km': Rk,
            'projection': 'equirectangular, east to the right; ROTATED 180 deg about the sub-planet axis (south up: the client '
                          'mesh +Y is the IAU south pole of this retrograde moon); lon 0 at the centre', 'max_tier': '%dk' % top,
            'normal': 'tangent space (x east, y north of the image), true scale from the artist\'s bump map, height span %g km' % cfg['relief_km'],
            'colour': 'artist\'s full-globe map (saturation x%g, measured tint, luminance mean %g from the albedo) x the low-passed '
                      '(%g deg) Voyager / artist luminance ratio over the imaged southern hemisphere (weight %g, clamp %s); unsharp mask %s'
                      % (ART_SAT, cfg['mean'], ART_SIG, ART_W, ART_CLAMP, ART_USM),
            'files': sorted(files), 'source': cfg['source'], 'credit': cfg['credit']}
    json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'), indent=1)
    print('  %s done in %d s' % (key, time.time() - t0), flush=True)


def build_direct(key, cfg, raw, out_root, Wt=None):
    """artist's colour map as published (RGB, no recolouring) + tangent-space normal from the artist's bump map (true scale,
    cfg['relief_km'] = height span of the bump range); no orientation change (lon 0 at the centre, north up)"""
    t0 = time.time()
    rawk = os.path.join(raw, key); os.makedirs(rawk, exist_ok=True)
    out = os.path.join(out_root, key); os.makedirs(out, exist_ok=True)
    top = cfg['top']
    W, H = (Wt, Wt // 2) if Wt else SIZES[top]
    print('== direct', key, W, 'x', H, flush=True)
    A = Image.open(art_src(rawk, cfg, cfg['art_color'])).convert('RGB')
    c8 = np.asarray(A.resize((W, H), Image.LANCZOS if A.size[0] < W else Image.BOX))
    del A
    B = Image.open(art_src(rawk, cfg, cfg['art_bump'])).convert('L')
    B = np.asarray(B.resize((W, H), Image.BICUBIC if B.size[0] < W else Image.BOX)).astype(np.float32)
    B = blur(B, 1.6 * W / 8192)                                       # 8-bit terraces of the painted bump
    lo, hi = np.percentile(B, [0.5, 99.5])
    hgt = (B - lo) / max(hi - lo, 1.0) * cfg['relief_km']
    del B
    Rk = cfg['radius_km']
    lat_r = 90 - (np.arange(H) + 0.5) * 180 / H
    dl, dp = 2 * np.pi / W, np.pi / H
    cl = np.maximum(np.cos(np.radians(lat_r)), 0.02)[:, None]
    hx = (np.roll(hgt, -1, 1) - np.roll(hgt, 1, 1)) / (2 * Rk * cl * dl)
    up, dn = np.vstack([hgt[:1], hgt[:-1]]), np.vstack([hgt[1:], hgt[-1:]])
    hy = (up - dn) / (2 * Rk * dp)
    del hgt, up, dn
    pw = smooth((np.abs(lat_r) - 84) / 5)[:, None]
    hx, hy = hx * (1 - pw), hy * (1 - pw)
    n = np.dstack([-hx, -hy, np.ones_like(hx)]).astype(np.float32)
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    n8 = np.clip(np.rint((n * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8)
    print('  relief: slope rms %.3f' % float(np.sqrt((hx * hx + hy * hy).mean())), flush=True)
    del n, hx, hy
    files = []
    for kk, (w, hh) in SIZES.items():
        if (Wt is None and kk > top) or (Wt and w > W):
            continue
        cc = c8 if w == W else cv2.resize(c8, (w, hh), interpolation=cv2.INTER_AREA)
        nm = '%s_color_%dk.webp' % (key, kk); webp(cc, os.path.join(out, nm), 90); files.append(nm)
        nn = n8 if w == W else cv2.resize(n8, (w, hh), interpolation=cv2.INTER_AREA)
        nm = '%s_normal_%dk.webp' % (key, kk); webp(nn, os.path.join(out, nm), 92); files.append(nm)
    meta = {'v': 1, 'built': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'radius_km': Rk,
            'projection': 'equirectangular, north up, east to the right, lon 0 at the centre; made by the artist for a 3-axis ellipsoid '
                          '(the client maps it onto the scaled sphere mesh)', 'max_tier': '%dk' % top,
            'ILLUSTRATIVE': "Haumea has never been imaged: artist's impression",
            'normal': 'tangent space (x east, y north), true scale from the artist bump map on the mean radius, height span %g km' % cfg['relief_km'],
            'colour': 'artist colour map as published (no recolouring), area-resampled',
            'files': sorted(files), 'source': cfg['source'], 'credit': cfg['credit']}
    json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'), indent=1)
    print('  %s done in %d s' % (key, time.time() - t0), flush=True)


def hl_cap(L, c, W):
    """soft cap of bright spots relative to the surrounding terrain (lowpass ~1/40 of the circumference): ratios above
    c are compressed (tanh), the dark side untouched -- a pure-white painted crater / ring no longer stands out"""
    base = lblur(L, W / 40.0)
    r = L / np.maximum(base, 1.0)
    k = 0.4 * (c - 1)
    r = np.where(r > c - k, (c - k) + k * np.tanh((r - (c - k)) / k), r)
    return (r * base).astype(np.float32)


def despot(L, thr, W):
    """remove large bright patches (Wunda's floor): where L exceeds thr x the broad lowpass, fill from the surroundings
    (1k TELEA inpaint, feathered) with a little grain, so no painted white disc can give the source away"""
    H = L.shape[0]
    r = L / np.maximum(lblur(L, W / 8.0), 1.0)
    m = ((r > thr) | (L / np.maximum(lblur(L, W / 30.0), 1.0) > 1.8)).astype(np.uint8)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(W / 60) * 2 + 1,) * 2))
    w1, h1 = 1024, 512
    sm = cv2.resize(L, (w1, h1), interpolation=cv2.INTER_AREA)
    mm = (cv2.resize(m.astype(np.float32), (w1, h1), interpolation=cv2.INTER_AREA) > 0.01).astype(np.uint8)
    p = 32
    inp = cv2.inpaint(wrap_pad(np.where(mm > 0, 0, sm).astype(np.float32), p), wrap_pad(mm, p), 6, cv2.INPAINT_TELEA)[:, p:-p]
    inp = cv2.GaussianBlur(inp, (0, 0), 3)
    F = cv2.resize(inp, (W, H), interpolation=cv2.INTER_CUBIC)
    F = F * (1 + 0.06 * sphere_noise(H, W, [W / 300.0, W / 700.0], 5))
    w = blur(m.astype(np.float32), W / 200.0)
    print('  despot: %.4f of the map filled' % float(m.mean()), flush=True)
    return (L * (1 - w) + F * w).astype(np.float32)


def sphere_rotate(img, Rm, chunk=256):
    """equirectangular field (H, W[, C]) seen through a rotation of the sphere: out(d) = img(Rm^T d)"""
    H, W = img.shape[:2]
    p = 4
    src_ = np.concatenate([img[:, -p:], img, img[:, :p]], 1).astype(np.float32)
    out = np.empty(img.shape, np.float32)
    lon = (np.arange(W) + 0.5) / W * 2 * np.pi - np.pi
    for r0 in range(0, H, chunk):
        la = np.radians(90 - (np.arange(r0, min(H, r0 + chunk)) + 0.5) * 180 / H)[:, None]
        d = np.stack([np.cos(la) * np.sin(lon), np.broadcast_to(np.sin(la), (la.shape[0], W)), np.cos(la) * np.cos(lon)], -1)
        s = d @ Rm                                                    # = (Rm^T d) per pixel
        sl = np.degrees(np.arcsin(np.clip(s[..., 1], -1, 1)))
        so = np.arctan2(s[..., 0], s[..., 2])
        px = ((so + np.pi) / (2 * np.pi) * W - 0.5 + p).astype(np.float32)
        py = ((90 - sl) / 180 * H - 0.5).astype(np.float32)
        out[r0:r0 + la.shape[0]] = cv2.remap(src_, px, py, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return out


def build_tno(key, cfg, raw, out_root, Wt=None):
    """no spacecraft image exists (Gonggong, Sedna): an ILLUSTRATIVE globe -- a Uranian moon's full-globe terrain from the
    artist's pack (colour detail + bump) seen through a random rotation of the sphere, recoloured to the body's measured
    colour / albedo with large soft colour provinces"""
    t0 = time.time()
    sc = BODIES[cfg['terrain']]
    rawk = os.path.join(raw, cfg['terrain']); os.makedirs(rawk, exist_ok=True)
    out = os.path.join(out_root, key); os.makedirs(out, exist_ok=True)
    top = cfg['top']
    W, H = (Wt, Wt // 2) if Wt else SIZES[top]
    print('== tno', key, 'from', cfg['terrain'], W, 'x', H, flush=True)
    a = np.radians(cfg['rot'])
    Rx = np.array([[1, 0, 0], [0, np.cos(a[0]), -np.sin(a[0])], [0, np.sin(a[0]), np.cos(a[0])]])
    Ry = np.array([[np.cos(a[1]), 0, np.sin(a[1])], [0, 1, 0], [-np.sin(a[1]), 0, np.cos(a[1])]])
    Rz = np.array([[np.cos(a[2]), -np.sin(a[2]), 0], [np.sin(a[2]), np.cos(a[2]), 0], [0, 0, 1]])
    Rm = Rz @ Ry @ Rx
    lat_r = 90 - (np.arange(H) + 0.5) * 180 / H
    # --- luminance detail ------------------------------------------------------------------------------------------------
    A = np.asarray(Image.open(art_src(rawk, sc, sc['art_color'])).convert('RGB'))
    up = W / A.shape[1]
    L = luma(cv2.resize(A, (W, H), interpolation=cv2.INTER_AREA if up <= 1 else cv2.INTER_LANCZOS4).astype(np.float32))
    del A
    if cfg.get('despot'):
        L = despot(L, cfg['despot'], W)
    L = sphere_rotate(L, Rm)
    L = L + ART_USM[1] * (L - blur(L, ART_USM[0] * max(up, 1.0)))
    if cfg.get('hl'):                                                 # bright spots (painted / copied terrain): soft cap
        L = hl_cap(L, cfg['hl'], W)
    L = np.maximum(L, 1.0) / float(np.mean(L))
    L = L ** cfg.get('contrast', 1.0)
    # --- colour: measured colour, soft provinces (one sphere-noise octave, ~1/7 of the circumference) --------------------------------
    N = sphere_noise(H, W, [W / 7.0], cfg.get('seed', 7))
    wv = (0.7 * smooth((N * 0.5 + 0.5 - 0.15) / 0.7))[..., None].astype(np.float32)
    c0, c1 = np.array(cfg['rgb'], np.float32), np.array(cfg['rgb2'], np.float32)
    col = L[..., None] * (c0 * (1 - wv) + c1 * wv) * (1 + 0.08 * N[..., None])
    del N, wv, L
    col *= float(cfg['mean']) / max(float(luma(col).mean()), 1e-3)
    T = KNEE * 255
    col = np.where(col > T, T + (255 - T) * (1 - np.exp(-(col - T) / (255 - T))), col)
    c8 = np.clip(np.rint(col), 0, 255).astype(np.uint8)
    del col
    # --- relief: the artist's bump through the same rotation, true scale --------------------------------------------------
    B = np.asarray(Image.open(art_src(rawk, sc, sc['art_bump'])).convert('L'))
    B = cv2.resize(B, (W, H), interpolation=cv2.INTER_AREA if B.shape[1] >= W else cv2.INTER_CUBIC).astype(np.float32)
    B = sphere_rotate(blur(B, 1.6 * W / 8192), Rm)
    lo, hi = np.percentile(B, [0.5, 99.5])
    hgt = (B - lo) / max(hi - lo, 1.0) * cfg['relief_km']
    del B
    Rk = cfg['radius_km']
    dl, dp = 2 * np.pi / W, np.pi / H
    cl = np.maximum(np.cos(np.radians(lat_r)), 0.02)[:, None]
    hx = (np.roll(hgt, -1, 1) - np.roll(hgt, 1, 1)) / (2 * Rk * cl * dl)
    up_, dn = np.vstack([hgt[:1], hgt[:-1]]), np.vstack([hgt[1:], hgt[-1:]])
    hy = (up_ - dn) / (2 * Rk * dp)
    del hgt, up_, dn
    pw = smooth((np.abs(lat_r) - 84) / 5)[:, None]
    hx, hy = hx * (1 - pw), hy * (1 - pw)
    n = np.dstack([-hx, -hy, np.ones_like(hx)]).astype(np.float32)
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    n8 = np.clip(np.rint((n * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8)
    del n, hx, hy
    files = []
    for kk, (w, hh) in SIZES.items():
        if (Wt is None and kk > top) or (Wt and w > W):
            continue
        cc = c8 if w == W else cv2.resize(c8, (w, hh), interpolation=cv2.INTER_AREA)
        nm = '%s_color_%dk.webp' % (key, kk); webp(cc, os.path.join(out, nm), 90); files.append(nm)
        nn = n8 if w == W else cv2.resize(n8, (w, hh), interpolation=cv2.INTER_AREA)
        nm = '%s_normal_%dk.webp' % (key, kk); webp(nn, os.path.join(out, nm), 92); files.append(nm)
    meta = {'v': 1, 'built': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'radius_km': Rk,
            'projection': 'equirectangular, north up, east to the right', 'max_tier': '%dk' % top,
            'ILLUSTRATIVE': 'no spacecraft image of %s exists: the terrain is %s\'s (artist\'s full-globe map) rotated by %s deg, '
                            'recoloured to the measured colour / albedo' % (cfg['name'], sc['name'], cfg['rot']),
            'normal': 'tangent space (x east, y north), true scale from the artist\'s bump map, height span %g km' % cfg['relief_km'],
            'colour': 'measured colour %s with provinces %s, luminance mean %g' % (cfg['rgb'], cfg['rgb2'], cfg['mean']),
            'files': sorted(files), 'source': cfg['source'], 'credit': cfg['credit']}
    json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'), indent=1)
    print('  %s done in %d s' % (key, time.time() - t0), flush=True)


def build_hybrid(key, cfg, raw, out_root, Wt=None):
    """real large-scale albedo of the mission mosaic (low-passed) x the artist's terrain detail (a Uranian moon rotated) +
    analytic big craters; relief = high-passed artist bump + analytic crater profiles, true scale"""
    t0 = time.time()
    hy = cfg['hybrid']
    tmp = os.path.join(SCRATCH, 'hyb_tmp')
    build(key, cfg, raw, tmp, Wt)                                     # the real colour map (deterministic)
    top = cfg['top']
    W, H = (Wt, Wt // 2) if Wt else SIZES[top]
    kk0 = [k for k, v in SIZES.items() if v == (W, H)][0]
    C = np.asarray(Image.open(os.path.join(tmp, key, '%s_color_%dk.webp' % (key, kk0))).convert('RGB'), np.float32)
    assert C.shape[:2] == (H, W), C.shape
    Rk = cfg['radius_km']
    lat_r = 90 - (np.arange(H) + 0.5) * 180 / H
    sig = hy['sig_deg'] * W / 360.0
    print('== hybrid', key, 'terrain', hy['terrain'], W, 'x', H, 'sigma %.1f px' % sig, flush=True)
    # --- crater check image (real 2k colour map, red = lon_east as given, cyan = west convention) --------------------------
    c2 = cv2.resize(C, (2048, 1024), interpolation=cv2.INTER_AREA).astype(np.uint8)
    chk = np.ascontiguousarray(c2[..., ::-1])
    for (la, lo, dm) in [c[:3] for c in hy['craters']]:
        for lo_, colr in ((lo, (0, 0, 255)), (-lo, (255, 255, 0))):
            ang = np.linspace(0, 2 * np.pi, 181)
            rr = (dm / 2) / Rk
            la0, lo0 = np.radians(la), np.radians(lo_)
            sl = np.arcsin(np.clip(np.sin(la0) * np.cos(rr) + np.cos(la0) * np.sin(rr) * np.cos(ang), -1, 1))
            ln = lo0 + np.arctan2(np.sin(ang) * np.sin(rr) * np.cos(la0), np.cos(rr) - np.sin(la0) * np.sin(sl))
            px = ((np.degrees(ln) + 180) % 360) / 360 * 2048
            py = (90 - np.degrees(sl)) / 180 * 1024
            cv2.polylines(chk, [np.stack([px, py], 1).astype(np.int32)], True, colr, 2)
    cv2.imwrite(os.path.join(SCRATCH, '%s_crater_check.png' % key), chk)
    # --- colour ----------------------------------------------------------------------------------------------------------
    Clo = np.dstack([iso_blur(np.ascontiguousarray(C[..., c]), sig, lat_r) for c in range(3)])
    del C
    a = np.radians(hy['rot'])
    Rx = np.array([[1, 0, 0], [0, np.cos(a[0]), -np.sin(a[0])], [0, np.sin(a[0]), np.cos(a[0])]])
    Ry = np.array([[np.cos(a[1]), 0, np.sin(a[1])], [0, 1, 0], [-np.sin(a[1]), 0, np.cos(a[1])]])
    Rz = np.array([[np.cos(a[2]), -np.sin(a[2]), 0], [np.sin(a[2]), np.cos(a[2]), 0], [0, 0, 1]])
    Rm = Rz @ Ry @ Rx
    sc = BODIES[hy['terrain']]
    rawk = os.path.join(raw, hy['terrain']); os.makedirs(rawk, exist_ok=True)
    A = np.asarray(Image.open(art_src(rawk, sc, sc['art_color'])).convert('RGB'))
    up = W / A.shape[1]
    La = luma(cv2.resize(A, (W, H), interpolation=cv2.INTER_AREA if up <= 1 else cv2.INTER_LANCZOS4).astype(np.float32))
    del A
    if cfg.get('despot'):
        La = despot(La, cfg['despot'], W)
    La = sphere_rotate(La, Rm)
    La = La + ART_USM[1] * (La - blur(La, ART_USM[0] * max(up, 1.0)))
    if cfg.get('hl'):                                                 # bright spots (painted / copied terrain): soft cap
        La = hl_cap(La, cfg['hl'], W)
    La = np.maximum(La, 1.0)
    D = np.clip(La / np.maximum(iso_blur(La, sig, lat_r), 1.0), *cfg.get('hyb_clip', (0.55, 1.6)))
    del La
    col = Clo * D[..., None]
    del Clo, D
    col *= float(cfg['mean']) / max(float(luma(col).mean()), 1e-3)
    T = KNEE * 255
    col = np.where(col > T, T + (255 - T) * (1 - np.exp(-(col - T) / (255 - T))), col)
    c8 = np.clip(np.rint(col), 0, 255).astype(np.uint8)
    del col
    # --- relief ----------------------------------------------------------------------------------------------------------
    B = np.asarray(Image.open(art_src(rawk, sc, sc['art_bump'])).convert('L'))
    B = cv2.resize(B, (W, H), interpolation=cv2.INTER_AREA if B.shape[1] >= W else cv2.INTER_CUBIC).astype(np.float32)
    B = sphere_rotate(blur(B, 1.6 * W / 8192), Rm)
    B = B - iso_blur(B, 10.0 * W / 360.0, lat_r)
    lo, hi = np.percentile(B, [0.5, 99.5])
    hgt = np.clip((B - lo) / max(hi - lo, 1.0), 0, 1).astype(np.float32) * hy['relief_km']
    del B
    lon = np.radians((np.arange(W) + 0.5) / W * 360 - 180)[None, :]
    la_all = np.radians(lat_r)
    for cr in hy['craters']:
        la, lo_e, dm, dep, rim, pk = cr[:6]
        op = cr[6] if len(cr) > 6 else {}
        R = dm / 2.0
        la0, lo0 = np.radians(la), np.radians(lo_e)
        for r0 in range(0, H, 256):
            r1 = min(H, r0 + 256)
            lt = la_all[r0:r1, None]
            hv = np.sin((lt - la0) / 2) ** 2 + np.cos(lt) * np.cos(la0) * np.sin((lon - lo0) / 2) ** 2
            d = 2 * Rk * np.arcsin(np.sqrt(np.clip(hv, 0, 1)))        # great-circle km
            fl = np.minimum(smooth((d / R - 0.85) / 0.3) + op.get('floor', 0.25), 1.0)
            x_ = d / R                                                 # s48: continuous at the rim (the old Gaussian + ejecta
            prof = np.where(x_ < 1, -dep * (1 - x_ ** 2) + rim * x_ ** 4,   # step drew a sharp ring line in the normals)
                            rim / np.maximum(x_, 1.0) ** 3)
            prof = prof + pk * np.exp(-(d / (0.12 * R)) ** 2)
            if op.get('dome'):
                prof = prof + op['dome'] * np.exp(-(d / (0.25 * R)) ** 2)
            hgt[r0:r1] = hgt[r0:r1] * fl + prof
        print('  crater lat %g lon_east %g diam %g km added' % (la, lo_e, dm), flush=True)
    for tr in hy.get('troughs', []):
        pla, plo = np.radians(tr['pole'][0]), np.radians(tr['pole'][1])
        Pv = np.array([np.cos(pla) * np.cos(plo), np.cos(pla) * np.sin(plo), np.sin(pla)])
        cl_ = np.radians(tr['centre_lon'])
        C0 = np.array([np.cos(cl_), np.sin(cl_), 0.0])
        C0 -= Pv * float(C0 @ Pv); C0 /= np.linalg.norm(C0)          # arc centre = equatorial crossing of the circle
        Tv = np.cross(Pv, C0)
        for r0 in range(0, H, 256):
            r1 = min(H, r0 + 256)
            lt = la_all[r0:r1, None]
            ux, uy, uz = np.cos(lt) * np.cos(lon), np.cos(lt) * np.sin(lon), np.broadcast_to(np.sin(lt), (r1 - r0, W))
            sv = np.arcsin(np.clip(ux * Pv[0] + uy * Pv[1] + uz * Pv[2], -1, 1))
            s_km = Rk * sv
            phi = np.degrees(np.abs(np.arctan2(ux * Tv[0] + uy * Tv[1] + uz * Tv[2], ux * C0[0] + uy * C0[1] + uz * C0[2])))
            wt = 1 - smooth((phi - (tr['half_deg'] - tr['taper_deg'])) / tr['taper_deg'])
            # s48: flat-floored graben with steep walls (half width tr['w'], walls tr['wall'] km wide) + raised shoulders
            prof = -tr['depth'] * (1 - smooth((np.abs(s_km) - (tr['w'] - tr['wall'])) / tr['wall'])) + tr['shoulder'] * np.exp(-((np.abs(s_km) - tr['shoulder_r']) / tr['shoulder_w']) ** 2)
            hgt[r0:r1] += (prof * wt).astype(np.float32)
        print('  trough %s added' % tr['name'], flush=True)
    hs_ = cv2.resize(hgt, (2048, 1024), interpolation=cv2.INTER_AREA)
    nh = height_normals(hs_, Rk)
    shd = np.clip(nh @ np.array([-0.5, 0.5, 0.7071], np.float32), 0, 1)
    cv2.imwrite(os.path.join(SCRATCH, '%s_relief_check.png' % key), np.clip(shd / max(float(np.percentile(shd, 99.5)), 1e-3) * 255, 0, 255).astype(np.uint8))
    pw = (1 - smooth((np.abs(lat_r) - 84) / 5))[:, None]
    n8 = n_to_8(height_normals(hgt, Rk, np.broadcast_to(pw, (H, W)).astype(np.float32)))
    print('  hybrid height min %.2f max %.2f km' % (float(hgt.min()), float(hgt.max())), flush=True)
    out = os.path.join(out_root, key); os.makedirs(out, exist_ok=True)
    files = []
    for kk, (w, hh) in SIZES.items():
        if (Wt is None and kk > top) or (Wt and w > W):
            continue
        cc = c8 if w == W else cv2.resize(c8, (w, hh), interpolation=cv2.INTER_AREA)
        nm = '%s_color_%dk.webp' % (key, kk); webp(cc, os.path.join(out, nm), 90); files.append(nm)
        nn = n8 if w == W else cv2.resize(n8, (w, hh), interpolation=cv2.INTER_AREA)
        nm = '%s_normal_%dk.webp' % (key, kk); webp(nn, os.path.join(out, nm), 92); files.append(nm)
    cn = ', '.join('%g km crater at %g N, %g E' % (c[2], c[0], c[1]) for c in hy['craters']) + ''.join(', trough ' + t['name'] for t in hy.get('troughs', []))
    meta = {'v': 1, 'built': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'radius_km': Rk,
            'projection': 'equirectangular, north up, east to the right, lon 0 at the centre', 'max_tier': '%dk' % top,
            'HYBRID': 'real large-scale albedo (USGS/DLR mosaic low-passed %g deg) x artist terrain detail (Astra-Planetshine %s, rotated) '
                      '+ analytic crater (%s)' % (hy['sig_deg'], hy['terrain'], cn),
            'normal': 'tangent space (x east, y north), true scale: artist bump high-passed (span %g km) + analytic crater profiles' % hy['relief_km'],
            'files': sorted(files), 'source': cfg['source'] + ' + Astra-Planetshine "Uranus Project Missing Data" terrain',
            'credit': cfg['credit'] + '; ' + sc['credit']}
    json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'), indent=1)
    print('  %s hybrid done in %d s' % (key, time.time() - t0), flush=True)


def rot_fill(L, M, rgb, lat_max, f, sig=SIG_ROT):
    """unseen hemisphere (Voyager / New Horizons saw one side): filled with the body's OWN imaged terrain seen through
    rotations of the sphere about the lon +-90 axis (180 deg first: (lat, lon) -> (-lat, 180 - lon); then +-90 deg, which
    bring mid-latitude terrain to the far pole). Rotations, not mirrors: craters keep their handedness. Only the band-pass
    detail (scales < SIG_ROT px) is copied; the low band is a smooth continuation of the data (copying the low band carried
    the mosaics' illumination blobs). Sources poleward of |lat| = lat_max are left out (a polar feature such as Charon's
    Mordor Macula must not be copied). Detail fades in over RAMP_ROT / 2 px from the data edge."""
    H, W = L.shape
    lat = np.radians(90 - (np.arange(H) + 0.5) * 180 / H)
    lon = np.radians((np.arange(W) + 0.5) / W * 360 - 180)
    cl = np.cos(lat)[:, None]
    X, Y, Z = cl * np.cos(lon)[None, :], cl * np.sin(lon)[None, :], np.broadcast_to(np.sin(lat)[:, None], (H, W))
    Ms = (M * smooth((lat_max - np.abs(np.degrees(lat))) / 4.0)[:, None]          # off the data rim (limb shading, dark fringe)
          * smooth((dist(M > 0) - 6 * f) / (10 * f))).astype(np.float32)
    gap = M < 1
    d = dist(gap)
    wh = smooth(d / (0.5 * RAMP_ROT * f)) * gap
    sl = sig * f
    p = 8
    chans = [L] if rgb is None else [L] + [np.ascontiguousarray(rgb[..., c]) for c in range(3)]
    e = np.exp(-d / (RAMP_ROT * f)).astype(np.float32)               # far from the data: the body's mean level (pushpull
    mw = M * np.cos(lat)[:, None]                                    # alone drags edge features into long columns)
    lows = [blur(pushpull(x, M), sl) * e + float((x * mw).sum() / mw.sum()) * (1 - e) for x in chans]
    acc = [np.zeros((H, W), np.float32) for _ in chans]
    # detail band taken on the SOURCE grid, then rotated (a lat/lon blur of rotated content smears radially at the poles)
    his = [(x - nblur(x, np.maximum(Ms, 1e-3), sl)) * (Ms > 0.5) for x in chans]
    his_ = []
    for hx_ in his:                                                    # s48: soft amplitude clip, no distinctive feature clones
        c_ = 1.5 * float(np.sqrt(np.mean(hx_[Ms > 0.5] ** 2))) + 1e-6
        his_.append((c_ * np.tanh(hx_ / c_)).astype(np.float32))
    his = his_
    wsum = np.zeros((H, W), np.float32)
    # 180 deg about the lon 90 axis first; then oblique axes (x/y-axis quarter turns map the equator onto meridians, so E-W
    # features of every pass converged into a star at the pole)
    for ax, ang in (((0, 1, 0), 180), ((1, 1, 0.6), 100), ((-1, 0.5, 0.8), -110), ((0.3, -1, 0.5), 95), ((1, -0.4, -0.7), -100)):
        k_ = np.array(ax, np.float64); k_ /= np.linalg.norm(k_)
        t = np.radians(-ang)                                        # source = R(-ang) . destination (Rodrigues)
        c, s_ = np.cos(t), np.sin(t)
        kd = k_[0] * X + k_[1] * Y + k_[2] * Z
        cx, cy, cz = k_[1] * Z - k_[2] * Y, k_[2] * X - k_[0] * Z, k_[0] * Y - k_[1] * X
        xs = X * c + cx * s_ + k_[0] * kd * (1 - c)
        ys = Y * c + cy * s_ + k_[1] * kd * (1 - c)
        zs = Z * c + cz * s_ + k_[2] * kd * (1 - c)
        lat_s = np.degrees(np.arcsin(np.clip(zs, -1, 1)))
        lon_s = np.degrees(np.arctan2(ys, xs))
        mx = ((lon_s + 180) / 360 * W - 0.5 + p).astype(np.float32)
        my = ((90 - lat_s) / 180 * H - 0.5).astype(np.float32)
        rm = lambda x: cv2.remap(wrap_pad(x, p).astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mr = rm(Ms)
        mr = smooth((blur(mr, 12 * f + 0.5) - 0.3) / 0.6) * gap       # soft source edge (no hard seams)
        w = (mr * (1 - wsum)).astype(np.float32)
        for k, hx in enumerate(his):
            acc[k] += w * rm(hx)
        wsum += w
        print('  rotation %s %+d deg: covers %.3f' % (str(ax), ang, float(w.mean())), flush=True)
    # what no rotation reaches: isotropic sphere-noise detail at the data's own relative detail level (plain, not blank)
    hp = L - blur(L, sl)
    rel = float(np.median((np.sqrt(nblur(hp * hp, M, 16 * f)) / np.maximum(nblur(L, M, 16 * f), 4.0))[M > 0]))
    Nz = sphere_noise(H, W, tuple(max(l * f, 1.0) for l in (4, 8, 16, 32)), 11) * rel * (1 - wsum)
    out = [np.where(gap, lo + (a_ + Nz * lo) * wh, x).astype(np.float32) for x, lo, a_ in zip(chans, lows, acc)]
    L = out[0]
    if rgb is not None:
        rgb = np.dstack(out[1:])
    print('  rotation fill: detail on %.3f of the map, plain continuation on %.3f' % (float(((wsum > 0.5) & gap).mean()), float(((wsum <= 0.5) & gap).mean())), flush=True)
    M2 = np.ones_like(M)                                            # filled everywhere: no ring inpaint (it streaks)
    return L, M2, rgb, ('; unseen areas: smooth continuation of the data + band-pass detail of the body\'s own imaged terrain '
                        'seen through rotations of the sphere (sources |lat| <= %g; a plausible stand-in, not observed)' % lat_max)


def smooth_mask(L, M, thr, f):
    """airbrushed fill inside a 'GlobalFill' mosaic (Triton): no fine detail -> treated as no data"""
    hp = L - blur(L, 2 * f + 0.5)
    rel = np.sqrt(nblur(hp * hp, M, 12 * f)) / np.maximum(nblur(L, M, 12 * f), 8.0)
    bad = ((rel < thr) & (M > 0)).astype(np.uint8)
    k = max(3, int(16 * f) | 1)
    bad = cv2.morphologyEx(bad, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    bad = cv2.dilate(bad, np.ones((k, k), np.uint8))
    print('  smooth (airbrushed) area masked: %.3f (rel detail median %.4f)' % (float(bad.mean()), float(np.median(rel[M > 0]))), flush=True)
    return M * (1 - bad)


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
        (build_hybrid if BODIES[k].get('hybrid') else build_tno if BODIES[k].get('tno') else build_direct if BODIES[k].get('direct') else build_art if BODIES[k].get('art') else build)(k, BODIES[k], raw, out, Wt)
    print('done in', round(time.time() - t0), 's')


if __name__ == '__main__':
    main()
