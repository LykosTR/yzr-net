#!/usr/bin/env python3
"""Milky Way panorama feed: ESO eso0932a (S. Brunier, CC BY 4.0) -> star-free, background-free
equirectangular galactic panorama (l=0 at the centre column, l increasing to the LEFT, b=0 at the middle row).
Output: out/mw_4k.webp (4096x2048), out/mw_2k.webp (2048x1024), out/meta.json.  Needs: pip install pillow numpy"""
import io, json, os, sys, time, urllib.request
import numpy as np
from PIL import Image, ImageFilter

Image.MAX_IMAGE_PIXELS = None
SRC = ['https://cdn.eso.org/images/original/eso0932a.tif',      # 6000x3000, largest free size
       'https://cdn.eso.org/images/large/eso0932a.jpg',
       'https://cdn.eso.org/images/publicationjpg/eso0932a.jpg']
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
W, H = 4096, 2048


def get():
    last = None
    for u in SRC:
        try:
            req = urllib.request.Request(u, headers={'User-Agent': 'yzr-net-data-feed'})
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
            im = Image.open(io.BytesIO(data)); im.load()
            return u, im.convert('RGB')
        except Exception as e:  # try the next size
            last = e; print('src failed', u, e, file=sys.stderr)
    raise SystemExit('no ESO source reachable: %s' % last)


def _opening(im, k):
    return im.filter(ImageFilter.MinFilter(k)).filter(ImageFilter.MaxFilter(k))


def starless(im):
    """Stars are bright features narrower than the structuring element: white top-hat of the max
    channel (grey opening, so dark dust lanes and extended glow are untouched) marks them.  The mask is
    dilated + feathered and those pixels are replaced by min(original, median-filtered copy) smoothed lightly
    (= the local background).  Two scales: ordinary stars (7 px), then saturated stars / planets (21 px)."""
    arr = np.asarray(im).astype(np.float32)
    cur = im
    for size, thr, dil, med in ((7, 5.0, 5, 9), (21, 40.0, 17, 25)):
        L = Image.fromarray(np.clip(arr.max(2), 0, 255).astype(np.uint8))
        th = np.asarray(L).astype(np.float32) - np.asarray(_opening(L, size)).astype(np.float32)
        m = Image.fromarray((th > thr).astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(dil)).filter(ImageFilter.GaussianBlur(1.2))
        mask = (np.asarray(m).astype(np.float32) / 255.0)[..., None]
        med_im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(med))
        fill = np.minimum(arr, np.asarray(med_im).astype(np.float32))
        fill = np.asarray(Image.fromarray(np.clip(fill, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))).astype(np.float32)
        arr = arr * (1 - mask) + fill * mask
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.5))


def main():
    t0 = time.time()
    url, im = get()
    print('source', url, im.size, '%.0fs' % (time.time() - t0))
    # ESO eso0932a is already a 2:1 equirectangular galactic panorama (l=0 centre, l increases leftwards).
    if abs(im.size[0] / im.size[1] - 2.0) > 0.01:
        raise SystemExit('unexpected aspect %s' % (im.size,))
    im = im.resize((W, H), Image.LANCZOS)
    im = starless(im)
    a = np.asarray(im).astype(np.float32)
    # background: per-row (galactic latitude) floor from the polar caps, constant inside |b|<35 deg, then a black point
    b = (0.5 - (np.arange(H) + 0.5) / H) * 180.0
    rows = np.percentile(a, 40, axis=1)                    # (H,3) robust per-row sky level
    from_pole = np.abs(b) > 35
    bg = np.zeros_like(rows)
    for c in range(3):
        n = np.convolve(rows[:, c], np.ones(41) / 41, mode='same')
        top, bot = n[from_pole & (b > 0)], n[from_pole & (b < 0)]
        edge_t = np.median(n[(b > 33) & (b < 40)]); edge_b = np.median(n[(b < -33) & (b > -40)])
        col = n.copy()
        col[(b > 0) & (b <= 35)] = edge_t; col[(b < 0) & (b >= -35)] = edge_b
        bg[:, c] = col
    a = a - bg[:, None, :]
    pole = np.abs(b) > 55
    bp = np.percentile(a[pole].reshape(-1, 3), 70, axis=0)  # residual black point (per channel)
    a = np.clip(a - bp, 0, None)
    a = a * (255.0 / max(a.max(), 1)) if a.max() < 200 else a
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    os.makedirs(OUT, exist_ok=True)
    out.save(os.path.join(OUT, 'mw_4k.webp'), quality=85, method=6)
    out.resize((W // 2, H // 2), Image.LANCZOS).save(os.path.join(OUT, 'mw_2k.webp'), quality=85, method=6)
    meta = {
        'at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'source': 'https://www.eso.org/public/images/eso0932a/', 'download': url,
        'credit': 'ESO/S. Brunier', 'license': 'CC BY 4.0',
        'projection': 'equirectangular (plate carree), galactic coordinates, 2:1',
        'l0_column': 'centre column (x = width/2); b=0 on the middle row',
        'direction': 'l increases to the LEFT (u = 0.5 - l/360, l in -180..180); b increases upwards',
        'width': W, 'height': H,
        'processing': 'stars removed (top-hat 7/21 px star mask, replaced by min(original, median 9/25) smoothed), per-row sky gradient subtracted (|b|<35 held constant), polar black point subtracted',
        'black_point_removed_8bit': [round(float(x), 2) for x in bp],
        'row_background_8bit_median': [round(float(x), 2) for x in np.median(bg, axis=0)],
    }
    with open(os.path.join(OUT, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print('done %.0fs' % (time.time() - t0), {k: os.path.getsize(os.path.join(OUT, k)) for k in os.listdir(OUT)})


if __name__ == '__main__':
    main()
