"""
Deterministic packshot layout: cut products out of plain light backgrounds and
place them on a flat #FAFAFA 16:9 canvas with soft, identical shadows.
"""
import threading

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter
from scipy import ndimage as ndi

W, H = 3840, 2160              # 16:9 output
BG = (250, 250, 250)           # #FAFAFA
MAX_PRODUCT_H = 1000           # product height in px
GAP = 80                       # space between products
BOTTOM = 1290                  # baseline the products stand on
TOP_LIMIT = 0.65               # products + shadows stay within top 65%
SIDE_MARGIN = 160              # minimum left/right margin
BG_TOLERANCE = 14              # max per-channel diff to count as background
MAX_INPUT_SIDE = 1400          # downscale large uploads to save memory

# Image decoding and cut-outs are memory-heavy, so they run one at a time even
# when Gemini calls run in parallel. Keeps peak memory inside a 512 MB instance.
cpu_lock = threading.RLock()


def load_image(fp):
    im = Image.open(fp)
    im.draft("RGB", (MAX_INPUT_SIDE, MAX_INPUT_SIDE))   # JPEGs decode at reduced size
    im = im.convert("RGBA") if im.mode in ("RGBA", "LA", "P") else im.convert("RGB")
    if im.mode == "RGBA":                       # flatten transparency onto white
        flat = Image.new("RGB", im.size, (255, 255, 255))
        flat.paste(im, mask=im.split()[-1])
        im = flat
    im.thumbnail((MAX_INPUT_SIDE, MAX_INPUT_SIDE), Image.LANCZOS)
    return im


def _border(a):
    return np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]).astype(np.int16)


def plain_background(im):
    """True if the photo sits on a plain, light, uniform background."""
    border = _border(np.asarray(im))
    bgc = np.median(border, axis=0)
    uniform = (np.abs(border - bgc).max(-1) < BG_TOLERANCE).mean()
    return uniform > 0.85 and bgc.mean() > 200


def cut_out(im):
    """Return an RGBA cutout of the product on a plain light background."""
    with cpu_lock:
        return _cut_out(im)


def _cut_out(im):
    im.thumbnail((MAX_INPUT_SIDE, MAX_INPUT_SIDE), Image.LANCZOS)
    a = np.asarray(im)
    bgc = np.median(_border(a), axis=0).astype(np.int16)
    close = np.ones(a.shape[:2], dtype=bool)
    for ch in range(3):                         # per channel keeps memory low
        close &= np.abs(a[..., ch].astype(np.int16) - bgc[ch]) < BG_TOLERANCE
    del a
    lab, _ = ndi.label(close)
    edge = set(np.unique(np.concatenate(
        [lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(edge))

    fg = ndi.binary_fill_holes(~bg)
    fg = ndi.binary_opening(fg, iterations=2)
    lbl, n = ndi.label(fg)
    if n == 0:
        raise ValueError("no product found")
    sizes = ndi.sum(fg, lbl, range(1, n + 1))
    fg = lbl == (np.argmax(sizes) + 1)

    mask = Image.fromarray((fg * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(0.8))
    out = im.convert("RGBA")
    out.putalpha(mask)
    bbox = mask.point(lambda v: 255 if v > 20 else 0).getbbox()
    return out.crop(bbox)


def compose(cutouts):
    """Lay cutouts out left-to-right on the canvas."""
    n = len(cutouts)
    ratios = [c.width / c.height for c in cutouts]
    usable_w = W - 2 * SIDE_MARGIN - GAP * (n - 1)
    h = int(min(MAX_PRODUCT_H, usable_w / sum(ratios)))
    bottom = min(BOTTOM, int(H * TOP_LIMIT) - 40)
    if bottom - h < 40:
        h = bottom - 40

    ims = [c.resize((max(1, round(c.width * h / c.height)), h), Image.LANCZOS)
           for c in cutouts]
    total = sum(i.width for i in ims) + GAP * (n - 1)
    x = (W - total) // 2

    canvas = Image.new("RGB", (W, H), BG)
    shadow = Image.new("L", (W, H), 0)
    contact = Image.new("L", (W, H), 0)
    ds, dc = ImageDraw.Draw(shadow), ImageDraw.Draw(contact)

    positions = []
    for im in ims:
        positions.append((x, bottom - h))
        cx = x + im.width / 2
        ds.ellipse([cx - im.width * 0.46, bottom - 14,
                    cx + im.width * 0.46, bottom + 30], fill=110)
        dc.ellipse([x + im.width * 0.08, bottom - 6,
                    x + im.width * 0.92, bottom + 10], fill=90)
        x += im.width + GAP

    shadow = ImageChops.lighter(shadow.filter(ImageFilter.GaussianBlur(22)),
                                contact.filter(ImageFilter.GaussianBlur(7)))
    del contact
    shadow = shadow.point(lambda v: int(v * 0.55))
    canvas.paste((60, 60, 60), (0, 0, W, H), shadow)   # dark tint through the shadow mask
    del shadow
    for (px, py), im in zip(positions, ims):
        canvas.paste(im, (px, py), im)
    return canvas
