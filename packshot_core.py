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
KEY_TOLERANCE = 70             # same, for Gemini's coloured key background
LOST_LIMIT = 0.003             # >0.3% of the product dropped = cut-out is suspect

# Background colours Gemini can be asked to use. The one least present in the
# product is chosen, so it can be removed without touching the pack.
KEY_COLOURS = {"bright green": (0, 255, 0), "magenta": (255, 0, 255),
               "pure blue": (0, 0, 255)}
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


def pick_key_colour(im):
    """Name and RGB of the key colour that appears least in the product."""
    small = np.asarray(im.resize((120, 120))).astype(np.int16).reshape(-1, 3)
    best, best_score = None, None
    for name, rgb in KEY_COLOURS.items():
        near = (np.abs(small - np.array(rgb)).max(-1) < KEY_TOLERANCE + 20).mean()
        if best_score is None or near < best_score:
            best, best_score = (name, rgb), near
    return best


def cut_out(im, tolerance=BG_TOLERANCE, erode=0):
    """Return an RGBA cutout of the product (see cut_out_checked)."""
    return cut_out_checked(im, tolerance, erode)[0]


def cut_out_checked(im, tolerance=BG_TOLERANCE, erode=0):
    """Return (RGBA cutout, share of product pixels that had to be dropped).

    A high share means parts of the pack matched the background colour (for
    example a white label touching a white background) and were cut away.
    """
    with cpu_lock:
        return _cut_out(im, tolerance, erode)


def _cut_out(im, tolerance, erode):
    im.thumbnail((MAX_INPUT_SIDE, MAX_INPUT_SIDE), Image.LANCZOS)
    a = np.asarray(im)
    bgc = np.median(_border(a), axis=0).astype(np.int16)
    close = np.ones(a.shape[:2], dtype=bool)
    for ch in range(3):                         # per channel keeps memory low
        close &= np.abs(a[..., ch].astype(np.int16) - bgc[ch]) < tolerance
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
    lost = 1 - sizes.max() / sizes.sum()
    fg = lbl == (np.argmax(sizes) + 1)
    if erode:                                   # trim the colour fringe at the edge
        fg = ndi.binary_erosion(fg, iterations=erode)

    mask = Image.fromarray((fg * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(0.8))
    out = im.convert("RGBA")
    out.putalpha(mask)
    bbox = mask.point(lambda v: 255 if v > 20 else 0).getbbox()
    return out.crop(bbox), float(lost)


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
