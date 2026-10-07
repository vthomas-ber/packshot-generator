"""
Deterministic packshot layout: cut products out of plain light backgrounds and
place them on a flat #FAFAFA 16:9 canvas with soft, identical shadows.
"""
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
MAX_INPUT_SIDE = 2000          # downscale large uploads to save memory


def load_image(fp):
    im = Image.open(fp)
    im = im.convert("RGBA") if im.mode in ("RGBA", "LA", "P") else im.convert("RGB")
    if im.mode == "RGBA":                       # flatten transparency onto white
        flat = Image.new("RGB", im.size, (255, 255, 255))
        flat.paste(im, mask=im.split()[-1])
        im = flat
    im.thumbnail((MAX_INPUT_SIDE, MAX_INPUT_SIDE), Image.LANCZOS)
    return im


def plain_background(im):
    """True if the photo sits on a plain, light, uniform background."""
    a = np.asarray(im).astype(int)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bgc = np.median(border, axis=0)
    uniform = (np.abs(border - bgc).max(-1) < BG_TOLERANCE).mean()
    return uniform > 0.85 and bgc.mean() > 200


def cut_out(im):
    """Return an RGBA cutout of the product on a plain light background."""
    a = np.asarray(im).astype(int)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bgc = np.median(border, axis=0)
    close = np.abs(a - bgc).max(-1) < BG_TOLERANCE
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

    canvas = Image.new("RGBA", (W, H), BG + (255,))
    soft = Image.new("L", (W, H), 0)
    contact = Image.new("L", (W, H), 0)
    ds, dc = ImageDraw.Draw(soft), ImageDraw.Draw(contact)

    positions = []
    for im in ims:
        positions.append((x, bottom - h))
        cx = x + im.width / 2
        ds.ellipse([cx - im.width * 0.46, bottom - 14,
                    cx + im.width * 0.46, bottom + 30], fill=110)
        dc.ellipse([x + im.width * 0.08, bottom - 6,
                    x + im.width * 0.92, bottom + 10], fill=90)
        x += im.width + GAP

    soft = soft.filter(ImageFilter.GaussianBlur(22))
    contact = contact.filter(ImageFilter.GaussianBlur(7))
    shadow = ImageChops.lighter(soft, contact).point(lambda v: int(v * 0.55))
    dark = Image.new("RGBA", (W, H), (60, 60, 60, 255))
    canvas = Image.composite(dark, canvas, shadow)
    for (px, py), im in zip(positions, ims):
        canvas.alpha_composite(im, (px, py))
    return canvas.convert("RGB")
