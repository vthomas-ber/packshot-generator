"""
Tray and Surprise scenes. Both take the same cleaned product cut-outs as the
neutral mode and build the scene with code, so counts, grouping and the TGTG
box are exact every time.

Tray:     products stand shoulder to shoulder in a kraft display tray, split
          evenly between product types and grouped left to right.
Surprise: products drop into the TGTG box. The box image never changes; the
          products sit between its back and front so the front rim hides
          their lower part.
"""
import math
import os

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from packshot_core import BG, H, W

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- tray ----

KRAFT = (196, 163, 118)
KRAFT_DARK = (160, 128, 88)
MAX_UNITS = 20


def split_units(n_types, units):
    """Even split of `units` across `n_types`; remainder goes to the first types."""
    base, extra = divmod(units, n_types)
    return [base + (1 if i < extra else 0) for i in range(n_types)]


def _kraft(size, colour, seed):
    """Flat kraft-cardboard texture: fine noise plus faint horizontal fibres."""
    w, h = size
    rng = np.random.default_rng(seed)
    a = np.empty((h, w, 3), dtype=np.float32)
    a[:] = colour
    a += rng.normal(0, 4.0, (h, w, 1))                       # grain
    a += rng.normal(0, 3.0, (h, 1, 1))                       # fibres
    a *= np.linspace(1.04, 0.95, h, dtype=np.float32)[:, None, None]  # top lit
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def compose_tray(cutouts, units):
    """Display tray with `units` products split evenly across the cut-outs."""
    counts = split_units(len(cutouts), units)
    rows = 1 if units <= 6 else (2 if units <= 12 else 3)
    per_row = math.ceil(units / rows)

    # Column-major fill keeps each product type in one block, left to right.
    seq = [i for i, c in enumerate(counts) for _ in range(c)]
    columns = [seq[c * rows:(c + 1) * rows] for c in range(per_row)]
    columns = [col for col in columns if col]

    ratios = [c.width / c.height for c in cutouts]
    col_ratio = [max(ratios[t] for t in col) for col in columns]

    # Size: fit tray width (~74% of canvas) and height (~66% incl. back rows).
    lift = 0.16                                   # back rows rise this much of h
    wall = 0.035                                  # side wall thickness, of h
    h_w = (W * 0.74) / (sum(col_ratio) + 2 * wall)
    h_h = (H * 0.66) / (1 + lift * (rows - 1) + 0.06)
    h = int(min(h_w, h_h))
    t = max(6, int(h * wall))

    widths = [round(r * h * 0.92) for r in col_ratio]   # packs touch, no gaps
    inner_w = sum(widths)
    tray_w = inner_w + 2 * t
    lip_h = int(h * 0.25)                         # lip covers bottom 25% of front row
    floor_h = max(4, int(h * 0.03))
    total_h = int(h * (1 + lift * (rows - 1))) + floor_h

    x0 = (W - tray_w) // 2
    base_y = (H + total_h) // 2                   # bottom of the tray (centred block)
    floor_top = base_y - floor_h                  # products stand here

    canvas = Image.new("RGB", (W, H), BG)

    # Floor shadow under the tray.
    sh = Image.new("L", (W, H), 0)
    ImageDraw.Draw(sh).rounded_rectangle(
        [x0 - 10, base_y - 10, x0 + tray_w + 10, base_y + 16], radius=12, fill=150)
    sh = sh.filter(ImageFilter.GaussianBlur(14)).point(lambda v: int(v * 0.45))
    canvas.paste((60, 60, 60), (0, 0, W, H), sh)

    # Products: back rows first (slightly smaller, raised), front row last.
    resized = {}
    for r in reversed(range(rows)):
        scale = 0.94 ** r
        hh = int(h * scale)
        x = x0 + t
        for ci, col in enumerate(columns):
            if r < len(col):
                cut = cutouts[col[r]]
                key = (col[r], hh)
                if key not in resized:
                    resized[key] = cut.resize(
                        (max(1, round(cut.width * hh / cut.height)), hh), Image.LANCZOS)
                im = resized[key]
                if r:                               # rows further back sit in shade
                    rgb = im.convert("RGB").point(lambda v: int(v * (1 - 0.07 * r)))
                    im = Image.merge("RGBA", (*rgb.split(), im.split()[-1]))
                cx = x + widths[ci] / 2
                y = floor_top - int(h * lift * r) - hh
                canvas.paste(im, (int(cx - im.width / 2), y), im)
            x += widths[ci]

    # Soft shadow the lip casts up onto the products just above it.
    lip_top = floor_top - lip_h
    band = Image.new("L", (W, H), 0)
    ImageDraw.Draw(band).rectangle([x0 + t, lip_top - 30, x0 + t + inner_w, lip_top], fill=70)
    band = band.filter(ImageFilter.GaussianBlur(12))
    canvas.paste((40, 30, 20), (0, 0, W, H), band)

    # Front lip.
    lip = _kraft((tray_w, lip_h + floor_h), KRAFT, 9)
    shade = np.ones(tray_w, dtype=np.float32)
    edge = max(1, int(tray_w * 0.04))
    shade[:edge] = np.linspace(0.86, 1, edge)                # rounded-looking ends
    shade[-edge:] = np.linspace(1, 0.86, edge)
    lip = Image.fromarray(np.clip(np.asarray(lip) * shade[None, :, None], 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(lip)
    d.line([(0, 1), (tray_w, 1)], fill=(224, 198, 158), width=3)          # cut edge
    d.line([(0, 4), (tray_w, 4)], fill=(150, 118, 80), width=1)
    canvas.paste(lip, (x0, lip_top))

    # Side walls rise above the lip at both ends with a curved cut, like a
    # shelf-ready tray, and overlap the outer packs slightly.
    rise, cheek = int(h * 0.11), int(h * 0.13)
    for side in (0, 1):
        pts = []
        for k in range(21):                                   # concave curve
            u = k / 20
            xx = t + cheek * u
            yy = lip_top - rise * (1 - u) ** 1.6
            pts.append((xx, yy))
        poly = [(0, lip_top - rise)] + pts + [(t + cheek, lip_top + 2), (0, lip_top + 2)]
        if side == 1:
            poly = [(tray_w - px, py) for px, py in poly]
        poly = [(x0 + px, py) for px, py in poly]
        m = Image.new("L", (W, H), 0)
        ImageDraw.Draw(m).polygon(poly, fill=255)
        tex = _kraft((W, H), KRAFT_DARK if side else (184, 150, 106), 5 + side)
        canvas.paste(tex, (0, 0), m)
        ImageDraw.Draw(canvas).line(poly[:22], fill=(222, 196, 156), width=3)

    return canvas, counts


# ------------------------------------------------------------ surprise ----

_BOX = None
# Front rim of the box opening in the box image's own pixels (1024 x 600):
# everything of the box below this line is in front of the products.
RIM = [(0, 257), (48, 260), (688, 300), (1023, 180), (1024, 180)]
RIM_Y = 285          # typical rim height, used for placement
OPENING = (205, 860)  # x range of the opening in box pixels
BOX_SCALE = 1.42


def _box_layers():
    global _BOX
    if _BOX is None:
        box = Image.open(os.path.join(HERE, "tgtg_box.png")).convert("RGBA")
        front = Image.new("L", box.size, 0)
        ImageDraw.Draw(front).polygon(
            RIM + [(box.width, box.height), (0, box.height)], fill=255)
        front = ImageChops.multiply(front, box.split()[-1])
        front_layer = box.copy()
        front_layer.putalpha(front)
        _BOX = (box, front_layer)
    return _BOX


def compose_surprise(cutouts):
    """Products dropping into the TGTG box. The box image is never altered."""
    box, front = _box_layers()
    s = BOX_SCALE
    bw, bh = round(box.width * s), round(box.height * s)
    box_s = box.resize((bw, bh), Image.LANCZOS)
    front_s = front.resize((bw, bh), Image.LANCZOS)
    bx = (W - bw) // 2 + round((box.width / 2 - 516) * s)   # centre the visible box
    by = H - bh                                              # bleeds off the bottom
    rim_y = by + round(RIM_Y * s)                            # ~65% of the canvas

    canvas = Image.new("RGB", (W, H), BG)
    canvas.paste(box_s, (bx, by), box_s)

    n = len(cutouts)
    ox0, ox1 = bx + OPENING[0] * s, bx + OPENING[1] * s
    span = ox1 - ox0
    tilt = [-11, 8, -5, 13, -8, 4]
    dip = [0.16, -0.10, 0.22, 0.06, 0.12, -0.12]           # share of h below the rim

    ratios = [c.width / c.height for c in cutouts]
    h = int(min(H * 0.44, (span * 1.25) / max(sum(ratios), 0.1),
                (rim_y - H * 0.06) / (1 - min(dip[:n]) + 0.15)))

    order = list(range(n))
    centres = [ox0 + span * (i + 0.5) / n for i in range(n)]
    # draw outer products first so central ones overlap them slightly
    order.sort(key=lambda i: -abs(i - (n - 1) / 2))
    for i in order:
        c = cutouts[i]
        im = c.resize((max(1, round(c.width * h / c.height)), h), Image.LANCZOS)
        im = im.rotate(tilt[i % len(tilt)], resample=Image.BICUBIC, expand=True)
        bottom = rim_y + h * dip[i % len(dip)]
        x = int(centres[i] - im.width / 2)
        y = int(bottom - im.height)
        # soft shadow to separate overlapping packs
        sh = im.split()[-1].filter(ImageFilter.GaussianBlur(10)).point(lambda v: int(v * 0.22))
        canvas.paste((40, 40, 40), (x + 8, y + 14, x + 8 + im.width, y + 14 + im.height), sh)
        canvas.paste(im, (x, y), im)

    canvas.paste(front_s, (bx, by), front_s)
    return canvas
