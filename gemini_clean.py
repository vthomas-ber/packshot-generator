"""
Gemini clean-up step: turns a messy product photo (shelves, hands, coloured
backgrounds, background graphics) into the same product on plain white, so the
deterministic layout step can cut it out cleanly.
"""
import io
import os

from PIL import Image

MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")

PROMPT = """You are a product retoucher preparing a packshot.
Return ONE image: the single main retail product from this photo, isolated on a
completely plain, flat, pure white (#FFFFFF) background.

Remove everything that is not the product itself: hands, fingers, shelves,
store fixtures, tables, price tags, other products, and any background shapes,
circles, splashes or graphics that sit behind the pack.

Keep the product exactly as it is: same packaging, colours, logos, text,
illustrations, shape and proportions. Do not redraw, restyle, translate or
"improve" anything printed on the pack. If part of the pack is hidden by a
hand, restore only that hidden part so it matches the visible design.

Show the whole product, front-facing, centred, filling most of the frame with a
small white margin on every side. No shadow, no reflection, no surface, no text
added. Output the image only."""

_client = None


def _get_client():
    global _client
    if _client is None:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        from google import genai
        _client = genai.Client(api_key=key)
    return _client


def clean(im):
    """Return a PIL image of the product on white, cleaned up by Gemini."""
    from google.genai import types

    resp = _get_client().models.generate_content(
        model=MODEL,
        contents=[PROMPT, im],
        config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
    )
    for cand in resp.candidates or []:
        for part in (cand.content.parts if cand.content else []) or []:
            if getattr(part, "inline_data", None) and part.inline_data.data:
                return Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
    raise RuntimeError("Gemini returned no image")
