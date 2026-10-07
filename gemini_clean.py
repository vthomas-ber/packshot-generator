"""
Gemini clean-up step: turns a messy product photo (shelves, hands, coloured
backgrounds, background graphics) into the same product on plain white, so the
deterministic layout step can cut it out cleanly.
"""
import io
import os
import threading
import time

from PIL import Image

MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image")

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

# One shared client for the whole app. Creating it under a lock matters: if two
# photos arrive at once, each thread would otherwise build its own client, and
# the one that gets discarded closes its connection while still in use
# ("Cannot send a request, as the client has been closed").
_client = None
_client_lock = threading.Lock()


def _get_client(fresh=False):
    global _client
    with _client_lock:
        if _client is None or fresh:
            key = os.environ.get("GEMINI_API_KEY")
            if not key:
                raise RuntimeError("GEMINI_API_KEY is not set")
            from google import genai
            _client = genai.Client(api_key=key)
        return _client


def _call(client, im):
    from google.genai import types

    resp = client.models.generate_content(
        model=MODEL,
        contents=[PROMPT, im],
        config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
    )
    for cand in resp.candidates or []:
        for part in (cand.content.parts if cand.content else []) or []:
            if getattr(part, "inline_data", None) and part.inline_data.data:
                return Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
    raise RuntimeError("Gemini returned no image")


def clean(im, attempts=3):
    """Return a PIL image of the product on white, cleaned up by Gemini.

    Retries on temporary failures (closed connection, rate limits, timeouts,
    no image returned), waiting a little longer each time.
    """
    last = None
    for i in range(attempts):
        client = _get_client(fresh=last is not None and "closed" in str(last))
        try:
            return _call(client, im)          # `client` stays referenced during the call
        except Exception as e:
            last = e
            if i < attempts - 1:
                time.sleep(2 * (i + 1))
    raise last
