# Packshot generator

Internal web tool that turns product photos into a 16:9 packshot on a flat
#FAFAFA background, with products centred in the top 65% of the frame and
identical soft shadows.

## How it works

1. **Per photo.** If the photo is already on a plain white or light-grey
   background, the product is cut out directly. This step is free and leaves the
   product untouched. Anything messier (shelves, hands, coloured backgrounds,
   background graphics) is first sent to Gemini, which returns the same product
   on pure white.
2. **Layout.** The cut-out products are scaled to the same height and placed
   left to right on the canvas in the order the user chose. Code handles the
   layout, so the background colour, spacing and shadows are the same every time.

Users can tick "Clean every photo with Gemini" to send all photos through
Gemini, for example when a "clean" photo still has a slight off-white tint.

## Deploy on Render

1. In Render, choose **New > Blueprint** and select this repository. Render
   reads `render.yaml` and creates the web service.
2. When asked, fill in the secret values:
   - `GEMINI_API_KEY`: your Gemini API key
   - `APP_PASSWORD`: the shared password users will type to log in
3. Click **Apply**. The first build takes a few minutes. When it is live, open
   the `.onrender.com` URL and log in.

`SECRET_KEY` is generated automatically. `GEMINI_IMAGE_MODEL` defaults to
`gemini-2.5-flash-image`; change it in the Render dashboard
(**Environment**) to switch models without touching the code.

The blueprint uses the **Starter** plan so the app stays awake. To use the free
plan instead, change `plan: starter` to `plan: free` in `render.yaml`. The free
plan sleeps when idle, so the first visit after a quiet period takes about a
minute.

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=... APP_PASSWORD=...
python app.py            # then open http://localhost:5000
```

## Limits and safeguards

- Up to 8 photos per packshot (PNG, JPG or WEBP). Large photos are downscaled
  to 2000 px on the longest side before processing.
- If Gemini fails on a photo, the original is used and the result page flags
  it, so one bad photo never blocks the rest.
- Gemini can still alter small print on a pack. Check packshots made from
  messy photos before publishing them.
- Restrict the API key to the Gemini API in Google Cloud and set a budget alert
  on the billing project.
