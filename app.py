"""
Packshot Generator web app.

Users log in with a shared password, upload product photos, and download a
#FAFAFA 16:9 packshot. Photos on plain backgrounds are cut out directly (free,
untouched); messy photos are cleaned up by Gemini first.

Environment variables (set in Render, never in code):
    GEMINI_API_KEY      Gemini API key
    APP_PASSWORD        shared password for users
    SECRET_KEY          random string for login sessions
    GEMINI_IMAGE_MODEL  optional, defaults to gemini-3-pro-image
"""
import base64
import hmac
import io
import os
import re
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, jsonify, redirect, render_template, request, session, url_for

import gemini_clean
import packshot_core as core

MAX_FILES = 8
ALLOWED = {".png", ".jpg", ".jpeg", ".webp"}

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY") or os.urandom(32)
app.config["MAX_CONTENT_LENGTH"] = 80 * 1024 * 1024
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("RENDER") is not None


def logged_in():
    return session.get("ok") is True


@app.get("/")
def index():
    if not logged_in():
        return render_template("login.html", error=None)
    return render_template("index.html", model=gemini_clean.MODEL)


@app.post("/login")
def login():
    expected = os.environ.get("APP_PASSWORD", "")
    given = request.form.get("password", "")
    if expected and hmac.compare_digest(given.encode(), expected.encode()):
        session["ok"] = True
        session.permanent = True
        return redirect(url_for("index"))
    error = "APP_PASSWORD is not configured on the server." if not expected \
        else "Wrong password."
    return render_template("login.html", error=error), 401


@app.post("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


def process_one(args):
    """Return (cutout, note) for one uploaded image."""
    name, data, force_gemini = args
    with core.cpu_lock:
        im = core.load_image(io.BytesIO(data))
        del data
        if not force_gemini and core.plain_background(im):
            return core.cut_out(im), f"{name}: already on a clean background, used as is"
    try:
        cleaned = gemini_clean.clean(im)
        return core.cut_out(cleaned), f"{name}: background cleaned up by Gemini"
    except Exception as e:  # fall back so one bad image doesn't block the rest
        return core.cut_out(im), f"{name}: Gemini clean-up failed ({e}); original used, check result"


@app.post("/generate")
def generate():
    if not logged_in():
        return jsonify(error="Please log in again."), 401

    files = [f for f in request.files.getlist("images") if f and f.filename]
    if not files:
        return jsonify(error="Add at least one image."), 400
    if len(files) > MAX_FILES:
        return jsonify(error=f"Maximum {MAX_FILES} images per packshot."), 400
    for f in files:
        if os.path.splitext(f.filename.lower())[1] not in ALLOWED:
            return jsonify(error=f"{f.filename}: use PNG, JPG or WEBP."), 400

    force = request.form.get("force_gemini") == "1"
    jobs = [(f.filename, f.read(), force) for f in files]
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(process_one, jobs))   # keeps upload order
    except Exception as e:
        return jsonify(error=f"Could not process the images: {e}"), 500

    del jobs
    with core.cpu_lock:
        packshot = core.compose([c for c, _ in results])
        buf = io.BytesIO()
        packshot.save(buf, "PNG")
        del packshot

    item = re.sub(r"[^A-Za-z0-9_-]+", "", request.form.get("item_id", ""))[:60]
    return jsonify(
        image="data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
        filename=f"{item or 'packshot'}_packshot.png",
        notes=[n for _, n in results],
    )


@app.get("/health")
def health():
    return "ok"


if __name__ == "__main__":
    app.run(debug=True, port=5000)
