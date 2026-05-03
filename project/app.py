"""
app.py — Context-Aware Privacy Filtering System
Renders a side-by-side results page (original vs. processed) in the browser.
"""

import os
import base64
import mimetypes
import uuid

from flask import Flask, render_template, request, abort, send_file
from detector import process_image

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}

for d in (UPLOADS_DIR, OUTPUTS_DIR, MODELS_DIR):
    os.makedirs(d, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.urandom(24)


def _allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _to_data_url(path, ext):
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    if "image" not in request.files:
        abort(400, description="No file part in the request.")
    file = request.files["image"]
    if file.filename == "":
        abort(400, description="No file selected.")
    if not _allowed(file.filename):
        abort(400, description="Unsupported type. Please upload JPEG, PNG, BMP, or WEBP.")

    ext = file.filename.rsplit(".", 1)[1].lower()
    upload_path = os.path.join(UPLOADS_DIR, f"{uuid.uuid4().hex}.{ext}")
    file.save(upload_path)

    try:
        result = process_image(upload_path, OUTPUTS_DIR, MODELS_DIR)
    except Exception as exc:
        try: os.remove(upload_path)
        except OSError: pass
        abort(500, description=f"Processing failed: {exc}")

    output_path = result["output_path"]

    # Encode both images for embedding — then delete temp files
    original_data  = _to_data_url(upload_path, ext)
    processed_data = _to_data_url(output_path, ext)

    try: os.remove(upload_path)
    except OSError: pass
    try: os.remove(output_path)
    except OSError: pass

    return render_template(
        "result.html",
        original=original_data,
        processed=processed_data,
        faces=result["faces_found"],
        plates=result["plates_found"],
        screens=result["screens_found"],
        ext=ext,
    )


@app.errorhandler(400)
def bad_request(e):
    return render_template("index.html", error=e.description), 400

@app.errorhandler(500)
def server_error(e):
    return render_template("index.html", error=e.description), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
