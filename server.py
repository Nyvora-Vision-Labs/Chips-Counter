"""
server.py — Flask backend for the Chip Rack Counter UI
=======================================================
Runs detection.py pipeline on an uploaded image and returns JSON results.

Usage:
    python3 server.py
    # then open http://localhost:5050 in your browser
"""

import os
import sys
import uuid
import tempfile
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, send_file

# ── Import the detection pipeline directly ───────────────────────────────────
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from detection import detect_chips, DEFAULT_REFS_DIR, CLIP_THRESHOLD

app = Flask(__name__, static_folder=str(HERE))

UPLOAD_DIR = HERE / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED = {".png", ".jpg", ".jpeg", ".webp"}

# uid → Path of the annotated image
_results: dict[str, Path] = {}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file(str(HERE / "index.html"))


@app.route("/detect", methods=["POST"])
def detect():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        f = request.files["image"]
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED:
            return jsonify({"error": f"Unsupported file type: {ext}"}), 400

        # Save uploaded file to a temp path
        uid = uuid.uuid4().hex[:8]
        save_path = UPLOAD_DIR / f"rack_{uid}{ext}"
        f.save(str(save_path))

        threshold = float(request.form.get("threshold", CLIP_THRESHOLD))

        counts = detect_chips(
            rack_path=save_path,
            refs_dir=DEFAULT_REFS_DIR,
            threshold=threshold,
            save_crops=False,
        )

        annotated_path = save_path.with_name(save_path.stem + "_detected.jpg")
        _results[uid] = annotated_path

        return jsonify({
            "ok": True,
            "counts": counts,
            "total": sum(counts.values()),
            "annotated_url": f"/result/{uid}",
            "threshold": threshold,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/result/<uid>")
def result_image(uid):
    """Serve the annotated output image for a given run uid."""
    path = _results.get(uid)
    if path and path.exists():
        return send_file(str(path), mimetype="image/jpeg")
    return "Not found", 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print("\n🚀  Chip Rack Counter server starting …")
    print(f"    Upload dir : {UPLOAD_DIR}")
    print(f"    Open       : http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
