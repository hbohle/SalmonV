import os
import io
import base64
import requests
from flask import Flask, render_template, request
from PIL import Image
import numpy as np

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "change-me")

RF_MODEL = "rayosx/1"
RF_API_KEY = os.getenv("RF_API_KEY", "TqEZgC3d5X2gJwutY4MG")

@app.route('/', methods=['GET', 'POST'])
def index():
    annotated = None
    areas = {}
    threshold = 0.5
    if request.method == 'POST':
        threshold = float(request.form.get('threshold', 0.5))
        file = request.files.get('image')
        if file:
            img_bytes = file.read()
            orig_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            # Determine API confidence param (0-100)
            effective_threshold = threshold * 100 if threshold <= 1 else threshold
            resp = requests.post(
                f"https://detect.roboflow.com/{RF_MODEL}",
                params={"api_key": RF_API_KEY, "format": "json", "confidence": effective_threshold},
                files={"file": img_bytes}
            )
            resp.raise_for_status()
            preds = resp.json().get("predictions", [])
            # Prepare overlay
            overlay = Image.new("RGBA", orig_img.size)
            for p in preds:
                cls = p.get("class", "unknown")
                conf = p.get("confidence", 0)
                if conf < effective_threshold:
                    continue
                mask_b64 = p.get("mask", "")
                if not mask_b64:
                    continue
                # Strip header if present
                if mask_b64.startswith("data"):
                    mask_b64 = mask_b64.split(",", 1)[1]
                mask = Image.open(io.BytesIO(base64.b64decode(mask_b64))).convert("L")
                mask_np = np.array(mask)
                area_px = int((mask_np > 0).sum())
                areas[cls] = areas.get(cls, 0) + area_px
                # Create colored overlay for this class
                alpha = Image.fromarray((mask_np > 0).astype(np.uint8) * 128)
                color = (255, 0, 0) if cls != "Vejiga_natatoria" else (0, 255, 0)
                mask_color = Image.new("RGBA", orig_img.size, color + (0,))
                mask_color.putalpha(alpha)
                overlay = Image.alpha_composite(overlay, mask_color)
            # Composite annotated image
            annotated_img = Image.alpha_composite(orig_img.convert("RGBA"), overlay)
            buf = io.BytesIO()
            annotated_img.save(buf, format="PNG")
            annotated = base64.b64encode(buf.getvalue()).decode()
    return render_template("index.html", annotated=annotated, areas=areas or None, threshold=threshold)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))