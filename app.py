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
    areas = None
    threshold = 0.5
    if request.method == 'POST':
        threshold = float(request.form.get('threshold', 0.5))
        file = request.files.get('image')
        if file:
            img_bytes = file.read()
            # Call Roboflow detect endpoint
            resp = requests.post(
                f"https://detect.roboflow.com/{RF_MODEL}",
                params={"api_key": RF_API_KEY, "format": "json"},
                files={"file": img_bytes}
            )
            resp.raise_for_status()
            preds = resp.json().get("predictions", [])
            areas = {}
            for p in preds:
                cls = p.get("class", "unknown")
                conf = p.get("confidence", 0)
                if conf < threshold:
                    continue
                mask_b64 = p.get("mask", {}).get("mask", "")
                if not mask_b64:
                    continue
                header, b64data = mask_b64.split(",", 1) if "," in mask_b64 else ("", mask_b64)
                mask = Image.open(io.BytesIO(base64.b64decode(b64data))).convert("L")
                mask_np = np.array(mask)
                count = int((mask_np > 0).sum())
                areas[cls] = areas.get(cls, 0) + count
    return render_template('index.html', areas=areas, threshold=threshold)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))