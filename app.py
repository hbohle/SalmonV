import os
import io
import base64
import requests
from flask import Flask, render_template, request
from PIL import Image, ImageDraw
import numpy as np

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "change-me")

# Tu modelo en Roboflow y API key
RF_MODEL   = "rayosx/1"
RF_API_KEY = os.getenv("RF_API_KEY", "TqEZgC3d5X2gJwutY4MG")

@app.route('/', methods=['GET', 'POST'])
def index():
    annotated = None       # Base64 de la imagen anotada
    areas     = {}         # Diccionario clase → área en px
    threshold = 0.5        # Umbral (0.0–1.0)

    if request.method == 'POST':
        threshold = float(request.form.get('threshold', 0.5))
        file = request.files.get('image')
        if file:
            img_bytes = file.read()
            orig_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            api_thresh = threshold * 100

            # Llamada a la API de Roboflow
            resp = requests.post(
                f"https://detect.roboflow.com/{RF_MODEL}",
                params={
                    "api_key": RF_API_KEY,
                    "format": "json",
                    "confidence": api_thresh
                },
                files={"file": img_bytes}
            )
            resp.raise_for_status()
            preds = resp.json().get("predictions", [])

            # Prepara overlay transparente del mismo tamaño
            overlay = Image.new("RGBA", orig_img.size, (0,0,0,0))
            draw_overlay = ImageDraw.Draw(overlay)

            for p in preds:
                cls  = p.get("class", "unknown")
                conf = p.get("confidence", 0)
                if conf < api_thresh:
                    continue

                # 1) Si viene máscara en base64
                mask_img = None
                if "mask" in p and p["mask"]:
                    mask_b64 = p["mask"]
                    if mask_b64.startswith("data:"):
                        mask_b64 = mask_b64.split(",",1)[1]
                    mask_img = Image.open(io.BytesIO(base64.b64decode(mask_b64))).convert("L")

                # 2) Si no, construye máscara a partir de los puntos del polígono
                elif "points" in p and p["points"]:
                    mask_img = Image.new("L", orig_img.size, 0)
                    draw_mask = ImageDraw.Draw(mask_img)
                    poly = [(point["x"], point["y"]) for point in p["points"]]
                    draw_mask.polygon(poly, outline=1, fill=1)

                if mask_img is None:
                    continue

                mask_np = np.array(mask_img)
                area_px = int((mask_np > 0).sum())
                areas[cls] = areas.get(cls, 0) + area_px

                # Dibuja el polígono coloreado en el overlay
                # Usamos el mismo poly si existe, o extraemos contorno de la máscara
                if "points" in p and p["points"]:
                    poly = [(point["x"], point["y"]) for point in p["points"]]
                    color = (255,0,0,120)  # RGBA: rojo semi-transparente
                    draw_overlay.polygon(poly, fill=color)
                else:
                    # si viene máscara, convertimos a polígono aproximado o pintamos píxel a píxel
                    # pero para simplificar usamos mask_img como alpha
                    rgba_mask = mask_img.point(lambda v: 120 if v>0 else 0)
                    color_img = Image.new("RGBA", orig_img.size, (255,0,0,0))
                    color_img.putalpha(rgba_mask)
                    overlay = Image.alpha_composite(overlay, color_img)

            # Superponemos overlay sobre la original
            annotated_img = Image.alpha_composite(orig_img.convert("RGBA"), overlay)

            # Convertimos a base64 para incrustar en HTML
            buf = io.BytesIO()
            annotated_img.save(buf, format="PNG")
            annotated = base64.b64encode(buf.getvalue()).decode()

    # Renderizamos siempre (GET y POST)
    return render_template(
        "index.html",
        annotated=annotated,
        areas=areas,
        threshold=threshold
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
