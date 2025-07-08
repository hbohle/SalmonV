import os
import io
import base64
import requests
from flask import Flask, render_template, request, flash
from PIL import Image
import numpy as np

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "change-me")

# --- CONFIGURACIÓN ROBOFLOW ---
RF_MODEL   = "rayosx/1"
RF_API_KEY = os.getenv("RF_API_KEY", "TqEZgC3d5X2gJwutY4MG")

@app.route("/", methods=["GET", "POST"])
def index():
    annotated = None       # cadena base64 de la imagen anotada
    areas     = {}         # dict: clase → área (px)
    threshold = 0.23       # umbral por defecto (0–1)

    if request.method == "POST":
        # 1) Leemos el umbral enviado por el slider (escala 0–1)
        threshold = float(request.form.get("threshold", threshold))
        # 2) Recogemos la imagen subida
        file = request.files.get("image")
        if not file:
            flash("Debes subir una imagen válida.", "danger")
            return render_template("index.html", annotated=None, areas={}, threshold=threshold)

        img_bytes = file.read()
        orig_img  = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        # La API de Roboflow pide confidence en 0–100
        api_conf = int(threshold * 100)

        # 3) Llamada HTTP a la API de segmentación
        resp = requests.post(
            f"https://detect.roboflow.com/{RF_MODEL}",
            params={
                "api_key": RF_API_KEY,
                "format": "json",
                "confidence": api_conf
            },
            # Aquí pasamos name, contenido y tipo MIME
            files={"file": (file.filename, img_bytes, file.mimetype)}
        )
        resp.raise_for_status()
        preds = resp.json().get("predictions", [])

        # 4) Preparamos un overlay RGBA transparente
        overlay = Image.new("RGBA", orig_img.size, (0,0,0,0))

        # 5) Iteramos sobre cada predicción
        for p in preds:
            cls  = p.get("class", "unknown")
            conf = p.get("confidence", 0.0)
            # filtramos por umbral en 0–1
            if conf < threshold:
                continue

            # a) intentamos extraer la máscara embebida
            mask_data = p.get("mask", {}).get("mask")
            mask_img  = None

            if mask_data:
                # Quítale el header `data:image/...;base64,`
                if mask_data.startswith("data:"):
                    mask_data = mask_data.split(",",1)[1]
                mask_img = Image.open(io.BytesIO(base64.b64decode(mask_data))).convert("L")

            # b) si no hay máscara, la generamos a partir de los puntos
            elif "points" in p and p["points"]:
                mask_img = Image.new("L", orig_img.size, 0)
                draw = Image.Draw.Draw(mask_img)
                poly = [(pt["x"], pt["y"]) for pt in p["points"]]
                draw.polygon(poly, fill=255)

            # si no pudo generarla, saltamos
            if mask_img is None:
                continue

            # 6) calculamos el área
            mask_np = np.array(mask_img)
            area_px = int((mask_np > 0).sum())
            areas[cls] = areas.get(cls, 0) + area_px

            # 7) construimos un overlay coloreado (rojo semitransparente)
            alpha_mask = mask_img.point(lambda v: 120 if v>0 else 0)
            color_img  = Image.new("RGBA", orig_img.size, (255,0,0,0))
            color_img.putalpha(alpha_mask)
            overlay = Image.alpha_composite(overlay, color_img)

        # 8) superponemos overlay sobre la original
        annotated_img = Image.alpha_composite(orig_img.convert("RGBA"), overlay)
        buf = io.BytesIO()
        annotated_img.save(buf, format="PNG")
        annotated = base64.b64encode(buf.getvalue()).decode()

    # Renderizamos en GET y POST (si no hubo POST, `annotated` será None y `areas` {})
    return render_template(
        "index.html",
        annotated=annotated,
        areas=areas,
        threshold=threshold
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
