import os
import io
import base64
import requests
from flask import Flask, render_template, request
from PIL import Image
import numpy as np

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "change-me")

# Configuración del modelo en Roboflow
RF_MODEL   = "rayosx/1"
RF_API_KEY = os.getenv("RF_API_KEY", "TqEZgC3d5X2gJwutY4MG")

@app.route('/', methods=['GET', 'POST'])
def index():
    # Inicializamos variables para la plantilla
    annotated = None       # imagen anotada en base64
    areas     = {}         # diccionario clase → área en px
    threshold = 0.5        # umbral por defecto (0.0–1.0)

    if request.method == 'POST':
        # Leemos el nuevo umbral del formulario
        threshold = float(request.form.get('threshold', 0.5))

        # Procesamos la imagen subida
        file = request.files.get('image')
        if file:
            img_bytes = file.read()
            orig_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            # Convertimos el umbral (0–1) a la escala 0–100 de la API
            api_thresh = threshold * 100

            # Llamada al endpoint de detección de Roboflow
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

            # Creamos un overlay transparente
            overlay = Image.new("RGBA", orig_img.size, (0, 0, 0, 0))

            # Para cada predicción, si supera el umbral, procesamos su máscara
            for p in preds:
                cls  = p.get("class", "unknown")
                conf = p.get("confidence", 0)
                if conf < api_thresh:
                    continue

                mask_b64 = p.get("mask", "")
                # Si viene con header data:, lo quitamos
                if mask_b64.startswith("data"):
                    mask_b64 = mask_b64.split(",", 1)[1]

                # Decodificamos la máscara y contamos sus píxeles
                mask = Image.open(io.BytesIO(base64.b64decode(mask_b64))).convert("L")
                mask_np = np.array(mask)
                area_px = int((mask_np > 0).sum())
                areas[cls] = areas.get(cls, 0) + area_px

                # Generamos un overlay coloreado para esta clase
                alpha      = Image.fromarray((mask_np > 0).astype(np.uint8) * 128)
                color      = (255, 0, 0)  # rojo por defecto
                mask_color = Image.new("RGBA", orig_img.size, color + (0,))
                mask_color.putalpha(alpha)
                overlay = Image.alpha_composite(overlay, mask_color)

            # Superponemos el overlay sobre la imagen original
            annotated_img = Image.alpha_composite(orig_img.convert("RGBA"), overlay)

            # Convertimos a base64 para incrustar en HTML
            buf = io.BytesIO()
            annotated_img.save(buf, format="PNG")
            annotated = base64.b64encode(buf.getvalue()).decode()

    # Renderizamos la plantilla siempre (GET y POST)
    return render_template(
        "index.html",
        annotated=annotated,  # None o cadena base64
        areas=areas,          # {} o dict con áreas
        threshold=threshold   # valor actual del slider
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
