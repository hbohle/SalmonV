import os
import io
import base64
import requests
from flask import Flask, render_template, request, flash
from PIL import Image, ImageDraw
import numpy as np

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "change-me")

# --- CONFIGURACIÓN DE ROBOFLOW ---
RF_MODEL   = "rayosx/1"
RF_API_KEY = os.getenv("RF_API_KEY", "TqEZgC3d5X2gJwutY4MG")

# Mapa de colores RGBA para cada clase (último valor = transparencia)
CLASS_COLORS = {
    "Estomago":         (255,   0,   0, 120),  # rojo
    "Pez":              (  0, 255,   0, 120),  # verde
    "Vejiga_natatoria": (  0,   0, 255, 120),  # azul
    "Columna":          (255, 255,   0, 120),  # amarillo
}

@app.route("/", methods=["GET", "POST"])
def index():
    annotated = None       # Base64 de la imagen anotada
    areas     = {}         # dict: clase → área en px
    threshold = 0.23       # umbral por defecto (0–1)

    if request.method == "POST":
        # Leemos el umbral enviado por el slider (escala 0–1)
        threshold = float(request.form.get("threshold", threshold))

        # Recogemos la imagen subida
        file = request.files.get("image")
        if not file:
            flash("Debes subir una imagen válida.", "danger")
            return render_template("index.html",
                                   annotated=None,
                                   areas={},
                                   threshold=threshold)

        img_bytes = file.read()
        orig_img  = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Roboflow espera confidence en 0–100
        api_conf = int(threshold * 100)

        # Llamada a la API de segmentación
        resp = requests.post(
            f"https://detect.roboflow.com/{RF_MODEL}",
            params={
                "api_key": RF_API_KEY,
                "format": "json",
                "confidence": api_conf
            },
            files={"file": (file.filename, img_bytes, file.mimetype)}
        )
        resp.raise_for_status()
        preds = resp.json().get("predictions", [])

        # Preparamos un overlay RGBA transparente
        overlay = Image.new("RGBA", orig_img.size, (0, 0, 0, 0))

        # Iteramos sobre cada predicción
        for p in preds:
            cls  = p.get("class", "unknown")
            conf = p.get("confidence", 0.0)

            # Filtramos por umbral (0–1)
            if conf < threshold:
                continue

            #-------------------------------
            # 1) Extraemos máscara embebida
            #-------------------------------
            mask_data = p.get("mask", {}).get("mask")
            mask_img  = None

            if mask_data:
                if mask_data.startswith("data:"):
                    mask_data = mask_data.split(",", 1)[1]
                mask_img = Image.open(io.BytesIO(base64.b64decode(mask_data))).convert("L")

            #---------------------------------------------------
            # 2) Si no hay máscara, la generamos desde "points"
            #---------------------------------------------------
            elif "points" in p and p["points"]:
                mask_img = Image.new("L", orig_img.size, 0)
                draw     = ImageDraw.Draw(mask_img)
                poly     = [(pt["x"], pt["y"]) for pt in p["points"]]
                draw.polygon(poly, fill=255)

            # Si no pudimos montar una máscara, saltamos este predicado
            if mask_img is None:
                continue

            #------------------------
            # 3) Calculamos área px²
            #------------------------
            mask_np = np.array(mask_img)
            area_px = int((mask_np > 0).sum())
            areas[cls] = areas.get(cls, 0) + area_px

            #------------------------------------------------
            # 4) Construimos overlay con color por clase
            #------------------------------------------------
            r, g, b, a = CLASS_COLORS.get(cls, (255, 0, 0, 120))
            # creamos un canal alpha donde mask>0
            alpha_mask = mask_img.point(lambda v: a if v > 0 else 0)
            # imagen del color
            color_img  = Image.new("RGBA", orig_img.size, (r, g, b, 0))
            color_img.putalpha(alpha_mask)
            overlay = Image.alpha_composite(overlay, color_img)

        #-------------------------------------
        # 5) Superponemos overlay sobre original
        #-------------------------------------
        annotated_img = Image.alpha_composite(orig_img.convert("RGBA"), overlay)
        buf = io.BytesIO()
        annotated_img.save(buf, format="PNG")
        annotated = base64.b64encode(buf.getvalue()).decode()

    # Renderizamos siempre (GET y POST)
    return render_template(
        "index.html",
        annotated=annotated,  # None o cadena base64
        areas=areas,          # dict con áreas
        threshold=threshold   # valor actual del slider
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
