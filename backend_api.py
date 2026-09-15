from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
from openai import OpenAI

import shutil
import uuid
import os
import json
import re
import base64
import numpy as np


# =========================================================
# CARGAR VARIABLES DE ENTORNO
# =========================================================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# =========================================================
# RUTAS / CARPETAS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"

UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)


# =========================================================
# APP FASTAPI
# =========================================================
app = FastAPI(
    title="Backend Viñedo TerraCore",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")


# =========================================================
# UTILIDADES
# =========================================================
def image_to_data_url(image_path: str) -> str:
    ext = Path(image_path).suffix.lower().replace(".", "")
    if ext == "jpg":
        ext = "jpeg"
    mime = f"image/{ext if ext else 'jpeg'}"

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime};base64,{b64}"


def extraer_json_desde_texto(texto: str) -> dict:
    if not texto:
        raise ValueError("OpenAI no devolvió texto.")

    texto = texto.strip()

    # Quitar bloque ```json ... ```
    texto = re.sub(r"^```json", "", texto, flags=re.IGNORECASE).strip()
    texto = re.sub(r"^```", "", texto).strip()
    texto = re.sub(r"```$", "", texto).strip()

    inicio = texto.find("{")
    fin = texto.rfind("}")

    if inicio == -1 or fin == -1:
        raise ValueError("No se encontró un JSON válido en la respuesta de OpenAI.")

    json_texto = texto[inicio:fin + 1]
    return json.loads(json_texto)


def analizar_imagen_con_openai(ruta_imagen: str) -> dict:
    if not client:
        raise RuntimeError(
            "No existe OPENAI_API_KEY en .env. "
            "Agrega tu clave real en el archivo .env"
        )

    data_url = image_to_data_url(ruta_imagen)

    prompt = """
Analiza esta imagen de viñedo y responde SOLO JSON válido.

Objetivo:
- Detectar si la imagen es un viñedo.
- Decidir si conviene analizar surcos.
- Estimar orientación principal de los surcos.
- Estimar cuántos surcos hay.
- Detectar zonas a excluir como caminos, techos, construcciones y árboles.

IMPORTANTE:
1. Responde SOLO JSON.
2. No agregues explicación.
3. Usa esta estructura exacta:

{
  "es_vinedo": true,
  "analizar_surcos": true,
  "orientacion_principal_grados": 90,
  "surcos_estimados": 80,
  "confianza_visual": 0.93,
  "hay_camino": true,
  "hay_techo": false,
  "hay_arboles": false,
  "hay_construccion": false,
  "resumen": "Texto corto",
  "zonas_excluir": [
    {
      "tipo": "camino",
      "puntos": [[0,0],[1000,0],[1000,50],[0,50]]
    }
  ],
  "detalle_usado": "low",
  "ancho_original": 0,
  "alto_original": 0
}

Reglas:
- "orientacion_principal_grados" usa 90 si los surcos son principalmente verticales en la imagen.
- usa 0 si son principalmente horizontales.
- "surcos_estimados" debe ser un entero razonable.
- "confianza_visual" entre 0 y 1.
- "zonas_excluir" usa coordenadas NORMALIZADAS de 0 a 1000.
- Si no hay zonas a excluir, devuelve [].
"""

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "Eres un analista visual de viñedos. Respondes únicamente JSON válido."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url,
                                "detail": "low"
                            }
                        }
                    ]
                }
            ]
        )

        contenido = resp.choices[0].message.content
        data = extraer_json_desde_texto(contenido)

        return data

    except Exception as e:
        raise RuntimeError(f"OpenAI no pudo analizar la imagen. {str(e)}")


# =========================================================
# PROCESAMIENTO DE SURCOS
# =========================================================
def asegurar_impar(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def suavizar_vector(v, ventana=9):
    v = np.asarray(v, dtype=np.float32)
    ventana = max(3, asegurar_impar(int(ventana)))
    kernel = np.ones(ventana, dtype=np.float32) / ventana
    return np.convolve(v, kernel, mode="same")


def exceso_verde(arr_rgb: np.ndarray) -> np.ndarray:
    r = arr_rgb[:, :, 0].astype(np.float32)
    g = arr_rgb[:, :, 1].astype(np.float32)
    b = arr_rgb[:, :, 2].astype(np.float32)

    # índice simple para vegetación
    exg = (2.0 * g) - r - b
    exg = np.clip(exg, 0, None)
    return exg


def escalar_puntos_zona(points, width, height):
    if not points:
        return []

    max_x = max(p[0] for p in points)
    max_y = max(p[1] for p in points)

    # Si parecen venir normalizados 0..1000
    if max_x <= 1000 and max_y <= 1000:
        sx = width / 1000.0
        sy = height / 1000.0
        return [(int(x * sx), int(y * sy)) for x, y in points]

    # Si ya vienen en pixeles
    return [(int(x), int(y)) for x, y in points]


def crear_mascara_permitida(width, height, zonas_excluir):
    mask_img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(mask_img)

    for zona in (zonas_excluir or []):
        puntos = zona.get("puntos", [])
        puntos_escalados = escalar_puntos_zona(puntos, width, height)

        if len(puntos_escalados) >= 3:
            draw.polygon(puntos_escalados, fill=0)

    mask = np.array(mask_img) > 0
    return mask


def perfil_surcos(arr_rgb, mask_permitida, orientacion_vertical=True):
    exg = exceso_verde(arr_rgb)
    exg = np.where(mask_permitida, exg, np.nan)

    if orientacion_vertical:
        # perfil por columnas
        perfil = np.nanmean(exg, axis=0)
    else:
        # perfil por filas
        perfil = np.nanmean(exg, axis=1)

    perfil = np.nan_to_num(perfil, nan=0.0)
    perfil = suavizar_vector(perfil, ventana=max(7, len(perfil) // 120))
    return perfil


def detectar_centros_surcos(perfil, esperado=0):
    perfil = np.asarray(perfil, dtype=np.float32)

    if perfil.max() > 0:
        perfil_n = perfil / (perfil.max() + 1e-6)
    else:
        perfil_n = perfil

    # umbral base
    umbral = max(0.20, float(np.percentile(perfil_n, 60)) * 0.85)

    activos = perfil_n >= umbral

    segmentos = []
    start = None

    for i, v in enumerate(activos):
        if v and start is None:
            start = i
        elif not v and start is not None:
            end = i - 1
            fuerza = float(np.mean(perfil_n[start:end + 1]))
            if end - start >= 1:
                segmentos.append((start, end, fuerza))
            start = None

    if start is not None:
        end = len(activos) - 1
        fuerza = float(np.mean(perfil_n[start:end + 1]))
        if end - start >= 1:
            segmentos.append((start, end, fuerza))

    centros = [int((s + e) / 2) for s, e, _ in segmentos]
    fuerzas = [f for _, _, f in segmentos]

    # si salen demasiados, conservar los más fuertes
    if esperado and len(centros) > int(esperado * 1.5):
        n_keep = max(esperado, int(esperado * 1.2))
        idxs = sorted(range(len(centros)), key=lambda i: fuerzas[i], reverse=True)[:n_keep]
        idxs = sorted(idxs)
        centros = [centros[i] for i in idxs]

    # unir centros demasiado cercanos
    if esperado > 0:
        gap_min = max(6, int((len(perfil) / esperado) * 0.45))
    else:
        gap_min = max(6, int(len(perfil) / 120))

    unidos = []
    for c in centros:
        if not unidos:
            unidos.append(c)
        else:
            if abs(c - unidos[-1]) < gap_min:
                unidos[-1] = int((unidos[-1] + c) / 2)
            else:
                unidos.append(c)

    return unidos


def detectar_tramos_secos(arr_rgb, mask_permitida, centro, orientacion_vertical=True, banda=4):
    exg = exceso_verde(arr_rgb)
    h, w = exg.shape[:2]

    if orientacion_vertical:
        x1 = max(0, centro - banda)
        x2 = min(w, centro + banda + 1)

        strip = exg[:, x1:x2]
        strip_mask = mask_permitida[:, x1:x2]

        vals = np.where(strip_mask, strip, np.nan)
        serie = np.nanmean(vals, axis=1)
    else:
        y1 = max(0, centro - banda)
        y2 = min(h, centro + banda + 1)

        strip = exg[y1:y2, :]
        strip_mask = mask_permitida[y1:y2, :]

        vals = np.where(strip_mask, strip, np.nan)
        serie = np.nanmean(vals, axis=0)

    serie = np.nan_to_num(serie, nan=0.0)
    serie = suavizar_vector(serie, ventana=max(5, len(serie) // 150))

    validos = serie[serie > 0]
    if len(validos) == 0:
        return []

    umbral_seco = max(float(np.percentile(validos, 28)), float(np.mean(validos) * 0.55))
    secos = serie < umbral_seco

    segmentos = []
    ini = None

    for i, v in enumerate(secos):
        if v and ini is None:
            ini = i
        elif not v and ini is not None:
            fin = i - 1
            if fin - ini >= 18:
                segmentos.append((ini, fin))
            ini = None

    if ini is not None:
        fin = len(secos) - 1
        if fin - ini >= 18:
            segmentos.append((ini, fin))

    return segmentos


def dibujar_surcos_en_imagen(ruta_entrada: str, ruta_salida: str, analisis: dict) -> dict:
    img = Image.open(ruta_entrada).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]

    zonas_excluir = analisis.get("zonas_excluir", [])
    orientacion = int(analisis.get("orientacion_principal_grados", 90))
    esperado = int(analisis.get("surcos_estimados", 0) or 0)

    vertical = 45 <= orientacion <= 135

    mask_permitida = crear_mascara_permitida(w, h, zonas_excluir)
    perfil = perfil_surcos(arr, mask_permitida, orientacion_vertical=vertical)
    centros = detectar_centros_surcos(perfil, esperado=esperado)

    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    lineas = []
    tramos_secos_total = 0

    for idx, centro in enumerate(centros, start=1):
        if vertical:
            # Línea verde base
            draw.line([(centro, 0), (centro, h - 1)], fill=(0, 255, 0), width=3)

            # Número arriba y abajo
            draw.text((centro + 2, 4), str(idx), fill=(255, 255, 0), font=font)
            draw.text((centro + 2, h - 16), str(idx), fill=(255, 255, 0), font=font)

            secos = detectar_tramos_secos(arr, mask_permitida, centro, orientacion_vertical=True, banda=4)

            for y1, y2 in secos:
                draw.line([(centro, y1), (centro, y2)], fill=(255, 0, 0), width=4)
                tramos_secos_total += 1

            lineas.append({"indice": idx, "x": int(centro)})

        else:
            draw.line([(0, centro), (w - 1, centro)], fill=(0, 255, 0), width=3)

            draw.text((4, centro + 2), str(idx), fill=(255, 255, 0), font=font)
            draw.text((w - 24, centro + 2), str(idx), fill=(255, 255, 0), font=font)

            secos = detectar_tramos_secos(arr, mask_permitida, centro, orientacion_vertical=False, banda=4)

            for x1, x2 in secos:
                draw.line([(x1, centro), (x2, centro)], fill=(255, 0, 0), width=4)
                tramos_secos_total += 1

            lineas.append({"indice": idx, "y": int(centro)})

    img.save(ruta_salida, quality=95)

    return {
        "surcos_detectados": len(centros),
        "tramos_secos_detectados": tramos_secos_total,
        "orientacion_vertical": vertical,
        "lineas": lineas
    }


# =========================================================
# ENDPOINTS
# =========================================================
@app.get("/")
def inicio():
    return {
        "ok": True,
        "mensaje": "Backend de viñedo funcionando",
        "docs": "/docs"
    }


@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        ext = Path(file.filename).suffix.lower()
        if not ext:
            ext = ".jpg"

        nombre_base = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        nombre_archivo = f"{nombre_base}{ext}"

        ruta_upload = UPLOADS_DIR / nombre_archivo

        with open(ruta_upload, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        analisis = analizar_imagen_con_openai(str(ruta_upload))

        nombre_resultado = f"resultado_{nombre_archivo}"
        ruta_resultado = OUTPUTS_DIR / nombre_resultado

        dibujo = dibujar_surcos_en_imagen(
            ruta_entrada=str(ruta_upload),
            ruta_salida=str(ruta_resultado),
            analisis=analisis
        )

        return {
            "ok": True,
            "archivo_original": nombre_archivo,
            "imagen_original_url": f"/uploads/{nombre_archivo}",
            "imagen_resultado_url": f"/outputs/{nombre_resultado}",
            "analisis": analisis,
            "dibujo": dibujo
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": str(e)
            }
        )
