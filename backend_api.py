from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI

from pathlib import Path
from datetime import datetime
from PIL import Image
import base64
import shutil
import uuid
import json
import re
import io
import os

# =========================================================
# CONFIGURACION
# =========================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"

UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini").strip()
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1").strip()

if not OPENAI_API_KEY:
    print("⚠️ No se encontró OPENAI_API_KEY en .env")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(
    title="Backend Viñedo TerraCore IA",
    version="1.3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")


# =========================================================
# PROMPTS
# =========================================================

PROMPT_ANALISIS = """
Analiza esta imagen aérea de un viñedo.

Necesito que respondas SOLO en JSON válido, sin explicación extra, sin markdown y sin texto adicional.

Devuelve exactamente esta estructura:

{
  "es_vinedo": true,
  "surcos_estimados": 0,
  "verde_pct": 0,
  "rojo_pct": 0,
  "orientacion_principal_grados": 90,
  "resumen": ""
}

Reglas:
- "es_vinedo": true si claramente es una parcela de viñedo.
- "surcos_estimados": número total aproximado de surcos visibles en la parcela principal.
- "verde_pct": porcentaje estimado de vegetación sana / verde.
- "rojo_pct": porcentaje estimado de tramos secos, faltantes o débiles.
- verde_pct + rojo_pct debe sumar aproximadamente 100.
- Ignora caminos, techos, construcciones y bordes externos.
- Si no es viñedo, responde:
  {
    "es_vinedo": false,
    "surcos_estimados": 0,
    "verde_pct": 0,
    "rojo_pct": 0,
    "orientacion_principal_grados": 0,
    "resumen": "No parece viñedo"
  }
- No pongas nada fuera del JSON.
"""

PROMPT_DIBUJO = """
Edita ESTA MISMA imagen del viñedo directamente sobre la fotografía original.

Instrucciones obligatorias:
- Dibuja una línea delgada siguiendo el centro de cada surco visible de la parcela principal.
- Donde el surco esté sano o con vegetación, pinta la línea en VERDE.
- Donde el surco tenga tramos secos, débiles o faltantes, pinta esos tramos en ROJO.
- Las líneas deben seguir la forma real del surco, no hacer una retícula rígida.
- No dibujes líneas sobre caminos, techos, construcciones ni fuera de la parcela principal.
- Numera cada surco con números pequeños y legibles.
- Coloca el número de cada surco ARRIBA y también ABAJO.
- El número de arriba y el de abajo deben corresponder al mismo surco.
- Mantén la foto original de fondo.
- No agregues título, leyenda, caja, tabla ni texto extra.
"""


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================

def extraer_json_de_texto(texto: str) -> dict:
    texto = texto.strip()

    try:
        return json.loads(texto)
    except Exception:
        pass

    match = re.search(r"\{.*\}", texto, re.DOTALL)
    if not match:
        raise ValueError("No se encontró un JSON válido en la respuesta del modelo.")

    bloque = match.group(0)
    return json.loads(bloque)


def clamp(valor, minimo, maximo):
    try:
        valor = int(round(float(valor)))
    except Exception:
        valor = minimo
    return max(minimo, min(valor, maximo))


def normalizar_porcentajes(verdes: int, rojos: int):
    verdes = clamp(verdes, 0, 100)
    rojos = clamp(rojos, 0, 100)

    total = verdes + rojos

    if total == 0:
        return 0, 0

    if total == 100:
        return verdes, rojos

    verdes = round((verdes / total) * 100)
    rojos = 100 - verdes

    verdes = clamp(verdes, 0, 100)
    rojos = clamp(rojos, 0, 100)

    return verdes, rojos


def guardar_upload(upload_file: UploadFile) -> Path:
    extension = Path(upload_file.filename or "imagen.jpg").suffix.lower()
    if extension not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
        extension = ".jpg"

    nombre = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{extension}"
    destino = UPLOADS_DIR / nombre

    with open(destino, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    return destino


def convertir_a_png_rgb(ruta_imagen: Path) -> Path:
    salida = ruta_imagen.with_suffix(".png")

    with Image.open(ruta_imagen) as img:
        img = img.convert("RGB")
        img.save(salida, format="PNG")

    return salida


def imagen_a_data_url(ruta_imagen: Path) -> str:
    mime = "image/png"
    with open(ruta_imagen, "rb") as f:
        contenido = f.read()
    b64 = base64.b64encode(contenido).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def elegir_size_para_edicion(ruta_imagen: Path) -> str:
    with Image.open(ruta_imagen) as img:
        w, h = img.size

    if abs(w - h) < 100:
        return "1024x1024"
    elif w > h:
        return "1536x1024"
    else:
        return "1024x1536"


def analizar_imagen_con_ia(ruta_png: Path) -> dict:
    data_url = imagen_a_data_url(ruta_png)

    respuesta = client.responses.create(
        model=OPENAI_VISION_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": PROMPT_ANALISIS
                    },
                    {
                        "type": "input_image",
                        "image_url": data_url,
                        "detail": "high"
                    }
                ]
            }
        ]
    )

    texto = getattr(respuesta, "output_text", "") or ""
    datos = extraer_json_de_texto(texto)

    es_vinedo = bool(datos.get("es_vinedo", False))
    surcos_estimados = clamp(datos.get("surcos_estimados", 0), 0, 1000)
    verde_pct = clamp(datos.get("verde_pct", 0), 0, 100)
    rojo_pct = clamp(datos.get("rojo_pct", 0), 0, 100)
    verde_pct, rojo_pct = normalizar_porcentajes(verde_pct, rojo_pct)

    orientacion = datos.get("orientacion_principal_grados", 90)
    try:
        orientacion = float(orientacion)
    except Exception:
        orientacion = 90

    resumen = str(datos.get("resumen", "")).strip()

    return {
        "es_vinedo": es_vinedo,
        "surcos_estimados": surcos_estimados,
        "verde_pct": verde_pct,
        "rojo_pct": rojo_pct,
        "orientacion_principal_grados": orientacion,
        "resumen": resumen
    }


def editar_imagen_con_ia(ruta_png: Path) -> bytes:
    size = elegir_size_para_edicion(ruta_png)

    with open(ruta_png, "rb") as f:
        respuesta = client.images.edit(
            model=OPENAI_IMAGE_MODEL,
            image=f,
            prompt=PROMPT_DIBUJO,
            size=size
        )

    item = respuesta.data[0]

    b64 = None
    if hasattr(item, "b64_json"):
        b64 = item.b64_json
    elif isinstance(item, dict):
        b64 = item.get("b64_json")

    if not b64:
        raise ValueError("La API no devolvió la imagen editada en base64.")

    return base64.b64decode(b64)


def guardar_resultado_png(imagen_bytes: bytes, nombre_base: str) -> Path:
    salida = OUTPUTS_DIR / f"resultado_{nombre_base}.png"

    img = Image.open(io.BytesIO(imagen_bytes)).convert("RGB")
    img.save(salida, format="PNG")

    return salida


def url_publica(request: Request, ruta_relativa: str) -> str:
    base = str(request.base_url).rstrip("/")
    return f"{base}{ruta_relativa}"


# =========================================================
# RUTAS
# =========================================================

@app.get("/")
def inicio():
    return {
        "ok": True,
        "mensaje": "Backend TerraCore IA activo",
        "version": "1.3.0",
        "normalizacion_imagen": "RGB PNG",
        "metodo_lineas": OPENAI_IMAGE_MODEL,
        "metodo_conteo": OPENAI_VISION_MODEL,
        "docs": "/docs"
    }


@app.post("/analyze-image")
async def analyze_image(request: Request, file: UploadFile = File(...)):
    try:
        if not OPENAI_API_KEY:
            return JSONResponse(
                status_code=500,
                content={
                    "ok": False,
                    "error": "Falta OPENAI_API_KEY en el archivo .env"
                }
            )

        # Guardar imagen original
        ruta_original = guardar_upload(file)
        nombre_archivo = ruta_original.name
        nombre_base = ruta_original.stem

        # Normalizar a PNG RGB para evitar errores de imagen inválida
        ruta_png = convertir_a_png_rgb(ruta_original)

        # 1) Analizar con IA
        analisis = analizar_imagen_con_ia(ruta_png)

        if not analisis["es_vinedo"]:
            return JSONResponse(
                status_code=200,
                content={
                    "ok": True,
                    "archivo_original": nombre_archivo,
                    "imagen_original_url": f"/uploads/{nombre_archivo}",
                    "imagen_original_url_publica": url_publica(request, f"/uploads/{nombre_archivo}"),
                    "analisis": analisis,
                    "mensaje": "La imagen no parece ser un viñedo."
                }
            )

        # 2) Editar con IA para dibujar líneas
        imagen_editada_bytes = editar_imagen_con_ia(ruta_png)
        ruta_salida = guardar_resultado_png(imagen_editada_bytes, nombre_base)

        return JSONResponse(
            status_code=200,
            content={
                "ok": True,
                "archivo_original": nombre_archivo,
                "imagen_original_url": f"/uploads/{nombre_archivo}",
                "imagen_original_url_publica": url_publica(request, f"/uploads/{nombre_archivo}"),
                "imagen_resultado_url": f"/outputs/{ruta_salida.name}",
                "imagen_resultado_url_publica": url_publica(request, f"/outputs/{ruta_salida.name}"),
                "analisis": {
                    "es_vinedo": analisis["es_vinedo"],
                    "surcos_estimados": analisis["surcos_estimados"],
                    "verde_pct": analisis["verde_pct"],
                    "rojo_pct": analisis["rojo_pct"],
                    "orientacion_principal_grados": analisis["orientacion_principal_grados"],
                    "resumen": analisis["resumen"]
                },
                "mensaje": "Imagen procesada correctamente con IA."
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": str(e)
            }
        )
