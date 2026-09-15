from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pathlib import Path
from datetime import datetime
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI
import base64
import io
import json
import os
import shutil
import uuid

# ============================================================
# CONFIGURACIÓN
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"

UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Backend Viñedo TerraCore")

app.mount(
    "/uploads",
    StaticFiles(directory=str(UPLOADS_DIR)),
    name="uploads"
)

app.mount(
    "/outputs",
    StaticFiles(directory=str(OUTPUTS_DIR)),
    name="outputs"
)


# ============================================================
# UTILIDADES
# ============================================================

def limpiar_json_respuesta(texto: str):
    texto = (texto or "").strip()

    if texto.startswith("```json"):
        texto = texto[len("```json"):].strip()
    elif texto.startswith("```"):
        texto = texto[3:].strip()

    if texto.endswith("```"):
        texto = texto[:-3].strip()

    return json.loads(texto)


def guardar_imagen(upload_file: UploadFile) -> Path:
    ext = Path(upload_file.filename or "").suffix.lower()

    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        raise HTTPException(
            status_code=400,
            detail="Formato no permitido. Usa JPG, JPEG, PNG o WEBP."
        )

    nombre = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
        f"{uuid.uuid4().hex[:8]}{ext}"
    )

    destino = UPLOADS_DIR / nombre

    with destino.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    return destino


def validar_imagen(ruta: Path):
    try:
        with Image.open(ruta) as img:
            img.verify()
    except Exception:
        if ruta.exists():
            ruta.unlink()
        raise HTTPException(
            status_code=400,
            detail="El archivo recibido no es una imagen válida."
        )


def preparar_imagen_para_ia(ruta: Path, max_side: int = 1600):
    """
    Reduce una copia para OpenAI si la fotografía es muy grande.
    La imagen original NO se modifica.
    """
    with Image.open(ruta) as img:
        img = img.convert("RGB")
        ancho, alto = img.size

        escala = min(
            1.0,
            float(max_side) / float(max(ancho, alto))
        )

        if escala < 1.0:
            nuevo_ancho = max(1, int(round(ancho * escala)))
            nuevo_alto = max(1, int(round(alto * escala)))
            img = img.resize(
                (nuevo_ancho, nuevo_alto),
                Image.Resampling.LANCZOS
            )

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=88)

        return buffer.getvalue(), ancho, alto


def analizar_imagen_con_openai(ruta_imagen: Path):
    """
    Envía una fotografía a OpenAI y obtiene una descripción
    estructurada del viñedo.

    En este paso todavía NO dibujamos líneas.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "No existe OPENAI_API_KEY en las variables de entorno."
        )

    image_bytes, original_width, original_height = preparar_imagen_para_ia(
        ruta_imagen
    )

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{image_b64}"

    prompt = """
Analiza esta fotografía de un viñedo desde una vista aérea o elevada.
Responde SOLO con JSON válido, sin markdown ni explicaciones fuera del JSON.

Objetivo futuro: usar tu respuesta para dibujar una línea por cada surco.
Todavía NO dibujes la imagen. Solo analiza la escena.

Devuelve exactamente esta estructura:
{
  "es_vinedo": true,
  "analizar_surcos": true,
  "orientacion_principal_grados": 90.0,
  "surcos_estimados": 80,
  "confianza_visual": 0.85,
  "hay_camino": true,
  "hay_techo": false,
  "hay_arboles": false,
  "hay_construccion": false,
  "resumen": "Descripción breve de la parcela.",
  "zonas_excluir": [
    {
      "tipo": "camino",
      "puntos": [[0,850],[1000,850],[1000,1000],[0,1000]]
    }
  ]
}

REGLAS:
- Las coordenadas de zonas_excluir están normalizadas de 0 a 1000.
- (0,0) es la esquina superior izquierda.
- (1000,1000) es la esquina inferior derecha.
- Excluye caminos, calles, techos, edificios, patios, jardines ornamentales,
  árboles grandes aislados y cualquier área que NO pertenezca a los surcos.
- NO excluyas huecos secos dentro de una hilera.
- NO excluyas tierra entre hileras.
- NO excluyas una hilera débil si sigue el patrón de la parcela.
- orientacion_principal_grados debe representar la dirección dominante de las hileras
  en la imagen, entre 0 y 180 grados.
- surcos_estimados debe ser un conteo visual aproximado, no inventes precisión falsa.
- confianza_visual debe ser un número entre 0 y 1.
- Si no hay un viñedo útil, analizar_surcos debe ser false.
- zonas_excluir puede ser una lista vacía.
- Usa de 4 a 12 puntos por polígono.
- Devuelve SOLO JSON válido.
"""

    client = OpenAI(api_key=OPENAI_API_KEY)

    errores = []

    # Primer intento: detalle alto.
    for detail in ["high", "low"]:
        try:
            response = client.responses.create(
                model="gpt-5.6-luna",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": prompt
                            },
                            {
                                "type": "input_image",
                                "image_url": data_url,
                                "detail": detail
                            }
                        ]
                    }
                ]
            )

            datos = limpiar_json_respuesta(response.output_text)

            if not isinstance(datos, dict):
                raise RuntimeError(
                    "OpenAI respondió, pero el resultado no fue un objeto JSON."
                )

            datos.setdefault("zonas_excluir", [])
            datos.setdefault("es_vinedo", False)
            datos.setdefault("analizar_surcos", False)
            datos.setdefault("surcos_estimados", 0)
            datos.setdefault("orientacion_principal_grados", None)
            datos.setdefault("confianza_visual", None)
            datos["detalle_usado"] = detail
            datos["ancho_original"] = int(original_width)
            datos["alto_original"] = int(original_height)

            return datos

        except Exception as e:
            errores.append(f"detail={detail}: {str(e)}")

    raise RuntimeError(
        "OpenAI no pudo analizar la imagen. " + " | ".join(errores)
    )


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
def inicio():
    return {
        "ok": True,
        "mensaje": "Backend TerraCore activo",
        "openai_configurado": bool(OPENAI_API_KEY)
    }


@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No se recibió ningún archivo."
        )

    ruta_imagen = guardar_imagen(file)
    validar_imagen(ruta_imagen)

    try:
        analisis = analizar_imagen_con_openai(ruta_imagen)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "archivo_original": ruta_imagen.name,
                "error": str(e)
            }
        )

    return JSONResponse({
        "ok": True,
        "archivo_original": ruta_imagen.name,
        "imagen_original_url": f"/uploads/{ruta_imagen.name}",
        "analisis": analisis
    })
