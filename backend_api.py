from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageOps, UnidentifiedImageError

from pathlib import Path
from datetime import datetime
import base64
import io
import json
import os
import uuid


# ============================================================
# CONFIGURACIÓN
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "Falta OPENAI_API_KEY. Configúrala como variable de entorno o secreto."
    )

client = OpenAI(api_key=OPENAI_API_KEY)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="TerraCore IA",
    version="1.2.0",
    description=(
        "Normaliza la imagen, dibuja líneas de surcos con GPT Image "
        "y cuenta los surcos marcados con visión."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/uploads",
    StaticFiles(directory=str(UPLOAD_DIR)),
    name="uploads",
)

app.mount(
    "/outputs",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="outputs",
)


# ============================================================
# UTILIDADES
# ============================================================

def crear_nombres() -> tuple[str, str]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    token = uuid.uuid4().hex[:8]

    input_name = f"{stamp}_{token}.png"
    output_name = f"resultado_{stamp}_{token}.png"

    return input_name, output_name


def url_publica(relative_path: str) -> str:
    if PUBLIC_BASE_URL:
        return f"{PUBLIC_BASE_URL}{relative_path}"
    return relative_path


def limpiar_json(texto: str) -> dict:
    texto = (texto or "").strip()

    if texto.startswith("```json"):
        texto = texto[7:].strip()
    elif texto.startswith("```"):
        texto = texto[3:].strip()

    if texto.endswith("```"):
        texto = texto[:-3].strip()

    try:
        data = json.loads(texto)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    inicio = texto.find("{")
    fin = texto.rfind("}")

    if inicio >= 0 and fin > inicio:
        data = json.loads(texto[inicio:fin + 1])
        if isinstance(data, dict):
            return data

    raise ValueError("La IA no devolvió JSON válido.")


# ============================================================
# NORMALIZACIÓN DE IMÁGENES
# ============================================================

def normalizar_imagen(raw: bytes) -> Image.Image:
    """
    Convierte cualquier JPG/JPEG/PNG/WEBP válido a:
    - orientación EXIF correcta
    - RGB
    - tamaño razonable
    - PNG estándar al guardarse

    Esto evita errores como:
    Invalid image file or mode
    """
    if not raw:
        raise ValueError("La imagen recibida está vacía.")

    try:
        imagen = Image.open(io.BytesIO(raw))
        imagen.load()
    except UnidentifiedImageError:
        raise ValueError(
            "El archivo recibido no pudo reconocerse como una imagen válida."
        )
    except Exception as exc:
        raise ValueError(f"No se pudo abrir la imagen: {exc}")

    try:
        imagen = ImageOps.exif_transpose(imagen)
    except Exception:
        pass

    # Eliminar modos problemáticos como CMYK, P, LA, I, etc.
    if imagen.mode != "RGB":
        imagen = imagen.convert("RGB")

    # Reducir imágenes excesivamente grandes para hacer la edición más estable.
    max_side = 2048
    width, height = imagen.size

    if max(width, height) > max_side:
        scale = max_side / float(max(width, height))
        new_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        imagen = imagen.resize(new_size, Image.Resampling.LANCZOS)

    return imagen


# ============================================================
# PROMPT DE EDICIÓN
# ============================================================

def vineyard_prompt() -> str:
    return """
Edit the provided aerial vineyard photo while preserving the original photograph,
same framing, camera angle, lighting, vineyard geometry, and parcel boundaries.

GOAL:
Overlay guide lines that follow the REAL vineyard rows as accurately as possible.

INSTRUCTIONS:
- Draw exactly ONE thin smooth guide line centered on EACH true vineyard row.
- Each line must follow the real row shape, including curves or small deviations.
- Use GREEN on portions of a row where vegetation is visibly present.
- Use RED only on portions of that SAME row where plants are visibly missing,
  dry, interrupted, or absent.
- Green and red portions belonging to the same physical row must remain aligned
  as one continuous row path.
- Do not create duplicate parallel lines on the same vineyard row.
- Do not invent extra vineyard rows.
- Do not draw lines in the spaces between rows.
- Do not draw on roads, perimeter lanes, roofs, buildings, patios, large trees,
  shadows, parking areas, or neighboring non-vineyard zones.
- Keep the original photograph visible and unchanged except for the thin overlay lines.
- Do not fill areas.
- Do not add decorative elements.
- Do not add labels or numbers.
- If a row is uncertain, omit it instead of inventing it.

STYLE:
- Thin clean overlay lines.
- Professional agronomic review style.
- Preserve as much original image detail as possible.
"""


# ============================================================
# CONTEO DE SURCOS EN LA IMAGEN YA MARCADA
# ============================================================

def contar_surcos_con_vision(image_bytes: bytes) -> dict:
    """
    Cuenta las trayectorias marcadas en la imagen final.
    Un mismo surco puede contener tramos verdes y rojos.
    """
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{image_b64}"

    prompt = """
Observa cuidadosamente esta imagen aérea de un viñedo YA MARCADA con líneas
verdes y rojas.

Tu tarea es contar únicamente los SURCOS MARCADOS.

REGLAS:
- Cuenta cada trayectoria o hilera física una sola vez.
- Una hilera puede tener segmentos verdes y rojos.
- Si los segmentos están alineados sobre la misma trayectoria, cuentan como UN solo surco.
- NO cuentes segmentos individuales.
- NO cuentes bordes de caminos, techos, árboles, sombras ni elementos originales de la foto.
- NO cuentes dos veces una línea que representa la misma hilera.
- Sigue visualmente cada trayectoria desde un extremo hasta el otro antes de sumar.
- Si una línea se interrumpe por un tramo seco y continúa en la misma alineación,
  sigue siendo el mismo surco.
- Cuenta solamente las hileras que tienen una línea de guía visible.

Devuelve SOLO JSON válido con exactamente esta estructura:

{
  "surcos_contados": 0,
  "confianza": "alta",
  "observacion": "breve explicación"
}

Para "confianza" usa solamente:
"alta", "media" o "baja".
"""

    response = client.responses.create(
        model="gpt-5.6-luna",
        reasoning={"effort": "medium"},
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                    {
                        "type": "input_image",
                        "image_url": data_url,
                        "detail": "high",
                    },
                ],
            }
        ],
    )

    data = limpiar_json(getattr(response, "output_text", "") or "")

    try:
        surcos = int(data.get("surcos_contados", 0))
    except Exception:
        surcos = 0

    surcos = max(0, surcos)

    confianza = str(data.get("confianza", "media")).lower().strip()

    if confianza not in {"alta", "media", "baja"}:
        confianza = "media"

    return {
        "surcos_contados": surcos,
        "confianza": confianza,
        "observacion": str(data.get("observacion", "")).strip(),
    }


# ============================================================
# ESTADO
# ============================================================

@app.get("/")
def root():
    return {
        "ok": True,
        "mensaje": "Backend TerraCore IA activo",
        "version": "1.2.0",
        "normalizacion_imagen": "RGB PNG",
        "metodo_lineas": "gpt-image-2.5-sunburst",
        "metodo_conteo": "gpt-5.6-luna",
        "docs": "/docs",
    }


# ============================================================
# ANALIZAR IMAGEN
# ============================================================

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        raw = await file.read()

        if not raw:
            return JSONResponse(
                status_code=400,
                content={
                    "ok": False,
                    "error": "La imagen recibida está vacía.",
                },
            )

        input_name, output_name = crear_nombres()

        input_path = UPLOAD_DIR / input_name
        output_path = OUTPUT_DIR / output_name

        # --------------------------------------------------------
        # 0. NORMALIZAR LA FOTO ANTES DE ENVIARLA A GPT IMAGE
        # --------------------------------------------------------
        imagen_normalizada = normalizar_imagen(raw)

        imagen_normalizada.save(
            input_path,
            format="PNG",
            optimize=False,
        )

        # Validación final: abrir de nuevo el PNG guardado.
        try:
            comprobacion = Image.open(input_path)
            comprobacion.load()

            if comprobacion.mode != "RGB":
                comprobacion = comprobacion.convert("RGB")
                comprobacion.save(
                    input_path,
                    format="PNG",
                    optimize=False,
                )
        except Exception as exc:
            raise RuntimeError(
                f"No se pudo validar el PNG normalizado: {exc}"
            )

        # --------------------------------------------------------
        # 1. GPT IMAGE DIBUJA LOS SURCOS
        # --------------------------------------------------------
        with input_path.open("rb") as img_file:
            result = client.images.edit(
                model="gpt-image-2.5-sunburst",
                image=img_file,
                prompt=vineyard_prompt(),
            )

        if not result.data:
            raise RuntimeError("OpenAI no devolvió ninguna imagen editada.")

        image_base64 = result.data[0].b64_json

        if not image_base64:
            raise RuntimeError(
                "OpenAI no devolvió la imagen editada en base64."
            )

        try:
            image_bytes = base64.b64decode(image_base64)
        except Exception as exc:
            raise RuntimeError(
                f"No se pudo decodificar la imagen generada: {exc}"
            )

        output_path.write_bytes(image_bytes)

        # Validar también la salida.
        try:
            salida = Image.open(io.BytesIO(image_bytes))
            salida.load()
        except Exception as exc:
            raise RuntimeError(
                f"OpenAI devolvió una imagen que no pudo abrirse: {exc}"
            )

        # --------------------------------------------------------
        # 2. GPT VISIÓN CUENTA LOS SURCOS YA MARCADOS
        # --------------------------------------------------------
        conteo_error = None

        try:
            conteo = contar_surcos_con_vision(image_bytes)
        except Exception as exc:
            conteo = {
                "surcos_contados": 0,
                "confianza": "baja",
                "observacion": "No se pudo completar el conteo visual.",
            }
            conteo_error = str(exc)

        original_relative = f"/uploads/{input_name}"
        result_relative = f"/outputs/{output_name}"

        response_data = {
            "ok": True,

            "archivo_original": input_name,

            "imagen_original_url": original_relative,
            "imagen_original_url_publica": url_publica(original_relative),

            "imagen_resultado_url": result_relative,
            "imagen_resultado_url_publica": url_publica(result_relative),

            # Compatible con tu app Streamlit actual.
            "surcos_estimados": int(conteo["surcos_contados"]),
            "surcos_contados": int(conteo["surcos_contados"]),

            "analisis": {
                "surcos_estimados": int(conteo["surcos_contados"]),
                "surcos_contados": int(conteo["surcos_contados"]),
                "confianza_conteo": conteo["confianza"],
                "observacion_conteo": conteo["observacion"],
            },

            "metodo": "gpt-image-2.5-sunburst",
            "metodo_conteo": "gpt-5.6-luna",
            "normalizacion": "RGB PNG",

            "tipo_proceso": "edicion_visual_con_ia_y_conteo_visual",

            "mensaje": (
                "Imagen normalizada, procesada y surcos contados correctamente."
            ),
        }

        if conteo_error:
            response_data["advertencia_conteo"] = conteo_error

        return response_data

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": str(exc),
            },
        )
