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
    version="1.4.2-conteo-verificado",
    description=(
        "Normaliza la imagen, dibuja líneas de surcos con GPT Image, "
        "cuenta los surcos y calcula porcentaje verde/rojo."
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

    if imagen.mode != "RGB":
        imagen = imagen.convert("RGB")

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
the same framing, camera angle, lighting, vineyard geometry, and parcel boundaries.

GOAL:
Overlay exactly ONE thin guide line centered on EACH true vineyard row.

IMPORTANT:
The row-counting logic will run AFTER this image is generated.
Therefore, do NOT add extra lines, duplicate lines, decorative marks, points,
arrows, labels, numbers, or any other overlay that could be mistaken for a vineyard row.

ROW GEOMETRY:
- Draw exactly ONE thin smooth guide trajectory for EACH true physical vineyard row.
- Follow the real row from one end to the other.
- Keep the line centered on the actual vineyard row.
- Follow curves and small deviations in the real row.
- Do not duplicate a row.
- Do not invent rows.
- Do not draw between rows.
- Do not draw on roads, perimeter lanes, buildings, roofs, patios, vehicles,
  shadows, large trees, or neighboring non-vineyard areas.
- If a row passes through a dry or missing section, keep following the SAME row.
- A dry gap does NOT mean the row stops.

COLOR CLASSIFICATION — CRITICAL:

GREEN means:
CLEARLY LIVING VINE VEGETATION IS PRESENT ON THAT EXACT ROW SEGMENT.

Use BRIGHT GREEN ONLY when:
- vine canopy is visibly present,
- vegetation is clearly alive,
- vegetation is sufficiently continuous,
- the segment is not mostly bare soil,
- the segment is not predominantly brown, beige, dry, weak, or empty.

RED means:
THE PHYSICAL VINEYARD ROW EXISTS, BUT THAT SEGMENT IS DRY, MISSING, WEAK,
INTERRUPTED, OR HAS LITTLE/NO LIVING VINE VEGETATION.

Use BRIGHT RED when:
- plants are missing,
- there is a visible gap in the row,
- exposed soil appears where vine canopy should be,
- the row is mostly brown or beige,
- the vegetation is sparse,
- the vegetation is clearly weaker than neighboring healthy row segments,
- the row is interrupted,
- the segment looks dry or dead,
- only isolated small green spots exist but the overall segment is bare or dry.

VERY IMPORTANT COLOR RULE:
DO NOT USE GREEN AS THE DEFAULT COLOR.

Before coloring every segment, inspect the underlying original photograph.

If the row position is known but vegetation is absent or doubtful:
USE RED.

If the segment is mostly bare, brown, beige, dry, interrupted, or missing:
USE RED.

GREEN is allowed ONLY when living vegetation is clearly visible.

STRICT DECISION RULE:
1. Clearly healthy living vegetation -> GREEN.
2. Clearly dry / bare / missing vegetation -> RED.
3. Mixed but mostly healthy -> GREEN.
4. Mixed but mostly dry / sparse / exposed soil -> RED.
5. Uncertain but row location is clear -> prefer RED.

DO NOT:
- paint bare soil GREEN,
- paint missing plants GREEN,
- paint dry gaps GREEN,
- paint weak brown sections GREEN just to maintain visual continuity,
- classify green weeds or grass BETWEEN rows as healthy vine canopy,
- treat shadows as vegetation,
- treat bare inter-row soil as a vineyard row.

ROW CONTINUITY:
A single physical row may look like:

GREEN -> RED -> GREEN -> RED -> GREEN

and it is still ONE vineyard row.

The COLOR may change,
but the TRAJECTORY must remain aligned as one continuous row.

FINAL COLOR CHECK:
Before returning the edited image, review every marked trajectory:

- Is there actual living vine canopy under every GREEN section?
- If not, change that GREEN section to RED.
- Is there visible exposed soil or missing vegetation within an established row?
- If yes, use RED.
- Is a weak/dry section incorrectly green?
- If yes, change it to RED.
- Are there green weeds between rows?
- Do not classify them as healthy vineyard vegetation.

STYLE:
- Thin clean guide lines.
- Bright saturated GREEN for healthy vegetation.
- Bright saturated RED for missing/dry/weak vegetation.
- Preserve the original image details.
- Do not fill areas.
- Do not add labels or numbers.
"""


# ============================================================
# CONTEO DE SURCOS EN LA IMAGEN YA MARCADA
# ============================================================

def contar_surcos_con_vision(image_bytes: bytes) -> dict:
    """
    Conteo 100% con IA sobre la imagen FINAL ya marcada.

    Objetivo:
    - contar cada trayectoria física una sola vez;
    - evitar duplicados;
    - evitar contar fragmentos cortos o líneas de borde como surcos extra;
    - verde + rojo alineados = un solo surco.
    """
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{image_b64}"

    prompt = """
Observa cuidadosamente esta imagen aérea de un viñedo YA MARCADA con líneas
verdes y rojas.

Tu única tarea es CONTAR LOS SURCOS MARCADOS.

REGLA PRINCIPAL:
Cada trayectoria física real del viñedo debe contarse UNA SOLA VEZ.

MUY IMPORTANTE:
- Una misma trayectoria puede tener segmentos VERDES y ROJOS.
- Si esos segmentos están alineados sobre la misma hilera, cuentan como UN SOLO SURCO.
- NO cuentes cada cambio de color como un surco diferente.
- NO cuentes dos líneas superpuestas o casi paralelas sobre la misma hilera como dos surcos.
- NO cuentes fragmentos cortos aislados que no representen una hilera completa.
- NO cuentes líneas accidentales sobre caminos, bordes, árboles, sombras o elementos de la foto.
- NO inventes surcos.
- NO sumes una línea dos veces porque esté interrumpida.

MÉTODO DE CONTEO OBLIGATORIO:

PASO 1 — IDENTIFICAR EL PATRÓN
- Determina la orientación principal de las hileras.
- Observa la separación regular entre surcos reales.
- Usa esa separación para detectar si dos líneas están duplicadas sobre la misma hilera.

PASO 2 — PRIMER CONTEO
- Recorre la imagen de IZQUIERDA A DERECHA.
- Sigue cada trayectoria desde su inicio hasta su final.
- Cuenta cada trayectoria física una sola vez.
- Verde + rojo alineados = 1.

PASO 3 — SEGUNDO CONTEO
- Repite el conteo de DERECHA A IZQUIERDA.
- Verifica especialmente los extremos izquierdo y derecho.
- Revisa si existe alguna línea duplicada, parcial o de borde.

PASO 4 — CONTROL DE DUPLICADOS
Antes de dar el total final, busca específicamente:
- dos líneas muy cercanas que sigan la misma hilera;
- líneas que se monten sobre una misma trayectoria;
- fragmentos que parezcan un surco nuevo pero en realidad sean continuación de otro;
- líneas cortas en los extremos que no correspondan a una hilera completa.

Si detectas uno de estos casos:
NO lo cuentes como un surco adicional.

PASO 5 — TOTAL FINAL
- El total final debe ser el número de trayectorias físicas ÚNICAS.
- Si el conteo izquierda→derecha y derecha→izquierda difieren,
  revisa la zona donde aparece la diferencia y corrige antes de responder.
- No promedies si puedes resolver visualmente cuál es el total correcto.

Devuelve SOLO JSON válido con exactamente esta estructura:

{
  "conteo_izquierda_derecha": 0,
  "conteo_derecha_izquierda": 0,
  "duplicados_descartados": 0,
  "fragmentos_descartados": 0,
  "surcos_contados": 0,
  "confianza": "alta",
  "observacion": "breve explicación"
}

Para "confianza" usa solamente:
"alta", "media" o "baja".
"""

    response = client.responses.create(
        model="gpt-5.6-luna",
        reasoning={"effort": "high"},
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

    data = limpiar_json(
        getattr(response, "output_text", "") or ""
    )

    def _entero(valor):
        try:
            return max(0, int(round(float(valor))))
        except Exception:
            return 0

    izquierda_derecha = _entero(
        data.get("conteo_izquierda_derecha", 0)
    )
    derecha_izquierda = _entero(
        data.get("conteo_derecha_izquierda", 0)
    )
    duplicados = _entero(
        data.get("duplicados_descartados", 0)
    )
    fragmentos = _entero(
        data.get("fragmentos_descartados", 0)
    )
    final = _entero(
        data.get("surcos_contados", 0)
    )

    # Respaldo: si la IA no devuelve total final, usar consenso visual.
    if final <= 0:
        if izquierda_derecha > 0 and derecha_izquierda > 0:
            if izquierda_derecha == derecha_izquierda:
                final = izquierda_derecha
            else:
                final = min(
                    izquierda_derecha,
                    derecha_izquierda,
                )
        else:
            final = max(
                izquierda_derecha,
                derecha_izquierda,
                0,
            )

    confianza = str(
        data.get("confianza", "media")
    ).lower().strip()

    if confianza not in {"alta", "media", "baja"}:
        confianza = "media"

    if izquierda_derecha > 0 and derecha_izquierda > 0:
        diferencia = abs(
            izquierda_derecha - derecha_izquierda
        )

        if diferencia >= 4:
            confianza = "baja"
        elif diferencia >= 2 and confianza == "alta":
            confianza = "media"

    return {
        "surcos_contados": int(final),
        "confianza": confianza,
        "observacion": str(
            data.get("observacion", "")
        ).strip(),
        "conteo_izquierda_derecha": int(
            izquierda_derecha
        ),
        "conteo_derecha_izquierda": int(
            derecha_izquierda
        ),
        "duplicados_descartados": int(
            duplicados
        ),
        "fragmentos_descartados": int(
            fragmentos
        ),
    }


# ============================================================
# PORCENTAJE VERDE / ROJO DE LAS LÍNEAS
# ============================================================

def calcular_porcentajes_lineas(image_bytes: bytes) -> dict:
    """
    Calcula el porcentaje relativo de los píxeles BRILLANTES/SATURADOS
    verdes y rojos del overlay.

    Importante:
    - No calcula el porcentaje de toda la vegetación del terreno.
    - Calcula la proporción entre los tramos de línea verdes y rojos
      que aparecen en la imagen generada.
    """
    imagen = Image.open(io.BytesIO(image_bytes)).convert("HSV")

    # Reducir un poco la imagen para acelerar el cálculo sin alterar proporciones.
    max_side = 1600
    w, h = imagen.size

    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        imagen = imagen.resize(
            (
                max(1, int(round(w * scale))),
                max(1, int(round(h * scale))),
            ),
            Image.Resampling.BILINEAR,
        )

    pixeles_verdes = 0
    pixeles_rojos = 0

    # En PIL HSV:
    # H: 0..255, S: 0..255, V: 0..255
    # Rojo ≈ H cercano a 0 o 255
    # Verde ≈ H alrededor de 60..110
    for h_val, s_val, v_val in imagen.getdata():

        # Solo colores muy vivos/brillantes para evitar contar
        # la vegetación natural de la fotografía.
        if s_val < 150 or v_val < 140:
            continue

        # Verde brillante
        if 45 <= h_val <= 105:
            pixeles_verdes += 1
            continue

        # Rojo brillante
        if h_val <= 12 or h_val >= 245:
            pixeles_rojos += 1

    total = pixeles_verdes + pixeles_rojos

    if total <= 0:
        return {
            "pixeles_verdes": 0,
            "pixeles_rojos": 0,
            "verde_pct": 0.0,
            "rojo_pct": 0.0,
        }

    verde_pct = round((pixeles_verdes / total) * 100.0, 1)
    rojo_pct = round(100.0 - verde_pct, 1)

    return {
        "pixeles_verdes": int(pixeles_verdes),
        "pixeles_rojos": int(pixeles_rojos),
        "verde_pct": float(verde_pct),
        "rojo_pct": float(rojo_pct),
    }


# ============================================================
# DIAGNÓSTICO VISUAL AGRONÓMICO
# ============================================================

def detectar_zona_mas_afectada(image_bytes: bytes) -> dict:
    """
    Divide la imagen procesada en izquierda / centro / derecha y estima
    en cuál hay mayor presencia de la línea roja del overlay.

    Esto NO diagnostica por sí solo la causa agronómica.
    """
    imagen = Image.open(io.BytesIO(image_bytes)).convert("HSV")

    max_side = 1600
    w, h = imagen.size

    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        imagen = imagen.resize(
            (
                max(1, int(round(w * scale))),
                max(1, int(round(h * scale))),
            ),
            Image.Resampling.BILINEAR,
        )
        w, h = imagen.size

    zonas = {
        "izquierda": {"rojo": 0, "verde": 0},
        "centro": {"rojo": 0, "verde": 0},
        "derecha": {"rojo": 0, "verde": 0},
    }

    tercio_1 = w / 3.0
    tercio_2 = 2.0 * w / 3.0

    pix = imagen.load()

    for y in range(h):
        for x in range(w):
            h_val, s_val, v_val = pix[x, y]

            # Solo overlay muy saturado/brillante.
            if s_val < 150 or v_val < 140:
                continue

            if x < tercio_1:
                zona = "izquierda"
            elif x < tercio_2:
                zona = "centro"
            else:
                zona = "derecha"

            if 45 <= h_val <= 105:
                zonas[zona]["verde"] += 1
            elif h_val <= 12 or h_val >= 245:
                zonas[zona]["rojo"] += 1

    # Proporción de rojo dentro de cada zona.
    detalle = {}

    for nombre, valores in zonas.items():
        total = valores["rojo"] + valores["verde"]

        if total > 0:
            rojo_pct = round(
                valores["rojo"] / total * 100.0,
                1,
            )
        else:
            rojo_pct = 0.0

        detalle[nombre] = {
            "rojo_pct": rojo_pct,
            "pixeles_rojos": int(valores["rojo"]),
            "pixeles_verdes": int(valores["verde"]),
        }

    zona_mas_afectada = max(
        detalle,
        key=lambda z: detalle[z]["rojo_pct"],
    )

    if all(
        detalle[z]["pixeles_rojos"] == 0
        for z in detalle
    ):
        zona_mas_afectada = "sin afectación clara"

    return {
        "zona_mas_afectada": zona_mas_afectada,
        "detalle_zonas": detalle,
    }


def construir_diagnostico_agronomico(
    verde_pct: float,
    rojo_pct: float,
    zona_mas_afectada: str,
) -> dict:
    """
    Genera una interpretación VISUAL PRELIMINAR.

    Importante:
    La imagen no permite confirmar por sí sola una deficiencia de N, P, K
    u otro nutriente. Las causas se reportan como hipótesis a validar con
    inspección, riego, suelo y análisis foliar.
    """
    verde_pct = float(max(0.0, min(100.0, verde_pct)))
    rojo_pct = float(max(0.0, min(100.0, rojo_pct)))

    if rojo_pct >= 45.0:
        nivel = "alto"
        diagnostico = (
            f"Se observa una afectación visual alta, con una proporción "
            f"importante de tramos secos, ralos o con pérdida de cobertura. "
            f"La mayor afectación visual aparece en la zona {zona_mas_afectada}."
        )
        causas = [
            "estrés hídrico o riego insuficiente/desuniforme",
            "baja disponibilidad de nutrientes",
            "compactación, mal drenaje o limitación de raíces",
            "salinidad u otra limitación química del suelo",
            "problemas sanitarios que requieren revisión en campo",
        ]
        recomendaciones = [
            "revisar presión, caudal y uniformidad del sistema de riego",
            "realizar análisis de suelo por zona afectada y zona sana de comparación",
            "realizar análisis foliar para confirmar o descartar deficiencias nutricionales",
            "medir pH, conductividad eléctrica y materia orgánica del suelo",
            "evaluar compactación, drenaje, raíces, plagas y enfermedades en campo",
        ]

    elif rojo_pct >= 20.0:
        nivel = "medio"
        diagnostico = (
            f"Se observa una afectación visual media, con mezcla de tramos "
            f"vigorosos y tramos débiles o secos. La mayor afectación visual "
            f"aparece en la zona {zona_mas_afectada}."
        )
        causas = [
            "estrés hídrico localizado",
            "distribución irregular del riego",
            "fertilidad o materia orgánica desuniforme",
            "compactación o variación física del suelo",
            "posible problema sanitario localizado",
        ]
        recomendaciones = [
            "inspeccionar en campo los tramos rojos",
            "comparar humedad y funcionamiento del riego entre zonas",
            "tomar muestras de suelo separadas en zona afectada y zona sana",
            "considerar análisis foliar para confirmar estado nutricional",
            "revisar raíces y presencia de plagas o enfermedades",
        ]

    elif rojo_pct > 0.0:
        nivel = "bajo"
        diagnostico = (
            f"Se observa una afectación visual baja o localizada. La mayor parte "
            f"de la cobertura marcada permanece verde, aunque existen algunos "
            f"tramos débiles o secos, principalmente en la zona {zona_mas_afectada}."
        )
        causas = [
            "estrés puntual por humedad",
            "variación normal del vigor",
            "pequeñas diferencias de suelo o fertilidad",
            "fallas puntuales de riego",
        ]
        recomendaciones = [
            "dar seguimiento a los puntos rojos en siguientes vuelos",
            "verificar emisores de riego cercanos a los tramos afectados",
            "hacer inspección de campo en los puntos más repetitivos",
            "muestrear suelo si la afectación aumenta o se concentra",
        ]

    else:
        nivel = "sin afectación roja detectada"
        diagnostico = (
            "No se detectó una proporción relevante de línea roja en la imagen "
            "procesada. Visualmente, el overlay indica predominio de vegetación."
        )
        causas = [
            "sin una causa de sequedad evidente en esta imagen procesada"
        ]
        recomendaciones = [
            "continuar monitoreo periódico",
            "mantener validación de campo y manejo agronómico preventivo",
        ]

    explicacion_nutrientes = (
        "Una fotografía por sí sola no permite afirmar qué nutriente falta. "
        "Nitrógeno, fósforo, potasio, magnesio, hierro u otros elementos pueden "
        "influir en el vigor, pero síntomas similares también pueden aparecer por "
        "falta o exceso de agua, salinidad, compactación, problemas de raíz, "
        "plagas o enfermedades. Para decidir una fertilización se recomienda "
        "confirmar con análisis de suelo y, de ser posible, análisis foliar."
    )

    return {
        "nivel_afectacion_visual": nivel,
        "diagnostico_visual": diagnostico,
        "causas_probables": causas,
        "explicacion_nutrientes": explicacion_nutrientes,
        "recomendaciones_iniciales": recomendaciones,
        "nota": (
            "Diagnóstico visual preliminar. No sustituye análisis de suelo, "
            "análisis foliar, revisión del riego ni diagnóstico agronómico en campo."
        ),
    }



# ============================================================
# ESTADO
# ============================================================

@app.get("/")
def root():
    return {
        "ok": True,
        "mensaje": "Backend TerraCore IA activo",
        "version": "1.4.2-conteo-verificado",
        "normalizacion_imagen": "RGB PNG",
        "metodo_lineas": "gpt-image-2.5-sunburst",
        "metodo_conteo": "gpt-5.6-luna",
        "porcentajes": "píxeles brillantes del overlay verde/rojo",
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
        # 0. NORMALIZAR FOTO
        # --------------------------------------------------------
        imagen_normalizada = normalizar_imagen(raw)

        imagen_normalizada.save(
            input_path,
            format="PNG",
            optimize=False,
        )

        # Validación final del PNG.
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

        # Validar salida generada.
        try:
            salida = Image.open(io.BytesIO(image_bytes))
            salida.load()
        except Exception as exc:
            raise RuntimeError(
                f"OpenAI devolvió una imagen que no pudo abrirse: {exc}"
            )

        # --------------------------------------------------------
        # 2. CONTAR SURCOS CON VISIÓN
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

        # --------------------------------------------------------
        # 3. CALCULAR PORCENTAJES VERDE / ROJO
        # --------------------------------------------------------
        try:
            porcentajes = calcular_porcentajes_lineas(image_bytes)
        except Exception as exc:
            porcentajes = {
                "pixeles_verdes": 0,
                "pixeles_rojos": 0,
                "verde_pct": 0.0,
                "rojo_pct": 0.0,
            }
            porcentaje_error = str(exc)
        else:
            porcentaje_error = None

        # --------------------------------------------------------
        # 4. DIAGNÓSTICO VISUAL AGRONÓMICO
        # --------------------------------------------------------
        try:
            zonas = detectar_zona_mas_afectada(image_bytes)

            diagnostico_agronomico = construir_diagnostico_agronomico(
                verde_pct=float(porcentajes["verde_pct"]),
                rojo_pct=float(porcentajes["rojo_pct"]),
                zona_mas_afectada=zonas["zona_mas_afectada"],
            )

            diagnostico_error = None

        except Exception as exc:
            zonas = {
                "zona_mas_afectada": "no determinada",
                "detalle_zonas": {},
            }

            diagnostico_agronomico = {
                "nivel_afectacion_visual": "no determinado",
                "diagnostico_visual": (
                    "No se pudo completar el diagnóstico visual agronómico."
                ),
                "causas_probables": [],
                "explicacion_nutrientes": (
                    "No se pudo generar una interpretación de nutrientes."
                ),
                "recomendaciones_iniciales": [],
                "nota": (
                    "Se requiere revisión del resultado y validación en campo."
                ),
            }

            diagnostico_error = str(exc)

        original_relative = f"/uploads/{input_name}"
        result_relative = f"/outputs/{output_name}"

        response_data = {
            "ok": True,

            "archivo_original": input_name,

            "imagen_original_url": original_relative,
            "imagen_original_url_publica": url_publica(original_relative),

            "imagen_resultado_url": result_relative,
            "imagen_resultado_url_publica": url_publica(result_relative),

            # Conteo
            "surcos_estimados": int(conteo["surcos_contados"]),
            "surcos_contados": int(conteo["surcos_contados"]),

            # Porcentajes directos para que tu Streamlit los lea fácil
            "verde_pct": float(porcentajes["verde_pct"]),
            "rojo_pct": float(porcentajes["rojo_pct"]),

            # Diagnóstico visual agronómico
            "zona_mas_afectada": zonas["zona_mas_afectada"],
            "nivel_afectacion_visual": diagnostico_agronomico[
                "nivel_afectacion_visual"
            ],
            "diagnostico_visual": diagnostico_agronomico[
                "diagnostico_visual"
            ],
            "causas_probables": diagnostico_agronomico[
                "causas_probables"
            ],
            "explicacion_nutrientes": diagnostico_agronomico[
                "explicacion_nutrientes"
            ],
            "recomendaciones_iniciales": diagnostico_agronomico[
                "recomendaciones_iniciales"
            ],
            "nota_diagnostico": diagnostico_agronomico["nota"],

            "analisis": {
                "surcos_estimados": int(conteo["surcos_contados"]),
                "surcos_contados": int(conteo["surcos_contados"]),

                "verde_pct": float(porcentajes["verde_pct"]),
                "rojo_pct": float(porcentajes["rojo_pct"]),

                "confianza_conteo": conteo["confianza"],
                "observacion_conteo": conteo["observacion"],

                "pixeles_linea_verdes": int(
                    porcentajes["pixeles_verdes"]
                ),
                "pixeles_linea_rojos": int(
                    porcentajes["pixeles_rojos"]
                ),

                "zona_mas_afectada": zonas["zona_mas_afectada"],
                "detalle_zonas": zonas["detalle_zonas"],
                "nivel_afectacion_visual": diagnostico_agronomico[
                    "nivel_afectacion_visual"
                ],
                "diagnostico_visual": diagnostico_agronomico[
                    "diagnostico_visual"
                ],
                "causas_probables": diagnostico_agronomico[
                    "causas_probables"
                ],
                "explicacion_nutrientes": diagnostico_agronomico[
                    "explicacion_nutrientes"
                ],
                "recomendaciones_iniciales": diagnostico_agronomico[
                    "recomendaciones_iniciales"
                ],
                "nota_diagnostico": diagnostico_agronomico["nota"],
            },

            "metodo": "gpt-image-2.5-sunburst",
            "metodo_lineas": "gpt-image-2.5-sunburst",
            "metodo_conteo": "gpt-5.6-luna",
            "metodo_porcentajes": "HSV líneas brillantes verde/rojo",

            "normalizacion": "RGB PNG",

            "tipo_proceso": (
                "edicion_visual_con_ia_conteo_porcentajes_y_diagnostico"
            ),

            "mensaje": (
                "Imagen procesada, surcos contados, porcentajes y diagnóstico visual calculados."
            ),
        }

        if conteo_error:
            response_data["advertencia_conteo"] = conteo_error

        if porcentaje_error:
            response_data["advertencia_porcentajes"] = porcentaje_error

        if diagnostico_error:
            response_data["advertencia_diagnostico"] = diagnostico_error

        return response_data

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": str(exc),
            },
        )
