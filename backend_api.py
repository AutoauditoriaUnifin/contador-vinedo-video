import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
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
    version="1.4.2",
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
the same framing, camera angle, parcel layout, vineyard geometry, and visible roads.

GOAL:
Overlay guide lines that follow the REAL vineyard rows as accurately as possible.

MAIN INSTRUCTIONS:
- Draw exactly ONE thin smooth guide line centered on EACH true vineyard row.
- Each line must follow the real row shape from one end of the row to the other.
- The line must stay on the row, not between rows.
- Do not invent rows.
- Do not duplicate rows.
- Do not draw on roads, bare lanes, borders, roofs, vehicles, shadows, buildings,
  patios, large trees, or non-vineyard areas.
- Keep the original photograph unchanged except for the thin overlay lines.
- Do not fill areas.
- Do not add decorative elements.

COLOR RULES:
- Use BRIGHT GREEN only on row segments where vegetation is clearly present and active.
- Use BRIGHT RED on row segments where vegetation is weak, sparse, interrupted,
  dry, missing, or absent.
- A single physical row may contain both green and red segments.
- Green and red segments of the same row must stay aligned as one continuous row path.

IMPORTANT RED RULES:
Use RED more aggressively whenever the row segment shows one or more of these conditions:
- visible brown or beige gaps
- exposed soil in the row center
- sparse vegetation
- discontinuous or interrupted plants
- missing plants
- weak canopy
- dry-looking sections
- long or short empty gaps

IMPORTANT DECISION RULE:
- If a segment is doubtful between green and red, prefer RED.
- Do NOT leave mostly dry or weak sections in green.
- There should be noticeable red segments wherever the row is incomplete or dry.

STYLE:
- Thin clean overlay lines.
- Bright saturated green and bright saturated red so the overlay is easy to measure.
- Professional agronomic review style.
- Preserve as much original image detail as possible.
- Do NOT add labels, text, or numbers.
- If a row is uncertain, omit it instead of inventing it.
"""


# ============================================================
# CONTEO DE SURCOS EN LA IMAGEN YA MARCADA
# ============================================================

def _v33_mascara_verde(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hh, ss, vv = cv2.split(hsv)

    exg = 2.0 * g - r - b
    ngrdi = (g - r) / (g + r + 1e-6)

    mask = (
        (exg > 7.0)
        & (ngrdi > -0.015)
        & (hh >= 21)
        & (hh <= 108)
        & (ss >= 16)
        & (vv >= 22)
        & (g >= r * 0.875)
        & (g >= b * 0.875)
    ).astype(np.uint8) * 255

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((2, 2), np.uint8),
        iterations=1,
    )
    return mask


def _v33_angulo_surcos(mask):
    h, w = mask.shape

    x0, x1 = int(w * 0.12), int(w * 0.88)
    y0, y1 = int(h * 0.12), int(h * 0.90)

    roi = mask[y0:y1, x0:x1]
    edges = cv2.Canny(roi, 30, 100)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=45,
        minLineLength=max(45, int(h * 0.08)),
        maxLineGap=18,
    )

    if lines is None:
        return 90.0

    vals = []

    for x1l, y1l, x2l, y2l in np.asarray(lines).reshape(-1, 4):
        dx = float(x2l - x1l)
        dy = float(y2l - y1l)
        length = np.hypot(dx, dy)

        if length < 35:
            continue

        angle = np.degrees(np.arctan2(dy, dx))

        while angle < 0:
            angle += 180
        while angle >= 180:
            angle -= 180

        if 55 <= angle <= 125:
            vals.append((angle, length))

    if not vals:
        return 90.0

    angles = np.array([v[0] for v in vals])
    weights = np.array([v[1] for v in vals])

    bins = np.arange(55, 126, 2)
    hist, edges_b = np.histogram(
        angles,
        bins=bins,
        weights=weights,
    )

    i = int(np.argmax(hist))
    lo = edges_b[i]
    hi = edges_b[i + 1]

    sel = (angles >= lo) & (angles < hi)

    if np.any(sel):
        return float(
            np.average(
                angles[sel],
                weights=weights[sel],
            )
        )

    return float(np.median(angles))


def _v33_rotar(img, angle, interpolation=cv2.INTER_LINEAR):
    h, w = img.shape[:2]
    centro = (w / 2.0, h / 2.0)
    rot_deg = 90.0 - angle

    M = cv2.getRotationMatrix2D(
        centro,
        rot_deg,
        1.0,
    )

    out = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=interpolation,
        borderMode=(
            cv2.BORDER_REFLECT
            if interpolation != cv2.INTER_NEAREST
            else cv2.BORDER_CONSTANT
        ),
    )

    return out


def _v33_limites_verticales(mask):
    h, w = mask.shape

    density = (
        (mask > 0)
        .mean(axis=1)
        .astype(np.float32)
    )

    density = gaussian_filter1d(
        density,
        sigma=max(7, h / 100),
    )

    top_range = np.arange(
        int(h * 0.05),
        int(h * 0.48),
    )

    bottom_range = np.arange(
        int(h * 0.55),
        int(h * 0.95),
    )

    if len(top_range) == 0 or len(bottom_range) == 0:
        return int(h * 0.14), int(h * 0.88)

    y_top_road = int(
        top_range[
            np.argmin(density[top_range])
        ]
    )

    y_bottom_road = int(
        bottom_range[
            np.argmin(density[bottom_range])
        ]
    )

    inset = max(8, int(h * 0.01))

    y0 = y_top_road + inset
    y1 = y_bottom_road - inset

    if y1 - y0 < h * 0.36:
        y0 = int(h * 0.14)
        y1 = int(h * 0.88)

    return max(0, y0), min(h - 1, y1)


def _v33_estimar_periodo(profile):
    p = profile.astype(np.float64)
    p -= np.mean(p)

    if np.std(p) < 1e-7:
        return 20.0

    ac = np.correlate(
        p,
        p,
        mode="full",
    )

    ac = ac[len(p) - 1:]

    width = len(profile)
    a = max(8, int(width * 0.006))
    b = min(
        int(width * 0.03),
        len(ac) - 1,
    )

    if b <= a:
        return 20.0

    peaks, _ = find_peaks(
        ac[a:b + 1]
    )

    if len(peaks) == 0:
        return max(
            16.0,
            width / 70.0,
        )

    vals = ac[a:b + 1][peaks]

    lag = float(
        a + peaks[np.argmax(vals)]
    )

    if lag < width / 85.0:
        lag *= 2.0

    return float(
        np.clip(
            lag,
            13.0,
            35.0,
        )
    )


def _v33_detectar_limites_laterales(mask, y0, y1):
    h, w = mask.shape

    zone = (
        (mask[y0:y1] > 0)
        .astype(np.float32)
    )

    density = zone.mean(axis=0)

    density = gaussian_filter1d(
        density,
        sigma=max(3.0, w / 450.0),
    )

    p20 = float(
        np.percentile(density, 20)
    )

    p65 = float(
        np.percentile(density, 65)
    )

    threshold = (
        p20
        + 0.22 * max(
            p65 - p20,
            1e-6,
        )
    )

    low = density < threshold

    bands = []
    i = 0

    while i < w:
        if not low[i]:
            i += 1
            continue

        j = i + 1

        while j < w and low[j]:
            j += 1

        width_band = j - i

        if width_band >= max(
            5,
            int(w * 0.004),
        ):
            bands.append(
                (i, j - 1, width_band)
            )

        i = j

    center = w / 2.0

    left_candidates = [
        b
        for b in bands
        if b[1] < center
        and b[1] > w * 0.03
    ]

    right_candidates = [
        b
        for b in bands
        if b[0] > center
        and b[0] < w * 0.97
    ]

    if left_candidates:
        left_band = max(
            left_candidates,
            key=lambda b:
                b[2] * 3.0
                - abs(center - b[1]) * 0.012,
        )
        x0 = int(
            left_band[1]
            + max(3, w * 0.003)
        )
    else:
        x0 = int(w * 0.08)

    if right_candidates:
        right_band = max(
            right_candidates,
            key=lambda b:
                b[2] * 3.0
                - abs(b[0] - center) * 0.012,
        )
        x1 = int(
            right_band[0]
            - max(3, w * 0.003)
        )
    else:
        x1 = int(w * 0.92)

    if x1 - x0 < w * 0.35:
        x0 = int(w * 0.08)
        x1 = int(w * 0.92)

    return (
        max(0, x0),
        min(w - 1, x1),
    )


def _v33_semillas_surcos(mask, y0, y1):
    """
    Lógica V3.3 anterior:
    estima el espaciado físico de las hileras usando varios
    cortes horizontales y genera una sola semilla por surco.
    """
    x0, x1 = _v33_detectar_limites_laterales(
        mask,
        y0,
        y1,
    )

    zone = (
        (mask[y0:y1, x0:x1] > 0)
        .astype(np.float32)
    )

    height, width = zone.shape

    if width < 30 or height < 30:
        return (
            np.array([], dtype=np.int32),
            20.0,
            int(x0),
            int(x1),
        )

    profiles = []
    periods = []
    contrasts = []

    centers = np.linspace(
        0.12,
        0.88,
        9,
    )

    band_h = max(
        12,
        int(height * 0.11),
    )

    for frac in centers:
        yc = int(
            round(
                frac * (height - 1)
            )
        )

        a = max(
            0,
            yc - band_h // 2,
        )

        b = min(
            height,
            yc + band_h // 2 + 1,
        )

        band = zone[a:b]

        if band.size == 0:
            continue

        profile = band.mean(axis=0)

        profile = gaussian_filter1d(
            profile,
            sigma=1.15,
        )

        period = _v33_estimar_periodo(
            profile
        )

        period = float(
            np.clip(
                period,
                9.0,
                40.0,
            )
        )

        profiles.append(profile)
        periods.append(period)
        contrasts.append(
            float(np.std(profile))
        )

    if not profiles:
        return (
            np.array([], dtype=np.int32),
            20.0,
            int(x0),
            int(x1),
        )

    spacing_candidates = []

    for profile, p0 in zip(
        profiles,
        periods,
    ):
        peaks, _ = find_peaks(
            profile,
            distance=max(
                5,
                int(p0 * 0.45),
            ),
            prominence=max(
                0.003,
                float(profile.max()) * 0.01,
            ),
            height=max(
                0.006,
                float(profile.max()) * 0.018,
            ),
        )

        if len(peaks) >= 5:
            diffs = np.diff(
                peaks.astype(np.float32)
            )

            good = diffs[
                (diffs >= 8.0)
                & (diffs <= 42.0)
            ]

            if len(good):
                med = float(
                    np.median(good)
                )

                near = good[
                    (good > med * 0.62)
                    & (good < med * 1.38)
                ]

                if len(near):
                    spacing_candidates.extend(
                        near.tolist()
                    )

    if spacing_candidates:
        spacing = float(
            np.median(
                spacing_candidates
            )
        )
    else:
        spacing = float(
            np.median(periods)
        )

    spacing = float(
        np.clip(
            spacing,
            9.0,
            36.0,
        )
    )

    half = spacing / 2.0

    if half >= 8.0:
        score_full = 0.0
        score_half = 0.0

        for profile in profiles:
            for candidate, bucket in [
                (spacing, "full"),
                (half, "half"),
            ]:
                best = -1.0

                phase_steps = max(
                    10,
                    int(round(candidate * 1.5)),
                )

                for phase in np.linspace(
                    0,
                    candidate,
                    phase_steps,
                    endpoint=False,
                ):
                    xs = np.arange(
                        phase,
                        len(profile),
                        candidate,
                    )

                    if len(xs) < 4:
                        continue

                    values = []

                    for x in xs:
                        xi = int(round(x))
                        a = max(0, xi - 2)
                        b = min(
                            len(profile),
                            xi + 3,
                        )

                        if b > a:
                            values.append(
                                float(
                                    profile[a:b].mean()
                                )
                            )

                    if values:
                        best = max(
                            best,
                            float(np.mean(values)),
                        )

                if bucket == "full":
                    score_full += max(best, 0.0)
                else:
                    score_half += max(best, 0.0)

        # IMPORTANTE:
        # El V3.3 original probaba el medio período.
        # Para conteo físico evitamos dividir por 2 salvo que
        # la evidencia sea claramente superior.
        if score_half >= score_full * 1.10:
            spacing = half

    spacing = float(
        np.clip(
            spacing,
            8.0,
            36.0,
        )
    )

    best_profile_index = int(
        np.argmax(
            np.asarray(
                contrasts,
                dtype=np.float32,
            )
        )
    )

    ref_profile = profiles[
        best_profile_index
    ]

    best_phase = 0.0
    best_score = -1e9

    phase_steps = max(
        18,
        int(round(spacing * 3)),
    )

    for phase in np.linspace(
        0,
        spacing,
        phase_steps,
        endpoint=False,
    ):
        xs = np.arange(
            phase,
            width,
            spacing,
        )

        if len(xs) < 4:
            continue

        values = []

        for x in xs:
            xi = int(round(x))
            a = max(0, xi - 2)
            b = min(
                width,
                xi + 3,
            )

            if b > a:
                values.append(
                    float(
                        ref_profile[a:b].mean()
                    )
                )

        if not values:
            continue

        score = float(
            np.mean(values)
        )

        if score > best_score:
            best_score = score
            best_phase = float(phase)

    local_seeds = np.arange(
        best_phase,
        width,
        spacing,
        dtype=np.float32,
    )

    edge_margin = max(
        1.0,
        spacing * 0.08,
    )

    local_seeds = local_seeds[
        (local_seeds >= edge_margin)
        & (
            local_seeds
            <= width - 1 - edge_margin
        )
    ]

    return (
        np.rint(
            local_seeds + x0
        ).astype(np.int32),
        float(spacing),
        int(x0),
        int(x1),
    )


def contar_surcos_con_vision(image_bytes: bytes) -> dict:
    """
    CONTEO V3.3 RESTAURADO.

    Aunque conserva el nombre de la función para no cambiar
    el resto del backend, ya NO usa otro modelo de IA para contar.

    Cuenta la estructura periódica de las hileras físicas directamente
    sobre la fotografía ORIGINAL.
    """
    try:
        pil = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        bgr = cv2.cvtColor(
            np.asarray(pil),
            cv2.COLOR_RGB2BGR,
        )

        mask0 = _v33_mascara_verde(
            bgr
        )

        angle = _v33_angulo_surcos(
            mask0
        )

        rot_mask = _v33_rotar(
            mask0,
            angle,
            interpolation=cv2.INTER_NEAREST,
        )

        y0, y1 = _v33_limites_verticales(
            rot_mask
        )

        seeds, spacing, x0, x1 = (
            _v33_semillas_surcos(
                rot_mask,
                y0,
                y1,
            )
        )

        count = int(len(seeds))

        if count < 3:
            raise RuntimeError(
                "No se detectaron suficientes hileras."
            )

        return {
            "surcos_contados": count,
            "confianza": "alta",
            "observacion": (
                f"Conteo V3.3 por espaciado físico. "
                f"Separación media aproximada: {spacing:.1f}px. "
                f"Ángulo aproximado: {angle:.1f}°."
            ),
        }

    except Exception as exc:
        return {
            "surcos_contados": 0,
            "confianza": "baja",
            "observacion": (
                "No se pudo completar el conteo V3.3: "
                + str(exc)
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
        "version": "1.4.5-v33-count",
        "normalizacion_imagen": "RGB PNG",
        "metodo_lineas": "gpt-image-2.5-sunburst",
        "metodo_conteo": "V3.3 espaciado físico de hileras",
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
        # 2. CONTAR SURCOS CON V3.3 SOBRE LA FOTO ORIGINAL
        # --------------------------------------------------------
        conteo_error = None

        try:
            original_bytes_conteo = input_path.read_bytes()
            conteo = contar_surcos_con_vision(
                original_bytes_conteo
            )
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
