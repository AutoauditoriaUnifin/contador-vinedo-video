from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

import os
import io
import cv2
import json
import uuid
import base64
import traceback
import numpy as np

# =========================================================
# CARGA .ENV
# =========================================================
load_dotenv()

# =========================================================
# OPENAI (OPCIONAL)
# Si no hay llave o no hay créditos, el backend NO se rompe.
# =========================================================
HAS_OPENAI = False
try:
    from openai import OpenAI
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini")

# =========================================================
# APP
# =========================================================
app = FastAPI(
    title="Backend Viñedo TerraCore",
    version="0.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # puedes cerrarlo después si quieres
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# RUTAS / CARPETAS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# =========================================================
# UTILIDADES GENERALES
# =========================================================
def now_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def unique_name(ext=".jpg"):
    return f"{now_stamp()}_{uuid.uuid4().hex[:8]}{ext}"

def normalize_filename(filename: str) -> str:
    if not filename:
        return "imagen.jpg"
    filename = filename.replace("\\", "_").replace("/", "_").strip()
    return filename

def read_image_from_bytes(file_bytes: bytes):
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def save_bytes(path: Path, data: bytes):
    path.write_bytes(data)

def image_to_data_url(image_path: Path):
    mime = "image/jpeg"
    ext = image_path.suffix.lower()
    if ext == ".png":
        mime = "image/png"
    elif ext == ".webp":
        mime = "image/webp"
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def extract_json_from_text(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    ini = text.find("{")
    fin = text.rfind("}")
    if ini != -1 and fin != -1 and fin > ini:
        sub = text[ini:fin+1]
        return json.loads(sub)

    raise ValueError("No se pudo extraer JSON de la respuesta")

def moving_average(arr, k=7):
    arr = np.asarray(arr, dtype=np.float32)
    if len(arr) == 0:
        return arr
    k = max(1, int(k))
    if k == 1:
        return arr
    pad_left = k // 2
    pad_right = k - 1 - pad_left
    padded = np.pad(arr, (pad_left, pad_right), mode="edge")
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(padded, kernel, mode="valid")

def rotate_bound(image, angle_deg, interp=cv2.INTER_LINEAR, border_value=0):
    (h, w) = image.shape[:2]
    cX, cY = w / 2.0, h / 2.0

    M = cv2.getRotationMatrix2D((cX, cY), angle_deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    rotated = cv2.warpAffine(
        image,
        M,
        (nW, nH),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )
    return rotated, M

def inverse_warp(image, M, out_w, out_h, interp=cv2.INTER_LINEAR, border_value=0):
    M_inv = cv2.invertAffineTransform(M)
    out = cv2.warpAffine(
        image,
        M_inv,
        (out_w, out_h),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )
    return out

# =========================================================
# MÁSCARA VERDE
# =========================================================
def compute_green_mask(img_bgr):
    img = cv2.GaussianBlur(img_bgr, (5, 5), 0)

    b, g, r = cv2.split(img.astype(np.float32))
    exg = 2 * g - r - b
    exg = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mask_hsv = cv2.inRange(hsv, (25, 20, 20), (105, 255, 255))
    _, mask_exg = cv2.threshold(exg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask = cv2.bitwise_and(mask_hsv, mask_exg)

    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2, iterations=2)

    return mask

# =========================================================
# ORIENTACIÓN PRINCIPAL
# =========================================================
def estimate_orientation_from_mask(mask):
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=max(30, min(mask.shape[:2]) // 12),
        maxLineGap=20
    )

    angles = []
    weights = []

    if lines is not None:
        for line in lines[:, 0]:
            x1, y1, x2, y2 = line
            dx = x2 - x1
            dy = y2 - y1
            length = np.hypot(dx, dy)
            if length < 20:
                continue

            angle = np.degrees(np.arctan2(dy, dx))
            # normalizar a 0..180
            while angle < 0:
                angle += 180
            while angle >= 180:
                angle -= 180

            # Convertimos a orientación principal tipo "vertical ~ 90"
            if angle > 90:
                angle = 180 - angle

            # nos interesan filas casi verticales u oblicuas, no horizontales
            if 45 <= angle <= 90:
                angles.append(angle)
                weights.append(length)

    if len(angles) == 0:
        return 90.0

    angles = np.asarray(angles, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)

    order = np.argsort(angles)
    angles = angles[order]
    weights = weights[order]

    cum = np.cumsum(weights)
    cutoff = cum[-1] / 2.0
    idx = np.searchsorted(cum, cutoff)
    angle = float(angles[min(idx, len(angles) - 1)])

    return angle

# =========================================================
# PERFIL DE COLUMNAS / ESPACIADO / PICOS
# =========================================================
def estimate_spacing(profile):
    p = np.asarray(profile, dtype=np.float32)
    if len(p) < 20:
        return 12

    p = p - p.mean()
    if np.allclose(p, 0):
        return 12

    ac = np.correlate(p, p, mode="full")
    ac = ac[len(ac)//2:]

    start = 6
    end = min(len(ac) - 1, 60)
    if end <= start:
        return 12

    window = ac[start:end]
    peak_idx = np.argmax(window) + start
    spacing = int(max(8, peak_idx))
    return spacing

def find_row_peaks(profile, spacing):
    prof = moving_average(profile, k=max(5, int(spacing * 0.7)))
    prof = np.asarray(prof, dtype=np.float32)

    if len(prof) < 3:
        return []

    threshold = np.percentile(prof, 55)
    peaks = []

    for i in range(1, len(prof) - 1):
        if prof[i] >= threshold and prof[i] >= prof[i - 1] and prof[i] >= prof[i + 1]:
            peaks.append((i, prof[i]))

    if not peaks:
        return []

    min_dist = max(5, int(spacing * 0.55))
    peaks = sorted(peaks, key=lambda x: x[1], reverse=True)

    selected = []
    for idx, val in peaks:
        if all(abs(idx - s) >= min_dist for s in selected):
            selected.append(idx)

    selected.sort()
    return selected

# =========================================================
# ZONAS A EXCLUIR
# Espera polígonos tipo:
# [
#   {"tipo":"camino","puntos":[[0,0],[1000,0],[1000,30],[0,30]]}
# ]
# =========================================================
def build_exclude_mask(img_shape, zonas_excluir):
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if not zonas_excluir:
        return mask

    for zona in zonas_excluir:
        puntos = zona.get("puntos", [])
        if len(puntos) < 3:
            continue

        pts = np.array(puntos, dtype=np.float32)

        max_x = pts[:, 0].max()
        max_y = pts[:, 1].max()

        # si viene normalizado 0..1000
        if max_x <= 1000.0 and max_y <= 1000.0:
            pts[:, 0] = (pts[:, 0] / 1000.0) * w
            pts[:, 1] = (pts[:, 1] / 1000.0) * h

        pts = pts.astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)

    return mask

# =========================================================
# TRACKING DE CADA SURCO
# =========================================================
def track_rows(rot_mask, rot_exclude_mask, peaks, spacing):
    h, w = rot_mask.shape[:2]
    step_h = max(8, h // 90)
    band_half = max(2, int(spacing * 0.20))
    search_radius = max(4, int(spacing * 0.40))

    rows = []

    for peak_x in peaks:
        points = []
        strengths = []
        excluded_flags = []

        current_x = int(peak_x)

        for y0 in range(0, h - 1, step_h):
            y1 = min(h, y0 + step_h)
            ymid = (y0 + y1) // 2

            best_x = current_x
            best_score = -1.0
            best_excluded = False

            x_start = max(band_half, current_x - search_radius)
            x_end = min(w - band_half - 1, current_x + search_radius)

            for x in range(x_start, x_end + 1):
                xa = max(0, x - band_half)
                xb = min(w, x + band_half + 1)

                excl_patch = rot_exclude_mask[y0:y1, xa:xb]
                excl_ratio = float(np.mean(excl_patch > 0)) if excl_patch.size > 0 else 0.0
                if excl_ratio > 0.45:
                    continue

                patch = rot_mask[y0:y1, xa:xb]
                score = float(np.sum(patch > 0))

                if score > best_score:
                    best_score = score
                    best_x = x
                    best_excluded = False

            patch_area = max(1, (y1 - y0) * (2 * band_half + 1))
            density = best_score / patch_area if best_score >= 0 else 0.0

            points.append((int(best_x), int(ymid)))
            strengths.append(float(density))
            excluded_flags.append(best_excluded)

            current_x = int(best_x)

        if len(points) < 5:
            continue

        xs = np.array([p[0] for p in points], dtype=np.float32)
        ys = np.array([p[1] for p in points], dtype=np.float32)

        xs_s = moving_average(xs, k=7)
        points_s = [(int(xs_s[i]), int(ys[i])) for i in range(len(ys))]

        strengths = np.array(strengths, dtype=np.float32)

        good_ratio = float(np.mean(strengths >= 0.10))
        std_x = float(np.std(xs_s))

        # filtros para quitar "líneas falsas" en árboles / ruido
        if good_ratio < 0.22:
            continue
        if std_x > spacing * 1.25:
            continue

        rows.append({
            "x_mean": float(np.mean(xs_s)),
            "points": points_s,
            "strengths": strengths.tolist()
        })

    return rows, step_h

def filter_regular_group(rows, spacing):
    if len(rows) <= 3:
        return rows

    rows = sorted(rows, key=lambda r: r["x_mean"])
    xs = [r["x_mean"] for r in rows]

    groups = []
    current = [rows[0]]

    for i in range(1, len(rows)):
        gap = xs[i] - xs[i - 1]
        if 0.45 * spacing <= gap <= 1.90 * spacing:
            current.append(rows[i])
        else:
            groups.append(current)
            current = [rows[i]]

    groups.append(current)

    groups = sorted(groups, key=lambda g: (len(g), sum(r["x_mean"] for r in g)), reverse=True)
    best = groups[0]

    # si el mejor grupo es muy pequeño, nos quedamos con todos
    if len(best) < max(4, len(rows) // 3):
        return rows

    return best

# =========================================================
# DIBUJADO FINAL
# =========================================================
def draw_rows_overlay(rot_shape, rows, dry_threshold=0.10):
    h, w = rot_shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    dry_segments = 0
    line_summaries = []

    rows = sorted(rows, key=lambda r: r["x_mean"])

    for idx, row in enumerate(rows, start=1):
        pts = row["points"]
        strengths = row["strengths"]

        if len(pts) < 2:
            continue

        x_top, _ = pts[0]
        x_bottom, _ = pts[-1]

        # numeración arriba
        cv2.putText(
            overlay, str(idx),
            (max(5, x_top - 8), 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 255, 255), 2, cv2.LINE_AA
        )

        # numeración abajo
        cv2.putText(
            overlay, str(idx),
            (max(5, x_bottom - 8), h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 255, 255), 2, cv2.LINE_AA
        )

        for i in range(len(pts) - 1):
            p1 = pts[i]
            p2 = pts[i + 1]
            strength = (strengths[i] + strengths[i + 1]) / 2.0

            if strength >= dry_threshold:
                color = (0, 220, 0)  # verde
            else:
                color = (0, 0, 255)  # rojo
                dry_segments += 1

            cv2.line(overlay, p1, p2, color, 2, cv2.LINE_AA)

        line_summaries.append({
            "indice": idx,
            "x_promedio": int(round(row["x_mean"]))
        })

    return overlay, dry_segments, line_summaries

# =========================================================
# FALLBACK SI OPENAI NO ESTÁ DISPONIBLE
# =========================================================
def fallback_analysis(img_bgr):
    h, w = img_bgr.shape[:2]
    mask = compute_green_mask(img_bgr)
    angle = estimate_orientation_from_mask(mask)

    # perfil rápido para estimación preliminar
    gray_profile = mask.mean(axis=0)
    spacing = estimate_spacing(gray_profile)
    peaks = find_row_peaks(gray_profile, spacing)

    # heurística simple de caminos en bordes
    hay_camino = True

    return {
        "es_vinedo": True,
        "analizar_surcos": True,
        "orientacion_principal_grados": int(round(angle)),
        "surcos_estimados": int(len(peaks)),
        "confianza_visual": 0.70,
        "hay_camino": hay_camino,
        "hay_techo": False,
        "hay_arboles": False,
        "hay_construccion": False,
        "resumen": "Análisis clásico local. Se detectan hileras de viñedo y se procede a dibujar los surcos.",
        "zonas_excluir": [],
        "detalle_usado": "fallback_local",
        "ancho_original": w,
        "alto_original": h
    }

# =========================================================
# ANÁLISIS OPENAI (OPCIONAL)
# Si falla por llave / crédito / etc, solo lanzará excepción y
# luego el backend seguirá con fallback clásico.
# =========================================================
def analyze_with_openai(image_path: Path):
    if not HAS_OPENAI:
        raise RuntimeError("La librería openai no está instalada.")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY no configurada.")

    client = OpenAI(api_key=OPENAI_API_KEY)
    data_url = image_to_data_url(image_path)

    system_prompt = """
Eres un analista visual especializado en viñedos.
Devuelve SOLO JSON válido.
No agregues texto extra.

Necesito este formato exacto:
{
  "es_vinedo": true,
  "analizar_surcos": true,
  "orientacion_principal_grados": 90,
  "surcos_estimados": 80,
  "confianza_visual": 0.95,
  "hay_camino": true,
  "hay_techo": false,
  "hay_arboles": false,
  "hay_construccion": false,
  "resumen": "texto corto",
  "zonas_excluir": [
    {
      "tipo": "camino",
      "puntos": [[x,y],[x,y],[x,y],[x,y]]
    }
  ]
}

Reglas:
- "orientacion_principal_grados" debe reflejar la dirección dominante de los surcos.
- Usa 90 si los surcos se ven predominantemente verticales.
- "surcos_estimados" debe ser una estimación visual razonable.
- "zonas_excluir" debe usar coordenadas normalizadas de 0 a 1000.
- Marca caminos, techos, construcciones o árboles SOLO si realmente estorban el conteo.
"""

    user_prompt = """
Analiza esta imagen aérea o de viñedo.

Quiero:
1) saber si es viñedo,
2) si conviene analizar surcos,
3) orientación principal,
4) estimación de número de surcos,
5) si hay camino, techos, árboles o construcciones,
6) zonas a excluir para que el motor de dibujo no trace sobre caminos o construcciones.

Devuelve SOLO JSON.
"""

    last_err = None

    for detail in ["high", "low"]:
        try:
            response = client.chat.completions.create(
                model=OPENAI_VISION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": data_url, "detail": detail}}
                        ]
                    }
                ],
                temperature=0
            )

            text = response.choices[0].message.content
            data = extract_json_from_text(text)
            data["detalle_usado"] = detail
            return data

        except Exception as e:
            last_err = e

    raise RuntimeError(f"OpenAI no pudo analizar la imagen. {last_err}")

# =========================================================
# MOTOR PRINCIPAL DE DIBUJO
# =========================================================
def process_vineyard_image(img_bgr, ai_analysis):
    h, w = img_bgr.shape[:2]

    green_mask = compute_green_mask(img_bgr)

    zonas_excluir = ai_analysis.get("zonas_excluir", []) if ai_analysis else []
    exclude_mask = build_exclude_mask(img_bgr.shape, zonas_excluir)

    orientation = 90.0
    if ai_analysis:
        orientation = float(ai_analysis.get("orientacion_principal_grados", 90))
    if not orientation:
        orientation = estimate_orientation_from_mask(green_mask)

    # Queremos que las hileras queden casi verticales
    rotate_angle = 90.0 - orientation

    rot_img, M = rotate_bound(img_bgr, rotate_angle, interp=cv2.INTER_LINEAR, border_value=(0, 0, 0))
    rot_mask, _ = rotate_bound(green_mask, rotate_angle, interp=cv2.INTER_NEAREST, border_value=0)
    rot_excl, _ = rotate_bound(exclude_mask, rotate_angle, interp=cv2.INTER_NEAREST, border_value=0)

    usable_mask = rot_mask.copy()
    usable_mask[rot_excl > 0] = 0

    profile = usable_mask.mean(axis=0)
    spacing = estimate_spacing(profile)
    peaks = find_row_peaks(profile, spacing)

    rows, step_h = track_rows(usable_mask, rot_excl, peaks, spacing)
    rows = filter_regular_group(rows, spacing)

    # Si se quedó demasiado corto, relajamos un poco
    if len(rows) < 8:
        extra_peaks = find_row_peaks(profile, max(8, int(spacing * 0.85)))
        rows2, _ = track_rows(usable_mask, rot_excl, extra_peaks, max(8, int(spacing * 0.85)))
        rows2 = filter_regular_group(rows2, max(8, int(spacing * 0.85)))
        if len(rows2) > len(rows):
            rows = rows2

    overlay_rot, dry_segments, line_summaries = draw_rows_overlay(
        rot_img.shape,
        rows,
        dry_threshold=0.10
    )

    overlay = inverse_warp(
        overlay_rot,
        M,
        out_w=w,
        out_h=h,
        interp=cv2.INTER_LINEAR,
        border_value=(0, 0, 0)
    )

    result = img_bgr.copy()

    # superponer solo donde haya dibujo
    nonzero = np.any(overlay > 0, axis=2)
    result[nonzero] = overlay[nonzero]

    drawing_info = {
        "surcos_detectados": int(len(rows)),
        "tramos_secos_detectados": int(dry_segments),
        "orientacion_vertical": True,
        "lineas": line_summaries
    }

    return result, drawing_info

# =========================================================
# ENDPOINTS
# =========================================================
@app.get("/")
def inicio():
    return {
        "ok": True,
        "mensaje": "Backend Viñedo TerraCore funcionando",
        "docs": "/docs"
    }

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        original_name = normalize_filename(file.filename or "imagen.jpg")
        file_bytes = await file.read()

        if not file_bytes:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": "No se recibió archivo"}
            )

        img_bgr = read_image_from_bytes(file_bytes)
        if img_bgr is None:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": "No se pudo leer la imagen"}
            )

        ext = ".jpg"
        lowname = original_name.lower()
        if lowname.endswith(".png"):
            ext = ".png"
        elif lowname.endswith(".webp"):
            ext = ".webp"
        elif lowname.endswith(".jpeg"):
            ext = ".jpeg"

        saved_name = unique_name(ext=ext)
        upload_path = UPLOADS_DIR / saved_name
        save_bytes(upload_path, file_bytes)

        # -------------------------------------------------
        # 1) Intentar OpenAI
        # -------------------------------------------------
        ai_warning = None
        ai_analysis = None

        try:
            ai_analysis = analyze_with_openai(upload_path)
        except Exception as e:
            ai_warning = str(e)
            ai_analysis = fallback_analysis(img_bgr)

        # -------------------------------------------------
        # 2) Dibujar surcos
        # -------------------------------------------------
        result_bgr, drawing_info = process_vineyard_image(img_bgr, ai_analysis)

        result_name = f"resultado_{saved_name.rsplit('.', 1)[0]}.jpeg"
        result_path = OUTPUTS_DIR / result_name

        cv2.imwrite(str(result_path), result_bgr)

        response = {
            "ok": True,
            "archivo_original": saved_name,
            "imagen_original_url": f"/uploads/{saved_name}",
            "imagen_resultado_url": f"/outputs/{result_name}",
            "analisis": ai_analysis,
            "dibujo": drawing_info
        }

        if ai_warning:
            response["aviso_ia"] = f"IA no disponible o falló, se usó detector local. Detalle: {ai_warning}"

        return JSONResponse(content=response)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": str(e),
                "trace": traceback.format_exc().splitlines()[-12:]
            }
        )
