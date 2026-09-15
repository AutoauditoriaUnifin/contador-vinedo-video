import os
import io
import re
import json
import math
import zipfile
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy.signal import find_peaks, savgol_filter

# IA (opcional)
try:
    from openai import OpenAI
    OPENAI_OK = True
except Exception:
    OPENAI_OK = False


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="TerroCore image AI", layout="wide")

APP_BG = "#722F37"
CARD_BG = "#7A2F39"
CARD_BORDER = "rgba(255,255,255,0.18)"

VIDEO_SAMPLE_SECONDS = 20        # cada cuántos segundos tomar una escena
VIDEO_MIN_DIFF = 4.0             # diferencia mínima entre escenas
MAX_IMAGE_SIDE_AI = 900
MAX_IMAGE_SIDE_ANALYSIS = 1600

LINE_THICKNESS_GREEN = 2
LINE_THICKNESS_RED = 2
ROW_NUMBER_FONT = 0.45
ROW_NUMBER_THICKNESS = 1

USE_AI_BY_DEFAULT = True


# =========================================================
# TEXTO ES / FR
# =========================================================
TXT = {
    "es": {
        "title": "TerroCore image AI",
        "subtitle": "Análisis inteligente del viñedo",
        "intro": "Analiza video e imágenes del viñedo.",
        "summary": "Resumen del análisis",
        "duration": "Duración",
        "images_count": "Imágenes",
        "rows_sum": "Surcos sumados",
        "summary_note": "Resumen calculado con las escenas que conservas.",
        "download_excel": "⬇ Descargar resultados (Excel)",
        "download_zip": "⬇ Descargar imágenes (ZIP)",
        "lang_es": "ES Español",
        "lang_fr": "FR Français",
        "test_ai": "🧠 Probar conexión IA",
        "type_file": "1. Tipo de archivo",
        "video": "🎥 Video",
        "images": "🖼️ Imágenes",
        "upload_video": "2. Subir archivo de video",
        "upload_images": "2. Subir imágenes",
        "video_help": "Haz clic para subir un video o arrástralo aquí",
        "video_formats": "Formatos: MP4, MOV, AVI, M4V",
        "images_help": "Selecciona una o varias fotografías",
        "analyze_video": "▶ Analizar video",
        "analyze_images": "▶ Analizar imágenes",
        "analyze_one_image_ai": "🔎 Analizar 1 imagen con IA",
        "ai_result": "Análisis con IA",
        "completed": "✅ Análisis completado",
        "full_response": "Ver respuesta completa",
        "scene_results": "Resultados por escena / parcela candidata",
        "actions": "Acciones",
        "delete": "🗑 Borrar",
        "analyzed_frames": "Fotogramas analizados",
        "analyzed_image": "Imagen analizada",
        "useful_scenes": "Escenas útiles",
        "total_rows": "Surcos totales",
        "frame": "Escena",
        "time": "Tiempo",
        "rows_est": "Surcos estimados",
        "green_pct": "Verde %",
        "red_pct": "Rojo %",
        "legend_green": "Surco detectado",
        "legend_red": "Tramo con poca vegetación",
        "ai_yes": "Sí",
        "ai_no": "No",
        "vineyard": "¿Es viñedo?",
        "construction": "¿Hay construcción?",
        "path": "¿Hay camino?",
        "roof": "¿Hay techo?",
        "analyze_rows_q": "¿Analizar surcos?",
        "ai_summary": "Resumen",
        "error_ai": "No se pudo consultar la IA. Se usó análisis local.",
        "empty_summary": "El resumen aparecerá aquí después de analizar el video o las imágenes.",
        "processing": "Procesando, espera por favor...",
        "no_results": "Aún no hay resultados.",
        "scene_label": "Escena",
        "image_label": "Imagen",
    },
    "fr": {
        "title": "TerroCore image AI",
        "subtitle": "Analyse intelligente du vignoble",
        "intro": "Analyse la vidéo et les images du vignoble.",
        "summary": "Résumé de l’analyse",
        "duration": "Durée",
        "images_count": "Images",
        "rows_sum": "Rangs cumulés",
        "summary_note": "Résumé calculé avec les scènes conservées.",
        "download_excel": "⬇ Télécharger les résultats (Excel)",
        "download_zip": "⬇ Télécharger les images (ZIP)",
        "lang_es": "ES Espagnol",
        "lang_fr": "FR Français",
        "test_ai": "🧠 Tester la connexion IA",
        "type_file": "1. Type de fichier",
        "video": "🎥 Vidéo",
        "images": "🖼️ Images",
        "upload_video": "2. Télécharger la vidéo",
        "upload_images": "2. Télécharger les images",
        "video_help": "Cliquez pour téléverser une vidéo ou glissez-la ici",
        "video_formats": "Formats : MP4, MOV, AVI, M4V",
        "images_help": "Sélectionnez une ou plusieurs photos",
        "analyze_video": "▶ Analyser la vidéo",
        "analyze_images": "▶ Analyser les images",
        "analyze_one_image_ai": "🔎 Analyser 1 image avec IA",
        "ai_result": "Analyse IA",
        "completed": "✅ Analyse terminée",
        "full_response": "Voir la réponse complète",
        "scene_results": "Résultats par scène / parcelle candidate",
        "actions": "Actions",
        "delete": "🗑 Supprimer",
        "analyzed_frames": "Images analysées",
        "analyzed_image": "Image analysée",
        "useful_scenes": "Scènes utiles",
        "total_rows": "Rangs totaux",
        "frame": "Scène",
        "time": "Temps",
        "rows_est": "Rangs estimés",
        "green_pct": "Vert %",
        "red_pct": "Rouge %",
        "legend_green": "Rang détecté",
        "legend_red": "Zone peu végétalisée",
        "ai_yes": "Oui",
        "ai_no": "Non",
        "vineyard": "Est-ce un vignoble ?",
        "construction": "Y a-t-il une construction ?",
        "path": "Y a-t-il un chemin ?",
        "roof": "Y a-t-il un toit ?",
        "analyze_rows_q": "Analyser les rangs ?",
        "ai_summary": "Résumé",
        "error_ai": "Impossible de consulter l’IA. Analyse locale utilisée.",
        "empty_summary": "Le résumé apparaîtra ici après l’analyse de la vidéo ou des images.",
        "processing": "Traitement en cours, merci de patienter...",
        "no_results": "Pas encore de résultats.",
        "scene_label": "Scène",
        "image_label": "Image",
    }
}


# =========================================================
# SESSION STATE
# =========================================================
if "lang" not in st.session_state:
    st.session_state.lang = "es"

if "mode" not in st.session_state:
    st.session_state.mode = "video"

if "results" not in st.session_state:
    st.session_state.results = []

if "deleted_ids" not in st.session_state:
    st.session_state.deleted_ids = set()

if "ai_debug_result" not in st.session_state:
    st.session_state.ai_debug_result = None


def T(key: str) -> str:
    return TXT[st.session_state.lang].get(key, key)


# =========================================================
# CSS
# =========================================================
st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(180deg, {APP_BG} 0%, #6b2831 100%);
        color: #FCEFEF;
    }}
    .block-container {{
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }}
    h1, h2, h3, h4, h5, h6, p, label, div, span {{
        color: #FFF6F6 !important;
    }}
    .tc-card {{
        background: rgba(255,255,255,0.03);
        border: 1px solid {CARD_BORDER};
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 14px;
    }}
    .tc-mini-card {{
        background: rgba(255,255,255,0.03);
        border: 1px solid {CARD_BORDER};
        border-radius: 14px;
        padding: 14px;
        text-align: left;
        min-height: 90px;
    }}
    .tc-summary {{
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.1;
        margin-top: 6px;
    }}
    .tc-muted {{
        opacity: 0.82;
        font-size: 0.95rem;
    }}
    .legend-wrap {{
        display: flex;
        gap: 16px;
        margin-top: 10px;
        margin-bottom: 8px;
        flex-wrap: wrap;
    }}
    .legend-pill {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        border-radius: 10px;
        border: 1px solid {CARD_BORDER};
        background: rgba(0,0,0,0.12);
        color: white;
        font-weight: 600;
    }}
    .legend-dot {{
        width: 14px;
        height: 14px;
        border-radius: 50%;
        display: inline-block;
    }}
    .stButton > button {{
        width: 100%;
        border-radius: 12px;
        border: none;
        background: #F4ECEB;
        color: #7A1F2E;
        font-weight: 700;
        min-height: 44px;
    }}
    .stDownloadButton > button {{
        width: 100%;
        border-radius: 12px;
        border: none;
        background: #F4ECEB;
        color: #7A1F2E;
        font-weight: 700;
        min-height: 44px;
    }}
    .small-btn button {{
        min-height: 36px !important;
    }}
    .tc-ok {{
        background: rgba(126, 179, 76, 0.25);
        border: 1px solid rgba(126,179,76,0.6);
        border-radius: 12px;
        padding: 10px 14px;
        font-weight: 700;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# UTILIDADES
# =========================================================
def load_logo():
    candidates = [
        "terrocore.png",
        "logo_terrocore.png",
        "terrocore_header.png",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def seconds_to_mmss(sec: float) -> str:
    sec = int(round(sec))
    m = sec // 60
    s = sec % 60
    return f"{m:02d}:{s:02d}"


def safe_json_parse(text: str):
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def resize_keep(image_bgr, max_side=1600):
    h, w = image_bgr.shape[:2]
    mx = max(h, w)
    if mx <= max_side:
        return image_bgr
    scale = max_side / float(mx)
    nw = int(w * scale)
    nh = int(h * scale)
    return cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_AREA)


def bgr_to_rgb(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def encode_image_for_ai(image_bgr):
    img = resize_keep(image_bgr, MAX_IMAGE_SIDE_AI)
    rgb = bgr_to_rgb(img)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    data = buf.getvalue()
    import base64
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("utf-8")


def get_openai_client():
    key = None
    try:
        key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        key = None
    if not key:
        key = os.getenv("OPENAI_API_KEY", "")
    if not key or not OPENAI_OK:
        return None
    try:
        return OpenAI(api_key=key)
    except Exception:
        return None


# =========================================================
# IA OPCIONAL DE ESCENA
# =========================================================
def ask_scene_ai(image_bgr):
    """
    IA opcional:
    - ayuda a clasificar si hay viñedo / techo / camino / construcción
    - NO dibuja las líneas, solo orienta el análisis
    """
    client = get_openai_client()
    if client is None:
        return {
            "ok": False,
            "es_vinedo": True,
            "hay_construccion": False,
            "hay_camino": True,
            "hay_techo": False,
            "analizar_surcos": True,
            "resumen": "Sin conexión IA. Se usó análisis local."
        }

    try:
        image_data = encode_image_for_ai(image_bgr)

        prompt = """
Responde SOLO un JSON válido.
Analiza la imagen agrícola aérea y devuelve:

{
  "es_vinedo": true/false,
  "hay_construccion": true/false,
  "hay_camino": true/false,
  "hay_techo": true/false,
  "analizar_surcos": true/false,
  "resumen": "texto corto"
}

Reglas:
- "es_vinedo": true si claramente se observan hileras de viñedo.
- "hay_construccion": true si hay casas, patios, construcciones o zonas urbanas.
- "hay_camino": true si hay caminos o vialidades visibles.
- "hay_techo": true si se ven techos o azoteas.
- "analizar_surcos": true solo si la imagen principal sí contiene viñedo útil para conteo.
- "resumen": máximo 25 palabras.
"""

        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_data},
                    ],
                }
            ],
            max_output_tokens=220,
        )

        txt = getattr(resp, "output_text", "")
        data = safe_json_parse(txt)
        if not data:
            raise ValueError("No se pudo parsear JSON de la IA")

        return {
            "ok": True,
            "raw": txt,
            "es_vinedo": bool(data.get("es_vinedo", True)),
            "hay_construccion": bool(data.get("hay_construccion", False)),
            "hay_camino": bool(data.get("hay_camino", True)),
            "hay_techo": bool(data.get("hay_techo", False)),
            "analizar_surcos": bool(data.get("analizar_surcos", True)),
            "resumen": str(data.get("resumen", "")).strip(),
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "es_vinedo": True,
            "hay_construccion": False,
            "hay_camino": True,
            "hay_techo": False,
            "analizar_surcos": True,
            "resumen": "Sin conexión IA. Se usó análisis local."
        }


# =========================================================
# MOTOR DE DETECCIÓN DE SURCOS
# =========================================================
def excess_green_u8(bgr):
    b = bgr[:, :, 0].astype(np.float32)
    g = bgr[:, :, 1].astype(np.float32)
    r = bgr[:, :, 2].astype(np.float32)
    exg = 2.0 * g - r - b
    exg = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX)
    return exg.astype(np.uint8)


def build_basic_masks(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    exg = excess_green_u8(bgr)

    # Vegetación
    veg = (
        (((h >= 25) & (h <= 100) & (s >= 35) & (v >= 25)) | (exg > 110))
    ).astype(np.uint8) * 255

    # limpiar máscara
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    veg = cv2.morphologyEx(veg, cv2.MORPH_OPEN, k3)
    veg = cv2.morphologyEx(veg, cv2.MORPH_CLOSE, k5)

    # Caminos / techos / zonas claras grises
    bright_gray = (((s < 50) & (v > 120))).astype(np.uint8) * 255
    bright_gray = cv2.morphologyEx(bright_gray, cv2.MORPH_CLOSE, k5)

    # Copas densas / árboles: mucha vegetación compacta
    dens = cv2.GaussianBlur(veg, (0, 0), 11)
    dense_tree = (dens > 215).astype(np.uint8) * 255

    return {
        "veg": veg,
        "bright_gray": bright_gray,
        "dense_tree": dense_tree,
        "exg": exg,
    }


def dominant_row_angle(veg_mask):
    """
    Devuelve el ángulo dominante de las hileras.
    Ángulo respecto al eje X, en grados [0,180).
    """
    edges = cv2.Canny(veg_mask, 50, 150)
    h, w = veg_mask.shape[:2]
    min_len = max(35, int(min(h, w) * 0.08))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=min_len,
        maxLineGap=15,
    )

    angles = []
    weights = []

    if lines is not None:
        for ln in lines[:, 0]:
            x1, y1, x2, y2 = ln
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length < min_len:
                continue
            ang = (math.degrees(math.atan2(dy, dx)) + 180.0) % 180.0
            angles.append(ang)
            weights.append(length)

    if len(angles) >= 3:
        angles_rad = np.deg2rad(np.array(angles) * 2.0)
        w = np.array(weights, dtype=np.float32)
        mean_angle = 0.5 * np.rad2deg(np.arctan2(np.sum(w * np.sin(angles_rad)),
                                                 np.sum(w * np.cos(angles_rad))))
        if mean_angle < 0:
            mean_angle += 180.0
        return mean_angle

    # Fallback por PCA de pixeles vegetales
    ys, xs = np.where(veg_mask > 0)
    if len(xs) < 30:
        return 90.0
    pts = np.column_stack([xs, ys]).astype(np.float32)
    mean = pts.mean(axis=0)
    pts0 = pts - mean
    cov = np.cov(pts0.T)
    vals, vecs = np.linalg.eig(cov)
    vec = vecs[:, np.argmax(vals)]
    ang = (math.degrees(math.atan2(vec[1], vec[0])) + 180.0) % 180.0
    return ang


def rotate_image(img, angle_deg, interp=cv2.INTER_LINEAR, border_value=0):
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return rotated, M


def inverse_transform_points(points_xy, M):
    """
    points_xy: (N, 2) en coordenadas rotadas
    regresa (N,2) en coordenadas originales
    """
    Mi = cv2.invertAffineTransform(M)
    pts = np.array(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.transform(pts, Mi).reshape(-1, 2)
    return out


def estimate_row_spacing(column_signal):
    peaks, _ = find_peaks(column_signal, distance=5)
    if len(peaks) < 4:
        return 12
    diffs = np.diff(peaks)
    diffs = diffs[(diffs >= 5) & (diffs <= 35)]
    if len(diffs) == 0:
        return 12
    return int(np.median(diffs))


def choose_peak_groups(peaks, spacing):
    """
    Conserva grupos coherentes y evita falsos grupos pequeños en árboles.
    """
    if len(peaks) == 0:
        return []

    groups = []
    cur = [peaks[0]]
    for p1, p2 in zip(peaks[:-1], peaks[1:]):
        if (p2 - p1) <= int(spacing * 2.6):
            cur.append(p2)
        else:
            groups.append(cur)
            cur = [p2]
    groups.append(cur)

    accepted = []
    for g in groups:
        if len(g) < 6:
            continue
        dif = np.diff(g)
        if len(dif) == 0:
            continue
        ratio = np.std(dif) / max(np.mean(dif), 1e-6)
        if ratio < 0.55:
            accepted.append(g)

    if not accepted and groups:
        accepted = sorted(groups, key=lambda x: len(x), reverse=True)[:2]

    return accepted


def trace_one_row(rot_exg, rot_veg, rot_bad, x0, step=4, search_radius=5):
    """
    Traza un surco de arriba a abajo siguiendo la mejor respuesta local,
    evitando brincarse al surco vecino.
    """
    h, w = rot_veg.shape[:2]
    x = float(x0)
    pts = []
    states = []  # green / red / skip

    for y in range(0, h, step):
        y1 = max(0, y - 4)
        y2 = min(h, y + 5)

        xmin = max(0, int(round(x - search_radius)))
        xmax = min(w - 1, int(round(x + search_radius)))
        xs = np.arange(xmin, xmax + 1)
        if len(xs) == 0:
            continue

        best_score = -1e9
        best_x = x

        for xx in xs:
            x1 = max(0, xx - 2)
            x2 = min(w, xx + 3)

            roi_exg = rot_exg[y1:y2, x1:x2]
            roi_veg = rot_veg[y1:y2, x1:x2]
            roi_bad = rot_bad[y1:y2, x1:x2]

            exg_score = float(np.mean(roi_exg)) if roi_exg.size else 0.0
            veg_ratio = float(np.mean(roi_veg > 0)) if roi_veg.size else 0.0
            bad_ratio = float(np.mean(roi_bad > 0)) if roi_bad.size else 0.0

            dist_pen = abs(xx - x) * 2.0
            score = exg_score + (veg_ratio * 120.0) - (bad_ratio * 220.0) - dist_pen

            if score > best_score:
                best_score = score
                best_x = float(xx)

        x = 0.72 * x + 0.28 * best_x

        x1 = max(0, int(round(x)) - 2)
        x2 = min(w, int(round(x)) + 3)
        roi_veg = rot_veg[y1:y2, x1:x2]
        roi_bad = rot_bad[y1:y2, x1:x2]
        veg_ratio = float(np.mean(roi_veg > 0)) if roi_veg.size else 0.0
        bad_ratio = float(np.mean(roi_bad > 0)) if roi_bad.size else 0.0

        pts.append([x, y])

        if bad_ratio > 0.45:
            states.append("skip")
        else:
            if veg_ratio >= 0.22:
                states.append("green")
            elif veg_ratio >= 0.06:
                states.append("red")
            else:
                states.append("skip")

    if len(pts) < 8:
        return None

    pts = np.array(pts, dtype=np.float32)
    ys = pts[:, 1]
    xs = pts[:, 0]

    # suavizar: que no sea muy curva y que quede más recta, pero sí siga el surco
    if len(xs) >= 11:
        window = min(15, len(xs) if len(xs) % 2 == 1 else len(xs) - 1)
        if window >= 5:
            xs = savgol_filter(xs, window_length=window, polyorder=2, mode="interp")
            pts[:, 0] = xs

    green_ratio = states.count("green") / len(states)
    red_ratio = states.count("red") / len(states)
    skip_ratio = states.count("skip") / len(states)

    # rechazar líneas con poca evidencia
    if green_ratio < 0.18:
        return None
    if (green_ratio + red_ratio) < 0.38:
        return None
    if skip_ratio > 0.72:
        return None

    return {
        "points": pts,
        "states": states,
        "green_ratio": green_ratio,
        "red_ratio": red_ratio,
    }


def split_segments(points, states):
    segments = {"green": [], "red": []}
    current_type = None
    current_pts = []

    for p, s in zip(points, states):
        if s not in ("green", "red"):
            if current_type is not None and len(current_pts) >= 2:
                segments[current_type].append(np.array(current_pts, dtype=np.float32))
            current_type = None
            current_pts = []
            continue

        if current_type is None:
            current_type = s
            current_pts = [p]
        elif s == current_type:
            current_pts.append(p)
        else:
            if len(current_pts) >= 2:
                segments[current_type].append(np.array(current_pts, dtype=np.float32))
            current_type = s
            current_pts = [p]

    if current_type is not None and len(current_pts) >= 2:
        segments[current_type].append(np.array(current_pts, dtype=np.float32))

    return segments


def analyze_rows_in_frame(bgr, use_ai=True):
    """
    Motor principal:
    - IA opcional clasifica escena
    - CV local detecta orientación y sigue surcos
    """
    bgr = resize_keep(bgr, MAX_IMAGE_SIDE_ANALYSIS)
    h0, w0 = bgr.shape[:2]

    ai_info = ask_scene_ai(bgr) if use_ai else {
        "ok": False,
        "es_vinedo": True,
        "hay_construccion": False,
        "hay_camino": True,
        "hay_techo": False,
        "analizar_surcos": True,
        "resumen": "Análisis local."
    }

    masks = build_basic_masks(bgr)
    veg = masks["veg"]
    bright_gray = masks["bright_gray"]
    dense_tree = masks["dense_tree"]
    exg = masks["exg"]

    # Máscara "mala": techos/caminos/árbol denso.
    bad = cv2.bitwise_or(bright_gray, dense_tree)

    # Si IA ve construcción/techo, endurecemos exclusión de zonas no vegetales
    if ai_info.get("hay_construccion") or ai_info.get("hay_techo"):
        extra_bad = ((bright_gray > 0) & (veg == 0)).astype(np.uint8) * 255
        bad = cv2.bitwise_or(bad, extra_bad)

    row_angle = dominant_row_angle(veg)
    rot_to_vertical = 90.0 - row_angle

    rot_exg, M = rotate_image(exg, rot_to_vertical, interp=cv2.INTER_LINEAR, border_value=0)
    rot_veg, _ = rotate_image(veg, rot_to_vertical, interp=cv2.INTER_NEAREST, border_value=0)
    rot_bad, _ = rotate_image(bad, rot_to_vertical, interp=cv2.INTER_NEAREST, border_value=255)

    # señal por columnas
    col_signal = np.mean(rot_veg > 0, axis=0).astype(np.float32)

    if np.max(col_signal) <= 0:
        annotated = bgr.copy()
        return {
            "annotated": annotated,
            "rows_count": 0,
            "green_pct": 0.0,
            "red_pct": 0.0,
            "ai_info": ai_info,
            "row_angle": row_angle,
        }

    # suavizado
    col_signal_sm = cv2.GaussianBlur(col_signal.reshape(1, -1), (1, 0), sigmaX=2).flatten()
    spacing = estimate_row_spacing(col_signal_sm)
    spacing = int(np.clip(spacing, 6, 28))

    prom = max(0.015, float(np.percentile(col_signal_sm, 70)) * 0.20)
    peaks, _ = find_peaks(col_signal_sm, distance=max(5, spacing - 2), prominence=prom)

    groups = choose_peak_groups(peaks, spacing)
    selected_peaks = []
    for g in groups:
        selected_peaks.extend(g)
    selected_peaks = sorted(selected_peaks)

    # si no hubo grupos, usa picos más fuertes
    if len(selected_peaks) == 0 and len(peaks) > 0:
        peak_scores = [(p, col_signal_sm[p]) for p in peaks]
        peak_scores = sorted(peak_scores, key=lambda x: x[1], reverse=True)
        selected_peaks = sorted([p for p, _ in peak_scores[:120]])

    # trazar
    traces = []
    for px in selected_peaks:
        tr = trace_one_row(rot_exg, rot_veg, rot_bad, px, step=4, search_radius=max(4, spacing // 2))
        if tr is None:
            continue

        pts = tr["points"]
        segments = split_segments(pts, tr["states"])
        green_segments = segments["green"]
        red_segments = segments["red"]

        # filtrar segmentos pobres
        green_len = sum(len(s) for s in green_segments)
        red_len = sum(len(s) for s in red_segments)
        total_len = green_len + red_len

        if total_len < 12:
            continue

        traces.append({
            "green": green_segments,
            "red": red_segments,
            "score": tr["green_ratio"] - 0.25 * tr["red_ratio"],
            "x_center": float(np.mean(pts[:, 0])),
        })

    # fusionar duplicados por cercanía en X
    traces = sorted(traces, key=lambda x: x["x_center"])
    merged = []
    for tr in traces:
        if not merged:
            merged.append(tr)
            continue
        if abs(tr["x_center"] - merged[-1]["x_center"]) < max(5, spacing * 0.55):
            if tr["score"] > merged[-1]["score"]:
                merged[-1] = tr
        else:
            merged.append(tr)

    # evitar contar de más: límite razonable por ancho y spacing
    max_reasonable = int(w0 / max(spacing, 8)) + 5
    if len(merged) > max_reasonable:
        merged = sorted(merged, key=lambda x: x["score"], reverse=True)[:max_reasonable]
        merged = sorted(merged, key=lambda x: x["x_center"])

    annotated = bgr.copy()

    total_green_drawn = 0
    total_red_drawn = 0

    for idx, tr in enumerate(merged, start=1):
        # dibujar verde
        for seg in tr["green"]:
            if len(seg) < 2:
                continue
            orig = inverse_transform_points(seg, M).astype(np.int32)
            cv2.polylines(
                annotated,
                [orig.reshape(-1, 1, 2)],
                False,
                (40, 255, 40),
                LINE_THICKNESS_GREEN,
                cv2.LINE_AA
            )
            total_green_drawn += len(orig)

        # dibujar rojo
        for seg in tr["red"]:
            if len(seg) < 2:
                continue
            orig = inverse_transform_points(seg, M).astype(np.int32)
            cv2.polylines(
                annotated,
                [orig.reshape(-1, 1, 2)],
                False,
                (30, 50, 255),
                LINE_THICKNESS_RED,
                cv2.LINE_AA
            )
            total_red_drawn += len(orig)

        # número arriba
        first_seg = None
        if len(tr["green"]) > 0:
            first_seg = tr["green"][0]
        elif len(tr["red"]) > 0:
            first_seg = tr["red"][0]

        if first_seg is not None and len(first_seg) >= 1:
            p0 = inverse_transform_points(first_seg[:1], M).astype(np.int32)[0]
            tx = int(p0[0])
            ty = max(16, int(p0[1]) - 8)
            cv2.putText(
                annotated,
                str(idx),
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                ROW_NUMBER_FONT,
                (255, 255, 255),
                ROW_NUMBER_THICKNESS,
                cv2.LINE_AA
            )

    rows_count = len(merged)

    total_draw = max(total_green_drawn + total_red_drawn, 1)
    green_pct = (total_green_drawn / total_draw) * 100.0
    red_pct = (total_red_drawn / total_draw) * 100.0

    return {
        "annotated": annotated,
        "rows_count": rows_count,
        "green_pct": round(green_pct, 1),
        "red_pct": round(red_pct, 1),
        "ai_info": ai_info,
        "row_angle": row_angle,
    }


# =========================================================
# VIDEO
# =========================================================
def extract_candidate_frames_from_video(uploaded_file, every_seconds=20, min_diff=4.0):
    """
    Extrae escenas útiles separadas en tiempo y con diferencia visual mínima.
    """
    suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"
    temp_path = None

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    cap = cv2.VideoCapture(temp_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if frame_count > 0 else 0.0

    step = max(1, int(fps * every_seconds))
    idx = 0
    frames = []
    last_gray = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx % step == 0:
            small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            keep = False
            if last_gray is None:
                keep = True
            else:
                diff = cv2.absdiff(gray, last_gray)
                mean_diff = float(diff.mean())
                if mean_diff >= min_diff:
                    keep = True

            if keep:
                second = idx / fps
                frames.append({
                    "time_sec": second,
                    "frame_bgr": frame.copy()
                })
                last_gray = gray

        idx += 1

    cap.release()
    try:
        os.remove(temp_path)
    except Exception:
        pass

    return frames, duration


# =========================================================
# EXPORTS
# =========================================================
def build_results_dataframe(results):
    rows = []
    for i, r in enumerate(results, start=1):
        if r.get("deleted", False):
            continue
        rows.append({
            T("frame"): i,
            T("time"): r.get("time_text", "-"),
            T("rows_est"): r.get("rows_count", 0),
            T("green_pct"): r.get("green_pct", 0.0),
            T("red_pct"): r.get("red_pct", 0.0),
            "es_vinedo": r.get("ai_info", {}).get("es_vinedo", True),
            "hay_construccion": r.get("ai_info", {}).get("hay_construccion", False),
            "hay_camino": r.get("ai_info", {}).get("hay_camino", False),
            "hay_techo": r.get("ai_info", {}).get("hay_techo", False),
            "analizar_surcos": r.get("ai_info", {}).get("analizar_surcos", True),
            T("ai_summary"): r.get("ai_info", {}).get("resumen", ""),
        })
    return pd.DataFrame(rows)


def build_excel_bytes(df):
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Resultados")
    bio.seek(0)
    return bio.getvalue()


def build_zip_images(results):
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, r in enumerate(results, start=1):
            if r.get("deleted", False):
                continue
            img = r["annotated_rgb"]
            pil = Image.fromarray(img)
            img_bytes = io.BytesIO()
            pil.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            zf.writestr(f"escena_{i:02d}.png", img_bytes.read())
    bio.seek(0)
    return bio.getvalue()


# =========================================================
# RENDER DE BLOQUES
# =========================================================
def render_header():
    logo = load_logo()
    c1, c2 = st.columns([4, 2], vertical_alignment="top")

    with c1:
        cols = st.columns([1.2, 4])
        with cols[0]:
            if logo:
                st.image(logo, width=220)
        with cols[1]:
            st.markdown(f"<h1 style='margin-bottom:0'>{T('title')}</h1>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='font-size:1.2rem;font-weight:600;margin-top:-6px'>{T('subtitle')}</div>",
                unsafe_allow_html=True
            )
            st.write(T("intro"))

    with c2:
        b1, b2 = st.columns(2)
        with b1:
            if st.button(T("lang_es"), key="lang_es_btn"):
                st.session_state.lang = "es"
        with b2:
            if st.button(T("lang_fr"), key="lang_fr_btn"):
                st.session_state.lang = "fr"

        if st.button(T("test_ai"), key="test_ai_button"):
            client = get_openai_client()
            if client is not None:
                st.success("IA conectada correctamente.")
            else:
                st.warning("No se encontró OPENAI_API_KEY o la librería openai.")


def render_summary(results):
    active = [r for r in results if not r.get("deleted", False)]
    total_rows = sum(r.get("rows_count", 0) for r in active)
    useful = len(active)

    # duración
    durations = [r.get("video_duration_sec", None) for r in active if r.get("video_duration_sec") is not None]
    duration_text = seconds_to_mmss(max(durations)) if durations else "—"

    st.markdown(f"<div class='tc-card'><h2>{T('summary')}</h2>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"<div class='tc-mini-card'><div>{T('duration')}</div><div class='tc-summary'>{duration_text}</div></div>",
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"<div class='tc-mini-card'><div>{T('images_count')}</div><div class='tc-summary'>{useful}</div></div>",
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"<div class='tc-mini-card'><div>{T('rows_sum')}</div><div class='tc-summary'>{total_rows}</div></div>",
            unsafe_allow_html=True
        )

    st.markdown(f"<div class='tc-muted' style='margin-top:12px'>{T('summary_note')}</div>", unsafe_allow_html=True)

    df = build_results_dataframe(results)
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        if not df.empty:
            st.download_button(
                T("download_excel"),
                data=build_excel_bytes(df),
                file_name="resultados_terrocore.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        else:
            st.button(T("download_excel"), disabled=True, use_container_width=True)

    with col_dl2:
        if active:
            st.download_button(
                T("download_zip"),
                data=build_zip_images(results),
                file_name="imagenes_terrocore.zip",
                mime="application/zip",
                use_container_width=True
            )
        else:
            st.button(T("download_zip"), disabled=True, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_ai_panel(ai_info, raw_text=None):
    st.markdown(f"<div class='tc-card'><h3>{T('ai_result')}</h3>", unsafe_allow_html=True)
    st.markdown(f"<div class='tc-ok'>{T('completed')}</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    yes = T("ai_yes")
    no = T("ai_no")

    with c1:
        st.markdown(f"<div class='tc-card'><b>{T('vineyard')}</b><br><span style='font-size:2rem'>{yes if ai_info.get('es_vinedo', False) else no}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='tc-card'><b>{T('path')}</b><br><span style='font-size:2rem'>{yes if ai_info.get('hay_camino', False) else no}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='tc-card'><b>{T('roof')}</b><br><span style='font-size:2rem'>{yes if ai_info.get('hay_techo', False) else no}</span></div>", unsafe_allow_html=True)

    with c2:
        st.markdown(f"<div class='tc-card'><b>{T('construction')}</b><br><span style='font-size:2rem'>{yes if ai_info.get('hay_construccion', False) else no}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='tc-card'><b>{T('analyze_rows_q')}</b><br><span style='font-size:2rem'>{yes if ai_info.get('analizar_surcos', False) else no}</span></div>", unsafe_allow_html=True)

    st.markdown(
        f"<p><b>{T('ai_summary')}:</b> {ai_info.get('resumen', '')}</p>",
        unsafe_allow_html=True
    )

    if raw_text:
        with st.expander(T("full_response")):
            st.code(raw_text, language="json")

    st.markdown("</div>", unsafe_allow_html=True)


def render_main_result_image(rgb_img):
    st.markdown(f"<div class='tc-card'><h3>{T('analyzed_image')}</h3>", unsafe_allow_html=True)
    st.image(rgb_img, use_container_width=True)
    st.markdown(
        f"""
        <div class="legend-wrap">
            <div class="legend-pill"><span class="legend-dot" style="background:#28FF28"></span> {T('legend_green')}</div>
            <div class="legend-pill"><span class="legend-dot" style="background:#1E32FF"></span> {T('legend_red')}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_results_table(results):
    active = [r for r in results if not r.get("deleted", False)]
    if not active:
        st.info(T("no_results"))
        return

    st.markdown(f"<div class='tc-card'><h3>{T('scene_results')}</h3>", unsafe_allow_html=True)

    table_rows = []
    for i, r in enumerate(results, start=1):
        if r.get("deleted", False):
            continue
        table_rows.append({
            T("frame"): i,
            T("time"): r.get("time_text", "-"),
            T("rows_est"): r.get("rows_count", 0),
            T("green_pct"): r.get("green_pct", 0.0),
            T("red_pct"): r.get("red_pct", 0.0),
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_gallery(results):
    active_count = sum(0 if r.get("deleted", False) else 1 for r in results)
    if active_count == 0:
        return

    st.markdown(f"<div class='tc-card'><h3>{T('analyzed_frames')}</h3>", unsafe_allow_html=True)

    for i, r in enumerate(results, start=1):
        if r.get("deleted", False):
            continue

        st.image(r["annotated_rgb"], use_container_width=True)
        c1, c2 = st.columns([4, 1])
        with c1:
            label_type = T("scene_label") if r.get("kind") == "video" else T("image_label")
            st.write(f"**{label_type} {i}** - {r.get('time_text', '-')} - {r.get('rows_count', 0)} surcos")
        with c2:
            if st.button(T("delete"), key=f"delete_{i}_{r['uid']}"):
                r["deleted"] = True
                st.rerun()

        st.divider()

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# PROCESAMIENTO DE IMÁGENES
# =========================================================
def read_uploaded_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


def process_images(uploaded_files, use_ai=True):
    out = []
    for idx, uf in enumerate(uploaded_files, start=1):
        img = read_uploaded_image(uf)
        if img is None:
            continue

        res = analyze_rows_in_frame(img, use_ai=use_ai)
        rgb = bgr_to_rgb(res["annotated"])

        out.append({
            "uid": f"img_{idx}_{uf.name}",
            "kind": "image",
            "time_sec": None,
            "time_text": f"{T('image_label')} {idx}",
            "rows_count": int(res["rows_count"]),
            "green_pct": float(res["green_pct"]),
            "red_pct": float(res["red_pct"]),
            "annotated_rgb": rgb,
            "ai_info": res["ai_info"],
            "video_duration_sec": None,
            "deleted": False,
        })
    return out


def process_video(uploaded_file, use_ai=True):
    scenes, duration = extract_candidate_frames_from_video(
        uploaded_file,
        every_seconds=VIDEO_SAMPLE_SECONDS,
        min_diff=VIDEO_MIN_DIFF
    )

    out = []
    for idx, item in enumerate(scenes, start=1):
        frame = item["frame_bgr"]
        sec = item["time_sec"]

        res = analyze_rows_in_frame(frame, use_ai=use_ai)
        rgb = bgr_to_rgb(res["annotated"])

        out.append({
            "uid": f"video_{idx}_{int(sec)}",
            "kind": "video",
            "time_sec": sec,
            "time_text": seconds_to_mmss(sec),
            "rows_count": int(res["rows_count"]),
            "green_pct": float(res["green_pct"]),
            "red_pct": float(res["red_pct"]),
            "annotated_rgb": rgb,
            "ai_info": res["ai_info"],
            "video_duration_sec": duration,
            "deleted": False,
        })
    return out


# =========================================================
# LAYOUT
# =========================================================
render_header()

left, right = st.columns([2.2, 1], vertical_alignment="top")

with left:
    render_summary(st.session_state.results)

with right:
    st.markdown(f"<div class='tc-card'><h3>{T('type_file')}</h3>", unsafe_allow_html=True)
    mode = st.radio(
        "",
        options=["video", "images"],
        index=0 if st.session_state.mode == "video" else 1,
        format_func=lambda x: T("video") if x == "video" else T("images"),
        horizontal=True
    )
    st.session_state.mode = mode
    st.markdown("</div>", unsafe_allow_html=True)

    if mode == "video":
        st.markdown(f"<div class='tc-card'><h3>{T('upload_video')}</h3>", unsafe_allow_html=True)
        video_file = st.file_uploader(
            T("video_help"),
            type=["mp4", "mov", "avi", "m4v"],
            key="video_upload"
        )
        st.caption(T("video_formats"))
        run_video = st.button(T("analyze_video"), key="run_video_btn", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if run_video and video_file is not None:
            with st.spinner(T("processing")):
                st.session_state.results = process_video(video_file, use_ai=USE_AI_BY_DEFAULT)
            st.rerun()

    else:
        st.markdown(f"<div class='tc-card'><h3>{T('upload_images')}</h3>", unsafe_allow_html=True)
        image_files = st.file_uploader(
            T("images_help"),
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="images_upload"
        )

        st.markdown(f"<h4>{T('analyze_one_image_ai')}</h4>", unsafe_allow_html=True)
        single_ai_img = st.file_uploader(
            "",
            type=["jpg", "jpeg", "png", "webp"],
            key="single_ai_upload"
        )
        if st.button(T("analyze_one_image_ai"), key="run_ai_one_btn", use_container_width=True):
            if single_ai_img is not None:
                img = read_uploaded_image(single_ai_img)
                ai = ask_scene_ai(img)
                st.session_state.ai_debug_result = ai
            else:
                st.warning("Sube una imagen primero.")

        if st.session_state.ai_debug_result is not None:
            render_ai_panel(
                st.session_state.ai_debug_result,
                raw_text=json.dumps(st.session_state.ai_debug_result, ensure_ascii=False, indent=2)
            )

        run_images = st.button(T("analyze_images"), key="run_images_btn", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if run_images and image_files:
            with st.spinner(T("processing")):
                st.session_state.results = process_images(image_files, use_ai=USE_AI_BY_DEFAULT)
            st.rerun()

# =========================================================
# RESULTADOS
# =========================================================
results_active = [r for r in st.session_state.results if not r.get("deleted", False)]
if results_active:
    # primero detalle / resumen de tabla
    render_results_table(st.session_state.results)

    # luego primera imagen
    render_main_result_image(results_active[0]["annotated_rgb"])

    # luego galería con borrar
    render_gallery(st.session_state.results)
else:
    st.info(T("empty_summary"))
