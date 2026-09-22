import io
import os
import re
import json
import math
import base64
import uuid
from datetime import datetime, timezone, date
from pathlib import Path

import requests
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload


# ============================================================
# TERRACORE — SOLO GEMINI
# - Sin OpenCV
# - Sin OpenAI
# - Gemini decide geometría, slots y Salud
# - PIL solo dibuja resultados
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "terrocore(1) (1).png"

st.set_page_config(
    page_title="TerroCore image AI",
    page_icon="🍷",
    layout="wide",
)


# ============================================================
# IDIOMA
# ============================================================
if "idioma_terrocore" not in st.session_state:
    st.session_state.idioma_terrocore = "ES"


def tr(es, fr):
    return es if st.session_state.idioma_terrocore == "ES" else fr


# ============================================================
# ESTADO
# ============================================================
_STATE_DEFAULTS = {
    "tc_parcela_nombre": "",
    "tc_fecha_captura": date.today(),
    "tc_captura_confirmada": False,
    "tc_inventario_procesado": False,
    "tc_inventario_confirmado": False,
    "tc_inventario_rows": [],
    "tc_inventario_table": None,
    "tc_inventario_image": None,
    "tc_inventario_fuente": "",
    "tc_inventario_confianza": 0.0,
    "tc_salud_procesada": False,
    "tc_salud_result": None,
    "tc_last_provider": "",
}
for _k, _v in _STATE_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def reiniciar_parcela():
    for k, v in _STATE_DEFAULTS.items():
        if k == "tc_parcela_nombre":
            continue
        st.session_state[k] = v


# ============================================================
# DISEÑO TERRACORE
# ============================================================
st.markdown(
    """
    <style>
      :root{
        --wine:#722F37;
        --wine-dark:#4F1E26;
        --cream:#FFF8F4;
        --soft:#F1DADF;
      }
      html,body,[data-testid="stAppViewContainer"],.stApp{
        background:linear-gradient(180deg,#722F37 0%,#642832 100%) !important;
        color:#fff !important;
      }
      [data-testid="stHeader"]{background:transparent !important;}
      .block-container{max-width:1540px;padding-top:.5rem;padding-bottom:2rem;}
      h1,h2,h3,h4,p,label,.stMarkdown,.stCaption,
      [data-testid="stMetricLabel"],[data-testid="stMetricValue"]{color:#fff !important;}
      [data-testid="stVerticalBlockBorderWrapper"]{
        border-color:rgba(255,255,255,.22)!important;
        background:rgba(67,18,28,.16)!important;
        border-radius:12px!important;
      }
      .stButton>button,.stDownloadButton>button{
        background:#FFFDFC!important;color:#722F37!important;
        border:1px solid #F2D5D8!important;border-radius:10px!important;
        font-weight:800!important;min-height:42px!important;
      }
      .stButton>button *,.stDownloadButton>button *{color:#722F37!important;}
      [data-testid="stFileUploaderDropzone"]{
        background:rgba(79,30,38,.38)!important;
        border:1.5px dashed rgba(255,255,255,.7)!important;
        border-radius:10px!important;
      }
      [data-testid="stFileUploaderDropzone"] *{color:#fff!important;}
      [data-testid="stMetric"]{
        background:rgba(86,27,38,.48);border:1px solid rgba(255,255,255,.16);
        border-radius:11px;padding:.8rem .9rem;
      }
      [data-testid="stAlert"]{
        background:rgba(255,255,255,.12)!important;
        border:1px solid rgba(255,255,255,.23)!important;
      }
      [data-testid="stAlert"] *{color:#fff!important;}
      .tc-flow{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin:.35rem 0 1rem 0;}
      .tc-step{padding:9px 6px;border:1px solid rgba(255,255,255,.25);border-radius:9px;text-align:center;font-weight:800;}
      .tc-step.active{background:#FFFDFC;color:#722F37;}
      .tc-note{font-size:.88rem;color:#F5DADD;margin-top:.3rem;}
      @media(max-width:800px){
        .tc-flow{grid-template-columns:1fr 1fr;}
        .block-container{padding-left:.7rem;padding-right:.7rem;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CABECERA
# ============================================================
logo_col, title_col = st.columns([1.15, 4.85], vertical_alignment="center")
with logo_col:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_container_width=True)
    else:
        st.markdown("# 🍷")
with title_col:
    st.markdown("## TerroCore image AI")
    st.markdown(f"**{tr('Análisis inteligente del viñedo — SOLO GEMINI', 'Analyse intelligente du vignoble — GEMINI UNIQUEMENT')}**")

st.caption(
    tr(
        "Gemini analiza la fotografía. Python/PIL únicamente organiza datos y dibuja la respuesta; no detecta surcos ni Salud.",
        "Gemini analyse la photographie. Python/PIL organise les données et dessine seulement la réponse; il ne détecte ni les rangs ni la santé.",
    )
)


# ============================================================
# SECRETS / GEMINI
# ============================================================
def secret_text(name, default=""):
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return str(os.getenv(name, default)).strip()


def gemini_api_key():
    value = secret_text("GEMINI_API_KEY")
    if value:
        return value
    # Compatibilidad si el usuario lo puso dentro de [gemini]
    try:
        block = st.secrets.get("gemini", {})
        if isinstance(block, dict):
            return str(block.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        pass
    return ""


def gemini_model():
    value = secret_text("GEMINI_VISION_MODEL", "gemini-3.5-flash-lite")
    if value == "gemini-2.5-flash-lite":
        value = "gemini-3.5-flash-lite"
    return value or "gemini-3.5-flash-lite"


def pil_to_jpeg_bytes(img, max_side=2600, quality=94):
    img = img.convert("RGB")
    w, h = img.size
    scale = min(1.0, float(max_side) / float(max(w, h)))
    if scale < 1.0:
        img = img.resize(
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            Image.Resampling.LANCZOS,
        )
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality), optimize=True)
    return buf.getvalue()


def parse_json_response(text):
    text = (text or "").strip()
    if text.startswith("```json"):
        text = text[7:].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    try:
        return json.loads(text)
    except Exception:
        a = text.find("{")
        b = text.rfind("}")
        if a >= 0 and b > a:
            return json.loads(text[a:b + 1])
        raise


def _extract_replacement_model(error_text):
    # Ejemplo del API: "use models/gemini-3.5-flash-lite"
    m = re.search(r"models/(gemini-[A-Za-z0-9._-]+)", str(error_text))
    return m.group(1) if m else ""


def gemini_json(images, prompt, max_tokens=32768, temperature=0.03):
    key = gemini_api_key()
    if not key:
        raise RuntimeError("Falta GEMINI_API_KEY en Streamlit Secrets.")

    parts = [{"text": str(prompt)}]
    for img in images:
        raw = pil_to_jpeg_bytes(img)
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(raw).decode("utf-8"),
            }
        })

    models_to_try = [gemini_model()]
    tried = set()
    last_error = ""

    while models_to_try:
        model = models_to_try.pop(0)
        if not model or model in tried:
            continue
        tried.add(model)

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": float(temperature),
                "maxOutputTokens": int(max_tokens),
            },
        }
        response = requests.post(
            url,
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            json=payload,
            timeout=300,
        )

        if response.status_code >= 400:
            last_error = f"Gemini HTTP {response.status_code}: {response.text[:1400]}"
            replacement = _extract_replacement_model(response.text)
            if response.status_code == 404 and replacement and replacement not in tried:
                models_to_try.append(replacement)
                continue
            raise RuntimeError(last_error)

        body = response.json()
        candidates = body.get("candidates") or []
        if not candidates:
            raise RuntimeError("Gemini no devolvió candidatos: " + json.dumps(body.get("promptFeedback") or {}, ensure_ascii=False))

        rparts = (candidates[0].get("content") or {}).get("parts") or []
        texts = [str(p.get("text", "")) for p in rparts if isinstance(p, dict) and p.get("text")]
        if not texts:
            raise RuntimeError("Gemini no devolvió texto JSON.")

        data = parse_json_response("\n".join(texts))
        if not isinstance(data, dict):
            raise RuntimeError("Gemini no devolvió un objeto JSON.")
        st.session_state.tc_last_provider = f"Gemini ({model})"
        return data

    raise RuntimeError(last_error or "Gemini no respondió.")


# Estado visible sin mostrar la clave
if gemini_api_key():
    st.success(f"✅ Gemini configurado: {gemini_model()}")
else:
    st.error("❌ Falta GEMINI_API_KEY en Streamlit Secrets.")


# ============================================================
# GEOMETRÍA SIMPLE PARA DIBUJAR RESPUESTAS DE GEMINI
# ============================================================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def norm_point(p):
    try:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            return None
        return [float(clamp(float(p[0]), 0, 1000)), float(clamp(float(p[1]), 0, 1000))]
    except Exception:
        return None


def norm_to_px(points, w, h):
    out = []
    for p in points or []:
        q = norm_point(p)
        if q is None:
            continue
        out.append((q[0] / 1000.0 * max(1, w - 1), q[1] / 1000.0 * max(1, h - 1)))
    return out


def polyline_lengths(points):
    if len(points) < 2:
        return [0.0]
    cum = [0.0]
    for a, b in zip(points[:-1], points[1:]):
        cum.append(cum[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    return cum


def point_on_polyline(points, frac):
    if not points:
        return (0.0, 0.0)
    if len(points) == 1:
        return points[0]
    frac = float(clamp(frac, 0.0, 1.0))
    cum = polyline_lengths(points)
    total = cum[-1]
    if total <= 1e-6:
        return points[0]
    target = frac * total
    for i in range(len(points) - 1):
        if cum[i + 1] >= target:
            seg = max(1e-9, cum[i + 1] - cum[i])
            a = (target - cum[i]) / seg
            return (
                points[i][0] * (1 - a) + points[i + 1][0] * a,
                points[i][1] * (1 - a) + points[i + 1][1] * a,
            )
    return points[-1]


def sort_rows(rows):
    def key(r):
        pts = r.get("points_norm", [])
        if not pts:
            return 1e9
        # x medio de la trayectoria; suficiente para parcelas verticales/diagonales.
        xs = [float(p[0]) for p in pts if isinstance(p, (list, tuple)) and len(p) >= 2]
        return sum(xs) / max(1, len(xs))
    out = sorted(rows, key=key)
    for i, r in enumerate(out, 1):
        r["id"] = i
    return out


def sanitize_rows(data):
    rows = data.get("rows", []) if isinstance(data, dict) else []
    clean = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        pts = []
        for p in item.get("points", item.get("points_norm", [])) or []:
            q = norm_point(p)
            if q is not None:
                pts.append(q)
        if len(pts) < 2:
            continue
        try:
            conf = float(item.get("confidence", 0.6) or 0.6)
        except Exception:
            conf = 0.6
        clean.append({
            "id": int(item.get("id", len(clean) + 1) or len(clean) + 1),
            "confidence": float(clamp(conf, 0, 1)),
            "points_norm": pts,
            "slot_count": int(item.get("slot_count", 0) or 0),
            "vacant_indices": [],
        })
    return sort_rows(clean)


def draw_row_guide(pil, rows, labels=True, slot_ticks=False):
    img = pil.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    font = ImageFont.load_default()
    for row in rows:
        rid = int(row.get("id", 0))
        pts = norm_to_px(row.get("points_norm", []), w, h)
        if len(pts) < 2:
            continue
        xy = [(int(x), int(y)) for x, y in pts]
        draw.line(xy, fill=(255, 0, 255), width=max(1, round(min(w, h) / 900)))
        if labels:
            mid = point_on_polyline(pts, 0.5)
            draw.text((int(mid[0]) + 2, int(mid[1]) - 10), f"R{rid:02d}", fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(25, 10, 20))
        if slot_ticks and int(row.get("slot_count", 0) or 0) > 1:
            count = int(row["slot_count"])
            # Solo marcas muy pequeñas para que Gemini localice slots; no se usan en salida final.
            for i in range(count):
                p = point_on_polyline(pts, i / max(1, count - 1))
                x, y = int(p[0]), int(p[1])
                draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(255, 255, 0))
    return img


def crop_bbox_for_rows(pil, rows, margin_factor=1.25):
    w, h = pil.size
    allp = []
    for r in rows:
        allp.extend(norm_to_px(r.get("points_norm", []), w, h))
    if not allp:
        return (0, 0, w, h)
    xs = [p[0] for p in allp]
    ys = [p[1] for p in allp]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    width = max(20.0, x1 - x0)
    height = max(20.0, y1 - y0)
    mx = width * max(0.12, (margin_factor - 1.0) / 2.0) + 20
    my = height * 0.05 + 15
    return (
        int(clamp(x0 - mx, 0, w - 1)),
        int(clamp(y0 - my, 0, h - 1)),
        int(clamp(x1 + mx, 1, w)),
        int(clamp(y1 + my, 1, h)),
    )


# ============================================================
# GEMINI: SURCOS
# ============================================================
def detect_rows_gemini(uploaded_file):
    pil = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")

    prompt1 = """
TERRACORE — DETECCIÓN DE SURCOS EN VIÑEDO.
Analiza exclusivamente la fotografía aérea ORIGINAL.

OBJETIVO:
Detectar TODAS las hileras físicas reales de vid. Una hilera física = una trayectoria continua desde su inicio visible hasta su final visible, aunque tenga huecos, plantas secas o vegetación débil.

REGLAS ESTRICTAS:
- UNA sola trayectoria por hilera física.
- La trayectoria debe ir por el CENTRO de la hilera de vid, nunca por el espacio de suelo entre dos hileras.
- No dibujes dos líneas sobre la misma hilera.
- No inventes líneas por sombras, caminos, bordes, postes, árboles o maleza.
- Incluye hileras secas si su eje físico es reconocible por el patrón regular de la parcela.
- Respeta perspectiva y pequeñas curvas: usa 6 a 12 puntos por hilera.
- Coordenadas [x,y] normalizadas 0..1000 sobre la imagen completa.
- Antes de responder, revisa el patrón de separación entre hileras y comprueba primera y última hilera.
- Devuelve las hileras ordenadas físicamente de izquierda a derecha cuando sea posible.

Devuelve SOLO JSON válido:
{
  "estimated_row_count": 0,
  "coverage_score": 0.0,
  "confidence": 0.0,
  "rows": [
    {"id":1,"confidence":0.0,"points":[[x,y],[x,y],[x,y],[x,y],[x,y],[x,y]]}
  ],
  "notes":""
}
"""
    first = gemini_json([pil], prompt1, temperature=0.02)
    rows1 = sanitize_rows(first)
    if not rows1:
        raise RuntimeError("Gemini no devolvió surcos utilizables.")

    guide = draw_row_guide(pil, rows1, labels=True, slot_ticks=False)
    proposal = [{"id": r["id"], "points": r["points_norm"]} for r in rows1]

    prompt2 = """
TERRACORE — AUDITORÍA FINAL DE SURCOS.
Imagen 1 = fotografía ORIGINAL.
Imagen 2 = propuesta de líneas magenta Rxx generada por una primera revisión de Gemini.

Propuesta actual:
__PROPOSAL__

CORRIGE la propuesta completa y devuelve la geometría FINAL.
Revisa una por una:
1) elimina duplicados;
2) agrega hileras reales omitidas, incluyendo extremos;
3) mueve cualquier línea que esté sobre suelo al CENTRO de la hilera correcta;
4) evita saltos de una hilera a otra;
5) sigue la misma hilera a través de huecos secos;
6) elimina líneas sobre caminos, árboles, bordes, maleza ajena o construcciones;
7) conserva el orden físico;
8) usa 6 a 12 puntos por hilera, coordenadas 0..1000 de la imagen COMPLETA;
9) una hilera física real debe aparecer EXACTAMENTE una vez.

No analices plantas, slots ni Salud.
Devuelve SOLO JSON válido:
{
  "estimated_row_count":0,
  "coverage_score":0.0,
  "confidence":0.0,
  "rows":[{"id":1,"confidence":0.0,"points":[[x,y],[x,y],[x,y],[x,y],[x,y],[x,y]]}],
  "audit_notes":""
}
""".replace("__PROPOSAL__", json.dumps(proposal, ensure_ascii=False, separators=(",", ":")))

    audited = gemini_json([pil, guide], prompt2, temperature=0.01)
    rows2 = sanitize_rows(audited) or rows1

    return {
        "pil": pil,
        "rows": sort_rows(rows2),
        "confidence": float(audited.get("confidence", first.get("confidence", 0.6)) or 0.6),
        "coverage_score": float(audited.get("coverage_score", first.get("coverage_score", 0.6)) or 0.6),
        "notes": audited.get("audit_notes", first.get("notes", "")),
    }


# ============================================================
# GEMINI: SLOTS / INVENTARIO
# ============================================================
def clean_indices(value, count):
    if isinstance(value, str):
        value = re.findall(r"\d+", value)
    if not isinstance(value, (list, tuple)):
        return []
    out, seen = [], set()
    for v in value:
        try:
            i = int(v)
        except Exception:
            continue
        if 1 <= i <= count and i not in seen:
            seen.add(i)
            out.append(i)
    return sorted(out)


def inventory_slots_gemini(pil, rows, batch_size=6):
    final = [dict(r) for r in rows]
    by_id = {int(r["id"]): r for r in final}

    for start in range(0, len(final), batch_size):
        group = final[start:start + batch_size]
        bbox = crop_bbox_for_rows(pil, group, 1.35)
        original = pil.crop(bbox).convert("RGB")
        guide_full = draw_row_guide(pil, group, labels=True, slot_ticks=False)
        guide = guide_full.crop(bbox).convert("RGB")
        ids = [int(r["id"]) for r in group]

        prompt = f"""
TERRACORE INVENTARIO — SOLO GEMINI.
Imagen 1 = recorte ORIGINAL.
Imagen 2 = mismas hileras marcadas en magenta con etiquetas Rxx.
Analiza SOLO estos IDs: {ids}.

Para cada hilera Rxx:
- identifica el inicio y final reales de la zona plantada visible;
- identifica el paso repetitivo entre posiciones de planta;
- slot_count = número TOTAL de posiciones esperadas de planta, incluyendo faltantes;
- vacant_indices = índices 1-based que están CLARAMENTE vacíos;
- una planta seca, amarilla, débil o con poco vigor sigue siendo OCUPADA si físicamente está presente;
- no cuentes postes, sombras, maleza, hojas sueltas ni textura del suelo como plantas;
- usa hileras vecinas solamente para validar la regularidad del espaciamiento;
- no cambies la geometría de los surcos.

Devuelve EXACTAMENTE todos los IDs.
SOLO JSON válido:
{{"rows":[{{"id":1,"slot_count":52,"vacant_indices":[7,19],"confidence":0.0}}]}}
"""
        data = gemini_json([original, guide], prompt, temperature=0.02)
        returned = data.get("rows", []) if isinstance(data, dict) else []
        for item in returned:
            if not isinstance(item, dict):
                continue
            try:
                rid = int(item.get("id"))
                count = int(item.get("slot_count", 0) or 0)
            except Exception:
                continue
            if rid not in by_id or count < 2 or count > 500:
                continue
            row = by_id[rid]
            row["slot_count"] = count
            row["vacant_indices"] = clean_indices(item.get("vacant_indices", []), count)
            try:
                row["slot_confidence"] = float(clamp(float(item.get("confidence", 0.6) or 0.6), 0, 1))
            except Exception:
                row["slot_confidence"] = 0.6

    # Segunda auditoría global de CONTEOS, no geometría.
    summary = [
        {
            "id": int(r["id"]),
            "slot_count": int(r.get("slot_count", 0) or 0),
            "vacant_indices": r.get("vacant_indices", []),
        }
        for r in final
    ]
    guide = draw_row_guide(pil, final, labels=True, slot_ticks=False)
    prompt_audit = f"""
TERRACORE — AUDITORÍA FINAL DE INVENTARIO.
Imagen 1 ORIGINAL, imagen 2 guía de hileras. Conteo preliminar:
{json.dumps(summary, ensure_ascii=False, separators=(',', ':'))}

Revisa SOLO los conteos de slots y faltantes. No cambies geometría ni IDs.
Corrige únicamente cuando el patrón repetitivo visible demuestra que el conteo preliminar está equivocado.
Una planta físicamente presente, aunque seca o débil, NO es vacante.
Devuelve todos los IDs.
SOLO JSON:
{{"rows":[{{"id":1,"slot_count":52,"vacant_indices":[7,19],"confidence":0.0}}]}}
"""
    audit = gemini_json([pil, guide], prompt_audit, temperature=0.01)
    for item in audit.get("rows", []) if isinstance(audit, dict) else []:
        if not isinstance(item, dict):
            continue
        try:
            rid = int(item.get("id"))
            count = int(item.get("slot_count", 0) or 0)
        except Exception:
            continue
        if rid not in by_id or count < 2 or count > 500:
            continue
        by_id[rid]["slot_count"] = count
        by_id[rid]["vacant_indices"] = clean_indices(item.get("vacant_indices", []), count)
        try:
            by_id[rid]["slot_confidence"] = max(
                float(by_id[rid].get("slot_confidence", 0.0) or 0.0),
                float(clamp(float(item.get("confidence", 0.6) or 0.6), 0, 1)),
            )
        except Exception:
            pass

    return final


def draw_inventory(pil, rows):
    base = pil.convert("RGB")
    w, h = base.size
    band_h = max(44, int(round(h * 0.065)))
    canvas = Image.new("RGB", (w, h + band_h), (91, 37, 46))
    canvas.paste(base, (0, band_h))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    table = []
    confs = []

    for row in rows:
        rid = int(row["id"])
        pts = norm_to_px(row.get("points_norm", []), w, h)
        if len(pts) < 2:
            continue
        shifted = [(int(x), int(y + band_h)) for x, y in pts]
        draw.line(shifted, fill=(245, 245, 245), width=max(1, int(round(min(w, h) / 1000))))

        # SOLO ARRIBA
        top = min(pts, key=lambda p: p[1])
        x = int(top[0])
        label = f"{rid:02d}"
        # alternancia vertical ligera solo para evitar superposición
        y = 5 if rid % 2 else 22
        tw = max(12, len(label) * 7)
        tx = int(clamp(x - tw // 2, 1, max(1, w - tw - 1)))
        draw.rounded_rectangle((tx - 2, y - 2, tx + tw + 2, y + 12), radius=2, fill=(41, 19, 28))
        draw.text((tx, y), label, fill=(255, 255, 255), font=font)

        count = int(row.get("slot_count", 0) or 0)
        vacant = clean_indices(row.get("vacant_indices", []), count)
        occupied = max(0, count - len(vacant))
        conf = float(row.get("slot_confidence", row.get("confidence", 0.0)) or 0.0)
        confs.append(conf)
        table.append({
            tr("Surco", "Rang"): label,
            tr("Slots", "Emplacements"): count,
            tr("Ocupados", "Occupés"): occupied,
            tr("Vacíos", "Vides"): len(vacant),
            tr("Confianza", "Confiance"): round(conf * 100, 1),
            tr("Estado", "État"): tr("Validado por Gemini", "Validé par Gemini"),
        })

    return canvas, pd.DataFrame(table), (sum(confs) / len(confs) if confs else 0.0)


def analyze_inventory(uploaded_file):
    base = detect_rows_gemini(uploaded_file)
    rows = inventory_slots_gemini(base["pil"], base["rows"], batch_size=6)
    image, table, slot_conf = draw_inventory(base["pil"], rows)
    if table.empty:
        raise RuntimeError("Gemini no pudo construir una tabla de Inventario.")
    return {
        "pil": base["pil"],
        "rows": rows,
        "image": image,
        "table": table,
        "count": len(rows),
        "confidence": (float(base.get("confidence", 0.0)) + slot_conf) / 2.0,
        "model": st.session_state.tc_last_provider or f"Gemini ({gemini_model()})",
        "notes": base.get("notes", ""),
    }


# ============================================================
# GEMINI: SALUD POR SLOT
# ============================================================
def health_slots_gemini(pil, rows, batch_size=5):
    result = {}
    valid = [r for r in rows if int(r.get("slot_count", 0) or 0) > 1]

    for start in range(0, len(valid), batch_size):
        group = valid[start:start + batch_size]
        bbox = crop_bbox_for_rows(pil, group, 1.45)
        original = pil.crop(bbox).convert("RGB")
        guide_full = draw_row_guide(pil, group, labels=True, slot_ticks=True)
        guide = guide_full.crop(bbox).convert("RGB")
        spec = [
            {
                "id": int(r["id"]),
                "slot_count": int(r["slot_count"]),
                "vacant_indices": clean_indices(r.get("vacant_indices", []), int(r["slot_count"])),
            }
            for r in group
        ]

        prompt = f"""
TERRACORE SALUD — CLASIFICACIÓN POR SLOT, SOLO GEMINI.
Imagen 1 = recorte ORIGINAL.
Imagen 2 = guía de las mismas hileras con Rxx y pequeñas referencias de posiciones.
Inventario confirmado:
{json.dumps(spec, ensure_ascii=False, separators=(',', ':'))}

Para CADA Rxx clasifica los slots que deben considerarse ROJOS.
El resto serán VERDES.

CRITERIO:
- VERDE: vid claramente viva con cobertura/vigor suficiente comparada con plantas sanas cercanas bajo iluminación similar.
- ROJO: slot vacío, planta seca, marrón/beige, amarillenta severa, muy débil, muy rala, interrumpida o con cobertura claramente inferior al patrón sano local.
- Todo vacant_indices del Inventario DEBE estar incluido en red_indices.
- No confundas maleza entre hileras con vid sana.
- No marques rojo únicamente por sombra; compara continuidad y estructura de la misma hilera.
- No cambies slot_count ni geometría.
- Si hay duda fuerte por baja resolución, usa confidence menor, pero no inventes precisión.

Devuelve todos los IDs.
SOLO JSON válido:
{{"rows":[{{"id":1,"red_indices":[7,8,19],"confidence":0.0}}]}}
"""
        data = gemini_json([original, guide], prompt, temperature=0.02)
        by = {}
        for item in data.get("rows", []) if isinstance(data, dict) else []:
            if not isinstance(item, dict):
                continue
            try:
                rid = int(item.get("id"))
            except Exception:
                continue
            row = next((r for r in group if int(r["id"]) == rid), None)
            if not row:
                continue
            count = int(row["slot_count"])
            red = set(clean_indices(item.get("red_indices", []), count))
            red.update(clean_indices(row.get("vacant_indices", []), count))
            try:
                conf = float(clamp(float(item.get("confidence", 0.6) or 0.6), 0, 1))
            except Exception:
                conf = 0.6
            by[rid] = {"red_indices": sorted(red), "confidence": conf}
        for r in group:
            rid = int(r["id"])
            result[rid] = by.get(rid, {
                "red_indices": clean_indices(r.get("vacant_indices", []), int(r["slot_count"])),
                "confidence": 0.35,
            })

    # Segunda auditoría final sobre la fotografía completa.
    guide = draw_row_guide(pil, valid, labels=True, slot_ticks=True)
    summary = [
        {
            "id": int(r["id"]),
            "slot_count": int(r["slot_count"]),
            "vacant_indices": clean_indices(r.get("vacant_indices", []), int(r["slot_count"])),
            "initial_red_indices": result.get(int(r["id"]), {}).get("red_indices", []),
        }
        for r in valid
    ]
    prompt_audit = f"""
TERRACORE SALUD — AUDITORÍA FINAL.
Imagen 1 ORIGINAL completa, imagen 2 guía completa. Clasificación preliminar:
{json.dumps(summary, ensure_ascii=False, separators=(',', ':'))}

Revisa especialmente falsos verdes y falsos rojos.
- Conserva rojo en vacíos físicos y plantas claramente secas/débiles.
- Verde requiere evidencia visible de vid viva suficiente.
- Sombra por sí sola no significa rojo.
- No cambies geometría ni slot_count.
- Todo vacant_indices debe permanecer rojo.

Devuelve SOLO JSON válido:
{{"rows":[{{"id":1,"red_indices":[7,8,19],"confidence":0.0}}]}}
"""
    audit = gemini_json([pil, guide], prompt_audit, temperature=0.01)
    by_row = {int(r["id"]): r for r in valid}
    for item in audit.get("rows", []) if isinstance(audit, dict) else []:
        if not isinstance(item, dict):
            continue
        try:
            rid = int(item.get("id"))
        except Exception:
            continue
        if rid not in by_row:
            continue
        row = by_row[rid]
        count = int(row["slot_count"])
        red = set(clean_indices(item.get("red_indices", []), count))
        red.update(clean_indices(row.get("vacant_indices", []), count))
        result[rid] = {
            "red_indices": sorted(red),
            "confidence": float(clamp(float(item.get("confidence", result.get(rid, {}).get("confidence", 0.6)) or 0.6), 0, 1)),
        }
    return result


def draw_health(pil, rows, health_map):
    img = pil.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    font = ImageFont.load_default()
    green = (25, 225, 55)
    red = (245, 45, 45)
    width = max(2, int(round(min(w, h) / 500)))

    total_slots = 0
    total_red = 0
    total_green = 0
    red_xy = []
    confs = []

    for row in rows:
        rid = int(row["id"])
        count = int(row.get("slot_count", 0) or 0)
        pts = norm_to_px(row.get("points_norm", []), w, h)
        if len(pts) < 2 or count < 2:
            continue
        red_indices = set(health_map.get(rid, {}).get("red_indices", []))
        confs.append(float(health_map.get(rid, {}).get("confidence", 0.0) or 0.0))

        positions = [point_on_polyline(pts, i / max(1, count - 1)) for i in range(count)]
        total_slots += count
        total_red += len(red_indices)
        total_green += max(0, count - len(red_indices))

        # Segmento entre slots: rojo si cualquiera de los extremos es rojo.
        for i in range(count - 1):
            idx1, idx2 = i + 1, i + 2
            p0, p1 = positions[i], positions[i + 1]
            is_red = idx1 in red_indices or idx2 in red_indices
            draw.line((int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1])), fill=red if is_red else green, width=width)

        for idx in red_indices:
            if 1 <= idx <= len(positions):
                red_xy.append(positions[idx - 1])

        # número SOLO ARRIBA
        top = min(pts, key=lambda p: p[1])
        label = f"{rid:02d}"
        x, y = int(top[0]), int(top[1])
        draw.text((x - 7, max(0, y - 13)), label, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(35, 20, 25))

    green_pct = 100.0 * total_green / total_slots if total_slots else 0.0
    red_pct = 100.0 * total_red / total_slots if total_slots else 0.0

    return {
        "annotated": img,
        "green_pct": green_pct,
        "red_pct": red_pct,
        "green_slots": total_green,
        "red_slots": total_red,
        "total_slots": total_slots,
        "red_xy": red_xy,
        "confidence": sum(confs) / len(confs) if confs else 0.0,
    }


def health_diagnosis_gemini(pil, visual, row_count):
    prompt = f"""
TERRACORE — DIAGNÓSTICO VISUAL PRELIMINAR.
Analiza esta fotografía aérea y este resumen ya calculado por slots:
- surcos: {row_count}
- slots totales: {visual['total_slots']}
- slots verdes: {visual['green_slots']} ({visual['green_pct']:.1f}%)
- slots rojos: {visual['red_slots']} ({visual['red_pct']:.1f}%)

No diagnostiques una enfermedad o nutriente específico solo por fotografía.
Describe la zona visualmente más afectada de forma espacial (izquierda/centro/derecha y superior/media/inferior cuando sea útil).
Devuelve SOLO JSON válido:
{{
 "zona_mas_afectada":"texto corto",
 "nivel_afectacion_visual":"bajo|medio|alto",
 "diagnostico_visual":"1 a 3 frases",
 "causas_probables":["..."],
 "recomendaciones_iniciales":["..."],
 "nota_diagnostico":"Diagnóstico visual preliminar; confirmar en campo."
}}
"""
    try:
        return gemini_json([pil], prompt, temperature=0.03)
    except Exception:
        return {}


def analyze_health(uploaded_file, rows):
    pil = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
    health_map = health_slots_gemini(pil, rows, batch_size=5)
    visual = draw_health(pil, rows, health_map)
    diag = health_diagnosis_gemini(pil, visual, len(rows))
    return {
        "annotated": visual["annotated"],
        "green_pct": float(visual["green_pct"]),
        "red_pct": float(visual["red_pct"]),
        "green_slots": int(visual["green_slots"]),
        "red_slots": int(visual["red_slots"]),
        "total_slots": int(visual["total_slots"]),
        "confidence": float(visual["confidence"]),
        "zona_mas_afectada": str(diag.get("zona_mas_afectada", "No determinada")),
        "nivel_afectacion_visual": str(diag.get("nivel_afectacion_visual", "No determinado")),
        "diagnostico_visual": str(diag.get("diagnostico_visual", "")),
        "causas_probables": diag.get("causas_probables", []) or [],
        "recomendaciones_iniciales": diag.get("recomendaciones_iniciales", []) or [],
        "nota_diagnostico": str(diag.get("nota_diagnostico", "Diagnóstico visual preliminar; confirmar en campo.")),
        "health_map": health_map,
    }


# ============================================================
# GOOGLE DRIVE / SHEETS (OPCIONAL)
# ============================================================
HISTORIAL_SHEET_NAME = "HistorialTerroCore"
HISTORIAL_HEADERS = [
    "ID", "Fecha", "Nombre", "ImagenOriginalFileID", "ImagenProcesadaFileID",
    "Surcos", "VerdePct", "RojoPct", "AmarilloPct", "NivelVisual",
    "ZonaMasAfectada", "DiagnosticoVisual", "CausasProbables",
    "ExplicacionNutrientes", "Recomendaciones", "NotaDiagnostico",
]


def service_account_info():
    try:
        if "gcp_service_account" in st.secrets:
            return dict(st.secrets["gcp_service_account"])
    except Exception:
        pass
    raw = secret_text("GCP_SERVICE_ACCOUNT_JSON")
    if raw:
        return json.loads(raw)
    return {}


def google_history_configured():
    return bool(service_account_info() and secret_text("GDRIVE_PARENT_FOLDER_ID") and secret_text("GSHEET_ID"))


@st.cache_resource(show_spinner=False)
def google_clients_cached(service_json, parent_id, sheet_id):
    info = json.loads(service_json)
    scopes = [
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/spreadsheets",
    ]
    cred = service_account.Credentials.from_service_account_info(info, scopes=scopes)
    drive = build("drive", "v3", credentials=cred, cache_discovery=False)
    sheets = build("sheets", "v4", credentials=cred, cache_discovery=False)
    return drive, sheets


def google_clients():
    info = service_account_info()
    if not info:
        raise RuntimeError("Falta la cuenta de servicio de Google.")
    parent = secret_text("GDRIVE_PARENT_FOLDER_ID")
    sheet_id = secret_text("GSHEET_ID")
    if not parent or not sheet_id:
        raise RuntimeError("Falta GDRIVE_PARENT_FOLDER_ID o GSHEET_ID.")
    return google_clients_cached(json.dumps(info, sort_keys=True), parent, sheet_id)


def drive_subfolder(name):
    drive, _ = google_clients()
    parent = secret_text("GDRIVE_PARENT_FOLDER_ID")
    safe = str(name).replace("'", "\\'")
    q = f"name = '{safe}' and '{parent}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    found = drive.files().list(q=q, spaces="drive", fields="files(id,name)", pageSize=10, supportsAllDrives=True, includeItemsFromAllDrives=True).execute().get("files", [])
    if found:
        return found[0]["id"]
    created = drive.files().create(
        body={"name": name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent]},
        fields="id", supportsAllDrives=True,
    ).execute()
    return created["id"]


def upload_drive_bytes(content, name, mime, subfolder):
    drive, _ = google_clients()
    folder_id = drive_subfolder(subfolder)
    media = MediaIoBaseUpload(io.BytesIO(content), mimetype=mime or "application/octet-stream", resumable=False)
    created = drive.files().create(
        body={"name": name, "parents": [folder_id]},
        media_body=media, fields="id,name", supportsAllDrives=True,
    ).execute()
    return created["id"]


def ensure_history_sheet():
    _, sheets = google_clients()
    sid = secret_text("GSHEET_ID")
    meta = sheets.spreadsheets().get(spreadsheetId=sid, fields="sheets.properties").execute()
    names = [s.get("properties", {}).get("title", "") for s in meta.get("sheets", [])]
    if HISTORIAL_SHEET_NAME not in names:
        sheets.spreadsheets().batchUpdate(
            spreadsheetId=sid,
            body={"requests": [{"addSheet": {"properties": {"title": HISTORIAL_SHEET_NAME}}}]},
        ).execute()
    current = sheets.spreadsheets().values().get(spreadsheetId=sid, range=f"{HISTORIAL_SHEET_NAME}!A1:P1").execute().get("values", [])
    if not current or current[0] != HISTORIAL_HEADERS:
        sheets.spreadsheets().values().update(
            spreadsheetId=sid, range=f"{HISTORIAL_SHEET_NAME}!A1:P1",
            valueInputOption="RAW", body={"values": [HISTORIAL_HEADERS]},
        ).execute()


def append_history(record):
    _, sheets = google_clients()
    ensure_history_sheet()
    sid = secret_text("GSHEET_ID")
    row = [[
        record.get("id", ""), record.get("fecha", ""), record.get("nombre", ""),
        record.get("imagen_original_file_id", ""), record.get("imagen_procesada_file_id", ""),
        int(record.get("surcos", 0) or 0), float(record.get("verde_pct", 0.0) or 0.0),
        float(record.get("rojo_pct", 0.0) or 0.0), 0.0,
        record.get("nivel_visual", ""), record.get("zona_mas_afectada", ""),
        record.get("diagnostico_visual", ""), json.dumps(record.get("causas_probables", []), ensure_ascii=False),
        "", json.dumps(record.get("recomendaciones", []), ensure_ascii=False),
        record.get("nota_diagnostico", ""),
    ]]
    sheets.spreadsheets().values().append(
        spreadsheetId=sid, range=f"{HISTORIAL_SHEET_NAME}!A:P",
        valueInputOption="RAW", insertDataOption="INSERT_ROWS", body={"values": row},
    ).execute()


def pil_png_bytes(img):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def save_health_to_google(uploaded, health, rows):
    if not google_history_configured():
        return False, "Historial de Google no configurado."
    original_id = upload_drive_bytes(uploaded.getvalue(), uploaded.name, uploaded.type or "image/jpeg", "Originales")
    processed_name = f"{Path(uploaded.name).stem}_salud_gemini.png"
    processed_id = upload_drive_bytes(pil_png_bytes(health["annotated"]), processed_name, "image/png", "Procesadas")
    record = {
        "id": str(uuid.uuid4()),
        "fecha": datetime.now(timezone.utc).isoformat(),
        "nombre": uploaded.name,
        "imagen_original_file_id": original_id,
        "imagen_procesada_file_id": processed_id,
        "surcos": len(rows),
        "verde_pct": health["green_pct"],
        "rojo_pct": health["red_pct"],
        "nivel_visual": health.get("nivel_afectacion_visual", ""),
        "zona_mas_afectada": health.get("zona_mas_afectada", ""),
        "diagnostico_visual": health.get("diagnostico_visual", ""),
        "causas_probables": health.get("causas_probables", []),
        "recomendaciones": health.get("recomendaciones_iniciales", []),
        "nota_diagnostico": health.get("nota_diagnostico", ""),
    }
    append_history(record)
    return True, record


# ============================================================
# FLUJO
# ============================================================
st.markdown(
    f"""
    <div class="tc-flow">
      <div class="tc-step active">① {tr('Captura','Capture')}</div>
      <div class="tc-step {'active' if st.session_state.tc_captura_confirmada else ''}">② {tr('Inventario','Inventaire')}</div>
      <div class="tc-step {'active' if st.session_state.tc_inventario_procesado else ''}">③ {tr('Validación','Validation')}</div>
      <div class="tc-step {'active' if st.session_state.tc_inventario_confirmado else ''}">④ {tr('Salud','Santé')}</div>
      <div class="tc-step {'active' if st.session_state.tc_salud_procesada else ''}">⑤ {tr('Reporte','Rapport')}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

side_col, main_col = st.columns([1.0, 2.2], gap="medium")

with side_col:
    es_col, fr_col = st.columns(2)
    with es_col:
        if st.button("🇪🇸 ES", use_container_width=True, disabled=st.session_state.idioma_terrocore == "ES"):
            st.session_state.idioma_terrocore = "ES"
            st.rerun()
    with fr_col:
        if st.button("🇫🇷 FR", use_container_width=True, disabled=st.session_state.idioma_terrocore == "FR"):
            st.session_state.idioma_terrocore = "FR"
            st.rerun()

    with st.container(border=True):
        st.markdown(tr("### 1. Parcela", "### 1. Parcelle"))
        parcela = st.text_input(
            tr("Nombre de la parcela", "Nom de la parcelle"),
            value=st.session_state.tc_parcela_nombre,
            placeholder=tr("Ej. Parcela 01", "Ex. Parcelle 01"),
        )
        st.session_state.tc_parcela_nombre = parcela
        fecha = st.date_input(tr("Fecha", "Date"), value=st.session_state.tc_fecha_captura)
        st.session_state.tc_fecha_captura = fecha

    if st.button(tr("🔄 Nueva parcela / Nuevo análisis", "🔄 Nouvelle parcelle / Nouvelle analyse"), use_container_width=True):
        reiniciar_parcela()
        st.rerun()

with main_col:
    with st.container(border=True):
        st.markdown(tr("### 2. Captura Base", "### 2. Capture de base"))
        uploaded_images = st.file_uploader(
            tr("Sube una o varias fotografías de la misma parcela", "Téléversez une ou plusieurs photos de la même parcelle"),
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
        )
        if uploaded_images:
            st.caption(tr(f"Imágenes cargadas: {len(uploaded_images)}", f"Images chargées : {len(uploaded_images)}"))
            preview_cols = st.columns(min(3, len(uploaded_images)))
            for i, up in enumerate(uploaded_images[:3]):
                with preview_cols[i % len(preview_cols)]:
                    st.image(Image.open(io.BytesIO(up.getvalue())).convert("RGB"), caption=up.name, use_container_width=True)

        if st.button(tr("✅ Confirmar Captura Base", "✅ Confirmer la capture"), use_container_width=True, disabled=not uploaded_images):
            st.session_state.tc_captura_confirmada = True
            st.session_state.tc_inventario_procesado = False
            st.session_state.tc_inventario_confirmado = False
            st.session_state.tc_salud_procesada = False
            st.rerun()

    if st.session_state.tc_captura_confirmada and uploaded_images:
        with st.container(border=True):
            st.markdown(tr("### 3. Inventario", "### 3. Inventaire"))
            st.caption(tr(
                "Gemini detecta los surcos, los audita y cuenta slots. La foto final muestra líneas finas y números únicamente arriba.",
                "Gemini détecte et audite les rangs puis compte les emplacements. L’image finale montre des lignes fines et des numéros uniquement en haut.",
            ))

            if st.button(tr("🌿 Analizar Inventario con Gemini", "🌿 Analyser l’inventaire avec Gemini"), use_container_width=True):
                progress = st.progress(5, text=tr("Gemini está detectando y auditando los surcos...", "Gemini détecte et audite les rangs..."))
                try:
                    candidates = []
                    errors = []
                    for up in uploaded_images:
                        try:
                            inv = analyze_inventory(up)
                            score = float(inv.get("confidence", 0.0))
                            candidates.append((score, up, inv))
                        except Exception as exc:
                            errors.append(f"{up.name}: {exc}")
                    if not candidates:
                        raise RuntimeError(" | ".join(errors) if errors else "Gemini no pudo analizar las imágenes.")
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    _, best_up, inv = candidates[0]
                    progress.progress(95, text=tr("Gemini terminó los slots...", "Gemini a terminé les emplacements..."))

                    st.session_state.tc_inventario_rows = inv["rows"]
                    st.session_state.tc_inventario_table = inv["table"]
                    st.session_state.tc_inventario_image = inv["image"]
                    st.session_state.tc_inventario_fuente = best_up.name
                    st.session_state.tc_inventario_confianza = float(inv.get("confidence", 0.0))
                    st.session_state.tc_inventario_procesado = True
                    st.session_state.tc_inventario_confirmado = False
                    st.session_state.tc_salud_procesada = False
                    progress.progress(100, text=tr("Inventario terminado.", "Inventaire terminé."))
                    st.rerun()
                except Exception as exc:
                    st.error(tr(f"No se pudo terminar el Inventario con Gemini: {exc}", f"Impossible de terminer l’inventaire avec Gemini : {exc}"))

    if st.session_state.tc_inventario_procesado:
        with st.container(border=True):
            st.markdown(tr("### Resultado de Inventario", "### Résultat de l’inventaire"))
            table = st.session_state.tc_inventario_table.copy()
            total_surcos = len(table)
            total_slots = int(pd.to_numeric(table[tr("Slots", "Emplacements")], errors="coerce").fillna(0).sum())
            total_occ = int(pd.to_numeric(table[tr("Ocupados", "Occupés")], errors="coerce").fillna(0).sum())
            total_empty = int(pd.to_numeric(table[tr("Vacíos", "Vides")], errors="coerce").fillna(0).sum())
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(tr("Surcos", "Rangs"), total_surcos)
            m2.metric(tr("Slots totales", "Emplacements totaux"), total_slots)
            m3.metric(tr("Ocupados", "Occupés"), total_occ)
            m4.metric(tr("Vacíos", "Vides"), total_empty)

            if st.session_state.tc_inventario_image is not None:
                st.caption(tr("Los slots se calculan en la tabla, pero no se dibujan sobre la fotografía.", "Les emplacements sont calculés dans le tableau mais ne sont pas dessinés sur la photo."))
                st.image(st.session_state.tc_inventario_image, use_container_width=True)

            edited = st.data_editor(
                table,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                key="tc_inventory_editor_gemini_only",
                column_config={
                    tr("Surco", "Rang"): st.column_config.TextColumn(disabled=True),
                    tr("Slots", "Emplacements"): st.column_config.NumberColumn(min_value=0, step=1),
                    tr("Ocupados", "Occupés"): st.column_config.NumberColumn(min_value=0, step=1),
                    tr("Vacíos", "Vides"): st.column_config.NumberColumn(min_value=0, step=1),
                    tr("Confianza", "Confiance"): st.column_config.NumberColumn(disabled=True, format="%.1f %%"),
                    tr("Estado", "État"): st.column_config.TextColumn(disabled=True),
                },
            )
            st.session_state.tc_inventario_table = edited

            valid = True
            for _, rr in edited.iterrows():
                if int(rr[tr("Slots", "Emplacements")]) != int(rr[tr("Ocupados", "Occupés")]) + int(rr[tr("Vacíos", "Vides")]):
                    valid = False
                    break
            if valid:
                st.success(tr("✅ Slots = Ocupados + Vacíos en todos los surcos.", "✅ Emplacements = Occupés + Vides pour tous les rangs."))
            else:
                st.error(tr("Corrige la tabla antes de confirmar.", "Corrigez le tableau avant de confirmer."))

            if st.button(tr("✅ Confirmar Inventario", "✅ Confirmer l’inventaire"), use_container_width=True, disabled=not valid):
                # Sincroniza posibles correcciones manuales de conteos con rows.
                rows = st.session_state.tc_inventario_rows
                col_s = tr("Surco", "Rang")
                col_slots = tr("Slots", "Emplacements")
                col_empty = tr("Vacíos", "Vides")
                for _, rr in edited.iterrows():
                    try:
                        rid = int(str(rr[col_s]))
                    except Exception:
                        continue
                    row = next((r for r in rows if int(r["id"]) == rid), None)
                    if row:
                        new_count = int(rr[col_slots])
                        # Si usuario cambia solo totales, conserva vacantes previos que sigan en rango.
                        row["slot_count"] = new_count
                        row["vacant_indices"] = clean_indices(row.get("vacant_indices", []), new_count)
                st.session_state.tc_inventario_rows = rows
                st.session_state.tc_inventario_confirmado = True
                st.session_state.tc_salud_procesada = False
                st.rerun()

    if st.session_state.tc_inventario_confirmado and uploaded_images:
        with st.container(border=True):
            st.markdown(tr("### 4. Salud", "### 4. Santé"))
            st.caption(tr(
                "Gemini reutiliza exactamente los surcos y slots del Inventario. Verde/Rojo se calcula por slots, no por longitud de líneas.",
                "Gemini réutilise exactement les rangs et emplacements de l’inventaire. Vert/Rouge est calculé par emplacements, pas par longueur de lignes.",
            ))
            if st.button(tr("🩺 Analizar Salud con Gemini", "🩺 Analyser la santé avec Gemini"), use_container_width=True):
                try:
                    fuente = st.session_state.tc_inventario_fuente
                    up = next((u for u in uploaded_images if u.name == fuente), uploaded_images[0])
                    with st.spinner(tr("Gemini está revisando la Salud slot por slot...", "Gemini analyse la santé emplacement par emplacement...")):
                        health = analyze_health(up, st.session_state.tc_inventario_rows)
                    st.session_state.tc_salud_result = health
                    st.session_state.tc_salud_procesada = True
                    st.rerun()
                except Exception as exc:
                    st.error(tr(f"No se pudo analizar Salud con Gemini: {exc}", f"Impossible d’analyser la santé avec Gemini : {exc}"))

    if st.session_state.tc_salud_procesada and st.session_state.tc_salud_result:
        health = st.session_state.tc_salud_result
        with st.container(border=True):
            st.markdown(tr("### Diagnóstico de Salud", "### Diagnostic de santé"))
            a, b = st.columns(2)
            a.metric(tr("Vegetación verde", "Végétation verte"), f"{health['green_pct']:.1f}%")
            b.metric(tr("Afectación roja", "Affectation rouge"), f"{health['red_pct']:.1f}%")
            st.caption(tr(
                f"Calculado por slots: {health['green_slots']} verdes + {health['red_slots']} rojos = {health['total_slots']} posiciones evaluadas.",
                f"Calcul par emplacements : {health['green_slots']} verts + {health['red_slots']} rouges = {health['total_slots']} positions évaluées.",
            ))

            fuente = st.session_state.tc_inventario_fuente
            up = next((u for u in uploaded_images if u.name == fuente), uploaded_images[0])
            c1, c2 = st.columns(2)
            with c1:
                st.caption(tr("Imagen original", "Image originale"))
                st.image(Image.open(io.BytesIO(up.getvalue())).convert("RGB"), use_container_width=True)
            with c2:
                st.caption(tr("Imagen procesada — Salud", "Image traitée — Santé"))
                st.image(health["annotated"], use_container_width=True)

            d1, d2, d3 = st.columns(3)
            d1.metric(tr("Verde", "Vert"), f"{health['green_pct']:.1f}%")
            d2.metric(tr("Rojo", "Rouge"), f"{health['red_pct']:.1f}%")
            d3.metric(tr("Zona más afectada", "Zone la plus touchée"), health.get("zona_mas_afectada", "No determinada"))

            if health.get("diagnostico_visual"):
                st.markdown(tr("**Diagnóstico visual**", "**Diagnostic visuel**"))
                st.write(health["diagnostico_visual"])
            if health.get("causas_probables"):
                st.markdown(tr("**Causas probables a revisar**", "**Causes probables à vérifier**"))
                for x in health["causas_probables"]:
                    st.markdown(f"- {x}")
            if health.get("recomendaciones_iniciales"):
                st.markdown(tr("**Recomendaciones iniciales**", "**Recommandations initiales**"))
                for x in health["recomendaciones_iniciales"]:
                    st.markdown(f"- {x}")

            # Reporte CSV
            report_df = st.session_state.tc_inventario_table.copy()
            report_df[tr("Parcela", "Parcelle")] = st.session_state.tc_parcela_nombre
            report_df[tr("Fecha", "Date")] = str(st.session_state.tc_fecha_captura)
            csv_bytes = report_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                tr("⬇️ Descargar reporte CSV", "⬇️ Télécharger le rapport CSV"),
                data=csv_bytes,
                file_name=f"TerroCore_{st.session_state.tc_parcela_nombre or 'parcela'}_{st.session_state.tc_fecha_captura}.csv",
                mime="text/csv",
                use_container_width=True,
            )

            if google_history_configured():
                if st.button(tr("💾 Guardar análisis en Google Drive / Sheets", "💾 Enregistrer dans Google Drive / Sheets"), use_container_width=True):
                    try:
                        ok, rec = save_health_to_google(up, health, st.session_state.tc_inventario_rows)
                        if ok:
                            st.success(tr("✅ Análisis guardado en Google.", "✅ Analyse enregistrée dans Google."))
                        else:
                            st.warning(str(rec))
                    except Exception as exc:
                        st.error(tr(f"No se pudo guardar en Google: {exc}", f"Impossible d’enregistrer dans Google : {exc}"))
            else:
                st.caption(tr("Historial Google no configurado; el análisis funciona de todos modos.", "Historique Google non configuré; l’analyse fonctionne quand même."))

st.markdown("---")
st.caption(
    tr(
        "Motor visual: Gemini únicamente. PIL se usa para dibujar las coordenadas devueltas por Gemini y para preparar las imágenes.",
        "Moteur visuel : Gemini uniquement. PIL sert uniquement à dessiner les coordonnées renvoyées par Gemini et à préparer les images.",
    )
)
