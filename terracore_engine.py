"""Motor OpenCV V2 de TerraCore, separado de la interfaz Streamlit.

Las funciones de inventario, salud visual y polígonos se conservan del
archivo app_surcos_VIDEO_V10.py. La interfaz y la persistencia se gestionan
por separado en la integración AppSheet.
"""
import io
import math
import re

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def tr(es, fr):
    """El servicio usa español como idioma base para las etiquetas."""
    return es


def _tc_quitar_nombre_motor_del_analisis(texto):
    """
    Quita referencias a OpenCV SOLO de los textos descriptivos del análisis.
    El campo 'metodo' conserva OpenCV para identificar el motor utilizado.
    """
    t = str(texto or "").strip()
    if not t:
        return ""

    # Resúmenes ES/FR antiguos.
    t = re.sub(r"^OpenCV evaluó\s+", "Se evaluaron ", t, flags=re.I)
    t = re.sub(r"^OpenCV a évalué\s+", "Ont été évaluées ", t, flags=re.I)

    # Referencias directas al motor en explicaciones/notas.
    t = re.sub(r"\bEl análisis OpenCV es visual\b", "El análisis es visual", t, flags=re.I)
    t = re.sub(r"\bL’analyse OpenCV est visuelle\b", "L’analyse est visuelle", t, flags=re.I)
    t = re.sub(r"\bpor OpenCV\b", "", t, flags=re.I)
    t = re.sub(r"\bpar OpenCV\b", "", t, flags=re.I)
    t = re.sub(r"\bOpenCV\b", "", t, flags=re.I)

    # Limpieza de espacios/puntuación que pudiera quedar al retirar el nombre.
    t = re.sub(r"\s+([,.;:])", r"\1", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

def mascara_verde(bgr):
    rgb = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB
    ).astype(np.float32)

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    hsv = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2HSV
    )

    hh, ss, vv = cv2.split(
        hsv
    )

    exg = (
        2.0 * g -
        r -
        b
    )

    ngrdi = (
        (g - r) /
        (g + r + 1e-6)
    )

    mask = (
        (exg > 5.0) &
        (ngrdi > -0.030) &
        (hh >= 18) &
        (hh <= 115) &
        (ss >= 10) &
        (vv >= 18) &
        (g >= r * 0.84) &
        (g >= b * 0.84)
    ).astype(np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones(
            (2, 2),
            np.uint8
        ),
        iterations=1
    )

    return mask

def diferencia_angular_rad(
    a,
    b

def campo_orientacion_local(
    bgr

def detectar_componentes_vinedo(
    bgr,
    tile=32

def estimar_espaciado_local(
    profile

def calcular_rejilla_global_surcos(
    green_mask,
    components

def asignar_track_a_rejilla(
    points,
    grid

def semillas_componente(
    component,
    response

def trazar_direccion(
    seed,
    initial_direction,
    component,
    response,
    theta,
    coherence,
    green,
    spacing,
    occupancy

def trazar_surco_local(
    seed,
    component,
    response,
    theta,
    coherence,
    green,
    spacing,
    occupancy

def estados_color(
    green_scores

def analizar(
    pil_img,
    exclusion_mask=None

def _tc_polyline_distances(points):
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 2:
        return np.array([0.0], dtype=np.float32)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float32)

def _tc_interp_point(points, cumulative, distance):
    pts = np.asarray(points, dtype=np.float32)
    cumulative = np.asarray(cumulative, dtype=np.float32)
    if len(pts) == 0:
        return np.array([0.0, 0.0], dtype=np.float32)
    if len(pts) == 1 or cumulative[-1] <= 1e-6:
        return pts[0].copy()
    d = float(np.clip(distance, 0.0, cumulative[-1]))
    idx = int(np.searchsorted(cumulative, d, side="right") - 1)
    idx = max(0, min(idx, len(pts) - 2))
    d0 = float(cumulative[idx])
    d1 = float(cumulative[idx + 1])
    alpha = 0.0 if d1 <= d0 else (d - d0) / (d1 - d0)
    return (pts[idx] * (1.0 - alpha) + pts[idx + 1] * alpha).astype(np.float32)

def _tc_sample_map_along_track(score_map, points, spacing, step_px=2.0):
    """Muestrea presencia visual a lo largo del eje del surco.

    Importante: score_map proviene de respuesta de textura/estructura, no de
    color verde/rojo. Por eso Inventario queda separado de Salud.
    """
    pts = np.asarray(points, dtype=np.float32)
    cumulative = _tc_polyline_distances(pts)
    length = float(cumulative[-1]) if len(cumulative) else 0.0
    if length < 2.0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), cumulative

    distances = np.arange(0.0, length + 1e-6, max(1.0, float(step_px)), dtype=np.float32)
    h, w = score_map.shape[:2]
    radius = max(2, int(round(float(spacing or 10.0) * 0.16)))
    values = []

    for d in distances:
        p = _tc_interp_point(pts, cumulative, float(d))
        x = int(np.clip(round(float(p[0])), 0, w - 1))
        y = int(np.clip(round(float(p[1])), 0, h - 1))
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        patch = score_map[y0:y1, x0:x1]
        values.append(float(np.mean(patch)) if patch.size else 0.0)

    arr = np.asarray(values, dtype=np.float32)
    if len(arr) >= 5:
        arr = gaussian_filter1d(arr, sigma=1.2, mode="nearest")
    return distances, arr, cumulative

def _tc_estimar_pitch_slots(distances, profile, row_spacing):
    """Estima automáticamente la distancia entre posiciones de planta.

    Primero busca periodicidad real de presencia sobre el surco. Si la foto no
    permite verla claramente, usa una relación geométrica conservadora con la
    separación entre surcos como respaldo.
    """
    if len(distances) < 8 or len(profile) < 8:
        return max(8.0, float(row_spacing or 18.0) * 0.55), 0.35

    step = float(np.median(np.diff(distances))) if len(distances) > 1 else 2.0
    row_spacing = max(8.0, float(row_spacing or 18.0))
    min_pitch = max(7.0, row_spacing * 0.24)
    max_pitch = max(min_pitch + 3.0, row_spacing * 1.10)

    smooth = np.asarray(profile, dtype=np.float32)
    broad_sigma = max(2.0, len(smooth) * 0.045)
    centered = smooth - gaussian_filter1d(smooth, sigma=broad_sigma, mode="nearest")
    centered = centered - float(np.mean(centered))

    energy = float(np.std(centered))
    if energy > 1e-5:
        ac = np.correlate(centered, centered, mode="full")[len(centered)-1:]
        if ac[0] > 1e-8:
            ac = ac / ac[0]
            lag_min = max(2, int(round(min_pitch / step)))
            lag_max = min(len(ac) - 2, int(round(max_pitch / step)))
            if lag_max > lag_min:
                segment = ac[lag_min:lag_max + 1]
                peaks, props = find_peaks(segment, prominence=0.035)
                if len(peaks):
                    peak_vals = segment[peaks]
                    best = int(peaks[int(np.argmax(peak_vals))]) + lag_min
                    corr = float(ac[best])
                    if corr >= 0.08:
                        confidence = float(np.clip(0.45 + corr * 0.55, 0.45, 0.90))
                        return float(best * step), confidence

    fallback = float(np.clip(row_spacing * 0.55, min_pitch, max_pitch))
    return fallback, 0.35

def _tc_score_at_distance(distances, profile, d, window):
    if len(distances) == 0:
        return 0.0
    mask = np.abs(distances - float(d)) <= float(window)
    if np.any(mask):
        return float(np.mean(profile[mask]))
    return float(np.interp(float(d), distances, profile))

def _tc_slots_track(score_map, track, forced_pitch=None):
    points = np.asarray(track.get("points", []), dtype=np.float32)
    spacing = float(track.get("spacing", 18.0) or 18.0)
    distances, profile, cumulative = _tc_sample_map_along_track(
        score_map, points, spacing, step_px=2.0
    )
    if len(distances) < 4:
        return [], 0.0, 0.0

    length = float(distances[-1])
    local_pitch, pitch_conf = _tc_estimar_pitch_slots(distances, profile, spacing)
    pitch = float(forced_pitch or local_pitch)
    pitch = max(6.0, min(pitch, max(7.0, length / 2.0)))

    # Encontrar la fase que mejor coincide con presencia repetitiva real.
    offsets = np.linspace(pitch * 0.25, pitch * 0.95, 12)
    best_offset = pitch * 0.5
    best_score = -1e9
    for off in offsets:
        ds = np.arange(off, max(off + 0.1, length - pitch * 0.15), pitch)
        if len(ds) < 2:
            continue
        scores = [_tc_score_at_distance(distances, profile, d, pitch * 0.18) for d in ds]
        score = float(np.mean(scores)) if scores else -1e9
        if score > best_score:
            best_score = score
            best_offset = float(off)

    slot_distances = np.arange(
        best_offset,
        max(best_offset + 0.1, length - pitch * 0.10),
        pitch,
        dtype=np.float32
    )
    if len(slot_distances) == 0:
        slot_distances = np.array([length * 0.5], dtype=np.float32)

    slot_scores = np.asarray([
        _tc_score_at_distance(distances, profile, float(d), pitch * 0.24)
        for d in slot_distances
    ], dtype=np.float32)

    # Umbral adaptativo de PRESENCIA. No representa salud.
    if len(slot_scores) >= 3 and float(np.max(slot_scores) - np.min(slot_scores)) > 0.012:
        norm = slot_scores - float(np.min(slot_scores))
        denom = float(np.max(norm)) or 1.0
        scaled = np.clip(norm / denom * 255.0, 0, 255).astype(np.uint8)
        otsu_value, _ = cv2.threshold(
            scaled.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        threshold = float(np.min(slot_scores) + (otsu_value / 255.0) * denom)
        # Evitar un corte demasiado exigente cuando el viñedo es homogéneo.
        threshold = min(threshold, float(np.percentile(slot_scores, 48)))
    else:
        threshold = float(np.median(slot_scores) * 0.45) if len(slot_scores) else 0.0

    threshold = max(0.006, threshold)
    occupied = slot_scores >= threshold

    slots = []
    for d, score, is_occ in zip(slot_distances, slot_scores, occupied):
        p = _tc_interp_point(points, cumulative, float(d))
        slots.append({
            "distance": float(d),
            "point": (float(p[0]), float(p[1])),
            "score": float(score),
            "occupied": bool(is_occ),
        })

    contrast = float(np.std(slot_scores) / (np.mean(slot_scores) + 1e-6)) if len(slot_scores) else 0.0
    confidence = float(np.clip(pitch_conf + min(0.25, contrast * 0.20), 0.30, 0.95))
    return slots, pitch, confidence

def _tg_clamp(v, lo, hi):
    try:
        return max(lo, min(hi, float(v)))
    except Exception:
        return lo

def _tg_norm_point(p):
    try:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            return None
        return [
            _tg_clamp(p[0], 0.0, 1000.0),
            _tg_clamp(p[1], 0.0, 1000.0),
        ]
    except Exception:
        return None

def _tg_norm_to_px(points, w, h):
    out = []
    for p in points or []:
        q = _tg_norm_point(p)
        if q is None:
            continue
        out.append((
            q[0] / 1000.0 * max(1, w - 1),
            q[1] / 1000.0 * max(1, h - 1),
        ))
    return out

def _tg_polyline_cum(points):
    if len(points) < 2:
        return [0.0]
    cum = [0.0]
    for a, b in zip(points[:-1], points[1:]):
        cum.append(cum[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    return cum

def _tg_point_on_polyline(points, frac):
    if not points:
        return (0.0, 0.0)
    if len(points) == 1:
        return points[0]
    frac = _tg_clamp(frac, 0.0, 1.0)
    cum = _tg_polyline_cum(points)
    total = cum[-1]
    if total <= 1e-9:
        return points[0]
    target = frac * total
    for i in range(len(points) - 1):
        if cum[i + 1] >= target:
            span = max(1e-9, cum[i + 1] - cum[i])
            a = (target - cum[i]) / span
            return (
                points[i][0] * (1.0 - a) + points[i + 1][0] * a,
                points[i][1] * (1.0 - a) + points[i + 1][1] * a,
            )
    return points[-1]

def _tg_sort_rows(rows):
    if not rows:
        return []

    # Ordenar perpendicularmente a la dirección dominante de las hileras.
    directions = []
    mids = []
    for r in rows:
        pts = r.get("points_norm", [])
        if len(pts) < 2:
            directions.append((0.0, 1.0))
            mids.append((500.0, 500.0))
            continue
        a, b = pts[0], pts[-1]
        dx = float(b[0]) - float(a[0])
        dy = float(b[1]) - float(a[1])
        length = math.hypot(dx, dy) or 1.0
        directions.append((dx / length, dy / length))
        px = sum(float(p[0]) for p in pts) / len(pts)
        py = sum(float(p[1]) for p in pts) / len(pts)
        mids.append((px, py))

    dx = sum(d[0] for d in directions) / max(1, len(directions))
    dy = sum(d[1] for d in directions) / max(1, len(directions))
    norm = math.hypot(dx, dy) or 1.0
    dx, dy = dx / norm, dy / norm
    nx, ny = -dy, dx

    # Orientar normal para que el orden sea visualmente izquierda->derecha
    # o arriba->abajo cuando las filas son horizontales.
    if abs(nx) >= abs(ny):
        if nx < 0:
            nx, ny = -nx, -ny
    else:
        if ny < 0:
            nx, ny = -nx, -ny

    ordered = sorted(
        zip(rows, mids),
        key=lambda rm: rm[1][0] * nx + rm[1][1] * ny,
    )
    out = []
    for i, (row, _) in enumerate(ordered, 1):
        rr = dict(row)
        rr["id"] = i
        out.append(rr)
    return out

def _tg_clean_indices(value, count):
    count = max(0, int(count or 0))
    if count <= 0:
        return []
    if isinstance(value, str):
        value = re.findall(r"\d+", value)
    if not isinstance(value, (list, tuple)):
        return []
    out = []
    seen = set()
    for v in value:
        try:
            i = int(v)
        except Exception:
            continue
        if 1 <= i <= count and i not in seen:
            seen.add(i)
            out.append(i)
    return sorted(out)

def _tg_zone_fallback(pil, red_xy):
    if not red_xy:
        return "No determinada"
    w, h = pil.size
    x = sum(p[0] for p in red_xy) / len(red_xy)
    y = sum(p[1] for p in red_xy) / len(red_xy)
    horiz = "izquierda" if x < w / 3 else "centro" if x < 2 * w / 3 else "derecha"
    vert = "superior" if y < h / 3 else "media" if y < 2 * h / 3 else "inferior"
    return f"{horiz}, zona {vert}"

def _ocv2_norm_points_from_px(points_px, w, h, n=6):
    pts = np.asarray(points_px, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 2:
        return []
    a = pts[0]
    b = pts[-1]
    out = []
    for t in np.linspace(0.0, 1.0, max(2, int(n))):
        p = a * (1.0 - t) + b * t
        out.append([
            float(np.clip(p[0] / max(1.0, w - 1) * 1000.0, 0.0, 1000.0)),
            float(np.clip(p[1] / max(1.0, h - 1) * 1000.0, 0.0, 1000.0)),
        ])
    return out

def _ocv2_group_consecutive(indices):
    vals = sorted(set(int(v) for v in indices if int(v) > 0))
    if not vals:
        return []
    groups = [[vals[0]]]
    for v in vals[1:]:
        if v == groups[-1][-1] + 1:
            groups[-1].append(v)
        else:
            groups.append([v])
    return groups

def _ocv2_regularize_peaks(peaks, spacing, width):
    """Elimina duplicados y rellena SOLO huecos pequeños de la rejilla."""
    vals = sorted(float(v) for v in np.asarray(peaks).reshape(-1))
    if len(vals) < 2:
        return vals

    spacing = max(6.0, float(spacing or np.median(np.diff(vals))))

    # Primero quitar duplicados demasiado cercanos.
    dedup = [vals[0]]
    for x in vals[1:]:
        if x - dedup[-1] < spacing * 0.62:
            dedup[-1] = 0.5 * (dedup[-1] + x)
        else:
            dedup.append(x)

    # Rellenar solo 1 o 2 hileras omitidas. Nunca puentea caminos grandes.
    out = []
    for a, b in zip(dedup[:-1], dedup[1:]):
        out.append(a)
        gap = b - a
        steps = int(round(gap / spacing))
        if 2 <= steps <= 3 and gap <= spacing * 3.35:
            step = gap / float(steps)
            for k in range(1, steps):
                out.append(a + step * k)
    out.append(dedup[-1])

    return [float(np.clip(x, 0, max(0, width - 1))) for x in out]

def _ocv2_inverse_affine_point(Minv, x, y):
    p = np.asarray([float(x), float(y), 1.0], dtype=np.float32)
    q = Minv @ p
    return np.asarray([float(q[0]), float(q[1])], dtype=np.float32)

def _ocv2_detect_straight_rows(pil):
    """Detecta la rejilla transversal y crea una línea recta por hilera.

    La decisión de qué hilera existe sale del perfil global del viñedo, no de
    una polilínea que pueda desviarse y brincar a la hilera vecina.
    """
    bgr = cv2.cvtColor(np.asarray(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    (
        green,
        theta,
        coherence,
        response,
        components,
        ny,
        nx,
    ) = detectar_componentes_vinedo(
        bgr,
        tile=max(24, int(min(h, w) / 18)),
    )

    if not components:
        raise RuntimeError("OpenCV no encontró una zona clara con patrón de viñedo.")

    grid = calcular_rejilla_global_surcos(green, components)
    if grid is None or len(grid.get("peaks", [])) < 4:
        # Respaldo con el detector V12 existente y ajuste recto robusto.
        base = analizar(pil, exclusion_mask=None)
        tracks = list(base.get("tracks", []) or [])
        rows = []
        by_idx = {}
        for t in tracks:
            idx = int(t.get("track_index", len(by_idx) + 1))
            old = by_idx.get(idx)
            if old is None or float(t.get("line_length", 0.0) or 0.0) > float(old.get("line_length", 0.0) or 0.0):
                by_idx[idx] = t

        for idx in sorted(by_idx):
            t = by_idx[idx]
            pts = np.asarray(t.get("points", []), dtype=np.float32).reshape(-1, 2)
            if len(pts) < 5:
                continue
            line = cv2.fitLine(pts.reshape(-1, 1, 2), cv2.DIST_HUBER, 0, 0.01, 0.01)
            vx, vy, x0, y0 = [float(v) for v in line.flatten()]
            d = np.asarray([vx, vy], dtype=np.float32)
            dn = float(np.linalg.norm(d))
            if dn < 1e-6:
                continue
            d /= dn
            c = np.asarray([x0, y0], dtype=np.float32)
            nvec = np.asarray([-d[1], d[0]], dtype=np.float32)
            residual = np.dot(pts - c, nvec)
            med = float(np.median(residual))
            mad = float(np.median(np.abs(residual - med))) + 1e-6
            spacing = float(t.get("spacing", 18.0) or 18.0)
            corridor = max(3.0, min(spacing * 0.38, 2.6 * mad + 2.5))
            keep = np.abs(residual - med) <= corridor
            if int(np.sum(keep)) >= max(5, int(len(pts) * 0.45)):
                pts2 = pts[keep]
                line = cv2.fitLine(pts2.reshape(-1, 1, 2), cv2.DIST_HUBER, 0, 0.01, 0.01)
                vx, vy, x0, y0 = [float(v) for v in line.flatten()]
                d = np.asarray([vx, vy], dtype=np.float32)
                d /= max(1e-6, float(np.linalg.norm(d)))
                c = np.asarray([x0, y0], dtype=np.float32)
                pts = pts2
            ref = np.asarray(t.get("points", [])[-1], dtype=np.float32) - np.asarray(t.get("points", [])[0], dtype=np.float32)
            if float(np.dot(d, ref)) < 0:
                d = -d
            proj = np.dot(pts - c, d)
            lo = float(np.percentile(proj, 1.0))
            hi = float(np.percentile(proj, 99.0))
            if hi <= lo:
                continue
            p0 = c + d * lo
            p1 = c + d * hi
            p0[0] = np.clip(p0[0], 0, w - 1); p0[1] = np.clip(p0[1], 0, h - 1)
            p1[0] = np.clip(p1[0], 0, w - 1); p1[1] = np.clip(p1[1], 0, h - 1)
            if float(np.linalg.norm(p1 - p0)) < min(h, w) * 0.08:
                continue
            rows.append({
                "id": len(rows) + 1,
                "confidence": 0.66,
                "points_norm": _ocv2_norm_points_from_px([p0, p1], w, h, n=6),
                "slot_count": 0,
                "vacant_indices": [],
                "row_spacing_px": spacing,
            })
        rows = _tg_sort_rows(rows)
        return {
            "pil": pil,
            "rows": rows,
            "response_map": np.asarray(base.get("response_map"), dtype=np.float32),
            "green_mask": np.asarray(base.get("green_mask"), dtype=np.uint8),
            "row_spacing_px": float(np.median([r.get("row_spacing_px", 18.0) for r in rows])) if rows else 18.0,
            "angle_deg": float(base.get("angle", 0.0) or 0.0),
        }

    M = np.asarray(grid["M"], dtype=np.float32)
    Minv = cv2.invertAffineTransform(M)
    spacing = float(grid.get("spacing", 18.0) or 18.0)
    peaks = _ocv2_regularize_peaks(grid.get("peaks", []), spacing, w)

    rot_green = cv2.warpAffine(
        (green > 0).astype(np.uint8),
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )

    # Soporte vertical general del viñedo. Sirve como respaldo para hileras secas.
    global_y_score = gaussian_filter1d(np.mean(rot_green.astype(np.float32), axis=1), sigma=2.0)
    global_ys = np.where(global_y_score > max(0.004, float(np.percentile(global_y_score, 45)) * 0.22))[0]
    if len(global_ys) >= 10:
        common_y0 = int(np.percentile(global_ys, 1.0))
        common_y1 = int(np.percentile(global_ys, 99.0))
    else:
        common_y0 = int(h * 0.05)
        common_y1 = int(h * 0.95)

    rows = []
    band_half = max(2, int(round(spacing * 0.24)))

    for x in peaks:
        xi = int(round(x))
        xa = max(0, xi - band_half)
        xb = min(w, xi + band_half + 1)
        if xb <= xa:
            continue

        col = np.mean(rot_green[:, xa:xb].astype(np.float32), axis=1)
        col = gaussian_filter1d(col, sigma=2.2)
        thr = max(0.010, float(np.percentile(col, 68)) * 0.24)
        ys = np.where(col > thr)[0]

        if len(ys) >= 8 and (ys[-1] - ys[0]) >= h * 0.30:
            y0 = int(np.percentile(ys, 1.5))
            y1 = int(np.percentile(ys, 98.5))
        else:
            y0, y1 = common_y0, common_y1

        # Evitar cabeceras extremas, pero permitir surcos que llegan cerca del borde.
        y0 = int(np.clip(y0, 0, h - 2))
        y1 = int(np.clip(y1, y0 + 1, h - 1))

        p0 = _ocv2_inverse_affine_point(Minv, x, y0)
        p1 = _ocv2_inverse_affine_point(Minv, x, y1)
        p0[0] = np.clip(p0[0], 0, w - 1); p0[1] = np.clip(p0[1], 0, h - 1)
        p1[0] = np.clip(p1[0], 0, w - 1); p1[1] = np.clip(p1[1], 0, h - 1)

        length = float(np.linalg.norm(p1 - p0))
        if length < min(h, w) * 0.18:
            continue

        support = float(np.mean(col[y0:y1 + 1])) if y1 > y0 else 0.0
        confidence = float(np.clip(0.58 + min(0.32, support * 2.2), 0.45, 0.92))
        rows.append({
            "id": len(rows) + 1,
            "confidence": confidence,
            "points_norm": _ocv2_norm_points_from_px([p0, p1], w, h, n=6),
            "slot_count": 0,
            "vacant_indices": [],
            "row_spacing_px": spacing,
        })

    rows = _tg_sort_rows(rows)
    for i, r in enumerate(rows, 1):
        r["id"] = i

    if len(rows) < 4:
        raise RuntimeError("OpenCV detectó muy pocos surcos rectos para un Inventario confiable.")

    return {
        "pil": pil,
        "rows": rows,
        "response_map": np.asarray(response, dtype=np.float32),
        "green_mask": (green > 0).astype(np.uint8) * 255,
        "row_spacing_px": spacing,
        "angle_deg": float(grid.get("angle_deg", 0.0) or 0.0),
    }

def _ocv2_slot_positions(row, full_size):
    w, h = full_size
    pts = _tg_norm_to_px(row.get("points_norm", []), w, h)
    count = int(row.get("slot_count", 0) or 0)
    if len(pts) < 2 or count <= 0:
        return []
    if count == 1:
        return [_tg_point_on_polyline(pts, 0.5)]
    return [_tg_point_on_polyline(pts, i / float(count - 1)) for i in range(count)]

def _tc_mejorar_calidad_imagen(pil_img):
    """
    Mejora moderadamente el contraste y la nitidez de la fotografía
    para facilitar la detección de surcos y patrones de plantas.

    No modifica el archivo original.
    No aumenta artificialmente la confianza.
    """
    original = pil_img.convert("RGB")

    try:
        # Convertir imagen a matriz RGB
        rgb = np.asarray(original, dtype=np.uint8)

        # Separar luminosidad y color
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Mejorar contraste local sin exagerarlo
        clahe = cv2.createCLAHE(
            clipLimit=1.6,
            tileGridSize=(8, 8)
        )
        l_mejorado = clahe.apply(l)

        # Reconstruir la imagen conservando los canales de color
        lab_mejorado = cv2.merge((l_mejorado, a, b))
        rgb_mejorado = cv2.cvtColor(
            lab_mejorado,
            cv2.COLOR_LAB2RGB
        )

        # Aplicar nitidez suave
        desenfoque = cv2.GaussianBlur(
            rgb_mejorado,
            (0, 0),
            sigmaX=0.8
        )

        nitidez = cv2.addWeighted(
            rgb_mejorado,
            1.12,
            desenfoque,
            -0.12,
            0
        )

        # Asegurar valores válidos de píxeles
        nitidez = np.clip(
            nitidez, 0, 255
        ).astype(np.uint8)

        return Image.fromarray(nitidez)

    except Exception as exc:
        # Si falla la mejora, utilizar la imagen original
        print(
            f"TerraCore: no se pudo mejorar la imagen: {exc}"
        )
        return original

def _ocv2_inventory_from_file(uploaded_file):
    # Leer la fotografía original
    pil_original = Image.open(
        io.BytesIO(uploaded_file.getvalue())
    ).convert("RGB")

    # Mejorar la imagen antes de analizarla
    pil = _tc_mejorar_calidad_imagen(pil_original)

    # Ejecutar la detección sobre la imagen mejorada
    w, h = pil.size
    base = _ocv2_detect_straight_rows(pil)

    # Continuar con el análisis original
    rows = [dict(r) for r in base["rows"]]
    response_map = np.asarray(
        base["response_map"], dtype=np.float32
    )

    # 1) Encontrar un paso común de planta para evitar conteos erráticos entre hileras.
    pitch_candidates = []
    for r in rows:
        pts = np.asarray(_tg_norm_to_px(r.get("points_norm", []), w, h), dtype=np.float32)
        if len(pts) < 2:
            continue
        spacing = float(r.get("row_spacing_px", base.get("row_spacing_px", 18.0)) or 18.0)
        sampled = np.asarray([_tg_point_on_polyline(pts.tolist(), t) for t in np.linspace(0, 1, 100)], dtype=np.float32)
        ds, prof, _ = _tc_sample_map_along_track(response_map, sampled, spacing, step_px=2.0)
        if len(ds) >= 8:
            p, c = _tc_estimar_pitch_slots(ds, prof, spacing)
            if np.isfinite(p) and 4.0 <= float(p) <= max(8.0, spacing * 1.25):
                pitch_candidates.append(float(p))

    forced_pitch = float(np.median(pitch_candidates)) if pitch_candidates else None

    confs = []
    for r in rows:
        pts = np.asarray(_tg_norm_to_px(r.get("points_norm", []), w, h), dtype=np.float32)
        spacing = float(r.get("row_spacing_px", base.get("row_spacing_px", 18.0)) or 18.0)
        sampled = np.asarray([_tg_point_on_polyline(pts.tolist(), t) for t in np.linspace(0, 1, 120)], dtype=np.float32)
        local_track = {"points": sampled, "spacing": spacing}
        slots, pitch_used, conf = _tc_slots_track(response_map, local_track, forced_pitch=forced_pitch)

        # Si la señal no permitió slots, usar la escala común SIN inventar cambios laterales.
        if len(slots) < 2 and forced_pitch:
            length = float(np.linalg.norm(pts[-1] - pts[0]))
            count = max(2, int(round(length / max(5.0, forced_pitch))) + 1)
            positions = [_tg_point_on_polyline(pts.tolist(), t) for t in np.linspace(0, 1, count)]
            slots = [{"point": p, "occupied": True, "score": 0.0} for p in positions]
            conf = 0.42
            pitch_used = forced_pitch

        r["slot_count"] = int(len(slots))
        r["vacant_indices"] = [i + 1 for i, s in enumerate(slots) if not bool(s.get("occupied", False))]
        r["slot_confidence"] = float(conf)
        r["slot_pitch_px"] = float(pitch_used or forced_pitch or 0.0)
        confs.append(float(conf))

    # 2) Dibujo Inventario. Línea y puntos comparten EXACTAMENTE el mismo eje recto.
    band_h = max(46, int(round(h * 0.065)))
    canvas = Image.new("RGB", (w, h + band_h), (91, 37, 46))
    canvas.paste(pil, (0, band_h))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    table = []
    line_w = max(1, int(round(min(w, h) / 950)))

    for r in rows:
        rid = int(r.get("id", 0) or 0)
        pts = _tg_norm_to_px(r.get("points_norm", []), w, h)
        if len(pts) < 2:
            continue
        a, b = pts[0], pts[-1]
        draw.line((a[0], a[1] + band_h, b[0], b[1] + band_h), fill=(245,245,245), width=line_w)

        count = int(r.get("slot_count", 0) or 0)
        vacant = set(_tg_clean_indices(r.get("vacant_indices", []), count))
        positions = _ocv2_slot_positions(r, (w, h))
        for idx, p in enumerate(positions, 1):
            x = float(p[0]); y = float(p[1] + band_h)
            if idx in vacant:
                rr = 2.5; fill = (255, 145, 35)   # vacío
            else:
                rr = 1.8; fill = (45, 190, 255)   # ocupado
            draw.ellipse((x-rr, y-rr, x+rr, y+rr), fill=fill)

        # número solo arriba
        top = a if a[1] <= b[1] else b
        label = f"{rid:02d}"
        bbox = draw.textbbox((0,0), label, font=font)
        tw = max(10, bbox[2]-bbox[0])
        tx = int(np.clip(top[0]-tw/2, 1, max(1, w-tw-2)))
        ty = 5 if rid % 2 else 20
        draw.text((tx,ty), label, fill=(255,255,255), font=font, stroke_width=2, stroke_fill=(35,20,25))

        occ = max(0, count - len(vacant))
        # Separamos dos conceptos para no confundir la detección de la línea
        # con la confianza de la estimación de posiciones (slots).
        # r["confidence"] viene del soporte visual de la línea del surco.
        # r["slot_confidence"] viene de la periodicidad y contraste de los slots.
        table.append({
            tr("Surco", "Rang"): label,
            tr("Slots", "Emplacements"): count,
            tr("Ocupados", "Occupés"): occ,
            tr("Vacíos", "Vides"): len(vacant),
            tr("Confianza del surco", "Confiance du rang"): round(
                float(np.clip(r.get("confidence", 0.0), 0.0, 1.0)) * 100.0, 1
            ),
            tr("Estado", "État"): tr("OpenCV local", "OpenCV local"),
        })

    if not table:
        raise RuntimeError("OpenCV no pudo construir el Inventario de slots.")

    return {
        "pil": pil,
        "rows": rows,
        "image": canvas,
        "table": pd.DataFrame(table),
        "count": len(rows),
        "confidence": float(np.mean(confs)) if confs else 0.0,
        "coverage_score": 1.0,
        "model": "OpenCV local — líneas rectas",
        "warnings": [],
        "debug": {
            "pitch_px": float(forced_pitch or 0.0),
            "row_spacing_px": float(base.get("row_spacing_px", 0.0) or 0.0),
            "angle_deg": float(base.get("angle_deg", 0.0) or 0.0),
            "motor": "opencv-v2-straight-grid",
        },
    }

def _ocv2_slot_patch_scores(bgr, center, direction, row_spacing, slot_pitch):
    """Mide verde y seco en un rectángulo orientado sin invadir el surco vecino."""
    h, w = bgr.shape[:2]
    cx, cy = float(center[0]), float(center[1])
    d = np.asarray(direction, dtype=np.float32)
    dn = float(np.linalg.norm(d))
    if dn < 1e-6:
        d = np.asarray([0.0, 1.0], dtype=np.float32)
    else:
        d = d / dn
    nvec = np.asarray([-d[1], d[0]], dtype=np.float32)

    along = max(3.0, min(max(5.0, float(slot_pitch or 8.0) * 0.42), max(6.0, float(row_spacing) * 0.52)))
    cross = max(2.5, float(row_spacing) * 0.24)
    radius = int(math.ceil(along + cross + 3.0))

    x0 = max(0, int(math.floor(cx)) - radius)
    x1 = min(w, int(math.ceil(cx)) + radius + 1)
    y0 = max(0, int(math.floor(cy)) - radius)
    y1 = min(h, int(math.ceil(cy)) + radius + 1)
    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0, 0.0, 0.0

    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx = xx.astype(np.float32) - cx
    dy = yy.astype(np.float32) - cy
    pa = dx * d[0] + dy * d[1]
    pc = dx * nvec[0] + dy * nvec[1]
    sel = (np.abs(pa) <= along) & (np.abs(pc) <= cross)
    if not np.any(sel):
        return 0.0, 0.0, 0.0

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    rr, gg, bb = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    hh, ss, vv = cv2.split(hsv)
    exg = 2.0*gg - rr - bb
    ngr = (gg-rr)/(gg+rr+1e-6)

    green = (
        (exg > 4.0) &
        (ngr > -0.025) &
        (hh >= 18) & (hh <= 115) &
        (ss >= 10) & (vv >= 18) &
        (gg >= rr*0.84) & (gg >= bb*0.84)
    )

    # Marrón / amarillo / seco visible. No se usa solo; se combina con verde bajo.
    dry = (
        (hh >= 3) & (hh <= 42) &
        (ss >= 18) & (vv >= 24) &
        (~green) &
        (rr >= bb * 1.05)
    )

    # Suelo claro/expuesto: ayuda a confirmar vacíos.
    soil = (
        (~green) &
        (vv >= 42) &
        (ss >= 8) &
        (hh >= 3) & (hh <= 45)
    )

    denom = max(1, int(np.sum(sel)))
    return (
        float(np.sum(green & sel) / denom),
        float(np.sum(dry & sel) / denom),
        float(np.sum(soil & sel) / denom),
    )

def _ocv2_classify_health(pil, rows):
    bgr = cv2.cvtColor(np.asarray(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    # Primera pasada: medir todos los slots para obtener referencia sana global.
    measurements = {}
    all_green = []
    for r in rows:
        rid = int(r.get("id", 0) or 0)
        count = int(r.get("slot_count", 0) or 0)
        positions = _ocv2_slot_positions(r, (w, h))
        pts = _tg_norm_to_px(r.get("points_norm", []), w, h)
        if count < 2 or len(positions) != count or len(pts) < 2:
            continue
        d = np.asarray(pts[-1], dtype=np.float32) - np.asarray(pts[0], dtype=np.float32)
        dn = float(np.linalg.norm(d))
        d = d / max(1e-6, dn)
        row_spacing = float(r.get("row_spacing_px", 18.0) or 18.0)
        slot_pitch = float(r.get("slot_pitch_px", 0.0) or 0.0)
        if slot_pitch <= 0 and count > 1:
            slot_pitch = dn / float(count - 1)
        vals = []
        for p in positions:
            g, dry, soil = _ocv2_slot_patch_scores(bgr, p, d, row_spacing, slot_pitch)
            vals.append((g, dry, soil))
            if g > 0.005:
                all_green.append(g)
        measurements[rid] = vals

    global_ref = float(np.percentile(all_green, 78)) if len(all_green) >= 8 else (float(np.max(all_green)) if all_green else 0.08)
    global_ref = max(0.045, global_ref)

    health_map = {}
    confs = []
    for r in rows:
        rid = int(r.get("id", 0) or 0)
        count = int(r.get("slot_count", 0) or 0)
        vals = measurements.get(rid, [])
        if count < 2 or len(vals) != count:
            health_map[rid] = {"red_indices": [], "confidence": 0.0}
            continue

        g_arr = np.asarray([v[0] for v in vals], dtype=np.float32)
        d_arr = np.asarray([v[1] for v in vals], dtype=np.float32)
        s_arr = np.asarray([v[2] for v in vals], dtype=np.float32)
        local_pos = g_arr[g_arr > 0.005]
        local_ref = float(np.percentile(local_pos, 75)) if len(local_pos) >= 4 else global_ref
        ref = max(global_ref * 0.72, local_ref)

        vacant = set(_tg_clean_indices(r.get("vacant_indices", []), count))
        red = set(vacant)

        for i, (g, dry, soil) in enumerate(zip(g_arr, d_arr, s_arr), 1):
            if i in vacant:
                continue
            very_low = g < max(0.020, ref * 0.26)
            weak = g < max(0.030, ref * 0.40)
            clearly_dry = dry > 0.20 and g < ref * 0.68
            bare_like = soil > 0.34 and g < ref * 0.32
            if very_low or clearly_dry or (weak and bare_like):
                red.add(i)

        # Evitar un micro-verde aislado dentro de un tramo afectado.
        for i in range(2, count):
            if i not in red and (i-1) in red and (i+1) in red and g_arr[i-1] < ref * 0.72:
                red.add(i)

        contrast = float(np.std(g_arr)/(np.mean(g_arr)+1e-6)) if len(g_arr) else 0.0
        conf = float(np.clip(0.62 + min(0.25, contrast*0.35), 0.50, 0.90))
        health_map[rid] = {
            "red_indices": sorted(red),
            "confidence": conf,
            "green_reference": ref,
        }
        confs.append(conf)

    return health_map, (float(np.mean(confs)) if confs else 0.0)

def _ocv2_polygon_for_group(positions, group, row_spacing, slot_pitch):
    if not positions or not group:
        return None
    n = len(positions)
    first = max(1, int(group[0])) - 1
    last = min(n, int(group[-1])) - 1
    p0 = np.asarray(positions[first], dtype=np.float32)
    p1 = np.asarray(positions[last], dtype=np.float32)

    prev_pt = np.asarray(positions[max(0, first-1)], dtype=np.float32)
    next_pt = np.asarray(positions[min(n-1, last+1)], dtype=np.float32)
    tangent = next_pt - prev_pt
    tn = float(np.linalg.norm(tangent))
    if tn < 1e-6:
        tangent = np.asarray([0.0, 1.0], dtype=np.float32)
        tn = 1.0
    tangent /= tn
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)

    pitch = max(5.0, float(slot_pitch or tn))
    cap = pitch * 0.42
    half_width = max(4.0, float(row_spacing) * 0.28)

    start = p0 - tangent * cap
    end = p1 + tangent * cap
    return [
        tuple((start + normal*half_width).tolist()),
        tuple((end + normal*half_width).tolist()),
        tuple((end - normal*half_width).tolist()),
        tuple((start - normal*half_width).tolist()),
    ]

def _ocv2_draw_health_and_polygons(pil, rows, health_map):
    base = pil.convert("RGB")
    w, h = base.size
    lines_img = base.copy()
    polygons_img = base.copy()
    draw_l = ImageDraw.Draw(lines_img, "RGBA")
    draw_p = ImageDraw.Draw(polygons_img, "RGBA")
    font = ImageFont.load_default()
    line_w = max(2, int(round(min(w,h)/600)))

    total = 0
    red_total = 0
    red_xy = []

    for r in rows:
        rid = int(r.get("id", 0) or 0)
        count = int(r.get("slot_count", 0) or 0)
        pts = _tg_norm_to_px(r.get("points_norm", []), w, h)
        positions = _ocv2_slot_positions(r, (w, h))
        if len(pts) < 2 or count < 2 or len(positions) != count:
            continue

        a = np.asarray(pts[0], dtype=np.float32)
        b = np.asarray(pts[-1], dtype=np.float32)
        red_indices = set(health_map.get(rid, {}).get("red_indices", []))
        red_indices.update(_tg_clean_indices(r.get("vacant_indices", []), count))
        red_indices = {i for i in red_indices if 1 <= i <= count}

        total += count
        red_total += len(red_indices)

        # Diagnóstico de líneas: eje VERDE recto completo.
        draw_l.line((float(a[0]),float(a[1]),float(b[0]),float(b[1])), fill=(25,220,55,255), width=line_w)

        row_spacing = float(r.get("row_spacing_px", 18.0) or 18.0)
        slot_pitch = float(r.get("slot_pitch_px", 0.0) or 0.0)
        if slot_pitch <= 0:
            slot_pitch = float(np.linalg.norm(b-a))/max(1, count-1)

        for group in _ocv2_group_consecutive(red_indices):
            # segmento rojo centrado exactamente en los slots afectados
            first = max(1, group[0])
            last = min(count, group[-1])
            t0 = max(0.0, (first - 1 - 0.45) / max(1, count - 1))
            t1 = min(1.0, (last - 1 + 0.45) / max(1, count - 1))
            p0 = _tg_point_on_polyline(pts, t0)
            p1 = _tg_point_on_polyline(pts, t1)
            draw_l.line((p0[0],p0[1],p1[0],p1[1]), fill=(245,45,45,255), width=max(3,line_w+1))

            poly = _ocv2_polygon_for_group(positions, group, row_spacing, slot_pitch)
            if poly:
                # Polígono rojo SOLO en la zona seca/vacía.
                draw_p.polygon(poly, fill=(40,120,255,88), outline=(30,110,245,235))

        # Puntos del diagnóstico: todos sobre el MISMO eje recto.
        for idx, p in enumerate(positions, 1):
            x, y = float(p[0]), float(p[1])
            if idx in red_indices:
                rr = 2.3
                col = (245,45,45,255)
                red_xy.append((x,y))
            else:
                rr = 1.6
                col = (25,225,55,255)
            draw_l.ellipse((x-rr,y-rr,x+rr,y+rr), fill=col)

        # Solo número arriba.
        top = a if a[1] <= b[1] else b
        label = f"{rid:02d}"
        draw_l.text((int(top[0])-6, max(0,int(top[1])-13)), label,
                    fill=(255,255,255,255), font=font, stroke_width=2, stroke_fill=(35,20,25,255))

    green_total = max(0, total - red_total)
    green_pct = 100.0 * green_total / total if total else 0.0
    red_pct = 100.0 * red_total / total if total else 0.0
    return {
        "annotated": lines_img,
        "polygon_image": polygons_img,
        "green_pct": green_pct,
        "red_pct": red_pct,
        "green_slots": green_total,
        "red_slots": red_total,
        "total_slots": total,
        "red_xy": red_xy,
    }

def _ocv2_analyze_health(uploaded_image, rows):
    pil = Image.open(io.BytesIO(uploaded_image.getvalue())).convert("RGB")
    if not rows:
        raise RuntimeError("No hay geometría de Inventario confirmada.")

    health_map, conf = _ocv2_classify_health(pil, rows)
    visual = _ocv2_draw_health_and_polygons(pil, rows, health_map)

    red_pct = float(visual["red_pct"])
    if red_pct < 15:
        nivel = "bajo"
    elif red_pct < 35:
        nivel = "medio"
    else:
        nivel = "alto"

    zona = _tg_zone_fallback(pil, visual.get("red_xy", []))
    diagnostico = (
        f"Se evaluaron {visual['total_slots']} posiciones sobre {len(rows)} surcos rectos. "
        f"Se detectaron {visual['red_slots']} posiciones secas, débiles o vacías ({red_pct:.1f}% rojo). "
        "Los polígonos rojos se limitan a los tramos afectados detectados sobre el mismo surco."
    )

    result = {
        "count": len(rows),
        "green_pct": float(visual["green_pct"]),
        "red_pct": red_pct,
        "green_slots": int(visual["green_slots"]),
        "red_slots": int(visual["red_slots"]),
        "total_slots": int(visual["total_slots"]),
        "angle": 0.0,
        "annotated": visual["annotated"],
        "polygon_image": visual["polygon_image"],
        "result_url": "",
        "zona_mas_afectada": zona,
        "nivel_afectacion_visual": nivel,
        "diagnostico_visual": _tc_quitar_nombre_motor_del_analisis(diagnostico),
        "causas_probables": [],
        "explicacion_nutrientes": "El análisis es visual y no identifica nutrientes ni enfermedades por sí solo.",
        "recomendaciones_iniciales": [
            "Revisar en campo los polígonos rojos para distinguir planta seca de faltante real.",
            "Comparar la misma parcela en capturas posteriores para confirmar si la afectación persiste.",
        ],
        "nota_diagnostico": "Diagnóstico visual automático; confirmar en campo.",
        "detalle_zonas": {},
        "metodo": "opencv-v2-straight-grid-polygons",
        "confidence": float(conf),
        "health_map": health_map,
    }
    result["backend"] = {
        "metodo": result["metodo"],
        "analisis": {
            "surcos_estimados": result["count"],
            "verde_pct": result["green_pct"],
            "rojo_pct": result["red_pct"],
            "zona_mas_afectada": result["zona_mas_afectada"],
            "nivel_afectacion_visual": result["nivel_afectacion_visual"],
            "diagnostico_visual": result["diagnostico_visual"],
            "causas_probables": [],
            "explicacion_nutrientes": result["explicacion_nutrientes"],
            "recomendaciones_iniciales": result["recomendaciones_iniciales"],
            "nota_diagnostico": result["nota_diagnostico"],
            "total_slots": result["total_slots"],
            "green_slots": result["green_slots"],
            "red_slots": result["red_slots"],
        },
    }
    return result
