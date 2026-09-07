import io
import os
import csv
import math
import zipfile
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline

st.set_page_config(
    page_title="Contador de surcos desde video",
    page_icon="🎥",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    max-width: 1600px;
    padding-top: 1rem;
    padding-bottom: 2rem;
}
.card {
    border:1px solid #303846;
    border-radius:10px;
    padding:14px;
    margin-top:10px;
}
.sub {
    color:#a9b2bf;
    margin-bottom:1rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🎥 Contador automático de surcos desde video – V10.2")
st.markdown(
    '<div class="sub">'
    'Extrae automáticamente los fotogramas más útiles, evita escenas consecutivas muy parecidas '
    'y estima los surcos visibles por escena/parcela candidata.'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# VEGETACIÓN
# ============================================================
def mascara_verde(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hh, ss, vv = cv2.split(hsv)

    exg = 2.0 * g - r - b
    ngrdi = (g - r) / (g + r + 1e-6)

    # Máscara calibrada para estas fotografías:
    # admite verde débil pero intenta rechazar suelo café.
    mask = (
        (exg > 7.0) &
        (ngrdi > -0.015) &
        (hh >= 21) & (hh <= 108) &
        (ss >= 16) &
        (vv >= 22) &
        (g >= r * 0.875) &
        (g >= b * 0.875)
    ).astype(np.uint8) * 255

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((2, 2), np.uint8),
        iterations=1
    )

    return mask


# ============================================================
# ÁNGULO DOMINANTE DE LOS SURCOS
# ============================================================
def angulo_surcos(mask):
    h, w = mask.shape

    # Solo zona central para no dejar que caminos/edificios dominen.
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
        maxLineGap=18
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

        # Solo estructuras aproximadamente verticales.
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
        weights=weights
    )

    i = int(np.argmax(hist))
    lo = edges_b[i]
    hi = edges_b[i + 1]

    sel = (angles >= lo) & (angles < hi)

    if np.any(sel):
        return float(np.average(angles[sel], weights=weights[sel]))

    return float(np.median(angles))


def rotar(img, angle):
    h, w = img.shape[:2]
    centro = (w / 2.0, h / 2.0)

    # Llevar los surcos a vertical.
    rot_deg = 90.0 - angle

    M = cv2.getRotationMatrix2D(centro, rot_deg, 1.0)
    Minv = cv2.invertAffineTransform(M)

    out = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    return out, M, Minv


def aplicar_matriz(points, M):
    pts = np.asarray(points, dtype=np.float32)

    if len(pts) == 0:
        return pts

    ones = np.ones((len(pts), 1), dtype=np.float32)
    aug = np.hstack([pts, ones])

    return aug @ M.T


# ============================================================
# DETECCIÓN AUTOMÁTICA DE PARCELA
# ============================================================
def limites_verticales(mask):
    """
    Busca el camino superior y el camino inferior.
    """
    h, w = mask.shape

    density = (mask > 0).mean(axis=1).astype(np.float32)
    density = gaussian_filter1d(
        density,
        sigma=max(7, h / 100)
    )

    # Camino superior.
    top_range = np.arange(
        int(h * 0.05),
        int(h * 0.48)
    )

    y_top_road = int(
        top_range[np.argmin(density[top_range])]
    )

    # Camino inferior.
    bottom_range = np.arange(
        int(h * 0.55),
        int(h * 0.95)
    )

    y_bottom_road = int(
        bottom_range[np.argmin(density[bottom_range])]
    )

    inset = max(8, int(h * 0.010))

    y0 = y_top_road + inset
    y1 = y_bottom_road - inset

    # Protección por si una imagen distinta confunde los mínimos.
    if y1 - y0 < h * 0.36:
        y0 = int(h * 0.14)
        y1 = int(h * 0.88)

    return max(0, y0), min(h - 1, y1)


def estimar_periodo(profile):
    """
    Estima la separación entre hileras.
    """
    p = profile.astype(np.float64)
    p -= np.mean(p)

    if np.std(p) < 1e-7:
        return 20.0

    ac = np.correlate(p, p, mode="full")
    ac = ac[len(p)-1:]

    width = len(profile)

    a = max(8, int(width * 0.006))
    b = min(int(width * 0.030), len(ac) - 1)

    if b <= a:
        return 20.0

    peaks, _ = find_peaks(ac[a:b+1])

    if len(peaks) == 0:
        return max(16.0, width / 70.0)

    vals = ac[a:b+1][peaks]
    lag = float(a + peaks[np.argmax(vals)])

    # En estas imágenes el primer armónico suele aparecer a media hilera.
    # Si sale demasiado pequeño, se duplica.
    if lag < width / 85.0:
        lag *= 2.0

    return float(np.clip(lag, 13.0, 35.0))


def detectar_limites_laterales(mask, y0, y1):
    """
    Detecta automáticamente los caminos laterales que delimitan
    la parcela principal. Devuelve x0, x1.
    """
    h, w = mask.shape

    zone = (mask[y0:y1] > 0).astype(np.float32)

    # Densidad verde por columna.
    density = zone.mean(axis=0)
    density = gaussian_filter1d(
        density,
        sigma=max(3.0, w / 450.0)
    )

    # Normalización robusta.
    p20 = float(np.percentile(density, 20))
    p65 = float(np.percentile(density, 65))

    # Los caminos son bandas verticales con muy poca vegetación.
    threshold = p20 + 0.22 * max(p65 - p20, 1e-6)

    low = density < threshold

    # Extraer bandas continuas de baja vegetación.
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

        # Camino/borde útil: debe tener cierto ancho.
        if width_band >= max(5, int(w * 0.004)):
            bands.append((i, j - 1, width_band))

        i = j

    center = w / 2.0

    left_candidates = [
        b for b in bands
        if b[1] < center and b[1] > w * 0.03
    ]

    right_candidates = [
        b for b in bands
        if b[0] > center and b[0] < w * 0.97
    ]

    # Preferir caminos anchos y relativamente cercanos al bloque central.
    if left_candidates:
        left_band = max(
            left_candidates,
            key=lambda b: (
                b[2] * 3.0
                - abs(center - b[1]) * 0.012
            )
        )
        x0 = int(left_band[1] + max(3, w * 0.003))
    else:
        x0 = int(w * 0.08)

    if right_candidates:
        right_band = max(
            right_candidates,
            key=lambda b: (
                b[2] * 3.0
                - abs(b[0] - center) * 0.012
            )
        )
        x1 = int(right_band[0] - max(3, w * 0.003))
    else:
        x1 = int(w * 0.92)

    # Salvaguardas.
    if x1 - x0 < w * 0.35:
        x0 = int(w * 0.08)
        x1 = int(w * 0.92)

    return max(0, x0), min(w - 1, x1)


def semillas_surcos(mask, y0, y1):
    """
    V3.3 - detectar TODOS los surcos.

    En vez de promediar toda la parcela (lo cual borra hileras
    curvas o débiles), toma varios cortes horizontales y calcula
    el espaciado típico entre surcos.

    Después construye una retícula completa. Una hilera seca no
    desaparece solo porque tenga poco verde.
    """
    x0, x1 = detectar_limites_laterales(
        mask,
        y0,
        y1
    )

    zone = (
        mask[
            y0:y1,
            x0:x1
        ] > 0
    ).astype(np.float32)

    height, width = zone.shape

    if width < 30 or height < 30:
        return (
            np.array([], dtype=np.int32),
            20.0,
            int(x0),
            int(x1)
        )

    # --------------------------------------------------------
    # PERFILES EN VARIAS ALTURAS
    # --------------------------------------------------------
    profiles = []
    periods = []
    contrasts = []

    centers = np.linspace(
        0.12,
        0.88,
        9
    )

    band_h = max(
        12,
        int(height * 0.11)
    )

    for frac in centers:
        yc = int(
            round(
                frac * (height - 1)
            )
        )

        a = max(
            0,
            yc - band_h // 2
        )

        b = min(
            height,
            yc + band_h // 2 + 1
        )

        band = zone[a:b]

        if band.size == 0:
            continue

        profile = band.mean(axis=0)
        profile = gaussian_filter1d(
            profile,
            sigma=1.15
        )

        period = estimar_periodo(
            profile
        )

        period = float(
            np.clip(
                period,
                9.0,
                40.0
            )
        )

        contrast = float(
            np.std(profile)
        )

        profiles.append(profile)
        periods.append(period)
        contrasts.append(contrast)

    if not profiles:
        return (
            np.array([], dtype=np.int32),
            20.0,
            int(x0),
            int(x1)
        )

    # --------------------------------------------------------
    # REFINAR ESPACIADO CON DISTANCIAS ENTRE PICOS DE CADA BANDA
    # --------------------------------------------------------
    spacing_candidates = []

    for profile, p0 in zip(
        profiles,
        periods
    ):
        peaks, _ = find_peaks(
            profile,
            distance=max(
                5,
                int(p0 * 0.45)
            ),
            prominence=max(
                0.003,
                float(profile.max()) * 0.010
            ),
            height=max(
                0.006,
                float(profile.max()) * 0.018
            )
        )

        if len(peaks) >= 5:
            diffs = np.diff(
                peaks.astype(np.float32)
            )

            good = diffs[
                (diffs >= 8.0) &
                (diffs <= 42.0)
            ]

            if len(good):
                # El valor más pequeño repetitivo suele corresponder
                # al paso real; huecos secos pueden producir 2x o 3x.
                med = float(
                    np.median(good)
                )

                near = good[
                    (good > med * 0.62) &
                    (good < med * 1.38)
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
        # Mediana robusta de las autocorrelaciones de todas las bandas.
        spacing = float(
            np.median(
                periods
            )
        )

    spacing = float(
        np.clip(
            spacing,
            9.0,
            36.0
        )
    )

    # --------------------------------------------------------
    # EVITAR ARMÓNICO 2X:
    # comprobar si la mitad del espaciado también tiene señal
    # repetitiva fuerte en las bandas.
    # --------------------------------------------------------
    half = spacing / 2.0

    if half >= 8.0:
        score_full = 0.0
        score_half = 0.0

        for profile in profiles:
            for candidate, bucket in [
                (spacing, "full"),
                (half, "half")
            ]:
                best = -1.0

                phase_steps = max(
                    10,
                    int(round(candidate * 1.5))
                )

                for phase in np.linspace(
                    0,
                    candidate,
                    phase_steps,
                    endpoint=False
                ):
                    xs = np.arange(
                        phase,
                        len(profile),
                        candidate
                    )

                    if len(xs) < 4:
                        continue

                    values = []

                    for x in xs:
                        xi = int(round(x))
                        a = max(0, xi - 2)
                        b = min(
                            len(profile),
                            xi + 3
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
                            float(np.mean(values))
                        )

                if bucket == "full":
                    score_full += max(best, 0.0)
                else:
                    score_half += max(best, 0.0)

        # Solo usar media separación si explica casi igual o mejor
        # la estructura. Esto recupera hileras omitidas por armónicos.
        if (
            score_half >=
            score_full * 0.94
        ):
            spacing = half

    spacing = float(
        np.clip(
            spacing,
            8.0,
            36.0
        )
    )

    # --------------------------------------------------------
    # ELEGIR LA BANDA MÁS NÍTIDA PARA FIJAR LA FASE INICIAL
    # --------------------------------------------------------
    best_profile_index = int(
        np.argmax(
            np.asarray(
                contrasts,
                dtype=np.float32
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
        int(round(spacing * 3))
    )

    for phase in np.linspace(
        0,
        spacing,
        phase_steps,
        endpoint=False
    ):
        xs = np.arange(
            phase,
            width,
            spacing
        )

        if len(xs) < 4:
            continue

        values = []

        for x in xs:
            xi = int(round(x))

            a = max(
                0,
                xi - 2
            )

            b = min(
                width,
                xi + 3
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

    # --------------------------------------------------------
    # RETÍCULA COMPLETA: NO DEPENDE DE QUE CADA SURCO SEA VERDE
    # --------------------------------------------------------
    local_seeds = np.arange(
        best_phase,
        width,
        spacing,
        dtype=np.float32
    )

    # Solo quitar posiciones prácticamente encima del camino.
    edge_margin = max(
        1.0,
        spacing * 0.08
    )

    local_seeds = local_seeds[
        (local_seeds >= edge_margin) &
        (
            local_seeds <=
            width - 1 - edge_margin
        )
    ]

    # --------------------------------------------------------
    # REFINAMIENTO MUY PEQUEÑO.
    # Nunca mover una semilla suficiente para entrar al vecino.
    # --------------------------------------------------------
    full_profile = zone.mean(axis=0)

    # Para surcos curvos, no usar el promedio puro:
    # conservar la mejor evidencia entre varias bandas.
    stacked = np.vstack(
        profiles
    )

    robust_profile = (
        0.55 *
        gaussian_filter1d(
            full_profile,
            sigma=1.0
        )
        +
        0.45 *
        np.percentile(
            stacked,
            70,
            axis=0
        )
    )

    radius = max(
        1,
        int(
            spacing * 0.18
        )
    )

    refined = []

    for s in local_seeds:
        xi = int(round(s))

        a = max(
            0,
            xi - radius
        )

        b = min(
            width,
            xi + radius + 1
        )

        if b <= a:
            refined.append(
                float(s + x0)
            )
            continue

        local = robust_profile[
            a:b
        ]

        # Si no hay señal, CONSERVAR la semilla de la retícula.
        # Esto es lo que mantiene los surcos secos.
        if (
            local.size and
            float(local.max()) >
            max(
                0.005,
                float(
                    np.median(
                        robust_profile
                    )
                ) * 0.80
            )
        ):
            best = (
                a +
                int(
                    np.argmax(
                        local
                    )
                )
            )
        else:
            best = xi

        refined.append(
            float(best + x0)
        )

    seeds = np.asarray(
        refined,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # FORZAR ORDEN Y SEPARACIÓN.
    # Si dos refinamientos se acercaron, volverlos a su retícula.
    # --------------------------------------------------------
    if len(seeds):
        base = float(
            local_seeds[0] + x0
        )

        regular = (
            base +
            np.arange(
                len(seeds),
                dtype=np.float32
            ) * spacing
        )

        max_deviation = (
            spacing * 0.20
        )

        seeds = np.clip(
            seeds,
            regular - max_deviation,
            regular + max_deviation
        )

        # Garantizar que dos surcos nunca colapsen en uno.
        for i in range(
            1,
            len(seeds)
        ):
            minimum = (
                seeds[i - 1] +
                spacing * 0.62
            )

            if seeds[i] < minimum:
                seeds[i] = max(
                    minimum,
                    regular[i] -
                    max_deviation
                )

    return (
        np.rint(
            seeds
        ).astype(np.int32),
        float(spacing),
        int(x0),
        int(x1)
    )


# ============================================================
# RESPUESTA VISUAL DEL SURCO
# ============================================================
def crear_respuesta(bgr, green_mask):
    """
    Combina vegetación con textura vertical.
    Esto permite seguir también hileras secas.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Bordes verticales de las plantas/surcos.
    sx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    sx = np.abs(sx)

    p99 = np.percentile(sx, 99)

    if p99 > 0:
        sx = np.clip(sx / p99, 0, 1)
    else:
        sx[:] = 0

    # Unir los dos bordes de una hilera para dar señal al centro.
    sx = cv2.GaussianBlur(
        sx,
        (0, 0),
        sigmaX=2.0,
        sigmaY=1.0
    )

    gm = (green_mask > 0).astype(np.float32)

    gm = cv2.GaussianBlur(
        gm,
        (0, 0),
        sigmaX=2.0,
        sigmaY=1.3
    )

    response = 0.70 * gm + 0.30 * sx

    return response


# ============================================================
# SEGUIR UN SURCO SIN SALTAR AL VECINO
# ============================================================
def seguir_surco(response, green_mask, seed, left, right, y0, y1):
    """
    V3.3 - línea adaptativa SIN saltar al vecino.

    Cada hilera tiene un carril independiente.
    La trayectoria puede curvarse, pero:
    - nunca cruza el punto medio hacia el surco vecino;
    - siempre tiene una pequeña atracción hacia su posición nominal;
    - si desaparece la vegetación, conserva la trayectoria.
    """
    height = max(
        1,
        y1 - y0
    )

    step = max(
        4,
        int(
            height / 170
        )
    )

    ys = np.arange(
        y0,
        y1,
        step,
        dtype=np.int32
    )

    lane_width = max(
        5.0,
        float(
            right - left
        )
    )

    # Margen de seguridad interno.
    safety = max(
        1.0,
        lane_width * 0.12
    )

    hard_left = float(
        left + safety
    )

    hard_right = float(
        right - safety
    )

    if (
        hard_right -
        hard_left <
        2.0
    ):
        hard_left = float(
            left + 0.5
        )
        hard_right = float(
            right - 0.5
        )

    seed = float(
        np.clip(
            seed,
            hard_left,
            hard_right
        )
    )

    xs = []
    greens = []

    prev_x = seed
    velocity = 0.0

    # Nunca mirar todo el carril:
    # solo alrededor de la trayectoria actual.
    search_radius = max(
        2,
        int(
            lane_width * 0.24
        )
    )

    # Movimiento por paso extremadamente limitado.
    max_step_shift = max(
        0.65,
        lane_width * 0.055
    )

    # Desviación máxima acumulada respecto del centro nominal.
    # Permite curva, pero no alcanza el surco vecino.
    max_total_deviation = (
        lane_width * 0.32
    )

    for y in ys:
        ya = max(
            y0,
            y - step // 2 - 1
        )

        yb = min(
            y1,
            y + step // 2 + 2
        )

        predicted = (
            prev_x +
            velocity
        )

        predicted = float(
            np.clip(
                predicted,
                seed -
                max_total_deviation,
                seed +
                max_total_deviation
            )
        )

        predicted = float(
            np.clip(
                predicted,
                hard_left,
                hard_right
            )
        )

        a = max(
            int(np.floor(hard_left)),
            int(round(predicted)) -
            search_radius
        )

        b = min(
            int(np.ceil(hard_right)),
            int(round(predicted)) +
            search_radius
        )

        if b <= a:
            x_new = predicted
            local_peak = 0.0
            local_median = 0.0

        else:
            candidates = np.arange(
                a,
                b + 1,
                dtype=np.int32
            )

            visual = np.zeros(
                len(candidates),
                dtype=np.float32
            )

            for j, x in enumerate(
                candidates
            ):
                xa = max(
                    int(np.floor(hard_left)),
                    x - 3
                )

                xb = min(
                    int(np.ceil(hard_right)) + 1,
                    x + 4
                )

                patch = response[
                    ya:yb,
                    xa:xb
                ]

                visual[j] = (
                    float(
                        patch.mean()
                    )
                    if patch.size
                    else 0.0
                )

            # Penalización 1: alejarse del punto previsto.
            dist_pred = (
                np.abs(
                    candidates -
                    predicted
                ) /
                max(
                    search_radius,
                    1
                )
            )

            # Penalización 2: alejarse demasiado del surco nominal.
            dist_seed = (
                np.abs(
                    candidates -
                    seed
                ) /
                max(
                    max_total_deviation,
                    1.0
                )
            )

            score = (
                visual
                - 0.26 * dist_pred
                - 0.11 * dist_seed
            )

            best_index = int(
                np.argmax(
                    score
                )
            )

            candidate_best = float(
                candidates[
                    best_index
                ]
            )

            local_peak = float(
                np.max(
                    visual
                )
            )

            local_median = float(
                np.median(
                    visual
                )
            )

            strong_evidence = (
                local_peak >= 0.040
                and
                (
                    local_peak -
                    local_median
                ) >= 0.007
            )

            if strong_evidence:
                target = candidate_best
            else:
                # Hueco seco: no perseguir otra hilera.
                target = predicted
                velocity *= 0.45

            # Cambio máximo en un paso.
            dx = float(
                np.clip(
                    target -
                    prev_x,
                    -max_step_shift,
                    max_step_shift
                )
            )

            x_new = (
                prev_x +
                dx
            )

            # Inercia suave.
            velocity = (
                0.86 *
                velocity
                +
                0.14 *
                dx
            )

        # Barreras absolutas.
        x_new = float(
            np.clip(
                x_new,
                seed -
                max_total_deviation,
                seed +
                max_total_deviation
            )
        )

        x_new = float(
            np.clip(
                x_new,
                hard_left,
                hard_right
            )
        )

        xi = int(
            round(
                x_new
            )
        )

        # Vegetación alrededor de la trayectoria.
        xa = max(
            int(np.floor(hard_left)),
            xi - 4
        )

        xb = min(
            int(np.ceil(hard_right)) + 1,
            xi + 5
        )

        patch_green = green_mask[
            ya:yb,
            xa:xb
        ]

        green_score = (
            float(
                np.mean(
                    patch_green > 0
                )
            )
            if patch_green.size
            else 0.0
        )

        xs.append(
            x_new
        )

        greens.append(
            green_score
        )

        prev_x = x_new

    xs = np.asarray(
        xs,
        dtype=np.float32
    )

    greens = np.asarray(
        greens,
        dtype=np.float32
    )

    # Suavizado leve. No puede producir overshoot.
    if len(xs) >= 7:
        smooth = gaussian_filter1d(
            xs,
            sigma=1.0,
            mode="nearest"
        )

        # Limitar otra vez después de suavizar.
        xs = np.clip(
            smooth,
            seed -
            max_total_deviation,
            seed +
            max_total_deviation
        )

        xs = np.clip(
            xs,
            hard_left,
            hard_right
        )

    return (
        np.column_stack(
            [
                xs,
                ys
            ]
        ).astype(
            np.float32
        ),
        greens
    )


# ============================================================
# CLASIFICACIÓN VERDE / ROJO
# ============================================================
def estado_verde(green_scores):
    positive = green_scores[green_scores > 0]

    if len(positive) >= 4:
        threshold = float(
            np.clip(
                np.percentile(positive, 35) * 0.62,
                0.025,
                0.085
            )
        )
    else:
        threshold = 0.045

    state = green_scores >= threshold

    # Mayoría local para evitar segmentos rojo/verde de 1 píxel.
    if len(state) >= 5:
        original = state.copy()

        for i in range(2, len(state) - 2):
            state[i] = (
                np.sum(original[i-2:i+3]) >= 3
            )

    return state


# ============================================================
# ANÁLISIS PRINCIPAL
# ============================================================
def analizar(pil_img):
    original = cv2.cvtColor(
        np.asarray(pil_img),
        cv2.COLOR_RGB2BGR
    )

    h, w = original.shape[:2]

    mask0 = mascara_verde(original)

    angle = angulo_surcos(mask0)

    rot_img, M, Minv = rotar(
        original,
        angle
    )

    rot_mask = cv2.warpAffine(
        mask0,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT
    )

    # Encontrar parcela solo después de hacer verticales los surcos.
    y0, y1 = limites_verticales(rot_mask)

    seeds, spacing, x0_detectado, x1_detectado = semillas_surcos(
        rot_mask,
        y0,
        y1
    )

    if len(seeds) < 5:
        raise RuntimeError(
            "No se detectó una parcela de surcos suficientemente clara."
        )

    # --------------------------------------------------------
    # Limites laterales automáticos:
    # desde medio espacio antes del primer surco
    # hasta medio espacio después del último.
    # --------------------------------------------------------
    # Límites duros detectados por los caminos laterales.
    # Ninguna línea puede dibujarse fuera de esta parcela.
    x0 = int(x0_detectado)
    x1 = int(x1_detectado)

    # Ignorar absolutamente todo fuera de esta parcela.
    parcel_mask = np.zeros_like(rot_mask)

    parcel_mask[y0:y1, x0:x1] = (
        rot_mask[y0:y1, x0:x1]
    )

    response = crear_respuesta(
        rot_img,
        parcel_mask
    )

    tracks = []

    for i, seed in enumerate(seeds):
        # Fronteras a mitad de distancia entre hileras.
        if i == 0:
            left = x0
        else:
            left = int(
                (seeds[i - 1] + seed) / 2
            )

        if i == len(seeds) - 1:
            right = x1
        else:
            right = int(
                (seed + seeds[i + 1]) / 2
            )

        if right - left < 5:
            continue

        pts, green = seguir_surco(
            response,
            parcel_mask,
            int(seed),
            left,
            right,
            y0,
            y1
        )

        # ====================================================
        # V3.3 - CONSERVAR TODAS LAS HILERAS DE LA RETÍCULA
        # ====================================================
        # Si la geometría determinó que aquí corresponde un surco,
        # no lo eliminamos por falta de verde.
        # Un surco seco sigue siendo un surco.

        tracks.append({
            "points": pts,
            "green": green
        })

    # --------------------------------------------------------
    # DIBUJAR EN LA FOTO ORIGINAL
    # --------------------------------------------------------
    # La rotación se usa únicamente para analizar.
    # La fotografía final NO se rota, recorta ni deforma.
    final = original.copy()

    total_green = 0
    total_red = 0

    # Máscara del rectángulo lógico en coordenadas rotadas.
    # No se dibuja; solo sirve como diagnóstico.
    mask_preview_rot = np.zeros_like(rot_mask)
    mask_preview_rot[y0:y1, x0:x1] = 255

    mask_preview = cv2.warpAffine(
        mask_preview_rot,
        Minv,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT
    )

    for number, tr in enumerate(tracks, 1):
        pts_rot = tr["points"]
        green = tr["green"]

        state = estado_verde(
            green
        )

        # Transformar SOLO las coordenadas de la trayectoria
        # de vuelta a la imagen original.
        pts = aplicar_matriz(
            pts_rot,
            Minv
        )

        first_drawn = None

        for j in range(
            len(pts) - 1
        ):
            p1 = pts[j]
            p2 = pts[j + 1]

            if not (
                np.all(np.isfinite(p1))
                and np.all(np.isfinite(p2))
            ):
                continue

            x1p = int(
                np.clip(
                    round(p1[0]),
                    0,
                    w - 1
                )
            )

            y1p = int(
                np.clip(
                    round(p1[1]),
                    0,
                    h - 1
                )
            )

            x2p = int(
                np.clip(
                    round(p2[0]),
                    0,
                    w - 1
                )
            )

            y2p = int(
                np.clip(
                    round(p2[1]),
                    0,
                    h - 1
                )
            )

            # Protección final:
            # los dos extremos deben seguir dentro de la parcela lógica.
            if (
                mask_preview[y1p, x1p] == 0
                or mask_preview[y2p, x2p] == 0
            ):
                continue

            is_green = bool(
                state[j]
                or state[
                    min(
                        j + 1,
                        len(state) - 1
                    )
                ]
            )

            if is_green:
                color = (
                    0,
                    240,
                    0
                )
                total_green += 1
            else:
                color = (
                    0,
                    0,
                    255
                )
                total_red += 1

            cv2.line(
                final,
                (x1p, y1p),
                (x2p, y2p),
                color,
                1,
                cv2.LINE_AA
            )

            if first_drawn is None:
                first_drawn = (
                    x1p,
                    y1p
                )

        # Numeración cerca del inicio real de la línea.
        if first_drawn is not None:
            label = str(
                number
            )

            label_x = max(
                0,
                first_drawn[0] - 4
            )

            label_y = max(
                13,
                first_drawn[1] - 4
            )

            cv2.putText(
                final,
                label,
                (
                    label_x,
                    label_y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                (
                    255,
                    255,
                    255
                ),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                final,
                label,
                (
                    label_x,
                    label_y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                (
                    10,
                    10,
                    10
                ),
                1,
                cv2.LINE_AA
            )

    total = total_green + total_red

    green_pct = (
        100.0 * total_green / total
        if total else 0.0
    )

    red_pct = (
        100.0 - green_pct
        if total else 0.0
    )

    return {
        "image": cv2.cvtColor(
            final,
            cv2.COLOR_BGR2RGB
        ),
        "mask": mask_preview,
        "count": len(tracks),
        "green_pct": green_pct,
        "red_pct": red_pct,
        "angle": angle
    }





# ============================================================
# VIDEO: CALIDAD Y SIMILITUD
# ============================================================

def calidad_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )

    gm = mascara_verde(frame)
    green_ratio = float(
        np.mean(gm > 0)
    )

    # Textura: ayuda a distinguir viñedo de caminos/cielo.
    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )
    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    texture = float(
        np.mean(
            np.sqrt(
                gx * gx +
                gy * gy
            )
        )
    )

    # Puntaje simple: nitidez + algo de vegetación + textura.
    vegetation_factor = float(
        np.clip(
            green_ratio / 0.10,
            0.15,
            1.0
        )
    )

    score = (
        np.log1p(max(sharpness, 0.0))
        *
        vegetation_factor
        *
        np.log1p(max(texture, 0.0))
    )

    return {
        "sharpness": sharpness,
        "green_ratio": green_ratio,
        "texture": texture,
        "score": float(score)
    }


def firma_frame(frame):
    small = cv2.resize(
        frame,
        (96, 54),
        interpolation=cv2.INTER_AREA
    )

    hsv = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2HSV
    )

    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [24, 24],
        [0, 180, 0, 256]
    )

    cv2.normalize(
        hist,
        hist,
        0,
        1,
        cv2.NORM_MINMAX
    )

    return hist


def similitud_firmas(a, b):
    return float(
        cv2.compareHist(
            a,
            b,
            cv2.HISTCMP_CORREL
        )
    )


# ============================================================
# EXTRAER CANDIDATOS DEL VIDEO
# ============================================================

def extraer_candidatos(video_path):
    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            "No se pudo abrir el video."
        )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    total_frames = float(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    duration = (
        total_frames / fps
        if fps > 0
        else 0.0
    )

    # Muestreo automático:
    # videos largos = cada 4 s; cortos = cada 2 s.
    if duration > 180:
        sample_seconds = 4.0
    elif duration > 60:
        sample_seconds = 3.0
    else:
        sample_seconds = 2.0

    candidates = []

    t = 0.0

    while t <= duration:
        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            t * 1000.0
        )

        ok, frame = cap.read()

        if not ok:
            t += sample_seconds
            continue

        q = calidad_frame(
            frame
        )

        # Filtro muy permisivo:
        # solo elimina frames claramente borrosos o sin información.
        if (
            q["sharpness"] >= 20.0
            and
            q["texture"] >= 5.0
        ):
            candidates.append({
                "time": float(t),
                "frame": frame,
                "quality": q,
                "signature": firma_frame(frame)
            })

        t += sample_seconds

    cap.release()

    if not candidates:
        raise RuntimeError(
            "No se encontraron fotogramas suficientemente claros."
        )

    # --------------------------------------------------------
    # Elegir el mejor frame de cada ventana temporal.
    # --------------------------------------------------------
    window_seconds = 12.0

    groups = {}

    for item in candidates:
        key = int(
            item["time"] //
            window_seconds
        )

        groups.setdefault(
            key,
            []
        ).append(
            item
        )

    best_by_window = []

    for key in sorted(groups):
        best = max(
            groups[key],
            key=lambda x: x["quality"]["score"]
        )
        best_by_window.append(
            best
        )

    # --------------------------------------------------------
    # Eliminar escenas consecutivas casi idénticas.
    # --------------------------------------------------------
    filtered = []

    for item in best_by_window:
        if not filtered:
            filtered.append(item)
            continue

        previous = filtered[-1]

        similarity = similitud_firmas(
            previous["signature"],
            item["signature"]
        )

        # Si es prácticamente la misma vista, conservar la más nítida.
        if (
            similarity >= 0.975
            and
            item["time"] -
            previous["time"] <= 24.0
        ):
            if (
                item["quality"]["score"]
                >
                previous["quality"]["score"]
            ):
                filtered[-1] = item
        else:
            filtered.append(item)

    # No procesar demasiadas imágenes en una sola corrida.
    max_frames = 20

    if len(filtered) > max_frames:
        indexes = np.linspace(
            0,
            len(filtered) - 1,
            max_frames
        ).astype(int)

        filtered = [
            filtered[i]
            for i in indexes
        ]

    return {
        "fps": fps,
        "duration": duration,
        "sample_seconds": sample_seconds,
        "frames": filtered
    }


# ============================================================
# ANALIZAR FRAMES CON V3.3
# ============================================================

def analizar_frames_video(info):
    analyzed = []

    for index, item in enumerate(
        info["frames"],
        1
    ):
        frame = item["frame"]

        pil = Image.fromarray(
            cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )
        )

        try:
            result = analizar(
                pil
            )

            annotated = cv2.cvtColor(
                result["image"],
                cv2.COLOR_RGB2BGR
            )

            analyzed.append({
                "index": index,
                "time": item["time"],
                "count": int(result["count"]),
                "angle": float(result["angle"]),
                "green_pct": float(result["green_pct"]),
                "red_pct": float(result["red_pct"]),
                "quality": float(item["quality"]["score"]),
                "sharpness": float(item["quality"]["sharpness"]),
                "signature": item["signature"],
                "annotated": annotated
            })

        except Exception:
            continue

    if not analyzed:
        raise RuntimeError(
            "Los fotogramas fueron extraídos, pero no se pudo detectar una parcela clara."
        )

    # --------------------------------------------------------
    # Segunda deduplicación:
    # misma apariencia + conteo parecido = misma escena probable.
    # --------------------------------------------------------
    unique = []

    for item in analyzed:
        if not unique:
            unique.append(item)
            continue

        previous = unique[-1]

        similarity = similitud_firmas(
            previous["signature"],
            item["signature"]
        )

        count_diff = abs(
            item["count"] -
            previous["count"]
        )

        allowed_diff = max(
            3,
            int(
                round(
                    max(
                        item["count"],
                        previous["count"]
                    )
                    * 0.14
                )
            )
        )

        same_scene = (
            similarity >= 0.92
            and
            count_diff <= allowed_diff
            and
            item["time"] -
            previous["time"] <= 30.0
        )

        if same_scene:
            # Quedarse con la toma más nítida.
            if (
                item["sharpness"]
                >
                previous["sharpness"]
            ):
                unique[-1] = item
        else:
            unique.append(item)

    # Reenumerar.
    for i, item in enumerate(
        unique,
        1
    ):
        item["scene"] = i

    return unique


def formato_tiempo(seconds):
    seconds = int(round(seconds))
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes:02d}:{secs:02d}"


def crear_zip_resultados(items):
    mem = io.BytesIO()

    with zipfile.ZipFile(
        mem,
        "w",
        compression=zipfile.ZIP_DEFLATED
    ) as z:
        for item in items:
            ok, buf = cv2.imencode(
                ".jpg",
                item["annotated"],
                [
                    int(
                        cv2.IMWRITE_JPEG_QUALITY
                    ),
                    96
                ]
            )

            if ok:
                name = (
                    f"escena_{item['scene']:02d}_"
                    f"{formato_tiempo(item['time']).replace(':','-')}_"
                    f"{item['count']}_surcos.jpg"
                )

                z.writestr(
                    name,
                    buf.tobytes()
                )

        csv_buffer = io.StringIO()
        writer = csv.writer(
            csv_buffer
        )

        writer.writerow([
            "Escena",
            "Tiempo",
            "Surcos_estimados",
            "Verde_pct",
            "Rojo_pct",
            "Angulo"
        ])

        for item in items:
            writer.writerow([
                item["scene"],
                formato_tiempo(item["time"]),
                item["count"],
                f"{item['green_pct']:.1f}",
                f"{item['red_pct']:.1f}",
                f"{item['angle']:.1f}"
            ])

        z.writestr(
            "conteo_surcos_video.csv",
            csv_buffer.getvalue()
        )

    mem.seek(0)
    return mem.getvalue()


# ============================================================
# INTERFAZ V10.2 - BORRADO REAL DE ESCENAS
# ============================================================

if "video_v102_items" not in st.session_state:
    st.session_state.video_v102_items = None

if "video_v102_duration" not in st.session_state:
    st.session_state.video_v102_duration = None

if "video_v102_signature" not in st.session_state:
    st.session_state.video_v102_signature = None


left, right = st.columns(
    [4.6, 1.4],
    gap="medium"
)

uploaded = None

with right:
    st.markdown("### 1. Subir video")

    uploaded = st.file_uploader(
        "Selecciona el video",
        type=["mp4", "mov", "avi", "m4v"],
        label_visibility="collapsed"
    )

    st.caption(
        "Primero busca fotogramas claros y después cuenta los surcos."
    )

    # Si se carga un video diferente, limpiar el análisis anterior.
    if uploaded is not None:
        current_signature = (
            f"{uploaded.name}:{getattr(uploaded, 'size', 0)}"
        )

        if (
            st.session_state.video_v102_signature is not None
            and
            st.session_state.video_v102_signature != current_signature
        ):
            st.session_state.video_v102_items = None
            st.session_state.video_v102_duration = None

    analyze_button = st.button(
        "🎥 Analizar video",
        type="primary",
        use_container_width=True,
        disabled=uploaded is None
    )


# ============================================================
# ANALIZAR VIDEO
# ============================================================

if analyze_button and uploaded is not None:

    suffix = Path(uploaded.name).suffix or ".mp4"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as tmp:
        tmp.write(uploaded.getbuffer())
        temp_path = Path(tmp.name)

    try:
        progress = st.progress(
            0,
            text="Buscando los mejores fotogramas..."
        )

        info = extraer_candidatos(temp_path)

        progress.progress(
            35,
            text=(
                f"Se seleccionaron {len(info['frames'])} fotogramas. "
                "Contando surcos..."
            )
        )

        items = analizar_frames_video(info)

        progress.progress(
            100,
            text="Análisis terminado."
        )

        st.session_state.video_v102_items = items
        st.session_state.video_v102_duration = float(
            info["duration"]
        )
        st.session_state.video_v102_signature = (
            f"{uploaded.name}:{getattr(uploaded, 'size', 0)}"
        )

    except Exception as exc:
        st.error(
            "No se pudo completar el análisis del video."
        )
        st.exception(exc)
        st.session_state.video_v102_items = None

    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass


# ============================================================
# RESULTADOS
# ============================================================

items = st.session_state.video_v102_items

if items:

    duration_text = formato_tiempo(
        st.session_state.video_v102_duration or 0
    )

    estimated_total = int(
        sum(
            item["count"]
            for item in items
        )
    )

    with right:
        st.markdown("### Resultados")

        st.metric(
            "Duración",
            duration_text
        )

        st.metric(
            "Escenas conservadas",
            len(items)
        )

        st.metric(
            "Suma estimada",
            estimated_total
        )

        st.caption(
            "Cuando borras una escena, desaparece de la tabla, "
            "de la suma y del ZIP final."
        )

        zip_bytes = crear_zip_resultados(
            items
        )

        st.download_button(
            "⬇️ Descargar resultados conservados",
            zip_bytes,
            file_name="resultado_surcos_video_v10_2.zip",
            mime="application/zip",
            use_container_width=True
        )

        if st.button(
            "🔄 Volver a analizar el video",
            use_container_width=True
        ):
            st.session_state.video_v102_items = None
            st.session_state.video_v102_duration = None
            st.rerun()


    with left:
        st.subheader(
            "Conteo por escena / parcela candidata"
        )

        df = pd.DataFrame([
            {
                "Escena": item["scene"],
                "Tiempo": formato_tiempo(item["time"]),
                "Surcos estimados": item["count"],
                "Verde %": round(item["green_pct"], 1),
                "Rojo %": round(item["red_pct"], 1)
            }
            for item in items
        ])

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # CADA ESCENA TIENE SU BOTÓN DE BORRAR
        # ----------------------------------------------------
        for item in list(items):

            col_title, col_delete = st.columns(
                [5, 1]
            )

            with col_title:
                st.markdown(
                    f"### Escena {item['scene']} · "
                    f"{formato_tiempo(item['time'])} · "
                    f"{item['count']} surcos"
                )

            with col_delete:
                delete_clicked = st.button(
                    "🗑️ Borrar",
                    key=f"borrar_escena_{item['scene']}",
                    use_container_width=True
                )

            if delete_clicked:
                st.session_state.video_v102_items = [
                    x
                    for x in st.session_state.video_v102_items
                    if x["scene"] != item["scene"]
                ]

                st.rerun()

            st.image(
                cv2.cvtColor(
                    item["annotated"],
                    cv2.COLOR_BGR2RGB
                ),
                use_container_width=True
            )


elif uploaded is None:

    with left:
        st.info(
            "Sube el video del dron para comenzar."
        )


else:

    with left:
        st.video(uploaded)

        st.info(
            "Presiona “Analizar video”. "
            "Después podrás borrar una por una las escenas que no quieras."
        )
