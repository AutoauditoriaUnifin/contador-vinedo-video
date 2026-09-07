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
    page_title="Contador de surcos",
    page_icon="🍇",
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

st.title("🍇 TerroCore image AI")
st.markdown(
    '<div class="sub">'
    'Extrae automáticamente los fotogramas más útiles, evita escenas consecutivas muy parecidas '
    'y estima los surcos visibles por escena/parcela candidata.'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# V12 - DETECTOR LOCAL DE SURCOS
# ============================================================
#
# Diferencia principal:
# - NO obliga toda la fotografía a una sola dirección.
# - Calcula la orientación local del surco en cada punto.
# - Puede seguir surcos verticales, horizontales, diagonales
#   y con curvas suaves.
# - Primero detecta regiones con patrón real de viñedo.
# - Los caminos y zonas sin patrón repetitivo ayudan a separar
#   parcelas.
# ============================================================


# ============================================================
# VEGETACIÓN
# ============================================================

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


# ============================================================
# DIFERENCIA ANGULAR
# ============================================================

def diferencia_angular_rad(
    a,
    b
):
    d = abs(
        float(a) -
        float(b)
    ) % np.pi

    return min(
        d,
        np.pi - d
    )


# ============================================================
# CAMPO DE ORIENTACIÓN LOCAL
# ============================================================

def campo_orientacion_local(
    bgr
):
    """
    Devuelve:
    - vegetación
    - dirección LOCAL del surco en cada píxel
    - coherencia de la dirección
    - respuesta visual del surco
    - energía de textura
    """
    green = mascara_verde(
        bgr
    ).astype(np.float32)

    gray = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2GRAY
    ).astype(np.float32) / 255.0

    smooth = cv2.GaussianBlur(
        gray,
        (0, 0),
        1.0
    )

    gx = cv2.Sobel(
        smooth,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        smooth,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    Jxx = cv2.GaussianBlur(
        gx * gx,
        (0, 0),
        3.0
    )

    Jyy = cv2.GaussianBlur(
        gy * gy,
        (0, 0),
        3.0
    )

    Jxy = cv2.GaussianBlur(
        gx * gy,
        (0, 0),
        3.0
    )

    coherence = (
        np.sqrt(
            (Jxx - Jyy) ** 2 +
            4.0 * Jxy ** 2
        )
        /
        (
            Jxx +
            Jyy +
            1e-8
        )
    )

    # Dirección del gradiente.
    gradient_angle = (
        0.5 *
        np.arctan2(
            2.0 * Jxy,
            Jxx - Jyy
        )
    )

    # La dirección del SURCO es perpendicular al gradiente.
    theta = (
        gradient_angle +
        np.pi / 2.0
    ) % np.pi

    # --------------------------------------------------------
    # RESPUESTA VISUAL
    # --------------------------------------------------------
    green_blur = cv2.GaussianBlur(
        green,
        (0, 0),
        1.7
    )

    local_background = cv2.GaussianBlur(
        gray,
        (0, 0),
        5.0
    )

    darkness = np.maximum(
        local_background -
        gray,
        0.0
    )

    p99 = float(
        np.percentile(
            darkness,
            99
        )
    )

    if p99 > 1e-6:
        darkness = np.clip(
            darkness / p99,
            0.0,
            1.0
        )
    else:
        darkness[:] = 0.0

    response = (
        (
            0.62 *
            green_blur
            +
            0.38 *
            darkness
        )
        *
        (
            0.35 +
            0.65 *
            coherence
        )
    )

    response = cv2.GaussianBlur(
        response.astype(np.float32),
        (0, 0),
        0.8
    )

    energy = (
        Jxx +
        Jyy
    )

    return (
        green,
        theta.astype(np.float32),
        coherence.astype(np.float32),
        response.astype(np.float32),
        energy.astype(np.float32)
    )


# ============================================================
# COMPONENTES / PARCELAS CON PATRÓN DE VIÑEDO
# ============================================================

def detectar_componentes_vinedo(
    bgr,
    tile=32
):
    """
    Divide la imagen en pequeñas celdas.

    Una celda se considera viñedo cuando tiene:
    - vegetación,
    - textura lineal,
    - una orientación local clara.

    Las celdas se conectan solo cuando sus direcciones son
    compatibles. Así no se impone una sola dirección a toda
    la fotografía.
    """
    (
        green,
        theta,
        coherence,
        response,
        energy
    ) = campo_orientacion_local(
        bgr
    )

    h, w = green.shape

    ny = (
        h +
        tile -
        1
    ) // tile

    nx = (
        w +
        tile -
        1
    ) // tile

    valid = np.zeros(
        (ny, nx),
        dtype=np.uint8
    )

    tile_angle = np.zeros(
        (ny, nx),
        dtype=np.float32
    )

    tile_weight = np.zeros(
        (ny, nx),
        dtype=np.float32
    )

    for iy in range(ny):
        for ix in range(nx):

            y0 = iy * tile
            y1 = min(
                h,
                (iy + 1) * tile
            )

            x0 = ix * tile
            x1 = min(
                w,
                (ix + 1) * tile
            )

            c = coherence[
                y0:y1,
                x0:x1
            ]

            th = theta[
                y0:y1,
                x0:x1
            ]

            g = green[
                y0:y1,
                x0:x1
            ]

            e = energy[
                y0:y1,
                x0:x1
            ]

            green_fraction = float(
                np.mean(
                    g > 0
                )
            )

            coherence_mean = float(
                np.mean(c)
            )

            energy_mean = float(
                np.mean(e)
            )

            # Promedio angular correcto para direcciones de 0..180°.
            weights = (
                np.maximum(
                    c - 0.20,
                    0.0
                )
                *
                (
                    0.30 +
                    0.70 * g
                )
            )

            z = np.sum(
                weights *
                np.exp(
                    1j *
                    2.0 *
                    th
                )
            )

            if abs(z) > 1e-7:
                local_angle = (
                    np.angle(z) /
                    2.0
                ) % np.pi
            else:
                local_angle = np.pi / 2.0

            tile_angle[
                iy,
                ix
            ] = local_angle

            tile_weight[
                iy,
                ix
            ] = (
                coherence_mean *
                (
                    0.20 +
                    green_fraction
                )
            )

            if (
                coherence_mean > 0.34
                and
                energy_mean > 0.00018
                and
                green_fraction > 0.018
            ):
                valid[
                    iy,
                    ix
                ] = 1

    # --------------------------------------------------------
    # UNION-FIND:
    # conectar solo celdas vecinas de orientación parecida.
    # Se usan 4 vecinos para que un camino tenga más facilidad
    # de separar dos parcelas.
    # --------------------------------------------------------
    parent = np.arange(
        ny * nx,
        dtype=np.int32
    )

    def find(a):
        while parent[a] != a:
            parent[a] = parent[
                parent[a]
            ]
            a = parent[a]

        return int(a)

    def union(a, b):
        ra = find(a)
        rb = find(b)

        if ra != rb:
            parent[rb] = ra

    for iy in range(ny):
        for ix in range(nx):

            if valid[
                iy,
                ix
            ] == 0:
                continue

            current = (
                iy * nx +
                ix
            )

            for dy, dx in [
                (1, 0),
                (0, 1)
            ]:
                jy = iy + dy
                jx = ix + dx

                if not (
                    0 <= jy < ny
                    and
                    0 <= jx < nx
                ):
                    continue

                if valid[
                    jy,
                    jx
                ] == 0:
                    continue

                if (
                    diferencia_angular_rad(
                        tile_angle[
                            iy,
                            ix
                        ],
                        tile_angle[
                            jy,
                            jx
                        ]
                    )
                    <=
                    np.deg2rad(
                        22.0
                    )
                ):
                    union(
                        current,
                        jy * nx + jx
                    )

    groups = {}

    for iy in range(ny):
        for ix in range(nx):

            if valid[
                iy,
                ix
            ] == 0:
                continue

            root = find(
                iy * nx + ix
            )

            groups.setdefault(
                root,
                []
            ).append(
                (
                    iy,
                    ix
                )
            )

    components = []

    minimum_tiles = max(
        8,
        int(
            ny *
            nx *
            0.020
        )
    )

    for cells in groups.values():

        if len(cells) < minimum_tiles:
            continue

        component_mask = np.zeros(
            (h, w),
            dtype=np.uint8
        )

        angles = []
        weights = []

        for iy, ix in cells:

            y0 = iy * tile
            y1 = min(
                h,
                (iy + 1) * tile
            )

            x0 = ix * tile
            x1 = min(
                w,
                (ix + 1) * tile
            )

            component_mask[
                y0:y1,
                x0:x1
            ] = 255

            angles.append(
                float(
                    tile_angle[
                        iy,
                        ix
                    ]
                )
            )

            weights.append(
                float(
                    tile_weight[
                        iy,
                        ix
                    ]
                )
            )

        angles = np.asarray(
            angles,
            dtype=np.float64
        )

        weights = np.asarray(
            weights,
            dtype=np.float64
        )

        z = np.sum(
            weights *
            np.exp(
                1j *
                2.0 *
                angles
            )
        )

        if abs(z) > 1e-7:
            mean_angle = (
                np.angle(z) /
                2.0
            ) % np.pi
        else:
            mean_angle = float(
                np.median(
                    angles
                )
            )

        ys, xs = np.where(
            component_mask > 0
        )

        if len(xs) == 0:
            continue

        components.append({
            "mask": component_mask,
            "angle": float(
                mean_angle
            ),
            "bbox": (
                int(xs.min()),
                int(ys.min()),
                int(xs.max()) + 1,
                int(ys.max()) + 1
            ),
            "tiles": int(
                len(cells)
            )
        })

    components.sort(
        key=lambda c:
        c["tiles"],
        reverse=True
    )

    return (
        green,
        theta,
        coherence,
        response,
        components,
        ny,
        nx
    )


# ============================================================
# ESPACIADO ENTRE SURCOS
# ============================================================

def estimar_espaciado_local(
    profile
):
    p = np.asarray(
        profile,
        dtype=np.float64
    )

    if len(p) < 20:
        return None

    p = gaussian_filter1d(
        p,
        sigma=1.0
    )

    trend = gaussian_filter1d(
        p,
        sigma=max(
            4.0,
            len(p) / 35.0
        )
    )

    p = (
        p -
        trend
    )

    p -= np.mean(
        p
    )

    if np.std(p) < 1e-7:
        return None

    ac = np.correlate(
        p,
        p,
        mode="full"
    )

    ac = ac[
        len(p) - 1:
    ]

    minimum = 6

    maximum = min(
        45,
        len(ac) - 1
    )

    if maximum <= minimum:
        return None

    segment = ac[
        minimum:
        maximum + 1
    ]

    peaks, _ = find_peaks(
        segment
    )

    if len(peaks) == 0:
        lag = (
            minimum +
            int(
                np.argmax(
                    segment
                )
            )
        )
    else:
        values = segment[
            peaks
        ]

        maximum_value = float(
            np.max(
                values
            )
        )

        strong = peaks[
            values >=
            maximum_value *
            0.55
        ]

        if len(strong):
            lag = (
                minimum +
                int(
                    strong[0]
                )
            )
        else:
            lag = (
                minimum +
                int(
                    peaks[
                        np.argmax(
                            values
                        )
                    ]
                )
            )

    return float(
        np.clip(
            lag,
            7.0,
            45.0
        )
    )


# ============================================================
# SEMILLAS DE UN COMPONENTE
# ============================================================

def semillas_componente(
    component,
    response
):
    """
    La orientación media se usa SOLO para colocar una semilla
    inicial en cada surco.

    Después de eso, la línea deja de depender del ángulo medio
    y sigue la orientación LOCAL.
    """
    h, w = response.shape

    x0, y0, x1, y1 = component[
        "bbox"
    ]

    pad = 16

    xa = max(
        0,
        x0 - pad
    )

    xb = min(
        w,
        x1 + pad
    )

    ya = max(
        0,
        y0 - pad
    )

    yb = min(
        h,
        y1 + pad
    )

    crop_response = response[
        ya:yb,
        xa:xb
    ]

    crop_mask = (
        component[
            "mask"
        ][
            ya:yb,
            xa:xb
        ] > 0
    ).astype(np.uint8)

    ch, cw = crop_response.shape

    if (
        ch < 20
        or
        cw < 20
    ):
        return (
            [],
            None
        )

    angle_deg = np.rad2deg(
        component[
            "angle"
        ]
    )

    rotation = (
        90.0 -
        angle_deg
    )

    M = cv2.getRotationMatrix2D(
        (
            cw / 2.0,
            ch / 2.0
        ),
        rotation,
        1.0
    )

    Minv = cv2.invertAffineTransform(
        M
    )

    rotated_response = cv2.warpAffine(
        crop_response,
        M,
        (cw, ch),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    rotated_mask = cv2.warpAffine(
        crop_mask,
        M,
        (cw, ch),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT
    )

    ys, xs = np.where(
        rotated_mask > 0
    )

    if len(xs) < 100:
        return (
            [],
            None
        )

    rx0 = int(
        xs.min()
    )

    rx1 = int(
        xs.max()
    ) + 1

    ry0 = int(
        ys.min()
    )

    ry1 = int(
        ys.max()
    ) + 1

    profiles = []

    # Varias franjas:
    # así una zona seca no borra el surco del perfil.
    for frac in np.linspace(
        0.20,
        0.80,
        5
    ):

        yc = int(
            ry0 +
            frac *
            (
                ry1 -
                ry0
            )
        )

        band_height = max(
            8,
            int(
                (
                    ry1 -
                    ry0
                ) *
                0.12
            )
        )

        a = max(
            ry0,
            yc -
            band_height // 2
        )

        b = min(
            ry1,
            yc +
            band_height // 2
        )

        if b <= a:
            continue

        local_mask = (
            rotated_mask[
                a:b,
                rx0:rx1
            ] > 0
        ).astype(np.float32)

        denominator = np.maximum(
            local_mask.sum(
                axis=0
            ),
            1.0
        )

        profile = (
            rotated_response[
                a:b,
                rx0:rx1
            ]
            *
            local_mask
        ).sum(
            axis=0
        ) / denominator

        profiles.append(
            gaussian_filter1d(
                profile,
                sigma=1.0
            )
        )

    if not profiles:
        return (
            [],
            None
        )

    stacked = np.vstack(
        profiles
    )

    profile = np.percentile(
        stacked,
        70,
        axis=0
    )

    spacing = estimar_espaciado_local(
        profile
    )

    if spacing is None:
        return (
            [],
            None
        )

    peaks, _ = find_peaks(
        profile,
        distance=max(
            4,
            int(
                spacing *
                0.55
            )
        ),
        prominence=max(
            0.008,
            float(
                profile.max()
            ) *
            0.040
        ),
        height=max(
            0.015,
            float(
                profile.max()
            ) *
            0.080
        )
    )

    if len(peaks) < 3:
        return (
            [],
            spacing
        )

    # --------------------------------------------------------
    # Retícula regular.
    # Esto ayuda a no perder una hilera seca.
    # --------------------------------------------------------
    mods = np.mod(
        peaks.astype(
            np.float32
        ),
        spacing
    )

    z = np.mean(
        np.exp(
            1j *
            2.0 *
            np.pi *
            mods /
            spacing
        )
    )

    phase = (
        (
            np.angle(z)
            %
            (
                2.0 *
                np.pi
            )
        )
        *
        spacing
        /
        (
            2.0 *
            np.pi
        )
    )

    candidates = np.arange(
        phase,
        len(profile),
        spacing
    )

    seeds = []

    for candidate in candidates:

        xr = float(
            candidate +
            rx0
        )

        xi = int(
            round(
                xr
            )
        )

        if not (
            0 <= xi < cw
        ):
            continue

        a = max(
            0,
            xi - 2
        )

        b = min(
            cw,
            xi + 3
        )

        yy = np.where(
            np.any(
                rotated_mask[
                    :,
                    a:b
                ] > 0,
                axis=1
            )
        )[0]

        if len(yy) < 8:
            continue

        # Usar el segmento continuo más largo.
        breaks = np.where(
            np.diff(
                yy
            ) > 1
        )[0]

        segments = np.split(
            yy,
            breaks + 1
        )

        segment = max(
            segments,
            key=len
        )

        if len(segment) < 8:
            continue

        yr = float(
            np.median(
                segment
            )
        )

        point_rotated = np.array(
            [
                xr,
                yr,
                1.0
            ],
            dtype=np.float32
        )

        point_crop = (
            point_rotated @
            Minv.T
        )

        seeds.append(
            (
                float(
                    point_crop[0] +
                    xa
                ),
                float(
                    point_crop[1] +
                    ya
                )
            )
        )

    return (
        seeds,
        float(
            spacing
        )
    )


# ============================================================
# TRAZADO LOCAL DE UN SURCO
# ============================================================

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
):
    """
    Sigue una sola dirección desde la semilla.

    En cada paso:
    1. consulta el ángulo LOCAL,
    2. avanza unos píxeles,
    3. busca el centro del surco cerca de la predicción,
    4. nunca abandona la región del viñedo,
    5. evita pegarse a una línea ya dibujada.
    """
    h, w = response.shape

    support = cv2.dilate(
        (
            component[
                "mask"
            ] > 0
        ).astype(np.uint8),
        np.ones(
            (17, 17),
            np.uint8
        ),
        iterations=1
    )

    point = np.asarray(
        seed,
        dtype=np.float64
    )

    direction = np.asarray(
        initial_direction,
        dtype=np.float64
    )

    direction /= (
        np.linalg.norm(
            direction
        ) +
        1e-9
    )

    points = [
        point.copy()
    ]

    green_scores = [
        0.0
    ]

    weak_steps = 0

    step_length = 4.0

    search_radius = max(
        2,
        int(
            spacing *
            0.22
        )
    )

    maximum_steps = max(
        120,
        int(
            2.2 *
            max(
                h,
                w
            ) /
            step_length
        )
    )

    for _ in range(
        maximum_steps
    ):

        xi = int(
            round(
                point[0]
            )
        )

        yi = int(
            round(
                point[1]
            )
        )

        if not (
            1 <= xi < w - 1
            and
            1 <= yi < h - 1
        ):
            break

        # ----------------------------------------------------
        # Dirección local del surco.
        # ----------------------------------------------------
        local_theta = float(
            theta[
                yi,
                xi
            ]
        )

        local_vector = np.array(
            [
                np.cos(
                    local_theta
                ),
                np.sin(
                    local_theta
                )
            ],
            dtype=np.float64
        )

        # La dirección de 0..180 no tiene signo.
        # Elegir el signo que continúa hacia adelante.
        if (
            np.dot(
                local_vector,
                direction
            ) < 0
        ):
            local_vector *= -1.0

        local_coherence = float(
            coherence[
                yi,
                xi
            ]
        )

        # No dejar que una textura lateral gire la línea de golpe.
        dot_value = float(
            np.clip(
                np.dot(
                    local_vector,
                    direction
                ),
                -1.0,
                1.0
            )
        )

        angle_change = float(
            np.arccos(
                dot_value
            )
        )

        if (
            local_coherence > 0.28
            and
            angle_change <
            np.deg2rad(
                40.0
            )
        ):
            direction = (
                0.78 *
                direction
                +
                0.22 *
                local_vector
            )

            direction /= (
                np.linalg.norm(
                    direction
                )
                +
                1e-9
            )

        predicted = (
            point +
            direction *
            step_length
        )

        perpendicular = np.array(
            [
                -direction[1],
                direction[0]
            ],
            dtype=np.float64
        )

        best_point = None
        best_score = -1e9
        best_response = 0.0
        best_green = 0.0

        # ----------------------------------------------------
        # Buscar el centro real cerca de la predicción.
        # ----------------------------------------------------
        for offset in np.linspace(
            -search_radius,
            search_radius,
            (
                search_radius *
                2 +
                1
            )
        ):

            candidate = (
                predicted +
                perpendicular *
                offset
            )

            cx = int(
                round(
                    candidate[0]
                )
            )

            cy = int(
                round(
                    candidate[1]
                )
            )

            if not (
                0 <= cx < w
                and
                0 <= cy < h
            ):
                continue

            if support[
                cy,
                cx
            ] == 0:
                continue

            visual = float(
                response[
                    cy,
                    cx
                ]
            )

            coherent = float(
                coherence[
                    cy,
                    cx
                ]
            )

            local_green = float(
                green[
                    cy,
                    cx
                ]
            )

            score = (
                visual
                +
                0.10 *
                coherent
                -
                0.018 *
                abs(
                    float(offset)
                )
            )

            # Evitar que una línea se pegue a otro surco ya trazado.
            if occupancy[
                cy,
                cx
            ] > 0:
                score -= 0.35

            if score > best_score:
                best_score = score
                best_point = candidate
                best_response = visual
                best_green = local_green

        if best_point is None:
            break

        # ----------------------------------------------------
        # Tramo seco:
        # seguir la trayectoria prevista durante un hueco corto.
        # No perseguir automáticamente otra hilera verde.
        # ----------------------------------------------------
        if best_response < 0.055:
            weak_steps += 1
            best_point = predicted

            bx = int(
                round(
                    best_point[0]
                )
            )

            by = int(
                round(
                    best_point[1]
                )
            )

            if not (
                0 <= bx < w
                and
                0 <= by < h
                and
                support[
                    by,
                    bx
                ] > 0
            ):
                break

        else:
            weak_steps = max(
                0,
                weak_steps - 1
            )

        # Camino/zona sin estructura demasiado larga = detener.
        if weak_steps > 10:
            break

        movement = (
            best_point -
            point
        )

        movement_norm = float(
            np.linalg.norm(
                movement
            )
        )

        if movement_norm > 1e-6:
            movement /= movement_norm

            # Suavizar cambios de trayectoria.
            if (
                np.dot(
                    movement,
                    direction
                ) > 0.72
            ):
                direction = (
                    0.85 *
                    direction
                    +
                    0.15 *
                    movement
                )

                direction /= (
                    np.linalg.norm(
                        direction
                    )
                    +
                    1e-9
                )

        point = best_point

        points.append(
            point.copy()
        )

        # Medir verde en una pequeña zona alrededor.
        px = int(
            round(
                point[0]
            )
        )

        py = int(
            round(
                point[1]
            )
        )

        y0 = max(
            0,
            py - 4
        )

        y1 = min(
            h,
            py + 5
        )

        x0 = max(
            0,
            px - 4
        )

        x1 = min(
            w,
            px + 5
        )

        patch = green[
            y0:y1,
            x0:x1
        ]

        green_scores.append(
            float(
                np.mean(
                    patch > 0
                )
            )
            if patch.size
            else 0.0
        )

    return (
        np.asarray(
            points,
            dtype=np.float32
        ),
        np.asarray(
            green_scores,
            dtype=np.float32
        )
    )


def trazar_surco_local(
    seed,
    component,
    response,
    theta,
    coherence,
    green,
    spacing,
    occupancy
):
    h, w = response.shape

    sx = int(
        np.clip(
            round(
                seed[0]
            ),
            0,
            w - 1
        )
    )

    sy = int(
        np.clip(
            round(
                seed[1]
            ),
            0,
            h - 1
        )
    )

    local_theta = float(
        theta[
            sy,
            sx
        ]
    )

    # Si la orientación local de la semilla es poco clara,
    # usar la orientación media SOLO para arrancar.
    if (
        float(
            coherence[
                sy,
                sx
            ]
        ) < 0.22
        or
        diferencia_angular_rad(
            local_theta,
            component[
                "angle"
            ]
        )
        >
        np.deg2rad(
            40
        )
    ):
        local_theta = float(
            component[
                "angle"
            ]
        )

    direction = np.array(
        [
            np.cos(
                local_theta
            ),
            np.sin(
                local_theta
            )
        ],
        dtype=np.float64
    )

    forward, forward_green = trazar_direccion(
        seed,
        direction,
        component,
        response,
        theta,
        coherence,
        green,
        spacing,
        occupancy
    )

    backward, backward_green = trazar_direccion(
        seed,
        -direction,
        component,
        response,
        theta,
        coherence,
        green,
        spacing,
        occupancy
    )

    if len(backward) > 1:
        points = np.vstack(
            [
                backward[
                    :0:-1
                ],
                forward
            ]
        )

        green_scores = np.concatenate(
            [
                backward_green[
                    :0:-1
                ],
                forward_green
            ]
        )
    else:
        points = forward
        green_scores = forward_green

    return (
        points,
        green_scores
    )


# ============================================================
# COLOR VERDE / ROJO
# ============================================================

def estados_color(
    green_scores
):
    values = np.asarray(
        green_scores,
        dtype=np.float32
    )

    positive = values[
        values > 0.005
    ]

    if len(positive) >= 5:
        threshold = float(
            np.clip(
                np.percentile(
                    positive,
                    32
                ) *
                0.65,
                0.025,
                0.12
            )
        )
    else:
        threshold = 0.045

    state = (
        values >=
        threshold
    )

    # Suavizar cambios aislados.
    if len(state) >= 5:
        original = state.copy()

        for i in range(
            2,
            len(state) - 2
        ):
            state[i] = (
                np.sum(
                    original[
                        i - 2:
                        i + 3
                    ]
                ) >= 3
            )

    return state


# ============================================================
# ANÁLISIS PRINCIPAL V12
# ============================================================

def analizar(
    pil_img
):
    original = cv2.cvtColor(
        np.asarray(
            pil_img.convert(
                "RGB"
            )
        ),
        cv2.COLOR_RGB2BGR
    )

    h, w = original.shape[:2]

    (
        green,
        theta,
        coherence,
        response,
        components,
        ny,
        nx
    ) = detectar_componentes_vinedo(
        original,
        tile=max(
            24,
            int(
                min(
                    h,
                    w
                ) /
                18
            )
        )
    )

    if not components:
        raise RuntimeError(
            "No se encontró una zona con patrón claro de surcos."
        )

    # --------------------------------------------------------
    # Evitar escenas donde solo hay una franja pequeña de viñedo.
    # Esto reduce líneas falsas sobre jardines, edificios o caminos.
    # --------------------------------------------------------
    total_component_tiles = sum(
        component[
            "tiles"
        ]
        for component in components
    )

    tile_coverage = (
        total_component_tiles /
        max(
            ny * nx,
            1
        )
    )

    if tile_coverage < 0.16:
        raise RuntimeError(
            "La imagen no contiene suficiente superficie de viñedo "
            "para hacer un trazado confiable."
        )

    final = original.copy()

    occupancy = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    all_tracks = []

    total_green = 0
    total_red = 0

    accepted_components = 0

    component_angles = []

    # Procesar primero los bloques más grandes.
    for component in components:

        seeds, spacing = semillas_componente(
            component,
            response
        )

        if (
            spacing is None
            or
            len(seeds) < 4
        ):
            continue

        accepted_components += 1

        component_angles.append(
            np.rad2deg(
                component[
                    "angle"
                ]
            )
        )

        for seed in seeds:

            points, green_scores = trazar_surco_local(
                seed,
                component,
                response,
                theta,
                coherence,
                green,
                spacing,
                occupancy
            )

            if len(points) < 8:
                continue

            # ------------------------------------------------
            # Conservar incluso surcos parciales.
            # La semilla ya proviene del patrón repetitivo.
            # ------------------------------------------------
            line_length = float(
                np.sum(
                    np.linalg.norm(
                        np.diff(
                            points,
                            axis=0
                        ),
                        axis=1
                    )
                )
            )

            if line_length < max(
                24.0,
                spacing * 2.5
            ):
                continue

            states = estados_color(
                green_scores
            )

            track_index = len(
                all_tracks
            ) + 1

            # Dibujar segmentos uno por uno.
            for j in range(
                len(points) - 1
            ):

                p1 = points[j]
                p2 = points[j + 1]

                if not (
                    np.all(
                        np.isfinite(
                            p1
                        )
                    )
                    and
                    np.all(
                        np.isfinite(
                            p2
                        )
                    )
                ):
                    continue

                x1 = int(
                    np.clip(
                        round(
                            p1[0]
                        ),
                        0,
                        w - 1
                    )
                )

                y1 = int(
                    np.clip(
                        round(
                            p1[1]
                        ),
                        0,
                        h - 1
                    )
                )

                x2 = int(
                    np.clip(
                        round(
                            p2[0]
                        ),
                        0,
                        w - 1
                    )
                )

                y2 = int(
                    np.clip(
                        round(
                            p2[1]
                        ),
                        0,
                        h - 1
                    )
                )

                green_segment = bool(
                    states[
                        min(
                            j,
                            len(
                                states
                            ) - 1
                        )
                    ]
                    or
                    states[
                        min(
                            j + 1,
                            len(
                                states
                            ) - 1
                        )
                    ]
                )

                if green_segment:
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
                    (
                        x1,
                        y1
                    ),
                    (
                        x2,
                        y2
                    ),
                    color,
                    1,
                    cv2.LINE_AA
                )

            # Numeración en la parte media para no amontonar arriba.
            middle = points[
                len(
                    points
                ) // 2
            ]

            mx = int(
                np.clip(
                    round(
                        middle[0]
                    ),
                    0,
                    w - 1
                )
            )

            my = int(
                np.clip(
                    round(
                        middle[1]
                    ),
                    0,
                    h - 1
                )
            )

            cv2.putText(
                final,
                str(
                    track_index
                ),
                (
                    mx + 4,
                    my - 4
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
                str(
                    track_index
                ),
                (
                    mx + 4,
                    my - 4
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

            # ------------------------------------------------
            # Marcar ocupación DESPUÉS de terminar el surco.
            # Esto evita que los siguientes se peguen al mismo.
            # ------------------------------------------------
            track_pixels = np.rint(
                points
            ).astype(
                np.int32
            )

            occupancy_line = np.zeros(
                (h, w),
                dtype=np.uint8
            )

            cv2.polylines(
                occupancy_line,
                [
                    track_pixels
                ],
                False,
                255,
                max(
                    2,
                    int(
                        spacing *
                        0.28
                    )
                ),
                cv2.LINE_AA
            )

            occupancy = np.maximum(
                occupancy,
                occupancy_line
            )

            all_tracks.append(
                points
            )

    if (
        accepted_components == 0
        or
        len(all_tracks) < 4
    ):
        raise RuntimeError(
            "Se encontró vegetación, pero no un patrón repetitivo "
            "de surcos suficientemente claro."
        )

    total = (
        total_green +
        total_red
    )

    green_pct = (
        100.0 *
        total_green /
        total
        if total
        else 0.0
    )

    red_pct = (
        100.0 -
        green_pct
        if total
        else 0.0
    )

    # Solo dato informativo:
    # promedio de orientaciones de componentes.
    mean_angle = float(
        np.mean(
            component_angles
        )
    ) if component_angles else 0.0

    return {
        "image": cv2.cvtColor(
            final,
            cv2.COLOR_BGR2RGB
        ),
        "count": int(
            len(
                all_tracks
            )
        ),
        "green_pct": float(
            green_pct
        ),
        "red_pct": float(
            red_pct
        ),
        "angle": mean_angle
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
# ZIP DE RESULTADOS DE IMÁGENES
# ============================================================

def crear_zip_resultados_imagenes(items):
    mem = io.BytesIO()

    with zipfile.ZipFile(
        mem,
        "w",
        compression=zipfile.ZIP_DEFLATED
    ) as z:

        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)

        writer.writerow([
            "Imagen",
            "Surcos_estimados",
            "Verde_pct",
            "Rojo_pct",
            "Angulo"
        ])

        for item in items:
            ok, buf = cv2.imencode(
                ".jpg",
                item["annotated"],
                [
                    int(cv2.IMWRITE_JPEG_QUALITY),
                    96
                ]
            )

            if ok:
                safe_name = Path(
                    item["name"]
                ).stem

                z.writestr(
                    f"{safe_name}_analizada.jpg",
                    buf.tobytes()
                )

            writer.writerow([
                item["name"],
                item["count"],
                f"{item['green_pct']:.1f}",
                f"{item['red_pct']:.1f}",
                f"{item['angle']:.1f}"
            ])

        z.writestr(
            "conteo_surcos_imagenes.csv",
            csv_buffer.getvalue()
        )

    mem.seek(0)
    return mem.getvalue()


# ============================================================
# INTERFAZ V10.3 - VIDEO E IMÁGENES
# ============================================================

# ----------------------------
# Estado de VIDEO
# ----------------------------
if "video_v103_items" not in st.session_state:
    st.session_state.video_v103_items = None

if "video_v103_duration" not in st.session_state:
    st.session_state.video_v103_duration = None

if "video_v103_signature" not in st.session_state:
    st.session_state.video_v103_signature = None

# ----------------------------
# Estado de IMÁGENES
# ----------------------------
if "imagenes_v103_items" not in st.session_state:
    st.session_state.imagenes_v103_items = None


left, right = st.columns(
    [4.6, 1.4],
    gap="medium"
)

with right:
    st.markdown("### 1. Tipo de archivo")

    modo = st.radio(
        "Selecciona qué quieres analizar:",
        [
            "🎥 Video",
            "🖼️ Imágenes"
        ],
        label_visibility="collapsed"
    )


# ============================================================
# MODO VIDEO
# ============================================================

if modo == "🎥 Video":

    uploaded_video = None

    with right:
        st.markdown("### 2. Subir video")

        uploaded_video = st.file_uploader(
            "Selecciona el video",
            type=["mp4", "mov", "avi", "m4v"],
            label_visibility="collapsed",
            key="uploader_video_v103"
        )

        st.caption(
            "Busca fotogramas útiles, cuenta los surcos "
            "y después puedes borrar las escenas que no quieras."
        )

        if uploaded_video is not None:
            current_signature = (
                f"{uploaded_video.name}:"
                f"{getattr(uploaded_video, 'size', 0)}"
            )

            if (
                st.session_state.video_v103_signature is not None
                and
                st.session_state.video_v103_signature != current_signature
            ):
                st.session_state.video_v103_items = None
                st.session_state.video_v103_duration = None

        analyze_video = st.button(
            "🎥 Analizar video",
            type="primary",
            use_container_width=True,
            disabled=uploaded_video is None,
            key="analizar_video_v103"
        )


    if analyze_video and uploaded_video is not None:

        suffix = Path(
            uploaded_video.name
        ).suffix or ".mp4"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as tmp:
            tmp.write(
                uploaded_video.getbuffer()
            )
            temp_path = Path(
                tmp.name
            )

        try:
            progress = st.progress(
                0,
                text="Buscando los mejores fotogramas..."
            )

            info = extraer_candidatos(
                temp_path
            )

            progress.progress(
                35,
                text=(
                    f"Se seleccionaron {len(info['frames'])} fotogramas. "
                    "Contando surcos..."
                )
            )

            items = analizar_frames_video(
                info
            )

            progress.progress(
                100,
                text="Análisis terminado."
            )

            st.session_state.video_v103_items = items
            st.session_state.video_v103_duration = float(
                info["duration"]
            )
            st.session_state.video_v103_signature = (
                f"{uploaded_video.name}:"
                f"{getattr(uploaded_video, 'size', 0)}"
            )

        except Exception as exc:
            st.error(
                "No se pudo completar el análisis del video."
            )
            st.exception(exc)
            st.session_state.video_v103_items = None

        finally:
            try:
                os.remove(
                    temp_path
                )
            except Exception:
                pass


    video_items = st.session_state.video_v103_items

    if video_items:

        duration_text = formato_tiempo(
            st.session_state.video_v103_duration or 0
        )

        estimated_total = int(
            sum(
                item["count"]
                for item in video_items
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
                len(video_items)
            )

            st.metric(
                "Suma estimada",
                estimated_total
            )

            st.caption(
                "Al borrar una escena desaparece de la tabla, "
                "de la suma, de la pantalla y del ZIP."
            )

            st.download_button(
                "⬇️ Descargar resultados",
                crear_zip_resultados(
                    video_items
                ),
                file_name="resultado_surcos_video.zip",
                mime="application/zip",
                use_container_width=True,
                key="descargar_video_v103"
            )

            if st.button(
                "🔄 Volver a analizar el video",
                use_container_width=True,
                key="reiniciar_video_v103"
            ):
                st.session_state.video_v103_items = None
                st.session_state.video_v103_duration = None
                st.rerun()


        with left:
            st.subheader(
                "Conteo por escena / parcela candidata"
            )

            df = pd.DataFrame([
                {
                    "Escena": item["scene"],
                    "Tiempo": formato_tiempo(
                        item["time"]
                    ),
                    "Surcos estimados": item["count"],
                    "Verde %": round(
                        item["green_pct"],
                        1
                    ),
                    "Rojo %": round(
                        item["red_pct"],
                        1
                    )
                }
                for item in video_items
            ])

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

            for item in list(
                video_items
            ):

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
                        key=(
                            f"borrar_video_"
                            f"{item['scene']}_v103"
                        ),
                        use_container_width=True
                    )

                if delete_clicked:
                    st.session_state.video_v103_items = [
                        x
                        for x
                        in st.session_state.video_v103_items
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

    elif uploaded_video is None:

        with left:
            st.info(
                "Sube el video del dron para comenzar."
            )

    else:

        with left:
            st.video(
                uploaded_video
            )

            st.info(
                "Presiona “Analizar video”."
            )


# ============================================================
# MODO IMÁGENES
# ============================================================

else:

    uploaded_images = None

    with right:
        st.markdown("### 2. Subir imágenes")

        uploaded_images = st.file_uploader(
            "Selecciona una o varias fotografías",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="uploader_imagenes_v103"
        )

        st.caption(
            "Puedes seleccionar varias fotografías desde el teléfono."
        )

        analyze_images = st.button(
            "🖼️ Analizar imágenes",
            type="primary",
            use_container_width=True,
            disabled=not uploaded_images,
            key="analizar_imagenes_v103"
        )


    if analyze_images and uploaded_images:

        analyzed_images = []

        progress = st.progress(
            0,
            text="Analizando fotografías..."
        )

        total_images = len(
            uploaded_images
        )

        for index, uploaded_image in enumerate(
            uploaded_images,
            1
        ):

            try:
                pil = Image.open(
                    uploaded_image
                ).convert("RGB")

                result = analizar(
                    pil
                )

                annotated = cv2.cvtColor(
                    result["image"],
                    cv2.COLOR_RGB2BGR
                )

                analyzed_images.append({
                    "id": (
                        f"{index}_"
                        f"{uploaded_image.name}"
                    ),
                    "name": uploaded_image.name,
                    "count": int(
                        result["count"]
                    ),
                    "green_pct": float(
                        result["green_pct"]
                    ),
                    "red_pct": float(
                        result["red_pct"]
                    ),
                    "angle": float(
                        result["angle"]
                    ),
                    "annotated": annotated
                })

            except Exception as exc:
                st.warning(
                    f"No se pudo analizar {uploaded_image.name}: {exc}"
                )

            progress.progress(
                int(
                    100 *
                    index /
                    max(
                        total_images,
                        1
                    )
                ),
                text=(
                    f"Analizando imagen "
                    f"{index} de {total_images}..."
                )
            )

        st.session_state.imagenes_v103_items = analyzed_images


    image_items = st.session_state.imagenes_v103_items

    if image_items:

        estimated_total = int(
            sum(
                item["count"]
                for item in image_items
            )
        )

        with right:
            st.markdown("### Resultados")

            st.metric(
                "Imágenes conservadas",
                len(image_items)
            )

            st.metric(
                "Suma estimada",
                estimated_total
            )

            st.caption(
                "Puedes borrar cualquier fotografía del resultado. "
                "Al borrarla deja de entrar en la suma y en el ZIP."
            )

            st.download_button(
                "⬇️ Descargar imágenes conservadas",
                crear_zip_resultados_imagenes(
                    image_items
                ),
                file_name="resultado_surcos_imagenes.zip",
                mime="application/zip",
                use_container_width=True,
                key="descargar_imagenes_v103"
            )

            if st.button(
                "🔄 Limpiar análisis de imágenes",
                use_container_width=True,
                key="limpiar_imagenes_v103"
            ):
                st.session_state.imagenes_v103_items = None
                st.rerun()


        with left:
            st.subheader(
                "Conteo por imagen"
            )

            df_images = pd.DataFrame([
                {
                    "Imagen": item["name"],
                    "Surcos estimados": item["count"],
                    "Verde %": round(
                        item["green_pct"],
                        1
                    ),
                    "Rojo %": round(
                        item["red_pct"],
                        1
                    )
                }
                for item in image_items
            ])

            st.dataframe(
                df_images,
                use_container_width=True,
                hide_index=True
            )

            for item in list(
                image_items
            ):

                col_title, col_delete = st.columns(
                    [5, 1]
                )

                with col_title:
                    st.markdown(
                        f"### {item['name']} · "
                        f"{item['count']} surcos"
                    )

                with col_delete:
                    delete_clicked = st.button(
                        "🗑️ Borrar",
                        key=(
                            f"borrar_imagen_"
                            f"{item['id']}_v103"
                        ),
                        use_container_width=True
                    )

                if delete_clicked:
                    st.session_state.imagenes_v103_items = [
                        x
                        for x
                        in st.session_state.imagenes_v103_items
                        if x["id"] != item["id"]
                    ]
                    st.rerun()

                st.image(
                    cv2.cvtColor(
                        item["annotated"],
                        cv2.COLOR_BGR2RGB
                    ),
                    use_container_width=True
                )

    elif not uploaded_images:

        with left:
            st.info(
                "Sube una o varias fotografías del viñedo."
            )

    else:

        with left:
            st.info(
                "Presiona “Analizar imágenes”."
            )
