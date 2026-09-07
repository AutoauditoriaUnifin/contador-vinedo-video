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
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import UnivariateSpline

st.set_page_config(
    page_title="TerroCore image AI",
    page_icon="🍇",
    layout="wide"
)

# ============================================================
# IDIOMA ES / FR
# ============================================================

if "idioma_terrocore" not in st.session_state:
    st.session_state.idioma_terrocore = "ES"


def tr(es, fr):
    """Texto visible según el idioma seleccionado."""
    return es if st.session_state.idioma_terrocore == "ES" else fr


# ============================================================
# DISEÑO - COLOR VINO #722F37
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --wine: #722F37;
        --wine-dark: #4F1E26;
        --wine-mid: #5E2630;
        --wine-light: #8D4A53;
        --cream: #FFF7F3;
        --soft: #F2DDE0;
        --green: #22D34F;
        --red: #FF3B3B;
    }

    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .stApp {
        background:
            radial-gradient(circle at 50% -10%, rgba(255,255,255,0.06), transparent 35%),
            linear-gradient(180deg, #722F37 0%, #662832 100%) !important;
        color: #FFFFFF !important;
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0) !important;
    }

    .block-container {
        max-width: 1550px;
        padding-top: 0.4rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3, h4, h5, h6,
    p, label, .stMarkdown, .stCaption,
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
    }

    h1 {
        font-size: 2.8rem !important;
        line-height: 1.05 !important;
        margin-bottom: 0.2rem !important;
        letter-spacing: -0.03em;
    }

    h2, h3 {
        letter-spacing: -0.015em;
    }

    .terro-brand {
        display:flex;
        align-items:center;
        gap:14px;
        margin-top:0.25rem;
    }

    .terro-grape {
        font-size:3rem;
        line-height:1;
        filter: drop-shadow(0 3px 8px rgba(0,0,0,.18));
    }

    .terro-kicker {
        font-family: Georgia, serif;
        color:#F5DADD !important;
        font-size:1.25rem;
        margin-top:-2px;
    }

    .terro-sub {
        color: #FFF3F1 !important;
        font-size: 1rem;
        margin-top: 0.7rem;
        margin-bottom: 0.9rem;
        opacity: .96;
    }

    /* CONTENEDORES */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: rgba(255,255,255,0.22) !important;
        background: rgba(73, 20, 29, 0.14) !important;
        border-radius: 12px !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.04);
    }

    /* BOTONES */
    .stButton > button,
    .stDownloadButton > button {
        background: linear-gradient(180deg, #FFFDFC, #F6E9E7) !important;
        color: #722F37 !important;
        border: 1px solid #F2D5D8 !important;
        border-radius: 10px !important;
        font-weight: 800 !important;
        min-height: 42px !important;
        box-shadow: 0 3px 10px rgba(0,0,0,0.18) !important;
        opacity: 1 !important;
    }

    .stButton > button *,
    .stDownloadButton > button * {
        color: #722F37 !important;
        fill: #722F37 !important;
        font-weight: 800 !important;
        opacity: 1 !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: #FFFFFF !important;
        color: #4F1E26 !important;
        border-color: #FFFFFF !important;
        transform: translateY(-1px);
        box-shadow: 0 5px 14px rgba(0,0,0,0.23) !important;
    }

    .stButton > button:disabled,
    .stDownloadButton > button:disabled {
        background: #D7C0C4 !important;
        color: #69454B !important;
        border-color: #CDB1B6 !important;
        opacity: .78 !important;
        box-shadow: none !important;
    }

    .stButton > button:disabled *,
    .stDownloadButton > button:disabled * {
        color:#69454B !important;
        fill:#69454B !important;
        opacity:1 !important;
    }

    /* FILE UPLOADER */
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(79,30,38,.34) !important;
        border: 1.5px dashed rgba(255,255,255,.75) !important;
        border-radius: 10px !important;
        min-height: 96px !important;
    }

    [data-testid="stFileUploaderDropzone"] * {
        color: #FFFFFF !important;
    }

    [data-testid="stFileUploader"] button {
        background: #FFFDFC !important;
        color: #722F37 !important;
        border: 1px solid #F2D5D8 !important;
        border-radius: 8px !important;
        font-weight: 800 !important;
    }

    [data-testid="stFileUploader"] button * {
        color:#722F37 !important;
        fill:#722F37 !important;
    }

    /* RADIO */
    div[role="radiogroup"] label,
    div[role="radiogroup"] label *,
    [data-testid="stRadio"] * {
        color: #FFFFFF !important;
        font-weight: 700;
    }

    /* TABLAS */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(255,255,255,.18);
        border-radius: 10px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 14px rgba(0,0,0,.11);
    }

    /* MÉTRICAS */
    [data-testid="stMetric"] {
        background: rgba(96, 31, 42, .45);
        border: 1px solid rgba(255,255,255,.15);
        border-radius: 11px;
        padding: .8rem .9rem;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
    }

    /* ALERTAS */
    [data-testid="stAlert"] {
        background: rgba(255,255,255,.13) !important;
        border: 1px solid rgba(255,255,255,.24) !important;
        border-radius: 10px !important;
    }

    [data-testid="stAlert"] * {
        color:#FFFFFF !important;
    }

    /* SELECT */
    div[data-baseweb="select"] > div {
        background: rgba(79,30,38,.65) !important;
        color:#FFFFFF !important;
        border-color: rgba(255,255,255,.2) !important;
    }

    div[data-baseweb="select"] * {
        color:#FFFFFF !important;
    }

    .legend-bar {
        display:flex;
        gap:0;
        align-items:center;
        margin-top:-6px;
        margin-bottom:8px;
        width:max-content;
        border-radius:0 0 8px 8px;
        overflow:hidden;
        box-shadow: 0 3px 12px rgba(0,0,0,.16);
    }

    .legend-chip {
        background:#301419;
        color:white;
        padding:7px 14px;
        font-size:.86rem;
        border-right:1px solid rgba(255,255,255,.12);
    }

    .dot-green, .dot-red {
        width:13px;
        height:13px;
        border-radius:50%;
        display:inline-block;
        margin-right:7px;
        vertical-align:-1px;
    }

    .dot-green { background:#22D34F; }
    .dot-red { background:#FF3B3B; }

    .scene-card-title {
        color:white;
        font-weight:700;
        font-size:.95rem;
        margin-top:.2rem;
    }

    .scene-card-sub {
        color:#F5E5E7;
        font-size:.82rem;
        margin-top:-.25rem;
        margin-bottom:.35rem;
    }

    hr {
        border-color: rgba(255,255,255,0.18) !important;
    }

    @media (max-width: 900px) {
        h1 { font-size: 2.15rem !important; }
        .block-container { padding-left: .8rem; padding-right: .8rem; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# CABECERA + SELECTOR ES / FR
# ============================================================

brand_col, lang_col = st.columns(
    [7.2, 2.8],
    vertical_alignment="center"
)

with brand_col:
    st.markdown(
        """
        <div class="terro-brand">
            <div class="terro-grape">🍇</div>
            <div>
                <div style="font-family:Georgia,serif;font-size:2.65rem;font-weight:700;color:white;line-height:1.0;">
                    TerroCore image AI
                </div>
                <div class="terro-kicker">
                    """ +
                    (
                        "Análisis inteligente del viñedo"
                        if st.session_state.idioma_terrocore == "ES"
                        else "Analyse intelligente du vignoble"
                    ) +
                    """
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with lang_col:
    l1, l2 = st.columns(2)

    with l1:
        if st.button(
            "🇪🇸 Español",
            key="lang_es_v13",
            use_container_width=True,
            disabled=st.session_state.idioma_terrocore == "ES"
        ):
            st.session_state.idioma_terrocore = "ES"
            st.rerun()

    with l2:
        if st.button(
            "🇫🇷 Français",
            key="lang_fr_v13",
            use_container_width=True,
            disabled=st.session_state.idioma_terrocore == "FR"
        ):
            st.session_state.idioma_terrocore = "FR"
            st.rerun()

st.markdown(
    f'<div class="terro-sub">'
    f'{tr("Analiza video e imágenes del viñedo, sigue la forma local de los surcos y reduce los saltos de una hilera a otra.", "Analyse les vidéos et les images du vignoble, suit la forme locale des rangs et réduit les sauts d’un rang à l’autre.")}'
    f'</div>',
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
# SUAVIZAR TRAYECTORIA SIN CAMBIAR DE SURCO
# ============================================================

def suavizar_trayectoria_surco(
    points,
    spacing
):
    """
    Suaviza el surco para que se vea menos ondulado,
    pero conservando su trayectoria real.

    La idea es enderezar ligeramente la línea sobre su eje
    principal y limitar cualquier movimiento lateral para
    no brincar a la hilera vecina.
    """
    pts = np.asarray(
        points,
        dtype=np.float32
    )

    n = len(pts)

    if n < 7:
        return pts

    try:
        # Dirección principal del surco.
        axis = pts[-1] - pts[0]
        axis_norm = float(np.linalg.norm(axis))

        if axis_norm < 1e-6:
            return pts

        u = axis / axis_norm
        v = np.array(
            [-u[1], u[0]],
            dtype=np.float32
        )

        base = pts[0].copy()

        # Coordenadas del surco en su eje principal (s)
        # y su desplazamiento lateral (d).
        rel = pts - base
        s = rel @ u
        d = rel @ v

        window = min(
            17,
            n if n % 2 == 1 else n - 1
        )

        if window < 7:
            return pts

        d_smooth = savgol_filter(
            d,
            window_length=window,
            polyorder=2,
            mode='interp'
        )

        # Acercar un poco el trazo al eje, pero sin hacerlo rígido.
        d_mix = (
            0.55 * d_smooth +
            0.45 * d
        )

        reconstructed = (
            base[None, :] +
            s[:, None] * u[None, :] +
            d_mix[:, None] * v[None, :]
        ).astype(np.float32)

        # Limitar desplazamiento para no saltar a otro surco.
        delta = reconstructed - pts
        distance = np.linalg.norm(delta, axis=1)
        maximum_move = max(
            1.0,
            float(spacing) * 0.10
        )

        too_far = distance > maximum_move
        if np.any(too_far):
            scale = maximum_move / (distance[too_far] + 1e-6)
            delta[too_far] *= scale[:, None]
            reconstructed = pts + delta

        reconstructed[0] = pts[0]
        reconstructed[-1] = pts[-1]

        return reconstructed.astype(np.float32)

    except Exception:
        return pts


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

    Mejora importante:
    - penaliza saltos laterales grandes;
    - conserva una banda alrededor del centro del surco;
    - no deja que la línea cruce fácilmente a la hilera vecina.
    """
    h, w = response.shape

    support = cv2.erode(
        (
            component[
                "mask"
            ] > 0
        ).astype(np.uint8),
        np.ones(
            (5, 5),
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

    initial_direction = direction.copy()
    initial_perpendicular = np.array(
        [
            -initial_direction[1],
            initial_direction[0]
        ],
        dtype=np.float64
    )
    seed_point = point.copy()

    points = [
        point.copy()
    ]

    green_scores = [
        0.0
    ]

    weak_steps = 0
    step_length = 3.5
    search_radius = max(
        2,
        int(
            spacing *
            0.10
        )
    )
    max_lateral_drift = max(
        3.0,
        float(spacing) * 0.28
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
            local_coherence > 0.35
            and
            angle_change <
            np.deg2rad(
                18.0
            )
        ):
            direction = (
                0.93 *
                direction
                +
                0.07 *
                local_vector
            )
            direction /= (
                np.linalg.norm(
                    direction
                ) +
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

        for offset in np.linspace(
            -search_radius,
            search_radius,
            (
                search_radius * 2 + 1
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
                0 <= cx < w and
                0 <= cy < h
            ):
                continue
            if support[cy, cx] == 0:
                continue

            visual = float(response[cy, cx])
            coherent = float(coherence[cy, cx])
            local_green = float(green[cy, cx])

            candidate_theta = float(theta[cy, cx])
            angle_penalty = diferencia_angular_rad(
                candidate_theta,
                float(np.arctan2(direction[1], direction[0]) % np.pi)
            )

            lateral_from_seed = abs(
                float(
                    np.dot(
                        candidate - seed_point,
                        initial_perpendicular
                    )
                )
            )

            # Castigar zonas ya ocupadas y saltos a la hilera vecina.
            score = (
                1.15 * visual
                + 0.16 * coherent
                + 0.05 * local_green
                - 0.030 * abs(float(offset))
                - 0.30 * angle_penalty
                - 0.085 * max(0.0, lateral_from_seed - max_lateral_drift)
            )

            if occupancy[cy, cx] > 0:
                score -= 0.80

            # Si el punto se aleja demasiado del eje original,
            # descartar casi por completo.
            if lateral_from_seed > max_lateral_drift + spacing * 0.15:
                score -= 1.40

            # Verificación de cresta local: el centro del surco debe ser
            # mejor que sus laterales cercanos, si no puede ser otra hilera.
            side1 = candidate + perpendicular * max(1.5, spacing * 0.22)
            side2 = candidate - perpendicular * max(1.5, spacing * 0.22)
            s_ok = True
            side_penalty = 0.0
            for side in (side1, side2):
                sx = int(round(side[0]))
                sy = int(round(side[1]))
                if 0 <= sx < w and 0 <= sy < h:
                    sv = float(response[sy, sx])
                    if sv > visual + 0.015:
                        side_penalty += 0.40
                else:
                    s_ok = False
            score -= side_penalty

            if score > best_score:
                best_score = score
                best_point = candidate
                best_response = visual

        if best_point is None:
            break

        lateral_jump = float(
            abs(
                np.dot(
                    best_point - predicted,
                    perpendicular
                )
            )
        )

        if best_response < 0.050:
            weak_steps += 1
            best_point = predicted
            bx = int(round(best_point[0]))
            by = int(round(best_point[1]))
            if not (
                0 <= bx < w and
                0 <= by < h and
                support[by, bx] > 0
            ):
                break
        else:
            weak_steps = max(0, weak_steps - 1)

        # Si hay un salto lateral grande, no seguir por otra hilera.
        if lateral_jump > max(2.0, float(spacing) * 0.18):
            break

        if weak_steps > 8:
            break

        movement = best_point - point
        movement_norm = float(np.linalg.norm(movement))
        if movement_norm > 1e-6:
            movement /= movement_norm
            if np.dot(movement, direction) > 0.88:
                direction = (
                    0.95 * direction +
                    0.05 * movement
                )
                direction /= (
                    np.linalg.norm(direction) + 1e-9
                )

        point = best_point
        points.append(point.copy())

        px = int(round(point[0]))
        py = int(round(point[1]))
        y0 = max(0, py - 4)
        y1 = min(h, py + 5)
        x0 = max(0, px - 4)
        x1 = min(w, px + 5)
        patch = green[y0:y1, x0:x1]
        green_scores.append(
            float(np.mean(patch > 0)) if patch.size else 0.0
        )

    return (
        np.asarray(points, dtype=np.float32),
        np.asarray(green_scores, dtype=np.float32)
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

    # Suavizar pequeñas ondulaciones manteniendo
    # la trayectoria dentro del mismo surco.
    points = suavizar_trayectoria_surco(
        points,
        spacing
    )

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
                    40
                ) * 0.90,
                0.040,
                0.16
            )
        )
    else:
        threshold = 0.060

    state = values >= threshold

    # Solo corregir ruido aislado de un punto.
    if len(state) >= 3:
        original = state.copy()
        for i in range(1, len(state) - 1):
            if original[i - 1] == original[i + 1] and original[i] != original[i - 1]:
                state[i] = original[i - 1]

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
            tr("No se encontró una zona con patrón claro de surcos.", "Aucune zone présentant un motif clair de rangs n’a été détectée.")
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
            tr("La imagen no contiene suficiente superficie de viñedo para hacer un trazado confiable.", "L’image ne contient pas une surface de vignoble suffisante pour effectuer un tracé fiable.")
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

        # Rechazar componentes con muy poca vegetación real.
        # Esto ayuda a evitar techos, caminos y otras texturas lineales.
        component_zone = (
            component["mask"] > 0
        )

        if np.any(component_zone):
            component_green = float(
                np.mean(
                    green[
                        component_zone
                    ] > 0
                )
            )
        else:
            component_green = 0.0

        if component_green < 0.035:
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

            # Descartar semillas que caen sobre un surco ya trazado.
            occupied_hits = 0
            for px_test, py_test in np.rint(points[::max(1, len(points)//12)]).astype(np.int32):
                if 0 <= px_test < w and 0 <= py_test < h and occupancy[py_test, px_test] > 0:
                    occupied_hits += 1
            if occupied_hits >= 3:
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

            # Evitar líneas falsas sobre construcciones:
            # debe existir algo de vegetación a lo largo de la trayectoria.
            green_support = float(
                np.mean(
                    np.asarray(
                        green_scores
                    ) >= 0.02
                )
            )

            if green_support < 0.10:
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

                idx1 = min(
                    j,
                    len(states) - 1
                )
                idx2 = min(
                    j + 1,
                    len(states) - 1
                )

                segment_green_score = float(
                    0.5 * (
                        float(green_scores[idx1]) +
                        float(green_scores[idx2])
                    )
                )

                green_segment = bool(
                    states[idx1]
                    and
                    states[idx2]
                    and
                    segment_green_score >= 0.055
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
            tr("Se encontró vegetación, pero no un patrón repetitivo de surcos suficientemente claro.", "De la végétation a été détectée, mais le motif répétitif des rangs n’est pas suffisamment clair.")
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
            tr("No se pudo abrir el video.", "Impossible d’ouvrir la vidéo.")
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
            tr("No se encontraron fotogramas suficientemente claros.", "Aucune image suffisamment nette n’a été trouvée.")
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
            tr("Los fotogramas fueron extraídos, pero no se pudo detectar una parcela clara.", "Les images ont été extraites, mais aucune parcelle suffisamment claire n’a pu être détectée.")
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
# EXPORTACIÓN EXCEL
# ============================================================

def crear_excel_video(items):
    buffer = io.BytesIO()

    rows = []
    for item in items:
        rows.append({
            "Escena": item["scene"],
            "Tiempo": formato_tiempo(item["time"]),
            "Surcos_estimados": item["count"],
            "Verde_pct": round(item["green_pct"], 1),
            "Rojo_pct": round(item["red_pct"], 1),
            "Angulo": round(item["angle"], 1)
        })

    df = pd.DataFrame(rows)

    with pd.ExcelWriter(
        buffer,
        engine="openpyxl"
    ) as writer:
        df.to_excel(
            writer,
            index=False,
            sheet_name="Resultados"
        )

        ws = writer.book["Resultados"]
        ws.freeze_panes = "A2"

        for col in ws.columns:
            width = max(
                len(str(cell.value))
                if cell.value is not None
                else 0
                for cell in col
            ) + 2

            ws.column_dimensions[
                col[0].column_letter
            ].width = min(
                max(width, 12),
                24
            )

    buffer.seek(0)
    return buffer.getvalue()


def crear_excel_imagenes(items):
    buffer = io.BytesIO()

    rows = []
    for item in items:
        rows.append({
            "Imagen": item["name"],
            "Surcos_estimados": item["count"],
            "Verde_pct": round(item["green_pct"], 1),
            "Rojo_pct": round(item["red_pct"], 1),
            "Angulo": round(item["angle"], 1)
        })

    df = pd.DataFrame(rows)

    with pd.ExcelWriter(
        buffer,
        engine="openpyxl"
    ) as writer:
        df.to_excel(
            writer,
            index=False,
            sheet_name="Resultados"
        )

        ws = writer.book["Resultados"]
        ws.freeze_panes = "A2"

        for col in ws.columns:
            width = max(
                len(str(cell.value))
                if cell.value is not None
                else 0
                for cell in col
            ) + 2

            ws.column_dimensions[
                col[0].column_letter
            ].width = min(
                max(width, 12),
                28
            )

    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# INTERFAZ PRO V15 - LINEAS SUAVES + RESUMEN ARRIBA
# ============================================================

if "video_v13_items" not in st.session_state:
    st.session_state.video_v13_items = None

if "video_v13_duration" not in st.session_state:
    st.session_state.video_v13_duration = None

if "video_v13_signature" not in st.session_state:
    st.session_state.video_v13_signature = None

if "imagenes_v13_items" not in st.session_state:
    st.session_state.imagenes_v13_items = None

if "escena_activa_v13" not in st.session_state:
    st.session_state.escena_activa_v13 = 0


main_col, side_col = st.columns(
    [2.15, 1.0],
    gap="medium"
)

# ============================================================
# PANEL DERECHO - CONTROLES
# ============================================================

with side_col:

    with st.container(border=True):
        st.markdown(
            tr(
                "### 1. Tipo de archivo",
                "### 1. Type de fichier"
            )
        )

        modo = st.radio(
            tr(
                "Selecciona qué quieres analizar:",
                "Sélectionnez ce que vous souhaitez analyser :"
            ),
            options=["video", "imagenes"],
            format_func=lambda value: (
                tr("🎥 Video", "🎥 Vidéo")
                if value == "video"
                else tr("🖼️ Imágenes", "🖼️ Images")
            ),
            horizontal=True,
            label_visibility="collapsed",
            key="modo_v13"
        )

    if modo == "video":
        with st.container(border=True):
            st.markdown(
                tr(
                    "### 2. Subir archivo de video",
                    "### 2. Importer un fichier vidéo"
                )
            )

            uploaded_video = st.file_uploader(
                tr(
                    "Haz clic para subir un video o arrástralo aquí",
                    "Cliquez pour importer une vidéo ou déposez-la ici"
                ),
                type=["mp4", "mov", "avi", "m4v"],
                key="uploader_video_v13",
                help=tr(
                    "Formatos: MP4, MOV, AVI, M4V",
                    "Formats : MP4, MOV, AVI, M4V"
                )
            )

            st.caption(
                tr(
                    "Formatos: MP4, MOV, AVI, M4V · máximo configurado: 500 MB",
                    "Formats : MP4, MOV, AVI, M4V · maximum configuré : 500 Mo"
                )
            )

            if uploaded_video is not None:
                current_signature = (
                    f"{uploaded_video.name}:"
                    f"{getattr(uploaded_video, 'size', 0)}"
                )

                if (
                    st.session_state.video_v13_signature is not None
                    and
                    st.session_state.video_v13_signature
                    != current_signature
                ):
                    st.session_state.video_v13_items = None
                    st.session_state.video_v13_duration = None

            analizar_video = st.button(
                tr(
                    "▶ Analizar video",
                    "▶ Analyser la vidéo"
                ),
                type="primary",
                use_container_width=True,
                disabled=uploaded_video is None,
                key="analizar_video_v13"
            )

        if analizar_video and uploaded_video is not None:
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
                    text=tr(
                        "Buscando fotogramas útiles...",
                        "Recherche des images utiles..."
                    )
                )

                info = extraer_candidatos(
                    temp_path
                )

                progress.progress(
                    35,
                    text=tr(
                        f"Se seleccionaron {len(info['frames'])} fotogramas. Contando surcos...",
                        f"{len(info['frames'])} images ont été sélectionnées. Comptage des rangs..."
                    )
                )

                items = analizar_frames_video(
                    info
                )

                progress.progress(
                    100,
                    text=tr(
                        "Análisis terminado.",
                        "Analyse terminée."
                    )
                )

                st.session_state.video_v13_items = items
                st.session_state.video_v13_duration = float(
                    info["duration"]
                )
                st.session_state.video_v13_signature = (
                    f"{uploaded_video.name}:"
                    f"{getattr(uploaded_video, 'size', 0)}"
                )
                st.session_state.escena_activa_v13 = 0

            except Exception as exc:
                st.error(
                    tr(
                        "No se pudo completar el análisis del video.",
                        "L’analyse de la vidéo n’a pas pu être terminée."
                    )
                )
                st.caption(str(exc))
                st.session_state.video_v13_items = None

            finally:
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    else:
        with st.container(border=True):
            st.markdown(
                tr(
                    "### 2. Subir imágenes",
                    "### 2. Importer des images"
                )
            )

            uploaded_images = st.file_uploader(
                tr(
                    "Selecciona una o varias fotografías",
                    "Sélectionnez une ou plusieurs photographies"
                ),
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key="uploader_images_v13"
            )

            st.caption(
                tr(
                    "Puedes seleccionar varias imágenes desde el teléfono.",
                    "Vous pouvez sélectionner plusieurs images depuis votre téléphone."
                )
            )

            analizar_imagenes = st.button(
                tr(
                    "🖼 Analizar imágenes",
                    "🖼 Analyser les images"
                ),
                type="primary",
                use_container_width=True,
                disabled=not uploaded_images,
                key="analizar_images_v13"
            )

        if analizar_imagenes and uploaded_images:
            analyzed_images = []

            progress = st.progress(
                0,
                text=tr(
                    "Analizando fotografías...",
                    "Analyse des photographies..."
                )
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
                        "id": f"{index}_{uploaded_image.name}",
                        "name": uploaded_image.name,
                        "count": int(result["count"]),
                        "green_pct": float(result["green_pct"]),
                        "red_pct": float(result["red_pct"]),
                        "angle": float(result["angle"]),
                        "annotated": annotated
                    })

                except Exception as exc:
                    st.warning(
                        tr(
                            f"No se pudo analizar {uploaded_image.name}: {exc}",
                            f"Impossible d’analyser {uploaded_image.name} : {exc}"
                        )
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
                    text=tr(
                        f"Analizando imagen {index} de {total_images}...",
                        f"Analyse de l’image {index} sur {total_images}..."
                    )
                )

            st.session_state.imagenes_v13_items = analyzed_images
            st.session_state.escena_activa_v13 = 0


# ============================================================
# RECUPERAR RESULTADOS ACTIVOS
# ============================================================

if modo == "video":
    active_items = st.session_state.video_v13_items or []
    duration_text = formato_tiempo(
        st.session_state.video_v13_duration or 0
    )
else:
    active_items = st.session_state.imagenes_v13_items or []
    duration_text = "—"


# ============================================================
# RESUMEN ARRIBA - OCUPA EL ESPACIO IZQUIERDO VACÍO
# ============================================================

with main_col:

    if active_items:

        with st.container(
            border=True
        ):

            st.subheader(
                tr(
                    "Resumen del análisis",
                    "Résumé de l’analyse"
                )
            )

            m1, m2, m3 = st.columns(
                3
            )

            with m1:
                st.metric(
                    tr(
                        "Duración",
                        "Durée"
                    ),
                    duration_text
                )

            with m2:
                st.metric(
                    tr(
                        "Escenas útiles"
                        if modo == "video"
                        else "Imágenes",
                        "Scènes utiles"
                        if modo == "video"
                        else "Images"
                    ),
                    len(
                        active_items
                    )
                )

            with m3:
                st.metric(
                    tr(
                        "Surcos sumados",
                        "Rangs cumulés"
                    ),
                    int(
                        sum(
                            item["count"]
                            for item
                            in active_items
                        )
                    )
                )

            st.caption(
                tr(
                    "Resumen calculado con las escenas que conservas.",
                    "Résumé calculé à partir des scènes conservées."
                )
            )

            d1, d2 = st.columns(
                2
            )

            with d1:

                if modo == "video":
                    excel_top = crear_excel_video(
                        active_items
                    )
                else:
                    excel_top = crear_excel_imagenes(
                        active_items
                    )

                st.download_button(
                    tr(
                        "⬇ Descargar resultados (Excel)",
                        "⬇ Télécharger les résultats (Excel)"
                    ),
                    excel_top,
                    file_name=(
                        "TerroCore_resultados_video.xlsx"
                        if modo == "video"
                        else "TerroCore_resultados_imagenes.xlsx"
                    ),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="excel_top_v14"
                )

            with d2:

                if modo == "video":
                    zip_top = crear_zip_resultados(
                        active_items
                    )
                else:
                    zip_top = crear_zip_resultados_imagenes(
                        active_items
                    )

                st.download_button(
                    tr(
                        "⬇ Descargar imágenes (ZIP)",
                        "⬇ Télécharger les images (ZIP)"
                    ),
                    zip_top,
                    file_name=(
                        "TerroCore_imagenes_video.zip"
                        if modo == "video"
                        else "TerroCore_imagenes.zip"
                    ),
                    mime="application/zip",
                    use_container_width=True,
                    key="zip_top_v14"
                )

    else:

        with st.container(
            border=True
        ):
            st.subheader(
                tr(
                    "Resumen del análisis",
                    "Résumé de l’analyse"
                )
            )

            st.info(
                tr(
                    "El resumen aparecerá aquí después de analizar el video o las imágenes.",
                    "Le résumé apparaîtra ici après l’analyse de la vidéo ou des images."
                )
            )


# ============================================================
# RESULTADOS GENERALES
# ============================================================

if active_items:

    st.markdown("---")

    st.subheader(
        tr(
            "Resultados por escena / parcela candidata",
            "Résultats par scène / parcelle candidate"
        )
    )

    if modo == "video":
        table_rows = [
            {
                tr("Escena", "Scène"): item["scene"],
                tr("Tiempo", "Temps"): formato_tiempo(
                    item["time"]
                ),
                tr(
                    "Surcos estimados",
                    "Rangs estimés"
                ): item["count"],
                tr(
                    "Verde %",
                    "Vert %"
                ): round(
                    item["green_pct"],
                    1
                ),
                tr(
                    "Rojo %",
                    "Rouge %"
                ): round(
                    item["red_pct"],
                    1
                )
            }
            for item in active_items
        ]

    else:
        table_rows = [
            {
                tr(
                    "Imagen",
                    "Image"
                ): item["name"],
                tr(
                    "Surcos estimados",
                    "Rangs estimés"
                ): item["count"],
                tr(
                    "Verde %",
                    "Vert %"
                ): round(
                    item["green_pct"],
                    1
                ),
                tr(
                    "Rojo %",
                    "Rouge %"
                ): round(
                    item["red_pct"],
                    1
                )
            }
            for item in active_items
        ]

    st.dataframe(
        pd.DataFrame(
            table_rows
        ),
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # FOTOGRAMAS ANALIZADOS - UNA IMAGEN POR FILA
    # ========================================================
    st.subheader(
        tr(
            "Fotogramas analizados"
            if modo == "video"
            else "Imágenes analizadas",
            "Images analysées"
        )
    )

    for idx, item in enumerate(
        list(active_items)
    ):

        with st.container(border=True):

            # Imagen grande: una por fila
            st.image(
                cv2.cvtColor(
                    item["annotated"],
                    cv2.COLOR_BGR2RGB
                ),
                use_container_width=True
            )

            # Solo botón BORRAR
            if st.button(
                tr(
                    "🗑 Borrar",
                    "🗑 Supprimer"
                ),
                key=f"delete_v13_{modo}_{idx}",
                use_container_width=True
            ):
                if modo == "video":
                    del st.session_state.video_v13_items[idx]
                else:
                    del st.session_state.imagenes_v13_items[idx]

                st.session_state.escena_activa_v13 = max(
                    0,
                    min(
                        st.session_state.escena_activa_v13,
                        len(active_items) - 2
                    )
                )

                st.rerun()
