import io
import base64
import json
import os
import csv
import math
import zipfile
import tempfile
import requests
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import UnivariateSpline
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload


# ============================================================
# LOGO TERROCORE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "terrocore(1) (1).png"


def cargar_logo_base64():
    """
    Convierte el logo PNG a Base64 para mostrarlo
    dentro del encabezado HTML de Streamlit.
    """
    try:
        if LOGO_PATH.exists():
            return base64.b64encode(
                LOGO_PATH.read_bytes()
            ).decode("utf-8")
    except Exception:
        pass

    return ""


LOGO_TERROCORE_BASE64 = cargar_logo_base64()


st.set_page_config(
    page_title="TerroCore image AI",
    page_icon="🍷",
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


def tr_diag_texto(texto):
    """Traduce únicamente textos del diagnóstico cuando el idioma es FR."""
    if st.session_state.idioma_terrocore == "ES":
        return str(texto or "")

    t = str(texto or "").strip()
    if not t:
        return ""

    exactos = {
        "centro": "centre",
        "izquierda": "gauche",
        "derecha": "droite",
        "medio": "moyen",
        "media": "moyenne",
        "alto": "élevé",
        "alta": "élevée",
        "bajo": "faible",
        "baja": "faible",
        "no determinado": "non déterminé",
        "no determinada": "non déterminée",

        "estrés hídrico localizado": "stress hydrique localisé",
        "distribución irregular del riego": "distribution irrégulière de l’irrigation",
        "fertilidad o materia orgánica desuniforme": "fertilité ou matière organique hétérogène",
