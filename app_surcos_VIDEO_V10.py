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
        "compactación o variación física del suelo": "compaction ou variation physique du sol",
        "posible problema sanitario localizado": "possible problème phytosanitaire localisé",

        "inspeccionar en campo los tramos rojos": "inspecter sur le terrain les sections rouges",
        "comparar humedad y funcionamiento del riego entre zonas": "comparer l’humidité et le fonctionnement de l’irrigation entre les zones",
        "tomar muestras de suelo separadas en zona afectada y zona sana": "prélever séparément des échantillons de sol dans la zone affectée et la zone saine",
        "considerar análisis foliar para confirmar estado nutricional": "envisager une analyse foliaire pour confirmer l’état nutritionnel",
        "revisar raíces y presencia de plagas o enfermedades": "vérifier les racines ainsi que la présence de ravageurs ou de maladies",

        "Una fotografía por sí sola no permite afirmar qué nutriente falta. Nitrógeno, fósforo, potasio, magnesio, hierro u otros elementos pueden influir en el vigor, pero síntomas similares también pueden aparecer por falta o exceso de agua, salinidad, compactación, problemas de raíz, plagas o enfermedades. Para decidir una fertilización se recomienda confirmar con análisis de suelo y, de ser posible, análisis foliar.":
        "Une photographie seule ne permet pas d’identifier avec certitude le nutriment manquant. L’azote, le phosphore, le potassium, le magnésium, le fer ou d’autres éléments peuvent influencer la vigueur, mais des symptômes similaires peuvent aussi être causés par un manque ou un excès d’eau, la salinité, la compaction, des problèmes racinaires, des ravageurs ou des maladies. Avant de décider d’une fertilisation, il est recommandé de confirmer par une analyse du sol et, si possible, une analyse foliaire.",

        "Diagnóstico visual preliminar. No sustituye análisis de suelo, análisis foliar, revisión del riego ni diagnóstico agronómico en campo.":
        "Diagnostic visuel préliminaire. Il ne remplace pas une analyse du sol, une analyse foliaire, une vérification de l’irrigation ni un diagnostic agronomique sur le terrain.",

        "La mezcla de tramos verdes y rojos indica heterogeneidad en el vigor del viñedo. Puede existir un problema localizado de humedad, fertilidad o compactación.":
        "Le mélange de sections vertes et rouges indique une hétérogénéité de la vigueur du vignoble. Il peut exister un problème localisé d’humidité, de fertilité ou de compaction.",
    }

    if t.lower() in exactos:
        return exactos[t.lower()]
    if t in exactos:
        return exactos[t]

    # Diagnóstico dinámico con zona al final.
    prefijo = (
        "Se observa una afectación visual media, con mezcla de tramos vigorosos "
        "y tramos débiles o secos. La mayor afectación visual aparece en la zona "
    )
    if t.startswith(prefijo):
        zona = t[len(prefijo):].rstrip(".")
        zona_fr = exactos.get(zona.lower(), zona)
        return (
            "Une affectation visuelle moyenne est observée, avec un mélange de sections "
            "vigoureuses et de sections faibles ou sèches. L’affectation visuelle la plus "
            f"importante se situe dans la zone {zona_fr}."
        )

    return t


def probar_openai():
    """
    Prueba únicamente la conexión con OpenAI.
    No analiza imágenes ni modifica el detector de surcos.
    """
    try:
        api_key = st.secrets["OPENAI_API_KEY"]

        client = OpenAI(
            api_key=api_key
        )

        response = client.responses.create(
            model="gpt-5.6-luna",
            input="Responde únicamente con la palabra: CONECTADO"
        )

        return True, response.output_text

    except Exception as e:
        return False, str(e)


def limpiar_json_respuesta(texto):
    """
    Convierte la respuesta de OpenAI a JSON aunque el modelo agregue
    accidentalmente texto o bloques ```json ... ``` alrededor.
    """
    texto = (texto or "").strip()

    if texto.startswith("```json"):
        texto = texto.replace("```json", "", 1).strip()
    elif texto.startswith("```"):
        texto = texto.replace("```", "", 1).strip()

    if texto.endswith("```"):
        texto = texto[:-3].strip()

    # Primer intento: JSON puro.
    try:
        return json.loads(texto)
    except Exception:
        pass

    # Segundo intento: extraer el primer objeto JSON completo visible.
    inicio = texto.find("{")
    fin = texto.rfind("}")
    if inicio >= 0 and fin > inicio:
        return json.loads(texto[inicio:fin + 1])

    raise ValueError("OpenAI no devolvió un objeto JSON válido.")



def _analizar_bytes_con_ia(image_bytes, mime_type="image/jpeg"):
    """
    Analiza la escena con OpenAI y devuelve polígonos aproximados de
    zonas donde NO se deben dibujar surcos.

    Esta versión reduce la imagen antes de enviarla para evitar fallos
    por payload grande y hace un segundo intento automático si la primera
    llamada falla.
    """
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        client = OpenAI(api_key=api_key)

        if not image_bytes:
            raise ValueError("La imagen está vacía.")

        # ----------------------------------------------------
        # NORMALIZAR / REDUCIR IMAGEN ANTES DE ENVIAR A OPENAI
        # ----------------------------------------------------
        try:
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if bgr is None:
                raise ValueError("OpenCV no pudo abrir la imagen.")

            h, w = bgr.shape[:2]
            max_side = 1600
            scale = min(1.0, max_side / float(max(h, w)))

            if scale < 1.0:
                bgr = cv2.resize(
                    bgr,
                    (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                    interpolation=cv2.INTER_AREA
                )

            ok_jpg, encoded = cv2.imencode(
                ".jpg",
                bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), 86]
            )

            if not ok_jpg:
                raise ValueError("No se pudo recomprimir la imagen.")

            image_bytes_send = encoded.tobytes()
            mime_send = "image/jpeg"

        except Exception:
            # Respaldo: enviar los bytes originales.
            image_bytes_send = image_bytes
            mime_send = mime_type or "image/jpeg"

        image_b64 = base64.b64encode(image_bytes_send).decode("utf-8")
        data_url = f"data:{mime_send};base64,{image_b64}"

        prompt = """
Analiza esta imagen agrícola desde arriba.
Responde SOLO con JSON válido, sin markdown.

Necesito distinguir el viñedo de zonas donde NO deben dibujarse líneas de surcos.

Devuelve exactamente esta estructura:
{
  "es_vinedo": true,
  "hay_camino": false,
  "hay_techo": false,
  "hay_construccion": false,
  "analizar_surcos": true,
  "resumen": "Texto breve en español.",
  "zonas_excluir": [
    {
      "tipo": "camino",
      "puntos": [[120,80],[250,80],[260,900],[110,900]]
    }
  ]
}

REGLAS IMPORTANTES:
- Las coordenadas de cada punto son [x,y] normalizadas de 0 a 1000.
- (0,0) es la esquina superior izquierda y (1000,1000) la inferior derecha.
- En zonas_excluir incluye áreas claramente ajenas al patrón de hileras: caminos, calles,
  techos, edificios, patios, estacionamientos, bodegas, superficies artificiales grandes,
  árboles aislados, copas de árboles, setos, jardines y vegetación ornamental que NO forme
  parte de las hileras regulares del viñedo.
- Si hay árboles grandes en un borde de la parcela, delimita sus copas completas en zonas_excluir.
- NO excluyas huecos secos dentro de un surco.
- NO excluyas tierra visible entre hileras del viñedo.
- NO excluyas una hilera débil o sin vegetación si forma parte del patrón del viñedo.
- Usa polígonos de 4 a 12 puntos y trata de ajustarlos al contorno visible.
- Si no hay zonas que excluir, devuelve "zonas_excluir": [].
- Si existe viñedo útil aunque también haya techo/camino, "analizar_surcos" puede ser true.
- Si NO hay viñedo útil, "analizar_surcos" debe ser false.
- Si la imagen es principalmente techo/patio/construcción y casi no hay viñedo útil,
  "analizar_surcos" debe ser false.
- Devuelve SOLO JSON válido.
"""

        def llamar_modelo(model_name, image_detail):
            return client.responses.create(
                model=model_name,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {
                                "type": "input_image",
                                "image_url": data_url,
                                "detail": image_detail
                            }
                        ]
                    }
                ]
            )

        errores = []
        response = None

        # Primer intento: Luna, costo bajo.
        try:
            response = llamar_modelo("gpt-5.6-luna", "high")
        except Exception as e1:
            errores.append(f"Luna/high: {e1}")

        # Segundo intento: misma imagen, menor detalle para reducir carga.
        if response is None:
            try:
                response = llamar_modelo("gpt-5.6-luna", "low")
            except Exception as e2:
                errores.append(f"Luna/low: {e2}")

        if response is None:
            raise RuntimeError(" | ".join(errores))

        output_text = getattr(response, "output_text", "") or ""
        datos = limpiar_json_respuesta(output_text)

        if not isinstance(datos, dict):
            raise ValueError("La respuesta de IA no fue un objeto JSON.")

        if "zonas_excluir" not in datos:
            datos["zonas_excluir"] = []

        if not isinstance(datos.get("zonas_excluir"), list):
            datos["zonas_excluir"] = []

        return True, datos

    except Exception as e:
        return False, str(e)



# ============================================================
# BACKEND IA VISUAL - GPT IMAGE
# ============================================================

BACKEND_ANALYZE_URL = os.getenv(
    "BACKEND_ANALYZE_URL",
    "https://potential-disco-gxxq7gj969q7hpg75-8000.app.github.dev/analyze-image"
).strip()

# Si existe BACKEND_ANALYZE_URL en Streamlit Secrets, tiene prioridad.
try:
    if "BACKEND_ANALYZE_URL" in st.secrets:
        BACKEND_ANALYZE_URL = str(
            st.secrets["BACKEND_ANALYZE_URL"]
        ).strip()
except Exception:
    pass


def _descargar_imagen_resultado_backend(data):
    """Descarga la imagen editada por el backend y la devuelve en BGR."""
    url = (
        data.get("imagen_resultado_url_publica")
        or data.get("imagen_resultado_url")
    )

    if not url:
        raise RuntimeError("El backend no devolvió imagen_resultado_url.")

    if url.startswith("/"):
        base = BACKEND_ANALYZE_URL.rsplit("/analyze-image", 1)[0]
        url = base + url

    response = requests.get(url, timeout=300)
    response.raise_for_status()

    pil_result = Image.open(
        io.BytesIO(response.content)
    ).convert("RGB")

    rgb = np.asarray(pil_result)
    bgr = rgb[:, :, ::-1].copy()

    return bgr, url


def procesar_imagen_backend_ia(uploaded_file):
    """
    Envía la imagen al backend TerraCore IA.
    El backend usa GPT Image para editar directamente la fotografía.
    No usa el detector local para crear las líneas.
    """
    try:
        if not BACKEND_ANALYZE_URL:
            raise RuntimeError("BACKEND_ANALYZE_URL no está configurada.")

        image_bytes = uploaded_file.getvalue()
        if not image_bytes:
            raise RuntimeError("La imagen está vacía.")

        mime_type = uploaded_file.type or "image/jpeg"

        files = {
            "file": (
                uploaded_file.name,
                image_bytes,
                mime_type
            )
        }

        response = requests.post(
            BACKEND_ANALYZE_URL,
            files=files,
            timeout=600
        )

        try:
            data = response.json()
        except Exception:
            raise RuntimeError(
                f"El backend respondió HTTP {response.status_code}, "
                f"pero no devolvió JSON válido: {response.text[:1200]}"
            )

        if response.status_code != 200 or not data.get("ok", False):
            raise RuntimeError(
                data.get("error")
                or f"Backend HTTP {response.status_code}"
            )

        annotated, result_url = _descargar_imagen_resultado_backend(data)

        # El backend de edición visual actual no devuelve un conteo técnico.
        # Si más adelante agrega 'analisis.surcos_estimados', la app lo toma automáticamente.
        analisis = data.get("analisis") or {}
        count = int(
            analisis.get("surcos_estimados")
            or data.get("surcos_estimados")
            or 0
        )

        return True, {
            "backend": data,
            "annotated": annotated,
            "result_url": result_url,
            "count": count,
            "green_pct": float(
          analisis.get("verde_pct")
          or data.get("verde_pct")
          or 0.0
          ),

            "red_pct": float(
          analisis.get("rojo_pct")
          or data.get("rojo_pct")
          or 0.0
          ), 
            "angle": float(
                analisis.get("orientacion_principal_grados")
                or 0.0
            ),

            # Diagnóstico agronómico devuelto por el backend.
            # Se conservan todos los campos anteriores.
            "zona_mas_afectada": (
                analisis.get("zona_mas_afectada")
                or data.get("zona_mas_afectada")
                or "No determinada"
            ),
            "nivel_afectacion_visual": (
                analisis.get("nivel_afectacion_visual")
                or analisis.get("nivel_visual")
                or data.get("nivel_afectacion_visual")
                or data.get("nivel_visual")
                or "No determinado"
            ),
            "diagnostico_visual": (
                analisis.get("diagnostico_visual")
                or data.get("diagnostico_visual")
                or ""
            ),
            "causas_probables": (
                analisis.get("causas_probables")
                or analisis.get("motivos_probables")
                or data.get("causas_probables")
                or data.get("motivos_probables")
                or []
            ),
            "explicacion_nutrientes": (
                analisis.get("explicacion_nutrientes")
                or data.get("explicacion_nutrientes")
                or ""
            ),
            "recomendaciones_iniciales": (
                analisis.get("recomendaciones_iniciales")
                or (
                    [analisis.get("recomendacion_corta")]
                    if analisis.get("recomendacion_corta")
                    else []
                )
                or data.get("recomendaciones_iniciales")
                or (
                    [data.get("recomendacion_corta")]
                    if data.get("recomendacion_corta")
                    else []
                )
                or []
            ),
            "nota_diagnostico": (
                analisis.get("nota_diagnostico")
                or data.get("nota_diagnostico")
                or ""
            ),
            "detalle_zonas": (
                analisis.get("detalle_zonas")
                or data.get("detalle_zonas")
                or {}
            ),

            "metodo": data.get("metodo", "gpt-image")
        }

    except Exception as e:
        return False, str(e)


def analizar_1_imagen_con_ia(uploaded_file):
    """Prueba manual: procesa una sola imagen usando el backend IA visual."""
    return procesar_imagen_backend_ia(uploaded_file)



def analizar_pil_con_ia(pil_img):
    """
    Versión usada internamente por el análisis normal de imágenes.
    """
    buffer = io.BytesIO()
    pil_img.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=90
    )

    return _analizar_bytes_con_ia(
        buffer.getvalue(),
        "image/jpeg"
    )



def crear_mascara_exclusion_ia(datos_ia, width, height):
    """
    Convierte zonas_excluir (0..1000) a una máscara OpenCV.
    255 = zona donde NO se deben dibujar surcos.
    """
    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    zonas = datos_ia.get(
        "zonas_excluir",
        []
    ) if isinstance(datos_ia, dict) else []

    for zona in zonas:
        if not isinstance(zona, dict):
            continue

        puntos = zona.get(
            "puntos",
            []
        )

        if not isinstance(puntos, list) or len(puntos) < 3:
            continue

        polygon = []

        for punto in puntos[:20]:
            if (
                not isinstance(punto, (list, tuple))
                or
                len(punto) < 2
            ):
                continue

            try:
                nx = float(punto[0])
                ny = float(punto[1])
            except Exception:
                continue

            nx = float(np.clip(nx, 0, 1000))
            ny = float(np.clip(ny, 0, 1000))

            px = int(round(nx / 1000.0 * (width - 1)))
            py = int(round(ny / 1000.0 * (height - 1)))

            polygon.append([px, py])

        if len(polygon) >= 3:
            poly = np.asarray(
                polygon,
                dtype=np.int32
            )

            cv2.fillPoly(
                mask,
                [poly],
                255
            )

    # Un pequeño margen de seguridad alrededor de caminos/techos.
    if np.any(mask > 0):
        kernel_size = max(
            3,
            int(round(min(width, height) * 0.004))
        )

        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel_size = min(kernel_size, 15)

        kernel = np.ones(
            (kernel_size, kernel_size),
            dtype=np.uint8
        )

        mask = cv2.dilate(
            mask,
            kernel,
            iterations=1
        )

    return mask




# ============================================================
# HISTORIAL PERSISTENTE - GOOGLE DRIVE + GOOGLE SHEETS
# ============================================================

HISTORIAL_SHEET_NAME = "HistorialTerroCore"

HISTORIAL_HEADERS = [
    "ID",
    "Fecha",
    "Nombre",
    "ImagenOriginalFileID",
    "ImagenProcesadaFileID",
    "Surcos",
    "VerdePct",
    "RojoPct",
    "AmarilloPct",
    "NivelVisual",
    "ZonaMasAfectada",
    "DiagnosticoVisual",
    "CausasProbables",
    "ExplicacionNutrientes",
    "Recomendaciones",
    "NotaDiagnostico",
]


def _secret_text(nombre, default=""):
    try:
        if nombre in st.secrets:
            return str(st.secrets[nombre]).strip()
    except Exception:
        pass
    return str(default).strip()


def _service_account_info():
    """
    Acepta cualquiera de estas dos formas en Streamlit Secrets:

    [gcp_service_account]
    type="service_account"
    ...

    o

    GCP_SERVICE_ACCOUNT_JSON='{"type":"service_account", ...}'
    """
    try:
        if "gcp_service_account" in st.secrets:
            return dict(st.secrets["gcp_service_account"])
    except Exception:
        pass

    raw = _secret_text("GCP_SERVICE_ACCOUNT_JSON")

    if raw:
        try:
            return json.loads(raw)
        except Exception as exc:
            raise RuntimeError(
                f"GCP_SERVICE_ACCOUNT_JSON no contiene JSON válido: {exc}"
            )

    return {}


def historial_google_configurado():
    return bool(
        _service_account_info()
        and _secret_text("GDRIVE_PARENT_FOLDER_ID")
        and _secret_text("GSHEET_ID")
    )


@st.cache_resource(show_spinner=False)
def _google_clients_cached(service_account_json, parent_folder_id, sheet_id):
    """
    Crea los clientes una sola vez por sesión de Streamlit.
    """
    info = json.loads(service_account_json)

    scopes = [
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/spreadsheets",
    ]

    credentials = service_account.Credentials.from_service_account_info(
        info,
        scopes=scopes,
    )

    drive_service = build(
        "drive",
        "v3",
        credentials=credentials,
        cache_discovery=False,
    )

    sheets_service = build(
        "sheets",
        "v4",
        credentials=credentials,
        cache_discovery=False,
    )

    return drive_service, sheets_service


def obtener_google_clients():
    info = _service_account_info()

    if not info:
        raise RuntimeError(
            "Falta la cuenta de servicio de Google en Streamlit Secrets."
        )

    parent_folder_id = _secret_text("GDRIVE_PARENT_FOLDER_ID")
    sheet_id = _secret_text("GSHEET_ID")

    if not parent_folder_id:
        raise RuntimeError(
            "Falta GDRIVE_PARENT_FOLDER_ID en Streamlit Secrets."
        )

    if not sheet_id:
        raise RuntimeError(
            "Falta GSHEET_ID en Streamlit Secrets."
        )

    return _google_clients_cached(
        json.dumps(info, sort_keys=True),
        parent_folder_id,
        sheet_id,
    )


def _escapar_query_drive(valor):
    return str(valor).replace("\\", "\\\\").replace("'", "\\'")


def obtener_o_crear_subcarpeta_drive(nombre):
    drive_service, _ = obtener_google_clients()

    parent_id = _secret_text("GDRIVE_PARENT_FOLDER_ID")

    nombre_q = _escapar_query_drive(nombre)

    query = (
        f"name = '{nombre_q}' "
        f"and '{parent_id}' in parents "
        "and mimeType = 'application/vnd.google-apps.folder' "
        "and trashed = false"
    )

    response = drive_service.files().list(
        q=query,
        spaces="drive",
        fields="files(id,name)",
        pageSize=10,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()

    files = response.get("files", [])

    if files:
        return files[0]["id"]

    metadata = {
        "name": nombre,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }

    created = drive_service.files().create(
        body=metadata,
        fields="id",
        supportsAllDrives=True,
    ).execute()

    return created["id"]


def subir_bytes_google_drive(
    contenido,
    nombre_archivo,
    mime_type,
    subcarpeta,
):
    drive_service, _ = obtener_google_clients()

    folder_id = obtener_o_crear_subcarpeta_drive(
        subcarpeta
    )

    media = MediaIoBaseUpload(
        io.BytesIO(contenido),
        mimetype=mime_type or "application/octet-stream",
        resumable=False,
    )

    metadata = {
        "name": nombre_archivo,
        "parents": [folder_id],
    }

    creado = drive_service.files().create(
        body=metadata,
        media_body=media,
        fields="id,name",
        supportsAllDrives=True,
    ).execute()

    return creado["id"]


def descargar_archivo_google_drive(file_id):
    drive_service, _ = obtener_google_clients()

    request = drive_service.files().get_media(
        fileId=str(file_id),
        supportsAllDrives=True,
    )

    buffer = io.BytesIO()

    downloader = MediaIoBaseDownload(
        buffer,
        request
    )

    done = False

    while not done:
        _, done = downloader.next_chunk()

    buffer.seek(0)

    return buffer.getvalue()


@st.cache_data(ttl=600, show_spinner=False)
def _tc_historial_imagen_bytes(file_id):
    """Descarga y cachea una imagen del historial desde Google Drive."""
    file_id = str(file_id or "").strip()
    if not file_id:
        return b""
    return descargar_archivo_google_drive(file_id)


def _tc_historial_pil(file_id):
    """Devuelve una imagen PIL lista para st.image o None si no existe."""
    try:
        contenido = _tc_historial_imagen_bytes(file_id)
        if not contenido:
            return None
        return Image.open(io.BytesIO(contenido)).convert("RGB")
    except Exception:
        return None


def _asegurar_hoja_historial():
    _, sheets_service = obtener_google_clients()

    spreadsheet_id = _secret_text("GSHEET_ID")

    meta = sheets_service.spreadsheets().get(
        spreadsheetId=spreadsheet_id,
        fields="sheets.properties"
    ).execute()

    nombres = [
        s.get("properties", {}).get("title", "")
        for s in meta.get("sheets", [])
    ]

    if HISTORIAL_SHEET_NAME not in nombres:
        sheets_service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={
                "requests": [
                    {
                        "addSheet": {
                            "properties": {
                                "title": HISTORIAL_SHEET_NAME
                            }
                        }
                    }
                ]
            }
        ).execute()

    rango_header = f"{HISTORIAL_SHEET_NAME}!A1:P1"

    actual = sheets_service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range=rango_header,
    ).execute().get("values", [])

    if not actual or actual[0] != HISTORIAL_HEADERS:
        sheets_service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=rango_header,
            valueInputOption="RAW",
            body={
                "values": [HISTORIAL_HEADERS]
            },
        ).execute()


def guardar_registro_google_sheets(registro):
    _, sheets_service = obtener_google_clients()

    _asegurar_hoja_historial()

    spreadsheet_id = _secret_text("GSHEET_ID")

    row = [[
        registro.get("id", ""),
        registro.get("fecha", ""),
        registro.get("nombre", ""),
        registro.get("imagen_original_file_id", ""),
        registro.get("imagen_procesada_file_id", ""),
        int(registro.get("surcos", 0) or 0),
        float(registro.get("verde_pct", 0.0) or 0.0),
        float(registro.get("rojo_pct", 0.0) or 0.0),
        float(registro.get("amarillo_pct", 0.0) or 0.0),
        registro.get("nivel_visual", ""),
        registro.get("zona_mas_afectada", ""),
        registro.get("diagnostico_visual", ""),
        json.dumps(
            registro.get("causas_probables", []),
            ensure_ascii=False
        ),
        registro.get("explicacion_nutrientes", ""),
        json.dumps(
            registro.get("recomendaciones", []),
            ensure_ascii=False
        ),
        registro.get("nota_diagnostico", ""),
    ]]

    sheets_service.spreadsheets().values().append(
        spreadsheetId=spreadsheet_id,
        range=f"{HISTORIAL_SHEET_NAME}!A:P",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body={"values": row},
    ).execute()


def _parse_lista_historial(valor):
    if isinstance(valor, list):
        return valor

    texto = str(valor or "").strip()

    if not texto:
        return []

    try:
        parsed = json.loads(texto)
        return parsed if isinstance(parsed, list) else [texto]
    except Exception:
        return [texto]



def _float_historial(valor):
    """
    Convierte números del historial aunque Google Sheets los devuelva
    con coma decimal, por ejemplo: '67,7' -> 67.7
    """
    if valor is None or valor == "":
        return 0.0

    if isinstance(valor, (int, float)):
        return float(valor)

    texto = str(valor).strip()
    texto = texto.replace("%", "").replace(" ", "")

    # Si viene en formato español: 67,7
    if "," in texto and "." not in texto:
        texto = texto.replace(",", ".")
    # Si viniera 1.234,56
    elif "," in texto and "." in texto:
        if texto.rfind(",") > texto.rfind("."):
            texto = texto.replace(".", "").replace(",", ".")

    return float(texto)


def _tg_es_numero_historial(valor):
    """True si el valor puede convertirse a número sin romper el historial."""
    if valor is None:
        return True

    texto = str(valor).strip()
    if texto == "":
        return True

    texto = texto.replace("%", "").replace(" ", "")

    if "," in texto and "." not in texto:
        texto = texto.replace(",", ".")
    elif "," in texto and "." in texto and texto.rfind(",") > texto.rfind("."):
        texto = texto.replace(".", "").replace(",", ".")

    try:
        float(texto)
        return True
    except Exception:
        return False


def _tg_int_historial_seguro(valor, default=0):
    """Convierte Surcos sin lanzar excepción si una fila antigua está corrida."""
    try:
        if valor is None or str(valor).strip() == "":
            return int(default)
        return int(round(_float_historial(valor)))
    except Exception:
        return int(default)


def _tg_float_historial_seguro(valor, default=0.0):
    """Convierte porcentajes sin detener toda la carga del historial."""
    try:
        if valor is None or str(valor).strip() == "":
            return float(default)
        return float(_float_historial(valor))
    except Exception:
        return float(default)


def _tg_detectar_offset_historial(row):
    """
    Detecta filas antiguas que quedaron desplazadas una o más columnas.

    Ejemplo del problema:
        ... ImagenOriginalFileID | ImagenProcesadaFileID | Surcos ...

    Si una columna como 'Parcelas' fue insertada antes del ID, los dos FileID
    se desplazan y el antiguo lector intentaba convertir un FileID a float.

    Devuelve el desplazamiento más probable del bloque estándar de 16 campos.
    """
    valores = list(row or [])

    mejor_offset = 0
    mejor_score = -1

    # Soportar hasta 6 columnas añadidas antes del bloque histórico original.
    for offset in range(0, min(7, max(1, len(valores)))):
        if len(valores) < offset + 6:
            continue

        score = 0

        # Campos típicos del bloque estándar.
        id_val = str(valores[offset + 0] or "").strip()
        fecha_val = str(valores[offset + 1] or "").strip()
        nombre_val = str(valores[offset + 2] or "").strip()
        orig_id = str(valores[offset + 3] or "").strip()
        proc_id = str(valores[offset + 4] or "").strip()
        surcos_val = valores[offset + 5] if len(valores) > offset + 5 else ""

        if id_val:
            score += 1

        # UUID/ID interno suele ser largo.
        if len(id_val) >= 8:
            score += 1

        # Fecha ISO o fecha reconocible.
        if ("-" in fecha_val or "/" in fecha_val or "T" in fecha_val) and len(fecha_val) >= 8:
            score += 2

        # Nombre de imagen/archivo.
        if any(ext in nombre_val.lower() for ext in (".jpg", ".jpeg", ".png", ".webp", ".heic")):
            score += 2
        elif nombre_val:
            score += 1

        # IDs de Google Drive normalmente son cadenas largas alfanuméricas.
        if len(orig_id) >= 15 and re.fullmatch(r"[A-Za-z0-9_-]+", orig_id or ""):
            score += 2
        if len(proc_id) >= 15 and re.fullmatch(r"[A-Za-z0-9_-]+", proc_id or ""):
            score += 2

        # Surcos DEBE ser numérico o vacío.
        if _tg_es_numero_historial(surcos_val):
            score += 4
        else:
            score -= 6

        # Los tres porcentajes siguientes también deben ser numéricos o vacíos.
        for idx in (6, 7, 8):
            if len(valores) > offset + idx:
                if _tg_es_numero_historial(valores[offset + idx]):
                    score += 2
                else:
                    score -= 2

        if score > mejor_score:
            mejor_score = score
            mejor_offset = offset

    return max(0, int(mejor_offset))


def obtener_historial_google(limite=100):
    _, sheets_service = obtener_google_clients()

    _asegurar_hoja_historial()

    spreadsheet_id = _secret_text("GSHEET_ID")

    # Leemos más columnas porque algunas filas históricas pueden tener
    # columnas extra como Parcelas/URLs/AppSheet.
    rows = sheets_service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range=f"{HISTORIAL_SHEET_NAME}!A2:AZ",
    ).execute().get("values", [])

    registros = []

    for row_original in rows:
        row_original = list(row_original or [])

        if not any(str(v).strip() for v in row_original):
            continue

        offset = _tg_detectar_offset_historial(row_original)

        row = row_original[offset:offset + 16]
        row = row + [""] * (16 - len(row))

        # Protección adicional:
        # si aun así Surcos no es numérico, intentar localizar un bloque
        # válido desplazado 1..6 columnas antes de descartar la fila.
        if not _tg_es_numero_historial(row[5]):
            encontrado = False

            for otro_offset in range(0, min(7, len(row_original))):
                candidato = row_original[otro_offset:otro_offset + 16]
                candidato = candidato + [""] * (16 - len(candidato))

                if (
                    _tg_es_numero_historial(candidato[5])
                    and _tg_es_numero_historial(candidato[6])
                    and _tg_es_numero_historial(candidato[7])
                    and _tg_es_numero_historial(candidato[8])
                ):
                    row = candidato
                    encontrado = True
                    break

            # No detenemos toda la aplicación por una fila dañada.
            if not encontrado and not _tg_es_numero_historial(row[5]):
                continue

        registros.append({
            "id": row[0],
            "fecha": row[1],
            "nombre": row[2],
            "imagen_original_file_id": row[3],
            "imagen_procesada_file_id": row[4],
            "surcos": _tg_int_historial_seguro(row[5]),
            "verde_pct": _tg_float_historial_seguro(row[6]),
            "rojo_pct": _tg_float_historial_seguro(row[7]),
            "amarillo_pct": _tg_float_historial_seguro(row[8]),
            "nivel_visual": row[9],
            "zona_mas_afectada": row[10],
            "diagnostico_visual": row[11],
            "causas_probables": _parse_lista_historial(row[12]),
            "explicacion_nutrientes": row[13],
            "recomendaciones": _parse_lista_historial(row[14]),
            "nota_diagnostico": row[15],
        })

    registros.reverse()

    return registros[:max(1, int(limite))]


def guardar_analisis_en_google(
    uploaded_image,
    backend_result,
):
    """
    Guarda automáticamente:
    - original en Drive/Originales
    - procesada en Drive/Procesadas
    - datos en Google Sheets
    """
    if not historial_google_configurado():
        return False, "Historial de Google no configurado."

    original_bytes = uploaded_image.getvalue()

    original_file_id = subir_bytes_google_drive(
        original_bytes,
        uploaded_image.name,
        uploaded_image.type or "image/jpeg",
        "Originales",
    )

    annotated = backend_result.get("annotated")

    if annotated is None:
        raise RuntimeError(
            "No existe imagen procesada para guardar."
        )

    ok_png, encoded_png = cv2.imencode(
        ".png",
        annotated
    )

    if not ok_png:
        raise RuntimeError(
            "No se pudo convertir la imagen procesada a PNG."
        )

    nombre_procesada = (
        f"{Path(uploaded_image.name).stem}_procesada.png"
    )

    processed_file_id = subir_bytes_google_drive(
        encoded_png.tobytes(),
        nombre_procesada,
        "image/png",
        "Procesadas",
    )

    backend_data = backend_result.get("backend") or {}
    analisis = (
        backend_data.get("analisis", {})
        if isinstance(backend_data, dict)
        else {}
    ) or {}

    amarillo_pct = float(
        analisis.get("amarillo_pct")
        or (
            backend_data.get("amarillo_pct")
            if isinstance(backend_data, dict)
            else 0.0
        )
        or 0.0
    )

    import uuid
    from datetime import datetime, timezone

    registro = {
        "id": str(uuid.uuid4()),
        "fecha": datetime.now(timezone.utc).isoformat(),
        "nombre": uploaded_image.name,
        "imagen_original_file_id": original_file_id,
        "imagen_procesada_file_id": processed_file_id,
        "surcos": int(
            backend_result.get("count", 0)
        ),
        "verde_pct": float(
            backend_result.get("green_pct", 0.0)
        ),
        "rojo_pct": float(
            backend_result.get("red_pct", 0.0)
        ),
        "amarillo_pct": amarillo_pct,
        "nivel_visual": str(
            backend_result.get(
                "nivel_afectacion_visual",
                "No determinado"
            )
        ),
        "zona_mas_afectada": str(
            backend_result.get(
                "zona_mas_afectada",
                "No determinada"
            )
        ),
        "diagnostico_visual": str(
            backend_result.get(
                "diagnostico_visual",
                ""
            )
        ),
        "causas_probables": backend_result.get(
            "causas_probables",
            []
        ),
        "explicacion_nutrientes": str(
            backend_result.get(
                "explicacion_nutrientes",
                ""
            )
        ),
        "recomendaciones": backend_result.get(
            "recomendaciones_iniciales",
            []
        ),
        "nota_diagnostico": str(
            backend_result.get(
                "nota_diagnostico",
                ""
            )
        ),
    }

    guardar_registro_google_sheets(
        registro
    )

    return True, registro


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

    .terro-logo-box {
        display:flex;
        align-items:center;
        justify-content:center;
        margin-right:14px;
        min-width:120px;
    }

    .terro-logo-box img {
        display:block;
        width:210px;
        max-width:100%;
        height:auto;
        object-fit:contain;
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


    /* BOTONES DE IDIOMA EN PANEL DERECHO */
    button[kind="secondary"] {
        min-height: 44px !important;
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

        .terro-logo-box {
            min-width:84px;
            margin-right:8px;
        }

        .terro-logo-box img {
            width:145px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# CABECERA - SIN HTML PARA EVITAR CUADRO NEGRO
# ============================================================

titulo_idioma = (
    "Análisis inteligente del viñedo"
    if st.session_state.idioma_terrocore == "ES"
    else "Analyse intelligente du vignoble"
)

header_logo_col, header_text_col = st.columns(
    [1.35, 4.65],
    vertical_alignment="center"
)

with header_logo_col:
    if LOGO_PATH.exists():
        st.image(
            str(LOGO_PATH),
            use_container_width=True
        )

with header_text_col:
    st.markdown(
        "## TerroCore image AI"
    )

    st.markdown(
        f"**{titulo_idioma}**"
    )

st.markdown(
    tr(
        "Analiza video e imágenes del viñedo.",
        "Analyse les vidéos et les images du vignoble."
    )
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
    """
    Estima la distancia REAL entre hileras.

    Evita el error frecuente de tomar como separación la mitad
    del espaciamiento verdadero (bordes izquierdo/derecho de una
    misma hilera), que era una de las causas del sobreconteo.
    """
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

    p = p - trend
    p -= np.mean(p)

    if np.std(p) < 1e-7:
        return None

    ac = np.correlate(
        p,
        p,
        mode="full"
    )

    ac = ac[len(p) - 1:]

    minimum = 6
    maximum = min(
        48,
        len(ac) - 1
    )

    if maximum <= minimum:
        return None

    segment = ac[
        minimum:
        maximum + 1
    ]

    peaks, _ = find_peaks(
        segment,
        distance=2
    )

    if len(peaks) == 0:
        lag = minimum + int(
            np.argmax(segment)
        )
    else:
        lags = minimum + peaks
        values = segment[peaks]

        # Pico de autocorrelación más fuerte.
        best_idx = int(
            np.argmax(values)
        )
        lag = int(lags[best_idx])
        best_value = float(values[best_idx])

        # Si el pico dominante es demasiado corto, comprobar 2x.
        # Los bordes de una misma hilera pueden crear un falso periodo
        # de aproximadamente la mitad del espaciamiento real.
        if lag <= 12:
            target = lag * 2
            near = np.where(
                np.abs(lags - target) <= 2
            )[0]

            if len(near):
                near_best = int(
                    near[
                        np.argmax(
                            values[near]
                        )
                    ]
                )

                if float(values[near_best]) >= best_value * 0.68:
                    lag = int(
                        lags[near_best]
                    )

    return float(
        np.clip(
            lag,
            8.0,
            45.0
        )
    )


# ============================================================
# REJILLA GLOBAL - UN ID POR HILERA REAL
# ============================================================

def calcular_rejilla_global_surcos(
    green_mask,
    components
):
    """
    Construye una sola rejilla transversal para toda la imagen.

    Esto evita contar dos o tres veces la misma hilera cuando el
    detector divide el viñedo en varios componentes por vigor,
    huecos secos o cambios de iluminación.
    """
    if not components:
        return None

    h, w = green_mask.shape

    angles = []
    weights = []

    for component in components:
        angles.append(
            float(component["angle"])
        )
        weights.append(
            max(
                1.0,
                float(component.get("tiles", 1))
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
        weights
        *
        np.exp(
            1j * 2.0 * angles
        )
    )

    if abs(z) < 1e-8:
        mean_angle = float(
            angles[0]
        )
    else:
        mean_angle = float(
            (np.angle(z) / 2.0) % np.pi
        )

    angle_deg = float(
        np.rad2deg(mean_angle)
    )

    # Después de rotar, las hileras quedan aproximadamente verticales.
    rotation = 90.0 - angle_deg

    M = cv2.getRotationMatrix2D(
        (w / 2.0, h / 2.0),
        rotation,
        1.0
    )

    rotated = cv2.warpAffine(
        (green_mask > 0).astype(np.uint8) * 255,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT
    )

    # Ignorar bordes extremos, donde suelen aparecer edificios/árboles.
    ya = int(h * 0.06)
    yb = int(h * 0.94)

    zone = (
        rotated[ya:yb] > 0
    ).astype(np.float32)

    if zone.size == 0:
        return None

    profile = np.mean(
        zone,
        axis=0
    )

    profile = gaussian_filter1d(
        profile,
        sigma=1.0
    )

    trend = gaussian_filter1d(
        profile,
        sigma=max(
            7.0,
            w / 55.0
        )
    )

    signal = profile - trend

    spacing = estimar_espaciado_local(
        signal
    )

    if spacing is None:
        return None

    # Para el conteo final no permitimos picos a media hilera.
    min_distance = max(
        6,
        int(
            round(
                spacing * 0.90
            )
        )
    )

    prominence = max(
        0.004,
        float(
            np.max(signal) - np.median(signal)
        ) * 0.13
    )

    peaks, props = find_peaks(
        signal,
        distance=min_distance,
        prominence=prominence
    )

    if len(peaks) < 4:
        return None

    # Quitar picos muy débiles respecto al perfil de vegetación.
    profile_floor = float(
        np.percentile(
            profile,
            32
        )
    )

    profile_span = max(
        1e-6,
        float(
            np.percentile(profile, 88)
            -
            profile_floor
        )
    )

    keep = []

    for p in peaks:
        strength = (
            float(profile[p])
            -
            profile_floor
        ) / profile_span

        if strength >= 0.10:
            keep.append(
                int(p)
            )

    peaks = np.asarray(
        keep,
        dtype=np.int32
    )

    if len(peaks) < 4:
        return None

    return {
        "angle": mean_angle,
        "angle_deg": angle_deg,
        "M": M,
        "peaks": peaks,
        "spacing": float(spacing)
    }


def asignar_track_a_rejilla(
    points,
    grid
):
    """
    Devuelve el índice de la hilera global más cercana al track.
    Si el track está entre hileras, devuelve None.
    """
    if grid is None or len(points) == 0:
        return None

    middle = np.asarray(
        points[len(points) // 2],
        dtype=np.float32
    )

    aug = np.array(
        [middle[0], middle[1], 1.0],
        dtype=np.float32
    )

    rotated_point = aug @ grid["M"].T
    lateral_x = float(
        rotated_point[0]
    )

    peaks = grid["peaks"].astype(
        np.float32
    )

    nearest = int(
        np.argmin(
            np.abs(peaks - lateral_x)
        )
    )

    distance = abs(
        float(peaks[nearest])
        -
        lateral_x
    )

    if distance > max(
        3.0,
        grid["spacing"] * 0.46
    ):
        return None

    return nearest


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
            0.16
        )
    )
    max_lateral_drift = max(
        3.0,
        float(spacing) * 0.42
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
            local_coherence > 0.30
            and
            angle_change <
            np.deg2rad(
                28.0
            )
        ):
            direction = (
                0.84 *
                direction
                +
                0.16 *
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
                score -= 0.50

            # Si el punto se aleja demasiado del eje original,
            # descartar casi por completo.
            if lateral_from_seed > max_lateral_drift + spacing * 0.22:
                score -= 0.90

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
                    if sv > visual + 0.03:
                        side_penalty += 0.24
                else:
                    s_ok = False
            score -= side_penalty

            if score > best_score:
                best_score = score
                best_point = candidate
                best_response = visual

        if best_point is None:
            break

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

        if weak_steps > 8:
            break

        movement = best_point - point
        movement_norm = float(np.linalg.norm(movement))
        if movement_norm > 1e-6:
            movement /= movement_norm
            if np.dot(movement, direction) > 0.82:
                direction = (
                    0.88 * direction +
                    0.12 * movement
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
    pil_img,
    exclusion_mask=None
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

    # --------------------------------------------------------
    # Paso 3: IA puede bloquear techo/camino/construcción.
    # La detección se ejecuta sobre una copia neutralizada,
    # pero el resultado final se dibuja sobre la foto original.
    # --------------------------------------------------------
    analysis_image = original.copy()

    if exclusion_mask is not None:
        if exclusion_mask.shape[:2] != (h, w):
            exclusion_mask = cv2.resize(
                exclusion_mask,
                (w, h),
                interpolation=cv2.INTER_NEAREST
            )

        exclusion_mask = (
            exclusion_mask > 0
        ).astype(np.uint8) * 255

        # Gris neutro: no genera vegetación falsa y evita bordes fuertes.
        analysis_image[
            exclusion_mask > 0
        ] = (128, 128, 128)

    (
        green,
        theta,
        coherence,
        response,
        components,
        ny,
        nx
    ) = detectar_componentes_vinedo(
        analysis_image,
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

    # Rejilla global: una posición transversal = una hilera real.
    global_grid = calcular_rejilla_global_surcos(
        green,
        components
    )

    used_grid_ids = set()

    final = original.copy()

    occupancy = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    all_tracks = []
    # Datos técnicos de cada trayectoria para Inventario.
    # No cambia el resultado visual de Salud; solo expone la geometría
    # necesaria para calcular slots ocupados/vacíos por separado.
    all_track_records = []

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

            # Descartar semillas que caen sobre un surco ya trazado.
            occupied_hits = 0
            for px_test, py_test in np.rint(points[::max(1, len(points)//12)]).astype(np.int32):
                if 0 <= px_test < w and 0 <= py_test < h and occupancy[py_test, px_test] > 0:
                    occupied_hits += 1
            if occupied_hits >= 2:
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

            # Exigir una trayectoria suficientemente larga para evitar
            # ramas, copas de árboles y fragmentos pequeños.
            if line_length < max(
                45.0,
                spacing * 4.0,
                min(h, w) * 0.075
            ):
                continue

            # Si OpenAI marcó zonas a excluir, al menos 70% del track
            # debe quedar dentro del viñedo permitido.
            if exclusion_mask is not None:
                sample = np.rint(
                    points
                ).astype(np.int32)

                valid_samples = 0
                total_samples = 0

                for sx, sy in sample[::max(1, len(sample)//30)]:
                    if 0 <= sx < w and 0 <= sy < h:
                        total_samples += 1
                        if exclusion_mask[sy, sx] == 0:
                            valid_samples += 1

                if total_samples > 0:
                    valid_fraction = valid_samples / total_samples
                    if valid_fraction < 0.70:
                        continue

            # Asociar el fragmento a UNA hilera global.
            grid_id = asignar_track_a_rejilla(
                points,
                global_grid
            )

            if global_grid is not None and grid_id is None:
                continue

            states = estados_color(
                green_scores
            )

            if grid_id is not None:
                track_index = int(grid_id) + 1
            else:
                track_index = len(all_tracks) + 1

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

                # No dibujar dentro de zonas excluidas por IA.
                if exclusion_mask is not None:
                    mid_x = int(round((x1 + x2) / 2.0))
                    mid_y = int(round((y1 + y2) / 2.0))

                    if (
                        exclusion_mask[y1, x1] > 0
                        or
                        exclusion_mask[y2, x2] > 0
                        or
                        exclusion_mask[mid_y, mid_x] > 0
                    ):
                        continue

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
                    2,
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

            # Si el centro cae en techo/camino, buscar otro punto del surco.
            if (
                exclusion_mask is not None
                and
                exclusion_mask[my, mx] > 0
            ):
                safe_label = None

                for candidate in points:
                    cx = int(np.clip(round(candidate[0]), 0, w - 1))
                    cy = int(np.clip(round(candidate[1]), 0, h - 1))

                    if exclusion_mask[cy, cx] == 0:
                        safe_label = (cx, cy)
                        break

                if safe_label is not None:
                    mx, my = safe_label

            if (
                exclusion_mask is None
                or
                exclusion_mask[my, mx] == 0
            ):
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
                    0.55,
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
                    0.55,
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
                        0.58
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

            all_track_records.append({
                "track_index": int(track_index),
                "grid_id": int(grid_id) if grid_id is not None else None,
                "points": np.asarray(points, dtype=np.float32),
                "green_scores": np.asarray(green_scores, dtype=np.float32),
                "states": np.asarray(states, dtype=bool),
                "spacing": float(spacing),
                "line_length": float(line_length),
            })

            if grid_id is not None:
                used_grid_ids.add(
                    int(grid_id)
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
            len(used_grid_ids)
            if global_grid is not None and len(used_grid_ids) > 0
            else len(all_tracks)
        ),
        "green_pct": float(
            green_pct
        ),
        "red_pct": float(
            red_pct
        ),
        "angle": mean_angle,
        # Campos técnicos usados exclusivamente por Inventario.
        "tracks": all_track_records,
        "green_mask": (green > 0).astype(np.uint8) * 255,
        "response_map": response.astype(np.float32),
    }




# ============================================================
# MOTOR DE VIDEO V3.3 - TODOS LOS SURCOS SIN SALTOS
# ============================================================
# IMPORTANTE:
# - Este bloque se usa SOLAMENTE cuando se analiza VIDEO.
# - El analizador normal "analizar()" de IMÁGENES queda intacto.
# - Cada hilera trabaja dentro de su propio carril.
# - Una hilera seca se conserva y puede marcarse en rojo.
# ============================================================

def video_v33_mascara_verde(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hh, ss, vv = cv2.split(hsv)
    exg = 2.0 * g - r - b
    ngrdi = (g - r) / (g + r + 1e-06)
    mask = ((exg > 7.0) & (ngrdi > -0.015) & (hh >= 21) & (hh <= 108) & (ss >= 16) & (vv >= 22) & (g >= r * 0.875) & (g >= b * 0.875)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    return mask

def video_v33_angulo_surcos(mask):
    h, w = mask.shape
    x0, x1 = (int(w * 0.12), int(w * 0.88))
    y0, y1 = (int(h * 0.12), int(h * 0.9))
    roi = mask[y0:y1, x0:x1]
    edges = cv2.Canny(roi, 30, 100)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=45, minLineLength=max(45, int(h * 0.08)), maxLineGap=18)
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
    hist, edges_b = np.histogram(angles, bins=bins, weights=weights)
    i = int(np.argmax(hist))
    lo = edges_b[i]
    hi = edges_b[i + 1]
    sel = (angles >= lo) & (angles < hi)
    if np.any(sel):
        return float(np.average(angles[sel], weights=weights[sel]))
    return float(np.median(angles))

def video_v33_rotar(img, angle):
    h, w = img.shape[:2]
    centro = (w / 2.0, h / 2.0)
    rot_deg = 90.0 - angle
    M = cv2.getRotationMatrix2D(centro, rot_deg, 1.0)
    Minv = cv2.invertAffineTransform(M)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return (out, M, Minv)

def video_v33_aplicar_matriz(points, M):
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) == 0:
        return pts
    ones = np.ones((len(pts), 1), dtype=np.float32)
    aug = np.hstack([pts, ones])
    return aug @ M.T

def video_v33_limites_verticales(mask):
    """
    Busca el camino superior y el camino inferior.
    """
    h, w = mask.shape
    density = (mask > 0).mean(axis=1).astype(np.float32)
    density = gaussian_filter1d(density, sigma=max(7, h / 100))
    top_range = np.arange(int(h * 0.05), int(h * 0.48))
    y_top_road = int(top_range[np.argmin(density[top_range])])
    bottom_range = np.arange(int(h * 0.55), int(h * 0.95))
    y_bottom_road = int(bottom_range[np.argmin(density[bottom_range])])
    inset = max(8, int(h * 0.01))
    y0 = y_top_road + inset
    y1 = y_bottom_road - inset
    if y1 - y0 < h * 0.36:
        y0 = int(h * 0.14)
        y1 = int(h * 0.88)
    return (max(0, y0), min(h - 1, y1))

def video_v33_estimar_periodo(profile):
    """
    Estima la separación entre hileras.
    """
    p = profile.astype(np.float64)
    p -= np.mean(p)
    if np.std(p) < 1e-07:
        return 20.0
    ac = np.correlate(p, p, mode='full')
    ac = ac[len(p) - 1:]
    width = len(profile)
    a = max(8, int(width * 0.006))
    b = min(int(width * 0.03), len(ac) - 1)
    if b <= a:
        return 20.0
    peaks, _ = find_peaks(ac[a:b + 1])
    if len(peaks) == 0:
        return max(16.0, width / 70.0)
    vals = ac[a:b + 1][peaks]
    lag = float(a + peaks[np.argmax(vals)])
    if lag < width / 85.0:
        lag *= 2.0
    return float(np.clip(lag, 13.0, 35.0))

def video_v33_detectar_limites_laterales(mask, y0, y1):
    """
    Detecta automáticamente los caminos laterales que delimitan
    la parcela principal. Devuelve x0, x1.
    """
    h, w = mask.shape
    zone = (mask[y0:y1] > 0).astype(np.float32)
    density = zone.mean(axis=0)
    density = gaussian_filter1d(density, sigma=max(3.0, w / 450.0))
    p20 = float(np.percentile(density, 20))
    p65 = float(np.percentile(density, 65))
    threshold = p20 + 0.22 * max(p65 - p20, 1e-06)
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
        if width_band >= max(5, int(w * 0.004)):
            bands.append((i, j - 1, width_band))
        i = j
    center = w / 2.0
    left_candidates = [b for b in bands if b[1] < center and b[1] > w * 0.03]
    right_candidates = [b for b in bands if b[0] > center and b[0] < w * 0.97]
    if left_candidates:
        left_band = max(left_candidates, key=lambda b: b[2] * 3.0 - abs(center - b[1]) * 0.012)
        x0 = int(left_band[1] + max(3, w * 0.003))
    else:
        x0 = int(w * 0.08)
    if right_candidates:
        right_band = max(right_candidates, key=lambda b: b[2] * 3.0 - abs(b[0] - center) * 0.012)
        x1 = int(right_band[0] - max(3, w * 0.003))
    else:
        x1 = int(w * 0.92)
    if x1 - x0 < w * 0.35:
        x0 = int(w * 0.08)
        x1 = int(w * 0.92)
    return (max(0, x0), min(w - 1, x1))

def video_v33_semillas_surcos(mask, y0, y1):
    """
    V3.3 - detectar TODOS los surcos.

    En vez de promediar toda la parcela (lo cual borra hileras
    curvas o débiles), toma varios cortes horizontales y calcula
    el espaciado típico entre surcos.

    Después construye una retícula completa. Una hilera seca no
    desaparece solo porque tenga poco verde.
    """
    x0, x1 = video_v33_detectar_limites_laterales(mask, y0, y1)
    zone = (mask[y0:y1, x0:x1] > 0).astype(np.float32)
    height, width = zone.shape
    if width < 30 or height < 30:
        return (np.array([], dtype=np.int32), 20.0, int(x0), int(x1))
    profiles = []
    periods = []
    contrasts = []
    centers = np.linspace(0.12, 0.88, 9)
    band_h = max(12, int(height * 0.11))
    for frac in centers:
        yc = int(round(frac * (height - 1)))
        a = max(0, yc - band_h // 2)
        b = min(height, yc + band_h // 2 + 1)
        band = zone[a:b]
        if band.size == 0:
            continue
        profile = band.mean(axis=0)
        profile = gaussian_filter1d(profile, sigma=1.15)
        period = video_v33_estimar_periodo(profile)
        period = float(np.clip(period, 9.0, 40.0))
        contrast = float(np.std(profile))
        profiles.append(profile)
        periods.append(period)
        contrasts.append(contrast)
    if not profiles:
        return (np.array([], dtype=np.int32), 20.0, int(x0), int(x1))
    spacing_candidates = []
    for profile, p0 in zip(profiles, periods):
        peaks, _ = find_peaks(profile, distance=max(5, int(p0 * 0.45)), prominence=max(0.003, float(profile.max()) * 0.01), height=max(0.006, float(profile.max()) * 0.018))
        if len(peaks) >= 5:
            diffs = np.diff(peaks.astype(np.float32))
            good = diffs[(diffs >= 8.0) & (diffs <= 42.0)]
            if len(good):
                med = float(np.median(good))
                near = good[(good > med * 0.62) & (good < med * 1.38)]
                if len(near):
                    spacing_candidates.extend(near.tolist())
    if spacing_candidates:
        spacing = float(np.median(spacing_candidates))
    else:
        spacing = float(np.median(periods))
    spacing = float(np.clip(spacing, 9.0, 36.0))
    half = spacing / 2.0
    if half >= 8.0:
        score_full = 0.0
        score_half = 0.0
        for profile in profiles:
            for candidate, bucket in [(spacing, 'full'), (half, 'half')]:
                best = -1.0
                phase_steps = max(10, int(round(candidate * 1.5)))
                for phase in np.linspace(0, candidate, phase_steps, endpoint=False):
                    xs = np.arange(phase, len(profile), candidate)
                    if len(xs) < 4:
                        continue
                    values = []
                    for x in xs:
                        xi = int(round(x))
                        a = max(0, xi - 2)
                        b = min(len(profile), xi + 3)
                        if b > a:
                            values.append(float(profile[a:b].mean()))
                    if values:
                        best = max(best, float(np.mean(values)))
                if bucket == 'full':
                    score_full += max(best, 0.0)
                else:
                    score_half += max(best, 0.0)
        if score_half >= score_full * 0.94:
            spacing = half
    spacing = float(np.clip(spacing, 8.0, 36.0))
    best_profile_index = int(np.argmax(np.asarray(contrasts, dtype=np.float32)))
    ref_profile = profiles[best_profile_index]
    best_phase = 0.0
    best_score = -1000000000.0
    phase_steps = max(18, int(round(spacing * 3)))
    for phase in np.linspace(0, spacing, phase_steps, endpoint=False):
        xs = np.arange(phase, width, spacing)
        if len(xs) < 4:
            continue
        values = []
        for x in xs:
            xi = int(round(x))
            a = max(0, xi - 2)
            b = min(width, xi + 3)
            if b > a:
                values.append(float(ref_profile[a:b].mean()))
        if not values:
            continue
        score = float(np.mean(values))
        if score > best_score:
            best_score = score
            best_phase = float(phase)
    local_seeds = np.arange(best_phase, width, spacing, dtype=np.float32)
    edge_margin = max(1.0, spacing * 0.08)
    local_seeds = local_seeds[(local_seeds >= edge_margin) & (local_seeds <= width - 1 - edge_margin)]
    full_profile = zone.mean(axis=0)
    stacked = np.vstack(profiles)
    robust_profile = 0.55 * gaussian_filter1d(full_profile, sigma=1.0) + 0.45 * np.percentile(stacked, 70, axis=0)
    radius = max(1, int(spacing * 0.18))
    refined = []
    for s in local_seeds:
        xi = int(round(s))
        a = max(0, xi - radius)
        b = min(width, xi + radius + 1)
        if b <= a:
            refined.append(float(s + x0))
            continue
        local = robust_profile[a:b]
        if local.size and float(local.max()) > max(0.005, float(np.median(robust_profile)) * 0.8):
            best = a + int(np.argmax(local))
        else:
            best = xi
        refined.append(float(best + x0))
    seeds = np.asarray(refined, dtype=np.float32)
    if len(seeds):
        base = float(local_seeds[0] + x0)
        regular = base + np.arange(len(seeds), dtype=np.float32) * spacing
        max_deviation = spacing * 0.2
        seeds = np.clip(seeds, regular - max_deviation, regular + max_deviation)
        for i in range(1, len(seeds)):
            minimum = seeds[i - 1] + spacing * 0.62
            if seeds[i] < minimum:
                seeds[i] = max(minimum, regular[i] - max_deviation)
    return (np.rint(seeds).astype(np.int32), float(spacing), int(x0), int(x1))

def video_v33_crear_respuesta(bgr, green_mask):
    """
    Combina vegetación con textura vertical.
    Esto permite seguir también hileras secas.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sx = np.abs(sx)
    p99 = np.percentile(sx, 99)
    if p99 > 0:
        sx = np.clip(sx / p99, 0, 1)
    else:
        sx[:] = 0
    sx = cv2.GaussianBlur(sx, (0, 0), sigmaX=2.0, sigmaY=1.0)
    gm = (green_mask > 0).astype(np.float32)
    gm = cv2.GaussianBlur(gm, (0, 0), sigmaX=2.0, sigmaY=1.3)
    response = 0.7 * gm + 0.3 * sx
    return response

def video_v33_seguir_surco(response, green_mask, seed, left, right, y0, y1):
    """
    V3.3 - línea adaptativa SIN saltar al vecino.

    Cada hilera tiene un carril independiente.
    La trayectoria puede curvarse, pero:
    - nunca cruza el punto medio hacia el surco vecino;
    - siempre tiene una pequeña atracción hacia su posición nominal;
    - si desaparece la vegetación, conserva la trayectoria.
    """
    height = max(1, y1 - y0)
    step = max(4, int(height / 170))
    ys = np.arange(y0, y1, step, dtype=np.int32)
    lane_width = max(5.0, float(right - left))
    safety = max(1.0, lane_width * 0.12)
    hard_left = float(left + safety)
    hard_right = float(right - safety)
    if hard_right - hard_left < 2.0:
        hard_left = float(left + 0.5)
        hard_right = float(right - 0.5)
    seed = float(np.clip(seed, hard_left, hard_right))
    xs = []
    greens = []
    prev_x = seed
    velocity = 0.0
    search_radius = max(2, int(lane_width * 0.24))
    max_step_shift = max(0.65, lane_width * 0.055)
    max_total_deviation = lane_width * 0.32
    for y in ys:
        ya = max(y0, y - step // 2 - 1)
        yb = min(y1, y + step // 2 + 2)
        predicted = prev_x + velocity
        predicted = float(np.clip(predicted, seed - max_total_deviation, seed + max_total_deviation))
        predicted = float(np.clip(predicted, hard_left, hard_right))
        a = max(int(np.floor(hard_left)), int(round(predicted)) - search_radius)
        b = min(int(np.ceil(hard_right)), int(round(predicted)) + search_radius)
        if b <= a:
            x_new = predicted
            local_peak = 0.0
            local_median = 0.0
        else:
            candidates = np.arange(a, b + 1, dtype=np.int32)
            visual = np.zeros(len(candidates), dtype=np.float32)
            for j, x in enumerate(candidates):
                xa = max(int(np.floor(hard_left)), x - 3)
                xb = min(int(np.ceil(hard_right)) + 1, x + 4)
                patch = response[ya:yb, xa:xb]
                visual[j] = float(patch.mean()) if patch.size else 0.0
            dist_pred = np.abs(candidates - predicted) / max(search_radius, 1)
            dist_seed = np.abs(candidates - seed) / max(max_total_deviation, 1.0)
            score = visual - 0.26 * dist_pred - 0.11 * dist_seed
            best_index = int(np.argmax(score))
            candidate_best = float(candidates[best_index])
            local_peak = float(np.max(visual))
            local_median = float(np.median(visual))
            strong_evidence = local_peak >= 0.04 and local_peak - local_median >= 0.007
            if strong_evidence:
                target = candidate_best
            else:
                target = predicted
                velocity *= 0.45
            dx = float(np.clip(target - prev_x, -max_step_shift, max_step_shift))
            x_new = prev_x + dx
            velocity = 0.86 * velocity + 0.14 * dx
        x_new = float(np.clip(x_new, seed - max_total_deviation, seed + max_total_deviation))
        x_new = float(np.clip(x_new, hard_left, hard_right))
        xi = int(round(x_new))
        xa = max(int(np.floor(hard_left)), xi - 4)
        xb = min(int(np.ceil(hard_right)) + 1, xi + 5)
        patch_green = green_mask[ya:yb, xa:xb]
        green_score = float(np.mean(patch_green > 0)) if patch_green.size else 0.0
        xs.append(x_new)
        greens.append(green_score)
        prev_x = x_new
    xs = np.asarray(xs, dtype=np.float32)
    greens = np.asarray(greens, dtype=np.float32)
    if len(xs) >= 7:
        smooth = gaussian_filter1d(xs, sigma=1.0, mode='nearest')
        xs = np.clip(smooth, seed - max_total_deviation, seed + max_total_deviation)
        xs = np.clip(xs, hard_left, hard_right)
    return (np.column_stack([xs, ys]).astype(np.float32), greens)

def video_v33_estado_verde(green_scores):
    positive = green_scores[green_scores > 0]
    if len(positive) >= 4:
        threshold = float(np.clip(np.percentile(positive, 35) * 0.62, 0.025, 0.085))
    else:
        threshold = 0.045
    state = green_scores >= threshold
    if len(state) >= 5:
        original = state.copy()
        for i in range(2, len(state) - 2):
            state[i] = np.sum(original[i - 2:i + 3]) >= 3
    return state

def video_v33_analizar(pil_img):
    original = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    h, w = original.shape[:2]
    mask0 = video_v33_mascara_verde(original)
    angle = video_v33_angulo_surcos(mask0)
    rot_img, M, Minv = video_v33_rotar(original, angle)
    rot_mask = cv2.warpAffine(mask0, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    y0, y1 = video_v33_limites_verticales(rot_mask)
    seeds, spacing, x0_detectado, x1_detectado = video_v33_semillas_surcos(rot_mask, y0, y1)
    if len(seeds) < 5:
        raise RuntimeError('No se detectó una parcela de surcos suficientemente clara.')
    x0 = int(x0_detectado)
    x1 = int(x1_detectado)
    parcel_mask = np.zeros_like(rot_mask)
    parcel_mask[y0:y1, x0:x1] = rot_mask[y0:y1, x0:x1]
    response = video_v33_crear_respuesta(rot_img, parcel_mask)
    tracks = []
    for i, seed in enumerate(seeds):
        if i == 0:
            left = x0
        else:
            left = int((seeds[i - 1] + seed) / 2)
        if i == len(seeds) - 1:
            right = x1
        else:
            right = int((seed + seeds[i + 1]) / 2)
        if right - left < 5:
            continue
        pts, green = video_v33_seguir_surco(response, parcel_mask, int(seed), left, right, y0, y1)
        tracks.append({'points': pts, 'green': green})
    final = original.copy()
    total_green = 0
    total_red = 0
    mask_preview_rot = np.zeros_like(rot_mask)
    mask_preview_rot[y0:y1, x0:x1] = 255
    mask_preview = cv2.warpAffine(mask_preview_rot, Minv, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    for number, tr in enumerate(tracks, 1):
        pts_rot = tr['points']
        green = tr['green']
        state = video_v33_estado_verde(green)
        pts = video_v33_aplicar_matriz(pts_rot, Minv)
        first_drawn = None
        for j in range(len(pts) - 1):
            p1 = pts[j]
            p2 = pts[j + 1]
            if not (np.all(np.isfinite(p1)) and np.all(np.isfinite(p2))):
                continue
            x1p = int(np.clip(round(p1[0]), 0, w - 1))
            y1p = int(np.clip(round(p1[1]), 0, h - 1))
            x2p = int(np.clip(round(p2[0]), 0, w - 1))
            y2p = int(np.clip(round(p2[1]), 0, h - 1))
            if mask_preview[y1p, x1p] == 0 or mask_preview[y2p, x2p] == 0:
                continue
            is_green = bool(state[j] or state[min(j + 1, len(state) - 1)])
            if is_green:
                color = (0, 240, 0)
                total_green += 1
            else:
                color = (0, 0, 255)
                total_red += 1
            cv2.line(final, (x1p, y1p), (x2p, y2p), color, 1, cv2.LINE_AA)
            if first_drawn is None:
                first_drawn = (x1p, y1p)
        if first_drawn is not None:
            label = str(number)
            label_x = max(0, first_drawn[0] - 4)
            label_y = max(13, first_drawn[1] - 4)
            cv2.putText(final, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(final, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (10, 10, 10), 1, cv2.LINE_AA)
    total = total_green + total_red
    green_pct = 100.0 * total_green / total if total else 0.0
    red_pct = 100.0 - green_pct if total else 0.0
    return {'image': cv2.cvtColor(final, cv2.COLOR_BGR2RGB), 'mask': mask_preview, 'count': len(tracks), 'green_pct': green_pct, 'red_pct': red_pct, 'angle': angle}

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

    # Muestreo automático más fino para mejorar la detección en video.
    if duration > 180:
        sample_seconds = 3.0
    elif duration > 60:
        sample_seconds = 2.0
    else:
        sample_seconds = 1.5

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
    max_frames = 28

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
# ANALIZAR FRAMES DEL VIDEO CON MOTOR V3.3 ORIGINAL
# ============================================================

def analizar_frames_video(info):
    analyzed = []

    for index, item in enumerate(
        info["frames"],
        1
    ):
        frame = item["frame"]

        pil_original = Image.fromarray(
            cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )
        )

        try:
            # ================================================
            # IA TAMBIÉN PARA VIDEO
            # Revisa cada fotograma antes de dibujar surcos.
            # La interfaz NO cambia.
            # ================================================
            ok_scene_ia, scene_ia = analizar_pil_con_ia(
                pil_original
            )

            exclusion_mask = None
            pil_para_detector = pil_original

            if ok_scene_ia:
                if not bool(scene_ia.get("es_vinedo", False)):
                    continue

                if not bool(scene_ia.get("analizar_surcos", False)):
                    continue

                exclusion_mask = crear_mascara_exclusion_ia(
                    scene_ia,
                    pil_original.width,
                    pil_original.height
                )

                # En V3.3 el motor de video no recibe exclusion_mask.
                # Por eso ocultamos temporalmente esas zonas SOLO para
                # el detector. Luego restauramos la imagen original.
                if (
                    exclusion_mask is not None
                    and
                    np.any(exclusion_mask > 0)
                ):
                    frame_detector = frame.copy()

                    # Oscurecer zonas prohibidas para que V3.3 no las
                    # interprete como surcos ni vegetación repetitiva.
                    frame_detector[
                        exclusion_mask > 0
                    ] = (
                        25,
                        25,
                        25
                    )

                    pil_para_detector = Image.fromarray(
                        cv2.cvtColor(
                            frame_detector,
                            cv2.COLOR_BGR2RGB
                        )
                    )

            result = video_v33_analizar(
                pil_para_detector
            )

            annotated = cv2.cvtColor(
                result["image"],
                cv2.COLOR_RGB2BGR
            )

            # Restaurar el fondo original en las zonas excluidas.
            # Así no quedan manchas negras en el resultado final.
            if (
                exclusion_mask is not None
                and
                exclusion_mask.shape[:2] == annotated.shape[:2]
            ):
                annotated[
                    exclusion_mask > 0
                ] = frame[
                    exclusion_mask > 0
                ]

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
                "annotated": annotated,
                "ia_scene": scene_ia if ok_scene_ia else None
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


# ============================================================
# INTERFAZ TERRACORE - FLUJO POR PARCELA / INVENTARIO
# Conserva diseño, logo, idiomas, backend e historial original.
# ============================================================

from datetime import date

# ------------------------------------------------------------
# ESTADO DE SESIÓN NUEVO
# ------------------------------------------------------------
_estado_nuevo = {
    "tc_parcela_nombre": "",
    "tc_fecha_captura": date.today(),
    "tc_captura_confirmada": False,
    "tc_inventario_procesado": False,
    "tc_inventario_confirmado": False,
    "tc_resultados_base": [],
    "tc_tabla_inventario": None,
    "tc_inventario_imagen": None,
    "tc_inventario_fuente": "",
    "tc_inventario_confianza": 0.0,
    "tc_inventario_rows_ai": [],
    "tc_inventario_modelo": "",
    "tc_inventario_debug": {},
    "tc_inventarios_por_imagen": {},
    "tc_salud_procesada": False,
}

for _k, _v in _estado_nuevo.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def _tc_reiniciar_parcela():
    st.session_state.tc_captura_confirmada = False
    st.session_state.tc_inventario_procesado = False
    st.session_state.tc_inventario_confirmado = False
    st.session_state.tc_resultados_base = []
    st.session_state.tc_tabla_inventario = None
    st.session_state.tc_inventario_imagen = None
    st.session_state.tc_inventario_fuente = ""
    st.session_state.tc_inventario_confianza = 0.0
    st.session_state.tc_inventario_rows_ai = []
    st.session_state.tc_inventario_modelo = ""
    st.session_state.tc_inventario_debug = {}
    st.session_state.tc_inventarios_por_imagen = {}
    st.session_state.tc_salud_procesada = False


# ============================================================
# INVENTARIO AUTOMÁTICO - SLOTS / OCUPADOS / VACÍOS
# ============================================================

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


def _tc_analizar_inventario_local(uploaded_image):
    """Detecta surcos + slots sin mezclar el diagnóstico de salud.

    Devuelve una imagen de Inventario, tabla automática y confianza media.
    """
    pil_img = Image.open(io.BytesIO(uploaded_image.getvalue())).convert("RGB")
    base = analizar(pil_img)
    tracks_raw = list(base.get("tracks", []) or [])
    response_map = np.asarray(base.get("response_map"), dtype=np.float32)

    if not tracks_raw or response_map.ndim != 2:
        raise RuntimeError(tr(
            "No se pudieron obtener trayectorias técnicas para Inventario.",
            "Impossible d’obtenir les trajectoires techniques pour l’inventaire."
        ))

    # Deduplicar fragmentos asignados a la misma hilera global.
    by_index = {}
    for track in tracks_raw:
        idx = int(track.get("track_index", len(by_index) + 1))
        current = by_index.get(idx)
        if current is None or float(track.get("line_length", 0.0)) > float(current.get("line_length", 0.0)):
            by_index[idx] = track

    tracks = [by_index[k] for k in sorted(by_index)]

    # Pitch común: mantiene una escala coherente de slots entre todos los surcos.
    pitches = []
    pitch_confidences = []
    for track in tracks:
        points = np.asarray(track.get("points", []), dtype=np.float32)
        spacing = float(track.get("spacing", 18.0) or 18.0)
        ds, prof, _ = _tc_sample_map_along_track(response_map, points, spacing, step_px=2.0)
        if len(ds) >= 8:
            p, c = _tc_estimar_pitch_slots(ds, prof, spacing)
            if np.isfinite(p) and p > 0:
                pitches.append(float(p))
                pitch_confidences.append(float(c))

    forced_pitch = float(np.median(pitches)) if pitches else None

    original = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    annotated = original.copy()
    h, w = annotated.shape[:2]

    rows = []
    confidences = []

    # Colores de INVENTARIO, deliberadamente distintos de Salud.
    color_line = (245, 245, 245)      # blanco
    color_occ = (255, 185, 30)        # azul/cian en BGR
    color_empty = (0, 170, 255)       # naranja en BGR

    for row_number, track in enumerate(tracks, 1):
        pts = np.asarray(track.get("points", []), dtype=np.float32)
        if len(pts) < 2:
            continue

        slots, pitch_used, conf = _tc_slots_track(response_map, track, forced_pitch=forced_pitch)
        confidences.append(conf)

        ipts = np.rint(pts).astype(np.int32)
        ipts[:, 0] = np.clip(ipts[:, 0], 0, w - 1)
        ipts[:, 1] = np.clip(ipts[:, 1], 0, h - 1)
        cv2.polylines(annotated, [ipts], False, color_line, 1, cv2.LINE_AA)

        # Numeración igual al INICIO y FINAL.
        label = f"{row_number:02d}"
        for endpoint in (ipts[0], ipts[-1]):
            ex, ey = int(endpoint[0]), int(endpoint[1])
            tx = int(np.clip(ex + 5, 0, max(0, w - 38)))
            ty = int(np.clip(ey - 5, 16, max(16, h - 4)))
            cv2.putText(annotated, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 40, 40), 3, cv2.LINE_AA)
            cv2.putText(annotated, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        occupied_count = 0
        empty_count = 0
        for slot in slots:
            x = int(np.clip(round(slot["point"][0]), 0, w - 1))
            y = int(np.clip(round(slot["point"][1]), 0, h - 1))
            if slot["occupied"]:
                occupied_count += 1
                cv2.circle(annotated, (x, y), 4, color_occ, 2, cv2.LINE_AA)
                cv2.circle(annotated, (x, y), 1, (255, 255, 255), -1, cv2.LINE_AA)
            else:
                empty_count += 1
                r = 5
                cv2.line(annotated, (x-r, y-r), (x+r, y+r), color_empty, 2, cv2.LINE_AA)
                cv2.line(annotated, (x-r, y+r), (x+r, y-r), color_empty, 2, cv2.LINE_AA)

        rows.append({
            tr("Surco", "Rang"): label,
            tr("Slots", "Emplacements"): int(len(slots)),
            tr("Ocupados", "Occupés"): int(occupied_count),
            tr("Vacíos", "Vides"): int(empty_count),
            tr("Confianza", "Confiance"): round(float(conf) * 100.0, 1),
        })

    if not rows:
        raise RuntimeError(tr(
            "No se pudo construir el inventario automático de slots.",
            "Impossible de construire l’inventaire automatique des emplacements."
        ))

    # Reordenar/renumerar de forma continua 01..N.
    for i, row in enumerate(rows, 1):
        row[tr("Surco", "Rang")] = f"{i:02d}"

    return {
        "image": annotated,
        "table": pd.DataFrame(rows),
        "count": len(rows),
        "pitch_px": float(forced_pitch or 0.0),
        "confidence": float(np.mean(confidences)) if confidences else 0.0,
    }


def _tc_resultado_a_fila_surcos(total_surcos):
    """Tabla de respaldo si el detector automático de slots no logra ejecutarse."""
    total_surcos = max(0, int(total_surcos or 0))
    return pd.DataFrame([
        {
            tr("Surco", "Rang"): f"{i:02d}",
            tr("Slots", "Emplacements"): 0,
            tr("Ocupados", "Occupés"): 0,
            tr("Vacíos", "Vides"): 0,
            tr("Confianza", "Confiance"): 0.0,
        }
        for i in range(1, total_surcos + 1)
    ])


def _tc_metricas_tabla(df):
    col_slots = tr("Slots", "Emplacements")
    col_occ = tr("Ocupados", "Occupés")
    col_empty = tr("Vacíos", "Vides")
    if df is None or df.empty:
        return 0, 0, 0, True
    for c in (col_slots, col_occ, col_empty):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    ok = bool((df[col_slots] == (df[col_occ] + df[col_empty])).all())
    return int(df[col_slots].sum()), int(df[col_occ].sum()), int(df[col_empty].sum()), ok



# ============================================================
# IA HIBRIDA - GEMINI FLASH-LITE + OPENAI DE RESPALDO
# ============================================================
# Gemini 2.5 Flash-Lite es el motor principal para Inventario y Salud.
# OpenAI se usa únicamente si Gemini falla, devuelve JSON incompleto
# o presenta confianza demasiado baja. OpenCV solo recorta y dibuja.
# El backend existente se conserva sin cambios.
# ============================================================

def _tc_openai_model():
    """Modelo económico usado para tareas repetitivas."""
    try:
        value = str(st.secrets.get("OPENAI_VISION_MODEL", "")).strip()
        if value:
            return value
    except Exception:
        pass
    return "gpt-5.6-luna"


def _tc_openai_precision_model():
    """Modelo para auditorías donde la geometría/Salud importa más que el costo."""
    try:
        value = str(st.secrets.get("OPENAI_PRECISION_MODEL", "")).strip()
        if value:
            return value
    except Exception:
        pass
    return "gpt-5.6-terra"


def _tc_openai_client():
    try:
        api_key = str(st.secrets["OPENAI_API_KEY"]).strip()
    except Exception:
        api_key = ""
    if not api_key:
        raise RuntimeError("Falta OPENAI_API_KEY en Streamlit Secrets.")
    return OpenAI(api_key=api_key)


def _tc_jpeg_bytes(pil_img, max_side=2200, quality=93):
    img = pil_img.convert("RGB")
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


def _tc_gemini_model():
    """Modelo económico principal para Inventario y Salud."""
    try:
        value = str(st.secrets.get("GEMINI_VISION_MODEL", "")).strip()
        if value:
            return value
    except Exception:
        pass
    value = str(os.getenv("GEMINI_VISION_MODEL", "")).strip()
    return value or "gemini-2.5-flash-lite"


def _tc_gemini_api_key():
    try:
        value = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
        if value:
            return value
    except Exception:
        pass
    return str(os.getenv("GEMINI_API_KEY", "")).strip()


def _tc_gemini_json(images, prompt, detail="high"):
    """
    Motor principal económico. Usa la API REST oficial de Gemini para no
    agregar dependencias nuevas a la app. Devuelve JSON parseado.
    """
    api_key = _tc_gemini_api_key()
    if not api_key:
        raise RuntimeError("Falta GEMINI_API_KEY en Streamlit Secrets o variables de entorno.")

    parts = [{"text": str(prompt)}]
    for img in images:
        b = _tc_jpeg_bytes(img) if isinstance(img, Image.Image) else bytes(img)
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(b).decode("utf-8"),
            }
        })

    model_name = _tc_gemini_model()
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent"
    )

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": parts,
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.05,
            "maxOutputTokens": 16384,
        },
    }

    response = requests.post(
        url,
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=240,
    )

    if response.status_code >= 400:
        detail_text = response.text[:1200]
        raise RuntimeError(
            f"Gemini HTTP {response.status_code}: {detail_text}"
        )

    body = response.json()
    candidates = body.get("candidates") or []
    if not candidates:
        feedback = body.get("promptFeedback") or {}
        raise RuntimeError(
            "Gemini no devolvió candidatos. "
            + json.dumps(feedback, ensure_ascii=False)[:800]
        )

    content = candidates[0].get("content") or {}
    response_parts = content.get("parts") or []
    texts = [
        str(part.get("text", ""))
        for part in response_parts
        if isinstance(part, dict) and part.get("text")
    ]
    if not texts:
        raise RuntimeError("Gemini no devolvió texto JSON.")

    data = limpiar_json_respuesta("\n".join(texts))
    if not isinstance(data, dict):
        raise RuntimeError("Gemini no devolvió un objeto JSON.")
    return data


def _tc_ai_result_useful(data, prompt):
    """Valida ESTRUCTURA mínima. La baja confianza ya NO obliga a usar OpenAI."""
    if not isinstance(data, dict) or not data:
        return False, "respuesta vacía"

    p = str(prompt or "").lower()
    rows = data.get("rows")
    if rows is None:
        rows = data.get("surcos")

    expects_rows = (
        '"rows"' in p
        or '"rows":' in p
        or '"surcos":[' in p.replace(" ", "")
    )

    if expects_rows:
        if not isinstance(rows, list) or len(rows) == 0:
            return False, "no devolvió filas/surcos"

        # Cuando el prompt pide geometría, exigimos al menos una fila con 2 puntos.
        expects_points = (
            '"points"' in p
            or '"puntos"' in p
            or 'trayectoria' in p
        )
        if expects_points:
            valid_geometry = 0
            for item in rows:
                if not isinstance(item, dict):
                    continue
                pts = item.get("points") or item.get("puntos") or item.get("trayectoria") or []
                if isinstance(pts, list) and len(pts) >= 2:
                    valid_geometry += 1
            if valid_geometry == 0:
                return False, "no devolvió geometría utilizable"

    # En Salud, si falta la zona no cancelamos; se marca como advertencia.
    return True, "ok"


def _tc_ai_quality_warning(data, prompt):
    """Devuelve una advertencia de calidad sin tumbar el análisis ni llamar OpenAI."""
    if not isinstance(data, dict):
        return ""

    p = str(prompt or "").lower()
    rows = data.get("rows")
    if rows is None:
        rows = data.get("surcos")

    warnings = []
    if isinstance(rows, list) and rows:
        confidences = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            raw_conf = item.get("confidence", item.get("confianza"))
            if raw_conf is not None:
                try:
                    confidences.append(float(raw_conf))
                except Exception:
                    pass
        if confidences:
            avg = sum(confidences) / max(1, len(confidences))
            if max(confidences) < 0.42:
                warnings.append("Gemini devolvió confianza baja; el resultado se conserva y queda marcado para revisión.")
            elif avg < 0.52:
                warnings.append("Gemini devolvió confianza media-baja en parte del análisis; revisa visualmente esas hileras.")

    if '"zona_mas_afectada"' in p and not str(data.get("zona_mas_afectada", "")).strip():
        warnings.append("Gemini no definió la zona más afectada; el diagnóstico continúa sin cancelar Salud.")

    return " ".join(warnings).strip()


def _tc_push_ai_warning(message):
    message = str(message or "").strip()
    if not message:
        return
    current = st.session_state.get("tc_ai_warnings_runtime", []) or []
    current = list(current)
    if message not in current:
        current.append(message)
    st.session_state.tc_ai_warnings_runtime = current


def _tc_openai_json_fallback(images, prompt, detail="high", model=None, effort="medium"):
    """OpenAI queda únicamente como respaldo cuando Gemini no alcanza calidad mínima."""
    content = [{"type": "input_text", "text": str(prompt)}]
    for img in images:
        b = _tc_jpeg_bytes(img) if isinstance(img, Image.Image) else bytes(img)
        data_url = "data:image/jpeg;base64," + base64.b64encode(b).decode("utf-8")
        content.append({"type": "input_image", "image_url": data_url, "detail": detail})

    client = _tc_openai_client()
    requested = str(model or _tc_openai_model()).strip()
    models = [requested]
    fallback = _tc_openai_model()
    if fallback not in models:
        models.append(fallback)

    errors = []
    for model_name in models:
        kwargs = dict(
            model=model_name,
            reasoning={"effort": effort},
            input=[{"role": "user", "content": content}],
        )
        try:
            response = client.responses.create(
                **kwargs,
                text={"format": {"type": "json_object"}},
            )
        except Exception:
            try:
                response = client.responses.create(**kwargs)
            except Exception as second_exc:
                errors.append(f"{model_name}: {second_exc}")
                continue
        try:
            data = limpiar_json_respuesta(getattr(response, "output_text", "") or "")
            useful, reason = _tc_ai_result_useful(data, prompt)
            if useful:
                st.session_state.tc_last_ai_provider = f"OpenAI respaldo ({model_name})"
                return data
            errors.append(f"{model_name}: {reason}")
        except Exception as parse_exc:
            errors.append(f"{model_name}: {parse_exc}")

    raise RuntimeError("OpenAI de respaldo no devolvió JSON utilizable. " + " | ".join(errors[-2:]))


def _tc_openai_json(images, prompt, detail="high", model=None, effort="medium"):
    """
    Motor híbrido robusto:
    1) Gemini 2.5 Flash-Lite SIEMPRE primero.
    2) Si Gemini devuelve estructura utilizable, se acepta aunque la confianza sea baja.
       La baja confianza solo genera una advertencia; NO gasta OpenAI.
    3) OpenAI se intenta únicamente si Gemini falla por completo o no devuelve estructura útil.
    4) Si OpenAI no tiene saldo, eso no afecta un resultado válido de Gemini.
    """
    gemini_error = None
    gemini_partial = None

    try:
        data = _tc_gemini_json(images, prompt, detail=detail)
        gemini_partial = data if isinstance(data, dict) else None
        useful, reason = _tc_ai_result_useful(data, prompt)
        if useful:
            warning = _tc_ai_quality_warning(data, prompt)
            st.session_state.tc_last_ai_provider = f"Gemini ({_tc_gemini_model()})"
            st.session_state.tc_last_ai_warning = warning
            if warning:
                _tc_push_ai_warning(warning)
                # Metadatos internos; las funciones existentes pueden ignorarlos.
                data = dict(data)
                data["_tc_review_required"] = True
                data["_tc_review_reason"] = warning
            return data
        gemini_error = f"Gemini: {reason}"
    except Exception as exc:
        gemini_error = f"Gemini: {exc}"

    # Solo llegamos aquí cuando Gemini NO produjo una estructura utilizable.
    try:
        return _tc_openai_json_fallback(
            images,
            prompt,
            detail=detail,
            model=model,
            effort=effort,
        )
    except Exception as openai_exc:
        # Si Gemini dejó algún JSON parcial, preferimos devolverlo marcado para revisión
        # antes que perder todo por falta de saldo en OpenAI, siempre que contenga datos.
        if isinstance(gemini_partial, dict) and gemini_partial:
            warning = (
                f"Gemini devolvió un resultado parcial ({gemini_error}). "
                "OpenAI de respaldo no está disponible; se conserva lo recuperable para revisión."
            )
            _tc_push_ai_warning(warning)
            partial = dict(gemini_partial)
            partial["_tc_review_required"] = True
            partial["_tc_review_reason"] = warning
            st.session_state.tc_last_ai_provider = f"Gemini parcial ({_tc_gemini_model()})"
            return partial

        msg = str(openai_exc)
        if "insufficient_quota" in msg or "credit_balance_exhausted" in msg or "no credits remaining" in msg.lower():
            raise RuntimeError(
                f"{gemini_error}. OpenAI de respaldo no tiene saldo. "
                "Configura GEMINI_API_KEY en Streamlit Secrets para que Gemini pueda trabajar sin depender de OpenAI."
            )
        raise RuntimeError(f"{gemini_error} | OpenAI respaldo: {openai_exc}")


def _tc_clamp01(value, default=0.0):
    try:
        return float(np.clip(float(value), 0.0, 1.0))
    except Exception:
        return float(default)


def _tc_norm_point(point):
    try:
        return [
            float(np.clip(float(point[0]), 0.0, 1000.0)),
            float(np.clip(float(point[1]), 0.0, 1000.0)),
        ]
    except Exception:
        return None


def _tc_norm_to_px(points_norm, w, h):
    arr = np.asarray(points_norm, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    out = arr.copy()
    out[:, 0] = out[:, 0] / 1000.0 * max(1, w - 1)
    out[:, 1] = out[:, 1] / 1000.0 * max(1, h - 1)
    return out


def _tc_point_on_polyline(points, t):
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) == 0:
        return np.array([0.0, 0.0], dtype=np.float32)
    if len(pts) == 1:
        return pts[0].copy()
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 1e-6:
        return pts[0].copy()
    d = float(np.clip(t, 0.0, 1.0)) * total
    j = int(np.searchsorted(cum, d, side="right") - 1)
    j = max(0, min(j, len(pts) - 2))
    d0, d1 = float(cum[j]), float(cum[j + 1])
    a = 0.0 if d1 <= d0 else (d - d0) / (d1 - d0)
    return pts[j] * (1.0 - a) + pts[j + 1] * a


def _tc_sanitize_rows(data):
    if not isinstance(data, dict):
        return []
    raw = data.get("rows") or data.get("surcos") or []
    result = []
    for idx, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            continue
        pts = []
        for p in (item.get("points") or item.get("puntos") or [])[:14]:
            q = _tc_norm_point(p)
            if q is not None:
                pts.append(q)
        if len(pts) < 2:
            continue
        result.append({
            "id": int(item.get("id", idx) or idx),
            "points_norm": pts,
            "confidence": _tc_clamp01(item.get("confidence", item.get("confianza", 0.7)), 0.7),
        })
    return result


def _tc_sort_rows_ai(rows, w, h):
    if not rows:
        return []
    directions = []
    prepared = []
    for row in rows:
        pts = _tc_norm_to_px(row.get("points_norm", []), w, h)
        if len(pts) < 2:
            continue
        v = pts[-1] - pts[0]
        n = float(np.linalg.norm(v))
        if n > 1e-6:
            v = v / n
            if abs(v[1]) >= abs(v[0]):
                if v[1] < 0:
                    v = -v
            elif v[0] < 0:
                v = -v
            directions.append(v)
        prepared.append((row, pts))
    if not prepared:
        return []
    d = np.mean(np.asarray(directions), axis=0) if directions else np.array([0.0, 1.0])
    dn = float(np.linalg.norm(d))
    d = d / dn if dn > 1e-6 else np.array([0.0, 1.0])
    normal = np.array([d[1], -d[0]], dtype=np.float32)
    if abs(d[1]) >= abs(d[0]) and normal[0] < 0:
        normal = -normal
    if abs(d[0]) > abs(d[1]) and normal[1] < 0:
        normal = -normal
    ordered = []
    for row, pts in prepared:
        mid = _tc_point_on_polyline(pts, 0.5)
        ordered.append((float(np.dot(mid, normal)), row))
    ordered.sort(key=lambda item: item[0])
    out = []
    for i, (_, row) in enumerate(ordered, 1):
        rr = dict(row)
        rr["id"] = i
        out.append(rr)
    return out


def _tc_remove_duplicate_rows_ai(rows, w, h):
    rows = _tc_sort_rows_ai(rows, w, h)
    if len(rows) < 3:
        return rows
    mids = np.asarray([
        _tc_point_on_polyline(_tc_norm_to_px(r["points_norm"], w, h), 0.5)
        for r in rows
    ])
    gaps = np.linalg.norm(np.diff(mids, axis=0), axis=1)
    good = gaps[gaps > 2.0]
    if len(good) == 0:
        return rows
    med = float(np.median(good))
    kept = [rows[0]]
    last_mid = mids[0]
    for i in range(1, len(rows)):
        distance = float(np.linalg.norm(mids[i] - last_mid))
        if distance < med * 0.30:
            if rows[i].get("confidence", 0) > kept[-1].get("confidence", 0):
                kept[-1] = rows[i]
                last_mid = mids[i]
            continue
        kept.append(rows[i])
        last_mid = mids[i]
    return _tc_sort_rows_ai(kept, w, h)


def _tc_render_row_guide_ai(pil, rows):
    """Guía visual para que OpenAI audite omisiones/duplicados de surcos."""
    bgr = cv2.cvtColor(np.asarray(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    for row in rows:
        rid = int(row.get("id", 0) or 0)
        pts = _tc_norm_to_px(row.get("points_norm", []), w, h)
        if len(pts) < 2:
            continue
        ip = np.rint(pts).astype(np.int32)
        cv2.polylines(bgr, [ip], False, (255, 255, 0), max(1, int(round(min(w,h)/900))), cv2.LINE_AA)
        mid = _tc_point_on_polyline(pts, 0.5)
        cv2.putText(bgr, f"R{rid:02d}", (int(mid[0])+2, int(mid[1])-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20,20,20), 3, cv2.LINE_AA)
        cv2.putText(bgr, f"R{rid:02d}", (int(mid[0])+2, int(mid[1])-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1, cv2.LINE_AA)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _tc_detect_rows_openai(uploaded_image):
    pil = Image.open(io.BytesIO(uploaded_image.getvalue())).convert("RGB")
    w, h = pil.size

    prompt_1 = """
Eres el módulo de visión agrícola de TerraCore. Analiza SOLO esta fotografía aérea.
Localiza TODAS las hileras físicas reales del viñedo. No cuentes plantas y no hagas Salud.

REGLAS:
- Una hilera física = UNA trayectoria continua aunque tenga huecos secos.
- Sigue el eje CENTRAL de cada hilera desde el inicio visible al final visible.
- No fragmentes una hilera, no unas dos vecinas y no dibujes entre hileras.
- Excluye caminos, cabeceras, bordes, árboles, postes, construcciones, sombras y maleza entre hileras.
- Incluye hileras débiles/secas si claramente forman parte del patrón de la parcela.
- Revisa especialmente primera/última hilera y zonas de perspectiva.
- Cada trayectoria: 6 a 12 puntos, coordenadas [x,y] normalizadas 0..1000.
- Antes de responder, recorre visualmente la parcela de izquierda a derecha (o perpendicular a las hileras) y verifica que no hayas saltado ninguna separación regular.

Devuelve SOLO JSON válido:
{"coverage_score":0.0,"confidence":0.0,"estimated_row_count":0,
 "rows":[{"id":1,"confidence":0.0,"points":[[x,y],[x,y],[x,y],[x,y],[x,y],[x,y]]}],"notes":""}
"""
    first = _tc_openai_json([pil], prompt_1, detail="high", model=_tc_openai_precision_model(), effort="high")
    first_rows = _tc_sanitize_rows(first)
    if not first_rows:
        raise RuntimeError("OpenAI no devolvió surcos en la primera revisión.")

    proposal = {
        "estimated_row_count": first.get("estimated_row_count", len(first_rows)) if isinstance(first, dict) else len(first_rows),
        "rows": [{"id": r["id"], "points": r["points_norm"]} for r in first_rows],
    }
    prompt_2 = """
Auditoría geométrica TerraCore sobre la MISMA foto.
Propuesta inicial: __PROPOSAL_JSON__

Corrige toda la propuesta:
1. elimina duplicados;
2. añade hileras reales omitidas, incluidos extremos;
3. mueve líneas que estén en suelo entre hileras al centro de la hilera correcta;
4. evita cruces y saltos de una hilera a otra;
5. conserva continuidad a través de huecos;
6. excluye caminos/árboles/edificios;
7. conserva el orden físico de las hileras;
8. usa 6 a 12 puntos por hilera, 0..1000.
No analices slots ni Salud.

Devuelve SOLO JSON válido:
{"coverage_score":0.0,"confidence":0.0,"estimated_row_count":0,
 "rows":[{"id":1,"confidence":0.0,"points":[[x,y],[x,y],[x,y],[x,y],[x,y],[x,y]]}],"audit_notes":""}
""".replace(
        "__PROPOSAL_JSON__",
        json.dumps(proposal, ensure_ascii=False, separators=(",", ":"))
    )
    audit = _tc_openai_json([pil], prompt_2, detail="high", model=_tc_openai_precision_model(), effort="high")
    rows = _tc_sanitize_rows(audit) or first_rows
    rows = _tc_remove_duplicate_rows_ai(rows, w, h)
    rows = _tc_sort_rows_ai(rows, w, h)

    # Tercera pasada: OpenAI ve la foto Y la guía dibujada. Es la mejor forma de
    # detectar huecos, duplicados y líneas que cayeron entre dos hileras.
    guide = _tc_render_row_guide_ai(pil, rows)
    proposal3 = [{"id": int(r["id"]), "points": r["points_norm"]} for r in rows]
    prompt_3 = """
AUDITORÍA FINAL DE SURCOS TERRACORE.
Imagen 1 = fotografía ORIGINAL. Imagen 2 = la misma foto con la propuesta de líneas cian Rxx.
Propuesta: __PROPOSAL_JSON__

Inspecciona fila por fila y devuelve la geometría FINAL.
- Cada línea debe estar encima del centro de UNA hilera real, nunca en el espacio entre hileras.
- Si falta una hilera entre dos Rxx, agrégala.
- Si dos Rxx pertenecen a la misma hilera, deja solo una.
- Si una Rxx cambia a la hilera vecina a mitad de camino, corrígela para seguir siempre la misma hilera.
- No marques vegetación ajena, caminos ni borde de parcela.
- Mantén hileras secas si su eje físico es reconocible.
- Usa 6 a 12 puntos por hilera, coordenadas 0..1000 de la imagen COMPLETA.

Devuelve SOLO JSON válido:
{"coverage_score":0.0,"confidence":0.0,"estimated_row_count":0,
 "rows":[{"id":1,"confidence":0.0,"points":[[x,y],[x,y],[x,y],[x,y],[x,y],[x,y]]}],"audit_notes":""}
""".replace(
        "__PROPOSAL_JSON__",
        json.dumps(proposal3, ensure_ascii=False, separators=(",", ":"))
    )
    try:
        final_audit = _tc_openai_json([pil, guide], prompt_3, detail="high", model=_tc_openai_precision_model(), effort="high")
        final_rows = _tc_sanitize_rows(final_audit)
        if final_rows:
            rows = _tc_remove_duplicate_rows_ai(final_rows, w, h)
            rows = _tc_sort_rows_ai(rows, w, h)
        else:
            final_audit = {}
    except Exception:
        final_audit = {}

    return {
        "pil": pil,
        "rows": rows,
        "coverage_score": _tc_clamp01((final_audit or {}).get("coverage_score", (audit or {}).get("coverage_score", (first or {}).get("coverage_score", 0.7))), 0.7),
        "confidence": _tc_clamp01((final_audit or {}).get("confidence", (audit or {}).get("confidence", (first or {}).get("confidence", 0.7))), 0.7),
        "debug": {"first": first, "audit": audit, "final_audit": final_audit},
    }

def _tc_crop_bbox_for_rows(pil, rows, margin_factor=1.5):
    w, h = pil.size
    groups = []
    mids = []
    for row in rows:
        pts = _tc_norm_to_px(row.get("points_norm", []), w, h)
        if len(pts):
            groups.append(pts)
            mids.append(_tc_point_on_polyline(pts, 0.5))
    if not groups:
        return (0, 0, w, h)
    all_pts = np.vstack(groups)
    spacing = max(14.0, min(w, h) * 0.02)
    if len(mids) > 1:
        ds = np.linalg.norm(np.diff(np.asarray(mids), axis=0), axis=1)
        ds = ds[ds > 3]
        if len(ds):
            spacing = float(np.median(ds))
    margin = max(18.0, spacing * margin_factor)
    x0 = max(0, int(np.floor(all_pts[:,0].min() - margin)))
    y0 = max(0, int(np.floor(all_pts[:,1].min() - margin)))
    x1 = min(w, int(np.ceil(all_pts[:,0].max() + margin)))
    y1 = min(h, int(np.ceil(all_pts[:,1].max() + margin)))
    return (x0, y0, max(x0+12, x1), max(y0+12, y1))


def _tc_reference_crop_ai(pil, rows, bbox, slots=False):
    x0, y0, x1, y1 = bbox
    crop = pil.crop(bbox).convert("RGB")
    bgr = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2BGR)
    fw, fh = pil.size
    cw, ch = crop.size
    for row in rows:
        pts = _tc_norm_to_px(row.get("points_norm", []), fw, fh)
        if len(pts) < 2:
            continue
        pts[:,0] -= x0
        pts[:,1] -= y0
        ip = np.rint(pts).astype(np.int32)
        cv2.polylines(bgr, [ip], False, (255, 0, 255), 1, cv2.LINE_AA)
        mid = _tc_point_on_polyline(pts, 0.5)
        label = f"R{int(row.get('id',0)):02d}"
        cv2.putText(bgr, label, (int(mid[0])+2, int(mid[1])-2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20,20,20), 3, cv2.LINE_AA)
        cv2.putText(bgr, label, (int(mid[0])+2, int(mid[1])-2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1, cv2.LINE_AA)
        if slots:
            for p in _tc_slot_positions_px_ai(row, (fw, fh)):
                xx, yy = int(round(p[0]-x0)), int(round(p[1]-y0))
                if 0 <= xx < cw and 0 <= yy < ch:
                    cv2.circle(bgr, (xx,yy), 2, (0,230,230), -1, cv2.LINE_AA)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _tc_crop_norm_to_global(points, bbox, fw, fh):
    x0, y0, x1, y1 = bbox
    cw, ch = max(1, x1-x0), max(1, y1-y0)
    out = []
    for p in points:
        q = _tc_norm_point(p)
        if q is None:
            continue
        x = x0 + q[0] / 1000.0 * max(1, cw-1)
        y = y0 + q[1] / 1000.0 * max(1, ch-1)
        out.append([
            float(np.clip(x / max(1, fw-1) * 1000.0, 0, 1000)),
            float(np.clip(y / max(1, fh-1) * 1000.0, 0, 1000)),
        ])
    return out


def _tc_clean_occupancy_ai(value, count):
    s = "".join(c for c in str(value or "").upper() if c in "OV")
    count = max(0, int(count or 0))
    if count <= 0:
        return s
    return s[:count]


def _tc_clean_index_list_ai(value, count):
    """Devuelve índices 1..count únicos/ordenados. Acepta lista, CSV o texto."""
    count = max(0, int(count or 0))
    if count <= 0:
        return []
    raw = value
    if isinstance(raw, str):
        import re
        raw = re.findall(r"\d+", raw)
    if not isinstance(raw, (list, tuple)):
        return []
    clean = []
    seen = set()
    for v in raw:
        try:
            i = int(v)
        except Exception:
            continue
        if 1 <= i <= count and i not in seen:
            seen.add(i)
            clean.append(i)
    clean.sort()
    return clean


def _tc_inventory_item_to_occ_ai(item):
    """Convierte respuesta compacta de IA a occupancy O/V sin exigir cadenas largas."""
    if not isinstance(item, dict):
        return 0, "", []
    try:
        count = int(item.get("slot_count", 0) or 0)
    except Exception:
        count = 0
    if count <= 1 or count > 500:
        return 0, "", []

    # Formato nuevo preferido: solo índices vacíos.
    vacant = _tc_clean_index_list_ai(
        item.get("vacant_indices", item.get("empty_indices", [])),
        count,
    )
    if "vacant_indices" in item or "empty_indices" in item:
        chars = ["O"] * count
        for idx in vacant:
            chars[idx - 1] = "V"
        return count, "".join(chars), vacant

    # Compatibilidad con respuestas antiguas.
    occ = "".join(c for c in str(item.get("occupancy", "") or "").upper() if c in "OV")
    if len(occ) == count:
        return count, occ, [i + 1 for i, c in enumerate(occ) if c == "V"]

    return 0, "", []


def _tc_inventory_item_valid_ai(item):
    if not isinstance(item, dict):
        return False
    count, occ, _ = _tc_inventory_item_to_occ_ai(item)
    # Para validar el INVENTARIO lo crítico es el conteo y los faltantes.
    # Si OpenAI no devuelve puntos suficientes, conservamos la trayectoria
    # de la detección global en lugar de cancelar toda la parcela.
    return count > 1 and len(occ) == count


def _tc_refine_inventory_batches_openai(pil, rough_rows, batch_size=8):
    fw, fh = pil.size
    rough_rows = _tc_sort_rows_ai(rough_rows, fw, fh)
    refined = []

    for start in range(0, len(rough_rows), batch_size):
        group = rough_rows[start:start + batch_size]
        bbox = _tc_crop_bbox_for_rows(pil, group, 1.45)
        original = pil.crop(bbox).convert("RGB")
        guide = _tc_reference_crop_ai(pil, group, bbox, slots=False)
        ids = [int(r["id"]) for r in group]

        prompt = f"""
TerraCore Inventario. Imagen 1 es el recorte ORIGINAL. Imagen 2 muestra una guía magenta aproximada con etiquetas Rxx.
Analiza SOLO estos IDs: {ids}.

Para cada Rxx:
1) Corrige la trayectoria para quedar en el CENTRO de la hilera real. Devuelve 5 a 9 puntos 0..1000 respecto a ESTE RECORTE.
2) Determina la secuencia REGULAR de posiciones reales de planta entre start_t y end_t.
3) slot_count = número TOTAL de posiciones esperadas, incluyendo faltantes.
4) NO devuelvas una cadena O/V. Devuelve SOLO vacant_indices: lista de índices 1-based de las posiciones realmente vacías.
5) Una planta seca, amarilla, débil o sin vigor sigue estando OCUPADA si físicamente existe. Solo una ausencia clara es vacía.
6) Usa la separación repetitiva de plantas en ESA hilera y en sus vecinas para fijar el paso real.
7) No cuentes hojas, manchas, postes, sombras, maleza ni textura del suelo.
8) start_t y end_t van de 0..1000 sobre la trayectoria corregida y delimitan la zona real con slots.

Devuelve SOLO JSON válido:
{{"rows":[{{"id":1,"confidence":0.0,"points":[[x,y],[x,y],[x,y],[x,y],[x,y]],"start_t":0,"end_t":1000,"slot_count":52,"vacant_indices":[7,19]}}]}}

Devuelve exactamente todos los IDs pedidos. Revisa dos veces que ningún vacant_index sea mayor que slot_count.
"""
        try:
            data = _tc_openai_json([original, guide], prompt, detail="high")
        except Exception:
            data = {}

        returned = data.get("rows", []) if isinstance(data, dict) else []
        by_id = {}
        for item in returned:
            if isinstance(item, dict):
                try:
                    by_id[int(item.get("id"))] = item
                except Exception:
                    pass

        for rough in group:
            rid = int(rough["id"])
            item = by_id.get(rid, {})

            # Reintentos puntuales. Usamos un formato compacto para evitar
            # errores por cadenas O/V demasiado largas.
            if not _tc_inventory_item_valid_ai(item):
                single_bbox = _tc_crop_bbox_for_rows(pil, [rough], 2.15)
                single_original = pil.crop(single_bbox).convert("RGB")
                single_guide = _tc_reference_crop_ai(pil, [rough], single_bbox, slots=False)

                retry_prompts = [
                    f"""
Revisa SOLO R{rid:02d}. Imagen 1 original; imagen 2 guía aproximada.
Cuenta posiciones reales de planta siguiendo la regularidad de la hilera.
Devuelve SOLO JSON válido:
{{"id":{rid},"confidence":0.0,"points":[[x,y],[x,y],[x,y],[x,y],[x,y]],"start_t":0,"end_t":1000,"slot_count":52,"vacant_indices":[7,19]}}
NO escribas occupancy O/V. vacant_indices son índices 1-based de faltantes claros. Planta seca pero presente NO es vacía.
""",
                    f"""
Auditoría final de R{rid:02d}. Ignora conteos previos si no coinciden con la fotografía.
Primero identifica el inicio y final reales del tramo plantado. Después estima el paso repetitivo entre plantas y cuenta TODOS los slots esperados.
Marca únicamente las AUSENCIAS CLARAS con vacant_indices.
Devuelve SOLO JSON:
{{"id":{rid},"confidence":0.0,"points":[[x,y],[x,y],[x,y],[x,y],[x,y]],"start_t":0,"end_t":1000,"slot_count":0,"vacant_indices":[]}}
""",
                ]

                for rp in retry_prompts:
                    try:
                        candidate = _tc_openai_json([single_original, single_guide], rp, detail="high")
                    except Exception:
                        candidate = {}
                    if _tc_inventory_item_valid_ai(candidate):
                        item = candidate
                        # Los puntos del recorte puntual deben transformarse con su bbox.
                        item = dict(item)
                        item["_single_bbox"] = single_bbox
                        break

            # Si una hilera sigue sin poder validarse, NO cancelamos toda la parcela.
            # Conservamos la geometría detectada y la marcamos para revisión, sin inventar slots.
            if not _tc_inventory_item_valid_ai(item):
                refined.append({
                    "id": rid,
                    "points_norm": rough.get("points_norm", []),
                    "confidence": min(0.35, _tc_clamp01(rough.get("confidence", 0.35), 0.35)),
                    "start_t": 0.0,
                    "end_t": 1000.0,
                    "slot_count": 0,
                    "occupancy": "",
                    "needs_review": True,
                    "review_reason": f"R{rid:02d}: la IA no pudo validar slots después de varios intentos.",
                })
                continue

            bbox_used = item.pop("_single_bbox", bbox)
            pts = _tc_crop_norm_to_global(item.get("points") or [], bbox_used, fw, fh)
            if len(pts) < 2:
                pts = rough["points_norm"]

            count, occ, vacant = _tc_inventory_item_to_occ_ai(item)
            try:
                start_t = float(np.clip(float(item.get("start_t", 0) or 0), 0, 1000))
                end_t = float(np.clip(float(item.get("end_t", 1000) or 1000), 0, 1000))
            except Exception:
                start_t, end_t = 0.0, 1000.0

            refined.append({
                "id": rid,
                "points_norm": pts,
                "confidence": _tc_clamp01(item.get("confidence", rough.get("confidence", 0.6)), 0.6),
                "start_t": start_t,
                "end_t": end_t,
                "slot_count": max(0, count),
                "occupancy": occ,
                "vacant_indices": vacant,
                "needs_review": False,
            })

    refined = _tc_sort_rows_ai(refined, fw, fh)
    for i, row in enumerate(refined, 1):
        row["id"] = i
    return refined


def _tc_slot_positions_px_ai(row, full_size):
    w, h = full_size
    pts = _tc_norm_to_px(row.get("points_norm", []), w, h)
    count = int(row.get("slot_count", 0) or 0)
    if len(pts) < 2 or count <= 0:
        return []
    a = float(np.clip(float(row.get("start_t", 0) or 0) / 1000.0, 0, 1))
    b = float(np.clip(float(row.get("end_t", 1000) or 1000) / 1000.0, 0, 1))
    if b < a:
        a, b = b, a
    ts = np.linspace(a, b, count) if count > 1 else np.array([(a+b)/2])
    return [_tc_point_on_polyline(pts, float(t)) for t in ts]


def _tc_audit_inventory_openai(pil, rows, batch_size=8):
    """Segunda auditoría de slots con OpenAI sin redetectar la parcela completa."""
    if not rows:
        return rows

    audited = [dict(r) for r in rows]
    by_id = {int(r.get("id", i + 1)): r for i, r in enumerate(audited)}

    for start in range(0, len(audited), batch_size):
        group = audited[start:start + batch_size]
        valid_group = [r for r in group if int(r.get("slot_count", 0) or 0) > 0]
        if not valid_group:
            continue

        bbox = _tc_crop_bbox_for_rows(pil, valid_group, 1.55)
        original = pil.crop(bbox).convert("RGB")
        guide = _tc_reference_crop_ai(pil, valid_group, bbox, slots=True)
        spec = []
        for r in valid_group:
            count = int(r.get("slot_count", 0) or 0)
            occ = _tc_clean_occupancy_ai(r.get("occupancy", ""), count)
            spec.append({
                "id": int(r["id"]),
                "slot_count": count,
                "vacant_indices": [i + 1 for i, c in enumerate(occ) if c == "V"],
            })

        prompt = f"""
TerraCore INVENTARIO - AUDITORÍA FINAL DE PRECISIÓN.
Imagen 1 = fotografía ORIGINAL. Imagen 2 = guía con hileras y slots calculados.
Resultado preliminar: {json.dumps(spec, ensure_ascii=False, separators=(',', ':'))}

Tu trabajo NO es inventar otro sistema: revisa visualmente cada Rxx y corrige SOLO si la fotografía lo justifica.

REGLAS ESTRICTAS:
- Una posición física esperada de planta = un slot.
- Usa el patrón repetitivo de separación entre plantas de la MISMA hilera y hileras vecinas.
- Planta seca, amarilla, pequeña o débil PERO físicamente presente = OCUPADO, no vacío.
- VACÍO solo cuando se ve una ausencia real en una posición esperada.
- No confundas suelo, sombras, postes, maleza o textura con plantas.
- No dupliques posiciones.
- No cambies drásticamente slot_count si la primera cuenta ya coincide con el patrón.
- Revisa especialmente inicio y final de cada hilera.

Devuelve SOLO JSON válido, sin texto extra:
{{"rows":[{{"id":1,"confidence":0.0,"slot_count":52,"vacant_indices":[7,19]}}]}}
Devuelve exactamente todos los IDs de la propuesta.
"""
        try:
            data = _tc_openai_json([original, guide], prompt, detail="high")
        except Exception:
            continue

        returned = data.get("rows", []) if isinstance(data, dict) else []
        for item in returned:
            if not isinstance(item, dict):
                continue
            try:
                rid = int(item.get("id"))
            except Exception:
                continue
            row = by_id.get(rid)
            if row is None:
                continue
            count, occ, vacant = _tc_inventory_item_to_occ_ai(item)
            if count <= 1 or len(occ) != count:
                continue

            old_count = int(row.get("slot_count", 0) or 0)
            # Protección contra saltos absurdos de conteo en la auditoría.
            if old_count > 4:
                ratio = count / float(old_count)
                if ratio < 0.72 or ratio > 1.35:
                    continue

            row["slot_count"] = count
            row["occupancy"] = occ
            row["vacant_indices"] = vacant
            row["confidence"] = max(
                _tc_clamp01(row.get("confidence", 0.0), 0.0),
                _tc_clamp01(item.get("confidence", 0.0), 0.0),
            )
            row["inventory_audited"] = True

    return audited


def _tc_inventory_consistency_ai(pil, rows):
    """Corrige solo outliers grandes de slot_count usando paso geométrico de hileras vecinas."""
    if not rows:
        return rows
    w, h = pil.size
    result = [dict(r) for r in rows]
    pitch = []
    lengths = []
    for r in result:
        pts = _tc_norm_to_px(r.get("points_norm", []), w, h)
        count = int(r.get("slot_count", 0) or 0)
        a = float(np.clip(float(r.get("start_t", 0) or 0)/1000.0, 0, 1))
        b = float(np.clip(float(r.get("end_t", 1000) or 1000)/1000.0, 0, 1))
        if b < a: a, b = b, a
        length = 0.0
        if len(pts) >= 2:
            # longitud aproximada del tramo plantado
            samples = np.asarray([_tc_point_on_polyline(pts, t) for t in np.linspace(a,b,40)], dtype=np.float32)
            length = float(np.sum(np.linalg.norm(np.diff(samples,axis=0),axis=1))) if len(samples)>1 else 0.0
        lengths.append(length)
        pitch.append(length/max(1,count-1) if count>2 and length>5 else np.nan)

    p = np.asarray(pitch,dtype=float)
    for i,r in enumerate(result):
        old = int(r.get("slot_count",0) or 0)
        if old <= 2 or not np.isfinite(p[i]):
            continue
        lo=max(0,i-3); hi=min(len(result),i+4)
        neigh=p[lo:hi]
        neigh=neigh[np.isfinite(neigh)]
        if len(neigh)<3:
            continue
        med=float(np.median(neigh))
        if med<=1:
            continue
        ratio=p[i]/med
        # solo corregir outliers claros, no pequeñas diferencias reales.
        if 0.68 <= ratio <= 1.47:
            continue
        new=max(2,int(round(lengths[i]/med))+1)
        if new<2 or new>500:
            continue
        old_vac=_tc_clean_index_list_ai(r.get("vacant_indices",[]),old)
        mapped=[]
        for idx in old_vac:
            q=0.0 if old<=1 else (idx-1)/float(old-1)
            mapped.append(1+int(round(q*(new-1))))
        mapped=_tc_clean_index_list_ai(mapped,new)
        r["slot_count"]=new
        r["vacant_indices"]=mapped
        r["occupancy"]="".join("V" if j in set(mapped) else "O" for j in range(1,new+1))
        r["confidence"]=min(float(r.get("confidence",0.7)),0.78)
        r["consistency_adjusted"]=True
    return result


def _tc_draw_inventory_ai(pil, rows):
    """
    INVENTARIO LIMPIO PARA TERRACORE.

    La IA sigue calculando:
    - surcos
    - slots
    - ocupados
    - vacíos

    Pero en la imagen SOLO se dibuja:
    - una línea fina por surco
    - número 01..N únicamente ARRIBA

    No se dibujan números abajo.
    Los círculos azules y las X naranjas tampoco se dibujan.
    """
    original = cv2.cvtColor(np.asarray(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = original.shape[:2]

    # Una sola banda exclusiva arriba para numeración.
    band_h = max(46, int(round(h * 0.075)))
    wine_bgr = (55, 47, 114)  # #722F37

    canvas = np.full((h + band_h, w, 3), wine_bgr, dtype=np.uint8)
    canvas[band_h:band_h + h, :, :] = original

    line_th = max(1, int(round(min(w, h) / 950)))
    table = []
    confs = []

    # Calcular separación aproximada entre hileras para adaptar el tamaño del texto.
    mids = []
    for r in rows:
        p = _tc_norm_to_px(r.get("points_norm", []), w, h)
        if len(p) >= 2:
            mids.append(_tc_point_on_polyline(p, 0.5))

    spacing = 24.0
    if len(mids) > 2:
        mm = np.asarray(mids, dtype=np.float32)
        mm = mm[np.argsort(mm[:, 0])]
        d = np.linalg.norm(np.diff(mm, axis=0), axis=1)
        d = d[d > 2]
        if len(d):
            spacing = float(np.median(d))

    font = float(np.clip(spacing / 38.0, 0.26, 0.46))
    font_th = 1

    # Dos carriles SOLO arriba para evitar que números vecinos se monten.
    lane_y_top = [max(14, int(band_h * 0.34)), max(30, int(band_h * 0.72))]

    for i, row in enumerate(rows, 1):
        row["id"] = i
        pts = _tc_norm_to_px(row.get("points_norm", []), w, h)
        if len(pts) < 2:
            continue

        ip = np.rint(pts).astype(np.int32)

        # Línea fina para saber exactamente a qué hilera pertenece el número.
        ip_canvas = ip.copy()
        ip_canvas[:, 1] += band_h
        cv2.polylines(
            canvas,
            [ip_canvas],
            False,
            (235, 235, 235),
            line_th,
            cv2.LINE_AA,
        )

        label = f"{i:02d}"

        # ÚNICAMENTE el extremo visual superior de la hilera.
        top_pt = ip[int(np.argmin(ip[:, 1]))]
        lane = (i - 1) % 2
        x = int(top_pt[0])
        ty = lane_y_top[lane]

        (tw, th), base = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font,
            font_th,
        )
        tx = int(np.clip(x - tw / 2, 1, max(1, w - tw - 2)))
        ty2 = int(np.clip(ty, th + 2, band_h - base - 2))

        cv2.putText(
            canvas,
            label,
            (tx, ty2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font,
            (15, 15, 15),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (tx, ty2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Los slots se siguen calculando para la tabla, pero NO se dibujan.
        positions = _tc_slot_positions_px_ai(row, (w, h))
        occ = _tc_clean_occupancy_ai(row.get("occupancy", ""), len(positions))
        row["occupancy"] = occ
        occupied = sum(1 for state in occ if state != "V")
        empty = sum(1 for state in occ if state == "V")

        conf = _tc_clamp01(row.get("confidence", 0), 0)
        confs.append(conf)

        status = tr("Validado por IA", "Validé par IA")
        if row.get("needs_review"):
            status = tr("Revisar", "À vérifier")
        elif row.get("consistency_adjusted"):
            status = tr("Ajustado por consistencia", "Ajusté par cohérence")

        table.append({
            tr("Surco", "Rang"): label,
            tr("Slots", "Emplacements"): len(positions),
            tr("Ocupados", "Occupés"): occupied,
            tr("Vacíos", "Vides"): empty,
            tr("Confianza", "Confiance"): round(conf * 100, 1),
            tr("Estado", "État"): status,
        })

    return canvas, pd.DataFrame(table), float(np.mean(confs)) if confs else 0.0, rows

def _tc_analyze_inventory_openai(uploaded_image, base=None):
    base = base or _tc_detect_rows_openai(uploaded_image)
    refined = _tc_refine_inventory_batches_openai(base["pil"], base["rows"], batch_size=8)
    if not refined:
        raise RuntimeError("OpenAI no devolvió un Inventario refinado.")
    # Segunda lectura IA: confirma conteo y vacíos sin redibujar surcos.
    refined = _tc_audit_inventory_openai(base["pil"], refined, batch_size=8)
    refined = _tc_inventory_consistency_ai(base["pil"], refined)
    image, table, local_conf, refined = _tc_draw_inventory_ai(base["pil"], refined)
    if table.empty:
        raise RuntimeError("OpenAI no pudo construir la tabla de Inventario.")
    return {
        "image": image,
        "table": table,
        "count": len(table),
        "confidence": float(np.mean([base.get("confidence",0), local_conf])),
        "coverage_score": base.get("coverage_score",0),
        "rows": refined,
        "model": st.session_state.get("tc_last_ai_provider", f"Gemini ({_tc_gemini_model()})"),
        "warnings": (
            [r.get("review_reason") for r in refined if r.get("needs_review") and r.get("review_reason")]
            + list(st.session_state.get("tc_ai_warnings_runtime", []) or [])
        ),
        "debug": base.get("debug",{}),
    }


def _tc_select_best_capture_openai(uploaded_images):
    # Advertencias nuevas para esta ejecución. No arrastrar mensajes de una parcela anterior.
    st.session_state.tc_ai_warnings_runtime = []
    st.session_state.tc_last_ai_warning = ""
    candidates = []
    errors = []
    for up in uploaded_images:
        try:
            base = _tc_detect_rows_openai(up)
            count = len(base.get("rows",[]))
            score = float(base.get("coverage_score",0))*0.55 + float(base.get("confidence",0))*0.30 + min(1.0,count/80.0)*0.15
            candidates.append((score, up, base))
        except Exception as exc:
            errors.append(f"{up.name}: {exc}")
    if not candidates:
        raise RuntimeError(" | ".join(errors) if errors else "No se pudo analizar la captura con IA.")
    candidates.sort(key=lambda item:item[0], reverse=True)
    _, up, base = candidates[0]
    return up, _tc_analyze_inventory_openai(up, base=base), errors


def _tc_clean_red_ranges_ai(value):
    """Normaliza intervalos rojos 0..1000 a lista ordenada sin solapamientos."""
    ranges=[]
    if isinstance(value, dict):
        value=value.get("red_ranges", value.get("ranges", []))
    if not isinstance(value,(list,tuple)):
        return []
    for item in value:
        if isinstance(item,dict):
            a=item.get("start",item.get("start_t",0)); b=item.get("end",item.get("end_t",0))
        elif isinstance(item,(list,tuple)) and len(item)>=2:
            a,b=item[0],item[1]
        else:
            continue
        try:
            a=float(np.clip(float(a),0,1000)); b=float(np.clip(float(b),0,1000))
        except Exception:
            continue
        if b<a: a,b=b,a
        if b-a<4: continue
        ranges.append([a,b])
    ranges.sort(key=lambda z:z[0])
    merged=[]
    for a,b in ranges:
        if merged and a<=merged[-1][1]+8:
            merged[-1][1]=max(merged[-1][1],b)
        else:
            merged.append([a,b])
    return merged


def _tc_health_segment_guide_ai(pil, rows, bbox):
    """Guía limpia: línea Rxx con marcas 0/25/50/75/100 para localizar tramos."""
    x0,y0,x1,y1=bbox
    crop=pil.crop(bbox).convert("RGB")
    bgr=cv2.cvtColor(np.asarray(crop),cv2.COLOR_RGB2BGR)
    fw,fh=pil.size; cw,ch=crop.size
    for r in rows:
        rid=int(r.get("id",0))
        pts=_tc_norm_to_px(r.get("points_norm",[]),fw,fh)
        if len(pts)<2: continue
        pts[:,0]-=x0; pts[:,1]-=y0
        ip=np.rint(pts).astype(np.int32)
        cv2.polylines(bgr,[ip],False,(255,0,255),1,cv2.LINE_AA)
        for frac,label in ((0.0,"0"),(0.25,"25"),(0.5,"50"),(0.75,"75"),(1.0,"100")):
            p=_tc_point_on_polyline(pts,frac)
            xx,yy=int(p[0]),int(p[1])
            if 0<=xx<cw and 0<=yy<ch:
                cv2.circle(bgr,(xx,yy),2,(0,255,255),-1,cv2.LINE_AA)
                if frac in (0.0,0.5,1.0):
                    cv2.putText(bgr,f"R{rid:02d}:{label}",(xx+2,yy-2),cv2.FONT_HERSHEY_SIMPLEX,0.33,(20,20,20),3,cv2.LINE_AA)
                    cv2.putText(bgr,f"R{rid:02d}:{label}",(xx+2,yy-2),cv2.FONT_HERSHEY_SIMPLEX,0.33,(255,255,255),1,cv2.LINE_AA)
    return Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))


def _tc_health_ranges_openai(pil, rows, batch_size=4):
    """OpenAI devuelve TRAMOS rojos continuos, no un color por slot."""
    result={}
    valid=[r for r in rows if len(r.get("points_norm",[]))>=2]
    for start in range(0,len(valid),batch_size):
        group=valid[start:start+batch_size]
        bbox=_tc_crop_bbox_for_rows(pil,group,1.85)
        original=pil.crop(bbox).convert("RGB")
        guide=_tc_health_segment_guide_ai(pil,group,bbox)
        spec=[{"id":int(r["id"]),"vacant_indices":_tc_clean_index_list_ai(r.get("vacant_indices",[]),int(r.get("slot_count",0) or 0))} for r in group]
        prompt=f"""
TerraCore SALUD — CLASIFICACIÓN POR TRAMOS.
Imagen 1 = recorte ORIGINAL de alta resolución.
Imagen 2 = guía magenta; cada hilera Rxx lleva referencias 0,25,50,75,100 desde un extremo al otro.
Hileras: {json.dumps(spec,ensure_ascii=False,separators=(',',':'))}

Devuelve para cada Rxx únicamente los TRAMOS que deben ser ROJOS como red_ranges en escala 0..1000 sobre la longitud total de la hilera.
Ejemplo: [120,260] significa rojo aproximadamente del 12% al 26% de esa hilera.

CRITERIO ESTRICTO (NO SOBREESTIMES VERDE):
- VERDE solo si se ve copa/vegetación de vid claramente viva, densa y continua, comparable con las zonas vigorosas cercanas bajo iluminación similar.
- ROJO si el tramo está vacío, con suelo expuesto donde debería existir vid, seco, marrón/beige, amarillento, muy ralo, interrumpido o con cobertura claramente inferior al patrón sano local.
- Una vid físicamente presente pero débil sigue siendo OCUPADA en Inventario, pero aquí el tramo debe ser ROJO.
- Un punto pequeño de verde NO convierte un tramo ralo/seco en verde.
- No confundas maleza entre hileras con copa de vid.
- No marques rojo únicamente por sombra fuerte; compara con la continuidad de la hilera.
- Sigue la MISMA hilera; no saltes a una vecina.
- Si dudas entre verde y rojo por baja cobertura, usa ROJO.
- Une zonas rojas cercanas separadas por un hueco verde muy corto.

Devuelve SOLO JSON válido:
{{"rows":[{{"id":1,"confidence":0.0,"red_ranges":[[120,260],[610,740]]}}]}}
Devuelve exactamente todos los IDs.
"""
        try:
            data=_tc_openai_json([original,guide],prompt,detail="high",model=_tc_openai_precision_model(),effort="high")
        except Exception:
            data={}
        by={}
        for item in (data.get("rows",[]) if isinstance(data,dict) else []):
            if isinstance(item,dict):
                try: by[int(item.get("id"))]=item
                except Exception: pass
        for r in group:
            rid=int(r["id"]); item=by.get(rid,{})
            result[rid]={
                "red_ranges":_tc_clean_red_ranges_ai(item.get("red_ranges",[])),
                "confidence":_tc_clamp01(item.get("confidence",0.65),0.65),
            }
    return result


def _tc_health_ranges_audit_openai(pil, rows, ranges_map, batch_size=4):
    """Segunda auditoría orientada a encontrar falsos verdes, no a borrar rojos."""
    final={int(k):dict(v) for k,v in (ranges_map or {}).items()}
    valid=[r for r in rows if len(r.get("points_norm",[]))>=2]
    for start in range(0,len(valid),batch_size):
        group=valid[start:start+batch_size]
        bbox=_tc_crop_bbox_for_rows(pil,group,2.0)
        original=pil.crop(bbox).convert("RGB")
        guide=_tc_health_segment_guide_ai(pil,group,bbox)
        spec=[]
        for r in group:
            rid=int(r["id"])
            spec.append({"id":rid,"initial_red_ranges":final.get(rid,{}).get("red_ranges",[])})
        prompt=f"""
AUDITORÍA FINAL TERRACORE SALUD.
Imagen 1 ORIGINAL, imagen 2 guía de las mismas hileras. Clasificación inicial: {json.dumps(spec,ensure_ascii=False,separators=(',',':'))}

Busca especialmente FALSOS VERDES. Para cada Rxx devuelve la lista FINAL de red_ranges 0..1000.
- Conserva rojo donde hay suelo expuesto, hueco, sequedad, color marrón/beige/amarillo, copa muy rala o vigor claramente menor.
- Verde exige vegetación de vid visible, suficiente, continua y comparable con referencia sana local.
- No basta un pequeño punto verde aislado.
- Si una zona inicial verde parece dudosa o claramente más débil que sus vecinas, conviértela a rojo.
- No conviertas a rojo una zona sana solo por sombra uniforme.
- No cambies la geometría de las hileras.

Devuelve SOLO JSON válido:
{{"rows":[{{"id":1,"confidence":0.0,"red_ranges":[[100,250],[600,800]]}}]}}
"""
        try:
            data=_tc_openai_json([original,guide],prompt,detail="high",model=_tc_openai_precision_model(),effort="high")
        except Exception:
            continue
        for item in (data.get("rows",[]) if isinstance(data,dict) else []):
            if not isinstance(item,dict): continue
            try: rid=int(item.get("id"))
            except Exception: continue
            if rid not in [int(r["id"]) for r in group]: continue
            rr=_tc_clean_red_ranges_ai(item.get("red_ranges",[]))
            final[rid]={"red_ranges":rr,"confidence":_tc_clamp01(item.get("confidence",0.72),0.72),"audited":True}
    return final


def _tc_row_vegetation_scores_ai(pil, row, samples=120):
    """Chequeo conservador de evidencia verde real alrededor del eje de la hilera."""
    rgb=np.asarray(pil.convert("RGB"))
    bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    h,w=bgr.shape[:2]
    pts=_tc_norm_to_px(row.get("points_norm",[]),w,h)
    if len(pts)<2: return np.zeros(samples,dtype=np.float32)
    scores=[]
    patch=max(3,int(round(min(w,h)*0.006)))
    for t in np.linspace(0,1,samples):
        p=_tc_point_on_polyline(pts,float(t)); x=int(round(p[0])); y=int(round(p[1]))
        x0=max(0,x-patch); x1=min(w,x+patch+1); y0=max(0,y-patch); y1=min(h,y+patch+1)
        crop=bgr[y0:y1,x0:x1]
        if crop.size==0: scores.append(0.0); continue
        arr=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB).astype(np.float32)
        r,g,b=arr[:,:,0],arr[:,:,1],arr[:,:,2]
        exg=2*g-r-b
        ngr=(g-r)/(g+r+1e-6)
        hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
        hh,ss,vv=cv2.split(hsv)
        mask=(exg>5)&(ngr>-0.02)&(hh>=18)&(hh<=115)&(ss>=12)&(vv>=20)&(g>=r*0.86)&(g>=b*0.86)
        scores.append(float(np.mean(mask)))
    return np.asarray(scores,dtype=np.float32)


def _tc_draw_health_ai_v4(pil, rows, ranges_map):
    bgr=cv2.cvtColor(np.asarray(pil.convert("RGB")),cv2.COLOR_RGB2BGR)
    h,w=bgr.shape[:2]
    thick=max(2,int(round(min(w,h)/620)))
    green,red=(45,210,50),(35,35,245)
    total_g=total_r=0.0; red_points=[]; confs=[]

    for row in rows:
        rid=int(row["id"])
        pts=_tc_norm_to_px(row.get("points_norm",[]),w,h)
        if len(pts)<2: continue
        info=ranges_map.get(rid,{})
        ranges=_tc_clean_red_ranges_ai(info.get("red_ranges",[]))
        confs.append(_tc_clamp01(info.get("confidence",0),0))

        n=140
        ts=np.linspace(0.0,1.0,n)
        pp=np.asarray([_tc_point_on_polyline(pts,float(t)) for t in ts],dtype=np.float32)
        veg=_tc_row_vegetation_scores_ai(pil,row,samples=n)
        positive=veg[veg>0]
        # Solo fuerza rojo cuando prácticamente no existe evidencia vegetal.
        low_floor=max(0.018,float(np.percentile(positive,18))*0.42) if len(positive)>=10 else 0.022
        states=[]
        for j,t in enumerate(ts):
            tn=t*1000.0
            red_ai=any(a<=tn<=b for a,b in ranges)
            red_low=bool(veg[j]<low_floor)
            states.append("R" if (red_ai or red_low) else "G")

        # suavizado 1D conservador: verde solo si forma una corrida real.
        s=states[:]
        for j in range(1,n-1):
            if states[j]=="G" and states[j-1]=="R" and states[j+1]=="R": s[j]="R"
        states=s
        # expande 1 muestra los tramos rojos: evita cortes verdes microscópicos.
        red_idx={j for j,v in enumerate(states) if v=="R"}
        expanded=set(red_idx)
        for j in red_idx:
            if j>0: expanded.add(j-1)
            if j<n-1: expanded.add(j+1)
        states=["R" if j in expanded else "G" for j in range(n)]

        for j in range(n-1):
            p0,p1=pp[j],pp[j+1]
            seglen=float(np.linalg.norm(p1-p0))
            is_red=(states[j]=="R" or states[j+1]=="R")
            color=red if is_red else green
            cv2.line(bgr,(int(round(p0[0])),int(round(p0[1]))),(int(round(p1[0])),int(round(p1[1]))),color,thick,cv2.LINE_AA)
            if is_red:
                total_r+=seglen; red_points.append((float((p0[0]+p1[0])/2),float((p0[1]+p1[1])/2)))
            else:
                total_g+=seglen

        # etiqueta SOLO ARRIBA, igual que Inventario
        ip=np.rint(pts).astype(np.int32); label=f"{rid:02d}"
        ep=ip[int(np.argmin(ip[:,1]))]
        x,y=int(ep[0]),int(ep[1]); font=0.44
        (tw,th),base=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,font,2)
        tx=int(np.clip(x-tw/2,2,max(2,w-tw-3))); off=6+(rid%2)*5
        ty=int(np.clip(y-off,th+4,h-4))
        cv2.rectangle(bgr,(max(0,tx-3),max(0,ty-th-3)),(min(w-1,tx+tw+3),min(h-1,ty+base+3)),(35,20,25),-1)
        cv2.putText(bgr,label,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,font,(255,255,255),2,cv2.LINE_AA)

    total=max(1e-6,total_g+total_r)
    return {"annotated":bgr,"green_pct":100.0*total_g/total,"red_pct":100.0*total_r/total,
            "red_points":red_points,"confidence":float(np.mean(confs)) if confs else 0.0}


def _tc_health_diagnosis_openai(pil, visual, row_count):
    w,h=pil.size
    distribution={"left":0,"center":0,"right":0,"top":0,"middle":0,"bottom":0}
    for x,y in visual.get("red_points",[]):
        distribution["left" if x<w/3 else "center" if x<2*w/3 else "right"] += 1
        distribution["top" if y<h/3 else "middle" if y<2*h/3 else "bottom"] += 1
    prompt=f"""
TerraCore. Haz un diagnóstico VISUAL PRELIMINAR de esta fotografía.
Resultado por tramos: surcos={row_count}, verde={visual.get('green_pct',0):.1f}%, rojo={visual.get('red_pct',0):.1f}%, distribución roja={json.dumps(distribution)}.
No afirmes enfermedad ni nutriente específico solo por la foto.
Devuelve SOLO JSON válido:
{{"zona_mas_afectada":"texto corto","nivel_afectacion_visual":"bajo|medio|alto","diagnostico_visual":"1 a 3 frases","causas_probables":["..."],"explicacion_nutrientes":"texto breve","recomendaciones_iniciales":["..."],"nota_diagnostico":"Diagnóstico visual preliminar..."}}
"""
    try:
        data=_tc_openai_json([pil],prompt,detail="high",model=_tc_openai_precision_model(),effort="medium")
        return data if isinstance(data,dict) else {}
    except Exception:
        return {}


def _tc_analyze_health_openai(uploaded_image, rows):
    pil=Image.open(io.BytesIO(uploaded_image.getvalue())).convert("RGB")
    if not rows:
        raise RuntimeError("No hay geometría de Inventario confirmada.")
    ranges=_tc_health_ranges_openai(pil,rows,batch_size=4)
    ranges=_tc_health_ranges_audit_openai(pil,rows,ranges,batch_size=4)
    visual=_tc_draw_health_ai_v4(pil,rows,ranges)
    diag=_tc_health_diagnosis_openai(pil,visual,len(rows))
    result={
        "count":len(rows),"green_pct":float(visual["green_pct"]),"red_pct":float(visual["red_pct"]),"angle":0.0,
        "annotated":visual["annotated"],"result_url":"","zona_mas_afectada":diag.get("zona_mas_afectada","No determinada"),
        "nivel_afectacion_visual":diag.get("nivel_afectacion_visual","No determinado"),"diagnostico_visual":diag.get("diagnostico_visual",""),
        "causas_probables":diag.get("causas_probables",[]),"explicacion_nutrientes":diag.get("explicacion_nutrientes",""),
        "recomendaciones_iniciales":diag.get("recomendaciones_iniciales",[]),"nota_diagnostico":diag.get("nota_diagnostico","Diagnóstico visual preliminar."),
        "detalle_zonas":{},"metodo":"gemini-flash-lite-openai-fallback","confidence":float(visual.get("confidence",0)),
    }
    result["backend"]={"metodo":"gemini-flash-lite-openai-fallback","analisis":{
        "surcos_estimados":result["count"],"verde_pct":result["green_pct"],"rojo_pct":result["red_pct"],
        "zona_mas_afectada":result["zona_mas_afectada"],"nivel_afectacion_visual":result["nivel_afectacion_visual"],
        "diagnostico_visual":result["diagnostico_visual"],"causas_probables":result["causas_probables"],
        "explicacion_nutrientes":result["explicacion_nutrientes"],"recomendaciones_iniciales":result["recomendaciones_iniciales"],
        "nota_diagnostico":result["nota_diagnostico"],
    }}
    return result



# ============================================================
# TERRACORE - MOTOR ACTIVO SOLO GEMINI
# ============================================================
# IMPORTANTE:
# - Inventario de FOTOS: SOLO Gemini decide surcos y slots.
# - Salud de FOTOS: SOLO Gemini clasifica los slots.
# - PIL se usa únicamente para recortar/dibujar la respuesta de Gemini.
# - NO se usa OpenAI como respaldo.
# - NO se usa OpenCV para decidir surcos, slots o Salud en este flujo.
# - Se conservan la interfaz, Google Drive, Google Sheets, historial,
#   reporte, idiomas y el resto de la aplicación original.
# ============================================================


def _tg_secret_text(name, default=""):
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return str(os.getenv(name, default)).strip()


def _tg_gemini_api_key():
    value = _tg_secret_text("GEMINI_API_KEY")
    if value:
        return value
    try:
        block = st.secrets.get("gemini", {})
        if hasattr(block, "get"):
            return str(block.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        pass
    return ""


def _tg_gemini_model():
    value = _tg_secret_text("GEMINI_VISION_MODEL", "gemini-3.5-flash-lite")
    # Compatibilidad con el modelo retirado que ya devolvió 404.
    if value == "gemini-2.5-flash-lite":
        value = "gemini-3.5-flash-lite"
    return value or "gemini-3.5-flash-lite"


def _tg_jpeg_bytes(pil_img, max_side=2800, quality=95):
    img = pil_img.convert("RGB")
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


def _tg_parse_json(text_value):
    s = str(text_value or "").strip()
    if s.startswith("```json"):
        s = s[7:].strip()
    elif s.startswith("```"):
        s = s[3:].strip()
    if s.endswith("```"):
        s = s[:-3].strip()
    try:
        return json.loads(s)
    except Exception:
        a = s.find("{")
        b = s.rfind("}")
        if a >= 0 and b > a:
            return json.loads(s[a:b + 1])
        raise


def _tg_extract_replacement_model(error_text):
    m = re.search(r"models/(gemini-[A-Za-z0-9._-]+)", str(error_text or ""))
    return m.group(1) if m else ""


def _tg_gemini_json(images, prompt, temperature=0.02, max_tokens=32768):
    key = _tg_gemini_api_key()
    if not key:
        raise RuntimeError("Falta GEMINI_API_KEY en Streamlit Secrets.")

    parts = [{"text": str(prompt)}]
    for img in images:
        raw = _tg_jpeg_bytes(img)
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(raw).decode("utf-8"),
            }
        })

    queue = [_tg_gemini_model()]
    tried = set()
    last_error = ""

    while queue:
        model_name = queue.pop(0)
        if not model_name or model_name in tried:
            continue
        tried.add(model_name)

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_name}:generateContent"
        )
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
            headers={
                "x-goog-api-key": key,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=300,
        )

        if response.status_code >= 400:
            last_error = f"Gemini HTTP {response.status_code}: {response.text[:1400]}"
            replacement = _tg_extract_replacement_model(response.text)
            if response.status_code == 404 and replacement and replacement not in tried:
                queue.append(replacement)
                continue
            raise RuntimeError(last_error)

        body = response.json()
        candidates = body.get("candidates") or []
        if not candidates:
            raise RuntimeError(
                "Gemini no devolvió candidatos: "
                + json.dumps(body.get("promptFeedback") or {}, ensure_ascii=False)[:900]
            )

        rparts = (candidates[0].get("content") or {}).get("parts") or []
        texts = [
            str(p.get("text", ""))
            for p in rparts
            if isinstance(p, dict) and p.get("text")
        ]
        if not texts:
            raise RuntimeError("Gemini no devolvió texto JSON.")

        data = _tg_parse_json("\n".join(texts))
        if not isinstance(data, dict):
            raise RuntimeError("Gemini no devolvió un objeto JSON.")

        st.session_state.tc_last_ai_provider = f"Gemini ({model_name})"
        return data

    raise RuntimeError(last_error or "Gemini no respondió.")


# Compatibilidad: cualquier función antigua del flujo nuevo que todavía llame
# _tc_openai_json queda redirigida a Gemini y NUNCA llama OpenAI.
def _tc_openai_json(images, prompt, detail="high", model=None, effort="medium"):
    return _tg_gemini_json(images, prompt, temperature=0.02)


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


def _tg_sanitize_rows(data):
    raw = data.get("rows", []) if isinstance(data, dict) else []
    clean = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        pts = []
        for p in item.get("points", item.get("points_norm", [])) or []:
            q = _tg_norm_point(p)
            if q is not None:
                pts.append(q)
        if len(pts) < 2:
            continue
        # Limitar ruido en geometría devuelta por el modelo.
        if len(pts) > 14:
            step = (len(pts) - 1) / 11.0
            pts = [pts[int(round(i * step))] for i in range(12)]
        try:
            confidence = _tg_clamp(item.get("confidence", 0.6), 0.0, 1.0)
        except Exception:
            confidence = 0.6
        clean.append({
            "id": len(clean) + 1,
            "confidence": confidence,
            "points_norm": pts,
            "slot_count": 0,
            "vacant_indices": [],
        })
    return _tg_sort_rows(clean)


def _tg_draw_guide(pil, rows, labels=True, slot_ticks=False):
    img = pil.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    font = ImageFont.load_default()
    line_w = max(1, int(round(min(w, h) / 900)))

    for row in rows:
        rid = int(row.get("id", 0) or 0)
        pts = _tg_norm_to_px(row.get("points_norm", []), w, h)
        if len(pts) < 2:
            continue
        xy = [(int(round(x)), int(round(y))) for x, y in pts]
        draw.line(xy, fill=(255, 0, 255), width=line_w)
        if labels:
            mid = _tg_point_on_polyline(pts, 0.5)
            draw.text(
                (int(mid[0]) + 2, int(mid[1]) - 10),
                f"R{rid:02d}",
                fill=(255, 255, 255),
                font=font,
                stroke_width=2,
                stroke_fill=(30, 10, 20),
            )
        if slot_ticks:
            count = int(row.get("slot_count", 0) or 0)
            if count > 1:
                for i in range(count):
                    p = _tg_point_on_polyline(pts, i / max(1, count - 1))
                    x, y = int(round(p[0])), int(round(p[1]))
                    draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(255, 255, 0))
    return img


def _tg_crop_bbox(pil, rows, margin_factor=1.35):
    w, h = pil.size
    pts = []
    for r in rows:
        pts.extend(_tg_norm_to_px(r.get("points_norm", []), w, h))
    if not pts:
        return (0, 0, w, h)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    ww = max(20.0, x1 - x0)
    hh = max(20.0, y1 - y0)
    mx = ww * max(0.12, (margin_factor - 1.0) / 2.0) + 20
    my = hh * 0.06 + 15
    return (
        int(_tg_clamp(x0 - mx, 0, w - 1)),
        int(_tg_clamp(y0 - my, 0, h - 1)),
        int(_tg_clamp(x1 + mx, 1, w)),
        int(_tg_clamp(y1 + my, 1, h)),
    )


def _tg_detect_rows(uploaded_file):
    pil = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")

    prompt_first = """
TERRACORE — DETECCIÓN EXACTA DE HILERAS FÍSICAS DE VID.
Analiza únicamente la fotografía aérea ORIGINAL.

OBJETIVO:
Encontrar TODAS las hileras físicas reales y devolver UNA SOLA línea central por hilera.

REGLAS CRÍTICAS:
- Una franja/copa lineal de vid puede tener dos bordes visibles: NO cuentes esos bordes como dos hileras.
- Cada hilera física debe aparecer EXACTAMENTE UNA VEZ, por su EJE CENTRAL.
- Nunca dibujes en el pasillo/suelo entre dos hileras.
- Sigue la MISMA hilera desde su inicio visible hasta su final visible, incluso a través de huecos secos.
- No unas dos hileras vecinas y no cambies de hilera a mitad de camino.
- Excluye caminos, cabeceras, árboles, postes, sombras, bordes y vegetación ajena.
- Incluye una hilera seca o débil solo si su eje físico es reconocible por la regularidad de la parcela.
- Usa la separación repetitiva entre hileras como control de calidad: no debe haber líneas duplicadas entre dos hileras reales.
- Revisa especialmente la primera y última hilera de cada bloque.
- Usa 6 a 12 puntos por trayectoria, coordenadas [x,y] normalizadas 0..1000 de la imagen COMPLETA.
- Ordena las hileras físicamente de un lado al otro de la parcela.

Devuelve SOLO JSON válido:
{
  "estimated_row_count":0,
  "coverage_score":0.0,
  "confidence":0.0,
  "rows":[{"id":1,"confidence":0.0,"points":[[x,y],[x,y],[x,y],[x,y],[x,y],[x,y]]}],
  "notes":""
}
"""
    first = _tg_gemini_json([pil], prompt_first, temperature=0.015)
    rows_first = _tg_sanitize_rows(first)
    if not rows_first:
        raise RuntimeError("Gemini no devolvió hileras utilizables.")

    # Lectura independiente solo del número de hileras para reducir sobreconteo
    # por bordes de vegetación.
    prompt_count = """
TERRACORE — CONTEO INDEPENDIENTE DE HILERAS.
Mira la fotografía ORIGINAL sin ninguna guía.
Cuenta HILERAS FÍSICAS de vid, no bordes de copa ni pasillos.
Una hilera = una banda longitudinal plantada, aunque tenga huecos o zonas secas.
No cuentes dos veces una hilera por sus dos costados visibles.
Revisa de un extremo al otro y devuelve SOLO JSON:
{"physical_row_count":0,"confidence":0.0,"notes":""}
"""
    count_data = _tg_gemini_json([pil], prompt_count, temperature=0.0)
    try:
        count_check = int(count_data.get("physical_row_count", 0) or 0)
    except Exception:
        count_check = 0

    guide = _tg_draw_guide(pil, rows_first, labels=True, slot_ticks=False)
    proposal = [
        {"id": int(r["id"]), "points": r["points_norm"]}
        for r in rows_first
    ]

    prompt_final = """
TERRACORE — AUDITORÍA FINAL DE HILERAS.
Imagen 1 = fotografía ORIGINAL.
Imagen 2 = propuesta magenta de una primera lectura de Gemini.

Primera propuesta: __PROPOSAL__
Conteo independiente de hileras físicas: __COUNT__

Devuelve la geometría FINAL. Inspecciona visualmente TODA la parcela antes de responder.
- Elimina cualquier línea duplicada sobre la misma hilera.
- Si la propuesta trazó los dos bordes de una misma franja de vegetación, reemplázalos por UNA línea CENTRAL.
- Agrega hileras reales omitidas.
- Mueve líneas que estén en pasillos/suelo al centro de la hilera real correspondiente.
- No cruces ni saltes a la hilera vecina.
- Sigue una hilera a través de huecos secos si el patrón físico continúa.
- No marques caminos, cabeceras, árboles, sombras, bordes ni vegetación ajena.
- Usa la separación regular como control de calidad.
- El conteo independiente es una referencia fuerte, pero corrígelo si la fotografía demuestra otra cantidad.
- Cada hilera física final debe aparecer EXACTAMENTE UNA VEZ.
- 6 a 12 puntos por hilera; coordenadas 0..1000 de la imagen COMPLETA.

Devuelve SOLO JSON válido:
{
 "estimated_row_count":0,
 "coverage_score":0.0,
 "confidence":0.0,
 "rows":[{"id":1,"confidence":0.0,"points":[[x,y],[x,y],[x,y],[x,y],[x,y],[x,y]]}],
 "audit_notes":""
}
""".replace(
        "__PROPOSAL__",
        json.dumps(proposal, ensure_ascii=False, separators=(",", ":")),
    ).replace("__COUNT__", str(count_check or "no disponible"))

    final = _tg_gemini_json([pil, guide], prompt_final, temperature=0.005)
    rows_final = _tg_sanitize_rows(final) or rows_first

    return {
        "pil": pil,
        "rows": _tg_sort_rows(rows_final),
        "confidence": _tg_clamp(final.get("confidence", first.get("confidence", 0.6)), 0.0, 1.0),
        "coverage_score": _tg_clamp(final.get("coverage_score", first.get("coverage_score", 0.6)), 0.0, 1.0),
        "count_check": count_check,
        "notes": str(final.get("audit_notes", first.get("notes", "")) or ""),
    }


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


def _tg_inventory_slots(pil, rows, batch_size=6):
    final = [dict(r) for r in rows]
    by_id = {int(r["id"]): r for r in final}

    for start in range(0, len(final), batch_size):
        group = final[start:start + batch_size]
        bbox = _tg_crop_bbox(pil, group, 1.45)
        original = pil.crop(bbox).convert("RGB")
        guide_full = _tg_draw_guide(pil, group, labels=True, slot_ticks=False)
        guide = guide_full.crop(bbox).convert("RGB")
        ids = [int(r["id"]) for r in group]

        prompt = f"""
TERRACORE INVENTARIO DE PLANTAS — SOLO GEMINI.
Imagen 1 = recorte ORIGINAL.
Imagen 2 = las mismas hileras marcadas Rxx.
Analiza SOLO estos IDs: {ids}.

Para cada Rxx:
1. sigue únicamente esa hilera, sin saltar a una vecina;
2. determina el patrón repetitivo de posiciones reales de planta;
3. slot_count = TOTAL de posiciones esperadas, incluyendo faltantes;
4. vacant_indices = índices 1-based de posiciones CLARAMENTE sin planta;
5. una planta seca, amarilla, pequeña o débil sigue siendo OCUPADA si físicamente existe;
6. no cuentes postes, sombras, maleza, hojas sueltas ni textura del suelo;
7. usa hileras vecinas solo para validar la distancia repetitiva;
8. no cambies geometría ni IDs.

Devuelve EXACTAMENTE todos los IDs. SOLO JSON válido:
{{"rows":[{{"id":1,"slot_count":52,"vacant_indices":[7,19],"confidence":0.0}}]}}
"""
        data = _tg_gemini_json([original, guide], prompt, temperature=0.01)
        for item in data.get("rows", []) if isinstance(data, dict) else []:
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
            row["vacant_indices"] = _tg_clean_indices(item.get("vacant_indices", []), count)
            row["slot_confidence"] = _tg_clamp(item.get("confidence", 0.6), 0.0, 1.0)

        # Auditoría del mismo lote; mejora conteos sin redibujar surcos.
        prelim = [
            {
                "id": int(r["id"]),
                "slot_count": int(r.get("slot_count", 0) or 0),
                "vacant_indices": r.get("vacant_indices", []),
            }
            for r in group
        ]
        prompt_audit = f"""
TERRACORE — AUDITORÍA DE SLOTS DEL MISMO LOTE.
Imagen 1 ORIGINAL, imagen 2 guía Rxx.
Conteo preliminar: {json.dumps(prelim, ensure_ascii=False, separators=(',', ':'))}

Revisa SOLO slot_count y vacant_indices.
Corrige un conteo únicamente si el patrón repetitivo visible lo demuestra.
Una planta físicamente presente, aunque seca/débil, NO es vacante.
No cambies geometría ni IDs. Devuelve todos los IDs.
SOLO JSON:
{{"rows":[{{"id":1,"slot_count":52,"vacant_indices":[7,19],"confidence":0.0}}]}}
"""
        audit = _tg_gemini_json([original, guide], prompt_audit, temperature=0.005)
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
            row = by_id[rid]
            old = int(row.get("slot_count", 0) or 0)
            # Protección simple contra saltos absurdos en una segunda lectura.
            if old > 4:
                ratio = count / float(old)
                if ratio < 0.65 or ratio > 1.55:
                    continue
            row["slot_count"] = count
            row["vacant_indices"] = _tg_clean_indices(item.get("vacant_indices", []), count)
            row["slot_confidence"] = max(
                float(row.get("slot_confidence", 0.0) or 0.0),
                _tg_clamp(item.get("confidence", 0.6), 0.0, 1.0),
            )

    for r in final:
        if int(r.get("slot_count", 0) or 0) < 2:
            r["needs_review"] = True
            r["review_reason"] = f"R{int(r['id']):02d}: Gemini no pudo validar los slots."
        else:
            r["needs_review"] = False
    return final


def _tg_draw_inventory(pil, rows):
    base = pil.convert("RGB")
    w, h = base.size
    band_h = max(42, int(round(h * 0.06)))
    canvas = Image.new("RGB", (w, h + band_h), (91, 37, 46))
    canvas.paste(base, (0, band_h))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    line_w = max(1, int(round(min(w, h) / 1000)))

    table = []
    confs = []

    for row in rows:
        rid = int(row.get("id", 0) or 0)
        pts = _tg_norm_to_px(row.get("points_norm", []), w, h)
        if len(pts) < 2:
            continue
        shifted = [(int(round(x)), int(round(y + band_h))) for x, y in pts]
        draw.line(shifted, fill=(245, 245, 245), width=line_w)

        # Número SOLO ARRIBA. Se alterna dentro de la franja superior, nunca abajo.
        top = min(pts, key=lambda p: p[1])
        x = int(round(top[0]))
        label = f"{rid:02d}"
        y = 4 if rid % 2 else 19
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = max(10, bbox[2] - bbox[0])
        tx = int(_tg_clamp(x - tw / 2, 1, max(1, w - tw - 2)))
        draw.text(
            (tx, y),
            label,
            fill=(255, 255, 255),
            font=font,
            stroke_width=2,
            stroke_fill=(36, 18, 25),
        )

        count = int(row.get("slot_count", 0) or 0)
        vacant = _tg_clean_indices(row.get("vacant_indices", []), count)
        occupied = max(0, count - len(vacant))
        conf = float(row.get("slot_confidence", row.get("confidence", 0.0)) or 0.0)
        confs.append(conf)
        status = tr("Validado por Gemini", "Validé par Gemini")
        if row.get("needs_review"):
            status = tr("Revisar", "À vérifier")
        table.append({
            tr("Surco", "Rang"): label,
            tr("Slots", "Emplacements"): count,
            tr("Ocupados", "Occupés"): occupied,
            tr("Vacíos", "Vides"): len(vacant),
            tr("Confianza", "Confiance"): round(conf * 100.0, 1),
            tr("Estado", "État"): status,
        })

    return canvas, pd.DataFrame(table), (sum(confs) / len(confs) if confs else 0.0)


def _tg_analyze_inventory(uploaded_file, base=None):
    base = base or _tg_detect_rows(uploaded_file)
    rows = _tg_inventory_slots(base["pil"], base["rows"], batch_size=6)
    image, table, slot_conf = _tg_draw_inventory(base["pil"], rows)
    if table.empty:
        raise RuntimeError("Gemini no pudo construir la tabla de Inventario.")
    warnings = [
        str(r.get("review_reason"))
        for r in rows
        if r.get("needs_review") and r.get("review_reason")
    ]
    return {
        "pil": base["pil"],
        "rows": rows,
        "image": image,
        "table": table,
        "count": len(rows),
        "confidence": (float(base.get("confidence", 0.0)) + float(slot_conf)) / 2.0,
        "coverage_score": float(base.get("coverage_score", 0.0)),
        "model": st.session_state.get("tc_last_ai_provider", f"Gemini ({_tg_gemini_model()})"),
        "warnings": warnings,
        "debug": {
            "conteo_independiente": base.get("count_check", 0),
            "notas": base.get("notes", ""),
        },
    }


def _tc_select_best_capture_openai(uploaded_images):
    """Nombre conservado por compatibilidad; internamente usa SOLO Gemini."""
    candidates = []
    errors = []
    for up in uploaded_images:
        try:
            base = _tg_detect_rows(up)
            count = len(base.get("rows", []))
            score = (
                float(base.get("coverage_score", 0.0)) * 0.50
                + float(base.get("confidence", 0.0)) * 0.35
                + min(1.0, count / 100.0) * 0.15
            )
            candidates.append((score, up, base))
        except Exception as exc:
            errors.append(f"{up.name}: {exc}")
    if not candidates:
        raise RuntimeError(" | ".join(errors) if errors else "Gemini no pudo analizar la captura.")
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, up, base = candidates[0]
    return up, _tg_analyze_inventory(up, base=base), errors


def _tg_health_slots(pil, rows, batch_size=5):
    result = {}
    valid = [r for r in rows if int(r.get("slot_count", 0) or 0) > 1]

    for start in range(0, len(valid), batch_size):
        group = valid[start:start + batch_size]
        bbox = _tg_crop_bbox(pil, group, 1.55)
        original = pil.crop(bbox).convert("RGB")
        guide_full = _tg_draw_guide(pil, group, labels=True, slot_ticks=True)
        guide = guide_full.crop(bbox).convert("RGB")
        spec = [
            {
                "id": int(r["id"]),
                "slot_count": int(r["slot_count"]),
                "vacant_indices": _tg_clean_indices(r.get("vacant_indices", []), int(r["slot_count"])),
            }
            for r in group
        ]

        prompt = f"""
TERRACORE SALUD POR SLOT — SOLO GEMINI.
Imagen 1 = recorte ORIGINAL.
Imagen 2 = guía Rxx con pequeñas referencias de posiciones de planta.
Inventario confirmado: {json.dumps(spec, ensure_ascii=False, separators=(',', ':'))}

Para CADA Rxx devuelve los índices 1-based que deben ser ROJOS.
El resto serán VERDES.

CRITERIO:
- VERDE: vid claramente viva, con cobertura y vigor suficiente comparada con plantas sanas cercanas bajo iluminación similar.
- ROJO: slot vacío, planta seca, marrón/beige, amarillenta severa, muy débil, muy rala, interrumpida o con cobertura claramente inferior al patrón sano local.
- Todo vacant_indices DEBE permanecer rojo.
- Una sombra por sí sola NO vuelve rojo un slot.
- No confundas maleza entre hileras con copa de vid.
- No cambies slot_count, geometría ni IDs.
- Si la resolución no permite asegurar un caso, baja confidence en vez de inventar precisión.

Devuelve todos los IDs. SOLO JSON válido:
{{"rows":[{{"id":1,"red_indices":[7,8,19],"confidence":0.0}}]}}
"""
        data = _tg_gemini_json([original, guide], prompt, temperature=0.01)
        by = {}
        for item in data.get("rows", []) if isinstance(data, dict) else []:
            if not isinstance(item, dict):
                continue
            try:
                rid = int(item.get("id"))
            except Exception:
                continue
            row = next((r for r in group if int(r["id"]) == rid), None)
            if row is None:
                continue
            count = int(row["slot_count"])
            red = set(_tg_clean_indices(item.get("red_indices", []), count))
            red.update(_tg_clean_indices(row.get("vacant_indices", []), count))
            by[rid] = {
                "red_indices": sorted(red),
                "confidence": _tg_clamp(item.get("confidence", 0.6), 0.0, 1.0),
            }

        for r in group:
            rid = int(r["id"])
            result[rid] = by.get(rid, {
                "red_indices": _tg_clean_indices(r.get("vacant_indices", []), int(r["slot_count"])),
                "confidence": 0.35,
            })

        # Segunda auditoría del mismo lote para corregir falsos verdes/falsos rojos.
        prelim = [
            {
                "id": int(r["id"]),
                "slot_count": int(r["slot_count"]),
                "vacant_indices": _tg_clean_indices(r.get("vacant_indices", []), int(r["slot_count"])),
                "initial_red_indices": result.get(int(r["id"]), {}).get("red_indices", []),
            }
            for r in group
        ]
        prompt_audit = f"""
TERRACORE SALUD — AUDITORÍA FINAL DEL LOTE.
Imagen 1 ORIGINAL, imagen 2 guía Rxx. Clasificación preliminar:
{json.dumps(prelim, ensure_ascii=False, separators=(',', ':'))}

Revisa especialmente falsos verdes y falsos rojos.
- Vacíos físicos permanecen rojos.
- Verde requiere evidencia visible suficiente de vid viva.
- Sombra sola no significa rojo.
- No cambies slot_count, geometría ni IDs.
Devuelve SOLO JSON:
{{"rows":[{{"id":1,"red_indices":[7,8,19],"confidence":0.0}}]}}
"""
        audit = _tg_gemini_json([original, guide], prompt_audit, temperature=0.005)
        for item in audit.get("rows", []) if isinstance(audit, dict) else []:
            if not isinstance(item, dict):
                continue
            try:
                rid = int(item.get("id"))
            except Exception:
                continue
            row = next((r for r in group if int(r["id"]) == rid), None)
            if row is None:
                continue
            count = int(row["slot_count"])
            red = set(_tg_clean_indices(item.get("red_indices", []), count))
            red.update(_tg_clean_indices(row.get("vacant_indices", []), count))
            result[rid] = {
                "red_indices": sorted(red),
                "confidence": _tg_clamp(
                    item.get("confidence", result.get(rid, {}).get("confidence", 0.6)),
                    0.0,
                    1.0,
                ),
            }

    return result


def _tg_draw_health(pil, rows, health_map):
    img = pil.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    font = ImageFont.load_default()
    green = (25, 225, 55)
    red = (245, 45, 45)
    width = max(2, int(round(min(w, h) / 520)))

    total_slots = 0
    total_red = 0
    total_green = 0
    red_xy = []
    confs = []

    for row in rows:
        rid = int(row.get("id", 0) or 0)
        count = int(row.get("slot_count", 0) or 0)
        pts = _tg_norm_to_px(row.get("points_norm", []), w, h)
        if len(pts) < 2 or count < 2:
            continue

        red_indices = set(health_map.get(rid, {}).get("red_indices", []))
        red_indices.update(_tg_clean_indices(row.get("vacant_indices", []), count))
        red_indices = {i for i in red_indices if 1 <= i <= count}
        confs.append(float(health_map.get(rid, {}).get("confidence", 0.0) or 0.0))

        positions = [
            _tg_point_on_polyline(pts, i / max(1, count - 1))
            for i in range(count)
        ]
        total_slots += count
        total_red += len(red_indices)
        total_green += max(0, count - len(red_indices))

        # La línea entre dos slots es roja si cualquiera de los extremos es rojo.
        for i in range(count - 1):
            idx1, idx2 = i + 1, i + 2
            p0, p1 = positions[i], positions[i + 1]
            is_red = idx1 in red_indices or idx2 in red_indices
            draw.line(
                (int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1])),
                fill=red if is_red else green,
                width=width,
            )

        for idx in red_indices:
            red_xy.append(positions[idx - 1])

        # Número SOLO ARRIBA.
        top = min(pts, key=lambda p: p[1])
        label = f"{rid:02d}"
        draw.text(
            (int(top[0]) - 6, max(0, int(top[1]) - 13)),
            label,
            fill=(255, 255, 255),
            font=font,
            stroke_width=2,
            stroke_fill=(35, 20, 25),
        )

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


def _tg_zone_fallback(pil, red_xy):
    if not red_xy:
        return "No determinada"
    w, h = pil.size
    x = sum(p[0] for p in red_xy) / len(red_xy)
    y = sum(p[1] for p in red_xy) / len(red_xy)
    horiz = "izquierda" if x < w / 3 else "centro" if x < 2 * w / 3 else "derecha"
    vert = "superior" if y < h / 3 else "media" if y < 2 * h / 3 else "inferior"
    return f"{horiz}, zona {vert}"


def _tg_health_diagnosis(pil, visual, row_count):
    prompt = f"""
TERRACORE — DIAGNÓSTICO VISUAL PRELIMINAR, SOLO GEMINI.
Analiza esta fotografía aérea junto con el resumen calculado por slots:
- surcos: {row_count}
- slots totales: {visual['total_slots']}
- slots verdes: {visual['green_slots']} ({visual['green_pct']:.1f}%)
- slots rojos: {visual['red_slots']} ({visual['red_pct']:.1f}%)

No afirmes una enfermedad ni un nutriente específico solo por fotografía.
Describe la zona más afectada de forma espacial.
Devuelve SOLO JSON válido:
{{
 "zona_mas_afectada":"texto corto",
 "nivel_afectacion_visual":"bajo|medio|alto",
 "diagnostico_visual":"1 a 3 frases",
 "causas_probables":["..."],
 "explicacion_nutrientes":"La fotografía por sí sola no confirma una deficiencia nutricional específica; confirmar con revisión de campo, suelo y/o análisis foliar.",
 "recomendaciones_iniciales":["..."],
 "nota_diagnostico":"Diagnóstico visual preliminar; confirmar en campo."
}}
"""
    try:
        return _tg_gemini_json([pil], prompt, temperature=0.02)
    except Exception:
        return {}


def _tc_analyze_health_openai(uploaded_image, rows):
    """Nombre conservado por compatibilidad; internamente usa SOLO Gemini."""
    pil = Image.open(io.BytesIO(uploaded_image.getvalue())).convert("RGB")
    if not rows:
        raise RuntimeError("No hay geometría de Inventario confirmada.")

    health_map = _tg_health_slots(pil, rows, batch_size=5)
    visual = _tg_draw_health(pil, rows, health_map)
    diag = _tg_health_diagnosis(pil, visual, len(rows))

    zona = str(diag.get("zona_mas_afectada", "") or "").strip()
    if not zona:
        zona = _tg_zone_fallback(pil, visual.get("red_xy", []))

    explicacion_default = (
        "La fotografía por sí sola no confirma una deficiencia nutricional específica; "
        "confirma con revisión de campo, análisis de suelo y, de ser posible, análisis foliar."
    )

    result = {
        "count": len(rows),
        "green_pct": float(visual["green_pct"]),
        "red_pct": float(visual["red_pct"]),
        "green_slots": int(visual["green_slots"]),
        "red_slots": int(visual["red_slots"]),
        "total_slots": int(visual["total_slots"]),
        "angle": 0.0,
        "annotated": visual["annotated"],
        "result_url": "",
        "zona_mas_afectada": zona or "No determinada",
        "nivel_afectacion_visual": str(diag.get("nivel_afectacion_visual", "No determinado") or "No determinado"),
        "diagnostico_visual": str(diag.get("diagnostico_visual", "") or ""),
        "causas_probables": diag.get("causas_probables", []) or [],
        "explicacion_nutrientes": str(diag.get("explicacion_nutrientes", explicacion_default) or explicacion_default),
        "recomendaciones_iniciales": diag.get("recomendaciones_iniciales", []) or [],
        "nota_diagnostico": str(diag.get("nota_diagnostico", "Diagnóstico visual preliminar; confirmar en campo.") or "Diagnóstico visual preliminar; confirmar en campo."),
        "detalle_zonas": {},
        "metodo": "gemini-only-slot-health",
        "confidence": float(visual.get("confidence", 0.0)),
        "health_map": health_map,
    }
    result["backend"] = {
        "metodo": "gemini-only-slot-health",
        "analisis": {
            "surcos_estimados": result["count"],
            "verde_pct": result["green_pct"],
            "rojo_pct": result["red_pct"],
            "zona_mas_afectada": result["zona_mas_afectada"],
            "nivel_afectacion_visual": result["nivel_afectacion_visual"],
            "diagnostico_visual": result["diagnostico_visual"],
            "causas_probables": result["causas_probables"],
            "explicacion_nutrientes": result["explicacion_nutrientes"],
            "recomendaciones_iniciales": result["recomendaciones_iniciales"],
            "nota_diagnostico": result["nota_diagnostico"],
            "total_slots": result["total_slots"],
            "green_slots": result["green_slots"],
            "red_slots": result["red_slots"],
        },
    }
    return result


# ============================================================
# HISTORIAL + PARCELAS - GUARDADO AUTOMÁTICO
# ============================================================
# - Conserva el historial existente.
# - Renombra las fotos con Parcela + fecha + hora + ID corto.
# - Guarda la parcela seleccionada en la columna "Parcelas"
#   de HistorialTerroCore.
# - Guarda ID + nombre de imagen en la columna correspondiente
#   de la hoja "Parcelas" (Parcela 1, Parcela 2, Parcela 10, etc.).
# - Llena automáticamente rutas/URLs para AppSheet.
# ============================================================

def _tg_normalizar_parcela(valor):
    """Normaliza '1' -> 'Parcela 1' y 'parcela 010' -> 'Parcela 10'."""
    raw = str(valor or "").strip()
    if not raw:
        return "Parcela"

    if re.fullmatch(r"\d+", raw):
        return f"Parcela {int(raw)}"

    m = re.fullmatch(r"(?i)parcela\s*0*(\d+)", raw)
    if m:
        return f"Parcela {int(m.group(1))}"

    return raw


def _tg_nombre_seguro(valor):
    """Nombre apto para Drive/AppSheet: letras, números y guion bajo."""
    import unicodedata

    txt = unicodedata.normalize("NFKD", str(valor or ""))
    txt = txt.encode("ascii", "ignore").decode("ascii")
    txt = re.sub(r"[^A-Za-z0-9]+", "_", txt).strip("_")
    return txt or "Parcela"


def _tg_letra_columna(numero):
    """1 -> A, 2 -> B, 27 -> AA."""
    numero = int(numero)
    salida = ""
    while numero > 0:
        numero, resto = divmod(numero - 1, 26)
        salida = chr(65 + resto) + salida
    return salida


def _tg_buscar_o_crear_encabezado(sheets_service, spreadsheet_id, hoja, encabezado):
    """
    Busca un encabezado en fila 1 (sin distinguir mayúsculas).
    Si no existe, lo agrega al final y devuelve el número de columna (1-based).
    """
    valores = sheets_service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range=f"'{hoja}'!1:1",
    ).execute().get("values", [])

    headers = list(valores[0]) if valores else []
    objetivo = str(encabezado or "").strip().casefold()

    for i, item in enumerate(headers, 1):
        if str(item or "").strip().casefold() == objetivo:
            return i

    nueva_col = len(headers) + 1
    letra = _tg_letra_columna(nueva_col)

    sheets_service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=f"'{hoja}'!{letra}1",
        valueInputOption="RAW",
        body={"values": [[encabezado]]},
    ).execute()

    return nueva_col


def _tg_buscar_fila_historial_por_id(sheets_service, spreadsheet_id, registro_id):
    """Devuelve la fila de HistorialTerroCore donde está el ID."""
    ids = sheets_service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range=f"'{HISTORIAL_SHEET_NAME}'!A2:A",
    ).execute().get("values", [])

    objetivo = str(registro_id or "").strip()
    for offset, fila in enumerate(ids, 2):
        valor = str(fila[0] if fila else "").strip()
        if valor == objetivo:
            return offset

    return None


def _tg_guardar_extras_historial(registro):
    """
    Guarda automáticamente columnas adicionales en HistorialTerroCore:
    Parcelas, URLs de Drive y rutas relativas para AppSheet.
    """
    _, sheets_service = obtener_google_clients()
    spreadsheet_id = _secret_text("GSHEET_ID")

    fila = _tg_buscar_fila_historial_por_id(
        sheets_service,
        spreadsheet_id,
        registro.get("id", ""),
    )
    if not fila:
        return

    extras = {
        "Parcelas": registro.get("parcela", ""),
        "ImagenOriginalURL": registro.get("imagen_original_url", ""),
        "ImagenProcesadaURL": registro.get("imagen_procesada_url", ""),
        "ImagenOriginalAppSheet": registro.get("imagen_original_appsheet", ""),
        "ImagenProcesadaAppSheet": registro.get("imagen_procesada_appsheet", ""),

        # SOLO IDs nuevos. Los links de AppSheet se generan con fórmulas
        # directamente en Google Sheets y NO se escriben desde Python.
        "ImagenInventarioFileID": registro.get("imagen_inventario_file_id", ""),
        "ImagenPoligonosFileID": registro.get("imagen_poligonos_file_id", ""),
    }

    data = []
    for encabezado, valor in extras.items():
        col = _tg_buscar_o_crear_encabezado(
            sheets_service,
            spreadsheet_id,
            HISTORIAL_SHEET_NAME,
            encabezado,
        )
        letra = _tg_letra_columna(col)
        data.append({
            "range": f"'{HISTORIAL_SHEET_NAME}'!{letra}{fila}",
            "values": [[valor]],
        })

    if data:
        sheets_service.spreadsheets().values().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={
                "valueInputOption": "RAW",
                "data": data,
            },
        ).execute()


def _tg_guardar_en_hoja_parcelas(registro):
    """
    En la hoja 'Parcelas':
      A = ID del análisis
      columna 'Parcela N' = nombre amigable de la imagen

    Cada análisis ocupa una fila nueva. Solo se llena la columna
    correspondiente a la parcela seleccionada.
    """
    _, sheets_service = obtener_google_clients()
    spreadsheet_id = _secret_text("GSHEET_ID")
    hoja = "Parcelas"

    # Asegurar que la hoja exista.
    meta = sheets_service.spreadsheets().get(
        spreadsheetId=spreadsheet_id,
        fields="sheets.properties",
    ).execute()

    nombres = [
        s.get("properties", {}).get("title", "")
        for s in meta.get("sheets", [])
    ]

    if hoja not in nombres:
        sheets_service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={
                "requests": [{
                    "addSheet": {
                        "properties": {"title": hoja}
                    }
                }]
            },
        ).execute()

    # Asegurar ID en A1.
    a1 = sheets_service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range=f"'{hoja}'!A1",
    ).execute().get("values", [])

    if not a1 or not a1[0] or str(a1[0][0]).strip().casefold() != "id":
        sheets_service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=f"'{hoja}'!A1",
            valueInputOption="RAW",
            body={"values": [["ID"]]},
        ).execute()

    parcela = _tg_normalizar_parcela(registro.get("parcela", ""))
    col = _tg_buscar_o_crear_encabezado(
        sheets_service,
        spreadsheet_id,
        hoja,
        parcela,
    )
    letra_parcela = _tg_letra_columna(col)

    # Primera fila libre según la columna ID.
    ids = sheets_service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range=f"'{hoja}'!A2:A",
    ).execute().get("values", [])
    siguiente_fila = len(ids) + 2

    sheets_service.spreadsheets().values().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={
            "valueInputOption": "RAW",
            "data": [
                {
                    "range": f"'{hoja}'!A{siguiente_fila}",
                    "values": [[registro.get("id", "")]],
                },
                {
                    "range": f"'{hoja}'!{letra_parcela}{siguiente_fila}",
                    "values": [[registro.get("nombre", "")]],
                },
            ],
        },
    ).execute()


# El historial original se conserva; esta versión agrega nombre amigable,
# parcela, rutas para AppSheet y guardado automático en la hoja Parcelas.
def guardar_analisis_en_google(uploaded_image, backend_result):
    if not historial_google_configurado():
        return False, "Historial de Google no configurado."

    import uuid
    from datetime import datetime, timezone

    # --------------------------------------------------------
    # IDENTIDAD DEL ANÁLISIS
    # --------------------------------------------------------
    parcela = _tg_normalizar_parcela(
        st.session_state.get("tc_parcela_nombre", "")
    )

    ahora = datetime.now(timezone.utc)

    fecha_captura = st.session_state.get("tc_fecha_captura", None)
    try:
        fecha_texto = fecha_captura.strftime("%Y%m%d")
    except Exception:
        fecha_texto = ahora.strftime("%Y%m%d")

    hora_texto = ahora.strftime("%H%M%S")
    corto = uuid.uuid4().hex[:6].upper()

    # ID solo letras y números.
    registro_id = f"TC{fecha_texto}{hora_texto}{corto}"

    parcela_archivo = _tg_nombre_seguro(parcela)
    base_nombre = f"{parcela_archivo}_{fecha_texto}_{hora_texto}_{corto}"

    extension_original = Path(uploaded_image.name).suffix.lower()
    if extension_original not in {".jpg", ".jpeg", ".png"}:
        mime = str(uploaded_image.type or "").lower()
        extension_original = ".png" if "png" in mime else ".jpg"

    nombre_original = f"{base_nombre}{extension_original}"
    nombre_procesada = f"{base_nombre}_procesada.png"
    nombre_inventario = f"{base_nombre}_inventario.png"
    nombre_poligonos = f"{base_nombre}_poligonos.png"

    # --------------------------------------------------------
    # GUARDAR ORIGINAL CON NOMBRE NUEVO
    # --------------------------------------------------------
    original_file_id = subir_bytes_google_drive(
        uploaded_image.getvalue(),
        nombre_original,
        uploaded_image.type or "image/jpeg",
        "Originales",
    )

    # --------------------------------------------------------
    # GUARDAR PROCESADA CON EL MISMO NOMBRE BASE
    # --------------------------------------------------------
    annotated = backend_result.get("annotated")
    if annotated is None:
        raise RuntimeError("No existe imagen procesada para guardar.")

    if isinstance(annotated, Image.Image):
        buf = io.BytesIO()
        annotated.convert("RGB").save(buf, format="PNG", optimize=True)
        processed_bytes = buf.getvalue()
    else:
        # Compatibilidad con resultados históricos/legados.
        try:
            pil_tmp = Image.fromarray(np.asarray(annotated))
            buf = io.BytesIO()
            pil_tmp.convert("RGB").save(buf, format="PNG", optimize=True)
            processed_bytes = buf.getvalue()
        except Exception as exc:
            raise RuntimeError(
                f"No se pudo convertir la imagen procesada a PNG: {exc}"
            )

    processed_file_id = subir_bytes_google_drive(
        processed_bytes,
        nombre_procesada,
        "image/png",
        "Procesadas",
    )

    # --------------------------------------------------------
    # SUBIR INVENTARIO Y POLÍGONOS YA GENERADOS
    # --------------------------------------------------------
    def _tg_a_png_bytes(imagen):
        if imagen is None:
            return None

        if isinstance(imagen, Image.Image):
            _buf = io.BytesIO()
            imagen.convert("RGB").save(_buf, format="PNG", optimize=True)
            return _buf.getvalue()

        _arr = np.asarray(imagen)
        if _arr.size == 0:
            return None

        _ok, _enc = cv2.imencode(".png", _arr)
        if not _ok:
            return None

        return _enc.tobytes()

    inventario_file_id = ""
    poligonos_file_id = ""

    # Inventario ya calculado y mostrado en la app.
    _inventario_img = st.session_state.get("tc_inventario_imagen")
    _inventario_bytes = _tg_a_png_bytes(_inventario_img)

    if _inventario_bytes:
        inventario_file_id = subir_bytes_google_drive(
            _inventario_bytes,
            nombre_inventario,
            "image/png",
            "Inventarios",
        )

    # Polígonos ya calculados por la lógica actual.
    _poligonos_img = backend_result.get("polygon_image")
    _poligonos_bytes = _tg_a_png_bytes(_poligonos_img)

    if _poligonos_bytes:
        poligonos_file_id = subir_bytes_google_drive(
            _poligonos_bytes,
            nombre_poligonos,
            "image/png",
            "Poligonos",
        )

    # --------------------------------------------------------
    # REGISTRO COMPLETO
    # --------------------------------------------------------
    backend_data = backend_result.get("backend") or {}
    analisis = (
        backend_data.get("analisis", {})
        if isinstance(backend_data, dict)
        else {}
    ) or {}

    registro = {
        "id": registro_id,
        "fecha": ahora.isoformat(),

        # Nombre que verá Historial/AppSheet. Ya NO usa "WhatsApp Image...".
        "nombre": nombre_original,
        "nombre_procesada": nombre_procesada,
        "parcela": parcela,

        "imagen_original_file_id": original_file_id,
        "imagen_procesada_file_id": processed_file_id,

        # IDs que se guardan en Q y R del HistorialTerroCore.
        "imagen_inventario_file_id": inventario_file_id,
        "imagen_poligonos_file_id": poligonos_file_id,

        "imagen_original_url":
            f"https://drive.google.com/file/d/{original_file_id}/view",
        "imagen_procesada_url":
            f"https://drive.google.com/file/d/{processed_file_id}/view",

        # Rutas relativas listas para columnas Image de AppSheet.
        "imagen_original_appsheet": f"Originales/{nombre_original}",
        "imagen_procesada_appsheet": f"Procesadas/{nombre_procesada}",

        "surcos": int(backend_result.get("count", 0) or 0),
        "verde_pct": float(backend_result.get("green_pct", 0.0) or 0.0),
        "rojo_pct": float(backend_result.get("red_pct", 0.0) or 0.0),
        "amarillo_pct": float(analisis.get("amarillo_pct", 0.0) or 0.0),
        "nivel_visual": str(
            backend_result.get(
                "nivel_afectacion_visual",
                "No determinado",
            ) or "No determinado"
        ),
        "zona_mas_afectada": str(
            backend_result.get(
                "zona_mas_afectada",
                "No determinada",
            ) or "No determinada"
        ),
        "diagnostico_visual": str(
            backend_result.get("diagnostico_visual", "") or ""
        ),
        "causas_probables":
            backend_result.get("causas_probables", []) or [],
        "explicacion_nutrientes": str(
            backend_result.get("explicacion_nutrientes", "") or ""
        ),
        "recomendaciones":
            backend_result.get("recomendaciones_iniciales", []) or [],
        "nota_diagnostico": str(
            backend_result.get("nota_diagnostico", "") or ""
        ),
    }

    # 1) HistorialTerroCore A:P existente.
    guardar_registro_google_sheets(registro)

    # 2) Columnas extra del mismo renglón:
    #    Parcelas + URLs + rutas AppSheet.
    _tg_guardar_extras_historial(registro)

    # 3) Hoja Parcelas: ID + nombre bajo la columna Parcela N correcta.
    _tg_guardar_en_hoja_parcelas(registro)

    return True, registro

# Gemini reemplaza también la ayuda visual de IA usada por el flujo legado/video.
# Devuelve el mismo esquema de exclusión para no romper el resto de la aplicación.
def analizar_pil_con_ia(pil_img):
    prompt = """
Analiza esta imagen agrícola desde arriba y distingue el viñedo de zonas donde NO deben trazarse surcos.
Devuelve SOLO JSON válido:
{
 "es_vinedo":true,
 "hay_camino":false,
 "hay_techo":false,
 "hay_construccion":false,
 "analizar_surcos":true,
 "resumen":"texto breve",
 "zonas_excluir":[{"tipo":"camino","puntos":[[x,y],[x,y],[x,y],[x,y]]}]
}
Coordenadas 0..1000. Excluye caminos, techos, edificios, árboles grandes y superficies ajenas.
NO excluyas huecos secos dentro de una hilera ni suelo normal entre hileras.
"""
    try:
        data = _tg_gemini_json([pil_img.convert("RGB")], prompt, temperature=0.02)
        if "zonas_excluir" not in data or not isinstance(data.get("zonas_excluir"), list):
            data["zonas_excluir"] = []
        return True, data
    except Exception as exc:
        return False, str(exc)


# ============================================================
# TERRACORE - MOTOR ACTIVO OPENCV V2
# ============================================================
# Este bloque NO elimina ninguna función existente. Solo redefine el motor
# activo de Inventario/Salud para que trabaje localmente con OpenCV/NumPy.
# Objetivos:
# - líneas RECTAS y centradas en cada surco;
# - sin saltos de una hilera a otra;
# - slots/puntos sobre el mismo eje recto;
# - polígonos rojos únicamente en plantas secas/débiles y vacíos;
# - porcentajes por slots, no por longitud de línea;
# - cero llamadas a Gemini/OpenAI durante Inventario/Salud.
# ============================================================


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


def _ocv2_inventory_from_file(uploaded_file):
    pil = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
    w, h = pil.size
    base = _ocv2_detect_straight_rows(pil)
    rows = [dict(r) for r in base["rows"]]
    response_map = np.asarray(base["response_map"], dtype=np.float32)

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
        table.append({
            tr("Surco","Rang"): label,
            tr("Slots","Emplacements"): count,
            tr("Ocupados","Occupés"): occ,
            tr("Vacíos","Vides"): len(vacant),
            tr("Confianza","Confiance"): round(float(r.get("slot_confidence",0.0))*100.0, 1),
            tr("Estado","État"): tr("OpenCV local","OpenCV local"),
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


def _ocv2_select_best_capture(uploaded_images):
    candidates = []
    errors = []
    for up in uploaded_images:
        try:
            inv = _ocv2_inventory_from_file(up)
            score = float(inv.get("confidence", 0.0))*0.50 + min(1.0, float(inv.get("count", 0))/100.0)*0.50
            candidates.append((score, up, inv))
        except Exception as exc:
            errors.append(f"{up.name}: {exc}")
    if not candidates:
        raise RuntimeError(" | ".join(errors) if errors else "OpenCV no pudo analizar la captura.")
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, up, inv = candidates[0]
    return up, inv, errors


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
        f"OpenCV evaluó {visual['total_slots']} posiciones sobre {len(rows)} surcos rectos. "
        f"Detectó {visual['red_slots']} posiciones secas, débiles o vacías ({red_pct:.1f}% rojo). "
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
        "diagnostico_visual": diagnostico,
        "causas_probables": [],
        "explicacion_nutrientes": "El análisis OpenCV es visual y no identifica nutrientes ni enfermedades por sí solo.",
        "recomendaciones_iniciales": [
            "Revisar en campo los polígonos rojos para distinguir planta seca de faltante real.",
            "Comparar la misma parcela en capturas posteriores para confirmar si la afectación persiste.",
        ],
        "nota_diagnostico": "Diagnóstico visual automático por OpenCV; confirmar en campo.",
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


# Alias de compatibilidad: la interfaz conserva los mismos nombres, pero ya NO llama IA.
def _tc_select_best_capture_openai(uploaded_images):
    return _ocv2_select_best_capture(uploaded_images)


def _tc_analyze_health_openai(uploaded_image, rows):
    return _ocv2_analyze_health(uploaded_image, rows)


def analizar_pil_con_ia(pil_img):
    """Compatibilidad con el flujo legado/video: no llama APIs externas."""
    return True, {
        "es_vinedo": True,
        "hay_camino": False,
        "hay_techo": False,
        "hay_construccion": False,
        "analizar_surcos": True,
        "resumen": "Modo local OpenCV activo.",
        "zonas_excluir": [],
    }


# ------------------------------------------------------------
# ESTILOS ADICIONALES: SOLO COMPLEMENTAN EL DISEÑO ORIGINAL
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .tc-flow-wrap{
        display:grid;
        grid-template-columns:repeat(5,1fr);
        gap:8px;
        margin:.35rem 0 1rem 0;
    }
    .tc-flow-step{
        border:1px solid rgba(255,255,255,.26);
        background:rgba(73,20,29,.18);
        border-radius:10px;
        padding:10px 7px;
        text-align:center;
        font-size:.84rem;
        font-weight:800;
        color:#FFF;
    }
    .tc-flow-step.active{
        background:#FFFDFC;
        color:#722F37;
        border-color:#F2D5D8;
    }
    .tc-flow-step.locked{
        opacity:.45;
    }
    .tc-parcela-title{
        font-size:1.03rem;
        font-weight:800;
        color:#FFF;
        margin-bottom:.15rem;
    }
    .tc-small-note{
        color:#F5DADD;
        font-size:.86rem;
    }
    .tc-row-number-demo{
        border:1px dashed rgba(255,255,255,.55);
        border-radius:10px;
        padding:12px;
        text-align:center;
        color:#FFF;
        font-weight:900;
        letter-spacing:.12em;
        margin:.35rem 0 .75rem 0;
        background:rgba(79,30,38,.25);
    }
    @media (max-width:900px){
        .tc-flow-wrap{grid-template-columns:1fr 1fr;}
    }
    /* Móvil: los controles de parcela/captura aparecen primero porque side_col
       es ahora la primera columna del DOM. También reducimos ruido del toolbar. */
    @media (max-width:700px){
        .tc-flow-wrap{grid-template-columns:1fr 1fr !important;gap:6px !important;}
        .tc-flow-step{padding:9px 5px !important;font-size:.78rem !important;letter-spacing:.03em !important;}
        [data-testid="stToolbar"]{display:none !important;}
        .block-container{padding-left:.65rem !important;padding-right:.65rem !important;}
        h1{font-size:2rem !important;}
        h2{font-size:1.55rem !important;}
        h3{font-size:1.25rem !important;}
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FLUJO VISUAL DEL VIDEO - SOLO INTERFAZ / PRESENTACIÓN
# ============================================================
# IMPORTANTE:
# - La lógica de líneas rectas NO se modifica.
# - La lógica de puntos/slots NO se modifica.
# - La lógica de polígonos azules NO se modifica.
# - El diagnóstico verde/rojo y porcentajes NO se modifican.
# - El guardado en Drive/Sheets NO se modifica.
# - El historial y descarga de imágenes NO se modifican.
# ============================================================

st.markdown(
    """
    <style>
    .tc-video-flow{
        display:grid;
        grid-template-columns:repeat(7, minmax(115px,1fr));
        gap:9px;
        margin: .75rem 0 1rem 0;
    }
    .tc-video-flow-item{
        min-height:44px;
        display:flex;
        align-items:center;
        justify-content:center;
        text-align:center;
        padding:8px 8px;
        border-radius:10px;
        border:1px solid rgba(255,255,255,.22);
        background:#FFFDFC;
        color:#722F37;
        font-weight:850;
        font-size:.82rem;
        line-height:1.15;
        box-shadow:0 2px 7px rgba(0,0,0,.10);
    }
    .tc-video-flow-item.locked{
        background:rgba(78,30,38,.18);
        color:rgba(255,255,255,.42);
        border-color:rgba(255,255,255,.12);
        box-shadow:none;
    }
    .tc-stage-title{
        font-size:1.65rem;
        font-weight:900;
        margin-top:.35rem;
        margin-bottom:.35rem;
        color:#FFF;
    }
    .tc-stage-sub{
        font-size:.92rem;
        color:#F4DDE0;
        margin-bottom:.75rem;
    }
    .tc-soft-card{
        border:1px solid rgba(255,255,255,.18);
        border-radius:12px;
        padding:14px;
        background:rgba(80,25,35,.18);
        margin:.6rem 0;
    }
    .tc-four-grid{
        display:grid;
        grid-template-columns:repeat(4,1fr);
        gap:12px;
    }
    .tc-label-mini{
        font-weight:800;
        margin-bottom:5px;
        color:#F8E8EA;
    }
    @media(max-width:1100px){
        .tc-video-flow{grid-template-columns:repeat(4,1fr);}
    }
    @media(max-width:700px){
        .tc-video-flow{grid-template-columns:1fr 1fr;}
        .tc-video-flow-item{font-size:.76rem; min-height:42px;}
    }
    </style>
    """,
    unsafe_allow_html=True
)


def _tc_ui_image(img, caption=None):
    """Muestra PIL o arreglo OpenCV sin alterar la imagen."""
    if img is None:
        return
    if isinstance(img, Image.Image):
        st.image(img, caption=caption, use_container_width=True)
        return
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[2] == 3:
        st.image(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB), caption=caption, use_container_width=True)
    else:
        st.image(arr, caption=caption, use_container_width=True)


def _tc_uploaded_pil(up):
    return Image.open(io.BytesIO(up.getvalue())).convert("RGB")


# ------------------------------------------------------------
# NAVEGACIÓN VISUAL DEL VIDEO
# ------------------------------------------------------------
st.markdown(
    f"""
    <div class="tc-video-flow">
      <div class="tc-video-flow-item">① {tr('Captura Base','Capture de base')}</div>
      <div class="tc-video-flow-item">② {tr('Inventario visual','Inventaire visuel')}</div>
      <div class="tc-video-flow-item">③ {tr('Validación QR','Validation QR')}</div>
      <div class="tc-video-flow-item">④ {tr('Regiones críticas','Régions critiques')}</div>
      <div class="tc-video-flow-item locked">⑤ {tr('Segundo vuelo','Deuxième vol')}</div>
      <div class="tc-video-flow-item locked">⑥ {tr('Análisis nivel 2','Analyse niveau 2')}</div>
      <div class="tc-video-flow-item">⑦ {tr('Historial','Historique')}</div>
    </div>
    """,
    unsafe_allow_html=True
)

side_col, main_col = st.columns([1.0, 2.15], gap="medium")

# ============================================================
# PANEL LATERAL - MISMO FLUJO DEL VIDEO
# ============================================================
with side_col:
    idioma_es_col, idioma_fr_col = st.columns(2, gap="medium")

    with idioma_es_col:
        if st.button(
            "🇪🇸 ES Español",
            key="lang_es_video_flow",
            use_container_width=True,
            disabled=st.session_state.idioma_terrocore == "ES"
        ):
            st.session_state.idioma_terrocore = "ES"
            st.rerun()

    with idioma_fr_col:
        if st.button(
            "🇫🇷 FR Français",
            key="lang_fr_video_flow",
            use_container_width=True,
            disabled=st.session_state.idioma_terrocore == "FR"
        ):
            st.session_state.idioma_terrocore = "FR"
            st.rerun()

    with st.container(border=True):
        st.markdown(tr("### 1. Parcela", "### 1. Parcelle"))

        parcela_nombre = st.text_input(
            tr("Nombre de la parcela", "Nom de la parcelle"),
            value=st.session_state.tc_parcela_nombre,
            placeholder=tr("Ej. Parcela 01", "Ex. Parcelle 01"),
            key="tc_parcela_input_video_flow"
        )
        st.session_state.tc_parcela_nombre = parcela_nombre

        fecha_captura = st.date_input(
            tr("Fecha de captura", "Date de capture"),
            value=st.session_state.tc_fecha_captura,
            key="tc_fecha_input_video_flow"
        )
        st.session_state.tc_fecha_captura = fecha_captura

    with st.container(border=True):
        st.markdown(tr("### 2. Captura Base", "### 2. Capture de base"))

        uploaded_images = st.file_uploader(
            tr(
                "Selecciona una o varias fotografías de la misma parcela",
                "Sélectionnez une ou plusieurs photos de la même parcelle"
            ),
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="tc_uploader_captura_base_video_flow"
        )

        misma_parcela = st.checkbox(
            tr(
                "Confirmo que todas las imágenes pertenecen a la misma parcela",
                "Je confirme que toutes les images appartiennent à la même parcelle"
            ),
            key="tc_misma_parcela_video_flow"
        )

        altura_vuelo = st.number_input(
            tr("Altura del vuelo base (m) — opcional", "Hauteur du vol de base (m) — facultatif"),
            min_value=0.0,
            value=0.0,
            step=1.0,
            key="tc_altura_video_flow"
        )

        if uploaded_images:
            st.caption(
                tr(
                    f"Imágenes seleccionadas: {len(uploaded_images)}",
                    f"Images sélectionnées : {len(uploaded_images)}"
                )
            )

        crear_captura = st.button(
            tr("📷 Crear captura base", "📷 Créer la capture de base"),
            type="primary",
            use_container_width=True,
            disabled=(not uploaded_images or not misma_parcela or not parcela_nombre.strip()),
            key="tc_crear_captura_video_flow"
        )

        if crear_captura:
            st.session_state.tc_captura_confirmada = True
            st.session_state.tc_inventario_procesado = False
            st.session_state.tc_inventario_confirmado = False
            st.session_state.tc_resultados_base = []
            st.session_state.tc_tabla_inventario = None
            st.session_state.tc_inventario_imagen = None
            st.session_state.tc_inventario_fuente = ""
            st.session_state.tc_inventario_confianza = 0.0
            st.session_state.tc_inventario_rows_ai = []
            st.session_state.tc_inventario_modelo = ""
            st.session_state.tc_inventario_debug = {}
            st.session_state.tc_inventarios_por_imagen = {}
            st.session_state.tc_salud_procesada = False
            st.success(
                tr(
                    "✅ Captura base creada. Ya puedes continuar con Inventario visual.",
                    "✅ Capture de base créée. Vous pouvez continuer avec l’inventaire visuel."
                )
            )

    with st.container(border=True):
        st.markdown(tr("### Estado de la parcela", "### État de la parcelle"))
        st.write(f"**{tr('Parcela','Parcelle')}:** {parcela_nombre or '—'}")
        st.write(f"**{tr('Fecha','Date')}:** {fecha_captura}")
        st.write(f"**{tr('Imágenes','Images')}:** {len(uploaded_images or [])}")
        st.write(
            f"**{tr('Estado','État')}:** " +
            (
                tr("Captura base lista", "Capture de base prête")
                if st.session_state.tc_captura_confirmada
                else tr("Pendiente", "En attente")
            )
        )

    if st.button(
        tr("🔄 Nueva parcela / Nuevo análisis", "🔄 Nouvelle parcelle / Nouvelle analyse"),
        use_container_width=True,
        key="tc_reiniciar_video_flow"
    ):
        _tc_reiniciar_parcela()
        st.rerun()


# ============================================================
# PANEL PRINCIPAL - FLUJO DEL VIDEO
# ============================================================
with main_col:

    # ========================================================
    # ① CAPTURA BASE
    # ========================================================
    st.markdown(
        f"<div class='tc-stage-title'>① {tr('Captura Base de la Parcela','Capture de base de la parcelle')}</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='tc-stage-sub'>{tr('La Captura Base es la referencia común para dos vertientes independientes: Inventario visual y Regiones críticas.','La capture de base est la référence commune pour deux volets indépendants : inventaire visuel et régions critiques.')}</div>",
        unsafe_allow_html=True
    )

    with st.container(border=True):
        st.markdown(tr("### Imagen Maestra de Parcela", "### Image maîtresse de la parcelle"))

        if uploaded_images:
            master_up = uploaded_images[0]
            if st.session_state.tc_inventario_fuente:
                master_up = next(
                    (u for u in uploaded_images if u.name == st.session_state.tc_inventario_fuente),
                    uploaded_images[0]
                )
            try:
                _tc_ui_image(
                    _tc_uploaded_pil(master_up),
                    caption=tr("Referencia principal", "Référence principale")
                )
            except Exception as exc:
                st.warning(str(exc))

            st.info(
                tr(
                    f"Se cargaron {len(uploaded_images)} fotografía(s). La Imagen Maestra funciona como referencia común sin modificar el análisis técnico.",
                    f"{len(uploaded_images)} photo(s) chargée(s). L’image maîtresse sert de référence commune sans modifier l’analyse technique."
                )
            )

            st.markdown(tr("### Fotografías fuente", "### Photographies source"))
            for idx, up in enumerate(uploaded_images, 1):
                with st.expander(f"{idx}. {up.name}", expanded=(idx == 1)):
                    try:
                        _tc_ui_image(_tc_uploaded_pil(up))
                    except Exception as exc:
                        st.warning(f"{up.name}: {exc}")
        else:
            st.info(
                tr(
                    "Carga las fotografías desde el panel de Captura Base.",
                    "Chargez les photographies depuis le panneau Capture de base."
                )
            )

    st.markdown("---")

    # ========================================================
    # ② INVENTARIO VISUAL
    # ========================================================
    st.markdown(
        f"<div class='tc-stage-title'>② {tr('Vertiente 1 · Inventario visual de la parcela','Volet 1 · Inventaire visuel de la parcelle')}</div>",
        unsafe_allow_html=True
    )
    st.caption(
        tr(
            "Detecta los surcos con la lógica actual, mantiene líneas rectas, puntos/slots y la numeración del Inventario.",
            "Détecte les rangs avec la logique actuelle, en conservant les lignes droites, les points/emplacements et la numérotation."
        )
    )

    analizar_inventario = st.button(
        tr("🌿 Analizar Inventario", "🌿 Analyser l’inventaire"),
        type="primary",
        use_container_width=True,
        disabled=(not st.session_state.tc_captura_confirmada or not uploaded_images),
        key="tc_analizar_inventario_video_flow"
    )

    if analizar_inventario and uploaded_images:
        progress = st.progress(
            5,
            text=tr(
                "OpenCV está detectando los surcos rectos y alineando los slots...",
                "OpenCV détecte les rangs droits et aligne les emplacements..."
            )
        )

        try:
            # Procesar TODAS las fotografías con el MISMO motor OpenCV.
            # La mejor captura se mantiene únicamente como referencia principal
            # para la tabla de validación, pero cada imagen conserva sus propios
            # surcos, slots e imagen de Inventario.
            inventarios_por_imagen = {}
            candidatos = []
            errores_inventario = []

            total_imgs = max(1, len(uploaded_images))

            for idx_img, up in enumerate(uploaded_images, 1):
                try:
                    inv_img = _ocv2_inventory_from_file(up)
                    inventarios_por_imagen[up.name] = inv_img

                    score = (
                        float(inv_img.get("confidence", 0.0)) * 0.50
                        + min(1.0, float(inv_img.get("count", 0)) / 100.0) * 0.50
                    )
                    candidatos.append((score, up, inv_img))

                    pct = 5 + int((idx_img / total_imgs) * 80)
                    progress.progress(
                        min(85, pct),
                        text=tr(
                            f"Inventario {idx_img}/{total_imgs}: {up.name}",
                            f"Inventaire {idx_img}/{total_imgs} : {up.name}"
                        )
                    )
                except Exception as exc_img:
                    errores_inventario.append(f"{up.name}: {exc_img}")

            if not candidatos:
                raise RuntimeError(
                    " | ".join(errores_inventario)
                    if errores_inventario
                    else "OpenCV no pudo analizar las fotografías."
                )

            candidatos.sort(key=lambda x: x[0], reverse=True)
            _, best_up, inv = candidatos[0]

            progress.progress(
                92,
                text=tr(
                    "OpenCV está terminando los inventarios...",
                    "OpenCV termine les inventaires..."
                )
            )

            st.session_state.tc_resultados_base = []
            st.session_state.tc_salud_procesada = False
            st.session_state.tc_inventario_confirmado = False

            # Referencia principal para la tabla editable/validación.
            st.session_state.tc_tabla_inventario = inv["table"]
            st.session_state.tc_inventario_imagen = inv["image"]
            st.session_state.tc_inventario_fuente = best_up.name
            st.session_state.tc_inventario_confianza = float(inv.get("confidence", 0.0))
            st.session_state.tc_inventario_rows_ai = inv.get("rows", [])
            st.session_state.tc_inventario_modelo = inv.get("model", "OpenCV local")
            st.session_state.tc_inventario_debug = inv.get("debug", {})
            st.session_state.tc_inventario_warnings = inv.get("warnings", [])

            # Mapa completo: una geometría/inventario independiente por foto.
            st.session_state.tc_inventarios_por_imagen = inventarios_por_imagen
            st.session_state.tc_inventario_procesado = True

            progress.progress(100, text=tr("Inventarios terminados.", "Inventaires terminés."))
            st.success(
                tr(
                    f"✅ Se analizaron {len(inventarios_por_imagen)} fotografía(s) con OpenCV.",
                    f"✅ {len(inventarios_por_imagen)} photographie(s) analysée(s) avec OpenCV."
                )
            )

            if errores_inventario:
                with st.expander(
                    tr("Fotografías que no pudieron procesarse", "Photographies non traitées"),
                    expanded=False
                ):
                    for msg in errores_inventario:
                        st.caption(msg)

        except Exception as exc:
            st.session_state.tc_inventario_procesado = False
            st.session_state.tc_tabla_inventario = None
            st.session_state.tc_inventario_imagen = None
            st.session_state.tc_inventario_rows_ai = []
            st.session_state.tc_inventario_warnings = []
            st.error(
                tr(
                    f"No se pudo terminar el Inventario con OpenCV: {exc}",
                    f"Impossible de terminer l’inventaire avec OpenCV : {exc}"
                )
            )

    if st.session_state.tc_inventario_procesado:
        tabla_actual = st.session_state.tc_tabla_inventario
        if tabla_actual is None:
            tabla_actual = _tc_resultado_a_fila_surcos(0)

        total_surcos = int(len(tabla_actual))
        total_slots, total_ocupados, total_vacios, inventario_valido = _tc_metricas_tabla(
            tabla_actual.copy()
        )

        st.markdown(tr("## Resultado de Inventario", "## Résultat de l’inventaire"))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(tr("Surcos", "Rangs"), total_surcos)
        m2.metric(tr("Slots totales", "Emplacements totaux"), total_slots)
        m3.metric(tr("Ocupados", "Occupés"), total_ocupados)
        m4.metric(tr("Vacíos", "Vides"), total_vacios)

        st.caption(
            tr(
                f"Imagen de referencia: {st.session_state.tc_inventario_fuente or '—'}. Confianza media: {float(st.session_state.tc_inventario_confianza or 0.0)*100:.1f}%.",
                f"Image de référence : {st.session_state.tc_inventario_fuente or '—'}. Confiance moyenne : {float(st.session_state.tc_inventario_confianza or 0.0)*100:.1f} %."
            )
        )

        st.markdown(tr("### Inventario por cada fotografía", "### Inventaire par photographie"))

        if uploaded_images:
            ref_name = st.session_state.tc_inventario_fuente or uploaded_images[0].name
            inv_map = st.session_state.get("tc_inventarios_por_imagen", {}) or {}

            for idx, up in enumerate(uploaded_images, 1):
                with st.expander(f"{idx}. {up.name}", expanded=(up.name == ref_name)):
                    inv_individual = inv_map.get(up.name)

                    if inv_individual:
                        i1, i2, i3 = st.columns(3)
                        i1.metric(
                            tr("Surcos", "Rangs"),
                            int(inv_individual.get("count", 0) or 0)
                        )
                        tabla_ind = inv_individual.get("table")
                        slots_ind = 0
                        if tabla_ind is not None:
                            try:
                                slots_ind = int(
                                    pd.to_numeric(tabla_ind["Slots"], errors="coerce")
                                    .fillna(0).sum()
                                )
                            except Exception:
                                slots_ind = 0
                        i2.metric(tr("Slots", "Emplacements"), slots_ind)
                        i3.metric(
                            tr("Confianza", "Confiance"),
                            f"{float(inv_individual.get('confidence', 0.0))*100:.1f}%"
                        )

                        _tc_ui_image(
                            inv_individual.get("image"),
                            caption=(
                                tr("Inventario de referencia", "Inventaire de référence")
                                if up.name == ref_name
                                else tr("Inventario independiente de esta fotografía", "Inventaire indépendant de cette photographie")
                            )
                        )
                    else:
                        st.warning(
                            tr(
                                "Esta fotografía no pudo generar Inventario.",
                                "Cette photographie n’a pas pu générer d’inventaire."
                            )
                        )

        st.markdown(tr("### Tabla automática por surco", "### Tableau automatique par rang"))
        st.caption(
            tr(
                "OpenCV estima Slots, Ocupados y Vacíos. Puedes corregir un valor antes de confirmar.",
                "OpenCV estime Emplacements, Occupés et Vides. Vous pouvez corriger une valeur avant de confirmer."
            )
        )

        edited = st.data_editor(
            tabla_actual,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            key="tc_editor_inventario_video_flow",
            column_config={
                "Surco": st.column_config.TextColumn("Surco", disabled=True, width="small"),
                "Slots": st.column_config.NumberColumn("Slots", min_value=0, step=1, format="%d"),
                "Ocupados": st.column_config.NumberColumn("Ocupados", min_value=0, step=1, format="%d"),
                "Vacíos": st.column_config.NumberColumn("Vacíos", min_value=0, step=1, format="%d"),
                "Confianza": st.column_config.NumberColumn(
                    "Confianza", min_value=0.0, max_value=100.0, format="%.1f %%", disabled=True
                ),
                "Estado": st.column_config.TextColumn("Estado", disabled=True, width="medium"),
            }
        )

        st.session_state.tc_tabla_inventario = edited
        total_slots, total_ocupados, total_vacios, inventario_valido = _tc_metricas_tabla(
            edited.copy()
        )

        if inventario_valido:
            st.success("✅ Validación correcta: Slots = Ocupados + Vacíos en todos los surcos.")
        else:
            st.error("Revisa la tabla: cada surco debe cumplir Slots = Ocupados + Vacíos.")

        # ====================================================
        # ③ VALIDACIÓN QR - MISMA CONFIRMACIÓN QUE YA EXISTÍA
        # ====================================================
        st.markdown(
            f"<div class='tc-stage-title'>③ {tr('Validación QR','Validation QR')}</div>",
            unsafe_allow_html=True
        )
        st.caption(
            tr(
                "La validación confirma el Inventario antes de desbloquear el estado visual y los polígonos.",
                "La validation confirme l’inventaire avant de déverrouiller l’état visuel et les polygones."
            )
        )

        if st.button(
            tr("➕ Agregar validación", "➕ Ajouter la validation"),
            type="primary",
            use_container_width=True,
            disabled=(not inventario_valido),
            key="tc_confirmar_inventario_video_flow"
        ):
            st.session_state.tc_inventario_confirmado = True
            st.success(
                tr(
                    "✅ Inventario validado. Se desbloqueó Regiones críticas.",
                    "✅ Inventaire validé. Les régions critiques sont déverrouillées."
                )
            )
            st.rerun()

        if st.session_state.tc_inventario_confirmado:
            st.success(
                tr(
                    "✅ Validación registrada.",
                    "✅ Validation enregistrée."
                )
            )

    st.markdown("---")

    # ========================================================
    # ④ REGIONES CRÍTICAS / ESTADO VISUAL
    # ========================================================
    st.markdown(
        f"<div class='tc-stage-title'>④ {tr('Vertiente 2 · Estado visual de la parcela','Volet 2 · État visuel de la parcelle')}</div>",
        unsafe_allow_html=True
    )

    if not st.session_state.tc_inventario_confirmado:
        st.info(
            tr(
                "🔒 Primero completa y valida el Inventario para ejecutar esta vertiente.",
                "🔒 Complétez et validez d’abord l’inventaire pour exécuter ce volet."
            )
        )
    else:
        if not st.session_state.tc_salud_procesada:
            analizar_salud = st.button(
                tr("🩺 Analizar estado visual", "🩺 Analyser l’état visuel"),
                type="primary",
                use_container_width=True,
                disabled=not uploaded_images,
                key="tc_analizar_salud_video_flow"
            )

            if analizar_salud and uploaded_images:
                resultados_salud = []
                progress_salud = st.progress(
                    5,
                    text=tr(
                        "OpenCV está revisando cada slot para detectar vegetación, seco y vacíos...",
                        "OpenCV examine chaque emplacement pour détecter végétation, sécheresse et vides..."
                    )
                )

                try:
                    inv_map = st.session_state.get("tc_inventarios_por_imagen", {}) or {}
                    total_imgs = max(1, len(uploaded_images))
                    errores_salud = []

                    # Guardar la referencia principal para restaurarla después.
                    inventario_principal = st.session_state.get("tc_inventario_imagen")

                    for idx_img, uploaded_image in enumerate(uploaded_images, 1):
                        try:
                            inv_individual = inv_map.get(uploaded_image.name)

                            # Cada fotografía usa SUS propios surcos.
                            if inv_individual:
                                rows_ai = inv_individual.get("rows", []) or []
                                inventario_img_actual = inv_individual.get("image")
                            else:
                                # Fallback para compatibilidad con sesiones anteriores.
                                rows_ai = st.session_state.tc_inventario_rows_ai or []
                                inventario_img_actual = inventario_principal

                            if not rows_ai:
                                raise RuntimeError(
                                    "No existe geometría de Inventario para esta fotografía."
                                )

                            progress_salud.progress(
                                max(5, min(80, int((idx_img - 1) / total_imgs * 80) + 5)),
                                text=tr(
                                    f"Analizando {idx_img}/{total_imgs}: {uploaded_image.name}",
                                    f"Analyse {idx_img}/{total_imgs} : {uploaded_image.name}"
                                )
                            )

                            # MISMO motor de líneas rectas / slots / verde-rojo / polígonos.
                            backend_result = _tc_analyze_health_openai(
                                uploaded_image,
                                rows_ai
                            )

                            # guardar_analisis_en_google toma el Inventario desde
                            # session_state. Cambiamos SOLO esa referencia durante
                            # el guardado para que cada registro suba SU inventario.
                            st.session_state.tc_inventario_imagen = inventario_img_actual

                            historial_google_ok = False
                            historial_google_info = ""
                            try:
                                historial_google_ok, historial_google_info = guardar_analisis_en_google(
                                    uploaded_image,
                                    backend_result
                                )
                            except Exception as historial_exc:
                                historial_google_info = str(historial_exc)

                            nombre_resultado = (
                                historial_google_info.get("nombre", uploaded_image.name)
                                if historial_google_ok and isinstance(historial_google_info, dict)
                                else uploaded_image.name
                            )
                            id_resultado = (
                                historial_google_info.get("id", f"{idx_img}_{uploaded_image.name}")
                                if historial_google_ok and isinstance(historial_google_info, dict)
                                else f"{idx_img}_{uploaded_image.name}"
                            )

                            resultados_salud.append({
                                "id": id_resultado,
                                "name": nombre_resultado,
                                "source_name": uploaded_image.name,
                                "inventory_image": inventario_img_actual,
                                "count": int(backend_result.get("count", 0)),
                                "green_pct": float(backend_result.get("green_pct", 0.0)),
                                "red_pct": float(backend_result.get("red_pct", 0.0)),
                                "green_slots": int(backend_result.get("green_slots", 0) or 0),
                                "red_slots": int(backend_result.get("red_slots", 0) or 0),
                                "total_slots": int(backend_result.get("total_slots", 0) or 0),
                                "angle": float(backend_result.get("angle", 0.0)),
                                "annotated": backend_result.get("annotated"),
                                "polygon_image": backend_result.get("polygon_image"),
                                "ia_scene": backend_result.get("backend"),
                                "result_url": backend_result.get("result_url"),
                                "historial_google_guardado": historial_google_ok,
                                "historial_google_info": historial_google_info,
                                "zona_mas_afectada": backend_result.get("zona_mas_afectada", "No determinada"),
                                "nivel_afectacion_visual": backend_result.get("nivel_afectacion_visual", "No determinado"),
                                "diagnostico_visual": backend_result.get("diagnostico_visual", ""),
                                "causas_probables": backend_result.get("causas_probables", []),
                                "explicacion_nutrientes": backend_result.get("explicacion_nutrientes", ""),
                                "recomendaciones_iniciales": backend_result.get("recomendaciones_iniciales", []),
                                "nota_diagnostico": backend_result.get("nota_diagnostico", ""),
                                "detalle_zonas": backend_result.get("detalle_zonas", {}),
                                "metodo": backend_result.get("metodo", "opencv-v2-straight-grid-polygons"),
                                "confidence": backend_result.get("confidence", 0.0),
                            })

                        except Exception as exc_img:
                            errores_salud.append(f"{uploaded_image.name}: {exc_img}")

                    # Restaurar referencia principal de la interfaz.
                    st.session_state.tc_inventario_imagen = inventario_principal

                    if not resultados_salud:
                        raise RuntimeError(
                            " | ".join(errores_salud)
                            if errores_salud
                            else "No se pudo analizar ninguna fotografía."
                        )

                    st.session_state.tc_resultados_base = resultados_salud
                    st.session_state.tc_salud_procesada = True

                    progress_salud.progress(
                        100,
                        text=tr(
                            f"Análisis terminado: {len(resultados_salud)} fotografía(s).",
                            f"Analyse terminée : {len(resultados_salud)} photographie(s)."
                        )
                    )

                    if errores_salud:
                        with st.expander(
                            tr("Fotografías con error", "Photographies en erreur"),
                            expanded=False
                        ):
                            for msg in errores_salud:
                                st.caption(msg)

                    st.rerun()

                except Exception as exc:
                    st.error(
                        tr(
                            f"No se pudo analizar con OpenCV: {exc}",
                            f"Impossible d’analyser avec OpenCV : {exc}"
                        )
                    )

        if st.session_state.tc_salud_procesada:
            resultados = st.session_state.tc_resultados_base or []

            if not resultados:
                st.info(tr("No hay resultados para mostrar.", "Aucun résultat à afficher."))
            else:
                st.markdown(tr("### Análisis por cada fotografía", "### Analyse par photographie"))

                for idx, item in enumerate(resultados, 1):
                    with st.expander(
                        f"{idx}. {item.get('name','')}",
                        expanded=True
                    ):
                        d1, d2, d3 = st.columns(3)
                        d1.metric(
                            tr("Vegetación verde", "Végétation verte"),
                            f"{float(item.get('green_pct',0.0)):.1f}%"
                        )
                        d2.metric(
                            tr("Afectación", "Affectation"),
                            f"{float(item.get('red_pct',0.0)):.1f}%"
                        )
                        d3.metric(
                            tr("Zona más afectada", "Zone la plus touchée"),
                            tr_diag_texto(item.get("zona_mas_afectada", "—"))
                        )

                        _tc_ui_image(
                            item.get("annotated"),
                            caption=tr("Análisis visual verde / rojo", "Analyse visuelle vert / rouge")
                        )

                        diagnostico = str(item.get("diagnostico_visual", "") or "").strip()
                        if diagnostico:
                            st.markdown(tr("#### Descripción visual preliminar", "#### Description visuelle préliminaire"))
                            st.write(tr_diag_texto(diagnostico))

                        recomendaciones = item.get("recomendaciones_iniciales", []) or []
                        if recomendaciones:
                            st.markdown(tr("#### Siguiente inspección sugerida", "#### Inspection suivante suggérée"))
                            for rec in recomendaciones:
                                st.markdown(f"- {tr_diag_texto(rec)}")

                # ----------------------------------------------------
                # POLÍGONOS - MISMA IMAGEN GENERADA POR LA LÓGICA ACTUAL
                # ----------------------------------------------------
                st.markdown(
                    tr(
                        "## Regiones críticas · Polígonos para seguimiento",
                        "## Régions critiques · Polygones de suivi"
                    )
                )
                st.caption(
                    tr(
                        "Los polígonos azules se muestran exactamente con la salida producida por la lógica actual.",
                        "Les polygones bleus sont affichés exactement avec la sortie produite par la logique actuelle."
                    )
                )

                for idx, item in enumerate(resultados, 1):
                    with st.expander(
                        f"{idx}. {item.get('name','')}",
                        expanded=True
                    ):
                        polygon_image = item.get("polygon_image")
                        if polygon_image is not None:
                            st.markdown(tr("### Imagen con polígonos azules", "### Image avec polygones bleus"))
                            _tc_ui_image(polygon_image)

                            p1, p2, p3 = st.columns(3)
                            red_pct_item = float(item.get("red_pct", 0.0) or 0.0)
                            p1.metric(
                                tr("Zonas detectadas", "Zones détectées"),
                                1 if red_pct_item > 0 else 0
                            )
                            p2.metric(
                                tr("Área visual diferente", "Zone visuelle différente"),
                                f"{red_pct_item:.1f}%"
                            )
                            p3.metric(
                                tr("Surcos analizados", "Rangs analysés"),
                                int(item.get("count", 0) or 0)
                            )

                            if item.get("historial_google_guardado"):
                                st.success(
                                    tr(
                                        "✅ El análisis fue guardado en Google Sheets y Google Drive.",
                                        "✅ L’analyse a été enregistrée dans Google Sheets et Google Drive."
                                    )
                                )
                        else:
                            st.info(
                                tr(
                                    "No hay una imagen de polígonos disponible en este resultado.",
                                    "Aucune image de polygones n’est disponible pour ce résultat."
                                )
                            )

                st.markdown("---")

                # ====================================================
                # ⑤ y ⑥ - SE CONSERVAN COMO ETAPAS VISUALES SIN
                # CAMBIAR LA LÓGICA ACTUAL.
                # ====================================================
                st.markdown(
                    f"<div class='tc-stage-title'>⑤ {tr('Segundo vuelo','Deuxième vol')}</div>",
                    unsafe_allow_html=True
                )
                st.info(
                    tr(
                        "Esta etapa queda preparada en el flujo. No se ejecuta un segundo vuelo automáticamente para no modificar la lógica actual.",
                        "Cette étape est préparée dans le flux. Aucun deuxième vol n’est exécuté automatiquement afin de ne pas modifier la logique actuelle."
                    )
                )

                st.markdown(
                    f"<div class='tc-stage-title'>⑥ {tr('Etapa 3 · Análisis visual de segundo nivel','Étape 3 · Analyse visuelle de deuxième niveau')}</div>",
                    unsafe_allow_html=True
                )
                st.caption(
                    tr(
                        "Primero registra el segundo vuelo de una región. Mientras tanto se conserva la vista final calculada con la lógica actual.",
                        "Enregistrez d’abord le deuxième vol d’une région. En attendant, la vue finale calculée avec la logique actuelle est conservée."
                    )
                )

                st.markdown(tr("## Vista final · Líneas sobre surcos", "## Vue finale · Lignes sur les rangs"))

                # Vista final para CADA fotografía analizada.
                for idx_item, item in enumerate(resultados, 1):
                    source_name = item.get("source_name", "")
                    up_ref = next(
                        (u for u in (uploaded_images or []) if u.name == source_name),
                        None
                    )

                    original_ref = None
                    if up_ref is not None:
                        try:
                            original_ref = _tc_uploaded_pil(up_ref)
                        except Exception:
                            original_ref = None

                    with st.container(border=True):
                        st.markdown(f"### {idx_item}. {source_name or item.get('name','')}")

                        f1, f2, f3 = st.columns(3)
                        f1.metric(
                            tr("Vegetación verde", "Végétation verte"),
                            f"{float(item.get('green_pct',0.0)):.1f}%"
                        )
                        f2.metric(
                            tr("Afectación roja", "Affectation rouge"),
                            f"{float(item.get('red_pct',0.0)):.1f}%"
                        )
                        f3.metric(
                            tr("Surcos usados", "Rangs utilisés"),
                            int(item.get("count",0) or 0)
                        )

                        vf1, vf2 = st.columns(2)
                        with vf1:
                            st.caption(tr("Imagen original", "Image originale"))
                            _tc_ui_image(original_ref)
                        with vf2:
                            st.caption(
                                tr(
                                    "Imagen procesada — Líneas guiadas por SU Inventario",
                                    "Image traitée — Lignes guidées par SON inventaire"
                                )
                            )
                            _tc_ui_image(item.get("annotated"))

                st.caption(
                    tr(
                        "Cada fotografía utiliza su propio Inventario, sus propios surcos y sus propios slots.",
                        "Chaque photographie utilise son propre inventaire, ses propres rangs et ses propres emplacements."
                    )
                )

                # ----------------------------------------------------
                # RESUMEN FINAL DE 4 IMÁGENES POR CADA FOTO
                # ----------------------------------------------------
                st.markdown("---")
                st.markdown(
                    tr(
                        "## Resumen final · 4 imágenes por fotografía",
                        "## Résumé final · 4 images par photographie"
                    )
                )
                st.caption(
                    tr(
                        "Se muestran: Original, Inventario, Análisis visual y Polígonos para cada fotografía cargada.",
                        "Affichage : Original, Inventaire, Analyse visuelle et Polygones pour chaque photographie chargée."
                    )
                )

                for idx_item, item in enumerate(resultados, 1):
                    source_name = item.get("source_name", "")
                    up_ref = next(
                        (u for u in (uploaded_images or []) if u.name == source_name),
                        None
                    )

                    original_ref = None
                    if up_ref is not None:
                        try:
                            original_ref = _tc_uploaded_pil(up_ref)
                        except Exception:
                            original_ref = None

                    with st.container(border=True):
                        st.markdown(f"### {idx_item}. {source_name or item.get('name','')}")
                        r1, r2, r3, r4 = st.columns(4)

                        with r1:
                            st.caption("1. Original")
                            _tc_ui_image(original_ref)

                        with r2:
                            st.caption("2. Inventario")
                            _tc_ui_image(item.get("inventory_image"))

                        with r3:
                            st.caption("3. Análisis")
                            _tc_ui_image(item.get("annotated"))

                        with r4:
                            st.caption("4. Polígonos")
                            _tc_ui_image(item.get("polygon_image"))


# ============================================================
# ⑦ HISTORIAL POR PARCELA
# ============================================================
st.markdown("---")
st.markdown(
    f"<div class='tc-stage-title'>⑦ 📂 {tr('Historial por parcela','Historique par parcelle')}</div>",
    unsafe_allow_html=True
)
st.caption(
    tr(
        "Consulta análisis anteriores y vuelve a abrir las imágenes guardadas sin cambiar la lógica de almacenamiento.",
        "Consultez les analyses précédentes et rouvrez les images enregistrées sans modifier la logique de stockage."
    )
)

if not historial_google_configurado():
    st.info(
        tr(
            "El historial de Google Drive aún no está configurado.",
            "L’historique Google Drive n’est pas encore configuré."
        )
    )
else:
    try:
        registros_historial = obtener_historial_google(limite=100)

        if not registros_historial:
            st.info(
                tr(
                    "Todavía no hay análisis guardados.",
                    "Aucune analyse enregistrée pour le moment."
                )
            )
        else:
            # Tabla conserva los mismos datos del historial.
            filas_historial = []
            for registro in registros_historial:
                filas_historial.append({
                    tr("Fecha", "Date"): registro.get("fecha", ""),
                    tr("Imagen", "Image"): registro.get("nombre", ""),
                    tr("Surcos", "Rangs"): registro.get("surcos", 0),
                    tr("Verde %", "Vert %"): registro.get("verde_pct", 0.0),
                    tr("Rojo %", "Rouge %"): registro.get("rojo_pct", 0.0),
                    tr("Zona más afectada", "Zone la plus touchée"):
                        tr_diag_texto(registro.get("zona_mas_afectada", "")),
                })

            st.dataframe(
                pd.DataFrame(filas_historial),
                use_container_width=True,
                hide_index=True
            )

            indice_historial = st.selectbox(
                tr(
                    "Selecciona un análisis para cargar sus imágenes",
                    "Sélectionnez une analyse pour charger ses images"
                ),
                options=list(range(len(registros_historial))),
                format_func=lambda i: (
                    f"{str(registros_historial[i].get('fecha',''))[:10]} | "
                    f"{registros_historial[i].get('nombre','')} | "
                    f"{registros_historial[i].get('surcos',0)} surcos"
                ),
                key="tc_historial_selector_video_flow"
            )

            registro_img = registros_historial[indice_historial]

            with st.container(border=True):
                h1, h2, h3 = st.columns(3)
                h1.metric(tr("Surcos", "Rangs"), registro_img.get("surcos", 0))
                h2.metric(
                    tr("Verde %", "Vert %"),
                    f"{float(registro_img.get('verde_pct',0.0) or 0.0):.1f}%"
                )
                h3.metric(
                    tr("Rojo %", "Rouge %"),
                    f"{float(registro_img.get('rojo_pct',0.0) or 0.0):.1f}%"
                )

                original_id = str(
                    registro_img.get("imagen_original_file_id", "") or ""
                ).strip()
                procesada_id = str(
                    registro_img.get("imagen_procesada_file_id", "") or ""
                ).strip()

                col_hist_original, col_hist_procesada = st.columns(2)

                with col_hist_original:
                    st.markdown("### 1. Original")
                    if original_id:
                        with st.spinner(
                            tr("Cargando imagen original...", "Chargement de l’image originale...")
                        ):
                            img_original_hist = _tc_historial_pil(original_id)

                        if img_original_hist is not None:
                            st.image(
                                img_original_hist,
                                caption=registro_img.get("nombre", ""),
                                use_container_width=True
                            )
                        else:
                            st.info(
                                tr(
                                    "No se pudo abrir la imagen original.",
                                    "Impossible d’ouvrir l’image originale."
                                )
                            )
                    else:
                        st.info(
                            tr(
                                "Este registro no tiene imagen original.",
                                "Cet enregistrement ne contient pas d’image originale."
                            )
                        )

                with col_hist_procesada:
                    st.markdown("### 2. Análisis")
                    if procesada_id:
                        with st.spinner(
                            tr("Cargando imagen procesada...", "Chargement de l’image traitée...")
                        ):
                            img_procesada_hist = _tc_historial_pil(procesada_id)

                        if img_procesada_hist is not None:
                            st.image(
                                img_procesada_hist,
                                caption=tr("Resultado del análisis", "Résultat de l’analyse"),
                                use_container_width=True
                            )
                        else:
                            st.info(
                                tr(
                                    "No se pudo abrir la imagen procesada.",
                                    "Impossible d’ouvrir l’image traitée."
                                )
                            )
                    else:
                        st.info(
                            tr(
                                "Este registro no tiene imagen procesada.",
                                "Cet enregistrement ne contient pas d’image traitée."
                            )
                        )

                diagnostico_guardado = str(
                    registro_img.get("diagnostico_visual", "") or ""
                ).strip()
                if diagnostico_guardado:
                    st.markdown(tr("## Diagnóstico guardado", "## Diagnostic enregistré"))
                    st.write(tr_diag_texto(diagnostico_guardado))

                recs_guardadas = registro_img.get("recomendaciones", []) or []
                if recs_guardadas:
                    st.markdown(tr("## Recomendaciones", "## Recommandations"))
                    for rec in recs_guardadas:
                        st.markdown(f"- {tr_diag_texto(rec)}")

    except Exception as exc:
        st.error(
            tr(
                f"No se pudo cargar el historial: {exc}",
                f"Impossible de charger l’historique : {exc}"
            )
        )
