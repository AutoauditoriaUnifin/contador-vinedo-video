import io
import base64
import json
import os
import csv
import math
import zipfile
import hashlib
import mimetypes
import uuid
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


def subir_video_google_drive(
    contenido,
    nombre_archivo,
    mime_type="video/mp4",
    subcarpeta="VideosDrone",
):
    """
    Sube el video original a Drive con carga reanudable para tolerar
    archivos grandes. Devuelve (DriveFileID, URL de apertura).
    """
    if not contenido:
        raise ValueError("El archivo de video está vacío.")

    drive_service, _ = obtener_google_clients()
    folder_id = obtener_o_crear_subcarpeta_drive(subcarpeta)

    media = MediaIoBaseUpload(
        io.BytesIO(contenido),
        mimetype=mime_type or "video/mp4",
        resumable=True,
        chunksize=8 * 1024 * 1024,
    )

    request = drive_service.files().create(
        body={
            "name": nombre_archivo,
            "parents": [folder_id],
        },
        media_body=media,
        fields="id,name,webViewLink",
        supportsAllDrives=True,
    )

    response = None
    while response is None:
        _, response = request.next_chunk()

    file_id = response["id"]
    url = response.get("webViewLink") or f"https://drive.google.com/file/d/{file_id}/view"
    return file_id, url


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


def obtener_historial_google(limite=100):
    _, sheets_service = obtener_google_clients()

    _asegurar_hoja_historial()

    spreadsheet_id = _secret_text("GSHEET_ID")

    rows = sheets_service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range=f"{HISTORIAL_SHEET_NAME}!A2:P",
    ).execute().get("values", [])

    registros = []

    for row in rows:
        row = list(row) + [""] * (16 - len(row))

        registros.append({
            "id": row[0],
            "fecha": row[1],
            "nombre": row[2],
            "imagen_original_file_id": row[3],
            "imagen_procesada_file_id": row[4],
            "surcos": int(float(row[5] or 0)),
            "verde_pct": _float_historial(row[6]),
            "rojo_pct": _float_historial(row[7]),
            "amarillo_pct": _float_historial(row[8]),
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
        displ
