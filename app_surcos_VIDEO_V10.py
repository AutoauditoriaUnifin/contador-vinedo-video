import io
import base64
import json
import os
import csv
import math
import zipfile
import tempfile
import requests
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import UnivariateSpline
from openai import OpenAI
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
            "verde_pct": float(row[6] or 0.0),
            "rojo_pct": float(row[7] or 0.0),
            "amarillo_pct": float(row[8] or 0.0),
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
        "angle": mean_angle
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

    # ========================================================
    # SELECTOR DE IDIOMA
    # Justo arriba de "1. Tipo de archivo"
    # ========================================================
    idioma_es_col, idioma_fr_col = st.columns(
        2,
        gap="medium"
    )

    with idioma_es_col:
        if st.button(
            "🇪🇸 ES Español",
            key="lang_es_v15_abajo",
            use_container_width=True,
            disabled=st.session_state.idioma_terrocore == "ES"
        ):
            st.session_state.idioma_terrocore = "ES"
            st.rerun()

    with idioma_fr_col:
        if st.button(
            "🇫🇷 FR Français",
            key="lang_fr_v15_abajo",
            use_container_width=True,
            disabled=st.session_state.idioma_terrocore == "FR"
        ):
            st.session_state.idioma_terrocore = "FR"
            st.rerun()

    # ========================================================
    # PASO 1 - PRUEBA DE CONEXIÓN OPENAI
    # ========================================================
    if st.button(
        tr(
            "🧠 Probar conexión IA",
            "🧠 Tester la connexion IA"
        ),
        key="probar_openai_paso1",
        use_container_width=True
    ):
        ok_openai, resultado_openai = probar_openai()

        if ok_openai:
            st.success(
                tr(
                    "✅ OpenAI conectado correctamente",
                    "✅ OpenAI connecté correctement"
                )
            )
            st.caption(resultado_openai)
        else:
            st.error(
                tr(
                    "❌ No se pudo conectar con OpenAI",
                    "❌ Impossible de se connecter à OpenAI"
                )
            )
            st.code(resultado_openai)

    # Espacio pequeño entre los botones y el panel siguiente
    st.markdown(
        "<div style='height:10px;'></div>",
        unsafe_allow_html=True
    )

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
                    # ====================================================
                    # IA VISUAL: enviar la fotografía completa al backend.
                    # GPT Image crea directamente las líneas sobre la foto.
                    # NO se llama al detector local analizar() para imágenes.
                    # ====================================================
                    ok_backend, backend_result = procesar_imagen_backend_ia(
                        uploaded_image
                    )

                    if not ok_backend:
                        raise RuntimeError(str(backend_result))

                    # ====================================================
                    # GUARDADO AUTOMÁTICO EN GOOGLE DRIVE + SHEETS
                    # Si falla, NO detiene el análisis actual.
                    # ====================================================
                    historial_google_ok = False
                    historial_google_info = ""

                    try:
                        historial_google_ok, historial_google_info = (
                            guardar_analisis_en_google(
                                uploaded_image,
                                backend_result
                            )
                        )
                    except Exception as historial_exc:
                        historial_google_info = str(historial_exc)

                    analyzed_images.append({
                        "id": f"{index}_{uploaded_image.name}",
                        "name": uploaded_image.name,
                        "count": int(backend_result.get("count", 0)),
                        "green_pct": float(backend_result.get("green_pct", 0.0)),
                        "red_pct": float(backend_result.get("red_pct", 0.0)),
                        "angle": float(backend_result.get("angle", 0.0)),
                        "annotated": backend_result["annotated"],
                        "ia_scene": backend_result.get("backend"),
                        "result_url": backend_result.get("result_url"),
                        "historial_google_guardado": historial_google_ok,
                        "historial_google_info": historial_google_info,

                        # Diagnóstico agronómico
                        "zona_mas_afectada": backend_result.get(
                            "zona_mas_afectada",
                            "No determinada"
                        ),
                        "nivel_afectacion_visual": backend_result.get(
                            "nivel_afectacion_visual",
                            "No determinado"
                        ),
                        "diagnostico_visual": backend_result.get(
                            "diagnostico_visual",
                            ""
                        ),
                        "causas_probables": backend_result.get(
                            "causas_probables",
                            []
                        ),
                        "explicacion_nutrientes": backend_result.get(
                            "explicacion_nutrientes",
                            ""
                        ),
                        "recomendaciones_iniciales": backend_result.get(
                            "recomendaciones_iniciales",
                            []
                        ),
                        "nota_diagnostico": backend_result.get(
                            "nota_diagnostico",
                            ""
                        ),
                        "detalle_zonas": backend_result.get(
                            "detalle_zonas",
                            {}
                        ),

                        "metodo": backend_result.get("metodo", "gpt-image")
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
                ),
                tr(
                    "Zona más afectada",
                    "Zone la plus touchée"
                ): item.get(
                    "zona_mas_afectada",
                    "—"
                ),
                tr(
                    "Nivel visual",
                    "Niveau visuel"
                ): item.get(
                    "nivel_afectacion_visual",
                    "—"
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
    # DIAGNÓSTICO AGRONÓMICO - SOLO IMÁGENES
    # Se agrega sin quitar la tabla ni las imágenes existentes.
    # ========================================================
    if modo == "imagenes":

        st.subheader(
            tr(
                "Diagnóstico agronómico",
                "Diagnostic agronomique"
            )
        )

        st.caption(
            tr(
                "Interpretación visual preliminar. Para confirmar deficiencias de nutrientes se requiere análisis de suelo, análisis foliar y revisión del riego.",
                "Interprétation visuelle préliminaire. Pour confirmer une carence en nutriments, une analyse du sol, une analyse foliaire et une vérification de l'irrigation sont nécessaires."
            )
        )

        for diag_idx, item in enumerate(active_items):

            with st.container(border=True):

                st.markdown(
                    f"### {item.get('name', tr('Imagen', 'Image'))}"
                )

                diag_col1, diag_col2, diag_col3, diag_col4 = st.columns(4)

                with diag_col1:
                    st.metric(
                        tr("Surcos", "Rangs"),
                        int(item.get("count", 0))
                    )

                with diag_col2:
                    st.metric(
                        tr("Verde", "Vert"),
                        f"{float(item.get('green_pct', 0.0)):.1f}%"
                    )

                with diag_col3:
                    st.metric(
                        tr("Seco / rojo", "Sec / rouge"),
                        f"{float(item.get('red_pct', 0.0)):.1f}%"
                    )

                with diag_col4:
                    st.metric(
                        tr("Nivel visual", "Niveau visuel"),
                        tr_diag_texto(
                            item.get(
                                "nivel_afectacion_visual",
                                "No determinado"
                            )
                        ).capitalize()
                    )

                st.markdown(
                    tr(
                        "**Zona más afectada:** ",
                        "**Zone la plus touchée :** "
                    )
                    +
                    tr_diag_texto(
                        item.get(
                            "zona_mas_afectada",
                            "No determinada"
                        )
                    ).capitalize()
                )

                diagnostico_visual = str(
                    item.get("diagnostico_visual", "") or ""
                ).strip()

                if not diagnostico_visual:
                    zona_tmp = str(
                        item.get("zona_mas_afectada", "No determinada")
                    )
                    nivel_tmp = str(
                        item.get(
                            "nivel_afectacion_visual",
                            "No determinado"
                        )
                    )

                    if (
                        zona_tmp not in {"", "No determinada", "—"}
                        or
                        nivel_tmp not in {"", "No determinado", "—"}
                    ):
                        diagnostico_visual = tr(
                            f"El análisis visual reporta un nivel {nivel_tmp.lower()} "
                            f"y señala como zona más afectada: {zona_tmp}.",
                            f"L'analyse visuelle indique un niveau {nivel_tmp.lower()} "
                            f"et signale comme zone la plus touchée : {zona_tmp}."
                        )

                if diagnostico_visual:
                    st.markdown(
                        tr(
                            "#### Diagnóstico visual",
                            "#### Diagnostic visuel"
                        )
                    )
                    st.write(tr_diag_texto(diagnostico_visual))

                causas = item.get("causas_probables", []) or []

                if causas:
                    st.markdown(
                        tr(
                            "#### Causas probables",
                            "#### Causes probables"
                        )
                    )

                    for causa in causas:
                        st.markdown(f"- {tr_diag_texto(causa)}")

                explicacion_nutrientes = str(
                    item.get("explicacion_nutrientes", "") or ""
                ).strip()

                if explicacion_nutrientes:
                    st.markdown(
                        tr(
                            "#### Suelo y nutrientes",
                            "#### Sol et nutriments"
                        )
                    )
                    st.write(tr_diag_texto(explicacion_nutrientes))

                recomendaciones = item.get(
                    "recomendaciones_iniciales",
                    []
                ) or []

                if recomendaciones:
                    st.markdown(
                        tr(
                            "#### Recomendaciones iniciales",
                            "#### Recommandations initiales"
                        )
                    )

                    for recomendacion in recomendaciones:
                        st.markdown(f"- {tr_diag_texto(recomendacion)}")

                nota = str(
                    item.get("nota_diagnostico", "") or ""
                ).strip()

                if nota:
                    st.info(tr_diag_texto(nota))


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

            # Mini-resumen agronómico debajo de cada imagen.
            if modo == "imagenes":
                st.markdown(
                    tr(
                        f"**Zona más afectada:** {item.get('zona_mas_afectada', '—')}  |  "
                        f"**Nivel visual:** {item.get('nivel_afectacion_visual', '—')}",
                        f"**Zone la plus touchée :** {tr_diag_texto(item.get('zona_mas_afectada', '—'))}  |  "
                        f"**Niveau visuel :** {tr_diag_texto(item.get('nivel_afectacion_visual', '—'))}"
                    )
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



# ============================================================
# HISTORIAL DE ANÁLISIS - GOOGLE DRIVE
# ============================================================

st.markdown("---")

with st.expander(
    tr(
        "📂 Historial de análisis",
        "📂 Historique des analyses"
    ),
    expanded=False
):

    if not historial_google_configurado():

        st.info(
            tr(
                "El historial de Google Drive aún no está configurado.",
                "L’historique Google Drive n’est pas encore configuré."
            )
        )

    else:

        try:
            registros_historial = obtener_historial_google(
                limite=100
            )

            if not registros_historial:

                st.info(
                    tr(
                        "Todavía no hay análisis guardados.",
                        "Aucune analyse enregistrée pour le moment."
                    )
                )

            else:

                filas_historial = []

                for registro in registros_historial:
                    filas_historial.append({
                        tr("Fecha", "Date"):
                            registro.get("fecha", ""),
                        tr("Imagen", "Image"):
                            registro.get("nombre", ""),
                        tr("Surcos", "Rangs"):
                            registro.get("surcos", 0),
                        tr("Verde %", "Vert %"):
                            registro.get("verde_pct", 0.0),
                        tr("Rojo %", "Rouge %"):
                            registro.get("rojo_pct", 0.0),
                        tr("Amarillo %", "Jaune %"):
                            registro.get("amarillo_pct", 0.0),
                        tr(
                            "Zona más afectada",
                            "Zone la plus touchée"
                        ):
                            tr_diag_texto(
                                registro.get(
                                    "zona_mas_afectada",
                                    "No determinada"
                                )
                            ),
                        tr(
                            "Nivel visual",
                            "Niveau visuel"
                        ):
                            tr_diag_texto(
                                registro.get(
                                    "nivel_visual",
                                    "No determinado"
                                )
                            ),
                    })

                st.dataframe(
                    pd.DataFrame(
                        filas_historial
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                seleccionado = st.selectbox(
                    tr(
                        "Selecciona un análisis para revisarlo",
                        "Sélectionnez une analyse à consulter"
                    ),
                    options=list(
                        range(
                            len(registros_historial)
                        )
                    ),
                    format_func=lambda i: (
                        f"{registros_historial[i].get('fecha', '')} — "
                        f"{registros_historial[i].get('nombre', '')}"
                    ),
                    key="historial_google_select",
                )

                registro = registros_historial[
                    int(seleccionado)
                ]

                st.markdown(
                    f"### {registro.get('nombre', tr('Imagen', 'Image'))}"
                )

                img_col1, img_col2 = st.columns(2)

                with img_col1:
                    st.markdown(
                        tr(
                            "**Imagen original**",
                            "**Image originale**"
                        )
                    )

                    original_id = str(
                        registro.get(
                            "imagen_original_file_id",
                            ""
                        )
                    ).strip()

                    if original_id:
                        try:
                            original_bytes = (
                                descargar_archivo_google_drive(
                                    original_id
                                )
                            )

                            st.image(
                                original_bytes,
                                use_container_width=True,
                            )
                        except Exception as exc:
                            st.warning(
                                tr(
                                    f"No se pudo abrir la imagen original: {exc}",
                                    f"Impossible d’ouvrir l’image originale : {exc}"
                                )
                            )

                with img_col2:
                    st.markdown(
                        tr(
                            "**Imagen procesada**",
                            "**Image traitée**"
                        )
                    )

                    processed_id = str(
                        registro.get(
                            "imagen_procesada_file_id",
                            ""
                        )
                    ).strip()

                    if processed_id:
                        try:
                            processed_bytes = (
                                descargar_archivo_google_drive(
                                    processed_id
                                )
                            )

                            st.image(
                                processed_bytes,
                                use_container_width=True,
                            )
                        except Exception as exc:
                            st.warning(
                                tr(
                                    f"No se pudo abrir la imagen procesada: {exc}",
                                    f"Impossible d’ouvrir l’image traitée : {exc}"
                                )
                            )

                hm1, hm2, hm3, hm4 = st.columns(4)

                with hm1:
                    st.metric(
                        tr("Surcos", "Rangs"),
                        registro.get("surcos", 0),
                    )

                with hm2:
                    st.metric(
                        tr("Verde", "Vert"),
                        f"{float(registro.get('verde_pct', 0) or 0):.1f}%",
                    )

                with hm3:
                    st.metric(
                        tr("Rojo", "Rouge"),
                        f"{float(registro.get('rojo_pct', 0) or 0):.1f}%",
                    )

                with hm4:
                    st.metric(
                        tr("Amarillo", "Jaune"),
                        f"{float(registro.get('amarillo_pct', 0) or 0):.1f}%",
                    )

                st.markdown(
                    tr(
                        "**Zona más afectada:** ",
                        "**Zone la plus touchée :** "
                    )
                    +
                    tr_diag_texto(
                        registro.get(
                            "zona_mas_afectada",
                            "No determinada"
                        )
                    ).capitalize()
                )

                st.markdown(
                    tr(
                        "**Nivel visual:** ",
                        "**Niveau visuel :** "
                    )
                    +
                    tr_diag_texto(
                        registro.get(
                            "nivel_visual",
                            "No determinado"
                        )
                    ).capitalize()
                )

                diagnostico_hist = str(
                    registro.get(
                        "diagnostico_visual",
                        ""
                    )
                    or ""
                ).strip()

                if diagnostico_hist:
                    st.markdown(
                        tr(
                            "#### Diagnóstico visual",
                            "#### Diagnostic visuel"
                        )
                    )
                    st.write(
                        tr_diag_texto(
                            diagnostico_hist
                        )
                    )

                causas_hist = registro.get(
                    "causas_probables",
                    []
                ) or []

                if causas_hist:
                    st.markdown(
                        tr(
                            "#### Causas probables",
                            "#### Causes probables"
                        )
                    )

                    for causa in causas_hist:
                        st.markdown(
                            f"- {tr_diag_texto(causa)}"
                        )

                nutrientes_hist = str(
                    registro.get(
                        "explicacion_nutrientes",
                        ""
                    )
                    or ""
                ).strip()

                if nutrientes_hist:
                    st.markdown(
                        tr(
                            "#### Suelo y nutrientes",
                            "#### Sol et nutriments"
                        )
                    )

                    st.write(
                        tr_diag_texto(
                            nutrientes_hist
                        )
                    )

                recomendaciones_hist = registro.get(
                    "recomendaciones",
                    []
                ) or []

                if recomendaciones_hist:
                    st.markdown(
                        tr(
                            "#### Recomendaciones iniciales",
                            "#### Recommandations initiales"
                        )
                    )

                    for rec in recomendaciones_hist:
                        st.markdown(
                            f"- {tr_diag_texto(rec)}"
                        )

                nota_hist = str(
                    registro.get(
                        "nota_diagnostico",
                        ""
                    )
                    or ""
                ).strip()

                if nota_hist:
                    st.info(
                        tr_diag_texto(
                            nota_hist
                        )
                    )

        except Exception as historial_exc:

            st.error(
                tr(
                    "No se pudo cargar el historial de Google Drive.",
                    "Impossible de charger l’historique Google Drive."
                )
            )

            st.code(
                str(historial_exc)
            )

