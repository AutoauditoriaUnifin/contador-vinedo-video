"""Integración de TerraCore con AppSheet.

AppSheet crea una solicitud en SolicitudesAnalisis y llama al endpoint
/appsheet/analysis/submit. El endpoint encola un trabajo en Cloud Tasks.
El worker procesa las imágenes con el motor OpenCV extraído de Streamlit
y guarda los resultados en el mismo Google Sheets y Drive.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import secrets
import unicodedata
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, Header, HTTPException
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from PIL import Image

from terracore_engine import (
    _ocv2_analyze_health,
    _ocv2_group_consecutive,
    _ocv2_inventory_from_file,
    _ocv2_polygon_for_group,
    _ocv2_slot_positions,
    _tc_quitar_nombre_motor_del_analisis,
    _tg_clean_indices,
    _tg_norm_to_px,
)

router = APIRouter(prefix="/appsheet", tags=["AppSheet"])

REQUEST_SHEET = "SolicitudesAnalisis"
REQUEST_HEADERS = [
    "SolicitudID",
    "FechaSolicitud",
    "ParcelaID",
    "ParcelaNombre",
    "Idioma",
    "Fotos",
    "SurcoID",
    "Estado",
    "FechaInicio",
    "FechaFin",
    "HistorialIDs",
    "TotalFotos",
    "ResumenResultado",
    "Error",
]

SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]

HISTORIAL_SHEET = "HistorialTerroCore"
DRIVE_ROOT_ID = os.getenv("GDRIVE_PARENT_FOLDER_ID", "").strip()
APPSHEET_DRIVE_ROOT_ID = (
    os.getenv("APPSHEET_DRIVE_ROOT_ID", "").strip() or DRIVE_ROOT_ID
)
SPREADSHEET_ID = os.getenv("GSHEET_ID", "").strip()
API_TOKEN = os.getenv("APPSHEET_API_TOKEN", "").strip()


class UploadedBytes:
    """Adaptador mínimo compatible con las funciones del motor Streamlit."""

    def __init__(self, name: str, mime_type: str, data: bytes):
        self.name = name
        self.type = mime_type or "image/jpeg"
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_header(value: Any) -> str:
    text = unicodedata.normalize("NFD", str(value or "").strip().casefold())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", text)


def _column_letter(index_zero_based: int) -> str:
    number = int(index_zero_based) + 1
    result = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _quote_sheet(name: str) -> str:
    return "'" + str(name).replace("'", "''") + "'"


def _verify_token(
    authorization: str | None = None,
    x_worker_token: str | None = None,
) -> None:
    expected = os.getenv("APPSHEET_API_TOKEN", "").strip()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Falta configurar el secreto APPSHEET_API_TOKEN en Cloud Run.",
        )

    bearer = ""
    if authorization:
        parts = authorization.strip().split(None, 1)
        if len(parts) == 2 and parts[0].casefold() == "bearer":
            bearer = parts[1].strip()

    supplied = str(x_worker_token or bearer or "").strip()
    if not supplied or not secrets.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Token de integración no válido.")


def _google_clients():
    if not SPREADSHEET_ID:
        raise RuntimeError("Falta la variable GSHEET_ID.")
    if not DRIVE_ROOT_ID:
        raise RuntimeError("Falta la variable GDRIVE_PARENT_FOLDER_ID.")

    raw = os.getenv("GCP_SERVICE_ACCOUNT_JSON", "").strip()
    if raw:
        try:
            info = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("GCP_SERVICE_ACCOUNT_JSON no contiene JSON válido.") from exc
        from google.oauth2 import service_account

        credentials = service_account.Credentials.from_service_account_info(
            info,
            scopes=SCOPES,
        )
    else:
        import google.auth

        credentials, _ = google.auth.default(scopes=SCOPES)

    drive = build(
        "drive",
        "v3",
        credentials=credentials,
        cache_discovery=False,
    )
    sheets = build(
        "sheets",
        "v4",
        credentials=credentials,
        cache_discovery=False,
    )
    return drive, sheets, SPREADSHEET_ID


def _sheet_metadata(sheets) -> dict[str, dict[str, Any]]:
    response = sheets.spreadsheets().get(
        spreadsheetId=SPREADSHEET_ID,
        fields="sheets.properties(sheetId,title,gridProperties(rowCount,columnCount))",
    ).execute()
    result = {}
    for item in response.get("sheets", []):
        props = item.get("properties", {})
        result[str(props.get("title", ""))] = props
    return result


def _ensure_sheet_and_headers(sheets, sheet_name: str, required_headers: list[str]) -> list[str]:
    metadata = _sheet_metadata(sheets)
    if sheet_name not in metadata:
        sheets.spreadsheets().batchUpdate(
            spreadsheetId=SPREADSHEET_ID,
            body={
                "requests": [
                    {
                        "addSheet": {
                            "properties": {
                                "title": sheet_name,
                                "gridProperties": {
                                    "rowCount": 1000,
                                    "columnCount": max(26, len(required_headers)),
                                },
                            }
                        }
                    }
                ]
            },
        ).execute()
        metadata = _sheet_metadata(sheets)

    props = metadata[sheet_name]
    response = sheets.spreadsheets().values().get(
        spreadsheetId=SPREADSHEET_ID,
        range=f"{_quote_sheet(sheet_name)}!1:1",
    ).execute()
    headers = list((response.get("values") or [[]])[0])

    if not any(str(x or "").strip() for x in headers):
        headers = list(required_headers)
        sheets.spreadsheets().values().update(
            spreadsheetId=SPREADSHEET_ID,
            range=f"{_quote_sheet(sheet_name)}!A1:{_column_letter(len(headers)-1)}1",
            valueInputOption="RAW",
            body={"values": [headers]},
        ).execute()
        return headers

    existing = {_normalize_header(x) for x in headers if str(x or "").strip()}
    missing = [x for x in required_headers if _normalize_header(x) not in existing]
    if missing:
        start_index = len(headers)
        required_columns = start_index + len(missing)
        current_columns = int(props.get("gridProperties", {}).get("columnCount", 26))
        if required_columns > current_columns:
            sheets.spreadsheets().batchUpdate(
                spreadsheetId=SPREADSHEET_ID,
                body={
                    "requests": [
                        {
                            "appendDimension": {
                                "sheetId": props["sheetId"],
                                "dimension": "COLUMNS",
                                "length": required_columns - current_columns,
                            }
                        }
                    ]
                },
            ).execute()

        updates = []
        for offset, header in enumerate(missing):
            col = _column_letter(start_index + offset)
            updates.append(
                {
                    "range": f"{_quote_sheet(sheet_name)}!{col}1",
                    "values": [[header]],
                }
            )
        sheets.spreadsheets().values().batchUpdate(
            spreadsheetId=SPREADSHEET_ID,
            body={"valueInputOption": "RAW", "data": updates},
        ).execute()
        headers.extend(missing)

    return headers


def _read_sheet(sheets, sheet_name: str) -> tuple[list[str], list[list[Any]]]:
    response = sheets.spreadsheets().values().get(
        spreadsheetId=SPREADSHEET_ID,
        range=f"{_quote_sheet(sheet_name)}!A:AZ",
        valueRenderOption="FORMATTED_VALUE",
    ).execute()
    values = response.get("values", []) or []
    if not values:
        return [], []
    return list(values[0]), values[1:]


def _row_as_dict(headers: list[str], row: list[Any]) -> dict[str, Any]:
    result = {}
    for index, header in enumerate(headers):
        if not str(header or "").strip():
            continue
        result[_normalize_header(header)] = row[index] if index < len(row) else ""
    return result


def _find_row_by_key(
    sheets,
    sheet_name: str,
    key_header: str,
    key_value: str,
) -> tuple[int | None, dict[str, Any] | None, list[str]]:
    headers, rows = _read_sheet(sheets, sheet_name)
    key = _normalize_header(key_header)
    target = str(key_value or "").strip()
    if not headers or not target:
        return None, None, headers

    for physical_row, row in enumerate(rows, start=2):
        data = _row_as_dict(headers, row)
        if str(data.get(key, "") or "").strip() == target:
            return physical_row, data, headers
    return None, None, headers


def _append_row_once(
    sheets,
    sheet_name: str,
    values: dict[str, Any],
    key_header: str,
    key_value: str,
) -> int:
    headers = _ensure_sheet_and_headers(
        sheets,
        sheet_name,
        list(values.keys()),
    )
    existing_row, _, headers = _find_row_by_key(
        sheets,
        sheet_name,
        key_header,
        key_value,
    )
    if existing_row:
        return existing_row

    index_map = {
        _normalize_header(header): index
        for index, header in enumerate(headers)
        if str(header or "").strip()
    }
    row = [""] * len(headers)
    for name, value in values.items():
        index = index_map.get(_normalize_header(name))
        if index is not None:
            row[index] = value

    last_col = _column_letter(max(0, len(headers) - 1))
    result = sheets.spreadsheets().values().append(
        spreadsheetId=SPREADSHEET_ID,
        range=f"{_quote_sheet(sheet_name)}!A:{last_col}",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body={"values": [row]},
    ).execute()

    updated_range = (
        result.get("updates", {}).get("updatedRange", "")
    )
    match = re.search(r"![A-Z]+(\d+)", updated_range)
    return int(match.group(1)) if match else 0


def _update_row_fields(
    sheets,
    sheet_name: str,
    physical_row: int,
    values: dict[str, Any],
) -> None:
    headers, _ = _read_sheet(sheets, sheet_name)
    index_map = {
        _normalize_header(header): index
        for index, header in enumerate(headers)
        if str(header or "").strip()
    }
    updates = []
    for name, value in values.items():
        index = index_map.get(_normalize_header(name))
        if index is None:
            continue
        col = _column_letter(index)
        updates.append(
            {
                "range": f"{_quote_sheet(sheet_name)}!{col}{physical_row}",
                "values": [[value]],
            }
        )
    if updates:
        sheets.spreadsheets().values().batchUpdate(
            spreadsheetId=SPREADSHEET_ID,
            body={"valueInputOption": "RAW", "data": updates},
        ).execute()


def _request_row(sheets, request_id: str) -> tuple[int, dict[str, Any]]:
    headers, rows = _read_sheet(sheets, REQUEST_SHEET)
    if not headers:
        raise RuntimeError(
            f"No existe la pestaña {REQUEST_SHEET} o no tiene encabezados. "
            "Ejecuta primero POST /appsheet/setup."
        )
    key = _normalize_header("SolicitudID")
    for physical_row, row in enumerate(rows, start=2):
        data = _row_as_dict(headers, row)
        if str(data.get(key, "") or "").strip() == str(request_id).strip():
            return physical_row, data
    raise HTTPException(
        status_code=404,
        detail=f"No encontré la solicitud {request_id} en {REQUEST_SHEET}.",
    )


def _parse_image_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_items = value
    else:
        text = str(value or "").strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            raw_items = parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            raw_items = re.split(r"[,;\n]", text)

    result = []
    seen = set()
    for item in raw_items:
        path = str(item or "").strip().strip('"').strip("'")
        if path and path not in seen:
            seen.add(path)
            result.append(path)
    return result


def _safe_name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")
    return text or "Parcela"


def _drive_escape(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace("'", "\\'")


def _drive_list(drive, query: str) -> list[dict[str, Any]]:
    results = []
    page_token = None
    while True:
        response = drive.files().list(
            q=query,
            spaces="drive",
            fields="nextPageToken,files(id,name,mimeType,parents,webViewLink)",
            pageSize=100,
            pageToken=page_token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        results.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break
        if len(results) >= 1000:
            break
    return results


def _find_child_folder(drive, parent_id: str, folder_name: str) -> str | None:
    name = _drive_escape(folder_name)
    query = (
        f"name = '{name}' and '{parent_id}' in parents "
        "and mimeType = 'application/vnd.google-apps.folder' "
        "and trashed = false"
    )
    matches = _drive_list(drive, query)
    return matches[0]["id"] if matches else None


def _resolve_appsheet_file(drive, path_or_id: str) -> tuple[bytes, str, str]:
    """Resuelve una ruta de archivo de AppSheet o un ID/URL de Drive."""
    value = str(path_or_id or "").strip()
    if not value:
        raise ValueError("La solicitud contiene una ruta de imagen vacía.")

    direct_id = ""
    match = re.search(r"/d/([A-Za-z0-9_-]{15,})", value)
    if match:
        direct_id = match.group(1)
    elif re.fullmatch(r"[A-Za-z0-9_-]{20,}", value):
        direct_id = value

    file_info = None
    if direct_id:
        file_info = drive.files().get(
            fileId=direct_id,
            fields="id,name,mimeType",
            supportsAllDrives=True,
        ).execute()
    else:
        normalized = value.replace("\\", "/")
        parts = [part for part in PurePosixPath(normalized).parts if part not in ("/", ".", "..")]
        filename = parts[-1] if parts else normalized
        folder_id = APPSHEET_DRIVE_ROOT_ID

        # Si la ruta es relativa a la carpeta de AppSheet, recorrer las subcarpetas.
        if folder_id and len(parts) > 1:
            for folder_name in parts[:-1]:
                next_id = _find_child_folder(drive, folder_id, folder_name)
                if not next_id:
                    folder_id = ""
                    break
                folder_id = next_id

        matches = []
        escaped_name = _drive_escape(filename)
        if folder_id:
            query = (
                f"name = '{escaped_name}' and '{folder_id}' in parents "
                "and trashed = false"
            )
            matches = _drive_list(drive, query)

        if not matches:
            query = f"name = '{escaped_name}' and trashed = false"
            matches = _drive_list(drive, query)

        if not matches:
            raise FileNotFoundError(
                f"No encontré el archivo de AppSheet '{value}' en Google Drive. "
                "Comparte la carpeta de archivos de AppSheet con la cuenta de servicio."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Encontré varios archivos llamados '{filename}'. "
                "Configura APPSHEET_DRIVE_ROOT_ID con la carpeta correcta de AppSheet."
            )
        file_info = matches[0]

    mime_type = str(file_info.get("mimeType") or "application/octet-stream")
    if not mime_type.startswith("image/"):
        raise ValueError(f"El archivo '{file_info.get('name')}' no es una imagen.")

    request = drive.files().get_media(
        fileId=file_info["id"],
        supportsAllDrives=True,
    )
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buffer.getvalue(), str(file_info.get("name") or "foto.jpg"), mime_type


def _folder_id(drive, folder_name: str) -> str:
    parent_id = DRIVE_ROOT_ID
    existing = _find_child_folder(drive, parent_id, folder_name)
    if existing:
        return existing

    created = drive.files().create(
        body={
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id],
        },
        fields="id,name",
        supportsAllDrives=True,
    ).execute()
    return created["id"]


def _upload_artifact(
    drive,
    content: bytes,
    filename: str,
    mime_type: str,
    folder_name: str,
) -> tuple[str, str]:
    folder = _folder_id(drive, folder_name)
    escaped_name = _drive_escape(filename)
    query = (
        f"name = '{escaped_name}' and '{folder}' in parents "
        "and trashed = false"
    )
    matches = _drive_list(drive, query)
    if matches:
        file_id = matches[0]["id"]
    else:
        media = MediaIoBaseUpload(
            io.BytesIO(content),
            mimetype=mime_type or "application/octet-stream",
            resumable=True,
        )
        created = drive.files().create(
            body={"name": filename, "parents": [folder]},
            media_body=media,
            fields="id,name,webViewLink",
            supportsAllDrives=True,
        ).execute()
        file_id = created["id"]

    return file_id, f"{folder_name}/{filename}"


def _drive_url(file_id: str) -> str:
    return f"https://drive.google.com/file/d/{file_id}/view"


def _image_to_png_bytes(image: Any) -> bytes:
    if image is None:
        return b""
    if isinstance(image, Image.Image):
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()
    array = np.asarray(image)
    if array.size == 0:
        return b""
    ok, encoded = cv2.imencode(".png", array)
    if not ok:
        raise RuntimeError("No se pudo convertir una imagen procesada a PNG.")
    return encoded.tobytes()


def _deterministic_id(seed: str, attempt: int = 0) -> str:
    raw = f"{seed}|{attempt}".encode("utf-8")
    number = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")
    return str(10_000_000 + (number % 90_000_000))


def _analysis_id_for_request(sheets, request_id: str, photo_index: int) -> str:
    for attempt in range(100):
        candidate = _deterministic_id(f"{request_id}|foto|{photo_index}", attempt)
        _, existing, _ = _find_row_by_key(
            sheets,
            HISTORIAL_SHEET,
            "ID",
            candidate,
        )
        if not existing:
            return candidate

        legacy_text = str(existing.get(_normalize_header("LegacyExtra"), "") or "")
        try:
            legacy = json.loads(legacy_text)
        except Exception:
            legacy = {}
        if (
            str(legacy.get("solicitud_id", "")) == request_id
            and int(legacy.get("foto_index", 0) or 0) == photo_index
        ):
            return candidate

    raise RuntimeError("No se pudo generar un ID único para el análisis.")


def _build_regions(
    uploaded_image: UploadedBytes,
    backend_result: dict[str, Any],
    rows: list[dict[str, Any]],
    historial_id: str,
) -> list[dict[str, Any]]:
    pil = Image.open(io.BytesIO(uploaded_image.getvalue())).convert("RGB")
    width, height = pil.size
    health_map = backend_result.get("health_map", {}) or {}
    regions = []

    for row in rows:
        row_id = int(row.get("id", 0) or 0)
        count = int(row.get("slot_count", 0) or 0)
        if row_id <= 0 or count < 2:
            continue

        positions = _ocv2_slot_positions(row, (width, height))
        if len(positions) != count:
            continue

        health = health_map.get(row_id, health_map.get(str(row_id), {})) or {}
        red_indices = set(health.get("red_indices", []) or [])
        red_indices.update(_tg_clean_indices(row.get("vacant_indices", []), count))
        red_indices = {int(i) for i in red_indices if 1 <= int(i) <= count}

        row_spacing = float(row.get("row_spacing_px", 18.0) or 18.0)
        points = _tg_norm_to_px(row.get("points_norm", []), width, height)
        if len(points) < 2:
            continue

        a = np.asarray(points[0], dtype=np.float32)
        b = np.asarray(points[-1], dtype=np.float32)
        slot_pitch = float(row.get("slot_pitch_px", 0.0) or 0.0)
        if slot_pitch <= 0:
            slot_pitch = float(np.linalg.norm(b - a)) / max(1, count - 1)

        for group_index, group in enumerate(_ocv2_group_consecutive(red_indices), 1):
            polygon = _ocv2_polygon_for_group(
                positions,
                group,
                row_spacing,
                slot_pitch,
            )
            if not polygon:
                continue

            points_json = [
                [round(float(x), 2), round(float(y), 2)]
                for x, y in polygon
            ]
            contour = np.asarray(points_json, dtype=np.float32)
            area_px = abs(float(cv2.contourArea(contour))) if len(contour) >= 3 else 0.0
            area_pct = (100.0 * area_px / float(width * height)) if width and height else 0.0
            region_id = _deterministic_id(
                f"{historial_id}|region|{row_id}|{group_index}"
            )
            regions.append(
                {
                    "region_id": region_id,
                    "surco": row_id,
                    "primer_slot": int(group[0]),
                    "ultimo_slot": int(group[-1]),
                    "points": points_json,
                    "area_visual_pct": round(area_pct, 4),
                }
            )
    return regions


def _append_analysis_results(
    drive,
    sheets,
    request_id: str,
    photo_index: int,
    source_path: str,
    source_name: str,
    source_mime: str,
    raw: bytes,
    parcela_id: str,
    parcela_name: str,
    idioma: str,
    fecha_captura: str,
    surco_id: str,
) -> dict[str, Any]:
    uploaded = UploadedBytes(source_name, source_mime, raw)

    inventory = _ocv2_inventory_from_file(uploaded)
    backend_result = _ocv2_analyze_health(uploaded, inventory["rows"])
    rows = inventory.get("rows", []) or []
    table = inventory.get("table")

    total_slots = sum(int(row.get("slot_count", 0) or 0) for row in rows)
    vacant_slots = sum(
        len(_tg_clean_indices(row.get("vacant_indices", []), int(row.get("slot_count", 0) or 0)))
        for row in rows
    )
    occupied_slots = max(0, total_slots - vacant_slots)
    row_count = int(inventory.get("count", len(rows)) or len(rows))
    confidence = float(backend_result.get("confidence", 0.0) or 0.0)
    green_pct = float(backend_result.get("green_pct", 0.0) or 0.0)
    red_pct = float(backend_result.get("red_pct", 0.0) or 0.0)
    level = str(backend_result.get("nivel_afectacion_visual", "No determinado") or "No determinado")
    zone = str(backend_result.get("zona_mas_afectada", "No determinada") or "No determinada")
    diagnosis = _tc_quitar_nombre_motor_del_analisis(
        backend_result.get("diagnostico_visual", "")
    )
    causes = backend_result.get("causas_probables", []) or []
    recommendations = backend_result.get("recomendaciones_iniciales", []) or []
    explanation = _tc_quitar_nombre_motor_del_analisis(
        backend_result.get("explicacion_nutrientes", "")
    )
    note = _tc_quitar_nombre_motor_del_analisis(
        backend_result.get("nota_diagnostico", "")
    )
    regions = _build_regions(uploaded, backend_result, rows, _analysis_id_for_request(
        sheets, request_id, photo_index
    ))

    historial_id = _analysis_id_for_request(sheets, request_id, photo_index)
    now = _now_iso()
    parcel_safe = _safe_name(parcela_name or parcela_id or "Parcela")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_name = f"{parcel_safe}_{stamp}_{historial_id}_F{photo_index:02d}"

    extension = PurePosixPath(source_name.replace("\\", "/")).suffix.lower()
    if extension not in (".jpg", ".jpeg", ".png", ".webp"):
        extension = ".jpg"

    original_name = f"{base_name}{extension}"
    inventory_name = f"{base_name}_inventario.png"
    processed_name = f"{base_name}_analisis.png"
    polygon_name = f"{base_name}_poligonos.png"

    original_id, original_path = _upload_artifact(
        drive, raw, original_name, source_mime, "Originales"
    )
    inventory_bytes = _image_to_png_bytes(inventory.get("image"))
    processed_bytes = _image_to_png_bytes(backend_result.get("annotated"))
    polygon_bytes = _image_to_png_bytes(backend_result.get("polygon_image"))

    inventory_id, inventory_path = _upload_artifact(
        drive, inventory_bytes, inventory_name, "image/png", "Inventarios"
    )
    processed_id, processed_path = _upload_artifact(
        drive, processed_bytes, processed_name, "image/png", "Procesadas"
    )

    polygon_id = ""
    polygon_path = ""
    if polygon_bytes:
        polygon_id, polygon_path = _upload_artifact(
            drive, polygon_bytes, polygon_name, "image/png", "Poligonos"
        )

    legacy_extra = json.dumps(
        {
            "origen": "AppSheet",
            "solicitud_id": request_id,
            "foto_index": photo_index,
            "foto_fuente": source_path,
            "surco_id": surco_id,
            "metodo": backend_result.get("metodo", "opencv-v2-straight-grid-polygons"),
        },
        ensure_ascii=False,
    )

    history_values = {
        "ID": historial_id,
        "Fecha": now,
        "Nombre": original_name,
        "ImagenOriginalFileID": original_id,
        "ImagenProcesadaFileID": processed_id,
        "Surcos": row_count,
        "VerdePct": green_pct,
        "RojoPct": red_pct,
        "AmarilloPct": 0,
        "NivelVisual": level,
        "ZonaMasAfectada": zone,
        "DiagnosticoVisual": diagnosis,
        "CausasProbables": json.dumps(causes, ensure_ascii=False),
        "ExplicacionNutrientes": explanation,
        "Recomendaciones": json.dumps(recommendations, ensure_ascii=False),
        "NotaDiagnostico": note,
        "ImagenInventarioFileID": inventory_id,
        "ImagenPoligonosFileID": polygon_id,
        "ImagenAnalisisFileID": processed_id,
        "LegacyExtra": legacy_extra,
        "ImagenOriginalURL": _drive_url(original_id),
        "ImagenProcesadaURL": _drive_url(processed_id),
        "ImagenOriginalAppSheet": original_path,
        "ImagenProcesadaAppSheet": processed_path,
        "ImagenInventarioAppSheet": inventory_path,
        "ImagenPoligonosAppSheet": polygon_path,
        "Parcelas": parcela_name,
        "ImagenAnalisisAppSheet": processed_path,
        "NombreFoto": source_path,
        "Idioma": idioma,
        "ImagenLineasAppSheet": processed_path,
        "SurcosAutomaticos": row_count,
        "SurcosReales": row_count,
        "PoligonosAutomaticos": len(regions),
        "PoligonosReales": len(regions),
        "SlotsReales": total_slots,
        "OcupadosReales": occupied_slots,
        "VaciosReales": vacant_slots,
        "ImagenPoligonosURL": _drive_url(polygon_id) if polygon_id else "",
    }
    _append_row_once(
        sheets,
        HISTORIAL_SHEET,
        history_values,
        "ID",
        historial_id,
    )

    analysis_values = {
        "AnalisisID": historial_id,
        "ParcelaID": parcela_id,
        "ParcelaNombre": parcela_name,
        "FechaCaptura": fecha_captura or now,
        "FechaAnalisis": now,
        "TipoAnalisis": "Inventario + Estado visual + Polígonos",
        "Metodo": backend_result.get("metodo", "opencv-v2-straight-grid-polygons"),
        "NombreImagen": original_name,
        "ImagenOriginalFileID": original_id,
        "ImagenInventarioFileID": inventory_id,
        "ImagenProcesadaFileID": processed_id,
        "Surcos": row_count,
        "SlotsTotales": total_slots,
        "Ocupados": occupied_slots,
        "Vacios": vacant_slots,
        "VerdePct": green_pct,
        "RojoPct": red_pct,
        "AmarilloPct": 0,
        "NivelVisual": level,
        "ZonaMasAfectada": zone,
        "DiagnosticoVisual": diagnosis,
        "Confianza": round(confidence * 100.0, 2),
        "CausasProbables": json.dumps(causes, ensure_ascii=False),
        "Recomendaciones": json.dumps(recommendations, ensure_ascii=False),
        "NotaDiagnostico": note,
        "Idioma": idioma,
        "SurcosAutomaticos": row_count,
        "SurcosReales": row_count,
        "PoligonosAutomaticos": len(regions),
        "PoligonosReales": len(regions),
        "SlotsReales": total_slots,
        "OcupadosReales": occupied_slots,
        "VaciosReales": vacant_slots,
        "ImagenPoligonosFileID": polygon_id,
        "ImagenPoligonosAppSheet": polygon_path,
    }
    _append_row_once(
        sheets,
        "Analisis",
        analysis_values,
        "AnalisisID",
        historial_id,
    )

    # Crear la parcela solo si no existe; nunca duplicar una clave ya usada.
    if parcela_id:
        _append_row_once(
            sheets,
            "Parcelas",
            {
                "ID": parcela_id,
                "NombreParcela": parcela_name or parcela_id,
                "RanchoVinedo": "",
                "FechaAlta": now,
                "KMLFileID": "",
                "Ubicacion": "",
                "Estado": "Activa",
                "Observaciones": "",
                "Idioma": idioma,
            },
            "ID",
            parcela_id,
        )

    # Guardar una fila por surco en InventarioSurcos.
    for row in rows:
        row_number = int(row.get("id", 0) or 0)
        slots = int(row.get("slot_count", 0) or 0)
        vacant = len(_tg_clean_indices(row.get("vacant_indices", []), slots))
        occupied = max(0, slots - vacant)
        confidence_row = round(float(row.get("slot_confidence", 0.0) or 0.0) * 100.0, 2)
        record_id = _deterministic_id(f"{historial_id}|inventario|{row_number}")
        _append_row_once(
            sheets,
            "InventarioSurcos",
            {
                "RegistroID": record_id,
                "AnalisisID": historial_id,
                "ParcelaID": parcela_id,
                "NumeroSurco": str(row_number),
                "Slots": slots,
                "Ocupados": occupied,
                "Vacios": vacant,
                "Confianza": confidence_row,
                "Idioma": idioma,
            },
            "RegistroID",
            record_id,
        )

    # Evidencias: originales, inventario, análisis y polígonos.
    evidence_items = [
        ("ORIGINAL", original_name, original_id, source_mime),
        ("INVENTARIO", inventory_name, inventory_id, "image/png"),
        ("ANALISIS", processed_name, processed_id, "image/png"),
    ]
    if polygon_id:
        evidence_items.append(("POLIGONOS", polygon_name, polygon_id, "image/png"))

    for evidence_type, filename, file_id, mime_type in evidence_items:
        evidence_id = _deterministic_id(
            f"{historial_id}|evidencia|{evidence_type}"
        )
        _append_row_once(
            sheets,
            "Evidencias",
            {
                "EvidenciaID": evidence_id,
                "AnalisisID": historial_id,
                "ParcelaID": parcela_id,
                "TipoEvidencia": evidence_type,
                "NombreArchivo": filename,
                "DriveFileID": file_id,
                "MimeType": mime_type,
                "Fecha": now,
                "Fuente": "AppSheet TerraCore",
                "Idioma": idioma,
            },
            "EvidenciaID",
            evidence_id,
        )

    if note:
        _append_row_once(
            sheets,
            "NotaDiagnostico",
            {
                "ID": historial_id,
                "NotaDiagnostico": note,
                "Idioma": idioma,
            },
            "ID",
            historial_id,
        )

    for index, region in enumerate(regions, 1):
        polygon_json = json.dumps(
            {"space": "image_px", "points": region["points"]},
            ensure_ascii=False,
        )
        area_pct = float(region["area_visual_pct"])
        region_id = region["region_id"]
        region_name = f"Región crítica {index}"
        segment = f"Slots {region['primer_slot']}-{region['ultimo_slot']}"
        _append_row_once(
            sheets,
            "RegionesCriticas",
            {
                "RegionID": region_id,
                "AnalisisID": historial_id,
                "ParcelaID": parcela_id,
                "NombreRegion": region_name,
                "PoligonoJSON": polygon_json,
                "PrimerSurco": region["surco"],
                "UltimoSurco": region["surco"],
                "TramoAprox": segment,
                "AreaVisualPct": area_pct,
                "AreaM2": 0,
                "NivelVisual": level,
                "Caracteristica": "Tramo seco, débil o vacío",
                "Confianza": round(confidence * 100.0, 2),
                "EstadoSeguimiento": "Pendiente",
                "Fecha": now,
                "Idioma": idioma,
            },
            "RegionID",
            region_id,
        )
        zone_id = _deterministic_id(f"{historial_id}|zona|{index}")
        _append_row_once(
            sheets,
            "ZonasDañadas",
            {
                "ZonaID": zone_id,
                "AnalisisID": historial_id,
                "ParcelaID": parcela_id,
                "NombreZona": f"Zona dañada {index}",
                "PoligonoJSON": polygon_json,
                "AreaM2": 0,
                "Nivel": level,
                "SurcosRelacionados": str(region["surco"]),
                "Estado": "Pendiente",
                "Fecha": now,
                "Idioma": idioma,
            },
            "ZonaID",
            zone_id,
        )

    return {
        "historial_id": historial_id,
        "parcela": parcela_name,
        "surcos": row_count,
        "slots": total_slots,
        "ocupados": occupied_slots,
        "vacios": vacant_slots,
        "verde_pct": round(green_pct, 2),
        "rojo_pct": round(red_pct, 2),
        "poligonos": len(regions),
        "imagenes": {
            "original": original_path,
            "inventario": inventory_path,
            "analisis": processed_path,
            "poligonos": polygon_path,
        },
    }


def _enqueue_analysis_task(request_id: str) -> str:
    from google.cloud import tasks_v2

    project = (
        os.getenv("CLOUD_TASKS_PROJECT", "").strip()
        or os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    )
    location = os.getenv("CLOUD_TASKS_LOCATION", "us-central1").strip()
    queue = os.getenv("CLOUD_TASKS_QUEUE", "terracore-analysis").strip()
    service_url = (
        os.getenv("CLOUD_RUN_SERVICE_URL", "").strip().rstrip("/")
        or os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
    )

    if not project or not service_url:
        raise RuntimeError(
            "Faltan CLOUD_TASKS_PROJECT/GOOGLE_CLOUD_PROJECT o CLOUD_RUN_SERVICE_URL."
        )

    token = os.getenv("APPSHEET_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Falta APPSHEET_API_TOKEN.")

    client = tasks_v2.CloudTasksClient()
    parent = client.queue_path(project, location, queue)
    worker_url = f"{service_url}/appsheet/analysis/process"

    http_request = {
        "http_method": tasks_v2.HttpMethod.POST,
        "url": worker_url,
        "headers": {
            "Content-Type": "application/json",
            "X-Worker-Token": token,
        },
        "body": json.dumps({"request_id": request_id}).encode("utf-8"),
    }

    task_service_account = os.getenv("CLOUD_TASKS_SERVICE_ACCOUNT_EMAIL", "").strip()
    if task_service_account:
        http_request["oidc_token"] = {
            "service_account_email": task_service_account,
            "audience": service_url,
        }

    created = client.create_task(
        parent=parent,
        task={"http_request": http_request},
    )
    return created.name


@router.get("/health")
def appsheet_health():
    return {
        "ok": True,
        "servicio": "TerraCore AppSheet API",
        "google_configurado": bool(SPREADSHEET_ID and DRIVE_ROOT_ID),
        "token_configurado": bool(os.getenv("APPSHEET_API_TOKEN", "").strip()),
        "cloud_tasks_configurado": bool(
            (
                os.getenv("CLOUD_TASKS_PROJECT", "").strip()
                or os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
            )
            and (
                os.getenv("CLOUD_RUN_SERVICE_URL", "").strip()
                or os.getenv("PUBLIC_BASE_URL", "").strip()
            )
        ),
    }


@router.post("/setup")
def appsheet_setup(authorization: str | None = Header(default=None)):
    _verify_token(authorization=authorization)
    try:
        _, sheets, _ = _google_clients()
        headers = _ensure_sheet_and_headers(
            sheets,
            REQUEST_SHEET,
            REQUEST_HEADERS,
        )
        return {
            "ok": True,
            "pestana": REQUEST_SHEET,
            "encabezados": headers,
            "mensaje": (
                "La pestaña quedó preparada en el mismo Google Sheets. "
                "Ahora agrégala como tabla en AppSheet."
            ),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/analysis/submit")
def appsheet_submit_analysis(
    payload: dict[str, Any],
    authorization: str | None = Header(default=None),
):
    _verify_token(authorization=authorization)
    request_id = str(payload.get("request_id") or payload.get("SolicitudID") or "").strip()
    if not request_id:
        raise HTTPException(status_code=400, detail="Falta request_id.")

    try:
        _, sheets, _ = _google_clients()
        row_number, row = _request_row(sheets, request_id)
        status = str(row.get(_normalize_header("Estado"), "") or "").strip().casefold()

        if status in ("completado", "completed"):
            return {
                "ok": True,
                "status": "Completado",
                "request_id": request_id,
                "mensaje": "Esta solicitud ya fue procesada.",
            }
        if status in ("en cola", "procesando"):
            return {
                "ok": True,
                "status": status.title(),
                "request_id": request_id,
                "mensaje": "La solicitud ya está en proceso.",
            }

        photo_paths = _parse_image_list(row.get(_normalize_header("Fotos"), ""))
        if not photo_paths:
            raise HTTPException(
                status_code=400,
                detail="Agrega al menos una fotografía en la columna Fotos.",
            )

        task_name = _enqueue_analysis_task(request_id)
        _update_row_fields(
            sheets,
            REQUEST_SHEET,
            row_number,
            {
                "Estado": "En cola",
                "Error": "",
            },
        )
        return {
            "ok": True,
            "status": "En cola",
            "request_id": request_id,
            "task": task_name,
            "mensaje": "La solicitud fue enviada al motor de análisis TerraCore.",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/analysis/process")
def appsheet_process_analysis(
    payload: dict[str, Any],
    x_worker_token: str | None = Header(default=None),
):
    _verify_token(x_worker_token=x_worker_token)
    request_id = str(payload.get("request_id") or "").strip()
    if not request_id:
        raise HTTPException(status_code=400, detail="Falta request_id.")

    drive = sheets = None
    try:
        drive, sheets, _ = _google_clients()
        row_number, row = _request_row(sheets, request_id)

        current_status = str(row.get(_normalize_header("Estado"), "") or "").strip().casefold()
        if current_status in ("completado", "completed"):
            return {
                "ok": True,
                "status": "Completado",
                "request_id": request_id,
                "mensaje": "La solicitud ya estaba terminada.",
            }

        photo_paths = _parse_image_list(row.get(_normalize_header("Fotos"), ""))
        if not photo_paths:
            raise ValueError("No hay fotografías registradas en la solicitud.")

        parcela_id = str(row.get(_normalize_header("ParcelaID"), "") or "").strip()
        parcela_name = str(row.get(_normalize_header("ParcelaNombre"), "") or "").strip()
        idioma = str(row.get(_normalize_header("Idioma"), "ES") or "ES").strip().upper()
        idioma = "FR" if idioma == "FR" else "ES"
        fecha_captura = str(row.get(_normalize_header("FechaSolicitud"), "") or "").strip()
        surco_id = str(row.get(_normalize_header("SurcoID"), "") or "").strip()

        _update_row_fields(
            sheets,
            REQUEST_SHEET,
            row_number,
            {
                "Estado": "Procesando",
                "FechaInicio": _now_iso(),
                "FechaFin": "",
                "Error": "",
            },
        )

        history_ids = _parse_image_list(row.get(_normalize_header("HistorialIDs"), ""))
        summaries = []

        for photo_index, source_path in enumerate(photo_paths, 1):
            raw, source_name, source_mime = _resolve_appsheet_file(drive, source_path)
            summary = _append_analysis_results(
                drive=drive,
                sheets=sheets,
                request_id=request_id,
                photo_index=photo_index,
                source_path=source_path,
                source_name=source_name,
                source_mime=source_mime,
                raw=raw,
                parcela_id=parcela_id,
                parcela_name=parcela_name,
                idioma=idioma,
                fecha_captura=fecha_captura,
                surco_id=surco_id,
            )
            if summary["historial_id"] not in history_ids:
                history_ids.append(summary["historial_id"])
            summaries.append(summary)

            _update_row_fields(
                sheets,
                REQUEST_SHEET,
                row_number,
                {
                    "HistorialIDs": json.dumps(history_ids, ensure_ascii=False),
                    "TotalFotos": len(history_ids),
                    "ResumenResultado": json.dumps(summaries, ensure_ascii=False),
                },
            )

        _update_row_fields(
            sheets,
            REQUEST_SHEET,
            row_number,
            {
                "Estado": "Completado",
                "FechaFin": _now_iso(),
                "HistorialIDs": json.dumps(history_ids, ensure_ascii=False),
                "TotalFotos": len(history_ids),
                "ResumenResultado": json.dumps(summaries, ensure_ascii=False),
                "Error": "",
            },
        )

        return {
            "ok": True,
            "status": "Completado",
            "request_id": request_id,
            "historial_ids": history_ids,
            "total_fotos": len(history_ids),
            "resultados": summaries,
        }

    except HTTPException:
        raise
    except Exception as exc:
        try:
            if sheets is not None:
                row_number, _ = _request_row(sheets, request_id)
                _update_row_fields(
                    sheets,
                    REQUEST_SHEET,
                    row_number,
                    {
                        "Estado": "Error",
                        "FechaFin": _now_iso(),
                        "Error": str(exc)[:4000],
                    },
                )
        except Exception:
            pass

        # Se responde 200 para no repetir automáticamente una solicitud
        # que requiere corregir datos, permisos o una fotografía.
        return {
            "ok": False,
            "status": "Error",
            "request_id": request_id,
            "error": str(exc),
        }
