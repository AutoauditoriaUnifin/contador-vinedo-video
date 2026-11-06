"""Integración de TerraCore con AppSheet.

El worker puede ejecutarse desde GitHub Actions sin servidor permanente.
Google Apps Script entrega un token OAuth temporal para acceder al mismo
Google Sheets y Drive; el motor OpenCV se reutiliza sin abrir Streamlit.
El endpoint FastAPI/Cloud Tasks se conserva como vía heredada opcional.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import secrets
import time
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
    _ocv2_render_inventory_with_rows,
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
    "Foto01",
    "Foto02",
    "Foto03",
    "Foto04",
    "Foto05",
    "Video",
    "Video simulación ",
    "SurcoID",
    "Estado",
    "FechaInicio",
    "FechaFin",
    "HistorialIDs",
    "TotalFotos",
    "ResumenResultado",
    "Error",
    "ValidacionInventario",
    "FechaInventario",
    "FechaValidacionInventario",
    "FechaInicioSalud",
    "ResumenInventario",
]

INVENTORY_SHEET = "InventarioSurcos"
INVENTORY_HEADERS = [
    "RegistroID",
    "SolicitudID",
    "AnalisisID",
    "ParcelaID",
    "ParcelaNombre",
    "FotoIndex",
    "FotoNombre",
    "FotoRuta",
    "ImagenInventarioAppSheet",
    "NumeroSurco",
    "Slots",
    "Ocupados",
    "Vacios",
    "VaciosIndices",
    "Confianza",
    "ConfianzaSurco",
    "ConfianzaSlots",
    "PuntoInicioX",
    "PuntoInicioY",
    "PuntoFinX",
    "PuntoFinY",
    "RowSpacingPx",
    "SlotPitchPx",
    "GeometriaJSON",
    "ManualPosition",
    "ManualInventory",
    "EstadoInventario",
    "FechaValidacion",
    "Idioma",
    "Fecha",
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


def _execute_sheets_request(request, max_attempts: int = 6):
    """Ejecuta una llamada de Sheets y espera si Google devuelve cuota 429."""
    for attempt in range(max_attempts):
        try:
            return request.execute()
        except Exception as exc:
            status = getattr(getattr(exc, "resp", None), "status", None)
            if status != 429 or attempt >= max_attempts - 1:
                raise
            retry_after = None
            response = getattr(exc, "resp", None)
            if response is not None:
                try:
                    retry_after = float(response.get("retry-after"))
                except (TypeError, ValueError, AttributeError):
                    retry_after = None
            delay = retry_after if retry_after and retry_after > 0 else min(60, 2 ** (attempt + 1))
            time.sleep(delay)
    raise RuntimeError("No se pudo completar la solicitud a Google Sheets.")


def _verify_token(
    authorization: str | None = None,
    x_worker_token: str | None = None,
) -> None:
    expected = os.getenv("APPSHEET_API_TOKEN", "").strip()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Falta configurar el secreto APPSHEET_API_TOKEN.",
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

    # Modo gratuito: Apps Script entrega un OAuth access token temporal
    # del propietario del archivo. Se usa solo en memoria durante este job.
    access_token = os.getenv("GOOGLE_OAUTH_ACCESS_TOKEN", "").strip()
    if access_token:
        from google.oauth2.credentials import Credentials

        credentials = Credentials(
            token=access_token,
            scopes=SCOPES,
        )
    else:
        raw = os.getenv("GCP_SERVICE_ACCOUNT_JSON", "").strip()
        if raw:
            try:
                info = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "GCP_SERVICE_ACCOUNT_JSON no contiene JSON válido."
                ) from exc
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


def _sheets_cache(sheets) -> dict[str, Any]:
    """Caché por cliente de Sheets; vive solo durante una ejecución del worker."""
    cache = getattr(sheets, "_terracore_cache", None)
    if not isinstance(cache, dict):
        cache = {"metadata": None, "sheets": {}}
        setattr(sheets, "_terracore_cache", cache)
    return cache


def _sheet_cache_entry(sheets, sheet_name: str) -> dict[str, Any]:
    cache = _sheets_cache(sheets)
    return cache["sheets"].setdefault(
        sheet_name,
        {"loaded": False, "headers": [], "rows": []},
    )


def _sheet_metadata(sheets) -> dict[str, dict[str, Any]]:
    cache = _sheets_cache(sheets)
    if cache.get("metadata") is not None:
        return cache["metadata"]
    response = _execute_sheets_request(
        sheets.spreadsheets().get(
            spreadsheetId=SPREADSHEET_ID,
            fields="sheets.properties(sheetId,title,gridProperties(rowCount,columnCount))",
        )
    )
    result = {}
    for item in response.get("sheets", []):
        props = item.get("properties", {})
        result[str(props.get("title", ""))] = props
    cache["metadata"] = result
    return result


def _read_sheet(sheets, sheet_name: str) -> tuple[list[str], list[list[Any]]]:
    """Lee una pestaña una sola vez por ejecución y reutiliza el resultado."""
    entry = _sheet_cache_entry(sheets, sheet_name)
    if not entry["loaded"]:
        response = _execute_sheets_request(
            sheets.spreadsheets().values().get(
                spreadsheetId=SPREADSHEET_ID,
                range=f"{_quote_sheet(sheet_name)}!A:AZ",
                valueRenderOption="FORMATTED_VALUE",
            )
        )
        values = response.get("values", []) or []
        entry["headers"] = list(values[0]) if values else []
        entry["rows"] = [list(row) for row in values[1:]] if values else []
        entry["loaded"] = True
    return list(entry["headers"]), [list(row) for row in entry["rows"]]


def _ensure_sheet_and_headers(sheets, sheet_name: str, required_headers: list[str]) -> list[str]:
    """Prepara la pestaña usando la caché para evitar lecturas y escrituras repetidas."""
    metadata = _sheet_metadata(sheets)
    if sheet_name not in metadata:
        _execute_sheets_request(
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
            )
        )
        _sheets_cache(sheets)["metadata"] = None
        metadata = _sheet_metadata(sheets)

    props = metadata[sheet_name]
    headers, _ = _read_sheet(sheets, sheet_name)
    entry = _sheet_cache_entry(sheets, sheet_name)

    if not any(str(x or "").strip() for x in headers):
        headers = list(required_headers)
        _execute_sheets_request(
            sheets.spreadsheets().values().update(
                spreadsheetId=SPREADSHEET_ID,
                range=f"{_quote_sheet(sheet_name)}!A1:{_column_letter(len(headers)-1)}1",
                valueInputOption="RAW",
                body={"values": [headers]},
            )
        )
        entry["headers"] = list(headers)
        entry["rows"] = []
        entry["loaded"] = True
        return headers

    existing = {_normalize_header(x) for x in headers if str(x or "").strip()}
    missing = [x for x in required_headers if _normalize_header(x) not in existing]
    if missing:
        start_index = len(headers)
        required_columns = start_index + len(missing)
        grid = props.setdefault("gridProperties", {})
        current_columns = int(grid.get("columnCount", 26))
        if required_columns > current_columns:
            _execute_sheets_request(
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
                )
            )
            grid["columnCount"] = required_columns

        updates = []
        for offset, header in enumerate(missing):
            col = _column_letter(start_index + offset)
            updates.append(
                {
                    "range": f"{_quote_sheet(sheet_name)}!{col}1",
                    "values": [[header]],
                }
            )
        _execute_sheets_request(
            sheets.spreadsheets().values().batchUpdate(
                spreadsheetId=SPREADSHEET_ID,
                body={"valueInputOption": "RAW", "data": updates},
            )
        )
        headers.extend(missing)
        entry["headers"] = list(headers)
    return headers

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


def _append_rows_once(
    sheets,
    sheet_name: str,
    records: list[dict[str, Any]],
    key_header: str,
) -> list[int]:
    """Agrega varios registros en una sola escritura y evita duplicados por clave."""
    if not records:
        return []

    required_headers = []
    for record in records:
        for header in record:
            if header not in required_headers:
                required_headers.append(header)

    headers = _ensure_sheet_and_headers(
        sheets,
        sheet_name,
        required_headers,
    )
    _, existing_rows = _read_sheet(sheets, sheet_name)
    key = _normalize_header(key_header)
    index_map = {
        _normalize_header(header): index
        for index, header in enumerate(headers)
        if str(header or "").strip()
    }
    key_index = index_map.get(key)
    if key_index is None:
        raise RuntimeError(
            f"No existe la columna clave '{key_header}' en '{sheet_name}'."
        )

    existing_positions = {}
    for physical_row, row in enumerate(existing_rows, start=2):
        value = row[key_index] if key_index < len(row) else ""
        value = str(value or "").strip()
        if value:
            existing_positions.setdefault(value, physical_row)

    pending_rows = []
    pending_positions = {}
    pending_record_indices = {}
    result_positions = [0] * len(records)

    for record_index, values in enumerate(records):
        key_value = str(values.get(key_header, "") or "").strip()
        if key_value and key_value in existing_positions:
            result_positions[record_index] = existing_positions[key_value]
            continue

        dedupe_key = key_value or f"__sin_clave_{record_index}"
        if dedupe_key in pending_positions:
            pending_record_indices[dedupe_key].append(record_index)
            continue

        row = [""] * len(headers)
        for name, value in values.items():
            column_index = index_map.get(_normalize_header(name))
            if column_index is not None:
                row[column_index] = value

        pending_positions[dedupe_key] = len(pending_rows)
        pending_record_indices[dedupe_key] = [record_index]
        pending_rows.append(row)

    if not pending_rows:
        return result_positions

    last_col = _column_letter(max(0, len(headers) - 1))
    response = _execute_sheets_request(
        sheets.spreadsheets().values().append(
            spreadsheetId=SPREADSHEET_ID,
            range=f"{_quote_sheet(sheet_name)}!A:{last_col}",
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body={"values": pending_rows},
        )
    )

    updated_range = response.get("updates", {}).get("updatedRange", "")
    match = re.search(r"![A-Z]+(\d+)", updated_range)
    entry = _sheet_cache_entry(sheets, sheet_name)

    if match:
        first_physical_row = int(match.group(1))
        for dedupe_key, pending_index in pending_positions.items():
            physical_row = first_physical_row + pending_index
            for record_index in pending_record_indices[dedupe_key]:
                result_positions[record_index] = physical_row

            cache_index = physical_row - 2
            if cache_index >= 0:
                while len(entry["rows"]) < cache_index:
                    entry["rows"].append([])
                row = pending_rows[pending_index]
                if cache_index == len(entry["rows"]):
                    entry["rows"].append(row)
                else:
                    entry["rows"][cache_index] = row

        entry["loaded"] = True
    else:
        entry["loaded"] = False

    return result_positions



def _json_safe(value: Any):
    """Convierte valores NumPy/Python a tipos compatibles con JSON."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, separators=(",", ":"))


def _record_value(record: dict[str, Any], name: str, default: Any = "") -> Any:
    return record.get(_normalize_header(name), default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return int(default)
        return int(round(float(value)))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return float(default)
        result = float(value)
        return result if np.isfinite(result) else float(default)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _truthy(value: Any) -> bool:
    return str(value or "").strip().casefold() in (
        "1", "true", "yes", "si", "sí", "verdadero", "x"
    )


def _parse_vacant_indices(value: Any) -> list[int]:
    if isinstance(value, (list, tuple, set)):
        result = []
        for item in value:
            try:
                result.append(int(item))
            except (TypeError, ValueError):
                continue
        return result
    text_value = str(value or "").strip()
    if not text_value:
        return []
    return [int(item) for item in re.findall(r"\d+", text_value)]


def _upsert_rows_by_key(
    sheets,
    sheet_name: str,
    records: list[dict[str, Any]],
    key_header: str,
) -> list[int]:
    """Inserta o actualiza un lote completo por clave con una sola escritura de actualización."""
    if not records:
        return []

    required_headers = []
    for record in records:
        for header in record:
            if header not in required_headers:
                required_headers.append(header)

    headers = _ensure_sheet_and_headers(sheets, sheet_name, required_headers)
    _, existing_rows = _read_sheet(sheets, sheet_name)
    index_map = {
        _normalize_header(header): index
        for index, header in enumerate(headers)
        if str(header or "").strip()
    }
    key_index = index_map.get(_normalize_header(key_header))
    if key_index is None:
        raise RuntimeError(
            f"No existe la columna clave '{key_header}' en '{sheet_name}'."
        )

    existing_positions = {}
    for physical_row, row in enumerate(existing_rows, start=2):
        value = row[key_index] if key_index < len(row) else ""
        value = str(value or "").strip()
        if value:
            existing_positions.setdefault(value, physical_row)

    update_data = []
    append_records = []
    result_positions = []
    last_col = _column_letter(max(0, len(headers) - 1))

    for record in records:
        key_value = str(record.get(key_header, "") or "").strip()
        physical_row = existing_positions.get(key_value) if key_value else None
        if physical_row:
            cache_index = physical_row - 2
            merged = list(existing_rows[cache_index]) if cache_index < len(existing_rows) else []
            while len(merged) < len(headers):
                merged.append("")
            for name, value in record.items():
                column_index = index_map.get(_normalize_header(name))
                if column_index is not None:
                    merged[column_index] = value
            update_data.append({
                "range": (
                    f"{_quote_sheet(sheet_name)}!A{physical_row}:"
                    f"{last_col}{physical_row}"
                ),
                "values": [merged],
            })
            existing_rows[cache_index] = merged
            result_positions.append(physical_row)
        else:
            append_records.append(record)
            result_positions.append(0)

    if update_data:
        _execute_sheets_request(
            sheets.spreadsheets().values().batchUpdate(
                spreadsheetId=SPREADSHEET_ID,
                body={"valueInputOption": "RAW", "data": update_data},
            )
        )
        entry = _sheet_cache_entry(sheets, sheet_name)
        entry["rows"] = existing_rows
        entry["loaded"] = True

    if append_records:
        appended_positions = _append_rows_once(
            sheets,
            sheet_name,
            append_records,
            key_header,
        )
        append_position_by_key = {}
        for record, position in zip(append_records, appended_positions):
            key_value = str(record.get(key_header, "") or "").strip()
            append_position_by_key.setdefault(key_value, position)
        for index, record in enumerate(records):
            if result_positions[index] == 0:
                key_value = str(record.get(key_header, "") or "").strip()
                result_positions[index] = append_position_by_key.get(key_value, 0)

    return result_positions


def _inventory_record_values(
    request_id: str,
    historial_id: str,
    parcela_id: str,
    parcela_name: str,
    photo_index: int,
    source_path: str,
    source_name: str,
    inventory_path: str,
    row: dict[str, Any],
    idioma: str,
    estado: str = "Pendiente",
) -> dict[str, Any]:
    row_id = int(row.get("id", 0) or 0)
    slots = max(0, int(row.get("slot_count", 0) or 0))
    vacant = _tg_clean_indices(row.get("vacant_indices", []), slots)
    occupied = max(0, slots - len(vacant))
    points = row.get("points_norm", []) or []
    start = points[0] if len(points) >= 2 else ["", ""]
    end = points[-1] if len(points) >= 2 else ["", ""]
    row_confidence = round(
        float(np.clip(row.get("confidence", 0.0) or 0.0, 0.0, 1.0)) * 100.0,
        2,
    )
    slot_confidence = round(
        float(np.clip(row.get("slot_confidence", 0.0) or 0.0, 0.0, 1.0)) * 100.0,
        2,
    )
    return {
        "RegistroID": _deterministic_id(
            f"{historial_id}|inventario|{row_id}"
        ),
        "SolicitudID": request_id,
        "AnalisisID": historial_id,
        "ParcelaID": parcela_id,
        "ParcelaNombre": parcela_name,
        "FotoIndex": int(photo_index),
        "FotoNombre": source_name,
        "FotoRuta": source_path,
        "ImagenInventarioAppSheet": inventory_path,
        "NumeroSurco": f"{row_id:02d}",
        "Slots": slots,
        "Ocupados": occupied,
        "Vacios": len(vacant),
        "VaciosIndices": ",".join(str(index) for index in vacant),
        "Confianza": slot_confidence,
        "ConfianzaSurco": row_confidence,
        "ConfianzaSlots": slot_confidence,
        "PuntoInicioX": round(_safe_float(start[0], 0.0), 3) if start[0] != "" else "",
        "PuntoInicioY": round(_safe_float(start[1], 0.0), 3) if start[1] != "" else "",
        "PuntoFinX": round(_safe_float(end[0], 0.0), 3) if end[0] != "" else "",
        "PuntoFinY": round(_safe_float(end[1], 0.0), 3) if end[1] != "" else "",
        "RowSpacingPx": round(_safe_float(row.get("row_spacing_px", 0.0)), 3),
        "SlotPitchPx": round(_safe_float(row.get("slot_pitch_px", 0.0)), 3),
        "GeometriaJSON": _json_dumps(row),
        "ManualPosition": bool(row.get("manual_position", False)),
        "ManualInventory": bool(row.get("manual_inventory", False)),
        "EstadoInventario": estado,
        "FechaValidacion": _now_iso() if estado == "Validado" else "",
        "Idioma": idioma,
        "Fecha": _now_iso(),
    }


def _apply_inventory_record_edits(record: dict[str, Any]) -> dict[str, Any]:
    """Replica la sincronización de la tabla editable de Streamlit sobre la geometría real."""
    geometry_raw = _record_value(record, "GeometriaJSON", "")
    try:
        geometry = json.loads(geometry_raw) if isinstance(geometry_raw, str) else geometry_raw
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("GeometriaJSON no contiene una geometría válida.") from exc
    if not isinstance(geometry, dict):
        raise ValueError("Falta la geometría original del surco.")

    row = dict(geometry)
    original_points = row.get("points_norm", []) or []
    if len(original_points) < 2:
        raise ValueError("El surco no contiene dos extremos de geometría.")

    row_id = _safe_int(_record_value(record, "NumeroSurco", row.get("id", 0)), 0)
    if row_id <= 0:
        raise ValueError("NumeroSurco debe ser mayor que cero.")

    slots = _safe_int(_record_value(record, "Slots", row.get("slot_count", 0)), 0)
    original_slots = _safe_int(geometry.get("slot_count", 0), 0)
    original_vacant = _tg_clean_indices(
        geometry.get("vacant_indices", []),
        max(0, original_slots),
    )
    default_vacant_count = len(original_vacant)
    vacant_count = _safe_int(
        _record_value(record, "Vacios", default_vacant_count),
        default_vacant_count,
    )
    occupied_count = _safe_int(
        _record_value(
            record,
            "Ocupados",
            max(0, slots - vacant_count),
        ),
        max(0, slots - vacant_count),
    )

    if slots < 0 or vacant_count < 0 or occupied_count < 0:
        raise ValueError(f"Surco {row_id:02d}: Slots, Ocupados y Vacíos no pueden ser negativos.")
    if slots != occupied_count + vacant_count:
        raise ValueError(
            f"Surco {row_id:02d}: Slots ({slots}) debe ser igual a "
            f"Ocupados ({occupied_count}) + Vacíos ({vacant_count})."
        )

    indices_raw = _record_value(record, "VaciosIndices", "")
    explicit_indices = _parse_vacant_indices(indices_raw)
    # Si el texto sigue igual que el valor automático original, la edición
    # corresponde a la tabla de conteos de Streamlit: ajustar índices existentes
    # y completar desde el final cuando cambie la cantidad de Vacíos.
    has_explicit_indices = (
        bool(str(indices_raw or "").strip())
        and explicit_indices != original_vacant
    )

    if has_explicit_indices:
        if len(explicit_indices) != vacant_count:
            raise ValueError(
                f"Surco {row_id:02d}: VaciosIndices debe contener exactamente "
                f"{vacant_count} número(s)."
            )
        if len(set(explicit_indices)) != len(explicit_indices):
            raise ValueError(f"Surco {row_id:02d}: VaciosIndices contiene números duplicados.")
        if any(index < 1 or index > slots for index in explicit_indices):
            raise ValueError(
                f"Surco {row_id:02d}: cada índice vacío debe estar entre 1 y {slots}."
            )
        vacant_indices = sorted(explicit_indices)
    else:
        vacant_indices = list(original_vacant)
        vacant_indices = [index for index in vacant_indices if index <= slots][:vacant_count]
        if len(vacant_indices) < vacant_count:
            available = [
                index for index in range(slots, 0, -1)
                if index not in set(vacant_indices)
            ]
            vacant_indices.extend(available[:vacant_count - len(vacant_indices)])
        vacant_indices = sorted(set(vacant_indices))[:vacant_count]

    row["id"] = row_id
    row["slot_count"] = int(slots)
    row["vacant_indices"] = vacant_indices

    endpoint_names = (
        "PuntoInicioX", "PuntoInicioY", "PuntoFinX", "PuntoFinY"
    )
    endpoint_values = [_record_value(record, name, "") for name in endpoint_names]
    endpoint_present = [str(value or "").strip() != "" for value in endpoint_values]
    if any(endpoint_present) and not all(endpoint_present):
        raise ValueError(
            f"Surco {row_id:02d}: completa las cuatro coordenadas para mover o girar la línea."
        )

    original_start = [float(original_points[0][0]), float(original_points[0][1])]
    original_end = [float(original_points[-1][0]), float(original_points[-1][1])]
    if all(endpoint_present):
        start_x, start_y, end_x, end_y = [
            _safe_float(value, 0.0) for value in endpoint_values
        ]
        coords = (start_x, start_y, end_x, end_y)
        if any(value < 0.0 or value > 1000.0 for value in coords):
            raise ValueError(
                f"Surco {row_id:02d}: las coordenadas deben estar entre 0 y 1000."
            )
        new_start = [start_x, start_y]
        new_end = [end_x, end_y]
        moved = any(
            abs(new_value - old_value) > 0.01
            for new_point, old_point in zip((new_start, new_end), (original_start, original_end))
            for new_value, old_value in zip(new_point, old_point)
        )
        if moved:
            point_count = max(2, len(original_points))
            row["points_norm"] = [
                [
                    float(new_start[0] * (1.0 - t) + new_end[0] * t),
                    float(new_start[1] * (1.0 - t) + new_end[1] * t),
                ]
                for t in np.linspace(0.0, 1.0, point_count)
            ]
            row["manual_position"] = True

    row_spacing = _safe_float(
        _record_value(record, "RowSpacingPx", row.get("row_spacing_px", 18.0)),
        _safe_float(row.get("row_spacing_px", 18.0), 18.0),
    )
    slot_pitch = _safe_float(
        _record_value(record, "SlotPitchPx", row.get("slot_pitch_px", 0.0)),
        _safe_float(row.get("slot_pitch_px", 0.0), 0.0),
    )
    if row_spacing > 0:
        row["row_spacing_px"] = row_spacing
    if slot_pitch >= 0:
        row["slot_pitch_px"] = slot_pitch

    row["manual_inventory"] = bool(
        row.get("manual_inventory", False)
        or slots != _safe_int(geometry.get("slot_count", 0), 0)
        or vacant_indices != _tg_clean_indices(
            geometry.get("vacant_indices", []),
            _safe_int(geometry.get("slot_count", 0), 0),
        )
    )
    return row


def _inventory_records_for_request(sheets, request_id: str) -> list[dict[str, Any]]:
    headers, rows = _read_sheet(sheets, INVENTORY_SHEET)
    if not headers:
        raise RuntimeError(
            f"No existe la pestaña {INVENTORY_SHEET}. Ejecuta primero la etapa de Inventario."
        )
    result = []
    for row in rows:
        data = _row_as_dict(headers, row)
        if str(data.get(_normalize_header("SolicitudID"), "") or "").strip() == request_id:
            result.append(data)
    result.sort(
        key=lambda item: (
            _safe_int(item.get(_normalize_header("FotoIndex"), 0), 0),
            _safe_int(item.get(_normalize_header("NumeroSurco"), 0), 0),
        )
    )
    return result


def _inventory_rows_for_photo(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [_apply_inventory_record_edits(record) for record in records]
    rows.sort(key=lambda item: int(item.get("id", 0) or 0))
    ids = [int(row.get("id", 0) or 0) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("El Inventario contiene números de surco duplicados para una fotografía.")
    if not rows:
        raise ValueError("No hay surcos de Inventario para esta fotografía.")
    return rows


def _append_row_once(
    sheets,
    sheet_name: str,
    values: dict[str, Any],
    key_header: str,
    key_value: str,
) -> int:
    """Compatibilidad para registros individuales."""
    record = dict(values)
    record[key_header] = key_value
    positions = _append_rows_once(
        sheets,
        sheet_name,
        [record],
        key_header,
    )
    return positions[0] if positions else 0


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
    indexed_values = []
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
        indexed_values.append((index, value))

    if updates:
        _execute_sheets_request(
            sheets.spreadsheets().values().batchUpdate(
                spreadsheetId=SPREADSHEET_ID,
                body={"valueInputOption": "RAW", "data": updates},
            )
        )

        entry = _sheet_cache_entry(sheets, sheet_name)
        cache_index = physical_row - 2
        if cache_index >= 0:
            while len(entry["rows"]) <= cache_index:
                entry["rows"].append([])
            cached_row = entry["rows"][cache_index]
            for index, value in indexed_values:
                while len(cached_row) <= index:
                    cached_row.append("")
                cached_row[index] = value

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


def _photo_paths(row: dict[str, Any]) -> list[str]:
    """Acepta EnumList Fotos o hasta cinco columnas Image individuales."""
    paths = _parse_image_list(row.get(_normalize_header("Fotos"), ""))
    for number in range(1, 6):
        value = row.get(_normalize_header(f"Foto{number:02d}"), "")
        for path in _parse_image_list(value):
            if path not in paths:
                paths.append(path)
    return paths


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
    inventory_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    uploaded = UploadedBytes(source_name, source_mime, raw)

    # El motor reutiliza el idioma elegido en el formulario AppSheet.
    os.environ["TERRACORE_LANGUAGE"] = (
        "FR" if str(idioma or "ES").strip().upper() == "FR" else "ES"
    )
    # En la segunda fase recibimos el Inventario validado por el usuario.
    # No volvemos a detectar ni sustituimos sus correcciones manuales.
    inventory = (
        inventory_override
        if inventory_override is not None
        else _ocv2_inventory_from_file(uploaded)
    )
    rows = inventory.get("rows", []) or []
    if not rows:
        raise RuntimeError("No hay geometría de Inventario confirmada para esta fotografía.")
    backend_result = _ocv2_analyze_health(uploaded, rows)

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
    historial_id = _analysis_id_for_request(sheets, request_id, photo_index)
    regions = _build_regions(uploaded, backend_result, rows, historial_id)
    now = _now_iso()
    parcel_safe = _safe_name(parcela_name or parcela_id or "Parcela")
    request_safe = _safe_name(request_id)
    base_name = f"{parcel_safe}_{request_safe}_{historial_id}_F{photo_index:02d}"

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

    # Actualizar el mismo lote que el usuario corrigió en AppSheet.
    # La clave RegistroID se conserva entre Inventario y análisis de Salud.
    inventory_records = [
        _inventory_record_values(
            request_id=request_id,
            historial_id=historial_id,
            parcela_id=parcela_id,
            parcela_name=parcela_name,
            photo_index=photo_index,
            source_path=source_path,
            source_name=source_name,
            inventory_path=inventory_path,
            row=row,
            idioma=idioma,
            estado="Validado",
        )
        for row in rows
    ]
    _upsert_rows_by_key(
        sheets,
        INVENTORY_SHEET,
        inventory_records,
        "RegistroID",
    )

    # Evidencias: originales, inventario, análisis y polígonos; un solo lote.
    evidence_items = [
        ("ORIGINAL", original_name, original_id, source_mime),
        ("INVENTARIO", inventory_name, inventory_id, "image/png"),
        ("ANALISIS", processed_name, processed_id, "image/png"),
    ]
    if polygon_id:
        evidence_items.append(("POLIGONOS", polygon_name, polygon_id, "image/png"))

    evidence_records = []
    for evidence_type, filename, file_id, mime_type in evidence_items:
        evidence_id = _deterministic_id(
            f"{historial_id}|evidencia|{evidence_type}"
        )
        evidence_records.append(
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
            }
        )
    _append_rows_once(
        sheets,
        "Evidencias",
        evidence_records,
        "EvidenciaID",
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

    critical_region_records = []
    damaged_zone_records = []
    for index, region in enumerate(regions, 1):
        polygon_json = json.dumps(
            {"space": "image_px", "points": region["points"]},
            ensure_ascii=False,
        )
        area_pct = float(region["area_visual_pct"])
        region_id = region["region_id"]
        region_name = f"Región crítica {index}"
        segment = f"Slots {region['primer_slot']}-{region['ultimo_slot']}"
        critical_region_records.append(
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
            }
        )

        zone_id = _deterministic_id(f"{historial_id}|zona|{index}")
        damaged_zone_records.append(
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
            }
        )

    _append_rows_once(
        sheets,
        "RegionesCriticas",
        critical_region_records,
        "RegionID",
    )
    _append_rows_once(
        sheets,
        "ZonasDañadas",
        damaged_zone_records,
        "ZonaID",
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


def _save_video_evidence(
    sheets,
    request_id: str,
    history_ids: list[str],
    parcela_name: str,
    idioma: str,
    video_path: str,
    simulation_video_path: str,
) -> int:
    """Asocia los videos ya cargados por AppSheet con cada historial generado.

    AppSheet almacena los archivos en la carpeta de la aplicación de Drive.
    Aquí guardamos sus rutas en HistorialTerroCore y Evidencias1 sin volver
    a descargar ni duplicar archivos de video pesados.
    """
    video_path = str(video_path or "").strip()
    simulation_video_path = str(simulation_video_path or "").strip()
    if not video_path and not simulation_video_path:
        return 0

    evidence_records = []
    for historial_id in history_ids:
        history_row, _, _ = _find_row_by_key(
            sheets,
            HISTORIAL_SHEET,
            "ID",
            historial_id,
        )
        if history_row:
            _update_row_fields(
                sheets,
                HISTORIAL_SHEET,
                history_row,
                {
                    "Video": video_path,
                    "Video simulación": simulation_video_path,
                },
            )

        evidence_id = _deterministic_id(
            f"{request_id}|video-evidencia|{historial_id}"
        )
        evidence_records.append(
            {
                "UNIQUEID": evidence_id,
                "NOMBRE": parcela_name,
                "DESCRIPCION": f"Videos de solicitud AppSheet {request_id}",
                "Video": video_path,
                "Video simulación ": simulation_video_path,
                "Idioma ": idioma,
                "HistorialID": historial_id,
            }
        )

    _append_rows_once(
        sheets,
        "Evidencias1",
        evidence_records,
        "UNIQUEID",
    )
    return len(evidence_records)


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
        _ensure_sheet_and_headers(sheets, REQUEST_SHEET, REQUEST_HEADERS)
        row_number, row = _request_row(sheets, request_id)
        status = str(row.get(_normalize_header("Estado"), "") or "").strip().casefold()
        inventory_validated = (
            _truthy(row.get(_normalize_header("ValidacionInventario"), ""))
            or status in ("validado", "validada")
        )

        if status in ("completado", "completed"):
            return {
                "ok": True,
                "status": "Completado",
                "request_id": request_id,
                "mensaje": "Esta solicitud ya fue procesada.",
            }
        if status in (
            "en cola",
            "procesando",
            "procesandoinventario",
            "procesandosalud",
        ):
            return {
                "ok": True,
                "status": status.title(),
                "request_id": request_id,
                "mensaje": "La solicitud ya está en proceso.",
            }
        if status in ("pendientevalidacion", "pendiente de validacion") and not inventory_validated:
            return {
                "ok": True,
                "status": "PendienteValidacion",
                "request_id": request_id,
                "mensaje": "Confirma el Inventario en AppSheet antes de continuar con Salud.",
            }

        photo_paths = _photo_paths(row)
        if not photo_paths:
            raise HTTPException(
                status_code=400,
                detail="Agrega una fotografía en Foto01–Foto05 o en Fotos.",
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



def _process_inventory_stage(
    drive,
    sheets,
    request_id: str,
    row_number: int,
    photo_paths: list[str],
    parcela_id: str,
    parcela_name: str,
    idioma: str,
    fecha_captura: str,
) -> dict[str, Any]:
    """Fase 1: ejecuta el inventario original y lo deja editable en AppSheet."""
    os.environ["TERRACORE_LANGUAGE"] = (
        "FR" if str(idioma or "ES").strip().upper() == "FR" else "ES"
    )
    inventory_records = []
    photo_summaries = []
    errors = []
    parcel_safe = _safe_name(parcela_name or parcela_id or "Parcela")
    request_safe = _safe_name(request_id)

    for photo_index, source_path in enumerate(photo_paths, 1):
        try:
            raw, source_name, source_mime = _resolve_appsheet_file(drive, source_path)
            uploaded = UploadedBytes(source_name, source_mime, raw)
            # MISMA función de inventario que utiliza Streamlit.
            inventory = _ocv2_inventory_from_file(uploaded)
            rows = inventory.get("rows", []) or []
            if not rows:
                raise RuntimeError("OpenCV no generó surcos para esta fotografía.")

            historial_id = _analysis_id_for_request(sheets, request_id, photo_index)
            inventory_name = (
                f"{parcel_safe}_{request_safe}_{historial_id}_F{photo_index:02d}_"
                "inventario_pendiente.png"
            )
            inventory_bytes = _image_to_png_bytes(inventory.get("image"))
            if not inventory_bytes:
                raise RuntimeError("No se pudo generar la imagen del Inventario.")
            _, inventory_path = _upload_artifact(
                drive,
                inventory_bytes,
                inventory_name,
                "image/png",
                "Inventarios",
            )

            for inventory_row in rows:
                inventory_records.append(
                    _inventory_record_values(
                        request_id=request_id,
                        historial_id=historial_id,
                        parcela_id=parcela_id,
                        parcela_name=parcela_name,
                        photo_index=photo_index,
                        source_path=source_path,
                        source_name=source_name,
                        inventory_path=inventory_path,
                        row=inventory_row,
                        idioma=idioma,
                        estado="Pendiente",
                    )
                )

            total_slots = sum(
                int(item.get("slot_count", 0) or 0) for item in rows
            )
            vacant_slots = sum(
                len(
                    _tg_clean_indices(
                        item.get("vacant_indices", []),
                        int(item.get("slot_count", 0) or 0),
                    )
                )
                for item in rows
            )
            photo_summaries.append(
                {
                    "foto_index": photo_index,
                    "foto": source_name,
                    "ruta": source_path,
                    "analisis_id": historial_id,
                    "surcos": len(rows),
                    "slots": total_slots,
                    "ocupados": max(0, total_slots - vacant_slots),
                    "vacios": vacant_slots,
                    "confianza_inventario": round(
                        float(inventory.get("confidence", 0.0) or 0.0) * 100.0,
                        2,
                    ),
                    "imagen_inventario": inventory_path,
                    "estado": "Pendiente de validación",
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "foto_index": photo_index,
                    "foto": str(source_path),
                    "error": str(exc)[:1000],
                }
            )

    if not inventory_records:
        detail = " | ".join(item["error"] for item in errors)
        raise RuntimeError(
            detail or "No se pudo generar el Inventario de ninguna fotografía."
        )

    # Un solo lote por pestaña: conserva las claves y evita duplicados en reintentos.
    _upsert_rows_by_key(
        sheets,
        INVENTORY_SHEET,
        inventory_records,
        "RegistroID",
    )

    now = _now_iso()
    inventory_summary = {
        "fotos": photo_summaries,
        "errores": errors,
        "instrucciones": (
            "Revisa los surcos de cada fotografía en InventarioSurcos. "
            "Corrige Slots, Ocupados, Vacios, VaciosIndices y, si hace falta, "
            "las cuatro coordenadas de los extremos. Confirma solo cuando "
            "Slots = Ocupados + Vacios en todos los surcos."
        ),
    }
    _update_row_fields(
        sheets,
        REQUEST_SHEET,
        row_number,
        {
            "Estado": "PendienteValidacion",
            "FechaInventario": now,
            "FechaFin": "",
            "TotalFotos": len(photo_summaries),
            "ResumenInventario": _json_dumps(inventory_summary),
            "ValidacionInventario": "Pendiente",
            "Error": "",
        },
    )

    return {
        "ok": True,
        "status": "PendienteValidacion",
        "request_id": request_id,
        "total_fotos": len(photo_summaries),
        "total_surcos": len(inventory_records),
        "inventario": inventory_summary,
        "mensaje": (
            "Inventario generado. Revisa y corrige los registros de InventarioSurcos "
            "en AppSheet y después confirma la validación."
        ),
    }


def _process_health_stage(
    drive,
    sheets,
    request_id: str,
    row_number: int,
    row: dict[str, Any],
    photo_paths: list[str],
    parcela_id: str,
    parcela_name: str,
    idioma: str,
    fecha_captura: str,
    surco_id: str,
) -> dict[str, Any]:
    """Fase 2: analiza Salud únicamente con el Inventario editado y confirmado."""
    os.environ["TERRACORE_LANGUAGE"] = (
        "FR" if str(idioma or "ES").strip().upper() == "FR" else "ES"
    )
    inventory_records = _inventory_records_for_request(sheets, request_id)
    records_by_photo: dict[int, list[dict[str, Any]]] = {}
    for record in inventory_records:
        photo_index = _safe_int(
            record.get(_normalize_header("FotoIndex"), 0),
            0,
        )
        if photo_index > 0:
            records_by_photo.setdefault(photo_index, []).append(record)

    history_ids = _parse_image_list(
        row.get(_normalize_header("HistorialIDs"), "")
    )
    summaries = []
    errors = []

    for photo_index, source_path in enumerate(photo_paths, 1):
        source_name = str(source_path)
        photo_records = records_by_photo.get(photo_index, [])
        if not photo_records:
            errors.append(
                f"Fotografía {photo_index}: no hay Inventario validado en {INVENTORY_SHEET}."
            )
            continue

        try:
            # Se aplican las correcciones guardadas en las columnas editables.
            rows = _inventory_rows_for_photo(photo_records)
            raw, source_name, source_mime = _resolve_appsheet_file(drive, source_path)
            uploaded = UploadedBytes(source_name, source_mime, raw)

            # Streamlit utiliza la fotografía original para volver a dibujar las
            # líneas movidas. Aquí se hace lo mismo sin volver a ejecutar detección.
            original_pil = Image.open(io.BytesIO(raw)).convert("RGB")
            inventory_image = _ocv2_render_inventory_with_rows(original_pil, rows)
            inventory_confidence = (
                float(np.mean([
                    float(item.get("slot_confidence", 0.0) or 0.0)
                    for item in rows
                ]))
                if rows else 0.0
            )
            inventory_override = {
                "rows": rows,
                "image": inventory_image,
                "count": len(rows),
                "confidence": inventory_confidence,
            }

            # MISMA función de Salud que utiliza Streamlit:
            # _tc_analyze_health_openai -> _ocv2_analyze_health.
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
                inventory_override=inventory_override,
            )
            if summary["historial_id"] not in history_ids:
                history_ids.append(summary["historial_id"])
            summaries.append(summary)

            _update_row_fields(
                sheets,
                REQUEST_SHEET,
                row_number,
                {
                    "HistorialIDs": _json_dumps(history_ids),
                    "TotalFotos": len(history_ids),
                    "ResumenResultado": _json_dumps(summaries),
                },
            )
        except Exception as exc:
            errors.append(f"{source_name}: {exc}")

    if not summaries:
        raise RuntimeError(
            "No se pudo completar el análisis de Salud con el Inventario validado. "
            + " | ".join(errors)
        )

    video_evidence_rows = _save_video_evidence(
        sheets=sheets,
        request_id=request_id,
        history_ids=history_ids,
        parcela_name=parcela_name,
        idioma=idioma,
        video_path=str(row.get(_normalize_header("Video"), "") or "").strip(),
        simulation_video_path=str(
            row.get(_normalize_header("Video simulación"), "") or ""
        ).strip(),
    )

    now = _now_iso()
    _update_row_fields(
        sheets,
        REQUEST_SHEET,
        row_number,
        {
            "Estado": "Completado",
            "FechaValidacionInventario": now,
            "FechaFin": now,
            "ValidacionInventario": "Validado",
            "HistorialIDs": _json_dumps(history_ids),
            "TotalFotos": len(history_ids),
            "ResumenResultado": _json_dumps(summaries),
            "Error": "",
        },
    )

    return {
        "ok": True,
        "status": "Completado",
        "request_id": request_id,
        "historial_ids": history_ids,
        "total_fotos": len(history_ids),
        "filas_evidencia_video": video_evidence_rows,
        "resultados": summaries,
        "advertencias": errors,
    }


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
        _ensure_sheet_and_headers(sheets, REQUEST_SHEET, REQUEST_HEADERS)
        row_number, row = _request_row(sheets, request_id)

        current_status = str(
            row.get(_normalize_header("Estado"), "") or ""
        ).strip().casefold()
        validation_value = row.get(
            _normalize_header("ValidacionInventario"),
            "",
        )
        inventory_validated = (
            _truthy(validation_value)
            or current_status in ("validado", "validada")
        )

        if current_status in ("completado", "completed"):
            return {
                "ok": True,
                "status": "Completado",
                "request_id": request_id,
                "mensaje": "La solicitud ya estaba terminada.",
            }

        if current_status in (
            "procesando",
            "procesandoinventario",
            "procesandosalud",
        ):
            return {
                "ok": True,
                "status": current_status,
                "request_id": request_id,
                "mensaje": "La solicitud ya está en proceso.",
            }

        if (
            current_status in ("pendientevalidacion", "pendiente de validacion")
            and not inventory_validated
        ):
            return {
                "ok": True,
                "status": "PendienteValidacion",
                "request_id": request_id,
                "mensaje": (
                    "El Inventario está esperando revisión y confirmación en AppSheet."
                ),
            }

        photo_paths = _photo_paths(row)
        if not photo_paths:
            raise ValueError("No hay fotografías registradas en la solicitud.")

        parcela_id = str(
            row.get(_normalize_header("ParcelaID"), "") or ""
        ).strip()
        if not parcela_id:
            raise ValueError("Selecciona una parcela en la columna ParcelaID.")

        parcela_name = str(
            row.get(_normalize_header("ParcelaNombre"), "") or ""
        ).strip()
        if not parcela_name:
            _, parcel_row, _ = _find_row_by_key(
                sheets,
                "Parcelas",
                "ID",
                parcela_id,
            )
            if parcel_row:
                parcela_name = str(
                    parcel_row.get(_normalize_header("NombreParcela"), "") or ""
                ).strip()
        if not parcela_name:
            parcela_name = parcela_id

        idioma = str(
            row.get(_normalize_header("Idioma"), "ES") or "ES"
        ).strip().upper()
        idioma = "FR" if idioma == "FR" else "ES"
        fecha_captura = str(
            row.get(_normalize_header("FechaSolicitud"), "") or ""
        ).strip()
        surco_id = str(
            row.get(_normalize_header("SurcoID"), "") or ""
        ).strip()

        if inventory_validated:
            _update_row_fields(
                sheets,
                REQUEST_SHEET,
                row_number,
                {
                    "Estado": "ProcesandoSalud",
                    "FechaInicioSalud": _now_iso(),
                    "FechaFin": "",
                    "Error": "",
                },
            )
            return _process_health_stage(
                drive=drive,
                sheets=sheets,
                request_id=request_id,
                row_number=row_number,
                row=row,
                photo_paths=photo_paths,
                parcela_id=parcela_id,
                parcela_name=parcela_name,
                idioma=idioma,
                fecha_captura=fecha_captura,
                surco_id=surco_id,
            )

        _update_row_fields(
            sheets,
            REQUEST_SHEET,
            row_number,
            {
                "Estado": "ProcesandoInventario",
                "FechaInicio": _now_iso(),
                "FechaFin": "",
                "ValidacionInventario": "Pendiente",
                "Error": "",
            },
        )
        return _process_inventory_stage(
            drive=drive,
            sheets=sheets,
            request_id=request_id,
            row_number=row_number,
            photo_paths=photo_paths,
            parcela_id=parcela_id,
            parcela_name=parcela_name,
            idioma=idioma,
            fecha_captura=fecha_captura,
        )

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

        # No se repite automáticamente: el usuario corrige el dato y vuelve a activar el Bot.
        return {
            "ok": False,
            "status": "Error",
            "request_id": request_id,
            "error": str(exc),
        }
