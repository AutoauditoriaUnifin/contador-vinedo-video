"""Ejecutor de análisis AppSheet para GitHub Actions, sin servidor permanente."""

from __future__ import annotations

import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _apps_script_post(payload: dict) -> dict:
    url = os.getenv("APPS_SCRIPT_WEBAPP_URL", "").strip()
    secret = os.getenv("APPS_SCRIPT_SHARED_SECRET", "").strip()
    if not url:
        raise RuntimeError("Falta el secreto APPS_SCRIPT_WEBAPP_URL.")
    if not secret:
        raise RuntimeError("Falta el secreto APPS_SCRIPT_SHARED_SECRET.")

    body = dict(payload)
    body["secret"] = secret
    request = Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=90) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Apps Script respondió HTTP {exc.code}: {detail[:1000]}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"No fue posible contactar Apps Script: {exc}") from exc

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Apps Script no devolvió JSON válido: {raw[:500]}"
        ) from exc

    if not isinstance(result, dict) or not result.get("ok"):
        raise RuntimeError(
            str(result.get("error", "Respuesta no válida de Apps Script"))
            if isinstance(result, dict)
            else "Respuesta no válida de Apps Script"
        )
    return result


def _mark_error(request_id: str, message: str) -> None:
    try:
        _apps_script_post(
            {
                "action": "mark_error",
                "request_id": request_id,
                "error": message[:3500],
            }
        )
    except Exception as exc:
        print(
            f"No se pudo actualizar el estado de error en Sheets: {exc}",
            file=sys.stderr,
        )


def main() -> int:
    request_id = (
        os.getenv("REQUEST_ID", "").strip()
        or (sys.argv[1].strip() if len(sys.argv) > 1 else "")
    )
    if not request_id:
        print("Uso: python run_appsheet_job.py SOLICITUD_ID", file=sys.stderr)
        return 2

    try:
        # El token OAuth se solicita al iniciar el job y solo vive en memoria.
        auth = _apps_script_post({"action": "get_google_token"})
        access_token = str(
            auth.get("access_token") or auth.get("token") or ""
        ).strip()
        if not access_token:
            raise RuntimeError("Apps Script no entregó un token OAuth.")

        shared_secret = os.getenv("APPS_SCRIPT_SHARED_SECRET", "").strip()
        os.environ["GOOGLE_OAUTH_ACCESS_TOKEN"] = access_token
        os.environ["APPSHEET_API_TOKEN"] = shared_secret

        # Importar después de preparar las variables que lee el módulo.
        from appsheet_integration import appsheet_process_analysis

        result = appsheet_process_analysis(
            {"request_id": request_id},
            x_worker_token=shared_secret,
        )
        if not result.get("ok"):
            raise RuntimeError(
                str(result.get("error") or "El motor TerraCore no completó el análisis.")
            )

        print(
            json.dumps(
                {
                    "ok": True,
                    "request_id": request_id,
                    "status": result.get("status", "Completado"),
                    "total_fotos": result.get("total_fotos", 0),
                    "filas_evidencia_video": result.get("filas_evidencia_video", 0),
                    "historial_ids": result.get("historial_ids", []),
                },
                ensure_ascii=False,
            )
        )
        return 0

    except Exception as exc:
        message = str(exc)
        print(f"Error en solicitud {request_id}: {message}", file=sys.stderr)
        _mark_error(request_id, message)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
