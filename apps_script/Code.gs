/**
 * Puente gratuito TerraCore: AppSheet -> Apps Script -> GitHub Actions.
 *
 * Configura estos Script Properties antes de desplegar:
 * GSHEET_ID
 * TERRACORE_SHARED_SECRET
 * GITHUB_DISPATCH_TOKEN
 * GITHUB_OWNER
 * GITHUB_REPO
 * GITHUB_WORKFLOW_FILE (opcional; default process-appsheet-analysis.yml)
 * GITHUB_BRANCH (opcional; default main)
 */

const TC_REQUEST_SHEET = "SolicitudesAnalisis";
const TC_REQUEST_HEADERS = [
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
  "ResumenInventario"
];

const TC_INVENTORY_SHEET = "InventarioSurcos";
const TC_INVENTORY_HEADERS = [
  "RegistroID", "SolicitudID", "AnalisisID", "ParcelaID", "ParcelaNombre",
  "FotoIndex", "FotoNombre", "FotoRuta", "ImagenInventarioAppSheet",
  "NumeroSurco", "Slots", "Ocupados", "Vacios", "VaciosIndices",
  "Confianza", "ConfianzaSurco", "ConfianzaSlots",
  "PuntoInicioX", "PuntoInicioY", "PuntoFinX", "PuntoFinY",
  "RowSpacingPx", "SlotPitchPx", "GeometriaJSON",
  "ManualPosition", "ManualInventory", "EstadoInventario",
  "FechaValidacion", "Idioma", "Fecha"
];

function doPost(e) {
  try {
    const payload = JSON.parse(
      e && e.postData && e.postData.contents
        ? e.postData.contents
        : "{}"
    );
    tcVerifySecret_(payload.secret);

    const action = String(payload.action || "").trim().toLowerCase();

    if (action === "setup") {
      return tcJson_(setupTerraCore());
    }

    if (action === "trigger") {
      return tcJson_(tcTriggerAnalysis_(payload.request_id));
    }

    if (action === "get_google_token") {
      return tcJson_({
        ok: true,
        access_token: ScriptApp.getOAuthToken()
      });
    }

    if (action === "mark_error") {
      return tcJson_(
        tcMarkError_(payload.request_id, payload.error || "Error no especificado")
      );
    }

    return tcJson_({
      ok: false,
      error: "Acción no reconocida."
    });
  } catch (error) {
    return tcJson_({
      ok: false,
      error: String(error && error.message ? error.message : error)
    });
  }
}

/**
 * Ejecuta manualmente desde el editor de Apps Script una sola vez.
 * Crea la pestaña o agrega los encabezados que hagan falta.
 */
function setupTerraCore() {
  const props = PropertiesService.getScriptProperties();
  const spreadsheetId = String(props.getProperty("GSHEET_ID") || "").trim();
  if (!spreadsheetId) {
    throw new Error("Configura GSHEET_ID en las propiedades del script.");
  }

  const ss = SpreadsheetApp.openById(spreadsheetId);
  tcEnsureSheetHeaders_(ss, TC_REQUEST_SHEET, TC_REQUEST_HEADERS);
  tcEnsureSheetHeaders_(ss, TC_INVENTORY_SHEET, TC_INVENTORY_HEADERS);

  return {
    ok: true,
    sheets: [TC_REQUEST_SHEET, TC_INVENTORY_SHEET],
    message: "SolicitudesAnalisis e InventarioSurcos quedaron preparadas en el mismo Google Sheets.",
    request_headers: TC_REQUEST_HEADERS,
    inventory_headers: TC_INVENTORY_HEADERS
  };
}

function tcEnsureSheetHeaders_(ss, sheetName, requiredHeaders) {
  let sheet = ss.getSheetByName(sheetName);
  if (!sheet) sheet = ss.insertSheet(sheetName);

  const lastColumn = Math.max(sheet.getLastColumn(), 1);
  const existing = sheet.getLastRow()
    ? sheet.getRange(1, 1, 1, lastColumn).getDisplayValues()[0]
    : [];
  const normalized = existing.map(tcNormalize_);

  if (!existing.length || existing.every(value => !String(value || "").trim())) {
    sheet.getRange(1, 1, 1, requiredHeaders.length).setValues([requiredHeaders]);
    sheet.setFrozenRows(1);
    return;
  }

  const missing = requiredHeaders.filter(
    header => !normalized.includes(tcNormalize_(header))
  );
  if (missing.length) {
    sheet.getRange(1, lastColumn + 1, 1, missing.length).setValues([missing]);
  }
  sheet.setFrozenRows(1);
}

function tcTriggerAnalysis_(requestId) {
  const id = String(requestId || "").trim();
  if (!id) throw new Error("Falta request_id.");

  const sheet = tcGetRequestSheet_();
  const found = tcFindRequest_(sheet, id);
  const row = found.values;
  const headers = found.headers;

  const status = String(row[tcCol_(headers, "Estado")] || "")
    .trim()
    .toLowerCase();

  if (["completado", "completed"].includes(status)) {
    return {
      ok: true,
      status: "Completado",
      request_id: id,
      message: "La solicitud ya estaba terminada."
    };
  }

  if (["en cola", "procesando"].includes(status)) {
    return {
      ok: true,
      status: status === "en cola" ? "En cola" : "Procesando",
      request_id: id,
      message: "La solicitud ya fue enviada o está en proceso."
    };
  }

  const photoColumns = ["Foto01", "Foto02", "Foto03", "Foto04", "Foto05"];
  const hasPhotoColumn = photoColumns.some(name => {
    const index = tcCol_(headers, name);
    return index >= 0 && String(row[index] || "").trim();
  });

  const fotosIndex = tcCol_(headers, "Fotos");
  const hasFotosList = fotosIndex >= 0 && tcHasListItems_(row[fotosIndex]);

  if (!hasPhotoColumn && !hasFotosList) {
    throw new Error("Agrega al menos una foto en Foto01–Foto05 o Fotos.");
  }

  const parcelaIndex = tcCol_(headers, "ParcelaID");
  if (parcelaIndex < 0 || !String(row[parcelaIndex] || "").trim()) {
    throw new Error("Selecciona una parcela en la columna ParcelaID.");
  }

  // Marcar primero para evitar que dos eventos lancen el mismo trabajo.
  tcSetCell_(sheet, found.rowNumber, headers, "Estado", "En cola");
  tcSetCell_(sheet, found.rowNumber, headers, "Error", "");

  try {
    tcDispatchGitHub_(id);
  } catch (error) {
    tcSetCell_(sheet, found.rowNumber, headers, "Estado", "Error");
    tcSetCell_(
      sheet,
      found.rowNumber,
      headers,
      "Error",
      String(error && error.message ? error.message : error).slice(0, 3500)
    );
    tcSetCell_(sheet, found.rowNumber, headers, "FechaFin", new Date());
    throw error;
  }

  return {
    ok: true,
    status: "En cola",
    request_id: id,
    message: "GitHub Actions recibió la solicitud TerraCore."
  };
}

function tcDispatchGitHub_(requestId) {
  const props = PropertiesService.getScriptProperties();
  const token = String(props.getProperty("GITHUB_DISPATCH_TOKEN") || "").trim();
  const owner = String(props.getProperty("GITHUB_OWNER") || "AutoauditoriaUnifin").trim();
  const repo = String(props.getProperty("GITHUB_REPO") || "contador-vinedo-video").trim();
  const workflow = String(
    props.getProperty("GITHUB_WORKFLOW_FILE") || "process-appsheet-analysis.yml"
  ).trim();
  const branch = String(
    props.getProperty("GITHUB_BRANCH") || "main"
  ).trim();

  if (!token) throw new Error("Falta GITHUB_DISPATCH_TOKEN en Apps Script.");

  const url =
    "https://api.github.com/repos/" + owner + "/" + repo +
    "/actions/workflows/" + encodeURIComponent(workflow) + "/dispatches";

  const response = UrlFetchApp.fetch(url, {
    method: "post",
    contentType: "application/json",
    headers: {
      Authorization: "Bearer " + token,
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28"
    },
    payload: JSON.stringify({
      ref: branch,
      inputs: { request_id: requestId }
    }),
    muteHttpExceptions: true
  });

  const status = response.getResponseCode();
  if (status !== 204) {
    throw new Error(
      "GitHub no aceptó el trabajo. HTTP " + status + ": " +
      response.getContentText().slice(0, 1200)
    );
  }
}

function tcMarkError_(requestId, message) {
  const id = String(requestId || "").trim();
  if (!id) throw new Error("Falta request_id.");

  const sheet = tcGetRequestSheet_();
  const found = tcFindRequest_(sheet, id);
  tcSetCell_(sheet, found.rowNumber, found.headers, "Estado", "Error");
  tcSetCell_(
    sheet,
    found.rowNumber,
    found.headers,
    "Error",
    String(message || "Error").slice(0, 3500)
  );
  tcSetCell_(sheet, found.rowNumber, found.headers, "FechaFin", new Date());

  return { ok: true, request_id: id, status: "Error" };
}

function tcGetRequestSheet_() {
  const spreadsheetId = String(
    PropertiesService.getScriptProperties().getProperty("GSHEET_ID") || ""
  ).trim();
  if (!spreadsheetId) throw new Error("Falta GSHEET_ID en Apps Script.");

  const ss = SpreadsheetApp.openById(spreadsheetId);
  const sheet = ss.getSheetByName(TC_REQUEST_SHEET);
  if (!sheet) {
    throw new Error(
      "No existe SolicitudesAnalisis. Ejecuta setupTerraCore() primero."
    );
  }
  return sheet;
}

function tcFindRequest_(sheet, requestId) {
  const values = sheet.getDataRange().getValues();
  if (values.length < 2) throw new Error("No hay solicitudes registradas.");

  const headers = values[0].map(value => String(value || "").trim());
  const idColumn = tcCol_(headers, "SolicitudID");
  if (idColumn < 0) throw new Error("Falta el encabezado SolicitudID.");

  for (let i = 1; i < values.length; i++) {
    if (String(values[i][idColumn] || "").trim() === requestId) {
      return {
        rowNumber: i + 1,
        headers: headers,
        values: values[i]
      };
    }
  }
  throw new Error("No se encontró la solicitud " + requestId + ".");
}

function tcSetCell_(sheet, rowNumber, headers, name, value) {
  const column = tcCol_(headers, name);
  if (column >= 0) {
    sheet.getRange(rowNumber, column + 1).setValue(value);
  }
}

function tcCol_(headers, name) {
  const target = tcNormalize_(name);
  return headers.findIndex(header => tcNormalize_(header) === target);
}

function tcNormalize_(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/\s+/g, " ");
}

function tcHasListItems_(value) {
  const text = String(value || "").trim();
  if (!text) return false;
  try {
    const parsed = JSON.parse(text);
    if (Array.isArray(parsed)) return parsed.some(item => String(item || "").trim());
    return Boolean(String(parsed || "").trim());
  } catch (error) {
    return text.split(/[,;\n]/).some(item => item.trim());
  }
}

function tcVerifySecret_(supplied) {
  const expected = String(
    PropertiesService.getScriptProperties().getProperty("TERRACORE_SHARED_SECRET") || ""
  ).trim();
  if (!expected || String(supplied || "").trim() !== expected) {
    throw new Error("Secreto de integración no válido.");
  }
}

function tcJson_(value) {
  return ContentService
    .createTextOutput(JSON.stringify(value))
    .setMimeType(ContentService.MimeType.JSON);
}
