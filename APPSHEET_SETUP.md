# TerraCore + AppSheet: integración de backend

Esta integración deja a AppSheet como única interfaz de usuario. Streamlit conserva su código y su despliegue actual; el motor OpenCV V2 se ejecuta desde un servicio backend en Cloud Run.

## 1. Flujo

1. AppSheet crea una fila en \`SolicitudesAnalisis\`.
2. Un Bot de AppSheet llama a \`POST /appsheet/analysis/submit\`.
3. La API crea una tarea en Cloud Tasks y responde rápidamente.
4. Cloud Tasks llama a \`POST /appsheet/analysis/process\`.
5. El backend descarga las fotos que AppSheet guardó en Google Drive.
6. El motor OpenCV V2 calcula inventario, slots, estado visual y polígonos.
7. Los resultados se guardan en el mismo Google Sheets y Drive.
8. La solicitud cambia a \`Completado\` o \`Error\`.

No se reemplaza ni se modifica el archivo principal de Streamlit.

## 2. Pestaña nueva: SolicitudesAnalisis

Después de desplegar el backend, llama una vez a:

\`POST https://TU-SERVICIO/appsheet/setup\`

Header:
\`Authorization: Bearer TU_APPSHEET_API_TOKEN\`

La API crea la pestaña \`SolicitudesAnalisis\` en el mismo archivo indicado por \`GSHEET_ID\`, con estos encabezados:

| Columna | Tipo recomendado en AppSheet | Configuración |
|---|---|---|
| SolicitudID | Text | Key; Initial value: \`UNIQUEID()\` |
| FechaSolicitud | DateTime | Initial value: \`NOW()\` |
| ParcelaID | Ref | Referencia a \`Parcelas\`; columna Key: \`ID\` |
| ParcelaNombre | Text | App formula: \`[ParcelaID].[NombreParcela]\` |
| Idioma | Enum | Valores ES, FR; Initial value: \`"ES"\` |
| Foto01 | Image | Primera fotografía |
| Foto02 | Image | Fotografía adicional opcional |
| Foto03 | Image | Fotografía adicional opcional |
| Foto04 | Image | Fotografía adicional opcional |
| Foto05 | Image | Fotografía adicional opcional |
| Fotos | LongText | Opcional; admite una lista separada por comas o JSON |
| SurcoID | Text | Opcional |
| Estado | Enum | Initial value: \`"Pendiente"\`; no editable por el usuario |
| FechaInicio | DateTime | No editable |
| FechaFin | DateTime | No editable |
| HistorialIDs | LongText | No editable |
| TotalFotos | Number | No editable |
| ResumenResultado | LongText | No editable |
| Error | LongText | No editable |

Los campos Foto01–Foto05 permiten enviar hasta cinco fotografías en una solicitud. Si una solicitud contiene varias fotos, se genera un HistorialID independiente para cada una y todos quedan vinculados al mismo SolicitudID.

## 3. Variables y secretos de Cloud Run

Variables de entorno:

- \`GSHEET_ID\`: el mismo ID que ya utiliza Streamlit.
- \`GDRIVE_PARENT_FOLDER_ID\`: la misma carpeta raíz de Drive que ya utiliza Streamlit.
- \`APPSHEET_DRIVE_ROOT_ID\`: opcional; ID de la carpeta de AppSheet donde se guardan las fotos. Si no se define, se utiliza GDRIVE_PARENT_FOLDER_ID.
- \`CLOUD_TASKS_PROJECT\`: ID del proyecto de Google Cloud.
- \`CLOUD_TASKS_LOCATION\`: región de Cloud Tasks, por ejemplo \`us-central1\`.
- \`CLOUD_TASKS_QUEUE\`: nombre de la cola, por ejemplo \`terracore-analysis\`.
- \`CLOUD_RUN_SERVICE_URL\`: URL base del servicio Cloud Run, sin barra final.
- \`GCP_SERVICE_ACCOUNT_JSON\`: opcional. Si se omite, se utiliza la cuenta de servicio asignada a Cloud Run mediante Application Default Credentials.
- \`OPENAI_API_KEY\`: opcional para la ruta antigua \`/analyze-image\`; el nuevo flujo AppSheet utiliza el motor OpenCV V2.

Secreto obligatorio:

- \`APPSHEET_API_TOKEN\`: token largo y aleatorio. Guárdalo en Secret Manager, no en GitHub, AppSheet como texto visible ni en el código.

## 4. Permisos de Google

Usa una cuenta de servicio dedicada para Cloud Run, por ejemplo:

\`terracore-api@TU_PROJECT_ID.iam.gserviceaccount.com\`

Dale el rol de proyecto:

- \`roles/cloudtasks.enqueuer\`

Comparte con esa cuenta de servicio, como Editor:

- El Google Sheets que utiliza TerraCore.
- La carpeta raíz de Drive configurada en \`GDRIVE_PARENT_FOLDER_ID\`.
- La carpeta de archivos de AppSheet, si es distinta.

No hace falta hacer públicos los archivos de Drive. La cuenta de servicio debe poder leer las fotos que AppSheet sube y escribir los resultados.

## 5. Cloud Tasks

Crea una cola en la misma región del servicio:

\`gcloud tasks queues create terracore-analysis --location=us-central1\`

El backend usa esta cola para ejecutar el análisis fuera del tiempo de espera del webhook de AppSheet.

## 6. Despliegue de Cloud Run

Desde la raíz del repositorio y estando en la rama \`integracion-appsheet\`:

\`\`\`bash
gcloud config set project TU_PROJECT_ID

gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com cloudtasks.googleapis.com secretmanager.googleapis.com

gcloud run deploy terracore-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --service-account terracore-api@TU_PROJECT_ID.iam.gserviceaccount.com \
  --memory 4Gi \
  --cpu 2 \
  --timeout 3600 \
  --set-env-vars GSHEET_ID=TU_GSHEET_ID,GDRIVE_PARENT_FOLDER_ID=TU_DRIVE_FOLDER_ID,CLOUD_TASKS_PROJECT=TU_PROJECT_ID,CLOUD_TASKS_LOCATION=us-central1,CLOUD_TASKS_QUEUE=terracore-analysis \
  --set-secrets APPSHEET_API_TOKEN=APPSHEET_API_TOKEN:latest
\`\`\`

Después del primer despliegue, copia la URL que devuelve Cloud Run y actualiza la variable:

\`\`\`bash
gcloud run services update terracore-api \
  --region us-central1 \
  --update-env-vars CLOUD_RUN_SERVICE_URL=https://TU-SERVICIO-URL
\`\`\`

La API queda pública para que AppSheet pueda llamar al webhook, pero las rutas de configuración y procesamiento exigen el token secreto.

## 7. Preparar la pestaña en Sheets y AppSheet

1. Prueba \`https://TU-SERVICIO/appsheet/health\`.
2. Ejecuta \`POST /appsheet/setup\` con el header Bearer del token.
3. En AppSheet, ve a **Data > Tables > + New Table**.
4. Agrega \`SolicitudesAnalisis\` desde el mismo Google Sheets.
5. Configura los tipos de columna de la tabla según la sección 2.
6. Configura \`SolicitudID\` como Key.

## 8. Bot de AppSheet

Crea un Bot con evento:

- Table: \`SolicitudesAnalisis\`
- Data change: Adds
- Condition: \`[Estado] = "Pendiente"\`

Agrega una tarea **Call a webhook**:

- URL: \`https://TU-SERVICIO/appsheet/analysis/submit\`
- HTTP Verb: POST
- HTTP Content Type: JSON
- Header \`Authorization\`: \`Bearer TU_APPSHEET_API_TOKEN\`
- Body:

\`\`\`json
{
  "request_id": "<<[SolicitudID]>>"
}
\`\`\`

La solicitud se procesa en segundo plano. AppSheet no debe esperar a que termine OpenCV; debe mostrar el campo \`Estado\` y actualizar la vista para ver \`Procesando\`, \`Completado\` o \`Error\`.

## 9. Prueba inicial

Haz una solicitud con una sola fotografía de una parcela bien visible. Comprueba:

- \`SolicitudesAnalisis.Estado\` cambia de Pendiente a En cola, Procesando y Completado.
- Se genera un ID en \`HistorialIDs\`.
- Aparece una fila en \`HistorialTerroCore\`.
- Se generan filas relacionadas en \`Analisis\`, \`InventarioSurcos\` y \`Evidencias\`.
- Las imágenes Originales, Inventarios, Procesadas y Poligonos aparecen en Drive.

Después prueba dos o más fotos en la misma solicitud.

## 10. Videos

El módulo actual de videos de Streamlit ya tiene una lógica que valida HistorialID + Idioma, actualiza Video y Video simulación en la fila correcta de HistorialTerroCore y hace upsert en Evidencias1.

La integración del formulario de videos de AppSheet se conecta como siguiente etapa, después de validar primero el ciclo de análisis de imágenes. Así evitamos que una falla de videos o permisos oculte un problema en el motor de imágenes.
