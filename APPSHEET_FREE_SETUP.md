# TerraCore + AppSheet sin Cloud Run ni facturación

Esta modalidad usa:

- **AppSheet** para capturar fotos, videos y parcela.
- **Google Apps Script** como puente seguro y para solicitar un token OAuth temporal.
- **GitHub Actions** para ejecutar Python, OpenCV y el motor TerraCore.
- **El mismo Google Sheets y Google Drive** que ya utiliza TerraCore.

No requiere Cloud Run, Cloud Tasks, una cuenta de servicio ni habilitar facturación de Google Cloud. GitHub Actions utiliza ejecutores estándar gratuitos para este repositorio público, sujetos a las condiciones vigentes de GitHub. Apps Script y AppSheet conservan sus propias cuotas y licencias.

> Importante: el PR debe revisarse antes de fusionarse. GitHub solo acepta el evento `workflow_dispatch` cuando el archivo del workflow existe en la rama predeterminada del repositorio. Por eso, el flujo automático quedará listo en esta rama, pero no podrá activarse desde AppSheet hasta que se autorice la integración en `main`.

## 1. Crear el puente de Apps Script

1. Abre [Google Apps Script](https://script.google.com/home).
2. Crea un proyecto llamado `TerraCore AppSheet Bridge`.
3. En el archivo `Code.gs`, pega el contenido de [apps_script/Code.gs](apps_script/Code.gs).
4. En **Configuración del proyecto**, activa la opción para mostrar el archivo de manifiesto `appsscript.json`.
5. Reemplaza su contenido con [apps_script/appsscript.json](apps_script/appsscript.json).

## 2. Configurar las propiedades de Apps Script

En **Configuración del proyecto → Propiedades del script**, agrega estas propiedades:

| Propiedad | Valor |
|---|---|
| `GSHEET_ID` | ID del Google Sheets actual de TerraCore |
| `TERRACORE_SHARED_SECRET` | Secreto largo y aleatorio que tú generes |
| `GITHUB_DISPATCH_TOKEN` | Token fino de GitHub con permiso Actions: Read and write |
| `GITHUB_OWNER` | `AutoauditoriaUnifin` |
| `GITHUB_REPO` | `contador-vinedo-video` |
| `GITHUB_WORKFLOW_FILE` | `process-appsheet-analysis.yml` |
| `GITHUB_BRANCH` | `integracion-appsheet` durante las pruebas |

Genera el secreto compartido en tu equipo o con un administrador de contraseñas. No lo publiques en el repositorio ni lo envíes por este chat.

### Crear el token de GitHub

1. En GitHub, abre **Settings → Developer settings → Fine-grained personal access tokens**.
2. Genera un token limitado al repositorio `contador-vinedo-video`.
3. Dale permiso **Actions: Read and write** y **Metadata: Read-only**.
4. Copia el token directamente a la propiedad `GITHUB_DISPATCH_TOKEN` de Apps Script.
5. No lo guardes en archivos del repositorio.

## 3. Autorizar y preparar la hoja

1. En el editor de Apps Script, selecciona la función `setupTerraCore`.
2. Pulsa **Ejecutar** y autoriza el acceso a Google Sheets y Drive.
3. La función crea la pestaña `SolicitudesAnalisis` dentro del mismo libro, con los encabezados necesarios.
4. Comprueba que la pestaña quedó creada.

El script solicita los permisos de Sheets, Drive y conexiones externas. No necesita una cuenta de facturación.

## 4. Publicar Apps Script como aplicación web

1. Pulsa **Implementar → Nueva implementación**.
2. Tipo: **Aplicación web**.
3. Ejecutar como: **Yo**.
4. Quién tiene acceso: **Cualquier persona**.
5. Implementa y copia la URL terminada en `/exec`.

El acceso público está protegido por `TERRACORE_SHARED_SECRET`, que se valida en cada petición. No compartas la URL junto con el secreto en capturas públicas.

## 5. Configurar los secretos del repositorio

En GitHub abre **Settings → Secrets and variables → Actions → New repository secret** y agrega:

| Nombre del secreto | Valor |
|---|---|
| `APPS_SCRIPT_WEBAPP_URL` | URL `/exec` de la aplicación web de Apps Script |
| `APPS_SCRIPT_SHARED_SECRET` | El mismo secreto configurado en Apps Script |
| `GSHEET_ID` | ID del libro actual de TerraCore |
| `GDRIVE_PARENT_FOLDER_ID` | ID de la carpeta raíz donde se guardan los resultados |
| `APPSHEET_DRIVE_ROOT_ID` | Carpeta donde AppSheet almacena las fotos y videos; puede ser la misma anterior |

No agregues estos valores a archivos Python, YAML, Markdown ni al código de Apps Script.

## 6. Agregar la tabla en AppSheet

1. Abre la aplicación TerraCore en AppSheet.
2. En **Data → Tables**, agrega la pestaña `SolicitudesAnalisis`.
3. Pulsa **Regenerate structure**.
4. Configura las columnas:

| Columna | Tipo AppSheet | Configuración |
|---|---|---|
| `SolicitudID` | Text | Key; Initial value: `UNIQUEID()` |
| `FechaSolicitud` | DateTime | Initial value: `NOW()` |
| `ParcelaID` | Ref | Referencia a `Parcelas`; columna Key: `ID` |
| `ParcelaNombre` | Text | App formula: `[ParcelaID].[NombreParcela]` |
| `Idioma` | Enum | Valores ES y FR; inicial: `"ES"` |
| `Fotos` | LongText | Opcional, para una lista de rutas |
| `Foto01` a `Foto05` | Image | Fotos de la solicitud |
| `Video` | File | Video principal opcional |
| `Video simulación` | File | Video de simulación opcional |
| `SurcoID` | Text | Opcional |
| `Estado` | Enum | Pendiente, En cola, Procesando, Completado, Error; inicial: `"Pendiente"` |
| `FechaInicio`, `FechaFin` | DateTime | No editables |
| `HistorialIDs`, `TotalFotos`, `ResumenResultado`, `Error` | LongText / Number | No editables |

La solicitud debe tener una parcela y al menos una foto. Los videos son opcionales.

## 7. Crear el Bot de AppSheet

1. Ve a **Automation → Bots → New Bot**.
2. Evento: **Data change → Adds only** en `SolicitudesAnalisis`.
3. Condición del evento: `[Estado] = "Pendiente"`.
4. Agrega un paso **Call a webhook**.
5. Configura:
   - Method: `POST`
   - URL: la URL `/exec` de Apps Script
   - Content type: `application/json`
   - Body:

```json
{
  "action": "trigger",
  "request_id": "<<[SolicitudID]>>",
  "secret": "PEGA_AQUI_EL_MISMO_SECRETO_COMPARTIDO"
}
```

Sustituye el texto de ejemplo por el secreto que guardaste en Apps Script. No uses un token de GitHub en este cuerpo.

## 8. Qué sucede al guardar una solicitud

1. AppSheet guarda la fila y sus archivos en Drive.
2. Apps Script valida la solicitud y la marca como `En cola`.
3. Apps Script llama a GitHub Actions.
4. GitHub Actions solicita a Apps Script un token OAuth temporal.
5. Python descarga las fotos desde Drive, ejecuta el motor OpenCV y guarda los resultados en las pestañas existentes.
6. Los videos adjuntos se relacionan con cada `HistorialID` generado y se registran en `HistorialTerroCore` y `Evidencias1`.
7. AppSheet muestra el estado `Completado` o `Error`.

El token OAuth se solicita al iniciar cada trabajo y se mantiene solo en memoria del ejecutor. No se imprime en los logs.

## 9. Activación final

El workflow `process-appsheet-analysis.yml` se encuentra en esta rama de preparación. Para que GitHub acepte los despachos automáticos de Apps Script, primero debe existir en la rama predeterminada:

1. Revisar el PR #3.
2. Autorizar su fusión a `main`.
3. En las propiedades de Apps Script, cambiar `GITHUB_BRANCH` a `main`.
4. Verificar que los secretos de Actions estén configurados.
5. Hacer una prueba con una solicitud y una foto de muestra.

**No se debe fusionar el PR hasta que se autorice expresamente.**

## Consideraciones

- Esta arquitectura evita mantener un servidor encendido.
- GitHub Actions puede tardar un poco en iniciar cada trabajo; no es procesamiento instantáneo.
- Las cuotas de Apps Script, el límite de ejecución de GitHub Actions y el tamaño máximo de archivos siguen aplicando.
- Los videos se guardan como rutas de archivo de AppSheet en Drive; no se vuelven a subir ni se procesan con OpenCV.
- La integración actual analiza imágenes; el video se conserva como evidencia asociada al historial.
