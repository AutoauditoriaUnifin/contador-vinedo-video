# TerraCore — análisis automático de AppSheet con el motor de Streamlit

## Objetivo

AppSheet debe ejecutar el análisis completo sin intervención manual:

1. Recibe la solicitud y las fotografías.
2. Ejecuta el Inventario con `_ocv2_inventory_from_file()`.
3. Conserva la geometría y los slots calculados por el motor.
4. Ejecuta inmediatamente Salud con `_ocv2_analyze_health()`.
5. Guarda imágenes y resultados en Drive, Sheets e historial.

**No se debe editar ni confirmar manualmente el Inventario.** El archivo `app_surcos_VIDEO_V10.py` de Streamlit permanece intacto. El worker utiliza las funciones equivalentes de `terracore_engine.py`.

## 1. Actualizar Apps Script

El puente actualizado está en [apps_script/Code.gs](https://github.com/AutoauditoriaUnifin/contador-vinedo-video/blob/main/apps_script/Code.gs).

1. Abre el proyecto de Apps Script que ya utiliza TerraCore.
2. Sustituye el contenido de `Code.gs` por el del enlace anterior.
3. Conserva las propiedades y el proyecto actuales. No compartas ni publiques secretos.
4. En **Project Settings → Script Properties**, comprueba que existan:
   - `GSHEET_ID`
   - `TERRACORE_SHARED_SECRET`
   - `GITHUB_DISPATCH_TOKEN`
   - `GITHUB_OWNER` = `AutoauditoriaUnifin`
   - `GITHUB_REPO` = `contador-vinedo-video`
   - `GITHUB_WORKFLOW_FILE` = `process-appsheet-analysis.yml`
5. Ejecuta una sola vez `setupTerraCore()` desde el editor y autoriza los permisos.
6. En **Deploy → Manage deployments**, edita el Web App existente y crea una nueva versión. Conserva el mismo URL y la misma configuración de acceso.

En GitHub → **Settings → Secrets and variables → Actions**, verifica:
- `APPS_SCRIPT_WEBAPP_URL`
- `APPS_SCRIPT_SHARED_SECRET`
- `GSHEET_ID`
- `GDRIVE_PARENT_FOLDER_ID`
- `APPSHEET_DRIVE_ROOT_ID` (si tu estructura lo utiliza)

`APPS_SCRIPT_SHARED_SECRET` debe coincidir exactamente con `TERRACORE_SHARED_SECRET`. Nunca pegues esos valores en el chat ni los guardes en archivos públicos.

## 2. Preparar las pestañas

`setupTerraCore()` prepara `SolicitudesAnalisis` e `InventarioSurcos` sin borrar las columnas existentes.

En AppSheet:

1. Abre **Data → Tables**.
2. Regenera las columnas de `SolicitudesAnalisis`.
3. `InventarioSurcos` puede agregarse como tabla de consulta si quieres ver el detalle detectado. Si la agregas, configura `RegistroID` como Key y deja Adds, Updates y Deletes desactivados para evitar que alguien modifique la geometría que el motor acaba de calcular.

La tabla `InventarioSurcos` es un registro técnico del resultado intermedio; el worker la utiliza para continuar Salud en la misma ejecución. No es una pantalla de validación manual.

## 3. Columnas de SolicitudesAnalisis

El worker agrega si faltan:
- `ValidacionInventario`
- `FechaInventario`
- `FechaValidacionInventario`
- `FechaInicioSalud`
- `ResumenInventario`

Conserva `SolicitudID` como Key y no elimines las columnas actuales de fotos, parcela, estado o historial.

## 4. Bot automático de AppSheet

**Conserva el Bot actual que dispara el worker cuando se crea o se pone en cola una solicitud.** Ese único Bot debe iniciar el proceso completo.

- No crees un segundo Bot para Salud.
- Si anteriormente creaste el Bot `TerraCore - Salud después de validar inventario`, desactívalo.
- No configures una acción `Confirmar Inventario`.
- No cambies `ValidacionInventario` manualmente.

El worker ahora ejecuta Inventario y Salud consecutivamente. Al terminar, la solicitud debe quedar en `Completado`. Si falla una fotografía o una etapa, debe quedar en `Error` con el detalle correspondiente.

Estados esperados:

`Pendiente` → `ProcesandoInventario` → `InventarioGenerado` → `ProcesandoSalud` → `Completado`

## 5. Prueba de punta a punta

1. Registra una solicitud nueva con una parcela y sus fotografías.
2. Deja que se active el Bot que ya existe.
3. En GitHub → **Actions**, abre `Process TerraCore AppSheet analysis` y comprueba que el job se ejecutó en la rama `main`.
4. En `SolicitudesAnalisis`, comprueba que el estado termine en `Completado`.
5. Revisa en el historial las imágenes de Inventario, análisis y polígonos, así como el número de surcos y los porcentajes.
6. Si queda en `Error`, copia únicamente el texto del error; no compartas tokens ni secretos.

## 6. Qué significa que sea el mismo análisis

El worker reutiliza las funciones del motor TerraCore que corresponden al flujo de Streamlit. La integración no debe volver a estimar, ajustar ni forzar el número de surcos. Cada foto se procesa por separado, con sus propios surcos, slots y geometría.

La comprobación definitiva requiere una prueba A/B con **el mismo archivo de imagen original**: ejecutar esa foto en Streamlit y en AppSheet, y comparar el Inventario y la imagen resultante. Los checks de GitHub validan que el código compile y se importe; por sí solos no prueban el resultado de una fotografía real ni el despliegue de Apps Script.
