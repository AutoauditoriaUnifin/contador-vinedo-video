# Homologación Streamlit + AppSheet — TerraCore

## Objetivo
AppSheet funciona como interfaz de captura y consulta; el motor OpenCV compartido de TerraCore realiza el análisis. Streamlit se conserva y continúa funcionando. Esta integración no reemplaza ni elimina la aplicación original.

## Fuente única del análisis
- La interfaz Streamlit activa invoca `tc_engine._ocv2_inventory_from_file(up)`.
- La integración AppSheet invoca `_ocv2_inventory_from_file(uploaded)`, importada del mismo módulo `terracore_engine.py`.
- Ambos caminos entregan un objeto con `getvalue()` al mismo motor, por lo que detección, geometría, slots y métricas provienen de la misma implementación.

La integración registra en `ResumenInventario` el SHA-256, tamaño, formato, dimensiones y orientación EXIF de la imagen recibida. Estos datos permiten verificar que AppSheet haya procesado exactamente el mismo archivo que Streamlit. Si los hashes difieren, primero debe corregirse la ruta/origen de la imagen; no se deben ajustar umbrales del detector para compensar una fotografía distinta.

## Métricas de confianza homologadas

| Métrica | Qué representa | Origen / cálculo |
|---|---|---|
| Confianza del surco | Soporte visual de la línea detectada | `row["confidence"] × 100`, por surco |
| Confianza de imagen | Periodicidad y contraste visual usados para calcular posiciones de plantas | `inventory["confidence"] × 100`; el motor calcula el promedio de `slot_confidence` de los surcos de esa fotografía |
| Confianza de slots | Confianza de posiciones de plantas para un surco | `row["slot_confidence"] × 100`, por surco |

La columna heredada `Confianza` se conserva como alias de `ConfianzaSlots` para no romper vistas ni expresiones existentes. Se agrega `ConfianzaImagen`; su valor de fotografía se repite en las filas de sus surcos para facilitar su visualización y agrupación en AppSheet.

## Validaciones automáticas
Antes de guardar el inventario se comprueba:
- Identificador único y válido para cada surco.
- Geometría con al menos dos puntos normalizados dentro de 0–1000.
- Al menos dos posiciones (slots) por surco.
- Índices de posiciones vacías dentro del rango válido y sin duplicados.
- Confianza del surco y de slots en el rango 0–100%.
- Totales de surcos, slots, ocupados y vacíos.
- Diagnóstico de la imagen fuente (SHA-256, bytes, formato, dimensiones y EXIF).

Las advertencias de confianza baja se conservan como advertencias; no se cambian automáticamente los umbrales ni se descartan surcos solo por tener confianza baja.

## Compatibilidad y conservación
- Se conserva `app_surcos_VIDEO_V10.py` y su interfaz Streamlit.
- Se conserva `terracore_engine.py` como motor compartido.
- Se conservan `backend_api.py`, Apps Script, los flujos de Drive/Sheets, historial y evidencias de video.
- No se fusiona esta rama a `main`.
- No se cambian hojas ni columnas existentes de manera destructiva; `ConfianzaImagen` se agrega como columna nueva.

## Pruebas
La validación de GitHub Actions compila los módulos Python, comprueba las rutas FastAPI y las columnas de inventario/video, y ejecuta pruebas unitarias para verificar la fórmula y compatibilidad de las métricas de confianza.

## Rama de revisión
`homologacion-streamlit-appsheet`

La rama se creó desde `main`. La integración queda pendiente de revisión y aprobación antes de cualquier fusión.
