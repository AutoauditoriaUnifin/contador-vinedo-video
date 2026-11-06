# TerraCore — terminar la integración de Inventario en AppSheet

## Estado del código

El worker de AppSheet ejecuta el motor de `terracore_engine.py`, que conserva las funciones de detección y Salud de Streamlit. El flujo queda dividido en dos ejecuciones:

1. **Primera ejecución:** genera el inventario de todas las fotos, guarda un registro por surco en `InventarioSurcos` y deja `SolicitudesAnalisis[Estado]` en `PendienteValidacion`.
2. **Revisión en AppSheet:** permite corregir conteos, índices vacíos y extremos de las líneas.
3. **Confirmación:** una acción cambia `ValidacionInventario` a `Validado`.
4. **Segunda ejecución:** un Bot vigila únicamente `ValidacionInventario` y vuelve a lanzar el worker. El motor lee las correcciones y ejecuta `_ocv2_analyze_health()` sin volver a detectar ni reemplazar el inventario.
5. **Cierre:** se guardan los resultados en las tablas de historial y análisis que ya utiliza TerraCore.

Estados esperados:

`Pendiente` → `ProcesandoInventario` → `PendienteValidacion` → `ProcesandoSalud` → `Completado`

El worker no marca una solicitud como completada si falta Inventario validado de alguna fotografía o si una de las fotos falla durante Salud.

## 1. Preparar las pestañas

El código prepara automáticamente los encabezados de `SolicitudesAnalisis` y `InventarioSurcos`. También puedes ejecutar una vez el endpoint `POST /appsheet/setup` si tienes desplegada la API FastAPI y su token.

Después de que exista la pestaña `InventarioSurcos` en el mismo Google Sheets:

1. En AppSheet abre **Data → Tables → + New Table**.
2. Selecciona la pestaña `InventarioSurcos`.
3. Agrega la tabla.
4. En **Data → Columns**, regenera la estructura de `SolicitudesAnalisis` e `InventarioSurcos`.

No elimines ni reemplaces columnas existentes.

## 2. Columnas de SolicitudesAnalisis

El worker agrega estas columnas si faltan:

- `ValidacionInventario` — Text o Enum.
- `FechaInventario` — DateTime.
- `FechaValidacionInventario` — DateTime.
- `FechaInicioSalud` — DateTime.
- `ResumenInventario` — LongText.

La clave existente `SolicitudID` se conserva.

## 3. Configurar InventarioSurcos

Configura `RegistroID` como **Key**. Desactiva **Deletes** y permite **Updates**.

| Columna | Tipo AppSheet | Editable |
|---|---|---:|
| RegistroID | Text (Key) | No |
| SolicitudID | Ref o Text | No |
| AnalisisID | Text | No |
| ParcelaID | Text | No |
| ParcelaNombre | Text | No |
| FotoIndex | Number | No |
| FotoNombre | Text | No |
| FotoRuta | Image o Text | No |
| ImagenInventarioAppSheet | Image o Text | No |
| NumeroSurco | Text | No |
| Slots | Number | Sí |
| Ocupados | Number | Sí |
| Vacios | Number | Sí |
| VaciosIndices | LongText | Sí |
| PuntoInicioX | Decimal | Sí |
| PuntoInicioY | Decimal | Sí |
| PuntoFinX | Decimal | Sí |
| PuntoFinY | Decimal | Sí |
| RowSpacingPx | Decimal | No |
| SlotPitchPx | Decimal | No |
| GeometriaJSON | LongText | No |
| Confianza | Decimal | No |
| ConfianzaSurco | Decimal | No |
| ConfianzaSlots | Decimal | No |
| ManualPosition | Yes/No | No |
| ManualInventory | Yes/No | No |
| EstadoInventario | Text | No |
| FechaValidacion | DateTime | No |
| Idioma | Text | No |
| Fecha | DateTime | No |

Para las columnas editables, usa esta expresión en **Editable_If**:

```appsheet
LOOKUP(
  [SolicitudID],
  "SolicitudesAnalisis",
  "SolicitudID",
  "Estado"
) = "PendienteValidacion"
```

Validaciones recomendadas:

- `Slots`: `[_THIS] >= 2`
- `Ocupados`: `AND([_THIS] >= 0, [_THIS] <= [Slots])`
- `Vacios`: `AND([_THIS] >= 0, [_THIS] <= [Slots])`
- Coordenadas: `AND([_THIS] >= 0, [_THIS] <= 1000)`

### Cómo editar los surcos

- **Slots:** número total de posiciones de ese surco.
- **Ocupados:** posiciones ocupadas.
- **Vacios:** posiciones vacías.
- **VaciosIndices:** números de las posiciones vacías separados por coma; ejemplo `2,5,8`.
- **PuntoInicioX/Y y PuntoFinX/Y:** extremos de la línea en escala normalizada de 0 a 1000. X va de izquierda a derecha; Y de arriba hacia abajo.

Si no modificas los cuatro extremos, se conserva la geometría original detectada. Si modificas los cuatro, el worker reconstruye el eje recto y recalcula las posiciones sobre esa línea. AppSheet no tiene el mismo canvas de arrastrar y girar de Streamlit; aquí se editan las coordenadas numéricas.

## 4. Acción Confirmar Inventario

En **Behavior → Actions → New Action**:

- **For a record of this table:** `SolicitudesAnalisis`
- **Do this:** `Data: set the values of some columns in this row`
- **Set these columns:** `ValidacionInventario` = `"Validado"`
- **Only if this condition is true:**

```appsheet
AND(
  [Estado] = "PendienteValidacion",
  COUNT(
    SELECT(
      InventarioSurcos[RegistroID],
      AND(
        [SolicitudID] = [_THISROW].[SolicitudID],
        OR(
          [Slots] < 2,
          [Ocupados] < 0,
          [Vacios] < 0,
          [Slots] <> ([Ocupados] + [Vacios]),
          [Ocupados] > [Slots],
          [Vacios] > [Slots],
          [PuntoInicioX] < 0,
          [PuntoInicioX] > 1000,
          [PuntoInicioY] < 0,
          [PuntoInicioY] > 1000,
          [PuntoFinX] < 0,
          [PuntoFinX] > 1000,
          [PuntoFinY] < 0,
          [PuntoFinY] > 1000
        )
      )
    )
  ) = 0,
  COUNT(
    SELECT(
      InventarioSurcos[RegistroID],
      [SolicitudID] = [_THISROW].[SolicitudID]
    )
  ) > 0,
  COUNT(
    UNIQUE(
      SELECT(
        InventarioSurcos[FotoIndex],
        [SolicitudID] = [_THISROW].[SolicitudID]
      )
    )
  ) = [TotalFotos]
)
```

Muestra la acción como botón **Confirmar Inventario**. No permitas editar `ValidacionInventario` directamente desde el formulario.

## 5. Bot de Salud posterior a la confirmación

Conserva el Bot actual que procesa las solicitudes nuevas.

Crea un segundo Bot:

- **Nombre:** `TerraCore - Salud después de validar inventario`
- **Tabla:** `SolicitudesAnalisis`
- **Evento:** Data change → Updates.
- **Columns to watch:** únicamente `ValidacionInventario`.
- **Condición:** `[ValidacionInventario] = "Validado"`.
- **Proceso:** Webhook al mismo Apps Script Web App que ya dispara el worker.

Usa el mismo formato de cuerpo y el mismo secreto que ya utiliza el Bot actual. Si el webhook actual envía:

```json
{
  "action": "trigger",
  "request_id": "<<[SolicitudID]>>",
  "secret": "EL_MISMO_SECRETO_CONFIGURADO"
}
```

reutiliza exactamente esa configuración; no generes ni publiques un secreto nuevo. No pongas el secreto en una columna visible de AppSheet.

Vigilar solo `ValidacionInventario` evita que los cambios de estado que hace el worker vuelvan a disparar el Bot.

## 6. Prueba de punta a punta

1. Crea una solicitud nueva desde AppSheet.
2. Espera a que el worker termine el Inventario y el estado cambie a `PendienteValidacion`.
3. Abre `InventarioSurcos` y filtra por el `SolicitudID` de esa solicitud.
4. Revisa todas las fotografías. Corrige conteos, índices o coordenadas si es necesario.
5. Pulsa **Confirmar Inventario**.
6. Comprueba que se active el segundo Bot y que el estado pase a `ProcesandoSalud`.
7. Espera `Completado` y revisa las imágenes y registros del historial.

Si alguna foto no generó inventario, el botón no debe habilitarse porque no se alcanza `TotalFotos`. El backend también vuelve a validar los registros antes de ejecutar Salud.

## Importante

La implementación del repositorio no puede crear por sí sola los Bots ni cambiar las propiedades de tablas dentro de tu cuenta AppSheet. Esos pasos de configuración se hacen en el editor de AppSheet. La primera prueba real debe confirmar que el webhook del segundo Bot está usando el mismo Apps Script y secreto del flujo actual.
