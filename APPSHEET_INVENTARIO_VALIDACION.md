# TerraCore — Inventario y validación antes de Salud en AppSheet

## Flujo implementado

La solicitud ahora se procesa en dos fases. El detector y el analizador son los mismos que utiliza la aplicación Streamlit:

1. **Inventario:** cada fotografía pasa por `_ocv2_inventory_from_file()`.
2. **Revisión en AppSheet:** se guardan los surcos por fotografía en `InventarioSurcos`. El usuario puede corregir los conteos, los índices vacíos y los extremos de cada línea.
3. **Confirmación:** AppSheet cambia `ValidacionInventario` a `Validado`.
4. **Salud:** solo después de esa confirmación se ejecuta `_ocv2_analyze_health()` con las filas corregidas. No se vuelve a detectar el Inventario ni se descartan las correcciones.
5. **Cierre:** se guardan los resultados en `HistorialTerroCore`, `Analisis`, `RegionesCriticas`, `ZonasDañadas` y las tablas de evidencias existentes.

Estados de `SolicitudesAnalisis`:

`Pendiente` → `ProcesandoInventario` → `PendienteValidacion` → `ProcesandoSalud` → `Completado`

## Tablas y columnas

Primero ejecuta una vez el endpoint existente `POST /appsheet/setup` con el token de integración. Ahora prepara tanto `SolicitudesAnalisis` como `InventarioSurcos` en el mismo Google Sheets. Después, en AppSheet agrega `InventarioSurcos` como tabla y abre **Data → Columns → Regenerate structure** para ambas tablas. El worker también asegura las columnas si faltan.

### SolicitudesAnalisis

Se agregan:

- `ValidacionInventario` — Text o Enum; valor inicial: `Pendiente`.
- `FechaInventario` — DateTime.
- `FechaValidacionInventario` — DateTime.
- `FechaInicioSalud` — DateTime.
- `ResumenInventario` — LongText.

No cambies la clave `SolicitudID) ni elimines las columnas actuales.

### InventarioSurcos

Se conservan las columnas que ya existen y se agregan:

| Columna | Tipo sugerido | Editable |
|---|---|---:|
| `RegistroID` | Text, Key | No |
| `SolicitudID` | Text o Ref | No |
| `AnalisisID` | Text | No |
| `ParcelaID` | Text o Ref | No |
| `ParcelaNombre` | Text | No |
| `FotoIndex` | Number | No |
| `FotoNombre` | Text | No |
| `FotoRuta` | Image | No |
| `ImagenInventarioAppSheet` | Image | No |
| `NumeroSurco` | Text | No |
| `Slots` | Number | Sí |
| `Ocupados` | Number | Sí |
| `Vacios` | Number | Sí |
| `VaciosIndices` | Text | Sí |
| `PuntoInicioX`, `PuntoInicioY` | Number | Sí |
| `PuntoFinX`, `PuntoFinY` | Number | Sí |
| `RowSpacingPx`, `SlotPitchPx` | Number | No |
| `GeometriaJSON` | LongText | No |
| `Confianza`, `ConfianzaSurco`, `ConfianzaSlots` | Number | No |
| `ManualPosition`, `ManualInventory` | Yes/No | No |
| `EstadoInventario`, `FechaValidacion`, `Idioma`, `Fecha` | Text/DateTime | No |

Configura `RegistroID` como Key. La columna `ImagenInventarioAppSheet` debe ser de tipo Image para ver la imagen de referencia.

En las propiedades de la tabla, permite **Updates** pero desactiva **Deletes** para no borrar accidentalmente geometrías. En las columnas editables configura **Editable_If** con:

```appsheet
LOOKUP([SolicitudID], "SolicitudesAnalisis", "SolicitudID", "Estado") = "PendienteValidacion"
```

Así los campos se bloquean cuando la solicitud ya pasa a Salud o queda completada.

### Cómo corregir

- **Slots:** total de posiciones de ese surco.
- **Ocupados:** posiciones ocupadas.
- **Vacios:** posiciones vacías.
- **VaciosIndices:** números de los slots vacíos separados por coma. Ejemplo: `2,5,8`. Debe contener exactamente tantos índices como el valor de `Vacios`.
- **PuntoInicioX/Y y PuntoFinX/Y:** extremos de la línea del surco en escala normalizada de 0 a 1000. X: 0 es izquierda y 1000 derecha. Y: 0 es arriba y 1000 abajo. Modificar los cuatro valores mueve y gira la línea; los slots se reconstruyen sobre el eje corregido.

La app Streamlit dispone de un canvas para arrastrar y girar líneas. AppSheet no ofrece ese mismo canvas nativo; en AppSheet la corrección de geometría se realiza editando los cuatro extremos numéricos mientras se consulta la imagen de Inventario. El motor recibe esos extremos y reconstruye la misma geometría normalizada que consume el análisis original.

Validaciones recomendadas en AppSheet:

- `Slots` → Valid_If: `[_THIS] >= 0`
- `Ocupados` → Valid_If: `AND([_THIS] >= 0, [_THIS] <= [Slots])`
- `Vacios` → Valid_If: `AND([_THIS] >= 0, [_THIS] <= [Slots])`
- Cada uno de los cuatro extremos → Valid_If: `AND([_THIS] >= 0, [_THIS] <= 1000)`

La acción de confirmación valida además la igualdad de los tres conteos y que todas las fotografías inventariadas tengan registros. El worker vuelve a comprobarlo antes de ejecutar Salud.

## Acción para confirmar Inventario

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
          [Slots] < 0,
          [Ocupados] < 0,
          [Vacios] < 0,
          [Slots] <> ([Ocupados] + [Vacios]),
          [Ocupados] > [Slots],
          [Vacios] > [Slots]
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

Muestra la acción como botón, por ejemplo **Confirmar Inventario**. No permitas que el usuario cambie `ValidacionInventario` directamente desde el formulario; usa la acción.

## Bot para iniciar Salud después de validar

Conserva el Bot actual de creación de solicitud:

- Tabla: `SolicitudesAnalisis`
- Evento: Adds
- Condición: `[Estado] = "Pendiente"`
- Webhook: conserva el Apps Script y el cuerpo que ya utiliza.

Crea un segundo Bot:

- Nombre: `TerraCore - Salud después de validar inventario`
- Tabla: `SolicitudesAnalisis`
- Evento: Updates
- **Columns to watch:** únicamente `ValidacionInventario`.
- Condición: `[ValidacionInventario] = "Validado"`.
- Webhook: usa el mismo Apps Script Web App y el mismo formato de solicitud:

```json
{
  "action": "trigger",
  "request_id": "<<[SolicitudID]>>",
  "secret": "EL_MISMO_SECRETO_CONFIGURADO"
}
```

No cambies ni publiques un secreto diferente. El Bot debe enviar el mismo secreto ya sincronizado con Apps Script y GitHub Actions. Vigilar únicamente `ValidacionInventario` evita que los cambios de estado escritos por el worker vuelvan a disparar el Bot.

## Verificación

1. Crea una solicitud nueva desde AppSheet. No reutilices una fila anterior porque el Bot inicial está configurado para Adds.
2. Espera a que el estado sea `PendienteValidacion`.
3. Abre `InventarioSurcos`, filtra por `SolicitudID) y revisa cada fotografía.
4. Corrige conteos o extremos si es necesario.
5. Confirma con la acción **Confirmar Inventario**.
6. Espera el segundo Bot. El estado debe pasar a `ProcesandoSalud` y después a `Completado`.
7. Comprueba que el análisis final y los polígonos correspondan a las correcciones guardadas.

El backend vuelve a validar `Slots = Ocupados + Vacios`; si detecta una inconsistencia, no ejecuta Salud.
