from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pathlib import Path
from datetime import datetime
from PIL import Image
from dotenv import load_dotenv
import shutil
import uuid
import os

# =========================
# CARGAR VARIABLES
# =========================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# =========================
# RUTAS
# =========================
BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"

UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# =========================
# APP
# =========================
app = FastAPI(title="Backend Viñedo TerraCore")

app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")


# =========================
# FUNCIONES AUXILIARES
# =========================
def guardar_imagen(upload_file: UploadFile) -> Path:
    """
    Guarda la imagen subida en uploads/
    """
    ext = Path(upload_file.filename).suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        raise HTTPException(status_code=400, detail="Formato no permitido. Usa JPG, PNG o WEBP.")

    nombre = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{ext}"
    destino = UPLOADS_DIR / nombre

    with destino.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    return destino


def validar_imagen(ruta: Path):
    """
    Verifica que sí sea una imagen válida
    """
    try:
        with Image.open(ruta) as img:
            img.verify()
    except Exception:
        if ruta.exists():
            ruta.unlink()
        raise HTTPException(status_code=400, detail="El archivo no es una imagen válida.")


def copiar_a_outputs(ruta_original: Path) -> Path:
    """
    Copia la imagen a outputs/ como resultado temporal
    """
    nombre_salida = f"resultado_{ruta_original.name}"
    ruta_salida = OUTPUTS_DIR / nombre_salida
    shutil.copy(ruta_original, ruta_salida)
    return ruta_salida


def analisis_base_mock(ruta_imagen: Path) -> dict:
    """
    Análisis temporal de prueba.
    En el Paso 3 aquí metemos OpenAI.
    """
    with Image.open(ruta_imagen) as img:
        ancho, alto = img.size

    return {
        "mensaje": "Imagen recibida correctamente",
        "ancho": ancho,
        "alto": alto,
        "tipo_analisis": "prueba_backend",
        "observacion": "Backend funcionando. En el siguiente paso aquí conectamos la IA."
    }


# =========================
# ENDPOINTS
# =========================
@app.get("/")
def inicio():
    return {
        "ok": True,
        "mensaje": "Backend TerraCore activo"
    }


@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    """
    Recibe una imagen, la guarda y devuelve un resultado base
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No se recibió ningún archivo.")

    ruta_imagen = guardar_imagen(file)
    validar_imagen(ruta_imagen)

    # análisis temporal
    analisis = analisis_base_mock(ruta_imagen)

    # copiamos la imagen como salida temporal
    ruta_salida = copiar_a_outputs(ruta_imagen)

    return JSONResponse({
        "ok": True,
        "archivo_original": ruta_imagen.name,
        "imagen_original_url": f"/uploads/{ruta_imagen.name}",
        "imagen_resultado_url": f"/outputs/{ruta_salida.name}",
        "analisis": analisis
    })
