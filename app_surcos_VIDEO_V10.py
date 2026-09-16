import streamlit as st
import requests
import pandas as pd
from dotenv import load_dotenv
import os

# =========================================================
# CONFIG
# =========================================================

load_dotenv()

BACKEND_ANALYZE_URL = os.getenv(
    "BACKEND_ANALYZE_URL",
    "http://localhost:8000/analyze-image"
).strip()

st.set_page_config(
    page_title="TerraCore - Conteo de Surcos con IA",
    layout="wide"
)

# =========================================================
# ESTILOS
# =========================================================

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: Arial, sans-serif;
}
.main {
    background: linear-gradient(180deg, #7a2933 0%, #6a2430 100%);
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
.titulo-principal {
    font-size: 34px;
    font-weight: 700;
    color: white;
    margin-bottom: 5px;
}
.subtitulo {
    font-size: 16px;
    color: #f2eaea;
    margin-bottom: 25px;
}
.card {
    background: rgba(255,255,255,0.07);
    padding: 18px;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.12);
}
.small-note {
    color: #f2eaea;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPERS
# =========================================================

def get_backend_base(url: str) -> str:
    if "/analyze-image" in url:
        return url.split("/analyze-image")[0]
    return url.rstrip("/")


def check_backend():
    base = get_backend_base(BACKEND_ANALYZE_URL)
    try:
        r = requests.get(base, timeout=30)
        if r.status_code == 200:
            return True, r.json()
        return False, {"error": f"HTTP {r.status_code}"}
    except Exception as e:
        return False, {"error": str(e)}


def analizar_una_imagen(uploaded_file):
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type if uploaded_file.type else "application/octet-stream"
        )
    }

    response = requests.post(
        BACKEND_ANALYZE_URL,
        files=files,
        timeout=600
    )

    try:
        data = response.json()
    except Exception:
        data = {"ok": False, "error": response.text}

    return response.status_code, data


# =========================================================
# UI
# =========================================================

st.markdown('<div class="titulo-principal">🌿 TerraCore - Conteo de Surcos con IA</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitulo">Sube imágenes del viñedo. El backend las envía a IA, la IA dibuja líneas verde/rojo, numera surcos arriba y abajo, y también devuelve conteo y porcentajes.</div>',
    unsafe_allow_html=True
)

col_a, col_b = st.columns([2, 1])

with col_a:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("### Cargar imágenes")
    archivos = st.file_uploader(
        "Selecciona una o varias imágenes",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True
    )
    analizar_btn = st.button("🔍 Analizar imágenes con IA", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_b:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("### Estado del backend")
    st.caption(BACKEND_ANALYZE_URL)

    ok_backend, info_backend = check_backend()
    if ok_backend:
        st.success("Backend conectado")
        st.json(info_backend)
    else:
        st.error("No se pudo conectar al backend")
        st.json(info_backend)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# PROCESAR
# =========================================================

if "resultados_rows" not in st.session_state:
    st.session_state["resultados_rows"] = []

if "resultados_imgs" not in st.session_state:
    st.session_state["resultados_imgs"] = []

if analizar_btn:
    st.session_state["resultados_rows"] = []
    st.session_state["resultados_imgs"] = []

    if not archivos:
        st.warning("Primero sube al menos una imagen.")
    else:
        progress = st.progress(0)
        total = len(archivos)

        for i, archivo in enumerate(archivos, start=1):
            with st.spinner(f"Procesando {archivo.name}..."):
                status_code, data = analizar_una_imagen(archivo)

                if status_code == 200 and data.get("ok"):
                    analisis = data.get("analisis", {})

                    fila = {
                        "Imagen": archivo.name,
                        "Surcos estimados": analisis.get("surcos_estimados", 0),
                        "Verde %": f"{analisis.get('verde_pct', 0)}%",
                        "Rojo %": f"{analisis.get('rojo_pct', 0)}%",
                    }
                    st.session_state["resultados_rows"].append(fila)

                    st.session_state["resultados_imgs"].append({
                        "nombre": archivo.name,
                        "imagen_original": data.get("imagen_original_url_publica", ""),
                        "imagen_resultado": data.get("imagen_resultado_url_publica", ""),
                        "surcos": analisis.get("surcos_estimados", 0),
                        "verde_pct": analisis.get("verde_pct", 0),
                        "rojo_pct": analisis.get("rojo_pct", 0),
                        "resumen": analisis.get("resumen", "")
                    })
                else:
                    error_msg = data.get("error", f"HTTP {status_code}")
                    st.error(f"❌ Error en {archivo.name}: {error_msg}")

            progress.progress(i / total)

# =========================================================
# MOSTRAR RESULTADOS
# =========================================================

if st.session_state["resultados_rows"]:
    st.write("")
    st.write("## Resultados por escena / parcela candidata")

    df = pd.DataFrame(st.session_state["resultados_rows"])
    st.dataframe(df, use_container_width=True)

    st.write("")
    st.write("## Imágenes analizadas")

    for item in st.session_state["resultados_imgs"]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write(f"### {item['nombre']}")
        st.write(
            f"**Surcos estimados:** {item['surcos']} | "
            f"**Verde %:** {item['verde_pct']}% | "
            f"**Rojo %:** {item['rojo_pct']}%"
        )
        if item["resumen"]:
            st.caption(item["resumen"])

        col1, col2 = st.columns(2)

        with col1:
            if item["imagen_original"]:
                st.image(item["imagen_original"], caption="Imagen original", use_container_width=True)

        with col2:
            if item["imagen_resultado"]:
                st.image(item["imagen_resultado"], caption="Imagen procesada con IA", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)
        st.write("")
