FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_backend.txt ./
RUN pip install --no-cache-dir -r requirements_backend.txt

# Solo se copian los tres módulos del backend. No se copian .env,
# Streamlit Secrets ni otros archivos del repositorio.
COPY backend_api.py appsheet_integration.py terracore_engine.py ./

RUN mkdir -p /app/uploads /app/outputs

EXPOSE 8080
CMD ["uvicorn", "backend_api:app", "--host", "0.0.0.0", "--port", "8080"]
