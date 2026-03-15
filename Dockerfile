# 1. Usar la imagen oficial de TensorFlow configurada para GPU
# Esta ya incluye Python, CUDA y cuDNN preinstalados.
FROM tensorflow/tensorflow:2.15.0-gpu

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 2. Instalamos dependencias del sistema necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. Instalamos las librerías de Python
COPY requirements.txt .
RUN pip install --upgrade pip
# NOTA: Asegúrate de que en requirements.txt NO diga 'tensorflow-cpu'
# Debe decir simplemente 'tensorflow' o nada (ya viene en la imagen base)
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .

RUN mkdir -p /app/data /app/artifacts

CMD ["python", "train.py"]