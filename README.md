Fruits 360 Trainer & Predictor 🍎🍌
Este proyecto permite entrenar un modelo de clasificación de imágenes utilizando el dataset Fruits 360 dentro de un entorno Dockerizado con soporte para GPU, además de realizar predicciones mediante un entorno de Conda.

🚀 Requisitos Previos
Docker instalado.

NVIDIA Container Toolkit (para soporte de GPU en Docker).

Conda (Miniconda o Anaconda).

🐳 Entrenamiento con Docker (GPU)
Para asegurar un entorno limpio y aprovechar la aceleración por hardware, utilizamos Docker.

1. Construir la imagen
Desde la raíz del proyecto, ejecuta:

Bash
docker build -t fruits360-trainer .
2. Ejecutar el entrenamiento
El siguiente comando monta los volúmenes para los datos y los artefactos (modelos guardados), y habilita el uso de todas las GPUs disponibles:

PowerShell
docker run --rm --gpus all `
  -e NVIDIA_VISIBLE_DEVICES=all `
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility `
  -v "${PWD}/data:/app/data" `
  -v "${PWD}/artifacts:/app/artifacts" `
  fruits360-trainer
Nota: Asegúrate de tener las carpetas /data (con las imágenes) y /artifacts creadas en tu directorio actual.

🔮 Predicciones (Entorno Local con Conda)
Una vez que el modelo esté entrenado, puedes realizar predicciones localmente.

3. Crear el entorno virtual
Bash
conda create -n fruits_ml python=3.10 -y
4. Activar el entorno
Bash
conda activate fruits_ml
5. Instalar dependencias
Bash
pip install -r requirements.txt
6. Ejecutar inferencia
Puedes usar el script predict.py para probar el modelo con nuevas imágenes:

Bash
python predict.py
Asegúrate de que el archivo del modelo entrenado esté en la ruta que espera el script.

📂 Estructura del Proyecto
/data: Imágenes para entrenamiento y validación.

/artifacts: Directorio donde se guardan los modelos .h5 o .pth.

predict.py: Script para realizar inferencias.

Dockerfile: Configuración del contenedor de entrenamiento.
