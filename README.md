# Fruits 360 Trainer & Predictor 🍎🍌

Este repositorio contiene las herramientas necesarias para entrenar un modelo de clasificación de imágenes utilizando el dataset **Fruits 360** mediante Docker y realizar inferencias en un entorno local de Conda.

---

## 🐳 Entrenamiento con Docker (GPU)

Para asegurar un entorno reproducible y aprovechar la aceleración por hardware (NVIDIA), utilizamos contenedores.

### 1. Construir la imagen

Ejecuta el siguiente comando en la raíz del proyecto para preparar el entorno de entrenamiento:

```bash
docker build -t fruits360-trainer .
2. Ejecutar el contenedor
El siguiente comando inicia el proceso de entrenamiento, vinculando tus carpetas locales de datos y resultados con el contenedor:

PowerShell
docker run --rm --gpus all `
  -e NVIDIA_VISIBLE_DEVICES=all `
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility `
  -v "${PWD}/data:/app/data" `
  -v "${PWD}/artifacts:/app/artifacts" `
  fruits360-trainer
Nota: Asegúrate de que las carpetas ./data y ./artifacts existan en tu directorio actual antes de correr el comando.

🔮 Predicciones (Entorno Local)
Si prefieres realizar las predicciones fuera del contenedor utilizando Conda, sigue estos pasos:

3. Crear el entorno virtual
Bash
conda create -n fruits_ml python=3.10 -y
4. Activar el entorno
Bash
conda activate fruits_ml
5. Instalar dependencias
Asegúrate de tener el archivo requirements.txt en la raíz:

Bash
pip install -r requirements.txt
6. Ejecutar inferencia
Una vez que el modelo esté entrenado y el archivo .h5 (o el formato que uses) esté generado, corre el script de predicción:

Bash
python predict.py
📂 Estructura de Carpetas
/data: Carpeta con las imágenes del dataset.

/artifacts: Ubicación donde se guardarán los pesos del modelo y logs.

predict.py: Script para probar el modelo con nuevas imágenes.

Dockerfile: Configuración del entorno de entrenamiento.
```
