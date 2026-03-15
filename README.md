# Fruits 360 Trainer & Predictor 🍎🍌

Este repositorio contiene las herramientas necesarias para **entrenar un modelo de clasificación de imágenes** utilizando el dataset **Fruits 360**.  
El entrenamiento se realiza dentro de un **contenedor Docker con soporte GPU**, mientras que las **predicciones pueden ejecutarse en un entorno local usando Conda**.

---

# ⚠️ IMPORTANTE
Del data set de fruits-360 solo se uso una parte del dataset orignal que fueron las imagenes 100x100 excluyendo las otras paara que el entrenamiento no fuera tan pesado. 
También es neecsario tener el dataset antes de  todo dentro de la carpeta data, con la siguiente estructurapara garantizar un entrenamiento sin errores:
````bash
└───fruits-360
    ├───Papers
    ├───Test
    │   ├───...
    ├───Training
    │   ├───...
    └───Validation
        ├───...
````

---

# 📌 Características

- Entrenamiento reproducible utilizando **Docker**
- Soporte para **GPU NVIDIA**
- Separación entre **entrenamiento y predicción**
- Uso del dataset **Fruits 360** para clasificación de imágenes
- Predicciones locales mediante **Python + Conda**

---

# 🐳 Entrenamiento con Docker (GPU)

Para asegurar un entorno reproducible y aprovechar la aceleración por hardware (NVIDIA), se utilizan contenedores Docker.

## 1. Construir la imagen

Ejecuta el siguiente comando en la raíz del proyecto para crear la imagen de entrenamiento:

```bash
docker build -t fruits360-trainer .
````
## 2. Ejecutar el contenedor

Este comando iniciará el proceso de entrenamiento y montará las carpetas locales de datos y resultados dentro del contenedor.

```bash
docker run --rm --gpus all `
  -e NVIDIA_VISIBLE_DEVICES=all `
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility `
  -v "${PWD}/data:/app/data" `
  -v "${PWD}/artifacts:/app/artifacts" `
  fruits360-trainer
````
⚠️ Importante:
Asegúrate de que las carpetas data y artifacts existan en el directorio raíz antes de ejecutar el comando.

---

# 🔮 Predicciones (Entorno Local)
Para ejecutar las predicciones fuera del contenedor usando Conda, sigue estos pasos.

## 3. Crear el entorno virtual
```bash
conda create -n fruits_ml python=3.10 -y
````

## 4. Activar el entorno
````bash
conda activate fruits_ml
````

## 5. Instalar dependencias
Asegúrate de tener el archivo requirements.txt en la raíz del proyecto.
````bash
pip install -r requirements.txt
````

## 6. Ejecutar predcción
Una vez que el modelo esté entrenado y el archivo del modelo (por ejemplo .h5) se encuentre disponible, ejecuta:
````bash
python predict.py
````

---

# 📂 Estructura del Proyecto
````bash
.
├── data/          # Dataset de imágenes Fruits 360
├── artifacts/     # Pesos del modelo, métricas y logs
├── predict.py     # Script para realizar predicciones
├── Dockerfile     # Configuración del entorno de entrenamiento
└── requirements.txt
````
