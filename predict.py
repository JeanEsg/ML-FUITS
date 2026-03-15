import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from pathlib import Path

# --- 1. DEFINICIÓN DE LA ARQUITECTURA ---
def build_model_architecture(num_classes):
    model = models.Sequential([
        layers.Input(shape=(100, 100, 3)),
        layers.Rescaling(1./255),
        
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model

def run_prediction():
    # Rutas
    MODEL_PATH = Path('artifacts/best_model.keras')
    # Usamos la carpeta de Test o Training solo para obtener los nombres de las frutas
    DATA_DIR = Path('data/fruits-360/Training') 
    VALIDATION_DIR = Path('data/fruits-360/validation')
    
    # 2. OBTENER NOMBRES DE FRUTAS
    class_names = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
    num_classes = len(class_names)

    # 3. CARGA DEL MODELO
    model = build_model_architecture(num_classes)
    
    try:
        model.load_weights(str(MODEL_PATH))
    except Exception as e:
        print(f">>> Error al cargar el modelo: {e}")
        return

    # 4. SELECCIÓN DE IMAGEN
    # Buscamos en todas las subcarpetas de validation para elegir una imagen al azar
    all_images = list(VALIDATION_DIR.glob('**/*.*'))
    if not all_images:
        print(">>> No se encontraron imágenes en la carpeta de validación.")
        return
    
    img_path = random.choice(all_images)

    # 5. PREPROCESAMIENTO E INFERENCIA
    img = image.load_img(img_path, target_size=(100, 100))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    
    preds = model.predict(img_array, verbose=0)
    idx = np.argmax(preds[0])
    confianza = 100 * np.max(preds[0])

    # 6. RESULTADOS (LIMPIOS)
    clase_real = img_path.parent.name 
    prediccion = class_names[idx]

    print("="*45)
    # Mostramos solo el nombre del archivo, no la ruta completa para que sea más estético
    print(f"ARCHIVO:    {img_path.name}")
    print(f"CLASE REAL:   {clase_real.upper()}")
    print(f"PREDICCIÓN:   {prediccion.split()[0].upper()}")
    print(f"CONFIANZA:  {confianza:.2f}%")
    print("="*45 + "\n")

if __name__ == "__main__":
    run_prediction()