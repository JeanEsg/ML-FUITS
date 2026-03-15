import os
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from itertools import cycle

# --- CONFIGURACIÓN ---
SEED = 42
IMG_SIZE = 100
BATCH_SIZE = 32  
EPOCHS = 30
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.2

DATASET_DIR = Path(os.getenv("DATASET_DIR", "/app/data/fruits-360"))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "/app/artifacts"))
TRAIN_DIR = DATASET_DIR / "Training"
TEST_DIR = DATASET_DIR / "Test"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
tf.random.set_seed(SEED)

# --- 1. PIPELINE DE DATOS CORREGIDO ---
def build_datasets():
    # 1. Carga inicial (Aquí es donde residen los class_names)
    base_train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    base_val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    base_test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )

    # EXTRAEMOS LOS NOMBRES ANTES DE TRANSFORMAR EL DATASET
    class_names = base_train_ds.class_names

    # 2. OPTIMIZACIÓN (Cache y Prefetch)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = base_train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = base_val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = base_test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names

# --- 2. MODELO ---
def create_optimized_model(num_classes):
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ])

    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        data_augmentation,
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

def save_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 20)) # Grande para que quepan las 138 clases
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión - Fruits 360')
    plt.ylabel('Clase Real')
    plt.xlabel('Predicción')
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "confusion_matrix.png")
    plt.close()

def save_roc_curves(y_true, y_score, n_classes, class_names):
    # Binarizar las etiquetas para multiclase
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Calcular ROC por cada clase
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Calcular Micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Graficar
    plt.figure(figsize=(10, 8))
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average ROC (area = {roc_auc["micro"]:0.2f})',
             color='deeppink', linestyle=':', linewidth=4)

    # Graficar las primeras 3 clases como ejemplo
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC de {class_names[i]} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curvas ROC - Clasificación de Frutas')
    plt.legend(loc="lower right")
    plt.savefig(ARTIFACTS_DIR / "roc_curves.png")
    plt.close()

def save_learning_curves(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    # Gráfica de Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'g', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
    plt.title('Curva de Aprendizaje - Precisión')
    plt.xlabel('Épocas')
    plt.ylabel('Exactitud')
    plt.legend()

    # Gráfica de Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'g', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Curva de Aprendizaje - Pérdida')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()

    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "learning_curves.png")
    plt.close()

# --- 3. MAIN ---
def main():

    # --- CONFIGURACIÓN DE MEMORIA GPU (Pégalo aquí) ---
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(">>> Memoria dinámica de GPU habilitada")
        except RuntimeError as e:
            print(f">>> Error configurando GPU: {e}")
    else:
        print(">>> ADVERTENCIA: No se detectó GPU. El entrenamiento será LENTO.")
    # --------------------------------------------------
    # Cargar datos y obtener nombres de clases
    train_ds, val_ds, test_ds, class_names = build_datasets()
    num_classes = len(class_names)

    print(f"\nClases detectadas: {num_classes}")
    
    model = create_optimized_model(num_classes)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3), # <--- ADD THIS COMMA
        ModelCheckpoint(str(ARTIFACTS_DIR / "best_model.keras"), save_best_only=True),
        tf.keras.callbacks.CSVLogger(
            str(ARTIFACTS_DIR / "training_log.csv"), 
            append=True
        ) # Added a closing paren here too just in case
    ]

    print("\nEntrenando...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Evaluación y Reporte
    print("\nEvaluando...")
    # Para el reporte de sklearn, necesitamos las etiquetas reales
    y_true = []
    y_pred = []
    
    for x, y in test_ds:
        y_true.extend(np.argmax(y.numpy(), axis=1))
        preds = model.predict(x, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))

    print("\nREPORT DE CLASIFICACIÓN:")

    # Creamos una lista de índices del 0 al 137
    all_possible_labels = list(range(len(class_names)))

    print(classification_report(y_true, y_pred, labels=all_possible_labels ,target_names=class_names, zero_division=0))

    print("\nGenerando matriz de confusión...")
    save_confusion_matrix(y_true, y_pred, class_names)

    print("\nGuardando resultados en JSON...")
    # 1. Guardar el historial de entrenamiento
    history_dict = {
        "accuracy": [float(x) for x in history.history["accuracy"]],
        "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
        "loss": [float(x) for x in history.history["loss"]],
        "val_loss": [float(x) for x in history.history["val_loss"]],
    }
    with open(ARTIFACTS_DIR / "training_history.json", "w") as f:
        json.dump(history_dict, f, indent=2)

    # 2. Guardar el reporte de clasificación detallado
    report_dict = classification_report(
        y_true, 
        y_pred,
        labels= all_possible_labels, 
        target_names=class_names, 
        output_dict=True,
        zero_division=0
    )
    with open(ARTIFACTS_DIR / "classification_report.json", "w") as f:
        json.dump(report_dict, f, indent=2)

    print(f">>> Éxito: Archivos guardados en {ARTIFACTS_DIR}")

    print("\nCalculando predicciones detalladas para métricas...")
    y_true = []
    y_score = [] # Aquí guardamos las probabilidades
    
    for x, y in test_ds:
        y_true.extend(np.argmax(y.numpy(), axis=1))
        preds = model.predict(x, verbose=0)
        y_score.extend(preds)
    
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_pred = np.argmax(y_score, axis=1)

    # Generar todo
    save_learning_curves(history)
    save_roc_curves(y_true, y_score, num_classes, class_names)
    save_confusion_matrix(y_true, y_pred, class_names) # La que definimos antes

if __name__ == "__main__":
    main()