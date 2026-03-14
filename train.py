import os
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report


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

np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)


def validate_dataset():
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"No existe TRAIN_DIR: {TRAIN_DIR}")
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"No existe TEST_DIR: {TEST_DIR}")

    train_classes = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
    test_classes = sorted([d.name for d in TEST_DIR.iterdir() if d.is_dir()])

    if not train_classes:
        raise RuntimeError("No se encontraron clases en Training/")
    if not test_classes:
        raise RuntimeError("No se encontraron clases en Test/")

    print("=" * 60)
    print("DATASET CHECK")
    print("=" * 60)
    print(f"Dataset dir: {DATASET_DIR}")
    print(f"Training dir: {TRAIN_DIR}")
    print(f"Test dir: {TEST_DIR}")
    print(f"Clases en train: {len(train_classes)}")
    print(f"Clases en test: {len(test_classes)}")
    print(f"Primeras 10 clases: {train_classes[:10]}")

    return train_classes


def build_generators():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=VALIDATION_SPLIT,
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=SEED,
    )

    validation_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=SEED,
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    return train_generator, validation_generator, test_generator


def create_cnn_model(input_shape=(100, 100, 3), num_classes=131):
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),

            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),

            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),

            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),

            layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),

            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="Fruits360_CNN",
    )
    return model


def main():
    classes = validate_dataset()

    print("=" * 60)
    print("ENVIRONMENT")
    print("=" * 60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")

    train_gen, val_gen, test_gen = build_generators()
    num_classes = train_gen.num_classes
    class_names = list(train_gen.class_indices.keys())

    print("=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"Train samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Test samples: {test_gen.samples}")
    print(f"Num classes: {num_classes}")

    model = create_cnn_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=num_classes)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    best_model_path = ARTIFACTS_DIR / "best_fruit_model.keras"

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1,
            mode="min",
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1,
            mode="min",
        ),
        ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
    ]

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )

    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)

    best_model = keras.models.load_model(best_model_path)
    test_loss, test_accuracy = best_model.evaluate(test_gen, verbose=1)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test accuracy %: {test_accuracy * 100:.2f}%")

    test_gen.reset()
    predictions = best_model.predict(test_gen, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_gen.classes

    report = classification_report(
        true_classes,
        predicted_classes,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    history_dict = {
        "accuracy": [float(x) for x in history.history["accuracy"]],
        "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
        "loss": [float(x) for x in history.history["loss"]],
        "val_loss": [float(x) for x in history.history["val_loss"]],
    }

    summary = {
        "dataset_dir": str(DATASET_DIR),
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "num_classes": num_classes,
        "train_samples": train_gen.samples,
        "validation_samples": val_gen.samples,
        "test_samples": test_gen.samples,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "target_passed": bool(test_accuracy > 0.80),
        "model_path": str(best_model_path),
    }

    with open(ARTIFACTS_DIR / "training_history.json", "w") as f:
        json.dump(history_dict, f, indent=2)

    with open(ARTIFACTS_DIR / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    with open(ARTIFACTS_DIR / "results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print("ARTIFACTS SAVED")
    print("=" * 60)
    for p in ARTIFACTS_DIR.iterdir():
        if p.is_file():
            print(f"- {p.name}")


if __name__ == "__main__":
    main()