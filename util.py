import os
from pathlib import Path

def check_dataset_consistency(train_path, test_path):
    train_dir = Path(train_path)
    test_dir = Path(test_path)

    train_classes = sorted([f.name for f in train_dir.iterdir() if f.is_dir()])
    test_classes = sorted([f.name for f in test_dir.iterdir() if f.is_dir()])

    print(f"--- Análisis de consistencia ---")
    print(f"Clases en Training: {len(train_classes)}")
    print(f"Clases en Test:     {len(test_classes)}")

    # 1. Buscar clases que están en uno pero no en otro
    only_in_train = set(train_classes) - set(test_classes)
    only_in_test = set(test_classes) - set(train_classes)

    if only_in_train:
        print(f"\n⚠️  ERROR: Estas carpetas están en TRAINING pero NO en TEST:\n{only_in_train}")
    
    if only_in_test:
        print(f"\n⚠️  ERROR: Estas carpetas están en TEST pero NO en TRAINING:\n{only_in_test}")

    # 2. Verificar el orden (Crucial para Keras)
    if train_classes == test_classes:
        print("\n✅ ¡Perfecto! Las carpetas coinciden exactamente en nombre y orden.")
    else:
        print("\n❌ DESFASE DETECTADO: El orden o los nombres no coinciden.")

# Ajusta estas rutas a tu PC local
check_dataset_consistency(
    "C:/Users/ASUS/Desktop/ML-FUITS/data/fruits-360/Training",
    "C:/Users/ASUS/Desktop/ML-FUITS/data/fruits-360/Test"
)