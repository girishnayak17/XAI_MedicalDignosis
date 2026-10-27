
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("/mnt/d/MCA_Final_Project/KDatasets")

MODELS = {
    "Original best": BASE_DIR / "models" / "kaggle_chest_xray_best.keras",
    "New V2 best": BASE_DIR / "models" / "kaggle_chest_xray_v2_best.keras",
}

CLASSES = ["normal", "pneumonia", "tuberculosis"]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8


def load_dataset(split):
    ds = keras.utils.image_dataset_from_directory(
        DATA_DIR / split,
        labels="inferred",
        label_mode="int",
        class_names=CLASSES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    return ds.prefetch(1)


def evaluate(model_path, dataset, y_true, split):
    print(f"\n{'=' * 60}")
    print(f"{model_path.name} | {split.upper()}")
    print("=" * 60)

    model = keras.models.load_model(model_path, compile=False)
    scores = model.predict(dataset, verbose=1)
    y_pred = np.argmax(scores, axis=1)

    print(f"Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=CLASSES,
            digits=4,
            zero_division=0,
        )
    )


def main():
    for split in ["val", "test"]:
        dataset = load_dataset(split)
        y_true = np.concatenate(
            [labels.numpy() for _, labels in dataset]
        )

        for name, path in MODELS.items():
            if path.exists():
                print(f"\nMODEL: {name}")
                evaluate(path, dataset, y_true, split)
            else:
                print(f"\nMissing model: {path}")


if __name__ == "__main__":
    main()
