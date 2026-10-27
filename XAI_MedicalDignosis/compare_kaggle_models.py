
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("/mnt/d/MCA_Final_Project/KDatasets")

MODEL_PATHS = {
    "BEST MODEL": BASE_DIR / "models" / "kaggle_chest_xray_best.keras",
    "FINAL MODEL": BASE_DIR / "models" / "kaggle_chest_xray_final.keras",
}

CLASS_NAMES = ["normal", "pneumonia", "tuberculosis"]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8


def load_validation_dataset():
    dataset = keras.utils.image_dataset_from_directory(
        DATA_DIR / "val",
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    return dataset.prefetch(1)


def main():
    print("Loading validation dataset...")
    dataset = load_validation_dataset()

    y_true = []
    for _, labels in dataset:
        y_true.extend(labels.numpy().tolist())

    print(f"Validation images: {len(y_true)}")
    print(f"Class order: {CLASS_NAMES}")

    for name, path in MODEL_PATHS.items():
        print("\n" + "=" * 60)
        print(name)
        print("=" * 60)

        if not path.exists():
            print(f"Model not found: {path}")
            continue

        model = keras.models.load_model(path, compile=False)

        predictions = model.predict(dataset, verbose=1)
        y_pred = predictions.argmax(axis=1)

        accuracy = accuracy_score(y_true, y_pred)
        print(f"\nValidation accuracy: {accuracy * 100:.2f}%")
        print("\nClassification report:")
        print(
            classification_report(
                y_true,
                y_pred,
                labels=[0, 1, 2],
                target_names=CLASS_NAMES,
                digits=4,
                zero_division=0,
            )
        )


if __name__ == "__main__":
    main()
