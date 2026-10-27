
import os
import csv
import numpy as np
import tensorflow as tf

DATASET_DIR = "/mnt/d/MCA_Final_Project/KDatasets"
MODEL_PATHS = {
    "Original": "models/kaggle_chest_xray_best.keras",
    "V2": "models/kaggle_chest_xray_v2_best.keras",
}

CLASS_NAMES = ["normal", "pneumonia", "tuberculosis"]

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "test"),
    class_names=CLASS_NAMES,
    image_size=(224, 224),
    batch_size=8,
    shuffle=False,
    label_mode="int",
)

# Save paths and labels before prefetching.
image_paths = test_ds.file_paths
y_true = np.concatenate([y.numpy() for _, y in test_ds])
test_ds = test_ds.prefetch(1)

os.makedirs("results", exist_ok=True)

for model_name, model_path in MODEL_PATHS.items():
    print(f"\nEvaluating {model_name}...")

    model = tf.keras.models.load_model(model_path, compile=False)
    scores = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(scores, axis=1)

    output_csv = f"results/{model_name.lower()}_misclassifications.csv"

    with open(output_csv, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "image_path", "actual_class", "predicted_class",
            "normal_score", "pneumonia_score", "tuberculosis_score"
        ])

        for i in range(len(y_true)):
            if y_true[i] != y_pred[i]:
                writer.writerow([
                    image_paths[i],
                    CLASS_NAMES[y_true[i]],
                    CLASS_NAMES[y_pred[i]],
                    *[round(float(x), 6) for x in scores[i]]
                ])

    print(f"Saved: {output_csv}")
    print("Total errors:", int(np.sum(y_true != y_pred)))

    # Focus on the Normal vs Tuberculosis confusion.
    normal_to_tb = np.sum((y_true == 0) & (y_pred == 2))
    tb_to_normal = np.sum((y_true == 2) & (y_pred == 0))

    print("Normal predicted as TB:", int(normal_to_tb))
    print("TB predicted as Normal:", int(tb_to_normal))
