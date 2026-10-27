
from pathlib import Path

import pandas as pd

from preprocessing import (
    load_metadata,
    load_image_list,
    create_split_dataframe,
    DISEASE_LABELS,
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR.parent / "Datasets"
TRAIN_LIST = DATASET_DIR / "train_val_list.txt"

metadata = load_metadata()
train_images = load_image_list(TRAIN_LIST)
train_df = create_split_dataframe(train_images, metadata)

total = len(train_df)
rows = []

for label in DISEASE_LABELS:
    positive_count = train_df["Finding Labels"].apply(
        lambda value: label in value.split("|")
    ).sum()

    negative_count = total - positive_count

    rows.append({
        "Label": label,
        "Positive": int(positive_count),
        "Negative": int(negative_count),
        "Positive_Percent": round(
            100 * positive_count / total, 2
        ),
        "Negative_to_Positive_Ratio": round(
            negative_count / max(positive_count, 1), 2
        ),
    })

result = pd.DataFrame(rows)
result = result.sort_values("Positive_Percent")

print("\n=== TRAINING-POOL CLASS IMBALANCE ===")
print(result.to_string(index=False))

output_path = BASE_DIR / "results" / "class_imbalance.csv"
result.to_csv(output_path, index=False)

print(f"\nSaved report: {output_path}")
