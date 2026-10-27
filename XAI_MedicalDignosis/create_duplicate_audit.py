
from pathlib import Path
from collections import defaultdict
import hashlib
import csv

DATASET_DIR = Path("/mnt/d/MCA_Final_Project/KDatasets")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Group images by exact file content.
hash_groups = defaultdict(list)

for split in ["train", "val", "test"]:
    for class_dir in sorted((DATASET_DIR / split).iterdir()):
        if not class_dir.is_dir():
            continue

        for image_path in class_dir.iterdir():
            if not image_path.is_file():
                continue

            try:
                digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
                hash_groups[digest].append({
                    "split": split,
                    "label": class_dir.name,
                    "path": str(image_path.resolve()),
                })
            except OSError as exc:
                print(f"Could not read {image_path}: {exc}")

# Save all images and their duplicate-group information.
csv_path = OUTPUT_DIR / "duplicate_audit.csv"

with csv_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "sha256",
            "duplicate_count",
            "conflicting_labels",
            "cross_split",
            "split",
            "label",
            "path",
        ],
    )
    writer.writeheader()

    for digest, records in sorted(hash_groups.items()):
        labels = sorted({record["label"] for record in records})
        splits = sorted({record["split"] for record in records})

        for record in records:
            writer.writerow({
                "sha256": digest,
                "duplicate_count": len(records),
                "conflicting_labels": len(labels) > 1,
                "cross_split": len(splits) > 1,
                **record,
            })

print(f"Unique image contents: {len(hash_groups)}")
print(f"Detailed audit saved to: {csv_path.resolve()}")
print("Original dataset was not modified.")
