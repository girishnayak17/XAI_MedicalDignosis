
from pathlib import Path
import csv
import hashlib
import shutil
import sys
from collections import Counter

PROJECT_DIR = Path(
    "/mnt/d/MCA_Final_Project/XAI_MedicalDignosis"
)
MANIFEST = PROJECT_DIR / "results" / "clean_dataset_proposal.csv"
OUTPUT = Path("/mnt/d/MCA_Final_Project/KDatasets_Clean")

if not MANIFEST.is_file():
    sys.exit(f"Missing proposal CSV: {MANIFEST}")

# Never overwrite an existing clean dataset.
if OUTPUT.exists():
    sys.exit(
        f"{OUTPUT} already exists. Nothing was changed. "
        "Choose a new output directory before proceeding."
    )

with MANIFEST.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

if not rows:
    sys.exit("The proposal CSV contains no images.")

expected_classes = {"normal", "pneumonia", "tuberculosis"}
expected_splits = {"train", "val", "test"}

for row in rows:
    if row["label"] not in expected_classes:
        sys.exit(f"Unexpected label: {row['label']}")
    if row["split"] not in expected_splits:
        sys.exit(f"Unexpected split: {row['split']}")
    if not Path(row["path"]).is_file():
        sys.exit(f"Source image missing: {row['path']}")

OUTPUT.mkdir(parents=True)

counts = Counter()
hashes_by_split = {s: set() for s in expected_splits}

try:
    for row in rows:
        source = Path(row["path"])
        digest = row["sha256"]

        # Use a hash prefix to prevent filename collisions.
        destination_dir = OUTPUT / row["split"] / row["label"]
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"{digest[:16]}_{source.name}"

        shutil.copy2(source, destination)

        # Verify the copied file matches the expected content.
        copied_hash = hashlib.sha256(
            destination.read_bytes()
        ).hexdigest()

        if copied_hash != digest:
            raise RuntimeError(
                f"Hash mismatch after copying: {destination}"
            )

        hashes_by_split[row["split"]].add(copied_hash)
        counts[(row["split"], row["label"])] += 1

    # No exact image content may appear in multiple splits.
    splits = list(expected_splits)
    for i, split_a in enumerate(splits):
        for split_b in splits[i + 1:]:
            overlap = hashes_by_split[split_a] & hashes_by_split[split_b]
            if overlap:
                raise RuntimeError(
                    f"Hash overlap between {split_a} and {split_b}"
                )

except Exception:
    print(
        "Build failed. The original dataset was not modified. "
        f"Inspect the partial output at {OUTPUT} before retrying."
    )
    raise

print("\nClean dataset copy created and hash-verified.")
print(f"Location: {OUTPUT}")
print("\nImage counts:")

for split in ("train", "val", "test"):
    for label in ("normal", "pneumonia", "tuberculosis"):
        print(f"{split:5s} | {label:15s} | {counts[(split, label)]}")

print("\nExact-hash overlap between splits: 0")
print(
    "Reminder: this process does not independently validate "
    "the medical labels."
)
