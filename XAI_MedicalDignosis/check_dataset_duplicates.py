
import os
import hashlib
from collections import defaultdict

DATASET_DIR = "/mnt/d/MCA_Final_Project/KDatasets"
SPLITS = ["train", "val", "test"]
CLASSES = ["normal", "pneumonia", "tuberculosis"]

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

hash_locations = defaultdict(list)
invalid_files = []
class_counts = defaultdict(int)

for split in SPLITS:
    for class_name in CLASSES:
        folder = os.path.join(DATASET_DIR, split, class_name)

        if not os.path.isdir(folder):
            print("MISSING FOLDER:", folder)
            continue

        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)

            if not os.path.isfile(path):
                continue

            try:
                file_hash = sha256_file(path)
                hash_locations[file_hash].append(
                    (split, class_name, path)
                )
                class_counts[(split, class_name)] += 1
            except OSError as error:
                invalid_files.append((path, str(error)))

print("\nIMAGE COUNTS")
for split in SPLITS:
    for class_name in CLASSES:
        print(
            f"{split:5s} | {class_name:12s} | "
            f"{class_counts[(split, class_name)]}"
        )

duplicates = {
    digest: locations
    for digest, locations in hash_locations.items()
    if len(locations) > 1
}

cross_split = []
cross_class = []

for locations in duplicates.values():
    splits = {item[0] for item in locations}
    classes = {item[1] for item in locations}

    if len(splits) > 1:
        cross_split.append(locations)

    if len(classes) > 1:
        cross_class.append(locations)

print("\nDUPLICATE CHECK")
print("Unique file hashes:", len(hash_locations))
print("Duplicate hash groups:", len(duplicates))
print("Groups spanning multiple splits:", len(cross_split))
print("Groups spanning multiple classes:", len(cross_class))
print("Unreadable files:", len(invalid_files))

os.makedirs("results", exist_ok=True)

with open("results/dataset_duplicate_report.txt", "w") as f:
    f.write(f"Unique file hashes: {len(hash_locations)}\n")
    f.write(f"Duplicate hash groups: {len(duplicates)}\n")
    f.write(f"Groups spanning splits: {len(cross_split)}\n")
    f.write(f"Groups spanning classes: {len(cross_class)}\n")
    f.write(f"Unreadable files: {len(invalid_files)}\n\n")

    for title, groups in [
        ("DUPLICATES", duplicates.values()),
        ("CROSS-SPLIT GROUPS", cross_split),
        ("CROSS-CLASS GROUPS", cross_class),
    ]:
        f.write(f"\n{title}\n")
        for locations in groups:
            for split, class_name, path in locations:
                f.write(f"{split} | {class_name} | {path}\n")
            f.write("\n")

    for path, error in invalid_files:
        f.write(f"UNREADABLE: {path} | {error}\n")

print("\nReport saved to results/dataset_duplicate_report.txt")
