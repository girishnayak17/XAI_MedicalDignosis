
from pathlib import Path
import hashlib

old = Path("/mnt/d/MCA_Final_Project/KDatasets")
new = Path("/mnt/d/MCA_Final_Project/PDatasets/chest_xray")

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

print("Scanning original dataset...", flush=True)

old_hashes = {}
count = 0

for split in ("train", "val", "test"):
    for p in (old / split).rglob("*"):
        if p.is_file():
            old_hashes.setdefault(digest(p), []).append(str(p))
            count += 1

            if count % 1000 == 0:
                print(f"Original files checked: {count}", flush=True)

print(f"Original scan complete: {count} files", flush=True)
print("Scanning new training dataset...", flush=True)

overlap_count = 0
new_count = 0

for p in (new / "train").rglob("*"):
    if not p.is_file():
        continue

    new_count += 1

    if digest(p) in old_hashes:
        overlap_count += 1
        print(f"Exact duplicate found: {p}", flush=True)

    if new_count % 500 == 0:
        print(f"New files checked: {new_count}", flush=True)

print("\nFinished.")
print("New training files checked:", new_count)
print("Files matching the original dataset:", overlap_count)
