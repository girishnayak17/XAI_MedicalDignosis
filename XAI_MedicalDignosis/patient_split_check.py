import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

CSV_PATH = Path("../Datasets/Data_Entry_2017_v2020.csv")
TRAIN_FILE = Path("../Datasets/train_val_list.txt")
TEST_FILE = Path("../Datasets/test_list.txt")


# ============================================================
# HELPER FUNCTION
# ============================================================

def read_image_list(file_path):
    """Read image filenames from train/test list."""

    with open(file_path, "r") as file:
        images = {line.strip() for line in file if line.strip()}

    return images


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("NIH CHEST X-RAY PATIENT-LEVEL SPLIT CHECK")
print("=" * 70)


# ------------------------------------------------------------
# 1. Check files
# ------------------------------------------------------------

print("\nChecking required files...")

if not CSV_PATH.exists():
    print(f"❌ CSV file not found: {CSV_PATH}")
    exit()

if not TRAIN_FILE.exists():
    print(f"❌ Training list not found: {TRAIN_FILE}")
    exit()

if not TEST_FILE.exists():
    print(f"❌ Test list not found: {TEST_FILE}")
    exit()

print("✓ All required files found")


# ------------------------------------------------------------
# 2. Load CSV
# ------------------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(CSV_PATH)

print("✓ Dataset loaded")
print(f"Total records: {len(df)}")


# ------------------------------------------------------------
# 3. Create image → patient mapping
# ------------------------------------------------------------

image_to_patient = dict(
    zip(
        df["Image Index"],
        df["Patient ID"]
    )
)


# ------------------------------------------------------------
# 4. Create image → labels mapping
# ------------------------------------------------------------

image_to_labels = dict(
    zip(
        df["Image Index"],
        df["Finding Labels"]
    )
)


# ------------------------------------------------------------
# 5. Read train and test image lists
# ------------------------------------------------------------

train_images = read_image_list(TRAIN_FILE)
test_images = read_image_list(TEST_FILE)

print("\n" + "=" * 70)
print("IMAGE SPLIT INFORMATION")
print("=" * 70)

print(f"Training/Validation images : {len(train_images)}")
print(f"Test images                : {len(test_images)}")


# ------------------------------------------------------------
# 6. Check image overlap
# ------------------------------------------------------------

image_overlap = train_images.intersection(test_images)

print(f"\nImages appearing in both   : {len(image_overlap)}")

if len(image_overlap) == 0:
    print("✓ No image-level overlap detected")
else:
    print("⚠ WARNING: Image overlap detected!")

    print("\nExample overlapping images:")

    for image in list(image_overlap)[:10]:
        print(" -", image)


# ------------------------------------------------------------
# 7. Convert images to patient IDs
# ------------------------------------------------------------

train_patients = {
    image_to_patient[image]
    for image in train_images
    if image in image_to_patient
}

test_patients = {
    image_to_patient[image]
    for image in test_images
    if image in image_to_patient
}


# ------------------------------------------------------------
# 8. Patient statistics
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PATIENT-LEVEL INFORMATION")
print("=" * 70)

print(f"Training/Validation patients : {len(train_patients)}")
print(f"Test patients                : {len(test_patients)}")


# ------------------------------------------------------------
# 9. Check patient overlap
# ------------------------------------------------------------

patient_overlap = train_patients.intersection(test_patients)

print(f"\nPatients appearing in both   : {len(patient_overlap)}")


if len(patient_overlap) == 0:

    print("✓ No patient-level overlap detected")

    print(
        "\nThis means the official train/validation and test "
        "lists are patient-independent."
    )

else:

    print("⚠ WARNING: Patient-level overlap detected!")

    print(
        "\nThe following patients appear in both "
        "training/validation and test sets:"
    )

    for patient in list(patient_overlap)[:20]:
        print(" -", patient)


# ------------------------------------------------------------
# 10. Check missing images
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("IMAGE VALIDATION")
print("=" * 70)

csv_images = set(df["Image Index"])

missing_train = train_images - csv_images
missing_test = test_images - csv_images

print(f"Train images missing from CSV : {len(missing_train)}")
print(f"Test images missing from CSV  : {len(missing_test)}")


if len(missing_train) == 0 and len(missing_test) == 0:
    print("✓ All train/test images exist in the CSV")
else:
    print("⚠ Some images from the split files are missing in the CSV")


# ------------------------------------------------------------
# 11. Label distribution
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DISEASE LABEL DISTRIBUTION")
print("=" * 70)


def calculate_label_distribution(image_set):

    label_counts = {}

    for image in image_set:

        if image not in image_to_labels:
            continue

        labels = image_to_labels[image].split("|")

        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1

    return label_counts


train_labels = calculate_label_distribution(train_images)
test_labels = calculate_label_distribution(test_images)


# ------------------------------------------------------------
# 12. Print disease distribution
# ------------------------------------------------------------

all_labels = sorted(
    set(train_labels.keys()).union(test_labels.keys())
)

print(
    f"\n{'Disease':<25}"
    f"{'Train':>12}"
    f"{'Test':>12}"
)

print("-" * 50)

for label in all_labels:

    train_count = train_labels.get(label, 0)
    test_count = test_labels.get(label, 0)

    print(
        f"{label:<25}"
        f"{train_count:>12}"
        f"{test_count:>12}"
    )


# ------------------------------------------------------------
# 13. Final conclusion
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)

if len(patient_overlap) == 0:

    print("✓ PATIENT-LEVEL SPLIT CHECK PASSED")

    print(
        "\nThe train/validation and test sets do not share patients."
    )

    print(
        "This reduces the risk of patient-level data leakage."
    )

else:

    print("⚠ PATIENT-LEVEL SPLIT CHECK FAILED")

    print(
        "\nPatients are present in both training and test sets."
    )

    print(
        "We should investigate this before model training."
    )


print("\nPatient split check completed.")
print("=" * 70)