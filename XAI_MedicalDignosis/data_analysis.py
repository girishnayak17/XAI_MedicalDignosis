import os
import pandas as pd
from collections import Counter


# ============================================================
# DATASET PATH
# ============================================================

CSV_PATH = "../Datasets/Data_Entry_2017_v2020.csv"


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("NIH CHEST X-RAY DATASET ANALYSIS")
print("=" * 70)

print("\nLoading dataset...")

try:
    df = pd.read_csv(CSV_PATH)
    print("✓ Dataset loaded successfully")

except Exception as e:
    print("✗ Error loading dataset")
    print("Error:", e)
    exit()


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("1. BASIC DATASET INFORMATION")
print("=" * 70)

print("Total records:", len(df))
print("Total columns:", len(df.columns))

print("\nColumns:")

for column in df.columns:
    print(" -", column)


# ============================================================
# UNIQUE IMAGES
# ============================================================

print("\n" + "=" * 70)
print("2. IMAGE INFORMATION")
print("=" * 70)

if "Image Index" in df.columns:

    total_records = len(df)

    unique_images = df["Image Index"].nunique()

    print("Total records:", total_records)
    print("Unique images:", unique_images)

else:
    print("Image Index column not found.")


# ============================================================
# PATIENT INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("3. PATIENT INFORMATION")
print("=" * 70)

if "Patient ID" in df.columns:

    unique_patients = df["Patient ID"].nunique()

    print("Unique patients:", unique_patients)

    images_per_patient = df.groupby("Patient ID").size()

    print("\nImages per patient statistics:")

    print("Minimum:", images_per_patient.min())
    print("Maximum:", images_per_patient.max())
    print("Average:", round(images_per_patient.mean(), 2))

else:
    print("Patient ID column not found.")


# ============================================================
# DISEASE LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("4. DISEASE LABEL DISTRIBUTION")
print("=" * 70)

if "Finding Labels" in df.columns:

    label_counts = df["Finding Labels"].value_counts()

    print("\nComplete label distribution:\n")

    for label, count in label_counts.items():

        percentage = (count / len(df)) * 100

        print(
            f"{label:<35} "
            f"{count:>8} "
            f"({percentage:.2f}%)"
        )

else:
    print("Finding Labels column not found.")


# ============================================================
# MULTI-LABEL ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("5. MULTI-LABEL ANALYSIS")
print("=" * 70)

if "Finding Labels" in df.columns:

    labels_per_image = df["Finding Labels"].apply(
        lambda x: len(str(x).split("|"))
    )

    single_label = (labels_per_image == 1).sum()
    multi_label = (labels_per_image > 1).sum()

    print("Images with one label:", single_label)
    print("Images with multiple labels:", multi_label)

    print(
        "Maximum labels in one image:",
        labels_per_image.max()
    )


# ============================================================
# INDIVIDUAL DISEASE COUNTS
# ============================================================

print("\n" + "=" * 70)
print("6. INDIVIDUAL DISEASE COUNTS")
print("=" * 70)

if "Finding Labels" in df.columns:

    disease_counter = Counter()

    for labels in df["Finding Labels"]:

        individual_labels = str(labels).split("|")

        for label in individual_labels:

            disease_counter[label] += 1

    print()

    for disease, count in disease_counter.most_common():

        percentage = (count / len(df)) * 100

        print(
            f"{disease:<35} "
            f"{count:>8} "
            f"({percentage:.2f}%)"
        )


# ============================================================
# NO FINDING ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("7. NO FINDING ANALYSIS")
print("=" * 70)

if "Finding Labels" in df.columns:

    no_finding_count = (
        df["Finding Labels"] == "No Finding"
    ).sum()

    percentage = (
        no_finding_count / len(df)
    ) * 100

    print("No Finding images:", no_finding_count)
    print(f"Percentage: {percentage:.2f}%")


# ============================================================
# TRAIN / TEST LIST ANALYSIS
# ============================================================

TRAIN_FILE = "../Datasets/train_val_list.txt"
TEST_FILE = "../Datasets/test_list.txt"


print("\n" + "=" * 70)
print("8. TRAIN / TEST SPLIT INFORMATION")
print("=" * 70)


# -------------------------------
# TRAIN / VALIDATION
# -------------------------------

if os.path.exists(TRAIN_FILE):

    with open(TRAIN_FILE, "r") as file:

        train_images = [
            line.strip()
            for line in file
            if line.strip()
        ]

    print(
        "Train/Validation images:",
        len(train_images)
    )

else:

    print("train_val_list.txt not found.")


# -------------------------------
# TEST
# -------------------------------

if os.path.exists(TEST_FILE):

    with open(TEST_FILE, "r") as file:

        test_images = [
            line.strip()
            for line in file
            if line.strip()
        ]

    print(
        "Test images:",
        len(test_images)
    )

else:

    print("test_list.txt not found.")


# ============================================================
# TRAIN / TEST OVERLAP CHECK
# ============================================================

print("\n" + "=" * 70)
print("9. TRAIN / TEST OVERLAP CHECK")
print("=" * 70)

if os.path.exists(TRAIN_FILE) and os.path.exists(TEST_FILE):

    train_set = set(train_images)
    test_set = set(test_images)

    overlap = train_set.intersection(test_set)

    print("Images appearing in both:", len(overlap))

    if len(overlap) == 0:

        print("✓ No image-level overlap detected.")

    else:

        print("⚠ Image-level overlap detected.")


# ============================================================
# DUPLICATE IMAGE NAME CHECK
# ============================================================

print("\n" + "=" * 70)
print("10. DUPLICATE IMAGE NAME CHECK")
print("=" * 70)

if "Image Index" in df.columns:

    duplicate_count = (
        df["Image Index"].duplicated().sum()
    )

    print(
        "Duplicate image-name records:",
        duplicate_count
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DATASET ANALYSIS COMPLETED")
print("=" * 70)

print("\nNext step:")
print("Use these results to design the classification problem")
print("and preprocessing pipeline before model training.")