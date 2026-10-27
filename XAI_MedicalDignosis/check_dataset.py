import os
import pandas as pd


# ============================================================
# 1. DATASET PATHS
# ============================================================

CSV_PATH = "../Datasets/Data_Entry_2017_v2020.csv"
IMAGE_DIR = "../Datasets/images"


# ============================================================
# 2. CHECK WHETHER PATHS EXIST
# ============================================================

print("=" * 60)
print("NIH CHEST X-RAY DATASET CHECK")
print("=" * 60)

print("\nChecking dataset paths...")

if os.path.exists(CSV_PATH):
    print("✓ CSV file found")
else:
    print("✗ CSV file NOT found")
    print("Expected:", CSV_PATH)

if os.path.exists(IMAGE_DIR):
    print("✓ Image folder found")
else:
    print("✗ Image folder NOT found")
    print("Expected:", IMAGE_DIR)


# ============================================================
# 3. LOAD CSV
# ============================================================

print("\nLoading CSV file...")

try:
    df = pd.read_csv(CSV_PATH)
    print("✓ CSV loaded successfully")

except Exception as e:
    print("✗ Error loading CSV")
    print("Error:", e)
    exit()


# ============================================================
# 4. BASIC DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nNumber of rows:", len(df))
print("Number of columns:", len(df.columns))

print("\nColumn names:")
for column in df.columns:
    print(" -", column)


# ============================================================
# 5. FIRST 5 ROWS
# ============================================================

print("\n" + "=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)

print(df.head())


# ============================================================
# 6. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(df.isnull().sum())


# ============================================================
# 7. UNIQUE IMAGES
# ============================================================

print("\n" + "=" * 60)
print("IMAGE INFORMATION")
print("=" * 60)

if "Image Index" in df.columns:

    total_images = len(df)
    unique_images = df["Image Index"].nunique()

    print("Total CSV records:", total_images)
    print("Unique image names:", unique_images)

else:
    print("Column 'Image Index' not found.")


# ============================================================
# 8. UNIQUE PATIENTS
# ============================================================

if "Patient ID" in df.columns:

    unique_patients = df["Patient ID"].nunique()

    print("Unique patients:", unique_patients)

else:
    print("Column 'Patient ID' not found.")


# ============================================================
# 9. DISEASE LABEL ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("DISEASE LABEL DISTRIBUTION")
print("=" * 60)

if "Finding Labels" in df.columns:

    label_counts = df["Finding Labels"].value_counts()

    print(label_counts)

else:
    print("Column 'Finding Labels' not found.")


# ============================================================
# 10. COUNT ACTUAL IMAGE FILES
# ============================================================

print("\n" + "=" * 60)
print("ACTUAL IMAGE FILES")
print("=" * 60)

if os.path.exists(IMAGE_DIR):

    image_files = [
        file for file in os.listdir(IMAGE_DIR)
        if file.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    print("Images found in image folder:", len(image_files))

else:

    print("Image folder does not exist.")


# ============================================================
# 11. CHECK FOR MISSING IMAGES
# ============================================================

if os.path.exists(IMAGE_DIR) and "Image Index" in df.columns:

    print("\nChecking whether CSV images exist in image folder...")

    image_set = set(image_files)

    csv_images = set(df["Image Index"])

    missing_images = csv_images - image_set

    print("Images referenced by CSV:", len(csv_images))
    print("Images actually found:", len(image_set))
    print("Missing images:", len(missing_images))

    if len(missing_images) == 0:
        print("✓ All CSV images were found.")
    else:
        print("⚠ Some images referenced in CSV are missing.")


# ============================================================
# 12. FINISHED
# ============================================================

print("\n" + "=" * 60)
print("DATASET CHECK COMPLETED")
print("=" * 60)