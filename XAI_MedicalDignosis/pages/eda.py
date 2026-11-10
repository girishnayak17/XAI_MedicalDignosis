"""Dataset EDA for the XAI Chest X-ray project.

Expected repository layout:

MCA_Final_Project/
├── KDatasets_Clean/
│   ├── train/{normal,pneumonia,tuberculosis}/
│   ├── val/{normal,pneumonia,tuberculosis}/
│   └── test/{normal,pneumonia,tuberculosis}/
├── covid19Dataset/
│   └── COVID-19_Radiography_Dataset/
│       ├── COVID/images/
│       ├── Normal/images/
│       ├── Viral Pneumonia/images/
│       └── Lung_Opacity/images/  # excluded from the four target classes
└── XAI_MedicalDignosis/pages/eda.py

This page reads images in place. It does not copy or modify datasets.
"""

from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.ui.core import BASE_DIR, CLASS_NAMES


# BASE_DIR is XAI_MedicalDignosis; datasets are siblings of that folder.
REPOSITORY_ROOT = BASE_DIR.parent
KDATASET_ROOT = REPOSITORY_ROOT / "KDatasets_Clean"
COVID_DATASET_ROOT = (
    REPOSITORY_ROOT
    / "covid19Dataset"
    / "COVID-19_Radiography_Dataset"
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SPLIT_ALIASES = {
    "train": "Train",
    "training": "Train",
    "val": "Validation",
    "valid": "Validation",
    "validation": "Validation",
    "test": "Test",
    "testing": "Test",
}

# The source label is retained in each record for transparency.
KDATASET_CLASS_MAP = {
    "normal": ("Normal", "Normal"),
    "pneumonia": ("Pneumonia", "Pneumonia"),
    "tuberculosis": ("Tuberculosis", "Tuberculosis"),
    "tb": ("Tuberculosis", "Tuberculosis"),
}

COVID_CLASS_MAP = {
    "normal": ("Normal", "Normal"),
    "covid": ("COVID-19", "COVID"),
    "covid-19": ("COVID-19", "COVID"),
    # This source class is grouped under the application's broader
    # Pneumonia label. The original label remains visible in the table.
    "viral pneumonia": ("Pneumonia", "Viral Pneumonia"),
}


st.title("📊 Exploratory Data Analysis")
st.write(
    "Explore the image datasets used by the project. The datasets are read "
    "from their existing locations in the MCA_Final_Project directory; "
    "nothing is copied into the application's data folder."
)


@st.cache_data(show_spinner="Scanning dataset images and reading image dimensions...")
def scan_datasets(kdataset_root_text: str, covid_root_text: str):
    """Scan supported class images and return metadata records."""
    records = []
    excluded_records = []
    errors = []

    k_root = Path(kdataset_root_text)
    if k_root.is_dir():
        for split_dir in sorted(p for p in k_root.iterdir() if p.is_dir()):
            split_name = SPLIT_ALIASES.get(split_dir.name.lower())
            if split_name is None:
                continue

            for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                mapping = KDATASET_CLASS_MAP.get(class_dir.name.lower())
                if mapping is None:
                    continue
                class_name, source_class = mapping

                for image_path in class_dir.rglob("*"):
                    if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    width, height = None, None
                    try:
                        with Image.open(image_path) as image:
                            width, height = image.size
                            image.verify()
                    except (OSError, UnidentifiedImageError, ValueError) as exc:
                        errors.append((str(image_path), str(exc)))
                        continue

                    records.append({
                        "Class": class_name,
                        "Source class": source_class,
                        "Dataset": "KDatasets_Clean",
                        "Split": split_name,
                        "Path": str(image_path),
                        "Filename": image_path.name,
                        "Width": width,
                        "Height": height,
                    })
    else:
        errors.append((str(k_root), "KDatasets_Clean directory was not found"))

    covid_root = Path(covid_root_text)
    if covid_root.is_dir():
        for class_dir in sorted(p for p in covid_root.iterdir() if p.is_dir()):
            source_name = class_dir.name.strip()
            source_key = source_name.lower()
            mapping = COVID_CLASS_MAP.get(source_key)

            # COVID-19 Radiography images are in each class's images/ folder.
            # Masks are deliberately never scanned.
            images_dir = class_dir / "images"
            if not images_dir.is_dir():
                continue

            if mapping is None:
                if source_key == "lung_opacity":
                    for image_path in images_dir.rglob("*"):
                        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                            excluded_records.append({
                                "Source class": source_name,
                                "Path": str(image_path),
                            })
                continue

            class_name, source_class = mapping
            for image_path in images_dir.rglob("*"):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                width, height = None, None
                try:
                    with Image.open(image_path) as image:
                        width, height = image.size
                        image.verify()
                except (OSError, UnidentifiedImageError, ValueError) as exc:
                    errors.append((str(image_path), str(exc)))
                    continue

                records.append({
                    "Class": class_name,
                    "Source class": source_class,
                    "Dataset": "COVID-19 Radiography Dataset",
                    # The source folder does not define train/validation/test.
                    "Split": "Not specified",
                    "Path": str(image_path),
                    "Filename": image_path.name,
                    "Width": width,
                    "Height": height,
                })
    else:
        errors.append((str(covid_root), "COVID-19_Radiography_Dataset directory was not found"))

    return records, excluded_records, errors


records, excluded_records, scan_errors = scan_datasets(
    str(KDATASET_ROOT),
    str(COVID_DATASET_ROOT),
)

if scan_errors:
    with st.expander(f"Dataset scan notices ({len(scan_errors)})"):
        for path_text, message in scan_errors[:30]:
            st.warning(f"{path_text}: {message}")
        if len(scan_errors) > 30:
            st.caption(f"Showing first 30 of {len(scan_errors)} notices.")

if not records:
    st.error("No supported class images were found. Check the two dataset paths and folder names.")
    st.code(
        f"KDatasets_Clean: {KDATASET_ROOT}\n"
        f"COVID dataset: {COVID_DATASET_ROOT}",
        language="text",
    )
    st.stop()


df = pd.DataFrame(records)

# -----------------------------------------------------------------------------
# Dataset overview
# -----------------------------------------------------------------------------

st.subheader("Dataset overview")
source_counts = df.groupby("Dataset").size().rename("Image files").reset_index()
class_counts = df.groupby("Class").size().reindex(CLASS_NAMES, fill_value=0)

metric_cols = st.columns(4)
for column, class_name in zip(metric_cols, CLASS_NAMES):
    column.metric(class_name, f"{int(class_counts[class_name]):,}")

st.caption(
    f"Scanned {len(df):,} readable image files across both sources. "
    "Counts represent files found in these folders, not necessarily unique patients or unique images."
)

# -----------------------------------------------------------------------------
# Class distribution
# -----------------------------------------------------------------------------

st.subheader("1. Class distribution")
class_source_counts = (
    df.groupby(["Class", "Dataset"]).size().unstack(fill_value=0)
    .reindex(CLASS_NAMES, fill_value=0)
)
st.bar_chart(class_source_counts)
st.dataframe(
    class_source_counts.reset_index().rename(columns={"index": "Class"}),
    use_container_width=True,
    hide_index=True,
)
st.caption(
    "The chart separates counts by source dataset. Viral Pneumonia images from "
    "the COVID-19 Radiography Dataset are grouped under Pneumonia; their source "
    "label is retained in the source breakdown. Lung Opacity is excluded from "
    "the four target classes."
)

with st.expander("Source-label breakdown"):
    source_label_counts = (
        df.groupby(["Dataset", "Source class", "Class"])
        .size().rename("Image count").reset_index()
    )
    st.dataframe(source_label_counts, use_container_width=True, hide_index=True)
    st.dataframe(source_counts, use_container_width=True, hide_index=True)
    if excluded_records:
        st.info(
            f"{len(excluded_records):,} Lung Opacity image files were found but "
            "excluded from the four-class totals because Lung Opacity is not one "
            "of the application's target labels."
        )

# -----------------------------------------------------------------------------
# Sample X-ray images
# -----------------------------------------------------------------------------

st.subheader("2. Sample X-ray images")
for class_name in CLASS_NAMES:
    class_df = df[df["Class"] == class_name]
    if class_df.empty:
        continue
    st.markdown(f"**{class_name}**")
    sample_count = min(4, len(class_df))
    samples = class_df.sample(n=sample_count, random_state=42)
    columns = st.columns(sample_count)
    for column, (_, row) in zip(columns, samples.iterrows()):
        try:
            with Image.open(row["Path"]) as image:
                column.image(
                    image.convert("RGB"),
                    caption=f"{row['Dataset']} · {row['Source class']}",
                    use_container_width=True,
                )
        except (OSError, UnidentifiedImageError):
            column.warning(f"Could not display {row['Filename']}")

# -----------------------------------------------------------------------------
# Image dimensions
# -----------------------------------------------------------------------------

st.subheader("3. Image dimensions")
dim_df = df.dropna(subset=["Width", "Height"]).copy()
dim_df["Dimensions"] = (
    dim_df["Width"].astype(int).astype(str)
    + " × "
    + dim_df["Height"].astype(int).astype(str)
)
dimension_counts = (
    dim_df.groupby(["Dimensions", "Dataset"]).size()
    .unstack(fill_value=0)
)
# Keep the most frequent 15 dimension combinations to make the chart readable.
top_dimensions = dim_df["Dimensions"].value_counts().head(15).index
dimension_counts = dimension_counts.reindex(top_dimensions, fill_value=0)
st.bar_chart(dimension_counts)
st.dataframe(
    dim_df.groupby(["Dataset", "Dimensions"]).size()
    .rename("Image count").reset_index()
    .sort_values("Image count", ascending=False),
    use_container_width=True,
    hide_index=True,
)

# -----------------------------------------------------------------------------
# Dataset split distribution
# -----------------------------------------------------------------------------

st.subheader("4. Dataset split distribution")
k_split_df = df[df["Dataset"] == "KDatasets_Clean"]
if not k_split_df.empty:
    split_table = (
        k_split_df.groupby(["Split", "Class"]).size()
        .unstack(fill_value=0)
        .reindex(index=["Train", "Validation", "Test"], fill_value=0)
        .reindex(columns=CLASS_NAMES, fill_value=0)
    )
    st.bar_chart(split_table)
    st.dataframe(
        split_table.reset_index(),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No train/validation/test split folders were found in KDatasets_Clean.")

covid_count = int((df["Dataset"] == "COVID-19 Radiography Dataset").sum())
if covid_count:
    st.info(
        f"The COVID-19 Radiography Dataset contributes {covid_count:,} readable "
        "images, but its current folder structure does not specify train, "
        "validation or test splits. It is therefore not assigned to any split "
        "in the chart above."
    )

# -----------------------------------------------------------------------------
# Methodology notes
# -----------------------------------------------------------------------------

with st.expander("Dataset interpretation and limitations"):
    st.markdown(
        """
        - Images are counted from the folders on disk; this page does not infer patient IDs.
        - `KDatasets_Clean` split labels are read from its `train`, `val`, and `test` folders.
        - The COVID-19 Radiography Dataset is scanned only inside each class's `images/` folder; `masks/` are not counted as X-rays.
        - `COVID` is mapped to **COVID-19**, `Normal` to **Normal**, and `Viral Pneumonia` to **Pneumonia**.
        - `Lung_Opacity` is excluded from the four-class totals. It should only be included if the project explicitly defines a compatible target class and retrains/evaluates accordingly.
        - The two sources are reported separately. Do not interpret their sum as a unique-image or unique-patient count without duplicate and patient-level checks.
        """
    )

st.caption(
    "EDA is descriptive dataset exploration. It does not establish clinical validity "
    "or prove that the model's evaluation split is free from patient-level leakage."
)
