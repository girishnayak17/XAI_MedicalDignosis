from pathlib import Path
from collections import Counter, defaultdict
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from src.ui.core import BASE_DIR, CLASS_NAMES

st.title("📊 Exploratory Data Analysis")
st.write("Dataset-level exploration. This page reads the dataset available under the project's `data/` directory; it does not use uploaded prediction images as training data.")
DATA_DIR = BASE_DIR / "data"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ALIASES = {
    "Normal": {"normal", "normal_images", "healthy"},
    "Pneumonia": {"pneumonia", "pneumonia_images"},
    "Tuberculosis": {"tuberculosis", "tb", "tuberculosis_images"},
    "COVID-19": {"covid", "covid19", "covid-19", "covid_images"},
}

def class_for_folder(name):
    value = name.lower().replace(" ", "_")
    for label, aliases in ALIASES.items():
        if value in aliases:
            return label
    return None

def scan_dataset(root):
    class_files = defaultdict(list)
    split_counts = Counter()
    dimension_counts = Counter()
    if not root.exists():
        return class_files, split_counts, dimension_counts
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        relative_parts = path.relative_to(root).parts
        label = next((class_for_folder(part) for part in relative_parts[:-1] if class_for_folder(part)), None)
        if label:
            class_files[label].append(path)
            split = next((part.lower() for part in relative_parts[:-1] if part.lower() in {"train", "training", "val", "valid", "validation", "test", "testing"}), "unspecified")
            if split in {"training"}: split = "train"
            if split in {"val", "valid"}: split = "validation"
            if split == "testing": split = "test"
            split_counts[(split, label)] += 1
            try:
                with Image.open(path) as im:
                    dimension_counts[im.size] += 1
            except Exception:
                pass
    return class_files, split_counts, dimension_counts

class_files, split_counts, dimension_counts = scan_dataset(DATA_DIR)
counts = {label: len(class_files.get(label, [])) for label in CLASS_NAMES}

if sum(counts.values()) == 0:
    st.warning("No class-folder image dataset was detected under `data/`. Place images in class folders for this page to build the requested charts.")
    st.code("data/\n  train/\n    normal/\n    pneumonia/\n    tuberculosis/\n    covid/\n  validation/\n    normal/ ...\n  test/\n    normal/ ...", language="text")
    st.caption("Folder names can also use: tb, covid19, covid-19, healthy. Images can be PNG, JPG, JPEG, BMP, or TIFF.")
else:
    st.subheader("Class distribution")
    df = pd.DataFrame({"Class": list(counts.keys()), "Image count": list(counts.values())})
    st.bar_chart(df.set_index("Class"))
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Sample X-ray images")
    for label in CLASS_NAMES:
        files = class_files.get(label, [])
        if files:
            st.markdown(f"**{label}**")
            sample_paths = files[:min(4, len(files))]
            cols = st.columns(len(sample_paths))
            for col, path in zip(cols, sample_paths):
                try:
                    with Image.open(path) as im:
                        col.image(im.convert("RGB"), caption=path.name, use_container_width=True)
                except Exception:
                    col.warning(f"Could not read {path.name}")

    st.subheader("Image dimensions")
    if dimension_counts:
        dim_df = pd.DataFrame([{"Width × Height": f"{w} × {h}", "Image count": n} for (w, h), n in dimension_counts.most_common(15)])
        st.bar_chart(dim_df.set_index("Width × Height"))
        st.dataframe(dim_df, use_container_width=True, hide_index=True)

    st.subheader("Dataset split distribution")
    if split_counts:
        split_df = pd.DataFrame([{"Split": split.title(), "Class": label, "Image count": n} for (split, label), n in sorted(split_counts.items())])
        st.bar_chart(split_df.pivot(index="Split", columns="Class", values="Image count").fillna(0))
        st.dataframe(split_df, use_container_width=True, hide_index=True)
    else:
        st.info("Split names were not found in the directory structure. Add train, validation, and test folders to show split distribution.")

st.caption("EDA counts are inferred from the image files physically present in the data directory. Verify dataset organization and patient-level separation before using these counts in a research report.")
