from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR.parent / "Datasets"

CSV_PATH = DATASET_DIR / "Data_Entry_2017_v2020.csv"
IMAGE_DIR = DATASET_DIR / "images"
TRAIN_LIST = DATASET_DIR / "train_val_list.txt"
TEST_LIST = DATASET_DIR / "test_list.txt"


# ============================================================
# IMAGE CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


# ============================================================
# NIH CHEST X-RAY LABELS
# ============================================================

DISEASE_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
    "No Finding"
]


# ============================================================
# LOAD DATASET METADATA
# ============================================================

def load_metadata():
    """
    Load NIH Chest X-ray metadata CSV.

    Returns
    -------
    pandas.DataFrame
        Dataset metadata.
    """

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Metadata CSV not found:\n{CSV_PATH}"
        )

    df = pd.read_csv(CSV_PATH)

    print(f"Metadata loaded successfully.")
    print(f"Total records: {len(df)}")

    return df


# ============================================================
# LOAD TRAIN / TEST IMAGE LIST
# ============================================================

def load_image_list(file_path):
    """
    Read image names from train/test list file.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"List file not found:\n{file_path}"
        )

    with open(file_path, "r") as file:
        image_names = [
            line.strip()
            for line in file
            if line.strip()
        ]

    return image_names


# ============================================================
# CREATE DATAFRAME FOR A SPLIT
# ============================================================

def create_split_dataframe(image_list, metadata):
    """
    Select metadata records corresponding to the supplied
    image list.
    """

    split_df = metadata[
        metadata["Image Index"].isin(image_list)
    ].copy()

    # Preserve the order from the image list
    order = {
        image_name: index
        for index, image_name in enumerate(image_list)
    }

    split_df["_order"] = split_df["Image Index"].map(order)

    split_df = (
        split_df
        .sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    return split_df


# ============================================================
# ENCODE MULTI-LABEL DISEASES
# ============================================================

def encode_labels(label_string):
    """
    Convert NIH multi-label string into a binary vector.

    Example:

    'Effusion|Infiltration'

    becomes something like:

    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, ...]
    """

    labels = set(label_string.split("|"))

    encoded = np.zeros(
        len(DISEASE_LABELS),
        dtype=np.float32
    )

    for index, disease in enumerate(DISEASE_LABELS):
        if disease in labels:
            encoded[index] = 1.0

    return encoded


# ============================================================
# ADD ENCODED LABELS TO DATAFRAME
# ============================================================

def add_encoded_labels(df):
    """
    Add multi-label binary vectors to dataframe.
    """

    df = df.copy()

    df["Encoded Labels"] = df["Finding Labels"].apply(
        encode_labels
    )

    return df


# ============================================================
# LOAD AND PREPROCESS ONE IMAGE
# ============================================================

def load_and_preprocess_image(image_name):
    """
    Load an X-ray image and preprocess it for the CNN.

    Steps:
    1. Load image
    2. Convert to RGB
    3. Resize to 224 x 224
    4. Convert to float32
    5. Normalize pixels to [0, 1]
    """

    image_path = IMAGE_DIR / image_name

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    image = Image.open(image_path)

    # Convert grayscale X-ray to RGB
    image = image.convert("RGB")

    # Resize
    image = image.resize(IMAGE_SIZE)

    # Convert to NumPy array
    image = np.array(
        image,
        dtype=np.float32
    )

    # Normalize pixel values
    image = image / 255.0

    return image


# ============================================================
# TEST ONE SAMPLE
# ============================================================

def get_sample(index=0):
    """
    Load one sample from the metadata.
    """

    metadata = load_metadata()

    image_name = metadata.iloc[index]["Image Index"]
    labels = metadata.iloc[index]["Finding Labels"]

    image = load_and_preprocess_image(image_name)

    encoded_labels = encode_labels(labels)

    return image, encoded_labels, image_name, labels


# ============================================================
# CREATE TENSORFLOW DATASET
# ============================================================

def create_tf_dataset(
    dataframe,
    shuffle=False,
    batch_size=BATCH_SIZE
):
    """
    Create a TensorFlow Dataset from a dataframe.

    Each sample contains:

        image -> (224, 224, 3)
        labels -> (15,)
    """

    image_names = dataframe["Image Index"].values

    labels = np.stack(
        dataframe["Encoded Labels"].values
    ).astype(np.float32)

    def generator():
        for image_name, label in zip(
            image_names,
            labels
        ):
            image = load_and_preprocess_image(
                image_name
            )

            yield image, label

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(
                shape=(
                    IMAGE_SIZE[0],
                    IMAGE_SIZE[1],
                    3
                ),
                dtype=tf.float32
            ),
            tf.TensorSpec(
                shape=(len(DISEASE_LABELS),),
                dtype=tf.float32
            )
        )
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(
                len(dataframe),
                10000
            )
        )

    dataset = dataset.batch(
        batch_size
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# PREPARE TRAIN AND TEST DATA
# ============================================================

def prepare_data():
    """
    Load metadata and create train/test dataframes.
    """

    metadata = load_metadata()

    train_images = load_image_list(
        TRAIN_LIST
    )

    test_images = load_image_list(
        TEST_LIST
    )

    train_df = create_split_dataframe(
        train_images,
        metadata
    )

    test_df = create_split_dataframe(
        test_images,
        metadata
    )

    train_df = add_encoded_labels(
        train_df
    )

    test_df = add_encoded_labels(
        test_df
    )

    print("\nDataset split:")
    print(f"Train/Validation images: {len(train_df)}")
    print(f"Test images: {len(test_df)}")

    return train_df, test_df


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("NIH CHEST X-RAY PREPROCESSING TEST")
    print("=" * 60)

    # Check paths
    print("\nDataset directory:")
    print(DATASET_DIR)

    print("\nImage directory:")
    print(IMAGE_DIR)

    # Prepare data
    train_df, test_df = prepare_data()

    # Load first sample
    image, encoded_labels, image_name, original_labels = (
        get_sample(0)
    )

    print("\nSample information:")
    print("Image:", image_name)
    print("Original labels:", original_labels)
    print("Image shape:", image.shape)
    print("Image dtype:", image.dtype)
    print(
        "Pixel range:",
        image.min(),
        "to",
        image.max()
    )

    print("\nEncoded labels:")
    print(encoded_labels)

    # Create small test dataset
    sample_df = train_df.head(4)

    dataset = create_tf_dataset(
        sample_df,
        shuffle=False,
        batch_size=2
    )

    print("\nTensorFlow dataset test:")

    for images, labels in dataset.take(1):
        print("Batch image shape:", images.shape)
        print("Batch label shape:", labels.shape)

    print("\n" + "=" * 60)
    print("PREPROCESSING TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)