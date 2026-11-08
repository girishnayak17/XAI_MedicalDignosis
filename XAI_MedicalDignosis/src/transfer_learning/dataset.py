import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[2]

OLD_DATA_DIR = Path(
    "/mnt/d/MCA_Final_Project/KDatasets_Clean"
)

COVID_DATA_DIR = Path(
    "/mnt/d/MCA_Final_Project/covid19Dataset/"
    "COVID-19_Radiography_Dataset"
)

IMAGE_SIZE = 224
BATCH_SIZE = 8
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

CLASS_NAMES = [
    "normal",
    "pneumonia",
    "tuberculosis",
    "covid"
]

CLASS_TO_LABEL = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


# --------------------------------------------------
# IMAGE COLLECTION
# --------------------------------------------------

def collect_images(folder, label, source, split=None):

    extensions = {".jpg", ".jpeg", ".png"}

    records = []

    if not folder.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {folder}"
        )

    for path in folder.rglob("*"):

        if path.is_file() and path.suffix.lower() in extensions:

            records.append({
                "path": str(path),
                "label": label,
                "class_name": CLASS_NAMES[label],
                "source": source,
                "split": split
            })

    return records


def collect_existing_dataset():

    records = []

    # Existing three-class dataset.
    # Preserve its original train/val/test partitions.

    for split in ["train", "val", "test"]:

        for class_name in [
            "normal",
            "pneumonia",
            "tuberculosis"
        ]:

            folder = OLD_DATA_DIR / split / class_name

            records.extend(
                collect_images(
                    folder=folder,
                    label=CLASS_TO_LABEL[class_name],
                    source="old",
                    split=split
                )
            )

    return records


def collect_covid_dataset():

    covid_folder = COVID_DATA_DIR / "COVID"

    records = collect_images(
        folder=covid_folder,
        label=CLASS_TO_LABEL["covid"],
        source="covid_dataset"
    )

    if len(records) < 10:
        raise ValueError(
            "Not enough COVID images found. "
            f"Found: {len(records)}"
        )

    return records


# --------------------------------------------------
# COVID TRAIN / VALIDATION / TEST SPLIT
# --------------------------------------------------

def split_covid_records(records):

    # One class only: random splitting is used.
    # The split is reproducible through SEED.

    train_records, temporary_records = train_test_split(
        records,
        test_size=0.30,
        random_state=SEED,
        shuffle=True
    )

    val_records, test_records = train_test_split(
        temporary_records,
        test_size=0.50,
        random_state=SEED,
        shuffle=True
    )

    return train_records, val_records, test_records


# --------------------------------------------------
# COMBINE DATASETS
# --------------------------------------------------

def create_records():

    old_records = collect_existing_dataset()
    covid_records = collect_covid_dataset()

    old_train = [
        r for r in old_records if r["split"] == "train"
    ]

    old_val = [
        r for r in old_records if r["split"] == "val"
    ]

    old_test = [
        r for r in old_records if r["split"] == "test"
    ]

    covid_train, covid_val, covid_test = split_covid_records(
        covid_records
    )

    train_records = old_train + covid_train
    val_records = old_val + covid_val
    test_records = old_test + covid_test

    if not train_records or not val_records or not test_records:
        raise ValueError("One or more dataset splits are empty.")

    return train_records, val_records, test_records


# --------------------------------------------------
# PRINT DATASET DISTRIBUTION
# --------------------------------------------------

def print_distribution(name, records):

    print(f"\n{name} distribution")

    for class_name in CLASS_NAMES:

        count = sum(
            r["class_name"] == class_name
            for r in records
        )

        print(f"{class_name:15s}: {count}")

    print(f"{'Total':15s}: {len(records)}")


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

def decode_image(path, label):

    image_bytes = tf.io.read_file(path)

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE]
    )

    # Keep pixel values in 0-255 range.
    # ResNet50 preprocessing is performed in models.py.

    image = tf.cast(image, tf.float32)

    return image, label


# --------------------------------------------------
# TENSORFLOW DATASET CREATION
# --------------------------------------------------

def make_dataset(records, training=False, class_weights=None):

    paths = np.asarray(
        [r["path"] for r in records],
        dtype=str
    )

    labels = np.asarray(
        [r["label"] for r in records],
        dtype=np.int32
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:

        dataset = dataset.shuffle(
            buffer_size=len(records),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    if class_weights is not None:

        weights_tensor = tf.constant(
            class_weights,
            dtype=tf.float32
        )

        def decode_with_weight(path, label):

            image, label = decode_image(path, label)

            sample_weight = tf.gather(
                weights_tensor,
                label
            )

            return image, label, sample_weight

        dataset = dataset.map(
            decode_with_weight,
            num_parallel_calls=AUTOTUNE
        )

    else:

        dataset = dataset.map(
            decode_image,
            num_parallel_calls=AUTOTUNE
        )

    return dataset.batch(BATCH_SIZE).prefetch(1)


# --------------------------------------------------
# MAIN DATASET LOADER
# --------------------------------------------------

def load_datasets():

    train_records, val_records, test_records = create_records()

    print_distribution("TRAIN", train_records)
    print_distribution("VALIDATION", val_records)
    print_distribution("TEST", test_records)

    y_train = np.asarray(
        [r["label"] for r in train_records],
        dtype=np.int32
    )

    present_classes = np.unique(y_train)

    if len(present_classes) != len(CLASS_NAMES):
        raise ValueError(
            "All four classes must be present in training data."
        )

    computed_weights = compute_class_weight(
        class_weight="balanced",
        classes=present_classes,
        y=y_train
    )

    class_weights = np.ones(
        len(CLASS_NAMES),
        dtype=np.float32
    )

    class_weights[present_classes] = computed_weights

    print("\nClass weights:", class_weights)

    train_ds = make_dataset(
        train_records,
        training=True,
        class_weights=class_weights
    )

    val_ds = make_dataset(val_records)
    test_ds = make_dataset(test_records)

    return (
        train_ds,
        val_ds,
        test_ds,
        train_records,
        val_records,
        test_records
    )