
import json
import shutil
from pathlib import Path

import tensorflow as tf

from src.transfer_learning.dataset import load_datasets
from src.transfer_learning.efficientnetb0 import (
    build_efficientnetb0,
    compile_efficientnetb0,
    load_efficientnetb0_checkpoint
)
from src.transfer_learning.evaluate import evaluate_model


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_DIR / "models"

RESULTS_DIR = PROJECT_DIR / "results" / "efficientnetb0"

STAGE1_PATH = MODEL_DIR / "efficientnetb0_stage1_best.keras"

STAGE2_PATH = MODEL_DIR / "efficientnetb0_stage2_best.keras"

FINAL_MODEL_PATH = MODEL_DIR / "efficientnetb0_best.keras"

RESULT_PATH = RESULTS_DIR / "efficientnetb0_results.json"

IMAGE_SIZE = 224

HEAD_EPOCHS = 15

FINETUNE_EPOCHS = 15

FINE_TUNE_LAST_LAYERS = 30

HEAD_LEARNING_RATE = 1e-3

FINETUNE_LEARNING_RATE = 1e-5


# ============================================================
# CREATE DIRECTORIES
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PREVENT ACCIDENTAL OVERWRITING
# ============================================================

for output_path in (
    STAGE1_PATH,
    STAGE2_PATH,
    FINAL_MODEL_PATH,
    RESULT_PATH
):

    if output_path.exists():

        raise FileExistsError(
            f"\nOutput already exists:\n{output_path}\n\n"
            "Back up or rename the existing file before "
            "running another training experiment."
        )


# ============================================================
# CALLBACKS
# ============================================================

def create_callbacks(checkpoint_path):

    return [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )

    ]


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    print("\n" + "=" * 75)

    print("EFFICIENTNETB0 TRANSFER LEARNING")

    print("=" * 75)

    print("TensorFlow version:", tf.__version__)

    print("Detected GPUs:", tf.config.list_physical_devices("GPU"))

    print("Project directory:", PROJECT_DIR)


    # --------------------------------------------------------
    # STEP 1: LOAD DATASET
    # --------------------------------------------------------

    print("\n[1/6] Loading existing four-class dataset...")

    (
        train_ds,
        val_ds,
        test_ds,
        train_records,
        val_records,
        test_records
    ) = load_datasets()

    print("\nDataset loaded successfully.")

    print("Training images:", len(train_records))

    print("Validation images:", len(val_records))

    print("Testing images:", len(test_records))


    # --------------------------------------------------------
    # STEP 2: BUILD EFFICIENTNETB0
    # --------------------------------------------------------

    print(
        "\n[2/6] Building ImageNet-pretrained EfficientNetB0..."
    )

    model, base_model = build_efficientnetb0()

    model.summary()

    print("\nEfficientNetB0 architecture created successfully.")


    # --------------------------------------------------------
    # STEP 3: TRAIN CLASSIFICATION HEAD
    # --------------------------------------------------------

    print("\n[3/6] Stage 1: Training classification head...")

    print("Freezing EfficientNetB0 backbone.")

    base_model.trainable = False

    compile_efficientnetb0(
        model,
        learning_rate=HEAD_LEARNING_RATE
    )

    history_stage1 = model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=HEAD_EPOCHS,

        callbacks=create_callbacks(STAGE1_PATH),

        verbose=1

    )

    print("\nStage 1 completed.")

    print("Stage 1 checkpoint:", STAGE1_PATH)


    # --------------------------------------------------------
    # STEP 4: FINE-TUNE EFFICIENTNETB0
    # --------------------------------------------------------

    print("\n[4/6] Stage 2: Fine-tuning EfficientNetB0...")

    print(
        f"Unfreezing the last {FINE_TUNE_LAST_LAYERS} "
        "backbone layers."
    )

    base_model.trainable = True

    # Freeze earlier layers.
    for layer in base_model.layers[:-FINE_TUNE_LAST_LAYERS]:

        layer.trainable = False

    # Keep BatchNormalization layers frozen.
    for layer in base_model.layers:

        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):

            layer.trainable = False

    trainable_layers = sum(
        1 for layer in base_model.layers
        if layer.trainable
    )

    print("Trainable backbone layers:", trainable_layers)

    # Recompile after modifying trainable layers.
    compile_efficientnetb0(
        model,
        learning_rate=FINETUNE_LEARNING_RATE
    )

    history_stage2 = model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=FINETUNE_EPOCHS,

        callbacks=create_callbacks(STAGE2_PATH),

        verbose=1

    )

    print("\nStage 2 completed.")

    print("Stage 2 checkpoint:", STAGE2_PATH)


    # --------------------------------------------------------
    # STEP 5: VALIDATION CHECKPOINT COMPARISON
    # --------------------------------------------------------

    print(
        "\n[5/6] Comparing Stage 1 and Stage 2 "
        "using validation Macro F1..."
    )

    print("\nEvaluating Stage 1 checkpoint...")

    stage1_result = evaluate_model(

        model_path=str(STAGE1_PATH),

        dataset=val_ds,

        dataset_name="EfficientNetB0 Stage 1 Validation"

    )

    print("\nEvaluating Stage 2 checkpoint...")

    stage2_result = evaluate_model(

        model_path=str(STAGE2_PATH),

        dataset=val_ds,

        dataset_name="EfficientNetB0 Stage 2 Validation"

    )

    stage1_f1 = stage1_result["metrics"]["Macro F1"]

    stage2_f1 = stage2_result["metrics"]["Macro F1"]

    print("\nValidation Macro F1 comparison:")

    print(f"Stage 1: {stage1_f1:.4f}")

    print(f"Stage 2: {stage2_f1:.4f}")


    # Select the better checkpoint.
    if stage1_f1 >= stage2_f1:

        selected_path = STAGE1_PATH

        selected_stage = "Stage 1"

    else:

        selected_path = STAGE2_PATH

        selected_stage = "Stage 2"

    print("\nSelected checkpoint:", selected_stage)


    # --------------------------------------------------------
    # SAVE SELECTED MODEL
    # --------------------------------------------------------

    print("\nSaving selected EfficientNetB0 model...")

    shutil.copy2(
        selected_path,
        FINAL_MODEL_PATH
    )

    # Verify the saved checkpoint.
    final_model = load_efficientnetb0_checkpoint(
        str(FINAL_MODEL_PATH)
    )

    print("\nSelected model saved successfully.")

    print("Model path:", FINAL_MODEL_PATH)

    print("Model input shape:", final_model.input_shape)

    print("Model output shape:", final_model.output_shape)


    # --------------------------------------------------------
    # STEP 6: FINAL TEST EVALUATION
    # --------------------------------------------------------

    print("\n[6/6] Final test evaluation...")

    test_result = evaluate_model(

        model_path=str(FINAL_MODEL_PATH),

        dataset=test_ds,

        dataset_name="EfficientNetB0 Final Test"

    )


    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    result_data = {

        "model": "EfficientNetB0",

        "architecture": "ImageNet pretrained EfficientNetB0",

        "selected_stage": selected_stage,

        "selection_metric": "Validation Macro F1",

        "training_configuration": {

            "image_size": IMAGE_SIZE,

            "head_epochs": HEAD_EPOCHS,

            "fine_tune_epochs": FINETUNE_EPOCHS,

            "fine_tune_last_layers": FINE_TUNE_LAST_LAYERS,

            "head_learning_rate": HEAD_LEARNING_RATE,

            "fine_tune_learning_rate": FINETUNE_LEARNING_RATE

        },

        "dataset_sizes": {

            "training": len(train_records),

            "validation": len(val_records),

            "testing": len(test_records)

        },

        "validation_results": {

            "stage_1_macro_f1": float(stage1_f1),

            "stage_2_macro_f1": float(stage2_f1)

        },

        "test_metrics": {

            key: float(value)

            for key, value in test_result["metrics"].items()

        },

        "confusion_matrix": (

            test_result["confusion_matrix"].tolist()

        ),

        "classification_report": (

            test_result["classification_report"]

        ),

        "model_path": str(FINAL_MODEL_PATH)

    }

    with open(
        RESULT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result_data,
            file,
            indent=4,
            allow_nan=False
        )


    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 75)

    print("EFFICIENTNETB0 TRAINING COMPLETED SUCCESSFULLY")

    print("=" * 75)

    print("Selected stage:", selected_stage)

    print("Best model:", FINAL_MODEL_PATH)

    print("Results:", RESULT_PATH)

    print("\nFinal test metrics:")

    for name, value in test_result["metrics"].items():

        print(f"{name:30s}: {value:.4f}")

    print("\nEfficientNetB0 implementation finished.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
