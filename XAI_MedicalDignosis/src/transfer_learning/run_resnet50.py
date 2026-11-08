
import json
import shutil
from pathlib import Path

from src.transfer_learning.dataset import load_datasets
from src.transfer_learning.models import (
    build_resnet50,
    load_resnet50_checkpoint
)
from src.transfer_learning.train import train_resnet50
from src.transfer_learning.evaluate import evaluate_model


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

STAGE1_PATH = MODEL_DIR / "resnet50_stage1_best.keras"
STAGE2_PATH = MODEL_DIR / "resnet50_stage2_best.keras"
FINAL_MODEL_PATH = MODEL_DIR / "resnet50_best.keras"

RESULT_PATH = RESULTS_DIR / "resnet50_results.json"


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("RESNET50 TRANSFER LEARNING")
    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1: LOAD DATASET
    # --------------------------------------------------------

    print("\n[1/6] Loading four-class dataset...")

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
    # STEP 2: BUILD RESNET50
    # --------------------------------------------------------

    print("\n[2/6] Building ImageNet-pretrained ResNet50...")

    model, backbone = build_resnet50()

    model.summary()

    print("\nResNet50 model created successfully.")

    # --------------------------------------------------------
    # STEP 3: TRAIN RESNET50
    # --------------------------------------------------------

    print("\n[3/6] Training ResNet50...")

    stage1_path, stage2_path = train_resnet50(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds
    )

    print("\nBoth training phases completed.")

    # --------------------------------------------------------
    # STEP 4: VALIDATION CHECKPOINT COMPARISON
    # --------------------------------------------------------

    print("\n[4/6] Comparing Phase 1 and Phase 2...")

    print("\nEvaluating Phase 1 checkpoint...")

    stage1_result = evaluate_model(
        model_path=stage1_path,
        dataset=val_ds,
        dataset_name="ResNet50 Phase 1 Validation"
    )

    print("\nEvaluating Phase 2 checkpoint...")

    stage2_result = evaluate_model(
        model_path=stage2_path,
        dataset=val_ds,
        dataset_name="ResNet50 Phase 2 Validation"
    )

    stage1_f1 = stage1_result["metrics"]["Macro F1"]
    stage2_f1 = stage2_result["metrics"]["Macro F1"]

    print("\nValidation Macro F1 comparison:")

    print(f"Phase 1: {stage1_f1:.4f}")
    print(f"Phase 2: {stage2_f1:.4f}")

    # Select the better checkpoint using validation Macro F1.
    if stage1_f1 >= stage2_f1:

        selected_path = stage1_path
        selected_stage = "Phase 1"

    else:

        selected_path = stage2_path
        selected_stage = "Phase 2"

    print("\nSelected checkpoint:", selected_stage)

    # --------------------------------------------------------
    # STEP 5: SAVE SELECTED RESNET50 MODEL
    # --------------------------------------------------------

    print("\n[5/6] Saving selected ResNet50 model...")

    if Path(selected_path).resolve() != FINAL_MODEL_PATH.resolve():

        shutil.copy2(
            selected_path,
            FINAL_MODEL_PATH
        )

    # Verify the saved model can be loaded.
    final_model = load_resnet50_checkpoint(
        FINAL_MODEL_PATH
    )

    print("\nSelected model saved successfully.")

    print("Model path:", FINAL_MODEL_PATH)

    print("Model output shape:", final_model.output_shape)

    # --------------------------------------------------------
    # STEP 6: FINAL TEST EVALUATION
    # --------------------------------------------------------

    print("\n[6/6] Evaluating selected model on test dataset...")

    test_result = evaluate_model(
        model_path=FINAL_MODEL_PATH,
        dataset=test_ds,
        dataset_name="ResNet50 Final Test"
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    result_data = {

        "model": "ResNet50",

        "selected_stage": selected_stage,

        "selection_metric": "Validation Macro F1",

        "validation_results": {

            "phase_1_macro_f1": float(stage1_f1),

            "phase_2_macro_f1": float(stage2_f1)

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

    print("\n" + "=" * 70)
    print("RESNET50 TRANSFER LEARNING COMPLETED")
    print("=" * 70)

    print("Selected phase:", selected_stage)

    print("Best model:", FINAL_MODEL_PATH)

    print("Results:", RESULT_PATH)

    print("\nFinal test metrics:")

    for name, value in test_result["metrics"].items():

        print(f"{name:30s}: {value:.4f}")

    print("\nResNet50 implementation finished successfully.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
