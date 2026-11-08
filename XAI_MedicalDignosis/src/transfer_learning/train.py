
import tensorflow as tf

from pathlib import Path

from src.transfer_learning.models import (
    compile_resnet50,
    load_resnet50_checkpoint
)


# ============================================================
# CONFIGURATION
# ============================================================

HEAD_EPOCHS = 8
FINE_TUNE_EPOCHS = 12

HEAD_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5

FINE_TUNE_LAYERS = 30

PROJECT_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CALLBACKS
# ============================================================

def get_callbacks(checkpoint_path):

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
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),

        tf.keras.callbacks.TerminateOnNaN()

    ]


# ============================================================
# FINE-TUNING
# ============================================================

def unfreeze_resnet_layers(model):

    backbone = model.get_layer("resnet50")

    backbone.trainable = True

    # Freeze every backbone layer first
    for layer in backbone.layers:

        layer.trainable = False

    # Unfreeze selected final layers
    for layer in backbone.layers[-FINE_TUNE_LAYERS:]:

        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):

            layer.trainable = False

        else:

            layer.trainable = True

    print("\nTrainable ResNet50 layers:")

    for layer in backbone.layers:

        if layer.trainable:

            print(
                f"{layer.name}: {layer.__class__.__name__}"
            )

    return backbone


# ============================================================
# TRAIN RESNET50
# ============================================================

def train_resnet50(
    model,
    train_ds,
    val_ds
):

    stage1_path = (
        MODEL_DIR / "resnet50_stage1_best.keras"
    )

    stage2_path = (
        MODEL_DIR / "resnet50_stage2_best.keras"
    )

    # --------------------------------------------------------
    # PHASE 1: TRAIN CLASSIFICATION HEAD
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("RESNET50 - PHASE 1: CLASSIFICATION HEAD")
    print("=" * 65)

    backbone = model.get_layer("resnet50")

    backbone.trainable = False

    compile_resnet50(
        model,
        learning_rate=HEAD_LEARNING_RATE
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=HEAD_EPOCHS,
        callbacks=get_callbacks(stage1_path),
        shuffle=False
    )

    if not stage1_path.exists():

        raise FileNotFoundError(
            "Phase 1 checkpoint was not created."
        )

    print("\nPhase 1 training completed.")

    # --------------------------------------------------------
    # LOAD BEST PHASE 1 CHECKPOINT
    # --------------------------------------------------------

    print("\nLoading Phase 1 checkpoint...")

    model = load_resnet50_checkpoint(
        stage1_path
    )

    # --------------------------------------------------------
    # PHASE 2: FINE-TUNE RESNET50
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("RESNET50 - PHASE 2: FINE-TUNING")
    print("=" * 65)

    unfreeze_resnet_layers(model)

    compile_resnet50(
        model,
        learning_rate=FINE_TUNE_LEARNING_RATE
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=get_callbacks(stage2_path),
        shuffle=False
    )

    if not stage2_path.exists():

        raise FileNotFoundError(
            "Phase 2 checkpoint was not created."
        )

    print("\nResNet50 training completed.")

    print("Phase 1:", stage1_path)
    print("Phase 2:", stage2_path)

    return (
        stage1_path,
        stage2_path
    )
