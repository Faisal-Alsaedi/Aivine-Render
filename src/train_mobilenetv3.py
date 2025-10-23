# src/train_mobilenetv3.py
"""
Two-phase training with MobileNetV3 and class weights.
- Phase 1: freeze base, train head
- Phase 2: partial unfreeze
- Class imbalance handled via class_weight
- Rolling checkpoints + resume per phase
- Safe defaults for CPU (workers=2, no multiprocessing)
"""

import os
import json
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications.mobilenet_v3 import MobileNetV3Large
from config import (
    SPLIT_DIR, MODELS_DIR, METRICS_DIR, IMG_SIZE, BATCH_SIZE, OUT_DIR,
    PHASE1_EPOCHS, PHASE1_LR, PHASE1_FREEZE,
    PHASE2_EPOCHS, PHASE2_LR, UNFREEZE_LAYERS,
    MODEL_NAME, SEED
)
from utils_data import set_seeds, make_generators, compute_balanced_class_weights

# ---------------------------- helpers ----------------------------

class HistorySaver(keras.callbacks.Callback):
    """Incrementally persist history after each epoch to avoid loss on interruption."""
    def __init__(self, path: str):
        super().__init__()
        self.path = path
        self.h = {}
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            self.h.setdefault(k, []).append(float(v))
        with open(self.path, "w") as f:
            json.dump(self.h, f, indent=2)

def build_model(num_classes: int, freeze_base: bool = True, alpha: float = 1.0):
    base = MobileNetV3Large(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        pooling="avg",
        alpha=alpha,
    )
    base.trainable = not freeze_base

    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base(inputs, training=not freeze_base)
    x = layers.BatchNormalization(name="head_bn1")(x)
    x = layers.Dropout(0.2, name="head_dropout1")(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(5e-4),
                     name="head_dense1")(x)
    x = layers.BatchNormalization(name="head_bn2")(x)
    x = layers.Dropout(0.2, name="head_dropout2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs, outputs, name="MobileNetV3_PlantDisease")
    return model, base

def callbacks(phase: int):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    return [
        # best checkpoint by val_loss
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / f"{MODEL_NAME}_phase{phase}_best.keras"),
            monitor="val_loss", save_best_only=True, mode="min", verbose=1
        ),
        # rolling checkpoint each epoch (resume point)
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / f"{MODEL_NAME}_phase{phase}_last.keras"),
            save_best_only=False, save_weights_only=False, verbose=0
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True, mode="min", verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, mode="min", verbose=1
        ),
        keras.callbacks.CSVLogger(str(METRICS_DIR / f"training_log_phase{phase}.csv"), append=True),
        HistorySaver(str(METRICS_DIR / f"live_history_phase{phase}.json")),
    ]

def _maybe_resume(model_path: Path, model: keras.Model | None):
    if model_path.exists():
        return keras.models.load_model(model_path)
    return model

def _fit_with_autosave(model, *args, autosave_path: Path, **kwargs):
    try:
        return model.fit(*args, **kwargs)
    except Exception:
        # fail-safe snapshot
        model.save(str(autosave_path))
        raise

# ------------------------------ main -----------------------------

def main():
    set_seeds(SEED)

    # Generators
    train_gen, val_gen, test_gen = make_generators(
        SPLIT_DIR / "train", SPLIT_DIR / "validation", SPLIT_DIR / "test",
        img_size=IMG_SIZE, batch_size=BATCH_SIZE
    )

    num_classes = len(train_gen.class_indices)
    class_weights = compute_balanced_class_weights(SPLIT_DIR / "train", train_gen.class_indices)

    # Optional caps for CPU runs via env (integers). Leave unset to use full dataset.
    steps_per_epoch = int(os.getenv("STEPS_PER_EPOCH", "0")) or None
    val_steps = int(os.getenv("VAL_STEPS", "0")) or None

    # Build or resume Phase 1
    model, base = build_model(num_classes=num_classes, freeze_base=PHASE1_FREEZE, alpha=1.0)
    phase1_last = MODELS_DIR / f"{MODEL_NAME}_phase1_last.keras"
    model = _maybe_resume(phase1_last, model)

    # Compile Phase 1
    model.compile(
        optimizer=optimizers.Adam(PHASE1_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_accuracy")],
    )

    # Train Phase 1
    h1 = _fit_with_autosave(
        model,
        train_gen,
        validation_data=val_gen,
        epochs=PHASE1_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks(1),
        workers=2, use_multiprocessing=False, max_queue_size=10, verbose=1,
        steps_per_epoch=steps_per_epoch, validation_steps=val_steps,
        autosave_path=MODELS_DIR / f"{MODEL_NAME}_phase1_autosave.keras",
    )

    with open(METRICS_DIR / "history_phase1.json", "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in h1.history.items()}, f, indent=2)

    # Phase 2: partial unfreeze
    base.trainable = True
    total_layers = len(base.layers)
    freeze_until = max(0, total_layers - UNFREEZE_LAYERS)
    for l in base.layers[:freeze_until]:
        l.trainable = False

    # Resume Phase 2 if exists
    phase2_last = MODELS_DIR / f"{MODEL_NAME}_phase2_last.keras"
    model = _maybe_resume(phase2_last, model)

    # Compile Phase 2
    model.compile(
        optimizer=optimizers.Adam(PHASE2_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_accuracy")],
    )

    # Train Phase 2
    h2 = _fit_with_autosave(
        model,
        train_gen,
        validation_data=val_gen,
        epochs=PHASE2_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks(2),
        workers=2, use_multiprocessing=False, max_queue_size=10, verbose=1,
        steps_per_epoch=steps_per_epoch, validation_steps=val_steps,
        autosave_path=MODELS_DIR / f"{MODEL_NAME}_phase2_autosave.keras",
    )

    with open(METRICS_DIR / "history_phase2.json", "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in h2.history.items()}, f, indent=2)

    # Save final model
    final_path = MODELS_DIR / f"{MODEL_NAME}_final.keras"
    model.save(str(final_path))

    # Persist class_indices for evaluation
    with open(METRICS_DIR / "class_indices.json", "w") as f:
        json.dump(train_gen.class_indices, f, indent=2)

    # Simple training metadata
    meta = {
        "img_size": list(IMG_SIZE),
        "batch_size": int(BATCH_SIZE),
        "phase1_epochs": int(len(h1.history.get("loss", []))),
        "phase2_epochs": int(len(h2.history.get("loss", []))),
        "phase1_best_val_acc": float(max(h1.history.get("val_accuracy", [0.0])) if h1 else 0.0),
        "phase2_best_val_acc": float(max(h2.history.get("val_accuracy", [0.0])) if h2 else 0.0),
        "unfreeze_layers": int(UNFREEZE_LAYERS),
        "freeze_until_index": int(freeze_until),
        "final_model_path": str(final_path),
    }
    with open(METRICS_DIR / "training_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Training complete.")

if __name__ == "__main__":
    # Ensure base dirs exist
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    main()