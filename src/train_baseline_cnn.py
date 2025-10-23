"""
Small baseline CNN. Trains on a small subset to validate the pipeline.
"""
import json, math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from config import SPLIT_DIR, OUT_DIR, MODELS_DIR, METRICS_DIR, IMG_SIZE, BATCH_SIZE, SEED
from utils_data import set_seeds, make_generators

def small_subset(gen, max_per_class=200):
    # Limit steps per epoch by capping samples
    steps = math.ceil(min(gen.samples, max_per_class*len(gen.class_indices))/gen.batch_size)
    return steps

def build_baseline(num_classes):
    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = layers.Conv2D(32,3,activation="relu")(inputs)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64,3,activation="relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(128,3,activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    return model

def main():
    set_seeds(SEED)
    train_gen, val_gen, _ = make_generators(SPLIT_DIR/"train", SPLIT_DIR/"validation", SPLIT_DIR/"test",
                                            img_size=IMG_SIZE, batch_size=BATCH_SIZE)

    model = build_baseline(num_classes=len(train_gen.class_indices))
    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss="categorical_crossentropy",
                  metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_accuracy")])

    steps = small_subset(train_gen, max_per_class=200)
    val_steps = max(1, val_gen.samples // val_gen.batch_size)

    hist = model.fit(train_gen, steps_per_epoch=steps,
                     validation_data=val_gen, validation_steps=val_steps,
                     epochs=5, verbose=1)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR/"baseline_small.keras")

    with open(METRICS_DIR/"baseline_history.json","w") as f:
        json.dump({k:[float(x) for x in v] for k,v in hist.history.items()}, f, indent=2)

    print("Baseline saved.")

if __name__ == "__main__":
    main()