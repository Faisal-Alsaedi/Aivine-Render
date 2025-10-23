import os, json, random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

def set_seeds(seed:int=42):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def class_counts_from_dir(dir_path: Path) -> Dict[str, int]:
    counts = {}
    for cls in sorted([d for d in dir_path.iterdir() if d.is_dir()]):
        n = len([f for f in cls.iterdir() if f.suffix.lower() in {".jpg",".jpeg",".png"}])
        counts[cls.name] = n
    return counts

def save_distribution_csv(counts: Dict[str,int], weights: Dict[int,float], class_indices: Dict[str,int], out_csv: Path):
    df = pd.DataFrame({
        "Class": list(counts.keys()),
        "Count": list(counts.values()),
        "Weight": [weights[class_indices[c]] for c in counts.keys()]
    }).sort_values("Count")
    df.to_csv(out_csv, index=False)

def make_generators(train_dir: Path, val_dir: Path, test_dir: Path, img_size=(224,224), batch_size=32) -> Tuple:
    train_aug = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.15, zoom_range=0.15, horizontal_flip=True, fill_mode="nearest"
    )
    plain = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_aug.flow_from_directory(
        str(train_dir), target_size=img_size, batch_size=batch_size,
        class_mode="categorical", shuffle=True, seed=42
    )
    val_gen = plain.flow_from_directory(
        str(val_dir), target_size=img_size, batch_size=batch_size,
        class_mode="categorical", shuffle=False
    )
    test_gen = plain.flow_from_directory(
        str(test_dir), target_size=img_size, batch_size=batch_size,
        class_mode="categorical", shuffle=False
    )
    return train_gen, val_gen, test_gen

def compute_balanced_class_weights(train_dir: Path, class_indices: Dict[str,int]) -> Dict[int,float]:
    from sklearn.utils.class_weight import compute_class_weight
    counts = {c: len([f for f in (train_dir / c).iterdir() if f.suffix.lower() in {".jpg",".jpeg",".png"}]) for c in class_indices}
    classes = np.array([class_indices[c] for c in counts.keys()])
    y = np.repeat(classes, [counts[c] for c in counts.keys()])
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return dict(zip(classes, w))