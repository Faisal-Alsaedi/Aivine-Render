import os
from pathlib import Path

SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 23

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SPLIT_DIR = DATA_DIR / "dataset_flat_structure"   # expects train/validation/test subdirs
OUT_DIR = ROOT / "outputs"
MODELS_DIR = OUT_DIR / "models"
METRICS_DIR = OUT_DIR / "metrics"
PLOTS_DIR = OUT_DIR / "plots"

MODEL_NAME = "mobilenetv3_plant_disease"

PHASE1_EPOCHS = 15
PHASE1_LR = 1e-3
PHASE1_FREEZE = True

PHASE2_EPOCHS = 15
PHASE2_LR = 1e-5
UNFREEZE_LAYERS = 40  # for MobileNetV3-Large

for d in [DATA_DIR, RAW_DIR, SPLIT_DIR, OUT_DIR, MODELS_DIR, METRICS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)