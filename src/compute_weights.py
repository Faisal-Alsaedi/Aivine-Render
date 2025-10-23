"""
Compute class weights and save distribution CSV + class_indices JSON.
"""
import json
from pathlib import Path
from config import SPLIT_DIR, METRICS_DIR
from utils_data import make_generators, compute_balanced_class_weights, save_distribution_csv

def main():
    train_gen, val_gen, test_gen = make_generators(SPLIT_DIR/"train", SPLIT_DIR/"validation", SPLIT_DIR/"test")
    class_indices = train_gen.class_indices
    weights = compute_balanced_class_weights(SPLIT_DIR/"train", class_indices)

    (METRICS_DIR).mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR/"class_indices.json","w") as f:
        json.dump(class_indices, f, indent=2)
    with open(METRICS_DIR/"class_weights.json","w") as f:
        json.dump({int(k): float(v) for k,v in weights.items()}, f, indent=2)

    # distribution CSV
    save_distribution_csv(
        {c: len(list((SPLIT_DIR/"train"/c).glob("*.*"))) for c in class_indices},
        weights, class_indices, METRICS_DIR/"class_distribution.csv"
    )
    print("Saved: class_indices.json, class_weights.json, class_distribution.csv")

if __name__ == "__main__":
    main()