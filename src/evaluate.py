"""
Evaluation: macro metrics, confusion matrix, per-class report, top-10 best/worst classes.
"""
import json, numpy as np, pandas as pd
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from config import SPLIT_DIR, MODELS_DIR, METRICS_DIR, MODEL_NAME, IMG_SIZE, BATCH_SIZE
from utils_data import make_generators

def main():
    _, _, test_gen = make_generators(SPLIT_DIR/"train", SPLIT_DIR/"validation", SPLIT_DIR/"test",
                                     img_size=IMG_SIZE, batch_size=BATCH_SIZE)
    with open(METRICS_DIR/"class_indices.json") as f: class_indices = json.load(f)
    idx_to_class = {v:k for k,v in class_indices.items()}

    model_path = MODELS_DIR/f"{MODEL_NAME}_final.keras"
    if not model_path.exists():
        # fallback to phase2 best
        model_path = MODELS_DIR/f"{MODEL_NAME}_phase2_best.keras"
    model = tf.keras.models.load_model(model_path)

    y_true = test_gen.classes
    probs = model.predict(test_gen, verbose=0)
    y_pred = probs.argmax(axis=1)

    # Reports
    report = classification_report(y_true, y_pred, output_dict=True, target_names=[idx_to_class[i] for i in range(len(idx_to_class))])
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(METRICS_DIR/"classification_report.csv")

    cm = confusion_matrix(y_true, y_pred)
    np.save(METRICS_DIR/"confusion_matrix.npy", cm)

    # Top-10 best/worst by recall
    per_class = df_report.iloc[:-3].copy()  # drop avg rows
    per_class["class_name"] = per_class.index
    best10 = per_class.sort_values("recall", ascending=False).head(10)
    worst10 = per_class.sort_values("recall", ascending=True).head(10)
    best10.to_csv(METRICS_DIR/"top10_best_classes.csv", index=False)
    worst10.to_csv(METRICS_DIR/"top10_worst_classes.csv", index=False)

    # Confidence analysis: predicted prob of chosen class
    chosen_conf = probs.max(axis=1)
    np.save(METRICS_DIR/"chosen_confidence.npy", chosen_conf)

    print("Saved: classification_report.csv, confusion_matrix.npy, top10_best_classes.csv, top10_worst_classes.csv, chosen_confidence.npy")

if __name__ == "__main__":
    main()