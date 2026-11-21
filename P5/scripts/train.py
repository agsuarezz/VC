"""Select HOG/SVM on validation and report untouched test scores."""
import argparse
import json
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from lebron.emotion import LABELS, SCHEMA, extract_hog


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--model", type=Path, default=Path("models/emotion.joblib"))
    p.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = p.parse_args()
    data = np.load(args.data, allow_pickle=False)
    labels, splits = data["labels"], data["splits"]
    features = np.stack([extract_hog(im) for im in data["images"]])
    masks = {s: splits == s for s in ("train", "val", "test")}
    for s, mask in masks.items():
        if set(labels[mask]) != set(LABELS):
            raise ValueError(f"Faltan clases en {s}")
    if set(data["subjects"]) != {"unknown"}:
        groups = {s: set(data["subjects"][m]) for s, m in masks.items()}
        if groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups["val"] & groups["test"]:
            raise ValueError("Fuga de sujetos entre particiones")
    candidates, best, best_score = [], None, -1.0
    for c in (0.01, 0.1, 1.0):
        model = make_pipeline(StandardScaler(), SVC(kernel="linear", C=c, class_weight="balanced", random_state=42))
        started = time.perf_counter()
        model.fit(features[masks["train"]], labels[masks["train"]])
        pred = model.predict(features[masks["val"]])
        score = f1_score(labels[masks["val"]], pred, labels=list(LABELS), average="macro", zero_division=0)
        candidates.append({"C": c, "validation_macro_f1": float(score), "training_seconds": time.perf_counter()-started})
        if score > best_score:
            best_score, best = score, model
    # Keep selected estimator trained on train only; validation is never refit into scaler.
    started = time.perf_counter()
    pred = best.predict(features[masks["test"]])
    inference_ms = 1000*(time.perf_counter()-started)/len(pred)
    truth = labels[masks["test"]]
    report = {"dataset": json.loads(str(data["metadata"])), "selection": candidates,
              "selected_C": best[-1].C, "test_accuracy": float(accuracy_score(truth, pred)),
              "test_macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
              "test_classification_report": classification_report(truth, pred, labels=list(LABELS), output_dict=True, zero_division=0),
              "label_order": list(LABELS), "confusion_matrix": confusion_matrix(truth, pred, labels=list(LABELS)).tolist(),
              "svm_inference_ms_per_crop_excluding_hog": inference_ms,
              "feature_count": features.shape[1], "python": platform.python_version(), "sklearn": sklearn.__version__,
              "note": "Small subject-disjoint evaluation; not a measure of webcam or emotion-reading reliability."}
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"schema": SCHEMA, "pipeline": best, "metadata": report}, args.model)
    (args.report_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(truth, pred, labels=list(LABELS), ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("HOG + SVM · test independiente")
    fig.tight_layout()
    fig.savefig(args.report_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"test_accuracy": report["test_accuracy"], "test_macro_f1": report["test_macro_f1"], "model": str(args.model)}, indent=2))


if __name__ == "__main__":
    main()
