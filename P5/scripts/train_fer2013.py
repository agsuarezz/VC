"""Train and evaluate a webcam-compatible classifier on the official FER2013 splits."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from lebron.emotion import LABELS, BlendshapeExtractor, geometry_vector

ROOT = Path(__file__).resolve().parents[1]


def extract(images):
    detector = BlendshapeExtractor(ROOT / 'models/face_landmarker.task')
    features = np.full((len(images), 128), np.nan, dtype=np.float32)
    try:
        for index, image in enumerate(images):
            face = cv2.resize(image, (160, 160), interpolation=cv2.INTER_CUBIC)
            blend = detector.extract(face)
            if blend is not None:
                features[index] = np.concatenate((blend, geometry_vector(detector.last_landmarks)))
    finally:
        detector.close()
    return features, detector.names


def scores(truth, prediction):
    return dict(accuracy=accuracy_score(truth, prediction),
                macro_f1=f1_score(truth, prediction, labels=LABELS, average='macro', zero_division=0))


def evaluate(model, features, truth):
    prediction = model.predict(features)
    return dict(**scores(truth, prediction),
                by_class=classification_report(truth, prediction, labels=LABELS,
                                               output_dict=True, zero_division=0),
                confusion_matrix=confusion_matrix(truth, prediction, labels=LABELS).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=ROOT / 'models/emotion.joblib')
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    with np.load(args.data, allow_pickle=False) as dataset:
        labels, splits, images = dataset['labels'], dataset['splits'], dataset['images']
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        batches = []
        for index, batch in enumerate(pool.map(extract, np.array_split(images, args.workers * 4)), 1):
            batches.append(batch)
            print(f'Extraccion {index}/{args.workers * 4}', flush=True)
    features = np.concatenate([batch[0] for batch in batches])
    names = next(names for _, names in batches if names is not None)
    valid = np.isfinite(features).all(axis=1)
    coverage = {}
    for split in ('train', 'val', 'test'):
        coverage[split] = {label: [int((valid & (splits == split) & (labels == label)).sum()),
                                    int(((splits == split) & (labels == label)).sum())]
                           for label in LABELS}
    train, val, test = [valid & (splits == split) for split in ('train', 'val', 'test')]
    candidates = []
    fitted = {}
    options = [(f'logistic_C={c}', make_pipeline(StandardScaler(), LogisticRegression(
        C=c, class_weight='balanced', max_iter=500))) for c in (.01, .1, 1.)]
    options.extend((f'boost_leaves={leaves}_iterations={steps}', HistGradientBoostingClassifier(
        max_leaf_nodes=leaves, max_iter=steps, min_samples_leaf=50, learning_rate=.08,
        l2_regularization=2., class_weight='balanced', early_stopping=False, random_state=42))
        for leaves in (7, 15) for steps in (60, 120, 240))
    with threadpool_limits(limits=4):
        for name, model in options:
            model.fit(features[train], labels[train])
            result = dict(name=name, train=scores(labels[train], model.predict(features[train])),
                          validation=scores(labels[val], model.predict(features[val])))
            candidates.append(result)
            print(json.dumps(result), flush=True)
            fitted[name] = model
        best_f1 = max(result['validation']['macro_f1'] for result in candidates)
        competitive = [result for result in candidates
                       if result['validation']['macro_f1'] >= best_f1 - .02]
        selection = min(competitive, key=lambda result: abs(
            result['train']['macro_f1'] - result['validation']['macro_f1']))
        selected = selection['name']
        best_model = fitted[selected]
        baseline = joblib.load(args.baseline)
        if baseline['feature_type'] != 'combined':
            raise ValueError('La referencia debe usar los mismos rasgos combinados.')
        report = dict(dataset='FER2013', label_order=LABELS, coverage=coverage, candidates=candidates,
                      selection_rule='Smallest absolute F1 gap within 0.02 of best validation F1',
                      selected=selected, test=evaluate(best_model, features[test], labels[test]),
                      baseline_test=evaluate(baseline['pipeline'], features[test], labels[test]))
    args.destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(schema=2, feature_type='combined', feature_names=names,
                     pipeline=best_model, metadata=report), args.destination / 'emotion_fer2013.joblib')
    (args.destination / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('coverage', 'selected', 'test', 'baseline_test')}))


if __name__ == '__main__':
    main()
