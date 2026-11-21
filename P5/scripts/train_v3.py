"""Entrena una SVM con JAFFE y WSEFEP, separando personas por corpus."""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
from xml.etree import ElementTree as ET
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from lebron.emotion import BlendshapeExtractor, LABELS, geometry_vector

ROOT = Path(__file__).resolve().parents[1]
WARSAW_SHA256 = 'af1abcc7700d8055abd5430157071ecae2b5a429d46df78816d3e4c609fb7b28'
WARSAW_LABELS = {'joy': 'happy', 'sadness': 'sad', 'anger': 'angry', 'surprise': 'surprise', 'neutral': 'neutral'}
JAFFE_LABELS = {'HA': 'happy', 'SA': 'sad', 'AN': 'angry', 'SU': 'surprise', 'NE': 'neutral'}


def warsaw_rows(archive):
    with archive.open('WSEFEP_v101_hq/WSEFEP - norms & FACS.xlsx') as metadata:
        with ZipFile(BytesIO(metadata.read())) as workbook:
            ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
            strings = [''.join(e.itertext()) for e in ET.fromstring(workbook.read('xl/sharedStrings.xml'))]
            rows = ET.fromstring(workbook.read('xl/worksheets/sheet1.xml')).findall('.//s:row', ns)
    for row in rows[1:]:
        cells = {}
        for cell in row:
            val = cell.findtext('s:v', namespaces=ns)
            if val is not None:
                cells[cell.get('r')[0]] = strings[int(val)] if cell.get('t') == 's' else val
        label = WARSAW_LABELS.get(cells.get('D', '').strip().lower())
        if label:
            yield 'warsaw:' + cells['A'], label, 'WSEFEP_v101_hq/' + cells['B']


def warsaw_images(archive, rows):
    files = {name.lower(): name for name in archive.namelist()}
    for person, label, filename in rows:
        if filename.lower() not in files:
            yield person, label, None
            continue
        raw = archive.read(files[filename.lower()])
        yield person, label, cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def jaffe_images(directory):
    for path in sorted(directory.rglob('*.tiff')):
        person, expression, *_ = path.name.split('.')
        if expression[:2] in JAFFE_LABELS:
            yield 'jaffe:' + person, JAFFE_LABELS[expression[:2]], cv2.imread(str(path))


def extract(args, old):
    with args.warsaw.open('rb') as source:
        if hashlib.file_digest(source, 'sha256').hexdigest() != WARSAW_SHA256:
            raise ValueError('El ZIP WSEFEP no coincide con la versión oficial utilizada.')
    detector = BlendshapeExtractor(ROOT / 'models/face_landmarker.task')
    features, labels, groups, augmented = [], [], [], []
    rejected = 0
    try:
        with ZipFile(args.warsaw) as archive:
            from itertools import chain
            # Sujetos de Varsovia reservados antes de extraer o evaluar rasgos.
            rows = list(warsaw_rows(archive))
            people = sorted({person for person, _, _ in rows})
            held = set(np.random.default_rng(431).permutation(people)[:8])
            held.update('jaffe:' + p for p, split in old['dataset']['subject_splits'].items() if split == 'test')
            for person, label, image in chain(jaffe_images(args.jaffe), warsaw_images(archive, rows)):
                if image is None:
                    rejected += 1
                    continue
                h, w = image.shape[:2]
                if max(h, w) > 720:
                    image = cv2.resize(image, (round(w * 720 / max(h,w)), round(h * 720 / max(h,w))), interpolation=cv2.INTER_AREA)
                variants = [image]
                if person not in held:
                    variants.append(cv2.flip(image, 1))
                    variants.append(cv2.convertScaleAbs(cv2.GaussianBlur(image, (5, 5), 1), alpha=.78, beta=7))
                for index, variant in enumerate(variants):
                    blend = detector.extract(variant)
                    if blend is None:
                        if index == 0:
                            rejected += 1
                            break
                        continue
                    geometry = geometry_vector(detector.last_landmarks, detector.last_aspect_ratio)
                    features.append(np.concatenate((blend, geometry)))
                    labels.append(label)
                    groups.append(person)
                    augmented.append(index > 0)
    finally:
        detector.close()
    args.features.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.features, X=features, y=labels, groups=groups,
                        augmented=augmented, holdout=sorted(held), names=detector.names, rejected=rejected)


def metrics(y, predicted):
    return {'accuracy': accuracy_score(y, predicted),
            'macro_f1': f1_score(y, predicted, labels=LABELS, average='macro', zero_division=0),
            'count': len(y), 'confusion_matrix': confusion_matrix(y, predicted, labels=LABELS).tolist(),
            'by_class': classification_report(y, predicted, labels=LABELS, output_dict=True, zero_division=0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jaffe', required=True, type=Path)
    parser.add_argument('--warsaw', required=True, type=Path)
    parser.add_argument('--features', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    args = parser.parse_args()
    old = joblib.load(ROOT / 'models/emotion_v2.joblib')
    if not args.features.exists():
        extract(args, old['metadata'])
    with np.load(args.features, allow_pickle=False) as data:
        X, y, groups, aug = (data[k] for k in ('X', 'y', 'groups', 'augmented'))
        test = np.isin(groups, data['holdout']) & ~aug
        development = ~np.isin(groups, data['holdout'])
        names, rejected = data['names'].tolist(), int(data['rejected'])
    corpus = np.array([person.split(':')[0] for person in groups])
    folds = [set() for _ in range(4)]
    for source in ('jaffe', 'warsaw'):
        persons = sorted(set(groups[development & (corpus == source)]))
        for fold, chunk in zip(folds, np.array_split(np.random.default_rng(88).permutation(persons), 4)):
            fold.update(chunk)
    candidates = []
    for kind, columns in [('blendshapes', slice(0, 52)), ('combined', slice(None))]:
        features = X[:, columns]
        for use_aug in (False, True):
            for kernel in ('linear', 'rbf'):
                for c in (.01, .1, 1, 10):
                    scores = []
                    for fold in folds:
                        inside = np.isin(groups, list(fold))
                        train = development & ~inside & (use_aug | ~aug)
                        val = development & inside & ~aug
                        model = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=c, class_weight='balanced'))
                        model.fit(features[train], y[train])
                        scores.append(f1_score(y[val], model.predict(features[val]), average='macro'))
                    candidates.append(dict(kind=kind, augmented=use_aug, kernel=kernel, C=c, cv_macro_f1=float(np.mean(scores))))
    best = max(candidates, key=lambda candidate: candidate['cv_macro_f1'])
    X_new = X if best['kind'] == 'combined' else X[:, :52]
    train = development & (best['augmented'] | ~aug)
    model = make_pipeline(StandardScaler(), SVC(kernel=best['kernel'], C=best['C'], class_weight='balanced'))
    model.fit(X_new[train], y[train])
    old_X = X if old['feature_type'] == 'combined' else X[:, :52]
    by_corpus = {}
    for source in ('jaffe', 'warsaw'):
        mask = test & (corpus == source)
        by_corpus[source] = {'v2': metrics(y[mask], old['pipeline'].predict(old_X[mask])),
                             'v3': metrics(y[mask], model.predict(X_new[mask]))}
    report = {'version': 3, 'selection': best, 'candidates': candidates, 'label_order': LABELS,
              'training_samples': int(train.sum()), 'development_originals': int((development & ~aug).sum()),
              'training_subjects': sorted(set(groups[train])), 'test_subjects': sorted(set(groups[test])),
              'development_people': len(set(groups[development])), 'rejected_originals': rejected,
              'held_out_people': {source: len(set(groups[test & (corpus == source)])) for source in by_corpus},
              'test': by_corpus}
    args.destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(schema=2, feature_type=best['kind'], feature_names=names, pipeline=model, metadata=report), args.destination / 'emotion.joblib', compress=3)
    (args.destination / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'selected': best, 'fit': int(train.sum()), 'originals': int((development & ~aug).sum()),
                      'results': {k: {v: [p['accuracy'], p['macro_f1'], p['count']] for v,p in s.items()}
                                  for k,s in by_corpus.items()}}, indent=2))


if __name__ == '__main__':
    main()
