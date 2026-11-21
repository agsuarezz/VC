"""Classificador propio SVM sobre geometrÃ­a MediaPipe; selecciÃ³n por sujetos."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from lebron.emotion import BlendshapeExtractor, LABELS, geometry_vector

ROOT = Path(__file__).resolve().parents[1]
CODES = {'HA':'happy','SA':'sad','AN':'angry','SU':'surprise','NE':'neutral'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--extract-only', action='store_true')
    p.add_argument('--select-only', action='store_true', help='No leer ni evaluar etiquetas del holdout')
    args = p.parse_args()
    old_report = ROOT/'reports/metrics.json'
    backup = ROOT/'reports/baseline_v1'
    if not backup.exists():
        backup.mkdir()
        shutil.copy2(old_report, backup/'metrics.json')
        shutil.copy2(ROOT/'reports/confusion_matrix.png', backup/'confusion_matrix.png')
        shutil.copy2(ROOT/'models/emotion.joblib', ROOT/'models/emotion_baseline.joblib')
    baseline = json.loads((backup/'metrics.json').read_text())
    if not args.features.exists():
        extractor = BlendshapeExtractor(ROOT/'models/face_landmarker.task')
        features, geometry, labels, subjects, names, failures = [], [], [], [], [], []
        try:
            for file in sorted(args.source.rglob('*.tiff')):
                parts = file.name.split('.')
                if len(parts) < 3 or parts[1][:2] not in CODES:
                    continue
                values = extractor.extract(cv2.imread(str(file)))
                if values is None:
                    failures.append(file.name)
                    continue
                features.append(values)
                geometry.append(geometry_vector(extractor.last_landmarks,extractor.last_aspect_ratio))
                labels.append(CODES[parts[1][:2]])
                subjects.append(parts[0])
                names.append(file.name)
            args.features.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.features, X=features, geometry=geometry, y=labels, subjects=subjects, names=names,
                                feature_names=extractor.names, failures=failures)
        finally:
            extractor.close()
    data = np.load(args.features, allow_pickle=False)
    X,y,subjects = data['X'], data['y'], data['subjects']
    if args.extract_only:
        print('Extracted', X.shape, 'failed', data['failures'])
        return
    split_map = baseline['dataset']['subject_splits']
    development = np.asarray([split_map[s] != 'test' for s in subjects])
    test = ~development
    devy, groups = y[development], subjects[development]
    feature_sets = {'blendshapes':X, 'geometry':data['geometry'], 'combined':np.column_stack((X,data['geometry']))}
    candidates = []
    for kind, features in feature_sets.items():
        devX = features[development]
        for kernel in ('linear', 'rbf'):
            for c in (.01,.1,1,10,100):
                scores=[]
                for tr,va in GroupKFold(n_splits=4).split(devX,devy,groups):
                    model=make_pipeline(StandardScaler(),SVC(C=c,kernel=kernel,class_weight='balanced'))
                    model.fit(devX[tr],devy[tr])
                    scores.append(f1_score(devy[va],model.predict(devX[va]),average='macro',zero_division=0))
                candidates.append({'features':kind,'kernel':kernel,'C':c,'cv_macro_f1':float(np.mean(scores)),'fold_f1':scores})
    chosen=max(candidates,key=lambda row:row['cv_macro_f1'])
    selection_report={'selected':chosen,'candidates':candidates,'subjects':sorted(set(groups))}
    (ROOT/'reports/development_selection.json').write_text(json.dumps(selection_report,indent=2))
    if args.select_only:
        print(json.dumps(selection_report['selected'],indent=2))
        return
    X=feature_sets[chosen['features']]
    devX=X[development]
    selected=make_pipeline(StandardScaler(),SVC(C=chosen['C'],kernel=chosen['kernel'],class_weight='balanced'))
    started=time.perf_counter()
    selected.fit(devX,devy)
    elapsed=time.perf_counter()-started
    # Only now use the historical holdout, once, after fixing the candidate from development CV.
    pred=selected.predict(X[test])
    report={'version':2,'dataset':baseline['dataset'],'selection':candidates,'selected':chosen,
            'feature_type':chosen['features'],'feature_count':int(X.shape[1]),'training_samples':len(devX),
            'training_subjects':sorted(set(groups)),'test_subjects':sorted(set(subjects[test])),
            'test_accuracy':float(accuracy_score(y[test],pred)),
            'test_macro_f1':float(f1_score(y[test],pred,average='macro',zero_division=0)),
            'test_classification_report':classification_report(y[test],pred,labels=list(LABELS),output_dict=True,zero_division=0),
            'label_order':list(LABELS),'confusion_matrix':confusion_matrix(y[test],pred,labels=list(LABELS)).tolist(),
            'training_seconds':elapsed,'failed_landmarks':data['failures'].tolist(),
            'evaluation_note':'Historical holdout previously reported for v1; no longer a fresh blind test. v2 chosen by grouped development CV only. Webcam performance unmeasured.'}
    joblib.dump({'schema':2,'pipeline':selected,'metadata':report,'feature_type':chosen['features'],
                 'feature_names':data['feature_names'].tolist()},ROOT/'models/emotion.joblib')
    old_report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(7,6))
    ConfusionMatrixDisplay.from_predictions(y[test],pred,labels=list(LABELS),ax=ax,cmap='Blues',colorbar=False)
    ax.set_title('SVM + coeficientes faciales Â· holdout histÃ³rico')
    fig.tight_layout()
    fig.savefig(ROOT/'reports/confusion_matrix.png',dpi=160)
    plt.close(fig)
    print(json.dumps({'selected':chosen,'test_accuracy':report['test_accuracy'],'test_macro_f1':report['test_macro_f1']},indent=2))


if __name__ == '__main__':
    main()
