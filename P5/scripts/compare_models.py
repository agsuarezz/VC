"""Compara SVM y MLP propios con DeepFace sobre las mismas particiones JAFFE."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('DEEPFACE_HOME', str(ROOT / 'data'))
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')

import cv2
import joblib
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from lebron.emotion import LABELS, BlendshapeExtractor, geometry_vector


def metrics(y, predictions):
    matrix_labels = list(LABELS) + sorted(set(predictions) - set(LABELS))
    return dict(accuracy=float(accuracy_score(y, predictions)),
                macro_f1=float(f1_score(y, predictions, labels=LABELS, average='macro', zero_division=0)),
                confusion_labels=matrix_labels,
                confusion_matrix=confusion_matrix(y, predictions, labels=matrix_labels).tolist(),
                by_class=classification_report(y, predictions, labels=LABELS, output_dict=True, zero_division=0))


def latency(values):
    return dict(mean_ms=float(np.mean(values)), median_ms=float(np.median(values)),
                p95_ms=float(np.percentile(values, 95)))


def plot_report(destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    report = json.loads((destination / 'comparison.json').read_text(encoding='utf-8'))
    models = list(report['models'])
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    x = np.arange(len(models))
    for offset, key, label in [(-.18, 'accuracy', 'Accuracy'), (.18, 'macro_f1', 'F1 macro')]:
        bars = axes[0].bar(x + offset, [report['models'][m][key] * 100 for m in models],
                           width=.34, label=label)
        axes[0].bar_label(bars, fmt='%.1f', fontsize=9, padding=3)
    axes[0].set(xticks=x, xticklabels=models, ylim=(0, 100), ylabel='Porcentaje (%)',
                title='JAFFE: 30 imágenes, 2 personas reservadas')
    axes[0].legend(frameon=False)
    times = [report['models'][m]['total_latency']['mean_ms'] for m in models]
    bars = axes[1].bar(models, times, color=['#3b82b6', '#24a08b', '#d18b36'][:len(models)])
    axes[1].bar_label(bars, fmt='%.2f ms', padding=5)
    axes[1].set(ylabel='Milisegundos por recorte', ylim=(0, max(times) * 1.3),
                title='Extracción + clasificación (CPU)')
    fig.tight_layout()
    fig.savefig(destination / 'comparison.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(models), figsize=(4.7 * len(models), 4.8), squeeze=False)
    translated = dict(zip(LABELS, ['Alegría', 'Tristeza', 'Enfado', 'Sorpresa', 'Neutral']))
    for ax, name in zip(axes[0], models):
        result = report['models'][name]
        labels = [translated.get(label, label) for label in result['confusion_labels']]
        display = ConfusionMatrixDisplay(np.array(result['confusion_matrix']), display_labels=labels)
        display.plot(ax=ax, colorbar=False, cmap='Blues', values_format='d')
        display.im_.set_clim(0, 6)
        ax.set(title=name, xlabel='Predicción', ylabel='Clase real')
        ax.tick_params(axis='x', rotation=40)
    fig.tight_layout()
    fig.savefig(destination / 'confusion_comparison.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, help='JAFFE preparado con prepare_dataset.py')
    parser.add_argument('--destination', type=Path, default=ROOT / 'reports')
    parser.add_argument('--deepface-only', action='store_true', help='Completar un informe propio ya generado')
    parser.add_argument('--plot-only', action='store_true', help='Regenerar gráficas desde el informe (entorno principal)')
    args = parser.parse_args()
    if args.plot_only:
        if args.deepface_only:
            parser.error('Elige --plot-only o --deepface-only')
        plot_report(args.destination)
        return
    if args.data is None:
        parser.error('--data es obligatorio para entrenar o evaluar')
    args.destination.mkdir(parents=True, exist_ok=True)
    output = args.destination / 'comparison.json'
    with np.load(args.data, allow_pickle=False) as data:
        images, labels, splits, subjects = [data[k] for k in ('images', 'labels', 'splits', 'subjects')]
        names = data['names']
    masks = {s: splits == s for s in ('train', 'val', 'test')}
    if set(subjects) == {'unknown'}:
        raise ValueError('Esta comparativa requiere sujetos conocidos y separados (JAFFE).')
    groups = {s: set(subjects[m]) for s, m in masks.items()}
    if any(groups[a] & groups[b] for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))):
        raise ValueError('Hay sujetos compartidos entre particiones.')
    if any(set(labels[m]) != set(LABELS) for m in masks.values()):
        raise ValueError('Todas las particiones deben contener las cinco clases.')
    digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
    cv2.setNumThreads(1)
    if not args.deepface_only:
        extractor = BlendshapeExtractor(ROOT / 'models/face_landmarker.task')
        features, extraction_ms = [], []
        try:
            for image in images:
                start = time.perf_counter()
                blend = extractor.extract(image)
                if blend is None:
                    raise ValueError('No se detectó un rostro: no se descartan muestras en la comparativa.')
                features.append(np.concatenate((blend, geometry_vector(
                    extractor.last_landmarks, extractor.last_aspect_ratio))))
                extraction_ms.append((time.perf_counter() - start) * 1000)
        finally:
            extractor.close()
        features = np.asarray(features)
        report = dict(dataset='JAFFE', dataset_sha256=digest, label_order=LABELS,
                      subjects={s: sorted(g) for s, g in groups.items()},
                      counts={s: int(m.sum()) for s, m in masks.items()},
                      environment=dict(python=platform.python_version(), sklearn=sklearn.__version__,
                                       platform=platform.platform(), processor=platform.processor(), threads=1),
                      protocol='Selection by validation macro F1; train only; shared 128 features; historical holdout.',
                      extraction=latency(np.array(extraction_ms)[masks['test']]), models={})
        configurations = {
            'SVM': [(f'C={c}', SVC(kernel='linear', C=c, class_weight='balanced', random_state=42))
                    for c in (.01, .1, 1.)],
            'MLP': [(f'hidden={hidden},alpha={alpha}', MLPClassifier(
                hidden_layer_sizes=hidden, alpha=alpha, solver='lbfgs', max_iter=1000,
                max_fun=25000, random_state=42, early_stopping=False))
                for hidden in ((32,), (64, 32)) for alpha in (.1, 1.)],
        }
        model_dir = args.destination / 'trained_models'
        model_dir.mkdir(exist_ok=True)
        with threadpool_limits(limits=1):
            for family, options in configurations.items():
                best, best_score, candidates = None, -1., []
                for name, estimator in options:
                    model = make_pipeline(StandardScaler(), estimator)
                    started = time.perf_counter()
                    model.fit(features[masks['train']], labels[masks['train']])
                    elapsed = time.perf_counter() - started
                    score = metrics(labels[masks['val']], model.predict(features[masks['val']]))['macro_f1']
                    candidates.append(dict(configuration=name, validation_macro_f1=score, training_seconds=elapsed))
                    if score > best_score:
                        best, best_score, selected = model, score, candidates[-1]
                x = features[masks['test']]
                best.predict(x[:1])
                timings, predictions = [], []
                for row in x:
                    started = time.perf_counter()
                    prediction = best.predict(row.reshape(1, -1))[0]
                    timings.append((time.perf_counter() - started) * 1000)
                    predictions.append(str(prediction))
                path = model_dir / f'{family.lower()}.joblib'
                joblib.dump(dict(schema=2, feature_type='combined', feature_names=extractor.names,
                                 pipeline=best), path, compress=3)
                result = dict(**metrics(labels[masks['test']], predictions), selected=selected,
                              candidates=candidates, classifier_latency=latency(timings),
                              total_latency=latency(np.array(timings) + np.array(extraction_ms)[masks['test']]),
                              model_bytes=path.stat().st_size, predictions=predictions)
                if family == 'MLP':
                    result['parameters'] = int(sum(a.size for a in (*best[-1].coefs_, *best[-1].intercepts_)))
                    result['iterations'] = int(best[-1].n_iter_)
                report['models'][family] = result
                print(f'{family}: {result["accuracy"]:.4f}', flush=True)
        report['evaluation_names'] = names[masks['test']].tolist()
        report['truth'] = labels[masks['test']].tolist()
        output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        return

    report = json.loads(output.read_text(encoding='utf-8'))
    if report['dataset_sha256'] != digest:
        raise ValueError('El dataset no coincide con el de los modelos propios.')
    from deepface import DeepFace
    from importlib.metadata import version
    DeepFace.build_model('Emotion', task='facial_attribute')
    def predict(image):
        return DeepFace.analyze(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), actions=['emotion'],
                                detector_backend='skip', enforce_detection=False,
                                align=False, silent=True)[0]['dominant_emotion']
    samples = images[masks['test']]
    predict(samples[0])  # Calentamiento excluido, igual que en los clasificadores propios.
    predictions, timings = [], []
    for image in samples:
        started = time.perf_counter()
        predictions.append(predict(image))
        timings.append((time.perf_counter() - started) * 1000)
    # Se mantienen fear/disgust si ganan: son errores, sin restringir artificialmente el modelo.
    report['models']['DeepFace'] = dict(**metrics(labels[masks['test']], predictions),
        total_latency=latency(timings), predictions=predictions,
        outside_target_classes=sum(p not in LABELS for p in predictions),
        backend='skip (mismos recortes, sin otro detector)', pretrained=True,
        versions={name: version(name) for name in ('deepface', 'tensorflow', 'tf-keras')})
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['models']['DeepFace'], indent=2))


if __name__ == '__main__':
    main()
