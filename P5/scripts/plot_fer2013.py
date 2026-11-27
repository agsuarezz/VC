"""Genera las figuras de LeBron a partir del informe de entrenamiento de FER2013."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


REPORTS = Path(__file__).resolve().parents[1] / 'reports'
EXPRESSIONS = {
    'happy': 'Alegría', 'sad': 'Tristeza', 'angry': 'Enfado',
    'surprise': 'Sorpresa', 'neutral': 'Neutral',
}
TRAIN = '#52758e'
VALIDATION = '#ae5a22'
SELECTED = '#136f61'


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor='white')
    plt.close(fig)


def plot_coverage(report, labels, output):
    fig, ax = plt.subplots(figsize=(9.5, 4.7), layout='constrained')
    rows = np.arange(len(labels))
    splits = [('train', 'Entrenamiento', TRAIN), ('val', 'Validación', VALIDATION),
              ('test', 'Evaluación', SELECTED)]
    for offset, (split, title, color) in zip((-.23, 0, .23), splits):
        valid, total = np.array([report['coverage'][split][label] for label in labels]).T
        bars = ax.barh(rows + offset, 100 * valid / total, height=.2, label=title, color=color)
        ax.bar_label(bars, fmt='%.1f', padding=4, fontsize=9)
    ax.set(yticks=rows, yticklabels=[EXPRESSIONS[label] for label in labels], xlim=(0, 105),
           xlabel='Imágenes con vector facial válido', title='Cobertura del extractor facial en FER2013')
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.legend(ncol=3, loc='lower center', bbox_to_anchor=(.5, 1.03), frameon=False)
    ax.set_title(ax.get_title(), y=1.15)
    ax.grid(axis='x', alpha=.2)
    ax.set_axisbelow(True)
    save(fig, output / 'fer2013_coverage.png')


def candidate_label(name):
    family, parameters = name.split('_', 1)
    if family == 'logistic':
        return f'Logística · {parameters.replace(".", ",")}'
    leaves, steps = (part.split('=')[1] for part in parameters.split('_'))
    return f'Boosting · {leaves} hojas, {steps} iter.'


def plot_selection(report, output):
    fig, ax = plt.subplots(figsize=(10, 5.5), layout='constrained')
    candidates = report['candidates']
    rows = np.arange(len(candidates))
    train = [run['train']['macro_f1'] for run in candidates]
    validation = [run['validation']['macro_f1'] for run in candidates]
    names = [run['name'] for run in candidates]
    chosen = names.index(report['selected'])
    ax.axhspan(chosen - .37, chosen + .37, color='#dcf0ea')
    ax.axvline(max(validation) - .02, color=SELECTED, linestyle='--', linewidth=1.3,
               label='Umbral mínimo de validación')
    ax.hlines(rows, validation, train, color='#b3bfc7', linewidth=2)
    ax.scatter(train, rows, color=TRAIN, marker='s', s=46, label='Entrenamiento', zorder=3)
    ax.scatter(validation, rows, color=VALIDATION, s=46, label='Validación', zorder=3)
    ax.annotate('Modelo elegido', xy=(train[chosen], chosen), xytext=(10, 0),
                textcoords='offset points', va='center', color=SELECTED, fontweight='bold')
    ax.set(yticks=rows, yticklabels=[candidate_label(name) for name in names],
           xlim=(min(validation) - .025, max(train) + .025), xlabel='F1 macro',
           title='Selección del clasificador de LeBron')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(loc='center right', frameon=False, fontsize=9)
    save(fig, output / 'fer2013_selection.png')


def plot_confusion(report, labels, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), layout='constrained')
    sample_count = sum(sum(row) for row in report['test']['confusion_matrix'])
    sample_count = f'{sample_count:,}'.replace(',', '.')
    fig.suptitle(f'FER2013 · Evaluación sobre {sample_count} imágenes con características válidas',
                 fontsize=14)
    titles = [('baseline_test', 'SVM anterior'), ('test', 'Boosting FER2013 · LeBron')]
    classes = [EXPRESSIONS[label] for label in labels]
    indices = np.arange(len(labels))
    for ax, (key, title) in zip(axes, titles):
        counts = np.asarray(report[key]['confusion_matrix'])
        proportions = counts / counts.sum(axis=1, keepdims=True)
        heatmap = ax.imshow(proportions, cmap='Blues', vmin=0, vmax=1)
        for (row, col), proportion in np.ndenumerate(proportions):
            ax.text(col, row, f'{proportion:.1%}\n({counts[row, col]})', ha='center', va='center',
                    color='white' if proportion > .55 else '#12283a', fontsize=9)
        ax.set(xticks=indices, xticklabels=classes, yticks=indices, yticklabels=classes,
               title=title, xlabel='Predicción', ylabel='Clase real')
        ax.tick_params(axis='x', rotation=24)
    fig.colorbar(heatmap, ax=axes, shrink=.75, format=PercentFormatter(xmax=1),
                 label='Porcentaje por clase real')
    save(fig, output / 'fer2013_confusion.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, default=REPORTS / 'metrics_fer2013.json')
    parser.add_argument('--output', type=Path, default=REPORTS)
    args = parser.parse_args()
    report = json.loads(args.metrics.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True, exist_ok=True)
    labels = report['label_order']
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})
    plot_coverage(report, labels, args.output)
    plot_selection(report, args.output)
    plot_confusion(report, labels, args.output)


if __name__ == '__main__':
    main()
