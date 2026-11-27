import glob
import json
import math
import os
import random
import shutil

# Configuración
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "annotations")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

RANDOM_SEED = 42
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

DOC_IMAGES = {
    "dirs",
    "rtdetr2l-detection",
    "yolo11-detect",
    "yolo11-pose",
    "yolo11-seg",
    "yolov8-tracker",
    "resultado_tracking_hud",
    "resultado_anonimizacion",
    "metricas_entrenamiento",
    "confusion_matrix_matriculas",
    "prediccion_val_matriculas",
    "resultado_tracking_ocr_hud",
    "comparativa_ocr_rendimiento_tiempos",
    "comparativa_ocr_resolucion_errores",
    "ejemplos_comparativa_ocr",
}


def create_link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def _clamp_yolo_box(xc: float, yc: float, w: float, h: float):
    """Recorta esquinas a [0, 1] y recalcula centro y tamaño dentro de [0, 1]."""
    if not all(math.isfinite(v) for v in (xc, yc, w, h)):
        return None
    if w <= 0.0 or h <= 0.0:
        return None
    x1 = max(0.0, min(1.0, xc - w / 2.0))
    y1 = max(0.0, min(1.0, yc - h / 2.0))
    x2 = max(0.0, min(1.0, xc + w / 2.0))
    y2 = max(0.0, min(1.0, yc + h / 2.0))
    nw = x2 - x1
    nh = y2 - y1
    if nw <= 1e-4 or nh <= 1e-4:
        return None
    nxc = round((x1 + x2) / 2.0, 6)
    nyc = round((y1 + y2) / 2.0, 6)
    max_w = max(0.0, 2.0 * min(nxc, 1.0 - nxc))
    max_h = max(0.0, 2.0 * min(nyc, 1.0 - nyc))
    nw = min(round(nw, 6), math.floor(max_w * 1_000_000) / 1_000_000)
    nh = min(round(nh, 6), math.floor(max_h * 1_000_000) / 1_000_000)
    if nw <= 0.0 or nh <= 0.0:
        return None
    return nxc, nyc, nw, nh


def main():
    if not math.isclose(TRAIN_RATIO + VAL_RATIO + TEST_RATIO, 1.0, abs_tol=1e-6):
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO debe sumar 1.0")
    if not os.path.isdir(IMAGES_DIR):
        raise FileNotFoundError(f"No se encontró la carpeta de imágenes: {IMAGES_DIR}")
    if not os.path.isdir(ANNOTATIONS_DIR):
        raise FileNotFoundError(
            f"No se encontró la carpeta de anotaciones: {ANNOTATIONS_DIR}. "
            "Coloca los JSON originales en P4/annotations/ para regenerar el dataset."
        )

    # 1. Emparejar imágenes y anotaciones
    all_images = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.*")))
    img_map = {}
    for p in all_images:
        if not os.path.isfile(p):
            continue
        base, ext = os.path.splitext(os.path.basename(p))
        if base in DOC_IMAGES:
            continue
        if ext.lower() in [".jpg", ".jpeg", ".png"]:
            if base in img_map:
                print(f"Aviso: extensión duplicada para '{base}', se conserva {os.path.basename(img_map[base])}.")
                continue
            img_map[base] = p

    all_anns = sorted(glob.glob(os.path.join(ANNOTATIONS_DIR, "*.json")))
    ann_map = {}
    for p in all_anns:
        if not os.path.isfile(p):
            continue
        base, _ = os.path.splitext(os.path.basename(p))
        ann_map[base] = p

    common_keys = sorted(set(img_map.keys()) & set(ann_map.keys()))
    print(f"Pares válidos encontrados: {len(common_keys)}")
    if not common_keys:
        raise RuntimeError("No se encontraron pares imagen/JSON coincidentes. No se modificó dataset/ ni el YAML.")

    # 2. Parsear y validar anotaciones
    valid_samples = []
    total_boxes = 0

    for key in common_keys:
        ann_path = ann_map[key]
        img_path = img_map[key]

        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"Aviso: error leyendo {os.path.basename(ann_path)} ({exc}); omitido.")
            continue

        if not isinstance(data, list):
            print(f"Aviso: formato JSON inesperado en {os.path.basename(ann_path)}; omitido.")
            continue

        boxes = []
        for item in data:
            if not isinstance(item, dict):
                continue
            rect = item.get("rectangle", [])
            if not isinstance(rect, (list, tuple)) or len(rect) != 4:
                continue
            try:
                xc, yc, w, h = (float(v) for v in rect)
            except (TypeError, ValueError):
                continue
            clamped = _clamp_yolo_box(xc, yc, w, h)
            if clamped is not None:
                boxes.append((0, *clamped))

        if boxes:
            valid_samples.append((key, img_path, boxes))
            total_boxes += len(boxes)

    print(f"Muestras válidas con al menos 1 matrícula: {len(valid_samples)}")
    print(f"Total de matrículas etiquetadas: {total_boxes}")
    if not valid_samples:
        raise RuntimeError("No hay muestras válidas con matrículas etiquetadas.")

    # 3. Barajar y dividir
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(valid_samples)
    n = len(valid_samples)
    n_train = max(1, int(n * TRAIN_RATIO)) if n >= 3 else n
    n_val = max(1, int(n * VAL_RATIO)) if n >= 3 else 0
    if n_train + n_val > n:
        n_val = max(0, n - n_train)

    train_samples = valid_samples[:n_train]
    val_samples = valid_samples[n_train:n_train + n_val]
    test_samples = valid_samples[n_train + n_val:]

    splits = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }

    # 4. Recrear splits limpios e invalidar caches anteriores de Ultralytics
    for split_name, samples in splits.items():
        split_dir = os.path.join(DATASET_DIR, split_name)
        if os.path.isdir(split_dir):
            shutil.rmtree(split_dir)
        img_split_dir = os.path.join(split_dir, "images")
        lbl_split_dir = os.path.join(split_dir, "labels")
        os.makedirs(img_split_dir, exist_ok=True)
        os.makedirs(lbl_split_dir, exist_ok=True)

        print(f"Generando split '{split_name}' ({len(samples)} imágenes)...")
        for key, src_img, boxes in samples:
            _, ext = os.path.splitext(src_img)
            dst_img = os.path.join(img_split_dir, f"{key}{ext}")
            dst_lbl = os.path.join(lbl_split_dir, f"{key}.txt")

            create_link_or_copy(src_img, dst_img)

            with open(dst_lbl, "w", encoding="utf-8") as lf:
                for cls_id, xc, yc, w, h in boxes:
                    lf.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    # 5. Generar archivo dataset_matriculas.yaml portable
    yaml_path = os.path.join(BASE_DIR, "dataset_matriculas.yaml")
    yaml_content = """# Dataset de detección de matrículas (Grupo 33)
path: ./dataset
train: train/images
val: val/images
test: test/images

nc: 1
names: ['matricula']
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"\n¡Dataset generado con éxito en: {DATASET_DIR}")
    print(f"Archivo de configuración guardado en: {yaml_path}")
    print(f"Resumen: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")


if __name__ == "__main__":
    main()
