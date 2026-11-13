import os
import glob
import json
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
    "dirs", "rtdetr2l-detection", "yolo11-detect", 
    "yolo11-pose", "yolo11-seg", "yolov8-tracker"
}

def create_link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)

def main():
    random.seed(RANDOM_SEED)
    
    # 1. Emparejar imágenes y anotaciones
    all_images = glob.glob(os.path.join(IMAGES_DIR, "*.*"))
    img_map = {}
    for p in all_images:
        base, ext = os.path.splitext(os.path.basename(p))
        if base in DOC_IMAGES:
            continue
        if ext.lower() in [".jpg", ".jpeg", ".png"]:
            img_map[base] = p

    all_anns = glob.glob(os.path.join(ANNOTATIONS_DIR, "*.json"))
    ann_map = {}
    for p in all_anns:
        base, _ = os.path.splitext(os.path.basename(p))
        ann_map[base] = p

    common_keys = sorted(list(set(img_map.keys()) & set(ann_map.keys())))
    print(f"Pares válidos encontrados: {len(common_keys)}")

    # 2. Parsear y validar anotaciones
    valid_samples = []
    total_boxes = 0
    
    for key in common_keys:
        ann_path = ann_map[key]
        img_path = img_map[key]
        
        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            continue
            
        boxes = []
        for item in data:
            rect = item.get("rectangle", [])
            if len(rect) == 4:
                xc, yc, w, h = rect
                if w > 0 and h > 0:
                    # Asegurar rango [0, 1]
                    xc = max(0.0, min(1.0, float(xc)))
                    yc = max(0.0, min(1.0, float(yc)))
                    w = max(0.001, min(1.0, float(w)))
                    h = max(0.001, min(1.0, float(h)))
                    boxes.append((0, xc, yc, w, h))
                    
        if boxes:
            valid_samples.append((key, img_path, boxes))
            total_boxes += len(boxes)

    print(f"Muestras válidas con al menos 1 matrícula: {len(valid_samples)}")
    print(f"Total de matrículas etiquetadas: {total_boxes}")

    # 3. Barajar y dividir
    random.shuffle(valid_samples)
    n = len(valid_samples)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    
    train_samples = valid_samples[:n_train]
    val_samples = valid_samples[n_train:n_train + n_val]
    test_samples = valid_samples[n_train + n_val:]
    
    splits = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples
    }

    # 4. Crear estructura de carpetas
    for split_name, samples in splits.items():
        img_split_dir = os.path.join(DATASET_DIR, split_name, "images")
        lbl_split_dir = os.path.join(DATASET_DIR, split_name, "labels")
        os.makedirs(img_split_dir, exist_ok=True)
        os.makedirs(lbl_split_dir, exist_ok=True)
        
        print(f"Generando split '{split_name}' ({len(samples)} imágenes)...")
        for key, src_img, boxes in samples:
            _, ext = os.path.splitext(src_img)
            dst_img = os.path.join(img_split_dir, f"{key}{ext}")
            dst_lbl = os.path.join(lbl_split_dir, f"{key}.txt")
            
            if not os.path.exists(dst_img):
                create_link_or_copy(src_img, dst_img)
                
            with open(dst_lbl, "w", encoding="utf-8") as lf:
                for cls_id, xc, yc, w, h in boxes:
                    lf.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    # 5. Generar archivo dataset_matriculas.yaml
    yaml_path = os.path.join(BASE_DIR, "dataset_matriculas.yaml")
    yaml_content = f"""# Dataset de detección de matrículas (Grupo 33)
path: {DATASET_DIR.replace(os.sep, '/')}
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
