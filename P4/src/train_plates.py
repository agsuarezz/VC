import argparse
import os
import torch
from ultralytics import YOLO


def main(epochs=20, imgsz=640, batch=16, workers=4, device=None):
    print("=== ENTRENAMIENTO DE DETECTOR DE MATRÍCULAS (YOLO11) ===")
    has_cuda = torch.cuda.is_available()
    if device is None:
        if has_cuda:
            device = "cuda:0"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Dispositivo de entrenamiento: {device}")
    if has_cuda and "cuda" in str(device):
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif device == "cpu":
        print("Aviso: entrenando en CPU; este proceso puede tardar considerablemente.")

    p4_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_path = os.path.join(p4_root, "dataset_matriculas.yaml")
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(
            f"No se encontró {yaml_path}. Ejecuta primero `python src/prepare_dataset.py`."
        )

    base_weights = os.path.join(p4_root, "yolo11n.pt")
    if not os.path.isfile(base_weights):
        print(f"No se encontró {base_weights}; Ultralytics intentará descargarlo.")

    # Cargar modelo base YOLO11 Nano preentrenado
    model = YOLO(base_weights)

    project_dir = os.path.join(p4_root, "runs", "detect", "runs_matriculas")
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        project=project_dir,
        name="yolo11n_matriculas",
        exist_ok=True,
        seed=42,
        verbose=True,
    )

    best_path = getattr(getattr(model, "trainer", None), "best", None)
    if best_path is None:
        best_path = os.path.join(project_dir, "yolo11n_matriculas", "weights", "best.pt")

    print("\n¡Entrenamiento completado!")
    print(f"Pesos guardados en: {best_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento YOLO11n para detección de matrículas")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    main(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
    )
