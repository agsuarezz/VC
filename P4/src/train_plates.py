import os
import torch
from ultralytics import YOLO

def main():
    print("=== ENTRENAMIENTO DE DETECTOR DE MATRÍCULAS (YOLO11) ===")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo de entrenamiento: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    p4_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_path = os.path.join(p4_root, "dataset_matriculas.yaml")
    
    # Cargar modelo base YOLO11 Nano preentrenado
    model = YOLO(os.path.join(p4_root, "yolo11n.pt"))
    
    # Lanzar entrenamiento
    results = model.train(
        data=yaml_path,
        epochs=20,
        imgsz=640,
        batch=16,
        device=device,
        workers=4,
        project="runs_matriculas",
        name="yolo11n_matriculas",
        exist_ok=True,
        seed=42,
        verbose=True
    )
    
    print("\n¡Entrenamiento completado!")
    print(f"Pesos guardados en: runs_matriculas/yolo11n_matriculas/weights/best.pt")

if __name__ == "__main__":
    main()
