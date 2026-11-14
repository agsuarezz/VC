import os
import sys
import argparse
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parent
P4_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from traffic_pipeline import TrafficAnalyzer


def _resolve_input(path_str: str) -> str:
    if os.path.exists(path_str):
        return path_str
    candidate = P4_ROOT / path_str
    return str(candidate) if candidate.exists() else path_str


def main():
    parser = argparse.ArgumentParser(description="Procesamiento de vídeo de tráfico con Detección, Tracking y OCR (Grupo 33)")
    parser.add_argument("--video", default="C0142.MP4", help="Ruta al vídeo de entrada")
    parser.add_argument("--output_video", default="results/resultado_C0142_tracking.mp4", help="Vídeo de salida")
    parser.add_argument("--output_csv", default="results/resultado_C0142_detecciones.csv", help="CSV de salida")
    parser.add_argument("--max_frames", type=int, default=None, help="Número máximo de fotogramas a procesar (None para vídeo completo)")
    parser.add_argument("--anonymize", action="store_true", help="Activar anonimización de privacidad (Extra 4)")
    parser.add_argument("--conf", type=float, default=0.10, help="Umbral mínimo de confianza de las detecciones enviadas al tracker")
    parser.add_argument("--model", default="yolo11m.pt", help="Modelo YOLO general (default: yolo11m.pt)")
    parser.add_argument("--imgsz", type=int, default=1280, help="Resolución de inferencia (default: 1280)")
    parser.add_argument("--device", default=None, help="Dispositivo de inferencia (por defecto, CUDA si está disponible y CPU en caso contrario)")
    parser.add_argument("--ocr", default="hybrid", choices=["hybrid", "fast_ocr", "easyocr", "smolvlm", "none"], help="Motor de OCR para matrículas (default: hybrid)")
    parser.add_argument("--use_contours", action="store_true", help="Usar fallback de contornos morfológicos para matrículas (Extra 3)")
    parser.add_argument("--include_parked", action="store_true", help="Desactivar filtro de movimiento e incluir vehículos estacionados")
    parser.add_argument("--motion_min_disp", type=float, default=30.0, help="Desplazamiento mínimo (px) para validar vehículo en movimiento (default: 30.0)")
    args = parser.parse_args()

    plate_candidate = _resolve_input("best_matriculas.pt")
    plate_model = plate_candidate if os.path.exists(plate_candidate) else None
    ocr_mode = "none" if args.anonymize else args.ocr

    analyzer = TrafficAnalyzer(
        general_model_path=_resolve_input(args.model),
        plate_model_path=plate_model,
        conf_thres=args.conf,
        imgsz=args.imgsz,
        device=args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"),
        ocr_mode=ocr_mode
    )

    summary = analyzer.process_video(
        video_path=_resolve_input(args.video),
        output_video_path=args.output_video,
        output_csv_path=args.output_csv,
        max_frames=args.max_frames,
        anonymize=args.anonymize,
        use_contour_plates=args.use_contours,
        only_moving=not args.include_parked,
        motion_min_span=args.motion_min_disp,
        display=False
    )

    print("\n¡Ejecución finalizada con éxito!")

if __name__ == "__main__":
    main()
