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

DEFAULT_OUTPUT_VIDEO = "results/resultado_C0142_tracking.mp4"
DEFAULT_OUTPUT_CSV = "results/resultado_C0142_detecciones.csv"


def _resolve_input(path_str: str) -> str:
    if not path_str:
        return path_str
    if os.path.isfile(path_str):
        return path_str
    candidate = P4_ROOT / path_str
    return str(candidate) if candidate.is_file() else path_str


def _resolve_output(path_str: str, default_rel: str) -> str:
    if not path_str:
        return path_str
    if path_str == default_rel and not os.path.isabs(path_str):
        return str(P4_ROOT / path_str)
    return path_str


def main():
    parser = argparse.ArgumentParser(description="Procesamiento de vídeo de tráfico con Detección, Tracking y OCR (Grupo 33)")
    parser.add_argument("--video", default="C0142.MP4", help="Ruta al vídeo de entrada")
    parser.add_argument("--output_video", default=DEFAULT_OUTPUT_VIDEO, help="Vídeo de salida")
    parser.add_argument("--output_csv", default=DEFAULT_OUTPUT_CSV, help="CSV de salida")
    parser.add_argument("--max_frames", type=int, default=None, help="Número máximo de fotogramas a procesar (None para vídeo completo)")
    parser.add_argument("--anonymize", action="store_true", help="Activar anonimización de privacidad (Extra 4)")
    parser.add_argument("--conf", type=float, default=0.10, help="Umbral mínimo de confianza de las detecciones enviadas al tracker")
    parser.add_argument("--model", default="yolo11m.pt", help="Modelo YOLO general (default: yolo11m.pt)")
    parser.add_argument("--plate_model", default="best_matriculas.pt", help="Modelo YOLO detector de matrículas (default: best_matriculas.pt)")
    parser.add_argument("--imgsz", type=int, default=1280, help="Resolución de inferencia (default: 1280)")
    parser.add_argument("--device", default=None, help="Dispositivo de inferencia (por defecto, CUDA si está disponible y CPU en caso contrario)")
    parser.add_argument("--ocr", default="hybrid", choices=["hybrid", "fast_ocr", "easyocr", "smolvlm", "none"], help="Motor de OCR para matrículas (default: hybrid)")
    parser.add_argument("--use_contours", action="store_true", help="Usar fallback de contornos morfológicos para matrículas (Extra 3)")
    parser.add_argument("--include_parked", action="store_true", help="Desactivar filtro de movimiento e incluir vehículos estacionados")
    parser.add_argument("--motion_min_disp", type=float, default=30.0, help="Desplazamiento mínimo (px) para validar vehículo en movimiento (default: 30.0)")
    args = parser.parse_args()

    if args.max_frames is not None and args.max_frames < 0:
        parser.error("--max_frames debe ser >= 0")

    plate_candidate = _resolve_input(args.plate_model)
    plate_model = plate_candidate if plate_candidate and os.path.isfile(plate_candidate) else None
    ocr_mode = "none" if args.anonymize else args.ocr

    analyzer = TrafficAnalyzer(
        general_model_path=_resolve_input(args.model),
        plate_model_path=plate_model,
        conf_thres=args.conf,
        imgsz=args.imgsz,
        device=args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"),
        ocr_mode=ocr_mode,
    )

    summary = analyzer.process_video(
        video_path=_resolve_input(args.video),
        output_video_path=_resolve_output(args.output_video, DEFAULT_OUTPUT_VIDEO),
        output_csv_path=_resolve_output(args.output_csv, DEFAULT_OUTPUT_CSV),
        max_frames=args.max_frames,
        anonymize=args.anonymize,
        use_contour_plates=args.use_contours,
        only_moving=not args.include_parked,
        motion_min_span=args.motion_min_disp,
        display=False,
    )

    print(
        f"\n¡Ejecución finalizada con éxito! "
        f"(Personas: {summary['unique_persons']}, Vehículos: {summary['unique_vehicles']}, "
        f"Matrículas: {summary['unique_vehicles_with_plate']}, OCR: {summary['recognized_plates_count']})"
    )


if __name__ == "__main__":
    main()
