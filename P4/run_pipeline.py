import os
import argparse
from traffic_pipeline import TrafficAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Procesamiento de vídeo de tráfico (Grupo 33)")
    parser.add_argument("--video", default="C0142.MP4", help="Ruta al vídeo de entrada")
    parser.add_argument("--output_video", default="resultado_C0142_tracking.mp4", help="Vídeo de salida")
    parser.add_argument("--output_csv", default="resultado_C0142_detecciones.csv", help="CSV de salida")
    parser.add_argument("--max_frames", type=int, default=None, help="Número máximo de fotogramas a procesar (None para vídeo completo)")
    parser.add_argument("--anonymize", action="store_true", help="Activar anonimización de privacidad (Extra 4)")
    parser.add_argument("--conf", type=float, default=0.10, help="Umbral mínimo de confianza de las detecciones enviadas al tracker")
    parser.add_argument("--model", default="yolo11m.pt", help="Modelo YOLO general (default: yolo11m.pt)")
    parser.add_argument("--imgsz", type=int, default=1280, help="Resolución de inferencia (default: 1280)")
    parser.add_argument("--use_contours", action="store_true", help="Usar fallback de contornos morfológicos para matrículas (Extra 3)")
    args = parser.parse_args()

    plate_model = "best_matriculas.pt" if os.path.exists("best_matriculas.pt") else None
    
    analyzer = TrafficAnalyzer(
        general_model_path=args.model,
        plate_model_path=plate_model,
        conf_thres=args.conf,
        imgsz=args.imgsz,
        device="cuda:0"
    )
    
    summary = analyzer.process_video(
        video_path=args.video,
        output_video_path=args.output_video,
        output_csv_path=args.output_csv,
        max_frames=args.max_frames,
        anonymize=args.anonymize,
        use_contour_plates=args.use_contours,
        display=False
    )
    
    print("\n¡Ejecución finalizada con éxito!")

if __name__ == "__main__":
    main()
