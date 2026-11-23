"""Prototipo 2 (P5): Filtro de arte generativo facial inspirado en 'Más Que la Cara'."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from generative_face import GenerativeFaceFilter, MODE_NAMES
from lebron.runtime import DemoRecorder

ROOT = Path(__file__).resolve().parent


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", type=Path, help="Vídeo de entrada para probar sin webcam")
    parser.add_argument(
        "--mode",
        type=int,
        default=1,
        choices=(1, 2, 3),
        help="Modo inicial: 1=Neon+Ondas, 2=Tipografia, 3=Topografia 3D",
    )
    parser.add_argument("--dark", action="store_true", help="Iniciar con fondo negro puro")
    parser.add_argument("--headless", action="store_true", help="Procesar vídeo sin ventana")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--record", type=Path, help="Grabar desde el primer fotograma, máximo 30s")
    parser.add_argument("--windowed", action="store_true", help="Abrir en ventana en lugar de pantalla completa")
    return parser.parse_args()


def _text(frame, message, y, color=(240, 240, 240)):
    cv2.putText(frame, message, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def _draw_hud(frame, art_filter: GenerativeFaceFilter, observation):
    panel_top = max(0, frame.shape[0] - 84)
    panel_w = min(frame.shape[1], 690)
    panel = frame[panel_top:, :panel_w]
    shade = np.full_like(panel, (18, 14, 28))
    panel[:] = cv2.addWeighted(shade, 0.78, panel, 0.22, 0)
    bg_label = "Fondo negro" if art_filter.dark_background else "Camara atenuada"
    _text(
        frame,
        "ARTE FACIAL | 1/2/3 o Guino: Modo | B Fondo | F Pantalla | H Ayuda | R Grabar | Q Salir",
        panel_top + 21,
    )
    sig = art_filter.last_signals
    if observation is None or sig is None:
        status = f"Modo {art_filter.mode + 1}: {art_filter.mode_name} | {bg_label} | Mira a la camara"
        detail = "Abre la boca (ondas), levanta cejas (chispas), mueve la cabeza (estela) o guina un ojo"
    else:
        status = (
            f"Modo {art_filter.mode + 1}/{len(MODE_NAMES)}: {art_filter.mode_name} | {bg_label}"
        )
        detail = (
            f"Boca {sig.mouth_ratio:.0%} | Sonrisa {sig.smile:.0%} | "
            f"Cejas {sig.brow_lift:.0%} | Movimiento {sig.head_speed:.0%}"
        )
    _text(frame, status, panel_top + 45, (110, 240, 255))
    _text(frame, detail, panel_top + 69, (220, 220, 235))


def main():
    args = arguments()
    if args.headless and not args.video:
        print("--headless requiere --video; no se abre la webcam sin vista previa.", file=sys.stderr)
        return 2

    from lebron.visuals import FaceTracker

    face = cap = recorder = None
    notices = []
    try:
        try:
            face = FaceTracker(ROOT / "models/face_landmarker.task")
        except Exception as exc:
            notices.append("Sin landmarks faciales: ejecuta scripts/download_models.py")
            print(
                f"No se pudo iniciar landmarks faciales: {exc}\nEjecuta python scripts/download_models.py",
                file=sys.stderr,
            )

        art_filter = GenerativeFaceFilter(mode=args.mode - 1, dark_background=args.dark)

        cap = cv2.VideoCapture(str(args.video) if args.video else args.camera)
        if not cap.isOpened():
            print(
                "No se pudo abrir la cámara/vídeo. Comprueba permisos, índice --camera y otras aplicaciones.",
                file=sys.stderr,
            )
            return 2
        if not args.video:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

        video_fps = cap.get(cv2.CAP_PROP_FPS) if args.video else 0
        if not np.isfinite(video_fps) or video_fps <= 0:
            video_fps = 25

        count = 0
        pending_record = args.record
        show_help = True
        fullscreen = not args.windowed
        title = "Arte Generativo Facial - VC P5"

        if not args.headless:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            if fullscreen:
                cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while True:
            ok, frame = cap.read()
            if not ok:
                if not args.video:
                    print("La cámara ha dejado de entregar imágenes.", file=sys.stderr)
                break
            if not args.video:
                frame = cv2.flip(frame, 1)
            if frame.shape[1] > 960:
                frame = cv2.resize(frame, (960, int(frame.shape[0] * 960 / frame.shape[1])))

            now = count / video_fps if args.video else time.monotonic()
            observation = face.detect(frame) if face is not None else None
            output = art_filter.render(frame, observation, now)

            if show_help:
                _draw_hud(output, art_filter, observation)
            for i, notice in enumerate(notices):
                _text(output, notice, output.shape[0] - (98 if show_help else 18) - 20 * i, (80, 190, 255))

            if pending_record is not None:
                recorder = DemoRecorder(pending_record, (output.shape[1], output.shape[0]), now)
                pending_record = None
            if recorder is not None:
                _text(output, "REC (video sin audio, max 30s)", 23, (90, 90, 255))
                if not recorder.write(output, now):
                    print(f"Grabación guardada: {recorder.path}")
                    recorder = None

            count += 1
            if args.max_frames > 0 and count >= args.max_frames:
                break
            if args.headless:
                continue

            cv2.imshow(title, output)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    title,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                )
            elif key == ord("h"):
                show_help = not show_help
            elif key == ord("b"):
                art_filter.toggle_background()
            elif key in (ord("1"), ord("2"), ord("3")):
                art_filter.set_mode(key - ord("1"), now=now)
            elif key == ord("r"):
                if recorder is not None:
                    recorder.close()
                    print(f"Grabación guardada: {recorder.path}")
                    recorder = None
                else:
                    pending_record = ROOT / "recordings" / (
                        "arte_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".avi"
                    )

        print(f"Procesados {count} fotogramas.")
        return 0 if count else 2
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        print(f"Error controlado: {exc}", file=sys.stderr)
        return 2
    finally:
        for resource in (recorder, face):
            if resource is not None:
                resource.close()
        if cap is not None:
            cap.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
