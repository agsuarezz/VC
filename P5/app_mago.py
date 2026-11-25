"""Filtro de mago con hechizos dibujados en el aire."""
import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from lebron.gestures import HandTracker
from lebron.runtime import DemoRecorder
from lebron.visuals import FaceTracker
from wizard_filter import AirSpell, SPELLS, draw_costume, draw_magic

ROOT = Path(__file__).resolve().parent


def draw_panel(frame, air):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = min(1., w / 650)
    status, color = SPELLS.get(air.spell, ('Traza tu primer hechizo', (223, 191, 249)))
    if air.points:
        status, color = 'Trazando hechizo...', (250, 201, 255)
    elif air.locked and air.cursor is not None:
        status = f'{status} | Baja el dedo para repetir'
    panel = frame[max(0, h-72):, :]
    cv2.addWeighted(panel, .26, np.full_like(panel, (40, 20, 49)), .74, 0, dst=panel)
    cv2.line(frame, (0, max(0, h-72)), (w, max(0, h-72)), (149, 88, 177), 1)
    cv2.putText(frame, status, (16, h-46), font, .71 * scale, color, 1, cv2.LINE_AA)
    cv2.putText(frame, 'Dibuja con el indice y cierra la figura | Baja el dedo para repetir',
                (16, h-26), font, .43 * scale, (241, 220, 245), 1, cv2.LINE_AA)
    cv2.putText(frame, 'Circulo: conejo   Cuadrado: oso   Triangulo: zorro   |   R Grabar   C Limpiar   Q Salir',
                (16, h-8), font, .42 * scale, (226, 186, 237), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--video', type=Path)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--record', type=Path)
    args = parser.parse_args()
    if args.headless and not args.video:
        parser.error('--headless requiere --video')

    cap = cv2.VideoCapture(str(args.video) if args.video else args.camera)
    face = hand = recorder = None
    try:
        if not cap.isOpened():
            raise RuntimeError('No se pudo abrir la camara o el video.')
        if not args.video:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 25
        face = FaceTracker(ROOT / 'models/face_landmarker.task', video=True)
        hand = HandTracker(
            ROOT / 'models/hand_landmarker.task',
            num_hands=2,
            min_detection_confidence=0.35,
            min_presence_confidence=0.30,
            min_tracking_confidence=0.30,
        )
        air = AirSpell()
        count = 0
        requested_record = args.record
        title = 'Hechizos animales - VC P5'
        if not args.headless:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        vignette = None

        while True:
            ok, frame = cap.read()
            if not ok:
                if not count:
                    raise RuntimeError('La camara o el video no entrego imagenes.')
                break
            if not args.video:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            if w > 960:
                frame = cv2.resize(frame, (960, round(h * 960 / w)))
                h, w = frame.shape[:2]
            now = count / fps if args.video else time.monotonic()
            observation = face.detect(frame)
            air.update(hand.detect(frame, now), (w, h), now)
            if vignette is None or vignette.shape[:2] != (h, w):
                x = np.linspace(-1., 1., w, dtype=np.float32)
                y = np.linspace(-1., 1., h, dtype=np.float32)
                vignette = (1 - .23 * np.minimum(1., x*x + y[:, None]**2))[..., None]
            frame = (frame * vignette).astype(np.uint8)
            if observation is not None:
                draw_costume(frame, observation, air.spell)
            draw_magic(frame, air, now)
            draw_panel(frame, air)
            if requested_record:
                recorder = DemoRecorder(requested_record, (w, h), now)
                requested_record = None
            if recorder is not None:
                cv2.putText(frame, 'REC', (14, 26), cv2.FONT_HERSHEY_DUPLEX, .65,
                            (113, 135, 255), 1, cv2.LINE_AA)
                if not recorder.write(frame, now):
                    recorder = None

            count += 1
            if not args.headless:
                cv2.imshow(title, frame)
                key = cv2.waitKey(1) & 255
                if key in (27, ord('q')) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord('c'):
                    air = AirSpell()
                if key == ord('r'):
                    if recorder is None:
                        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        requested_record = ROOT / 'recordings' / f'mago_{stamp}.avi'
                    else:
                        recorder.close()
                        recorder = None
            if args.max_frames and count >= args.max_frames:
                break
        print(f'Procesados {count} fotogramas.')
        return 0
    except (RuntimeError, OSError, ValueError, cv2.error) as exc:
        print(exc, file=sys.stderr)
        return 2
    finally:
        if recorder is not None:
            recorder.close()
        if hand is not None:
            hand.close()
        if face is not None:
            face.close()
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    raise SystemExit(main())
