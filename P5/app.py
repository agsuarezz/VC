"""Filtro LeBron con barba, corona, sonidos por expresión, tiro y talco."""
import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from lebron.runtime import DemoRecorder, ExpressionCue, PredictionSmoother

ROOT = Path(__file__).resolve().parent


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--video', type=Path, help='Vídeo de entrada para probar sin webcam')
    parser.add_argument('--model', type=Path, default=ROOT / 'models/emotion_fer2013.joblib')
    parser.add_argument('--headless', action='store_true', help='Procesar vídeo sin ventana')
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--record', type=Path, help='Grabar desde el primer fotograma, máximo 30s')
    parser.add_argument('--mute', action='store_true')
    parser.add_argument('--windowed', action='store_true', help='Abrir en ventana en lugar de pantalla completa')
    return parser.parse_args()


def draw_predominant_emotion(frame, active, names):
    """Muestra únicamente el nombre de la emoción predominante, sin paneles."""
    name = names.get(active)
    if not name:
        return
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = max(0.75, min(1.05, frame.shape[1] / 960.0))
    pos = (20, int(round(44 * scale)))
    cv2.putText(frame, name, pos, font, scale, (15, 15, 15), 4, cv2.LINE_AA)
    cv2.putText(frame, name, pos, font, scale, (250, 250, 250), 2, cv2.LINE_AA)


def draw_ball(frame, origin, age):
    x = int(origin[0] + 130 * age)
    y = int(origin[1] - 330 * age + 215 * age * age)
    radius = 15
    cv2.circle(frame, (x, y), radius, (35, 150, 245), -1, cv2.LINE_AA)
    cv2.circle(frame, (x, y), radius, (35, 45, 65), 2, cv2.LINE_AA)
    cv2.line(frame, (x-radius+1, y), (x+radius-1, y), (35, 45, 65), 2, cv2.LINE_AA)
    cv2.ellipse(frame, (x, y), (7, radius), 0, 0, 360, (35, 45, 65), 1, cv2.LINE_AA)


def main():
    args = arguments()
    if args.headless and not args.video:
        print('--headless requiere --video; no se abre la webcam sin vista previa.', file=sys.stderr)
        return 2
    from lebron.visuals import (
        FaceTracker,
        TacoRain,
        apply_sad_grayscale,
        draw_lebron_filter,
        draw_sun,
        draw_taco_tuesday,
    )
    from lebron.gestures import HandTracker, GestureEngine, ParticleSystem
    from lebron.audio import AudioPlayer, EXPRESSION_SOUNDS
    from lebron.emotion import EmotionModel, ExpressionFaceCropper

    expression_names = {label: name for label, (name, _) in EXPRESSION_SOUNDS.items()}
    expression_names['neutral'] = 'Neutral'
    face = hands = cap = audio = recorder = emotion = None
    try:
        try:
            face = FaceTracker(ROOT / 'models/face_landmarker.task', video=True)
        except Exception as exc:
            print(f'No se pudo iniciar landmarks faciales: {exc}\nEjecuta python scripts/download_models.py', file=sys.stderr)
        try:
            gesture_model = ROOT / 'models/gesture_recognizer.task'
            hands = HandTracker(gesture_model if gesture_model.is_file() else ROOT / 'models/hand_landmarker.task')
        except Exception as exc:
            print(f'Manos no disponibles: {exc}', file=sys.stderr)
        emotion = None
        if args.model.is_file():
            try:
                emotion = EmotionModel(args.model)
            except Exception as exc:
                print(f'No se pudo cargar el modelo de expresiones: {exc}', file=sys.stderr)
        audio = AudioPlayer(ROOT / 'assets')
        audio.muted = args.mute
        gestures, particles = GestureEngine(posture_effects=False), ParticleSystem()
        taco_rain = TacoRain()
        smoother = PredictionSmoother()
        reaction = ExpressionCue()
        use_observation = emotion is not None and emotion.needs_observation
        expression_cropper = ExpressionFaceCropper() if emotion is not None and not use_observation else None
        cap = cv2.VideoCapture(str(args.video) if args.video else args.camera)
        if not cap.isOpened():
            print('No se pudo abrir la cámara/vídeo. Comprueba permisos, índice --camera y otras aplicaciones.', file=sys.stderr)
            return 2
        if not args.video:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        count = 0
        label = None
        last_predict, last_event_at = -1., -10.
        last_event, event_origin = None, (0, 0)
        video_fps = cap.get(cv2.CAP_PROP_FPS) if args.video else 0
        if not np.isfinite(video_fps) or video_fps <= 0:
            video_fps = 25
        pending_record = args.record
        last_face_y, last_face_at = None, -10.
        fullscreen = not args.windowed
        title = 'LeBron Lab - VC P5'
        if not args.headless:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            if fullscreen:
                cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        while True:
            ok, frame = cap.read()
            if not ok:
                if not args.video:
                    print('La cámara ha dejado de entregar imágenes.', file=sys.stderr)
                break
            if not args.video:
                frame = cv2.flip(frame, 1)
            if frame.shape[1] > 960:
                frame = cv2.resize(frame, (960, int(frame.shape[0] * 960 / frame.shape[1])))
            wall_now = time.monotonic()
            now = count / video_fps if args.video else wall_now
            observation = face.detect(frame) if face is not None else None
            bbox = (observation.bbox if observation is not None else None)
            if expression_cropper is not None:
                bbox = expression_cropper.detect(frame)
            if bbox is None:
                smoother.reset()
                label = None
            elif emotion is not None and now - last_predict >= .15:
                x, y, w, h = bbox
                crop = frame[y:y+h, x:x+w]
                if crop.size:
                    if use_observation and observation is not None:
                        predicted, _ = emotion.predict_observation(observation, include_scores=True)
                    else:
                        predicted, _ = emotion.predict(crop, include_scores=True)
                    if predicted is not None:
                        label, _ = smoother.update(predicted, None)
                    else:
                        smoother.reset()
                        label = None
                last_predict = now
            detected_hands = hands.detect(frame, now) if hands is not None else []
            face_y = None
            if observation is not None:
                if label == 'happy':
                    draw_sun(frame, observation)
                draw_lebron_filter(frame, observation)
                _, y, _, h = observation.bbox
                face_y = (y + h / 2) / frame.shape[0]
                last_face_y, last_face_at = face_y, now
            elif now - last_face_at <= .3:
                face_y = last_face_y
            cue = reaction.update(label if observation is not None else None)
            events = gestures.update(detected_hands, face_y, now, hands.poses if hands is not None else None)
            if events:
                last_event = events[0]
                audio.play(last_event)
                if last_event == 'chalk':
                    particles.trigger(detected_hands, now)
                event_origin = tuple(np.rint(gestures.event_origin * frame.shape[1::-1]).astype(int))
                last_event_at = now
            elif cue is not None:
                audio.play(cue)
            particles.draw(frame, now)
            if now - last_event_at < 1.2 and last_event == 'shot':
                draw_ball(frame, event_origin, now - last_event_at)
            if label == 'sad':
                apply_sad_grayscale(frame)
            taco_rain.draw(frame, now, active=(label == 'surprise'))
            if label == 'surprise':
                draw_taco_tuesday(frame, observation, now)
            draw_predominant_emotion(frame, label, expression_names)
            if pending_record is not None:
                recorder = DemoRecorder(pending_record, (frame.shape[1], frame.shape[0]), now)
                pending_record = None
            if recorder is not None:
                if not recorder.write(frame, now):
                    print(f'Grabación guardada: {recorder.path}')
                    recorder = None
            count += 1
            if args.max_frames > 0 and count >= args.max_frames:
                break
            if args.headless:
                continue
            cv2.imshow(title, frame)
            key = cv2.waitKey(1) & 0xff
            if key in (27, ord('q')) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord('f'):
                fullscreen = not fullscreen
                cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            elif key == ord('m'):
                audio.toggle()
            elif key == ord('r'):
                if recorder is not None:
                    recorder.close()
                    print(f'Grabación guardada: {recorder.path}')
                    recorder = None
                else:
                    pending_record = ROOT / 'recordings' / (datetime.now().strftime('%Y%m%d_%H%M%S') + '.avi')
        print(f'Procesados {count} fotogramas.')
        return 0 if count else 2
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        print(f'Error controlado: {exc}', file=sys.stderr)
        return 2
    finally:
        for resource in (recorder, face, hands, audio, emotion):
            if resource is not None:
                resource.close()
        if cap is not None:
            cap.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(main())
