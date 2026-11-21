"""MediaPipe hands, temporal gesture rules and a bounded chalk animation."""
from pathlib import Path
from dataclasses import dataclass
import math
import numpy as np


class HandTracker:
    def __init__(self, model_path):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Falta el modelo de manos: {path}")
        self._mp = mp
        self.recognizes_poses = path.name == 'gesture_recognizer.task'
        self.poses = None
        options_type = vision.GestureRecognizerOptions if self.recognizes_poses else vision.HandLandmarkerOptions
        detector_type = vision.GestureRecognizer if self.recognizes_poses else vision.HandLandmarker
        options = options_type(
            base_options=python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.45,
            min_hand_presence_confidence=0.45, min_tracking_confidence=0.45)
        self._detector = detector_type.create_from_options(options)
        self._last_frame_ms = -1

    def detect(self, frame_bgr, now):
        import cv2
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        self._last_frame_ms = max(self._last_frame_ms + 1, int(now * 1000))
        if self.recognizes_poses:
            result = self._detector.recognize_for_video(image, self._last_frame_ms)
            self.poses = [next(((p.category_name, p.score) for p in options), (None, 0.0))
                          for options in result.gestures]
        else:
            result = self._detector.detect_for_video(image, self._last_frame_ms)
        return [np.array([(p.x, p.y, p.z) for p in hand], dtype=np.float32)
                for hand in result.hand_landmarks]

    def close(self):
        self._detector.close()


def extended_fingers(hand):
    extended = []
    for mcp, pip, tip in ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)):
        a, b = hand[mcp, :2] - hand[pip, :2], hand[tip, :2] - hand[pip, :2]
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        straight = norm > 1e-8 and np.dot(a, b) / norm < -0.65
        away = np.linalg.norm(hand[tip, :2] - hand[0, :2]) > 1.15 * np.linalg.norm(hand[pip, :2] - hand[0, :2])
        extended.append(bool(straight and away))
    return tuple(extended)


def open_palm(hand):
    return sum(extended_fingers(hand)) >= 3


def fist_pose(hand):
    wrist = hand[0, :2]
    return all(np.linalg.norm(hand[tip, :2] - wrist) < 1.12 * np.linalg.norm(hand[tip-2, :2] - wrist)
               for tip in (8, 12, 16, 20))


@dataclass(eq=False)
class HandMotion:
    center: np.ndarray
    seen: float
    raised_from: tuple | None = None
    shot_since: float | None = None
    pose: str | None = None
    pose_since: float = 0.0
    pose_seen: float = -math.inf
    latched: str | None = None


class GestureEngine:
    """Two-hand rise for chalk and distinct one-hand gestures."""
    def __init__(self, cooldown=1.15, window=1.8, rise=0.075,
                 chalk_hold=0.12, dropout=0.30, posture_effects=True):
        self.cooldown, self.window, self.rise = cooldown, window, rise
        self.chalk_hold, self.dropout = chalk_hold, dropout
        self.posture_effects = posture_effects
        self.reset()

    def reset(self):
        self._tracks = []
        self._last_event = -math.inf
        self._previous_time = None
        self._last_two_seen = -math.inf
        self._chalk_ready = None
        self._chalk_seconds = 0.0
        self._previous_chalk_pose = False
        self.event_origin = np.asarray((.5, .5))
        self.pose_names = []
        self.hand_count = self.open_count = 0
        self.chalk_progress = 0.0
        self.status = "Talco: empieza con ambas manos bajas"

    def _chalk(self, centers, palms, face_y, now, dt):
        ys = centers[:, 1]
        if np.all(ys > face_y + .04):
            self._chalk_ready = (now, centers.copy(), face_y)
            self._chalk_seconds = 0.0
            self._previous_chalk_pose = False
        if self._chalk_ready is None:
            self.status = "Talco: baja las dos manos para preparar"
            return []
        started, start, start_face = self._chalk_ready
        if now - started > 2.2 or np.max(np.abs(centers[:, 0] - start[:, 0])) > .32:
            self._chalk_ready = None
            self._chalk_seconds = 0.0
            self._previous_chalk_pose = False
            self.status = "Talco: vuelve a empezar con las manos bajas"
            return []
        rise = np.minimum(start[:,1] - ys, (start[:,1]-start_face) - (ys-face_y))
        high = all(palms) and np.all(rise >= .09) and np.all(ys <= face_y + .035)
        if high and self._previous_chalk_pose:
            self._chalk_seconds += dt
        elif not high:
            self._chalk_seconds = 0.0
        self._previous_chalk_pose = high
        self.chalk_progress = .8 * float(np.clip(np.min(rise) / .09, 0, 1))
        if high:
            self.chalk_progress += .2 * min(self._chalk_seconds / self.chalk_hold, 1.)
        self.status = f"Talco preparado: sube y abre ambas ({self.chalk_progress:.0%})"
        if high and self._chalk_seconds + 1e-9 >= self.chalk_hold:
            self._chalk_ready = None
            self._last_event = now
            self.event_origin = centers.mean(axis=0)
            self.status = "TALCO! Baja las dos manos para repetir"
            return ["chalk"]
        return []

    def _assign_tracks(self, centers, now):
        self._tracks = [track for track in self._tracks if now - track.seen <= .28]
        result = [None] * len(centers)
        matches = sorted((abs(center[0]-track.center[0]) + .25 * abs(center[1]-track.center[1]), i, j)
                         for i, center in enumerate(centers) for j, track in enumerate(self._tracks))
        used = set()
        for distance, i, j in matches:
            if distance <= .24 and result[i] is None and j not in used:
                result[i] = self._tracks[j]
                used.add(j)
        for i, center in enumerate(centers):
            if result[i] is None:
                result[i] = HandMotion(center, now)
                self._tracks.append(result[i])
            result[i].center, result[i].seen = center, now
        return result

    def update(self, hands, face_y, now, poses=None):
        rows = [(np.asarray(hand), poses[i] if poses is not None and i < len(poses) else (None, 0.0))
                for i, hand in enumerate(hands) if np.asarray(hand).shape == (21, 3) and np.isfinite(hand).all()]
        rows = sorted(rows, key=lambda row: float(row[0][0, 0]))[:2]
        centers, palms, actions = [], [], []
        self.pose_names = []
        for hand, (name, score) in rows:
            fingers = extended_fingers(hand)
            palms.append((name == 'Open_Palm' and score >= .45) or sum(fingers) >= 3)
            action = {'Victory': 'victory', 'Closed_Fist': 'fist'}.get(name) if score >= .5 else None
            if poses is None:
                if fingers == (True, True, False, False):
                    action = 'victory'
                elif not any(fingers) and fist_pose(hand):
                    action = 'fist'
            actions.append(action)
            centers.append(hand[[0, 5, 9, 13, 17], :2].mean(axis=0))
            self.pose_names.append({'victory': 'V', 'fist': 'puno'}.get(action, 'palma' if palms[-1] else '-'))
        self.hand_count, self.open_count = len(rows), sum(palms)
        self.chalk_progress = 0.0
        dt = 0.0 if self._previous_time is None else now - self._previous_time
        self._previous_time = now
        since_two = now - self._last_two_seen
        if since_two > self.dropout or dt < 0 or dt > .4 or face_y is None or not np.isfinite(face_y):
            self._chalk_ready = None
            self._chalk_seconds = 0.0
            self._previous_chalk_pose = False
        if dt < 0 or dt > .4 or face_y is None or not np.isfinite(face_y):
            self._tracks.clear()
        if len(rows) == 2:
            self._last_two_seen = now
        else:
            self._previous_chalk_pose = False
        if face_y is None or not np.isfinite(face_y):
            self.status = 'Muestra la cara y las manos'
            return []
        tracks = self._assign_tracks(centers, now)
        if not tracks:
            for track in self._tracks:
                track.raised_from = track.shot_since = None
        self.status = 'Sube una mano para tirar' if rows else 'Muestra una mano a la camara'
        cooling = now - self._last_event < self.cooldown
        if len(rows) == 2 and not cooling:
            event = self._chalk(np.asarray(centers), palms, face_y, now, dt)
            if event:
                for track in tracks:
                    track.raised_from = track.shot_since = None
                return event
        both_rising = self._chalk_ready is not None and (
            len(rows) < 2 or np.all(self._chalk_ready[1][:,1] - np.asarray(centers)[:,1] >= .045))
        for track, palm, action in zip(tracks, palms, actions):
            hand_y = track.center[1]
            action = action if hand_y <= face_y + .24 else None
            if action is not None:
                if action != track.pose or now - track.pose_seen > .20:
                    track.pose, track.pose_since = action, now
                track.pose_seen = now
            elif now - track.pose_seen > .20:
                track.pose = track.latched = None
            if hand_y >= face_y + .025:
                if track.raised_from is None or hand_y > track.raised_from[1] - .025:
                    track.raised_from = (now, hand_y)
            shot = False
            if track.raised_from is not None:
                started, low = track.raised_from
                if now - started > self.window:
                    track.raised_from = None
                else:
                    shot = hand_y <= face_y + .075 and low-hand_y >= self.rise and (palm or action is None)
            if shot and track.shot_since is None:
                track.shot_since = now
            elif not shot:
                track.shot_since = None
            if cooling:
                continue
            if self.posture_effects and action and action != track.latched:
                self.status = 'Manten la V' if action == 'victory' else 'Manten el puno'
                if now - track.pose_since >= .23:
                    track.latched = action
                    return self._finish(action, track, now)
            if shot and not both_rising and (len(rows) == 1 or now - track.shot_since >= .12):
                return self._finish('shot', track, now)
        if cooling:
            self.status = 'Espera un momento'
        return []

    def _finish(self, event, track, now):
        self._last_event = now
        self.event_origin = track.center.copy()
        for tracked in self._tracks:
            tracked.raised_from = tracked.shot_since = None
        return [event]


class ParticleSystem:
    def __init__(self, max_particles=320, seed=None):
        self.max_particles = max_particles
        self._rng = np.random.default_rng(seed)
        self._particles = []

    def trigger(self, hands, now):
        for hand in hands[:2]:
            x, y = np.asarray(hand)[[0, 5, 9, 13, 17], :2].mean(axis=0)
            for _ in range(150):
                self._particles.append((float(now), float(x), float(y),
                    self._rng.uniform(-0.28, 0.28), self._rng.uniform(-0.28, -0.05),
                    self._rng.uniform(1.3, 2.4), self._rng.uniform(0.010, 0.025)))
        self._particles = self._particles[-self.max_particles:]

    def draw(self, frame, now):
        import cv2
        height, width = frame.shape[:2]
        self._particles = [p for p in self._particles if 0 <= now - p[0] < p[5]]
        if not self._particles:
            return frame
        opacity = np.zeros((height, width), dtype=np.uint8)
        for born, x, y, vx, vy, life, radius in self._particles:
            age = now - born
            px, py = int((x + vx * age) * width), int((y + vy * age + 0.07 * age ** 2) * height)
            r = max(1, int(radius * width * (1 + age)))
            alpha = int(185 * (1 - age / life))
            if -r < px < width + r and -r < py < height + r:
                cv2.circle(opacity, (px, py), r, alpha, -1, cv2.LINE_AA)
        # White particles fade to transparency, never toward black.
        blend = opacity[:, :, None].astype(np.float32) / 255.0
        frame[:] = np.clip(frame * (1 - blend) + 255 * blend, 0, 255).astype(np.uint8)
        return frame
