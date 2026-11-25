"""Landmarks y accesorios dibujados sobre la cámara."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import monotonic

import cv2
import numpy as np


@dataclass
class FaceObservation:
    bbox: tuple[int, int, int, int]
    landmarks: np.ndarray
    mouth_open: bool = False
    blendshapes: np.ndarray | None = None
    frame_size: tuple[int, int] | None = None
    blendshape_map: dict[str, float] | None = None


class FaceTracker:
    """Una cara por fotograma; sin hilos ni resultados atrasados."""

    def __init__(self, model_path: str | Path, video=False):
        self.video = video
        self.timestamp = -1
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Falta el modelo facial: {model_path}. Ejecuta scripts/download_models.py")
        import mediapipe as mp
        self._mp = mp
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=(mp.tasks.vision.RunningMode.VIDEO if video else mp.tasks.vision.RunningMode.IMAGE),
            num_faces=1,
            output_face_blendshapes=True,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )
        self._detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray) -> FaceObservation | None:
        if frame_bgr is None or frame_bgr.size == 0:
            return None
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if self.video:
            self.timestamp = max(self.timestamp + 1, int(monotonic() * 1000))
            result = self._detector.detect_for_video(image, self.timestamp)
        else:
            result = self._detector.detect(image)
        if not result.face_landmarks:
            return None
        lm = np.asarray([(p.x, p.y, p.z) for p in result.face_landmarks[0]], dtype=np.float32)
        h, w = frame_bgr.shape[:2]
        points = lm[:, :2] * (w, h)
        lo, hi = points.min(axis=0), points.max(axis=0)
        padding = (hi - lo) * 0.08
        x0, y0 = np.maximum(0, np.floor(lo - padding)).astype(int)
        x1, y1 = np.minimum((w, h), np.ceil(hi + padding)).astype(int)
        if x1 <= x0 or y1 <= y0:
            return None
        lip_gap = np.linalg.norm(points[13] - points[14])
        mouth_width = np.linalg.norm(points[78] - points[308])
        scores = None
        score_map = None
        if result.face_blendshapes:
            ordered = sorted(result.face_blendshapes[0], key=lambda p: p.category_name)
            scores = np.asarray([p.score for p in ordered], dtype=np.float32)
            score_map = {p.category_name: float(p.score) for p in ordered}
        return FaceObservation((int(x0), int(y0), int(x1-x0), int(y1-y0)), lm,
                               bool(lip_gap / max(mouth_width, 1.0) > 0.30), scores, (w, h), score_map)

    def close(self):
        if self._detector is not None:
            self._detector.close()
            self._detector = None


JAW = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152,
       377, 400, 378, 379, 365, 397, 288, 361, 323, 454]
LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
        375, 321, 405, 314, 17, 84, 181, 91, 146]


def _draw_beard(frame, points, across, down, eye_width):
    """Barba cerrada, con patillas y bigote, dejando labios y boca visibles."""
    jaw = points[JAW].copy()
    jaw += down * eye_width * np.sin(np.linspace(0, np.pi, len(jaw)))[:, None] * .085
    lip_center = points[LIPS].mean(axis=0)
    top = np.stack((
        .59 * points[454] + .41 * points[291] - down * eye_width * .035,
        points[291] - down * eye_width * .085 + across * eye_width * .05,
        .39 * points[2] + .61 * points[0],
        points[61] - down * eye_width * .085 - across * eye_width * .05,
        .59 * points[234] + .41 * points[61] - down * eye_width * .035,
    ))
    outline = np.rint(np.concatenate((jaw, top))).astype(np.int32)
    x, y, width, height = cv2.boundingRect(outline)
    x0, y0 = max(0, x-3), max(0, y-3)
    x1, y1 = min(frame.shape[1], x+width+3), min(frame.shape[0], y+height+3)
    if x1 <= x0 or y1 <= y0:
        return
    offset = np.asarray((x0, y0))
    mask = np.zeros((y1-y0, x1-x0), np.uint8)
    lips = np.rint(lip_center + 1.17 * (points[LIPS] - lip_center) - offset).astype(np.int32)
    cv2.fillPoly(mask, [outline-offset.astype(np.int32)], 255, cv2.LINE_AA)
    cv2.fillPoly(mask, [lips], 0, cv2.LINE_AA)
    alpha = cv2.GaussianBlur(mask, (0, 0), max(.65, eye_width * .012)).astype(np.float32) / 255
    # Fine directional texture follows the orientation of the jaw.
    yy, xx = np.ogrid[y0:y1, x0:x1]
    u = ((xx-lip_center[0])*across[0] + (yy-lip_center[1])*across[1]) / eye_width
    v = ((xx-lip_center[0])*down[0] + (yy-lip_center[1])*down[1]) / eye_width
    texture = 5*np.sin(186*u + 24*v) + 3*np.sin(265*u - 112*v)
    shade = np.float32((23, 28, 37)) + texture[..., None]
    region = frame[y0:y1, x0:x1]
    alpha = (alpha * .92)[..., None]
    region[:] = np.clip(region * (1-alpha) + shade * alpha, 0, 255).astype(np.uint8)


def draw_lebron_filter(frame, observation: FaceObservation | None):
    """Ajusta la corona a la frente y la barba a los puntos de mandíbula y boca."""
    if observation is None or len(observation.landmarks) < 468:
        return frame
    points = observation.landmarks[:, :2] * frame.shape[1::-1]
    eye_axis = points[263] - points[33]
    eye_width = float(np.linalg.norm(eye_axis))
    if not np.isfinite(points).all() or eye_width < 5:
        return frame
    across = eye_axis / eye_width
    down = np.asarray((-across[1], across[0]))
    if np.dot(down, points[152] - points[10]) < 0:
        down = -down
    vertical = points[152] - points[10]
    height = max(float(np.linalg.norm(vertical)), 1.)
    beard_down = vertical / height
    _draw_beard(frame, points, across, beard_down, height * .55)

    origin = points[10] + down * eye_width * .07
    size = eye_width * 1.24

    def place(coords):
        local = np.asarray(coords)
        return np.rint(origin + size * (local[:, :1]*across + local[:, 1:]*down)).astype(np.int32)

    crown = place([(-.52, .08), (-.57, -.42), (-.29, -.20), (0, -.57),
                   (.29, -.20), (.57, -.42), (.52, .08)])
    cv2.fillPoly(frame, [crown], (160, 107, 0), cv2.LINE_AA)
    cv2.fillPoly(frame, [place([(-.53, .015), (.53, .015), (.52, .08), (-.52, .08)])],
                 (66, 23, 237), cv2.LINE_AA)
    cv2.polylines(frame, [crown], True, (22, 55, 105), max(1, round(size*.018)), cv2.LINE_AA)
    jewel = place([(0, -.29), (.093, -.16), (0, -.03), (-.093, -.16)])
    cv2.fillConvexPoly(frame, jewel, (250, 250, 250), cv2.LINE_AA)
    cv2.polylines(frame, [jewel], True, (185, 235, 255), max(1, round(size*.012)), cv2.LINE_AA)
    return frame


def draw_sun(frame, observation):
    x, y, w, h = observation.bbox
    center = np.array((x + w // 2, y + h // 2))
    radius = max(8, int(max(w, h) * .55))
    gold = (30, 210, 255)
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.circle(mask, tuple(center), radius, 255, -1)
    for angle in np.linspace(0, 2*np.pi, 20, endpoint=False):
        direction = np.array((np.cos(angle), np.sin(angle)))
        start = center + np.rint(direction * (radius + 4)).astype(int)
        end = center + np.rint(direction * radius * 1.19).astype(int)
        cv2.line(mask, tuple(start), tuple(end), 255, max(3, radius // 15))
    points = np.rint(observation.landmarks[:468, :2] * frame.shape[1::-1]).astype(np.int32)
    cv2.fillConvexPoly(mask, cv2.convexHull(points), 0)
    frame[mask != 0] = gold


def draw_taco(frame, hand):
    points = hand[:, :2] * frame.shape[1::-1]
    left, top = points.min(axis=0)
    right, bottom = points.max(axis=0)
    width, height = right - left, bottom - top
    radius_x = max(8, round(max(width * .60, height * .50)))
    radius_y = max(5, round(max(height * .92, width * .36)))
    center_x = round((left + right) / 2)
    base_y = round(bottom - height * .07)
    for lift, color in ((.17, (58, 95, 133)), (.10, (48, 173, 68)), (0, (60, 182, 245))):
        origin = (center_x, base_y - round(radius_y * lift))
        cv2.ellipse(frame, origin, (radius_x, radius_y), 0, 180, 360, color, -1, cv2.LINE_AA)
