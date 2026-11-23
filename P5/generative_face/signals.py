"""Extracción de contornos y señales faciales continuas para arte generativo."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from lebron.visuals import FaceObservation

# Subconjuntos de índices de MediaPipe FaceMesh (468 puntos base).
FACE_OVAL = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
)
LEFT_EYE = (33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7)
RIGHT_EYE = (263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249)
LEFT_EYEBROW = (70, 63, 105, 66, 107)
RIGHT_EYEBROW = (300, 293, 334, 296, 336)
NOSE_BRIDGE = (168, 6, 197, 195, 5, 4, 1)
NOSE_BASE = (98, 97, 2, 326, 327)
OUTER_LIPS = (
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    375, 321, 405, 314, 17, 84, 181, 91, 146,
)
INNER_LIPS = (
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
    324, 318, 402, 317, 14, 87, 178, 88, 95,
)
INNER_CHEEK_RING = (
    151, 337, 299, 333, 298, 301, 368, 264, 447, 366, 401, 435,
    367, 364, 394, 395, 369, 396, 175, 171, 140, 170, 169, 135,
    138, 215, 177, 137, 227, 34, 139, 71, 68, 104, 69, 108,
)

# Pares (indices, closed) usados para dibujar la silueta y las estelas.
CONTOUR_SPECS = (
    ("oval", FACE_OVAL, True),
    ("left_eye", LEFT_EYE, True),
    ("right_eye", RIGHT_EYE, True),
    ("left_brow", LEFT_EYEBROW, False),
    ("right_brow", RIGHT_EYEBROW, False),
    ("nose_bridge", NOSE_BRIDGE, False),
    ("nose_base", NOSE_BASE, False),
    ("outer_lips", OUTER_LIPS, True),
    ("inner_lips", INNER_LIPS, True),
)

# Costillas radiales para el modo de topografía 3D.
Z_RADIAL_RIBS = (
    (1, 168, 151, 10),
    (1, 195, 6, 107, 109),
    (1, 195, 6, 336, 338),
    (1, 36, 205, 123, 234),
    (1, 266, 425, 352, 454),
    (1, 98, 206, 212, 172),
    (1, 327, 426, 432, 397),
    (1, 2, 0, 17, 152),
)


@dataclass
class FaceSignals:
    points: np.ndarray
    z_values: np.ndarray
    center: np.ndarray
    eye_width: float
    mouth_center: np.ndarray
    left_eye_center: np.ndarray
    right_eye_center: np.ndarray
    mouth_ratio: float
    smile: float
    brow_lift: float
    blink_left: float
    blink_right: float
    head_tilt: float
    head_speed: float
    wink_triggered: bool = False


def _eye_aspect_ratio(points: np.ndarray, outer: int, inner: int, top: int, bottom: int) -> float:
    width = float(np.linalg.norm(points[outer] - points[inner]))
    height = float(np.linalg.norm(points[top] - points[bottom]))
    return height / max(width, 1.0)


class SignalExtractor:
    """Convierte observaciones faciales en señales suavizadas y detecta guiños."""

    def __init__(
        self,
        wink_close: float = 0.60,
        wink_open: float = 0.34,
        wink_hold: float = 0.14,
        wink_cooldown: float = 0.90,
        smoothing: float = 0.45,
    ):
        self.wink_close = wink_close
        self.wink_open = wink_open
        self.wink_hold = wink_hold
        self.wink_cooldown = wink_cooldown
        self.smoothing = smoothing
        self.reset()

    def reset(self):
        self._prev_center: np.ndarray | None = None
        self._prev_time: float | None = None
        self._wink_side: str | None = None
        self._wink_since: float | None = None
        self._wink_latched: bool = False
        self._last_wink_at: float = -math.inf
        self._smooth_mouth: float = 0.0
        self._smooth_smile: float = 0.0
        self._smooth_brow: float = 0.0
        self._smooth_speed: float = 0.0

    def extract(
        self,
        observation: FaceObservation | None,
        frame_shape: tuple[int, ...],
        now: float,
    ) -> FaceSignals | None:
        if observation is None or len(observation.landmarks) < 468:
            self._prev_center = None
            self._prev_time = None
            self._wink_side = None
            self._wink_since = None
            self._wink_latched = False
            return None

        h, w = int(frame_shape[0]), int(frame_shape[1])
        lm = np.asarray(observation.landmarks, dtype=np.float32)
        if not np.isfinite(lm).all():
            return None

        points = lm[:, :2] * np.asarray((w, h), dtype=np.float32)
        eye_axis = points[263] - points[33]
        eye_width = float(np.linalg.norm(eye_axis))
        if eye_width < 5.0:
            return None

        head_tilt = float(np.clip(math.atan2(float(eye_axis[1]), float(eye_axis[0])), -0.75, 0.75))
        center = points[1].copy()
        mouth_center = points[[13, 14]].mean(axis=0)
        left_eye_center = points[list(LEFT_EYE)].mean(axis=0)
        right_eye_center = points[list(RIGHT_EYE)].mean(axis=0)

        lip_gap = float(np.linalg.norm(points[13] - points[14]))
        mouth_width = float(np.linalg.norm(points[78] - points[308]))
        geom_mouth = float(np.clip((lip_gap / max(mouth_width, 1.0) - 0.07) / 0.50, 0.0, 1.0))

        bmap = observation.blendshape_map or {}
        if bmap:
            raw_mouth = max(geom_mouth, float(bmap.get("jawOpen", 0.0)))
            raw_smile = 0.5 * (
                float(bmap.get("mouthSmileLeft", 0.0)) + float(bmap.get("mouthSmileRight", 0.0))
            )
            raw_brow = max(
                float(bmap.get("browInnerUp", 0.0)),
                0.5 * (
                    float(bmap.get("browOuterUpLeft", 0.0)) + float(bmap.get("browOuterUpRight", 0.0))
                ),
            )
            blink_left = float(bmap.get("eyeBlinkLeft", 0.0))
            blink_right = float(bmap.get("eyeBlinkRight", 0.0))
        else:
            raw_mouth = geom_mouth
            raw_smile = float(np.clip((mouth_width / eye_width - 0.42) / 0.25, 0.0, 1.0))
            brow_gap = 0.5 * (
                float(np.linalg.norm(points[105] - left_eye_center))
                + float(np.linalg.norm(points[334] - right_eye_center))
            )
            raw_brow = float(np.clip((brow_gap / eye_width - 0.20) / 0.14, 0.0, 1.0))
            ear_l = _eye_aspect_ratio(points, 33, 133, 159, 145)
            ear_r = _eye_aspect_ratio(points, 263, 362, 386, 374)
            blink_left = float(np.clip((0.27 - ear_l) / 0.14, 0.0, 1.0))
            blink_right = float(np.clip((0.27 - ear_r) / 0.14, 0.0, 1.0))

        dt = None if self._prev_time is None else (now - self._prev_time)
        alpha = 1.0 if (dt is None or dt < 0 or dt > 0.5) else self.smoothing
        self._smooth_mouth = (1.0 - alpha) * self._smooth_mouth + alpha * float(np.clip(raw_mouth, 0.0, 1.0))
        self._smooth_smile = (1.0 - alpha) * self._smooth_smile + alpha * float(np.clip(raw_smile, 0.0, 1.0))
        self._smooth_brow = (1.0 - alpha) * self._smooth_brow + alpha * float(np.clip(raw_brow, 0.0, 1.0))

        raw_speed = 0.0
        if self._prev_center is not None and dt is not None and 1e-4 < dt < 0.5:
            dist = float(np.linalg.norm(center - self._prev_center)) / eye_width
            raw_speed = float(np.clip((dist / dt) / 2.2, 0.0, 1.0))
        self._smooth_speed = 0.65 * self._smooth_speed + 0.35 * raw_speed
        self._prev_center = center.copy()
        self._prev_time = now

        wink_triggered = self._update_wink(blink_left, blink_right, now)

        return FaceSignals(
            points=points,
            z_values=lm[:, 2].copy(),
            center=center,
            eye_width=eye_width,
            mouth_center=mouth_center,
            left_eye_center=left_eye_center,
            right_eye_center=right_eye_center,
            mouth_ratio=self._smooth_mouth,
            smile=self._smooth_smile,
            brow_lift=self._smooth_brow,
            blink_left=blink_left,
            blink_right=blink_right,
            head_tilt=head_tilt,
            head_speed=self._smooth_speed,
            wink_triggered=wink_triggered,
        )

    def _update_wink(self, blink_left: float, blink_right: float, now: float) -> bool:
        both_open = blink_left < self.wink_open and blink_right < self.wink_open
        if both_open:
            self._wink_latched = False

        left_wink = blink_left >= self.wink_close and blink_right <= self.wink_open
        right_wink = blink_right >= self.wink_close and blink_left <= self.wink_open
        side = "left" if left_wink else ("right" if right_wink else None)

        if side is None:
            self._wink_side = None
            self._wink_since = None
            return False

        if side != self._wink_side:
            self._wink_side = side
            self._wink_since = now
            return False

        if (
            not self._wink_latched
            and self._wink_since is not None
            and (now - self._wink_since) >= self.wink_hold
            and (now - self._last_wink_at) >= self.wink_cooldown
        ):
            self._wink_latched = True
            self._last_wink_at = now
            return True
        return False
