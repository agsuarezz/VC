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


def apply_sad_grayscale(frame: np.ndarray) -> np.ndarray:
    """Convierte el fotograma a blanco y negro conservando 3 canales BGR."""
    if frame is None or frame.size == 0:
        return frame
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame[:] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return frame


def _build_taco_sprite(size: int = 128) -> np.ndarray:
    """Crea un sprite BGRA de un taco dibujado con primitivas vectoriales."""
    sprite = np.zeros((size, size, 4), dtype=np.uint8)
    scale = size / 128.0

    def pt(x: float, y: float) -> tuple[int, int]:
        return (int(round(x * scale)), int(round(y * scale)))

    def ax(rx: float, ry: float) -> tuple[int, int]:
        return (max(1, int(round(rx * scale))), max(1, int(round(ry * scale))))

    outline = (24, 62, 108, 255)

    # Cara trasera de la tortilla asomando por arriba
    cv2.ellipse(sprite, pt(64, 86), ax(49, 46), 0, 180, 360, (44, 152, 208, 255), -1, cv2.LINE_AA)
    cv2.ellipse(sprite, pt(64, 86), ax(49, 46), 0, 180, 360, outline, max(1, int(round(2.5 * scale))), cv2.LINE_AA)

    # Lechuga rizada verde asomando por el borde superior
    lettuce_fill = (52, 198, 72, 255)
    lettuce_edge = (26, 128, 42, 255)
    lettuce_spots = (
        (24, 66, 11), (34, 54, 12), (46, 46, 12), (60, 42, 13),
        (74, 44, 12), (88, 50, 12), (99, 60, 11), (104, 70, 9),
    )
    for x, y, r in lettuce_spots:
        cv2.circle(sprite, pt(x, y), max(2, int(round(r * scale))), lettuce_fill, -1, cv2.LINE_AA)
        cv2.circle(sprite, pt(x, y), max(2, int(round(r * scale))), lettuce_edge, max(1, int(round(1.5 * scale))), cv2.LINE_AA)

    # Carne sazonada
    meat_fill = (34, 66, 122, 255)
    for x, y, r in ((32, 64, 9), (46, 56, 10), (62, 53, 10), (78, 56, 10), (92, 64, 9)):
        cv2.circle(sprite, pt(x, y), max(2, int(round(r * scale))), meat_fill, -1, cv2.LINE_AA)

    # Tomate rojo en dados
    tomato_fill = (42, 48, 232, 255)
    tomato_edge = (22, 24, 155, 255)
    for x, y, r in ((36, 56, 5.5), (56, 48, 6), (76, 50, 5.5), (92, 60, 5)):
        cv2.circle(sprite, pt(x, y), max(2, int(round(r * scale))), tomato_fill, -1, cv2.LINE_AA)
        cv2.circle(sprite, pt(x, y), max(2, int(round(r * scale))), tomato_edge, 1, cv2.LINE_AA)

    # Queso rallado amarillo brillante y crema
    cheese_fill = (30, 220, 255, 255)
    for p1, p2 in (((41, 51), (50, 57)), ((63, 45), (72, 52)), ((80, 51), (88, 58)), ((29, 62), (38, 66))):
        cv2.line(sprite, pt(*p1), pt(*p2), cheese_fill, max(2, int(round(3.5 * scale))), cv2.LINE_AA)
    for x, y in ((50, 51), (70, 48)):
        cv2.circle(sprite, pt(x, y), max(2, int(round(3.2 * scale))), (245, 250, 255, 255), -1, cv2.LINE_AA)

    # Tortilla frontal doblada (semicirculo dorado con base ligeramente redondeada)
    shell_fill = (62, 202, 248, 255)
    shell_shade = (44, 168, 232, 255)
    shell_pts = cv2.ellipse2Poly(pt(64, 92), ax(50, 42), 0, 180, 360, 6)
    bottom_pts = np.array([pt(114, 92), pt(106, 97), pt(64, 99), pt(22, 97), pt(14, 92)], dtype=np.int32)
    shell_poly = np.vstack((shell_pts, bottom_pts))
    cv2.fillPoly(sprite, [shell_poly], shell_fill, cv2.LINE_AA)

    # Sombra inferior suave de la tortilla
    shade_poly = np.array([pt(16, 89), pt(38, 86), pt(64, 85), pt(90, 86), pt(112, 89),
                           pt(106, 97), pt(64, 99), pt(22, 97)], dtype=np.int32)
    cv2.fillPoly(sprite, [shade_poly], shell_shade, cv2.LINE_AA)

    # Manchas tostadas caracteristicas de la tortilla
    toast = (42, 122, 178, 255)
    for x, y, rx, ry, ang in (
        (36, 78, 3.8, 2.4, -15), (52, 68, 3.2, 2.2, 10), (68, 76, 4.2, 2.6, 5),
        (85, 70, 3.5, 2.2, -20), (96, 82, 3.4, 2.2, 15), (46, 88, 3.5, 2.0, 0),
        (76, 89, 3.8, 2.1, -8), (28, 86, 2.8, 1.9, 12),
    ):
        cv2.ellipse(sprite, pt(x, y), ax(rx, ry), ang, 0, 360, toast, -1, cv2.LINE_AA)

    # Contorno cartoon de la tortilla frontal
    cv2.polylines(sprite, [shell_poly], True, outline, max(1, int(round(3 * scale))), cv2.LINE_AA)
    return sprite


class TacoRain:
    """Lluvia continua de tacos por toda la pantalla mientras dure la sorpresa."""

    def __init__(self, count: int = 30, seed: int | None = None):
        self.count = count
        self._rng = np.random.default_rng(seed)
        self._sprite = _build_taco_sprite(128)
        self._tacos: list[list[float]] = []
        self._last_time: float | None = None

    def reset(self):
        self._tacos.clear()
        self._last_time = None

    def _spawn(self, y: float | None = None) -> list[float]:
        size = float(self._rng.uniform(0.06, 0.11))
        speed_factor = 0.75 + (size - 0.06) / 0.05 * 0.55
        return [
            float(self._rng.uniform(-0.02, 1.02)),                     # 0: x norm
            float(self._rng.uniform(-0.35, 0.92) if y is None else y), # 1: y norm
            float(self._rng.uniform(0.32, 0.56) * speed_factor),       # 2: vy norm/s
            float(self._rng.uniform(-0.05, 0.05)),                     # 3: vx norm/s
            size,                                                      # 4: size / frame_w
            float(self._rng.uniform(-35.0, 35.0)),                     # 5: angle deg
            float(self._rng.uniform(-75.0, 75.0)),                     # 6: angular vel deg/s
            float(self._rng.uniform(0.0, 2 * np.pi)),                  # 7: sway phase
            float(self._rng.uniform(0.008, 0.022)),                    # 8: sway amp
        ]

    def _stamp(self, frame: np.ndarray, cx: int, cy: int, px_size: int, angle: float):
        h, w = frame.shape[:2]
        dst_size = max(16, int(round(px_size * 1.35)))
        half = dst_size // 2
        if cx + half <= 0 or cx - half >= w or cy + half <= 0 or cy - half >= h:
            return
        scale = px_size / float(self._sprite.shape[0])
        center = (self._sprite.shape[1] / 2.0, self._sprite.shape[0] / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        matrix[0, 2] += dst_size / 2.0 - center[0]
        matrix[1, 2] += dst_size / 2.0 - center[1]
        rotated = cv2.warpAffine(
            self._sprite, matrix, (dst_size, dst_size),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
        )
        x0, y0 = cx - half, cy - half
        x1, y1 = x0 + dst_size, y0 + dst_size
        sx0, sy0 = max(0, -x0), max(0, -y0)
        dx0, dy0 = max(0, x0), max(0, y0)
        dx1, dy1 = min(w, x1), min(h, y1)
        if dx1 <= dx0 or dy1 <= dy0:
            return
        patch = rotated[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
        alpha = patch[:, :, 3:4].astype(np.float32) * (1.0 / 255.0)
        roi = frame[dy0:dy1, dx0:dx1]
        roi[:] = np.clip(roi * (1.0 - alpha) + patch[:, :, :3] * alpha, 0, 255).astype(np.uint8)

    def draw(self, frame: np.ndarray, now: float, active: bool) -> np.ndarray:
        if not active:
            self.reset()
            return frame
        if not self._tacos:
            self._tacos = [self._spawn() for _ in range(self.count)]
            self._last_time = now
        dt = 0.0 if self._last_time is None else float(np.clip(now - self._last_time, 0.0, 0.1))
        self._last_time = now
        h, w = frame.shape[:2]
        for idx, taco in enumerate(self._tacos):
            taco[0] += taco[3] * dt
            taco[1] += taco[2] * dt
            taco[5] = (taco[5] + taco[6] * dt) % 360.0
            if taco[1] > 1.15:
                self._tacos[idx] = self._spawn(y=float(self._rng.uniform(-0.22, -0.06)))
                taco = self._tacos[idx]
            sway_x = taco[0] + np.sin(now * 2.6 + taco[7]) * taco[8]
            cx = int(round(sway_x * w))
            cy = int(round(taco[1] * h))
            px_size = max(24, int(round(taco[4] * w)))
            self._stamp(frame, cx, cy, px_size, taco[5])
        return frame


def draw_taco_tuesday(frame: np.ndarray, observation: FaceObservation | None, now: float = 0.0) -> np.ndarray:
    """Coloca el letrero 'TACO TUESDAY' encima de la cabeza del usuario."""
    h, w = frame.shape[:2]
    label = "TACO TUESDAY"
    font = cv2.FONT_HERSHEY_DUPLEX
    center_x = w * 0.5
    anchor_y = h * 0.16
    scale = max(0.9, min(1.5, w / 680.0))

    if observation is not None:
        bx, by, bw, _ = observation.bbox
        center_x = bx + bw * 0.5
        anchor_y = by - max(18.0, bw * 0.14)
        scale = max(0.8, min(1.65, bw / 170.0))
        if len(observation.landmarks) >= 468:
            points = observation.landmarks[:, :2] * frame.shape[1::-1]
            eye_axis = points[263] - points[33]
            eye_width = float(np.linalg.norm(eye_axis))
            if np.isfinite(points).all() and eye_width >= 5:
                across = eye_axis / eye_width
                down = np.asarray((-across[1], across[0]))
                if np.dot(down, points[152] - points[10]) < 0:
                    down = -down
                crown_top = points[10] - down * (eye_width * 0.64)
                center_x = float(crown_top[0])
                anchor_y = float(crown_top[1]) - max(16.0, eye_width * 0.24)
                scale = max(0.85, min(1.7, eye_width / 95.0))

    thickness = max(2, int(round(scale * 2.4)))
    outline_thickness = thickness + max(4, int(round(scale * 4.5)))
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    max_width = max(80, w - 28)
    if tw > max_width:
        scale *= max_width / float(tw)
        thickness = max(2, int(round(scale * 2.4)))
        outline_thickness = thickness + max(4, int(round(scale * 4.5)))
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)

    bounce = int(round(4.0 * np.sin(now * 6.5)))
    tx = int(np.clip(round(center_x - tw / 2.0), 14, max(14, w - tw - 14)))
    ty = int(np.clip(round(anchor_y + bounce), th + 16, max(th + 16, h - 14)))

    # Sombra, contorno oscuro y relleno dorado estilo Taco Tuesday
    cv2.putText(frame, label, (tx + 3, ty + 4), font, scale, (18, 22, 35), outline_thickness, cv2.LINE_AA)
    cv2.putText(frame, label, (tx, ty), font, scale, (24, 58, 128), outline_thickness, cv2.LINE_AA)
    cv2.putText(frame, label, (tx, ty), font, scale, (35, 215, 255), thickness, cv2.LINE_AA)
    return frame


def draw_angry_fire(frame: np.ndarray, observation: FaceObservation | None, now: float = 0.0) -> np.ndarray:
    """Dibuja llamas animadas detrás del rostro, fuego en la base y brasas ascendentes."""
    if frame is None or frame.size == 0:
        return frame
    h, w = frame.shape[:2]

    # Sutil tinte cálido rojizo para reforzar la atmósfera de enfado
    warm_tint = np.full_like(frame, (10, 24, 125))
    cv2.addWeighted(frame, 0.90, warm_tint, 0.10, 0, dst=frame)

    sw, sh = max(1, w // 4), max(1, h // 4)
    sx_f, sy_f = sw / float(w), sh / float(h)
    back_small = np.zeros((sh, sw, 3), dtype=np.uint8)
    front_small = np.zeros((sh, sw, 3), dtype=np.uint8)
    face_mask_small = None

    def _flame_contour(cx, cy, half_w, height, sway, bend):
        """Polígono orgánico en forma de lengua de fuego con base redondeada y curva S."""
        return np.array([
            (cx, cy - half_w * 0.42),
            (cx - half_w * 0.72, cy - half_w * 0.15),
            (cx - half_w, cy + height * 0.18),
            (cx - half_w * 0.78 + bend * 0.35, cy + height * 0.48),
            (cx - half_w * 0.34 + sway * 0.65, cy + height * 0.78),
            (cx + sway, cy + height),
            (cx + half_w * 0.34 + sway * 0.65, cy + height * 0.78),
            (cx + half_w * 0.78 + bend * 0.35, cy + height * 0.48),
            (cx + half_w, cy + height * 0.18),
            (cx + half_w * 0.72, cy - half_w * 0.15),
        ], dtype=np.float32)

    # 1) Llamas principales envolviendo la cabeza (detrás de la silueta facial)
    if observation is not None and len(observation.landmarks) >= 468:
        points = observation.landmarks[:, :2] * frame.shape[1::-1]
        eye_axis = points[263] - points[33]
        eye_width = float(np.linalg.norm(eye_axis))
        if np.isfinite(points).all() and eye_width >= 5:
            across = eye_axis / eye_width
            down = np.asarray((-across[1], across[0]), dtype=np.float32)
            if np.dot(down, points[152] - points[10]) < 0:
                down = -down
            up = -down
            head_center = 0.5 * (points[10] + points[152])
            scale = max(eye_width, 0.55 * float(np.linalg.norm(points[152] - points[10])))

            def local_small(xy):
                world = head_center + scale * (xy[..., :1] * across + xy[..., 1:] * up)
                return np.rint(world * (sx_f, sy_f)).astype(np.int32)

            pulse = 1.0 + 0.07 * float(np.sin(now * 13.0))
            flames = []
            for k in range(13):
                frac = (k / 12.0) * 2.0 - 1.0
                flicker = 0.18 * float(np.sin(now * 11.5 + k * 1.9)) + 0.09 * float(np.cos(now * 16.5 + k * 2.7))
                sway = 0.18 * float(np.sin(now * 8.5 + k * 1.4))
                bend = 0.12 * float(np.cos(now * 10.0 + k * 2.1))
                bx = frac * 0.88
                # Base curvada siguiendo la mandíbula/mejillas para evitar cortes planos
                by = -0.35 - 0.30 * (1.0 - frac * frac)
                height = (1.58 - 0.30 * abs(frac) + flicker) * pulse
                half_w = 0.28 + 0.08 * (1.0 - abs(frac))
                flames.append((bx, by, half_w, height, sway, bend))

            # Dibujar por capas (rojo -> naranja -> amarillo) para una mezcla continua
            for color, w_mul, h_mul, s_mul in (
                ((8, 36, 235), 1.00, 1.00, 1.00),
                ((16, 128, 255), 0.72, 0.78, 0.76),
                ((110, 232, 255), 0.42, 0.52, 0.48),
            ):
                polys = [
                    local_small(_flame_contour(bx, by, hw * w_mul, fh * h_mul, sw_v * s_mul, bd * s_mul))
                    for bx, by, hw, fh, sw_v, bd in flames
                ]
                cv2.fillPoly(back_small, polys, color, cv2.LINE_AA)

            # Brasas ascendentes alrededor del rostro
            for i in range(22):
                phase = (now * 1.05 + i * 0.17) % 1.0
                sx_e = (float(np.sin(i * 2.9 + now * 1.8)) * 1.32) * scale
                sy_e = (-0.75 + phase * 2.55) * scale
                pos = (head_center + sx_e * across + sy_e * up) * (sx_f, sy_f)
                rad = max(1, int(round(scale * sx_f * 0.045 * (1.0 - phase * 0.6))))
                color = (105, 235, 255) if i % 3 == 0 else (25, 135, 255)
                cv2.circle(front_small, tuple(np.rint(pos).astype(int)), rad, color, -1, cv2.LINE_AA)

            mask = np.zeros((sh, sw), dtype=np.uint8)
            hull = cv2.convexHull(np.rint(points[:468] * (sx_f, sy_f)).astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 255, cv2.LINE_AA)
            face_mask_small = cv2.GaussianBlur(mask, (0, 0), 2.5)

    # 2) Lenguas de fuego suaves en el borde inferior del fotograma
    base_count = 15
    step_x = sw / float(base_count)
    bottom_specs = []
    for idx in range(base_count + 1):
        cx = idx * step_x
        flicker = 0.24 * float(np.sin(now * 10.5 + idx * 1.6)) + 0.14 * float(np.cos(now * 15.0 + idx * 2.3))
        flame_h = sh * (0.14 + 0.045 * ((idx % 3) * 0.5)) * (1.0 + flicker)
        flame_w = step_x * 0.92
        sway = step_x * 0.26 * float(np.sin(now * 9.0 + idx * 1.3))
        bend = step_x * 0.16 * float(np.cos(now * 11.0 + idx * 1.9))
        bottom_specs.append((cx, flame_w, flame_h, sway, bend))

    for color, w_mul, h_mul, s_mul in (
        ((8, 40, 230), 1.00, 1.00, 1.00),
        ((18, 132, 255), 0.70, 0.74, 0.72),
        ((120, 238, 255), 0.38, 0.46, 0.45),
    ):
        polys = []
        for cx, fw, fh, sw_v, bd in bottom_specs:
            hw = fw * w_mul
            hh = fh * h_mul
            sw_s, bd_s = sw_v * s_mul, bd * s_mul
            polys.append(np.rint([
                (cx - hw, sh + 3),
                (cx - hw * 0.80 + bd_s * 0.35, sh - hh * 0.38),
                (cx - hw * 0.38 + sw_s * 0.65, sh - hh * 0.74),
                (cx + sw_s, sh - hh),
                (cx + hw * 0.38 + sw_s * 0.65, sh - hh * 0.74),
                (cx + hw * 0.80 + bd_s * 0.35, sh - hh * 0.38),
                (cx + hw, sh + 3),
            ]).astype(np.int32))
        cv2.fillPoly(front_small, polys, color, cv2.LINE_AA)

    def _soften(layer_small: np.ndarray, sigma_inner: float, sigma_bloom: float) -> np.ndarray:
        soft = cv2.GaussianBlur(layer_small, (0, 0), sigma_inner)
        bloom = cv2.GaussianBlur(layer_small, (0, 0), sigma_bloom)
        return cv2.addWeighted(soft, 0.65, bloom, 0.85, 0)

    combined_small = _soften(front_small, 1.4, 4.2)
    if face_mask_small is not None:
        back_glow_small = _soften(back_small, 1.8, 5.0)
        keep = (1.0 - face_mask_small.astype(np.float32) / 255.0)[..., None]
        back_glow_small = np.clip(back_glow_small * keep, 0, 255).astype(np.uint8)
        combined_small = cv2.add(combined_small, back_glow_small)

    glow_full = cv2.resize(combined_small, (w, h), interpolation=cv2.INTER_LINEAR)
    cv2.addWeighted(frame, 1.0, glow_full, 0.92, 0, dst=frame)
    return frame
