"""Renderizadores de arte generativo facial (Neón, Tipografía ASCII y Topografía 3D)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import cv2
import numpy as np

from generative_face.signals import (
    CONTOUR_SPECS,
    FACE_OVAL,
    INNER_CHEEK_RING,
    INNER_LIPS,
    LEFT_EYE,
    LEFT_EYEBROW,
    NOSE_BASE,
    NOSE_BRIDGE,
    OUTER_LIPS,
    RIGHT_EYE,
    RIGHT_EYEBROW,
    Z_RADIAL_RIBS,
    FaceSignals,
    SignalExtractor,
)
from lebron.visuals import FaceObservation

MODE_NAMES = (
    "Neon + Ondas",
    "Tipografia Facial",
    "Topografia 3D",
)

ASCII_GLYPHS = ("@", "#", "&", "*", "+", ":", ".")
MOUTH_GLYPHS = ("O", "A", "!", "*", "#", "~", "+")


def hsv_to_bgr(hue_deg: float, sat: float = 0.92, val: float = 1.0) -> tuple[int, int, int]:
    """Convierte HSV (hue en grados [0, 360)) a tupla BGR de OpenCV."""
    h = (hue_deg % 360.0) / 60.0
    s = float(np.clip(sat, 0.0, 1.0))
    v = float(np.clip(val, 0.0, 1.0))
    c = v * s
    x = c * (1.0 - abs((h % 2.0) - 1.0))
    m = v - c
    if h < 1.0:
        r, g, b = c, x, 0.0
    elif h < 2.0:
        r, g, b = x, c, 0.0
    elif h < 3.0:
        r, g, b = 0.0, c, x
    elif h < 4.0:
        r, g, b = 0.0, x, c
    elif h < 5.0:
        r, g, b = x, 0.0, c
    else:
        r, g, b = c, 0.0, x
    return (int(round((b + m) * 255)), int(round((g + m) * 255)), int(round((r + m) * 255)))


@dataclass
class _Wave:
    born: float
    x: float
    y: float
    start_r: float
    speed_r: float
    life: float
    hue: float
    glyph: str | None = None


@dataclass
class _Spark:
    born: float
    x: float
    y: float
    vx: float
    vy: float
    life: float
    size: float
    hue: float
    glyph: str | None = None


class GenerativeFaceFilter:
    """Lienzo facial interactivo con 3 modos visuales y reacciones continuas."""

    def __init__(self, mode: int = 0, dark_background: bool = False, seed: int | None = 42):
        self.mode = int(mode) % len(MODE_NAMES)
        self.dark_background = bool(dark_background)
        self.extractor = SignalExtractor()
        self._rng = np.random.default_rng(seed)
        self._trail: deque[tuple[float, dict[str, np.ndarray], float]] = deque(maxlen=9)
        self._waves: list[_Wave] = []
        self._sparks: list[_Spark] = []
        self._last_wave_at: float = -math.inf
        self._last_spark_at: float = -math.inf
        self._mode_changed_at: float = -math.inf
        self.last_signals: FaceSignals | None = None

    @property
    def mode_name(self) -> str:
        return MODE_NAMES[self.mode]

    def set_mode(self, mode: int, now: float | None = None):
        target = int(mode) % len(MODE_NAMES)
        if target != self.mode:
            self.mode = target
            self._trail.clear()
            if now is not None:
                self._mode_changed_at = now

    def next_mode(self, now: float):
        self.set_mode(self.mode + 1, now=now)

    def toggle_background(self):
        self.dark_background = not self.dark_background

    def reset(self):
        self.extractor.reset()
        self._trail.clear()
        self._waves.clear()
        self._sparks.clear()
        self._last_wave_at = -math.inf
        self._last_spark_at = -math.inf
        self.last_signals = None

    def _base_canvas(self, frame: np.ndarray) -> np.ndarray:
        if self.dark_background:
            canvas = np.empty_like(frame)
            canvas[:] = (10, 7, 18)
            return canvas
        # Conserva la presencia de la webcam atenuada para realzar el arte generativo.
        dimmed = (frame.astype(np.float32) * 0.22 + np.asarray((8.0, 5.0, 14.0), dtype=np.float32))
        return np.clip(dimmed, 0, 255).astype(np.uint8)

    def _prune(self, now: float):
        self._waves = [w for w in self._waves if 0.0 <= now - w.born < w.life]
        self._sparks = [s for s in self._sparks if 0.0 <= now - s.born < s.life]

    def _spawn_reactions(self, signals: FaceSignals, hue: float, now: float):
        if signals.wink_triggered:
            self.next_mode(now)
            self._spawn_ring_burst(signals.points[list(FACE_OVAL)][::3], hue + 60.0, now, count=18)

        if signals.mouth_ratio > 0.24 and now - self._last_wave_at >= 0.11:
            self._last_wave_at = now
            mx, my = float(signals.mouth_center[0]), float(signals.mouth_center[1])
            scale = 0.6 + 1.5 * signals.mouth_ratio
            glyph = MOUTH_GLYPHS[int(self._rng.integers(0, len(MOUTH_GLYPHS)))]
            self._waves.append(
                _Wave(
                    born=now,
                    x=mx,
                    y=my,
                    start_r=max(6.0, signals.eye_width * 0.10),
                    speed_r=signals.eye_width * (1.1 + 1.6 * signals.mouth_ratio),
                    life=0.65 + 0.45 * signals.mouth_ratio,
                    hue=(hue + 35.0 * signals.mouth_ratio) % 360.0,
                    glyph=glyph,
                )
            )
            self._waves = self._waves[-24:]
            # En modo tipográfico también lanza caracteres sueltos desde la boca.
            if self.mode == 1:
                for _ in range(int(2 + 4 * signals.mouth_ratio)):
                    angle = float(self._rng.uniform(0.0, 2.0 * math.pi))
                    speed = signals.eye_width * float(self._rng.uniform(0.45, 1.35)) * scale
                    self._sparks.append(
                        _Spark(
                            born=now,
                            x=mx,
                            y=my,
                            vx=math.cos(angle) * speed,
                            vy=math.sin(angle) * speed - 20.0,
                            life=float(self._rng.uniform(0.55, 1.05)),
                            size=0.45 + 0.45 * signals.mouth_ratio,
                            hue=(hue + float(self._rng.uniform(-30.0, 45.0))) % 360.0,
                            glyph=MOUTH_GLYPHS[int(self._rng.integers(0, len(MOUTH_GLYPHS)))],
                        )
                    )

        if signals.brow_lift > 0.42 and now - self._last_spark_at >= 0.14:
            self._last_spark_at = now
            for eye_center in (signals.left_eye_center, signals.right_eye_center):
                ex, ey = float(eye_center[0]), float(eye_center[1])
                for _ in range(5):
                    angle = float(self._rng.uniform(-math.pi * 0.95, -math.pi * 0.05))
                    speed = signals.eye_width * float(self._rng.uniform(0.55, 1.35))
                    self._sparks.append(
                        _Spark(
                            born=now,
                            x=ex,
                            y=ey - signals.eye_width * 0.12,
                            vx=math.cos(angle) * speed,
                            vy=math.sin(angle) * speed,
                            life=float(self._rng.uniform(0.45, 0.85)),
                            size=float(self._rng.uniform(2.0, 4.5)),
                            hue=(hue + 140.0 + float(self._rng.uniform(-25.0, 25.0))) % 360.0,
                        )
                    )
        self._sparks = self._sparks[-90:]

    def _spawn_ring_burst(self, pts: np.ndarray, hue: float, now: float, count: int = 16):
        center = pts.mean(axis=0)
        for p in pts[:count]:
            direction = p - center
            norm = max(float(np.linalg.norm(direction)), 1.0)
            vel = (direction / norm) * float(self._rng.uniform(80.0, 190.0))
            self._sparks.append(
                _Spark(
                    born=now,
                    x=float(p[0]),
                    y=float(p[1]),
                    vx=float(vel[0]),
                    vy=float(vel[1]),
                    life=0.65,
                    size=3.5,
                    hue=(hue + float(self._rng.uniform(-30.0, 30.0))) % 360.0,
                )
            )
        self._sparks = self._sparks[-90:]

    def render(self, frame: np.ndarray, observation: FaceObservation | None, now: float) -> np.ndarray:
        canvas = self._base_canvas(frame)
        self._prune(now)

        signals = self.extractor.extract(observation, frame.shape, now)
        self.last_signals = signals
        if signals is None:
            self._trail.clear()
            self._draw_waves_and_sparks(canvas, now, glow_layer=None)
            return canvas

        # Tono principal modulado por inclinación de la cabeza y sonrisa.
        base_hue = (195.0 + math.degrees(signals.head_tilt) * 3.2 + signals.smile * 115.0) % 360.0
        self._spawn_reactions(signals, base_hue, now)

        contours = {
            name: np.rint(signals.points[list(ids)]).astype(np.int32)
            for name, ids, _ in CONTOUR_SPECS
        }
        self._trail.append((now, contours, base_hue))

        if self.mode == 0:
            self._render_neon(canvas, signals, contours, base_hue, now)
        elif self.mode == 1:
            self._render_ascii(canvas, signals, contours, base_hue, now)
        else:
            self._render_z_mesh(canvas, signals, base_hue, now)

        if now - self._mode_changed_at < 1.5:
            fade = 1.0 - (now - self._mode_changed_at) / 1.5
            banner_color = hsv_to_bgr(base_hue, 0.7, 0.5 + 0.5 * fade)
            cv2.putText(
                canvas,
                f"MODO {self.mode + 1}: {self.mode_name.upper()}",
                (16, 46),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                banner_color,
                2,
                cv2.LINE_AA,
            )
        return canvas

    def _draw_waves_and_sparks(self, canvas: np.ndarray, now: float, glow_layer: np.ndarray | None):
        target = glow_layer if glow_layer is not None else canvas
        h, w = canvas.shape[:2]
        for wave in self._waves:
            age = now - wave.born
            progress = age / wave.life
            radius = int(round(wave.start_r + wave.speed_r * age))
            if radius <= 1:
                continue
            val = max(0.0, 1.0 - progress)
            color = hsv_to_bgr(wave.hue, 0.9, val)
            thickness = max(1, int(round(3 * val)))
            cv2.circle(target, (int(round(wave.x)), int(round(wave.y))), radius, color, thickness, cv2.LINE_AA)
            if glow_layer is not None:
                core = hsv_to_bgr(wave.hue, 0.35, min(1.0, val * 1.15))
                cv2.circle(canvas, (int(round(wave.x)), int(round(wave.y))), radius, core, 1, cv2.LINE_AA)

        for spark in self._sparks:
            age = now - spark.born
            progress = age / spark.life
            val = max(0.0, 1.0 - progress)
            px = int(round(spark.x + spark.vx * age))
            py = int(round(spark.y + spark.vy * age + 65.0 * age * age))
            if not (0 <= px < w and 0 <= py < h):
                continue
            color = hsv_to_bgr(spark.hue, 0.85, val)
            if spark.glyph is not None:
                cv2.putText(
                    canvas,
                    spark.glyph,
                    (px, py),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.35, spark.size * (0.7 + 0.4 * val)),
                    color,
                    1,
                    cv2.LINE_AA,
                )
            else:
                r = max(1, int(round(spark.size * val)))
                cv2.circle(target, (px, py), r + 1, color, -1, cv2.LINE_AA)
                cv2.circle(canvas, (px, py), max(1, r - 1), (255, 255, 255), -1, cv2.LINE_AA)

    def _render_neon(
        self,
        canvas: np.ndarray,
        signals: FaceSignals,
        contours: dict[str, np.ndarray],
        base_hue: float,
        now: float,
    ):
        glow = np.zeros_like(canvas)

        # Estela temporal de fotogramas previos.
        trail_items = list(self._trail)[:-1]
        trail_boost = 0.45 + 0.55 * signals.head_speed
        for idx, (_, past_contours, past_hue) in enumerate(trail_items):
            ratio = ((idx + 1) / max(len(trail_items) + 1, 1)) ** 1.5
            ghost_color = hsv_to_bgr(past_hue - 25.0 * (1.0 - ratio), 0.9, 0.55 * ratio * trail_boost)
            for name, _, closed in CONTOUR_SPECS:
                if name in ("oval", "outer_lips", "left_eye", "right_eye"):
                    cv2.polylines(glow, [past_contours[name]], closed, ghost_color, 2, cv2.LINE_AA)

        # Contorno principal en capa de brillo + núcleo brillante.
        thick = max(2, int(round(3 + 3 * signals.smile + 2 * signals.mouth_ratio)))
        for name, _, closed in CONTOUR_SPECS:
            offset = 0.0
            if "lip" in name:
                offset = 45.0 + 40.0 * signals.mouth_ratio
            elif "eye" in name or "brow" in name:
                offset = -35.0 + 35.0 * signals.brow_lift
            neon_color = hsv_to_bgr(base_hue + offset, 0.95, 1.0)
            core_color = hsv_to_bgr(base_hue + offset, 0.25, 1.0)
            pts = contours[name]
            cv2.polylines(glow, [pts], closed, neon_color, thick, cv2.LINE_AA)
            cv2.polylines(canvas, [pts], closed, core_color, 1, cv2.LINE_AA)

        # Puntos de iris si el modelo incluye los 478 landmarks.
        if len(signals.points) >= 478:
            for iris_idx in (468, 473):
                ix, iy = int(round(signals.points[iris_idx, 0])), int(round(signals.points[iris_idx, 1]))
                iris_r = max(3, int(round(signals.eye_width * (0.045 + 0.03 * signals.brow_lift))))
                cv2.circle(glow, (ix, iy), iris_r + 2, hsv_to_bgr(base_hue + 140.0, 0.95, 1.0), 2, cv2.LINE_AA)
                cv2.circle(canvas, (ix, iy), max(2, iris_r - 1), (255, 255, 255), -1, cv2.LINE_AA)

        self._draw_waves_and_sparks(canvas, now, glow_layer=glow)
        blurred = cv2.GaussianBlur(glow, (0, 0), 5.0)
        cv2.add(canvas, blurred, dst=canvas)

    def _render_ascii(
        self,
        canvas: np.ndarray,
        signals: FaceSignals,
        contours: dict[str, np.ndarray],
        base_hue: float,
        now: float,
    ):
        # Líneas guía muy tenues + estela suave.
        trail_items = list(self._trail)[:-1]
        for idx, (_, past_contours, past_hue) in enumerate(trail_items[-4:]):
            val = 0.25 * (idx + 1) / 4.0
            cv2.polylines(canvas, [past_contours["oval"]], True, hsv_to_bgr(past_hue, 0.8, val), 1, cv2.LINE_AA)

        groups = (
            (FACE_OVAL[::2], 0.42, 0.0, "@#*+"),
            (LEFT_EYEBROW, 0.46 + 0.16 * signals.brow_lift, -40.0, "^~="),
            (RIGHT_EYEBROW, 0.46 + 0.16 * signals.brow_lift, -40.0, "^~="),
            (LEFT_EYE[::2], 0.40, 120.0, "o0*"),
            (RIGHT_EYE[::2], 0.40, 120.0, "o0*"),
            (NOSE_BRIDGE[::2] + NOSE_BASE[::2], 0.40, 25.0, "|:+"),
            (OUTER_LIPS[::2], 0.45 + 0.45 * signals.mouth_ratio, 60.0, "O#@!"),
            (INNER_LIPS[::3], 0.42 + 0.35 * signals.mouth_ratio, 90.0, "*+o"),
        )
        phase = int(now * 6.0)
        h, w = canvas.shape[:2]
        for indices, scale, hue_shift, charset in groups:
            color = hsv_to_bgr(base_hue + hue_shift, 0.85, 0.98)
            for k, idx in enumerate(indices):
                px = int(round(signals.points[idx, 0]))
                py = int(round(signals.points[idx, 1]))
                if 0 <= px < w and 0 <= py < h:
                    glyph = charset[(k + phase) % len(charset)]
                    cv2.putText(
                        canvas,
                        glyph,
                        (px - 4, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        scale,
                        color,
                        1,
                        cv2.LINE_AA,
                    )

        # Palabra reactiva dentro de la boca cuando se abre.
        if signals.mouth_ratio > 0.32:
            mx, my = int(round(signals.mouth_center[0])), int(round(signals.mouth_center[1]))
            word = "WOW" if signals.brow_lift > 0.38 else "AAH"
            cv2.putText(
                canvas,
                word,
                (mx - 24, my + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55 + 0.55 * signals.mouth_ratio,
                hsv_to_bgr(base_hue + 110.0, 0.75, 1.0),
                2,
                cv2.LINE_AA,
            )

        self._draw_waves_and_sparks(canvas, now, glow_layer=None)

    def _render_z_mesh(
        self,
        canvas: np.ndarray,
        signals: FaceSignals,
        base_hue: float,
        now: float,
    ):
        glow = np.zeros_like(canvas)
        pts = signals.points
        z = signals.z_values[:468]
        z_min, z_max = float(z.min()), float(z.max())
        z_span = max(z_max - z_min, 1e-5)
        # depth = 1.0 en la punta de la nariz (más cerca), 0.0 en los bordes.
        depth = 1.0 - (z - z_min) / z_span

        # Efecto paralaje anaglifo al inclinar o mover la cabeza.
        parallax = int(round(signals.head_tilt * signals.eye_width * 0.14 + signals.head_speed * 6.0))
        if abs(parallax) >= 2:
            oval_pts = np.rint(pts[list(FACE_OVAL)]).astype(np.int32)
            left_shift = oval_pts + np.asarray((-parallax, 0), dtype=np.int32)
            right_shift = oval_pts + np.asarray((parallax, 0), dtype=np.int32)
            cv2.polylines(glow, [left_shift], True, (220, 180, 40), 2, cv2.LINE_AA)
            cv2.polylines(glow, [right_shift], True, (180, 50, 235), 2, cv2.LINE_AA)

        # Anillos concéntricos + costillas radiales deformadas levemente al abrir la boca.
        pulse = 1.0 + 0.08 * signals.mouth_ratio * math.sin(now * 10.0)
        center = signals.center

        def _deformed(idx: int) -> tuple[int, int]:
            p = center + (pts[idx] - center) * pulse
            return int(round(p[0])), int(round(p[1]))

        rings = (
            (FACE_OVAL, True),
            (INNER_CHEEK_RING, True),
            (LEFT_EYE, True),
            (RIGHT_EYE, True),
            (OUTER_LIPS, True),
        )
        for ring, closed in rings:
            pairs = list(zip(ring, ring[1:])) + ([(ring[-1], ring[0])] if closed else [])
            for a, b in pairs:
                d_avg = 0.5 * (float(depth[a]) + float(depth[b]))
                color = hsv_to_bgr(base_hue + d_avg * 145.0, 0.9, 0.35 + 0.65 * d_avg)
                pa, pb = _deformed(a), _deformed(b)
                cv2.line(glow, pa, pb, color, 2, cv2.LINE_AA)
                cv2.line(canvas, pa, pb, color, 1, cv2.LINE_AA)

        for rib in Z_RADIAL_RIBS:
            for a, b in zip(rib, rib[1:]):
                d_avg = 0.5 * (float(depth[a]) + float(depth[b]))
                color = hsv_to_bgr(base_hue + 40.0 + d_avg * 120.0, 0.85, 0.4 + 0.6 * d_avg)
                cv2.line(canvas, _deformed(a), _deformed(b), color, 1, cv2.LINE_AA)

        # Nodos 3D escalados según profundidad Z.
        sample_nodes = set(FACE_OVAL[::2]) | set(INNER_CHEEK_RING[::2]) | set(OUTER_LIPS[::2]) | set(NOSE_BRIDGE)
        for idx in sample_nodes:
            d = float(depth[idx])
            node_r = max(2, int(round(2 + 4 * d)))
            color = hsv_to_bgr(base_hue + d * 150.0, 0.85, 0.55 + 0.45 * d)
            cv2.circle(canvas, _deformed(idx), node_r, color, -1, cv2.LINE_AA)

        self._draw_waves_and_sparks(canvas, now, glow_layer=glow)
        blurred = cv2.GaussianBlur(glow, (0, 0), 4.0)
        cv2.add(canvas, blurred, dst=canvas)
