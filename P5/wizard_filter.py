"""Recognize closed air gestures and draw the wizard's face effects."""
import math

import cv2
import numpy as np


SPELLS = {
    'circle': ('Doctor Strange', (45, 165, 255)),
    'square': ('Super Saiyan', (50, 228, 255)),
    'triangle': ('Sharingan', (55, 55, 245)),
    'failed': ('Power-Up fallido', (245, 195, 110)),
}

LEFT_EYE_RING = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
RIGHT_EYE_RING = (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398)
LEFT_BROW = (70, 63, 105, 66, 107)
RIGHT_BROW = (300, 293, 334, 296, 336)


def classify_shape(points):
    """Classify a single closed contour in pixel coordinates."""
    if len(points) < 8:
        return 'failed'
    points = np.asarray(points, dtype=np.float32)
    span = np.ptp(points, axis=0)
    size = float(max(span))
    if size < 15 or min(span) < .40 * size:
        return 'failed'
    if np.linalg.norm(points[0] - points[-1]) > .27 * size:
        return 'failed'
    contour = points.reshape(-1, 1, 2)
    perimeter = cv2.arcLength(contour, True)
    if not 2.2 * size < perimeter < 5.6 * size:
        return 'failed'
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    area = cv2.contourArea(contour)
    if hull_area <= 0 or area / hull_area < .83:
        return 'failed'
    corners = cv2.approxPolyDP(contour, .037 * perimeter, True).reshape(-1, 2)
    if len(corners) in (3, 4) and cv2.contourArea(corners) >= .85 * area:
        edges = np.roll(corners, -1, axis=0) - corners
        sides = np.linalg.norm(edges, axis=1)
        if min(sides) <= .42 * max(sides):
            return 'failed'
        if len(corners) == 3:
            return 'triangle'
        units = edges / sides[:, None]
        angle_cosines = np.sum(units * np.roll(units, -1, axis=0), axis=1)
        if min(sides) >= .59 * max(sides) and np.max(np.abs(angle_cosines)) < .42:
            return 'square'
        return 'failed'
    if len(corners) >= 4 and 4 * math.pi * area / perimeter**2 >= .73:
        return 'circle'
    return 'failed'


class OneEuroFilter:
    """Adaptive low-pass filter: suppresses jitter at rest, minimizes lag in motion."""

    def __init__(self, min_cutoff=1.6, beta=6.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x = None
        self.dx = np.zeros(2, dtype=np.float64)
        self.t = None

    def reset(self):
        self.x = None
        self.dx = np.zeros(2, dtype=np.float64)
        self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def update(self, raw, now, scale):
        point = np.asarray(raw, dtype=np.float64)
        if self.x is None or self.t is None:
            self.x = point.copy()
            self.dx[:] = 0.0
            self.t = float(now)
            return self.x.copy()
        dt = float(now) - self.t
        if dt <= 1e-5:
            return self.x.copy()
        if dt > 0.35:
            self.x = point.copy()
            self.dx[:] = 0.0
            self.t = float(now)
            return self.x.copy()
        vel = (point - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * vel + (1.0 - a_d) * self.dx
        speed_norm = float(np.linalg.norm(self.dx)) / max(float(scale), 1.0)
        cutoff = self.min_cutoff + self.beta * speed_norm
        alpha = self._alpha(cutoff, dt)
        self.x = alpha * point + (1.0 - alpha) * self.x
        self.t = float(now)
        return self.x.copy()

    def predict(self, now):
        if self.x is None or self.t is None:
            return None
        dt = min(max(0.0, float(now) - self.t), 0.10)
        return self.x + self.dx * dt


def pointing_tip(hand, size, tracing):
    """Detect an extended index finger robustly across curls, knuckle folds and tilts."""
    hand = np.asarray(hand, dtype=np.float32)
    if hand.shape != (21, 3) or not np.isfinite(hand).all():
        return None
    w, h = size
    points = hand * np.array((w, h, w), dtype=np.float32)
    wrist = points[0]
    palm = max(float(np.linalg.norm(points[9] - wrist)),
               float(np.linalg.norm(points[5] - points[17])))
    if palm <= 3.0:
        return None

    fingers = points[np.arange(5, 21).reshape(4, 4)]
    segments = np.linalg.norm(np.diff(fingers, axis=1), axis=2).sum(axis=1)
    reach = np.linalg.norm(fingers[:, 3] - fingers[:, 0], axis=1)
    straightness = reach / np.maximum(segments, 1e-5)

    wrist_to_mcp = np.linalg.norm(fingers[:, 0] - wrist, axis=1)
    wrist_to_pip = np.linalg.norm(fingers[:, 1] - wrist, axis=1)
    wrist_to_tip = np.linalg.norm(fingers[:, 3] - wrist, axis=1)

    # Index finger must extend beyond its PIP and knuckle, with relaxed straightness mid-stroke.
    index_extended = (
        wrist_to_tip[0] > wrist_to_pip[0] * (1.02 if tracing else 1.06)
        and wrist_to_tip[0] > wrist_to_mcp[0] + (0.16 if tracing else 0.24) * palm
        and straightness[0] > (0.60 if tracing else 0.72)
    )
    if not index_extended:
        return None

    # Other fingers only count as raised if both straight AND extended away from the palm.
    other_raised = (
        (straightness[1:] > (0.84 if tracing else 0.80))
        & (wrist_to_tip[1:] > wrist_to_pip[1:] * (1.08 if tracing else 1.05))
        & (wrist_to_tip[1:] > wrist_to_mcp[1:] + (0.34 if tracing else 0.28) * palm)
        & (wrist_to_tip[1:] > (0.82 if tracing else 0.76) * wrist_to_tip[0])
    )
    if np.count_nonzero(other_raised) <= (1 if tracing else 0):
        return points[8, :2]
    return None


class AirSpell:
    def __init__(self):
        self.points = []
        self._arc_lengths = []
        self.spell = None
        self.cast_at = -100.
        self.last_seen = -100.
        self.locked = False
        self.cursor = None
        self._filter = OneEuroFilter()
        self._pending_tip = None
        self._pending_at = -100.
        self._outlier_tip = None
        self._outlier_at = -100.

    def _append_point(self, tip):
        point = np.asarray(tip, dtype=np.float64)
        if not self.points:
            self.points.append(point)
            self._arc_lengths.append(0.0)
            return
        step = float(np.linalg.norm(point - self.points[-1]))
        self.points.append(point)
        self._arc_lengths.append(self._arc_lengths[-1] + step)

    def _find_closed_loop(self, min_dim):
        if len(self.points) < 12:
            return None
        pts = np.asarray(self.points, dtype=np.float64)
        tip = pts[-1]
        total_len = self._arc_lengths[-1]

        # 1) Direct closure back near the starting point.
        span = np.ptp(pts, axis=0)
        box_max, box_min = float(max(span)), float(min(span))
        if (box_min >= 0.08 * min_dim and total_len >= 2.1 * box_max
                and float(np.linalg.norm(tip - pts[0])) < 0.16 * box_max):
            return 0

        # 2) Closure onto an early point (trims short lead-in tail from raising the hand).
        max_start = min(len(pts) - 10, max(2, len(pts) // 3))
        if box_min >= 0.08 * min_dim and total_len >= 2.1 * box_max:
            dists = np.linalg.norm(pts[1:max_start] - tip, axis=1)
            for offset in np.flatnonzero(dists < 0.13 * box_max):
                idx = int(offset) + 1
                loop_len = total_len - self._arc_lengths[idx]
                loop_span = np.ptp(pts[idx:], axis=0)
                loop_max, loop_min = float(max(loop_span)), float(min(loop_span))
                if loop_min >= 0.08 * min_dim and loop_len >= 2.1 * loop_max:
                    return idx
        return None

    def update(self, hands, size, now):
        min_dim, max_dim = float(min(size)), float(max(size))
        tips = [tip for hand in hands
                if (tip := pointing_tip(hand, size, tracing=bool(self.points))) is not None]
        if tips:
            ref = self._filter.predict(now)
            if ref is None and self._pending_tip is not None:
                ref = self._pending_tip
            if ref is None and self.points:
                ref = self.points[-1]
            raw_tip = (min(tips, key=lambda xy: float(np.linalg.norm(xy - ref)))
                       if ref is not None else tips[0])

            if self.locked:
                self.last_seen = now
                self.cursor = self._filter.update(raw_tip, now, min_dim)
                self._pending_tip = None
                self._outlier_tip = None
                return

            if not self.points:
                # Confirm pointing across 2 close frames to prevent 1-frame stray starts.
                if self._pending_tip is None or now - self._pending_at > 0.25:
                    self._filter.reset()
                    self.cursor = self._filter.update(raw_tip, now, min_dim)
                    self._pending_tip = self.cursor.copy()
                    self._pending_at = now
                    self.last_seen = now
                    return
                if float(np.linalg.norm(raw_tip - self._pending_tip)) > 0.22 * max_dim:
                    self._filter.reset()
                    self.cursor = self._filter.update(raw_tip, now, min_dim)
                    self._pending_tip = self.cursor.copy()
                    self._pending_at = now
                    self.last_seen = now
                    return
                first_tip = self._pending_tip
                self._pending_tip = None
                self.cursor = self._filter.update(raw_tip, now, min_dim)
                self.last_seen = now
                self._append_point(first_tip)
                if float(np.linalg.norm(self.cursor - first_tip)) >= max(3.0, min_dim * 0.005):
                    self._append_point(self.cursor)
                return

            # Time-aware outlier gate: reject isolated 1-frame teleports, recover on fast moves.
            dt = max(1.0 / 30.0, float(now) - self.last_seen)
            max_jump = min(0.45, 0.16 + 0.9 * dt) * max_dim
            anchor = self.cursor if self.cursor is not None else self.points[-1]
            raw_distance = float(np.linalg.norm(raw_tip - anchor))
            if raw_distance > max_jump:
                if (self._outlier_tip is not None
                        and now - self._outlier_at <= 0.18
                        and float(np.linalg.norm(raw_tip - self._outlier_tip)) <= 0.14 * max_dim):
                    self._outlier_tip = None
                else:
                    self._outlier_tip = np.asarray(raw_tip, dtype=np.float64)
                    self._outlier_at = now
                    if now - self.last_seen > 0.45:
                        self.finish(now)
                    return
            else:
                self._outlier_tip = None

            self.last_seen = now
            tip = self._filter.update(raw_tip, now, min_dim)
            self.cursor = tip
            min_step = max(3.0, min_dim * 0.005)
            if float(np.linalg.norm(tip - self.points[-1])) >= min_step:
                self._append_point(tip)

            loop_start = self._find_closed_loop(min_dim)
            if loop_start is not None:
                if loop_start > 0:
                    self.points = self.points[loop_start:]
                self.finish(now)
            elif len(self.points) >= 600:
                self.finish(now)
            return

        # No pointing finger detected in this frame.
        if now - self._pending_at > 0.18:
            self._pending_tip = None
        if now - self.last_seen >= 0.15:
            self.cursor = None
        if self.points:
            if now - self.last_seen >= 0.45:
                self.finish(now)
        elif self.locked and now - self.last_seen >= 0.22:
            self.locked = False
            self._filter.reset()

    def finish(self, now):
        if len(self.points) >= 4 and max(np.ptp(self.points, axis=0)) >= 20:
            self.spell = classify_shape(self.points)
            self.cast_at = now
        self.points.clear()
        self._arc_lengths.clear()
        self._pending_tip = None
        self._outlier_tip = None
        self.locked = True


def _face_exclusion_mask(frame_shape, points):
    """Máscara de la silueta facial para que halos y mandalas queden detrás de la cabeza."""
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(np.rint(points[:468]).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255, cv2.LINE_AA)
    return cv2.GaussianBlur(mask, (0, 0), 5.0)


def _add_glow_layer(frame, layer, weight=0.9, blur_sigma=4.0, exclude_mask=None):
    """Añade una capa luminosa con bloom suave (opcionalmente oculta tras la cara)."""
    if blur_sigma > 0:
        h, w = frame.shape[:2]
        sw, sh = max(1, w // 3), max(1, h // 3)
        small = cv2.resize(layer, (sw, sh), interpolation=cv2.INTER_AREA)
        bloom = cv2.GaussianBlur(small, (0, 0), max(1.0, blur_sigma / 3.0))
        bloom = cv2.resize(bloom, (w, h), interpolation=cv2.INTER_LINEAR)
        combined = cv2.addWeighted(layer, 0.85, bloom, 1.05, 0)
    else:
        combined = layer
    if exclude_mask is not None:
        keep = (1.0 - exclude_mask.astype(np.float32) / 255.0)[..., None]
        combined = np.clip(combined * keep, 0, 255).astype(np.uint8)
    cv2.addWeighted(frame, 1.0, combined, weight, 0, dst=frame)


def _draw_mandala(canvas, center, radius, angle, primary, bright, thickness=2):
    """Dibuja un escudo/mandala rúnico giratorio estilo Doctor Strange."""
    cx, cy = float(center[0]), float(center[1])
    r = max(12.0, float(radius))
    ri = int(round(r))
    c_int = (int(round(cx)), int(round(cy)))

    # Anillos exteriores y marcas rúnicas
    cv2.circle(canvas, c_int, ri, primary, thickness, cv2.LINE_AA)
    cv2.circle(canvas, c_int, max(4, int(round(r * 0.87))), primary, max(1, thickness - 1), cv2.LINE_AA)
    cv2.circle(canvas, c_int, max(3, int(round(r * 0.52))), bright, max(1, thickness - 1), cv2.LINE_AA)

    for i in range(16):
        a = angle + i * (2.0 * math.pi / 16.0)
        p0 = (int(round(cx + r * 0.87 * math.cos(a))), int(round(cy + r * 0.87 * math.sin(a))))
        p1 = (int(round(cx + r * 0.99 * math.cos(a))), int(round(cy + r * 0.99 * math.sin(a))))
        cv2.line(canvas, p0, p1, bright, max(1, thickness - 1), cv2.LINE_AA)
        if i % 2 == 0:
            dot = (int(round(cx + r * 0.93 * math.cos(a + math.pi / 16.0))),
                   int(round(cy + r * 0.93 * math.sin(a + math.pi / 16.0))))
            cv2.circle(canvas, dot, max(1, thickness), bright, -1, cv2.LINE_AA)

    # Dos cuadrados superpuestos (estrella de 8 puntas de Kamar-Taj)
    sq_r = r * 0.85
    for offset in (0.0, math.pi / 4.0):
        pts = np.array([
            (int(round(cx + sq_r * math.cos(angle + offset + k * math.pi / 2.0))),
             int(round(cy + sq_r * math.sin(angle + offset + k * math.pi / 2.0))))
            for k in range(4)
        ], dtype=np.int32)
        cv2.polylines(canvas, [pts], True, bright, thickness, cv2.LINE_AA)

    # Triángulos interiores girando en sentido contrario
    tri_r = r * 0.50
    for offset in (0.0, math.pi):
        pts = np.array([
            (int(round(cx + tri_r * math.cos(-angle * 1.4 + offset + k * 2.0 * math.pi / 3.0))),
             int(round(cy + tri_r * math.sin(-angle * 1.4 + offset + k * 2.0 * math.pi / 3.0))))
            for k in range(3)
        ], dtype=np.int32)
        cv2.polylines(canvas, [pts], True, primary, max(1, thickness - 1), cv2.LINE_AA)


def _draw_doctor_strange(frame, points, right, up, scale, now):
    """Mandalas rúnicos naranjas girando + Ojo de Agamotto y chispas arcanas."""
    h, w = frame.shape[:2]
    head_center = 0.5 * (points[10] + points[152])
    primary = (22, 130, 255)
    bright = (110, 220, 255)

    # Gran mandala arcano detrás de la cabeza
    back_glow = np.zeros_like(frame)
    _draw_mandala(back_glow, head_center, scale * 1.52, now * 0.75, primary, bright, thickness=2)

    # Chispas doradas orbitando el portal
    for i in range(20):
        a = now * 1.2 + i * (2.0 * math.pi / 20.0)
        dist = scale * (1.58 + 0.14 * math.sin(now * 3.5 + i * 1.7))
        sx = int(round(head_center[0] + dist * math.cos(a)))
        sy = int(round(head_center[1] + dist * math.sin(a)))
        tx = int(round(sx - 9 * math.sin(a)))
        ty = int(round(sy + 9 * math.cos(a)))
        cv2.line(back_glow, (sx, sy), (tx, ty), bright, 2, cv2.LINE_AA)

    face_mask = _face_exclusion_mask(frame.shape, points)
    _add_glow_layer(frame, back_glow, weight=0.95, blur_sigma=7.0, exclude_mask=face_mask)

    # Dos mini-mandalas flotantes a ambos lados de la cabeza
    front_glow = np.zeros_like(frame)
    for side in (-1.0, 1.0):
        side_center = head_center + scale * (side * 1.38 * right - 0.10 * up)
        _draw_mandala(front_glow, side_center, scale * 0.42, -side * now * 1.4, primary, bright, thickness=2)

    # Ojo de Agamotto en la base del cuello (debajo de la barbilla)
    amulet = points[152] - scale * 0.36 * up
    ax, ay = int(round(amulet[0])), int(round(amulet[1]))
    arx, ary = max(6, int(round(scale * 0.24))), max(4, int(round(scale * 0.15)))
    tilt = math.degrees(math.atan2(right[1], right[0]))
    if 0 <= ax < w and 0 <= ay < h:
        cv2.ellipse(frame, (ax, ay), (arx, ary), tilt, 0, 360, (32, 165, 225), -1, cv2.LINE_AA)
        cv2.ellipse(frame, (ax, ay), (arx, ary), tilt, 0, 360, (18, 95, 160), 2, cv2.LINE_AA)
        cv2.ellipse(frame, (ax, ay), (max(3, arx // 2), max(2, ary // 2)), tilt, 0, 360, (95, 235, 55), -1, cv2.LINE_AA)
        cv2.ellipse(front_glow, (ax, ay), (max(4, int(arx * 0.65)), max(3, int(ary * 0.65))),
                    tilt, 0, 360, (35, 165, 35), 2, cv2.LINE_AA)

    _add_glow_layer(frame, front_glow, weight=0.88, blur_sigma=5.0)


def _draw_eye_iris(frame, points, ring_ids, iris_idx, mode, now):
    """Dibuja iris personalizados (Super Saiyan o Sharingan) recortados al contorno real del ojo."""
    h, w = frame.shape[:2]
    ring = np.rint(points[list(ring_ids)]).astype(np.int32)
    x, y, bw, bh = cv2.boundingRect(ring)
    if bw < 6 or bh < 3:
        return
    x0, y0 = max(0, x - 2), max(0, y - 2)
    x1, y1 = min(w, x + bw + 2), min(h, y + bh + 2)
    if x1 <= x0 or y1 <= y0:
        return

    if len(points) > iris_idx:
        center = points[iris_idx]
    else:
        center = points[list(ring_ids)].mean(axis=0)
    cx, cy = float(center[0]) - x0, float(center[1]) - y0
    radius = max(3, int(round(min(bw * 0.27, max(bh * 0.52, bw * 0.19)))))

    roi = frame[y0:y1, x0:x1]
    overlay = roi.copy()
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillPoly(mask, [ring - np.array((x0, y0), dtype=np.int32)], 255, cv2.LINE_AA)

    ic = (int(round(cx)), int(round(cy)))
    if mode == 'saiyan':
        cv2.circle(overlay, ic, radius, (205, 240, 80), -1, cv2.LINE_AA)
        cv2.circle(overlay, ic, radius, (110, 165, 35), 1, cv2.LINE_AA)
        cv2.circle(overlay, ic, max(1, int(round(radius * 0.42))), (25, 45, 25), -1, cv2.LINE_AA)
        cv2.circle(overlay, (ic[0] - max(1, radius // 3), ic[1] - max(1, radius // 3)),
                   max(1, radius // 4), (255, 255, 240), -1, cv2.LINE_AA)
        alpha_strength = 0.85
    else:  # Sharingan
        cv2.circle(overlay, ic, radius, (24, 24, 215), -1, cv2.LINE_AA)
        cv2.circle(overlay, ic, max(2, int(round(radius * 0.82))), (35, 35, 248), -1, cv2.LINE_AA)
        cv2.circle(overlay, ic, radius, (15, 12, 35), max(1, radius // 6), cv2.LINE_AA)
        inner_r = max(2, int(round(radius * 0.56)))
        cv2.circle(overlay, ic, inner_r, (18, 12, 22), 1, cv2.LINE_AA)
        cv2.circle(overlay, ic, max(1, int(round(radius * 0.25))), (12, 10, 14), -1, cv2.LINE_AA)
        # 3 tomoe girando sobre el anillo interior
        tomoe_r = max(1, int(round(radius * 0.17)))
        for k in range(3):
            a = now * 2.4 + k * (2.0 * math.pi / 3.0)
            tx = int(round(cx + inner_r * math.cos(a)))
            ty = int(round(cy + inner_r * math.sin(a)))
            cv2.circle(overlay, (tx, ty), tomoe_r, (10, 8, 12), -1, cv2.LINE_AA)
            tail = (int(round(tx - tomoe_r * 1.4 * math.sin(a))),
                    int(round(ty + tomoe_r * 1.4 * math.cos(a))))
            cv2.line(overlay, (tx, ty), tail, (10, 8, 12), max(1, tomoe_r), cv2.LINE_AA)
        cv2.circle(overlay, (ic[0] - max(1, radius // 3), ic[1] - max(1, radius // 3)),
                   max(1, radius // 5), (245, 245, 255), -1, cv2.LINE_AA)
        alpha_strength = 0.92

    alpha = (cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32) / 255.0 * alpha_strength)[..., None]
    roi[:] = np.clip(roi * (1.0 - alpha) + overlay * alpha, 0, 255).astype(np.uint8)


def _draw_lightning_bolt(canvas, start, end, seed, color, core_color, spread):
    """Dibuja un rayo quebrado determinista para el aura Super Saiyan."""
    rng = np.random.default_rng(seed)
    p0 = np.asarray(start, dtype=np.float32)
    p1 = np.asarray(end, dtype=np.float32)
    diff = p1 - p0
    length = max(float(np.linalg.norm(diff)), 1.0)
    perp = np.array((-diff[1], diff[0]), dtype=np.float32) / length
    pts = [p0]
    for step in range(1, 6):
        t = step / 6.0
        jitter = rng.uniform(-spread, spread) * (1.0 - abs(t - 0.5) * 1.4)
        pts.append(p0 + diff * t + perp * jitter)
    pts.append(p1)
    poly = np.rint(pts).astype(np.int32)
    cv2.polylines(canvas, [poly], False, color, 4, cv2.LINE_AA)
    cv2.polylines(canvas, [poly], False, core_color, 1, cv2.LINE_AA)


def _draw_super_saiyan(frame, points, right, up, scale, now):
    """Aura eléctrica dorada, rayos chispeantes, pelo puntiagudo y ojos/cejas Saiyan."""
    forehead = points[10]
    head_center = 0.5 * (forehead + points[152])
    pulse = 1.0 + 0.06 * math.sin(now * 14.0)

    def local(anchor, xy):
        coords = np.asarray(xy, dtype=np.float32)
        return np.rint(anchor + scale * (coords[..., :1] * right + coords[..., 1:] * up)).astype(np.int32)

    # 1) Aura dorada flamígera ascendente (estilo Dragon Ball) detrás de la cabeza
    aura = np.zeros_like(frame)
    for k in range(13):
        frac = (k / 12.0) * 2.0 - 1.0  # -1 (izquierda) a +1 (derecha)
        wobble = 0.16 * math.sin(now * 12.0 + k * 1.7)
        base_x = frac * 1.05
        base_y = -1.10 + 0.35 * (1.0 - abs(frac))
        peak_h = (1.65 - 0.35 * abs(frac) + wobble) * pulse
        width = 0.28 + 0.08 * (1.0 - abs(frac))
        flame = local(head_center, [
            (base_x - width, base_y),
            (base_x - width * 0.55, base_y + peak_h * 0.55),
            (base_x + 0.08 * math.sin(now * 9.0 + k), base_y + peak_h),
            (base_x + width * 0.55, base_y + peak_h * 0.55),
            (base_x + width, base_y),
        ])
        cv2.fillPoly(aura, [flame], (25, 185, 255), cv2.LINE_AA)

    # Partículas doradas ascendentes
    for i in range(16):
        phase = (now * 1.15 + i * 0.19) % 1.0
        px = (math.sin(i * 3.1) * 1.35) * scale
        py = (-1.15 + phase * 2.75) * scale
        pt = head_center + px * right + py * up
        rad = max(2, int(round(scale * 0.045 * (1.0 - phase * 0.7))))
        cv2.circle(aura, tuple(np.rint(pt).astype(int)), rad, (110, 240, 255), -1, cv2.LINE_AA)

    face_mask = _face_exclusion_mask(frame.shape, points)
    # Suavizado intenso para que el aura sea energía difusa y no polígonos duros
    h, w = frame.shape[:2]
    small_aura = cv2.resize(aura, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    small_aura = cv2.GaussianBlur(small_aura, (0, 0), 5.5)
    soft_aura = cv2.resize(small_aura, (w, h), interpolation=cv2.INTER_LINEAR)
    keep = (1.0 - face_mask.astype(np.float32) / 255.0)[..., None]
    soft_aura = np.clip(soft_aura * keep, 0, 255).astype(np.uint8)
    cv2.addWeighted(frame, 1.0, soft_aura, 0.95, 0, dst=frame)

    # 2) Pelo energético dorado puntiagudo sobre la cabeza
    sway = 0.025 * math.sin(now * 10.0)
    spikes_outer = [
        (-0.82, -0.08), (-1.05, 0.42 + sway), (-0.68, 0.36),
        (-0.84, 0.96 - sway), (-0.42, 0.56), (-0.34, 1.28 + sway),
        (-0.08, 0.72), (0.04, 1.40 - sway), (0.22, 0.72),
        (0.46, 1.24 + sway), (0.48, 0.56), (0.88, 0.90 - sway),
        (0.70, 0.36), (1.05, 0.38 + sway), (0.82, -0.08),
        (0.58, 0.06), (0.24, 0.14), (0.0, 0.02), (-0.24, 0.14), (-0.58, 0.06),
    ]
    hair_poly = local(forehead, spikes_outer)
    cv2.fillPoly(frame, [hair_poly], (42, 218, 255), cv2.LINE_AA)

    spikes_inner = [
        (-0.60, 0.12), (-0.62, 0.74 - sway), (-0.28, 0.44),
        (-0.22, 1.00 + sway), (-0.02, 0.56), (0.05, 1.12 - sway),
        (0.18, 0.56), (0.34, 0.96 + sway), (0.36, 0.42),
        (0.64, 0.70 - sway), (0.56, 0.12), (0.0, 0.08),
    ]
    cv2.fillPoly(frame, [local(forehead, spikes_inner)], (135, 248, 255), cv2.LINE_AA)
    cv2.polylines(frame, [hair_poly], True, (18, 128, 210), 2, cv2.LINE_AA)

    # 3) Cejas doradas y ojos verde-turquesa Saiyan
    for brow_ids in (LEFT_BROW, RIGHT_BROW):
        brow = np.rint(points[list(brow_ids)]).astype(np.int32)
        cv2.polylines(frame, [brow], False, (35, 215, 255), max(3, int(round(scale * 0.055))), cv2.LINE_AA)
        cv2.polylines(frame, [brow], False, (145, 250, 255), max(1, int(round(scale * 0.02))), cv2.LINE_AA)
    _draw_eye_iris(frame, points, LEFT_EYE_RING, 468, 'saiyan', now)
    _draw_eye_iris(frame, points, RIGHT_EYE_RING, 473, 'saiyan', now)

    # 4) Relámpagos eléctricos azul-cian saltando alrededor del aura (12 FPS estilo anime)
    bolt_layer = np.zeros_like(frame)
    tick = int(now * 12.0)
    for b in range(5):
        a0 = ((tick * 1.7 + b * 1.25) % (2.0 * math.pi))
        a1 = a0 + 0.55 * (1 if b % 2 == 0 else -1)
        r0 = scale * (0.95 + 0.15 * ((tick + b) % 3))
        r1 = scale * (1.42 + 0.18 * ((tick + 2 * b) % 3))
        start = head_center + r0 * (math.cos(a0) * right + math.sin(a0) * up)
        end = head_center + r1 * (math.cos(a1) * right + math.sin(a1) * up)
        _draw_lightning_bolt(bolt_layer, start, end, tick * 17 + b * 31,
                             (255, 215, 70), (255, 255, 240), spread=scale * 0.16)
    _add_glow_layer(frame, bolt_layer, weight=0.95, blur_sigma=4.0)


def _draw_sharingan(frame, points, right, up, scale, now):
    """Iris Sharingan con 3 tomoe giratorios, marcas místicas y aura oscura carmesí."""
    forehead = points[10]
    head_center = 0.5 * (forehead + points[152])

    # 1) Ligero oscurecimiento dramático con tinte carmesí en el entorno
    tint = np.full_like(frame, (28, 16, 54))
    cv2.addWeighted(frame, 0.86, tint, 0.14, 0, dst=frame)

    # 2) Aura oscura carmesí flamígera (estilo chakra / Susanoo) detrás de la cabeza
    def local(anchor, xy):
        coords = np.asarray(xy, dtype=np.float32)
        return np.rint(anchor + scale * (coords[..., :1] * right + coords[..., 1:] * up)).astype(np.int32)

    aura = np.zeros_like(frame)
    pulse = 1.0 + 0.06 * math.sin(now * 6.0)
    for k in range(11):
        frac = (k / 10.0) * 2.0 - 1.0
        wobble = 0.18 * math.sin(now * 7.5 + k * 2.1)
        bx = frac * 1.02
        by = -1.05 + 0.30 * (1.0 - abs(frac))
        peak = (1.55 - 0.28 * abs(frac) + wobble) * pulse
        flame = local(head_center, [
            (bx - 0.30, by),
            (bx - 0.16, by + peak * 0.55),
            (bx + 0.10 * math.cos(now * 6.5 + k), by + peak),
            (bx + 0.16, by + peak * 0.55),
            (bx + 0.30, by),
        ])
        color = (55, 18, 210) if k % 2 == 0 else (95, 22, 155)
        cv2.fillPoly(aura, [flame], color, cv2.LINE_AA)
    for i in range(18):
        phase = (now * 0.85 + i * 0.23) % 1.0
        side_off = math.sin(i * 2.4 + now * 1.5) * scale * 1.25
        y_off = (-1.0 + phase * 2.45) * scale
        pt = head_center + side_off * right + y_off * up
        rad = max(2, int(round(scale * 0.06 * (1.0 - phase * 0.7))))
        cv2.circle(aura, tuple(np.rint(pt).astype(int)), rad, (45, 30, 245), -1, cv2.LINE_AA)

    face_mask = _face_exclusion_mask(frame.shape, points)
    h, w = frame.shape[:2]
    small_aura = cv2.resize(aura, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    small_aura = cv2.GaussianBlur(small_aura, (0, 0), 5.0)
    soft_aura = cv2.resize(small_aura, (w, h), interpolation=cv2.INTER_LINEAR)
    keep = (1.0 - face_mask.astype(np.float32) / 255.0)[..., None]
    soft_aura = np.clip(soft_aura * keep, 0, 255).astype(np.uint8)
    cv2.addWeighted(frame, 1.0, soft_aura, 0.92, 0, dst=frame)

    # 3) Marcas místicas bajo los ojos y en la frente (estilo ninja/maldición)

    mark_glow = np.zeros_like(frame)
    for side in (-1.0, 1.0):
        cheek = 0.55 * points[291 if side > 0 else 61] + 0.45 * points[454 if side > 0 else 234]
        for row, dy in enumerate((0.12, 0.03, -0.06)):
            seg = local(cheek, [(-side * 0.08, dy), (side * (0.18 - row * 0.03), dy - 0.04)])
            cv2.line(frame, tuple(seg[0]), tuple(seg[1]), (25, 16, 48), 2, cv2.LINE_AA)
            cv2.line(mark_glow, tuple(seg[0]), tuple(seg[1]), (40, 35, 225), 2, cv2.LINE_AA)

    # Sello místico en el centro de la frente
    seal = local(forehead, [(0.0, -0.08), (0.06, -0.17), (0.0, -0.26), (-0.06, -0.17)])
    cv2.fillPoly(frame, [seal], (25, 15, 75), cv2.LINE_AA)
    cv2.polylines(frame, [seal], True, (50, 50, 245), 2, cv2.LINE_AA)
    cv2.polylines(mark_glow, [seal], True, (45, 45, 255), 3, cv2.LINE_AA)

    # 4) Iris Sharingan giratorios en ambos ojos
    _draw_eye_iris(frame, points, LEFT_EYE_RING, 468, 'sharingan', now)
    _draw_eye_iris(frame, points, RIGHT_EYE_RING, 473, 'sharingan', now)

    # Pequeño destello rojo alrededor de los ojos
    for eye_ring in (LEFT_EYE_RING, RIGHT_EYE_RING):
        eye_center = points[list(eye_ring)].mean(axis=0)
        cv2.circle(mark_glow, tuple(np.rint(eye_center).astype(int)),
                   max(4, int(round(scale * 0.14))), (30, 25, 195), 2, cv2.LINE_AA)
    _add_glow_layer(frame, mark_glow, weight=0.65, blur_sigma=4.0)


def _draw_anime_sweatdrop(frame, points, right, up, scale, now):
    """Gota de sudor anime deslizándose en la sien + líneas verticales de desánimo."""
    forehead = points[10]
    ink = (54, 27, 72)

    def local(anchor, xy):
        coords = np.asarray(xy, dtype=np.float32)
        return np.rint(anchor + scale * (coords[..., :1] * right + coords[..., 1:] * up)).astype(np.int32)

    # Líneas verticales azuladas de "gloom" sobre la frente
    for idx, dx in enumerate((-0.34, -0.22, -0.10, 0.02, 0.14)):
        p0 = local(forehead, (dx, -0.02))
        p1 = local(forehead, (dx, -0.32 - 0.06 * (idx % 2)))
        cv2.line(frame, tuple(p0), tuple(p1), (150, 105, 80), 2, cv2.LINE_AA)

    # Gota de sudor anime grande en la sien derecha cayendo en bucle suave
    slide = ((now * 0.65) % 1.0) * 0.16
    drop_anchor = forehead + scale * (0.58 * right + (0.04 - slide) * up)
    drop = local(drop_anchor, [
        (0.0, 0.24),
        (0.09, 0.07),
        (0.13, -0.04),
        (0.10, -0.14),
        (0.0, -0.18),
        (-0.10, -0.14),
        (-0.13, -0.04),
        (-0.09, 0.07),
    ])
    cv2.fillPoly(frame, [drop], (250, 205, 115), cv2.LINE_AA)
    cv2.polylines(frame, [drop], True, ink, 2, cv2.LINE_AA)
    # Brillo blanco característico de gota anime
    glint = local(drop_anchor, [(-0.05, 0.02), (-0.07, -0.07), (-0.03, -0.10), (-0.02, -0.02)])
    cv2.fillPoly(frame, [glint], (255, 255, 255), cv2.LINE_AA)


def draw_costume(frame, observation, spell, now=0.0):
    points = observation.landmarks[:, :2] * frame.shape[1::-1]
    axis = points[263] - points[33]
    eye_width = float(np.linalg.norm(axis))
    if eye_width < 8 or not np.isfinite(points).all():
        return
    right = axis / eye_width
    up = np.array((right[1], -right[0]))
    if np.dot(up, points[10] - points[152]) < 0:
        up = -up
    scale = max(eye_width, .53 * np.linalg.norm(points[152] - points[10]))

    if spell == 'circle':
        _draw_doctor_strange(frame, points, right, up, scale, now)
        return
    if spell == 'square':
        _draw_super_saiyan(frame, points, right, up, scale, now)
        return
    if spell == 'triangle':
        _draw_sharingan(frame, points, right, up, scale, now)
        return
    if spell == 'failed':
        _draw_anime_sweatdrop(frame, points, right, up, scale, now)
        return

    ink = (54, 27, 72)

    def local(anchor, xy):
        coords = np.asarray(xy)
        return np.rint(anchor + scale * (coords[..., :1] * right + coords[..., 1:] * up)).astype(np.int32)

    def polygon(anchor, xy, fill, outline=ink):
        contour = local(anchor, xy)
        cv2.fillPoly(frame, [contour], fill, cv2.LINE_AA)
        cv2.polylines(frame, [contour], True, outline, 2, cv2.LINE_AA)

    def ellipse(anchor, xy, radii, color, angle=0):
        center = tuple(local(anchor, xy))
        axes = tuple(max(1, round(value * scale)) for value in radii)
        tilt = math.degrees(math.atan2(right[1], right[0]))
        cv2.ellipse(frame, center, axes, tilt + angle, 0, 360, color, -1, cv2.LINE_AA)

    forehead = points[10]
    ellipse(forehead, (0, -.025), (.76, .135), ink)
    polygon(forehead, [(-.47, .015), (-.24, .54), (.11, 1.06), (.45, 1.10),
                       (.30, .79), (.33, .39), (.49, .015)], (104, 58, 141))
    polygon(forehead, [(-.43, .035), (-.33, .24), (.39, .21), (.48, .035)], (191, 112, 243))
    ellipse(forehead, (0, -.035), (.72, .061), (130, 77, 172))
    ellipse(forehead, (.04, .132), (.083, .075), (119, 217, 255))
    ellipse(forehead, (.04, .132), (.042, .040), (225, 122, 195))
    for x, y in ((-.19, .39), (.15, .64), (.085, .46)):
        polygon(forehead, [(x-.043,y), (x,y+.059), (x+.043,y), (x,y-.059)],
                (124, 228, 255), (124, 228, 255))


def draw_magic(frame, air, now):
    h, w = frame.shape[:2]
    color = SPELLS.get(air.spell, ('', (255, 196, 217)))[1]
    if air.points:
        render_pts = air.points if air.cursor is None else [*air.points, air.cursor]
        trail = np.rint(render_pts).astype(np.int32)
        sw, sh = max(1, w // 4), max(1, h // 4)
        glow = np.zeros((sh, sw, 3), dtype=np.uint8)
        small_trail = np.rint(np.asarray(render_pts) * (sw / w, sh / h)).astype(np.int32)
        cv2.polylines(glow, [small_trail], False, (218, 100, 241), 4, cv2.LINE_AA)
        glow = cv2.GaussianBlur(glow, (0, 0), 2.2)
        glow = cv2.resize(glow, (w, h), interpolation=cv2.INTER_LINEAR)
        cv2.addWeighted(frame, 1., glow, .85, 0, dst=frame)
        cv2.polylines(frame, [trail], False, (255, 215, 251), 3, cv2.LINE_AA)
        if len(air.points) >= 8 and min(np.ptp(air.points, axis=0)) >= 0.05 * min(w, h):
            start = tuple(trail[0])
            cv2.circle(frame, start, 9, (218, 140, 255), 1, cv2.LINE_AA)
            cv2.circle(frame, start, 3, (255, 225, 255), -1, cv2.LINE_AA)
        tip = tuple(trail[-1])
        cv2.circle(frame, tip, 8, (252, 203, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, tip, 3, (255, 255, 255), -1, cv2.LINE_AA)
    elif air.cursor is not None:
        tip = tuple(np.rint(air.cursor).astype(int))
        if air.locked:
            cv2.circle(frame, tip, 7, (170, 140, 195), 1, cv2.LINE_AA)
        else:
            cv2.circle(frame, tip, 7, (252, 203, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, tip, 2, (255, 255, 255), -1, cv2.LINE_AA)

    age = now - air.cast_at
    if 0 <= age < 1.35:
        flash = max(0., 1 - age * 5)
        if flash:
            glow = np.full_like(frame, color)
            cv2.addWeighted(frame, 1 - flash * .15, glow, flash * .15, 0, dst=frame)
        center = np.array((w / 2, h * .43))
        fade = 1 - age / 1.35
        for i in range(32):
            angle = i * 2.39996 + age * .6
            radius = min(w,h) * (.14 + age * (.12 + (i % 5) * .032))
            x, y = np.rint(center + radius * np.array((math.cos(angle), math.sin(angle)))).astype(int)
            span = max(1, round((3 + i % 5) * fade))
            light = tuple(int(c * (.5 + .5 * fade)) for c in color)
            cv2.line(frame, (x-span, y), (x+span, y), light, 1, cv2.LINE_AA)
            cv2.line(frame, (x, y-span), (x, y+span), light, 1, cv2.LINE_AA)
