"""Recognize closed air gestures and draw the wizard's face effects."""
import math

import cv2
import numpy as np


SPELLS = {
    'circle': ('Conejo lunar', (255, 205, 162)),
    'square': ('Oso del bosque', (145, 218, 255)),
    'triangle': ('Zorro de fuego', (105, 167, 255)),
    'failed': ('Hechizo fallido', (250, 126, 255)),
}


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


def draw_costume(frame, observation, spell):
    points = observation.landmarks[:, :2] * frame.shape[1::-1]
    axis = points[263] - points[33]
    eye_width = float(np.linalg.norm(axis))
    if eye_width < 8 or not np.isfinite(points).all():
        return
    right = axis / eye_width
    up = np.array((right[1], -right[0]))
    scale = max(eye_width, .53 * np.linalg.norm(points[152] - points[10]))
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
    # Animal ears and facial accents follow the pose after the transformation.
    ears = {'circle': ('rabbit', 'rabbit'), 'square': ('bear', 'bear'),
            'triangle': ('fox', 'fox'), 'failed': ('rabbit', 'fox')}
    if spell in ears:
        fh, fw = frame.shape[:2]
        sw, sh = max(1, fw // 4), max(1, fh // 4)
        aura = np.zeros((sh, sw, 3), dtype=np.uint8)
        tint = SPELLS[spell][1]
        center_full = (forehead + points[152]) / 2
        center = (round(float(center_full[0]) * sw / fw), round(float(center_full[1]) * sh / fh))
        tilt = math.degrees(math.atan2(right[1], right[0]))
        rx = max(1, round(scale * .91 * sw / fw))
        ry = max(1, round(scale * 1.12 * sh / fh))
        cv2.ellipse(aura, center, (rx, ry), tilt,
                    0, 360, tint, max(2, round(scale * .085 * sw / fw)), cv2.LINE_AA)
        aura = cv2.GaussianBlur(aura, (0, 0), max(1.5, scale * .085 * sw / fw))
        aura = cv2.resize(aura, (fw, fh), interpolation=cv2.INTER_LINEAR)
        cv2.addWeighted(frame, 1., aura, .78, 0, dst=frame)
        for side, animal in zip((-1, 1), ears[spell]):
            anchor = local(forehead, (side * .66, -.07))
            if animal == 'rabbit':
                ellipse(anchor, (side * .17, .43), (.16, .49), (242, 224, 247), side * 19)
                ellipse(anchor, (side * .19, .48), (.075, .34), (190, 129, 243), side * 19)
            elif animal == 'bear':
                ellipse(anchor, (side * .13, .18), (.25, .24), (89, 138, 191))
                ellipse(anchor, (side * .13, .18), (.14, .13), (141, 193, 226))
            else:
                polygon(anchor, [(side*.03, -.08), (side*.19, .65), (side*.43, .03)],
                        (80, 141, 246))
                polygon(anchor, [(side*.14, .04), (side*.21, .47), (side*.33, .08)],
                        (176, 217, 255))

        nose = points[2]
        for side in (-1, 1):
            cheek = .52 * points[291 if side == 1 else 61] + .48 * points[454 if side == 1 else 234]
            cheek_style = ('circle' if side < 0 else 'triangle') if spell == 'failed' else spell
            if cheek_style == 'circle':
                ellipse(cheek, (0, .05), (.16, .093), (187, 139, 240))
                for dy in (-.065, 0, .065):
                    a, b = local(cheek, [(side * .04, dy), (side * .25, 1.7 * dy)])
                    cv2.line(frame, tuple(a), tuple(b), (238, 226, 252), 2, cv2.LINE_AA)
            elif cheek_style == 'triangle':
                polygon(cheek, [(side * .22, .17), (side * -.16, .08), (side * .15, -.065)],
                        (167, 222, 255))
            else:
                for dx in (-.065, .02, .105):
                    ellipse(cheek, (dx, abs(dx) / 2), (.024, .024), (105, 158, 221))
        polygon(nose, [(-.115,.035), (0,.075), (.115,.035), (.07,-.04), (0,-.075), (-.07,-.04)],
                (124, 93, 231) if spell == 'circle' else (73, 40, 89))
        ellipse(nose, (-.034, .025), (.031, .016), (214, 180, 247))
        for side in (-1, 1):
            for dx, dy, radius in ((.87, .03, .027), (.98, -.14, .018), (.81, -.27, .013)):
                ellipse(forehead, (side * dx, dy), (radius, radius), tint)
        return

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
