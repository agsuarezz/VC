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


def pointing_tip(hand, size, tracing):
    """Allow a slightly bent index and track finger bends in three dimensions."""
    w, h = size
    points = hand * (w, h, w)
    palm = np.linalg.norm(points[9] - points[0])
    fingers = points[np.arange(5, 21).reshape(4, 4)]
    segments = np.linalg.norm(np.diff(fingers, axis=1), axis=2).sum(axis=1)
    reach = np.linalg.norm(fingers[:, 3] - fingers[:, 0], axis=1)
    straightness = reach / np.maximum(segments, 1e-5)
    index_above_knuckle = np.linalg.norm(points[8] - points[0]) > .22 * palm + np.linalg.norm(points[5] - points[0])
    raised = straightness[1:] > .85
    if (palm > 3 and index_above_knuckle and straightness[0] > (.69 if tracing else .78)
            and np.count_nonzero(raised) <= (1 if tracing else 0)):
        return points[8, :2]
    return None


class AirSpell:
    def __init__(self):
        self.points = []
        self.spell = None
        self.cast_at = -100.
        self.last_seen = -100.
        self.locked = False

    def update(self, hands, size, now):
        # Keep the hand closest to the current stroke when both index fingers are up.
        tips = [tip for hand in hands
                if (tip := pointing_tip(hand, size, tracing=bool(self.points))) is not None]
        if tips:
            if self.locked:
                self.last_seen = now
                return
            if not self.points:
                self.last_seen = now
                self.points.append(tips[0])
                return
            tip = min(tips, key=lambda xy: np.linalg.norm(xy - self.points[-1]))
            distance = np.linalg.norm(tip - self.points[-1])
            # Ignore isolated teleports from the tracker without cutting the stroke.
            if distance > .27 * max(size):
                if now - self.last_seen > .32:
                    self.finish(now)
                return
            self.last_seen = now
            weight = .55 + .39 * min(1., distance / (.04 * min(size)))
            tip = weight * tip + (1 - weight) * self.points[-1]
            if distance >= max(1.5, min(size) * .003):
                self.points.append(tip)
            span = np.ptp(self.points, axis=0)
            closed = np.linalg.norm(tip - self.points[0]) < .12 * max(span)
            traced_loop = len(self.points) >= 13 and min(span) >= .083 * min(size) and closed
            if traced_loop or len(self.points) >= 600:
                self.finish(now)
        elif now - self.last_seen >= .32:
            if self.points:
                self.finish(now)
            self.locked = False

    def finish(self, now):
        if len(self.points) >= 4 and max(np.ptp(self.points, axis=0)) >= 20:
            self.spell = classify_shape(self.points)
            self.cast_at = now
        self.points.clear()
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
        aura = np.zeros_like(frame)
        tint = SPELLS[spell][1]
        center = tuple(np.rint((forehead + points[152]) / 2).astype(int))
        tilt = math.degrees(math.atan2(right[1], right[0]))
        cv2.ellipse(aura, center, (round(scale * .91), round(scale * 1.12)), tilt,
                    0, 360, tint, max(4, round(scale * .085)), cv2.LINE_AA)
        aura = cv2.GaussianBlur(aura, (0, 0), max(3., scale * .085))
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
        trail = np.rint(air.points).astype(np.int32)
        glow = np.zeros_like(frame)
        cv2.polylines(glow, [trail], False, (218, 100, 241), 13, cv2.LINE_AA)
        cv2.addWeighted(frame, 1., cv2.GaussianBlur(glow, (0, 0), 7), .85, 0, dst=frame)
        cv2.polylines(frame, [trail], False, (255, 215, 251), 3, cv2.LINE_AA)
        cv2.circle(frame, tuple(trail[-1]), 8, (252, 203, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, tuple(trail[-1]), 3, (255, 255, 255), -1, cv2.LINE_AA)

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
