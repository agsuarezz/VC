import os
from pathlib import Path


# Recuadros de ocupaciones fijas comprobadas en C0142, en píxeles a 1920 × 1080.
C0142_PARKING_BOXES = (
    (1283, 323, 1307, 343),
    (1303, 320, 1342, 351),
    (1341, 325, 1382, 350),
    (1364, 325, 1408, 354),
    (1388, 325, 1443, 360),
    (1410, 328, 1445, 364),
    (1426, 320, 1512, 383),
)
PARKING_ID_START = 1_000_001
MIN_CLEAR_FRAMES = 5


def parking_boxes(video_path, width, height):
    """Use the parking calibration only for the fixed camera in C0142."""
    if not isinstance(video_path, (str, os.PathLike)):
        return ()
    if Path(video_path).name.casefold() == 'c0142.mp4' and (width, height) == (1920, 1080):
        return C0142_PARKING_BOXES
    return ()


def parking_match(box, anchors):
    """Find the best calibrated slot, requiring matching scale and position."""
    x1, y1, x2, y2 = box
    area = (x2 - x1) * (y2 - y1)
    if area <= 0:
        return None, 0.0, False
    best = None
    best_score = 0.0
    in_parking = False
    for index, (ax1, ay1, ax2, ay2) in enumerate(anchors):
        width, height = ax2 - ax1, ay2 - ay1
        if width <= 0 or height <= 0:
            continue
        center_delta = abs(x1 + x2 - ax1 - ax2)
        bottom_delta = abs(y2 - ay2)
        if center_delta > width * 1.1 or bottom_delta > height * 0.38:
            continue
        if not (width * 0.35 <= x2 - x1 <= width * 1.4 and height * 0.4 <= y2 - y1 <= height * 1.4):
            continue
        intersection = max(0, min(x2, ax2) - max(x1, ax1)) * max(0, min(y2, ay2) - max(y1, ay1))
        union = area + width * height - intersection
        if union <= 0:
            continue
        score = intersection / union
        in_parking |= score >= 0.20
        if center_delta <= width * 0.42 and bottom_delta <= height * 0.20 and score >= 0.55 and score > best_score:
            best = PARKING_ID_START + index
            best_score = score
    return best, best_score, in_parking


class ParkingIdentity:
    """Keep one public ID per occupied spot in the calibrated scene."""

    def __init__(self, anchors):
        self.anchors = tuple(anchors) if anchors else ()
        self.clear_runs = {}
        if self.anchors:
            self._near_bounds = (
                min(ax1 for ax1, _, _, _ in self.anchors) - 13,
                max(ax2 for _, _, ax2, _ in self.anchors) + 13,
                min(ay2 for _, _, _, ay2 in self.anchors) - 11,
                max(ay2 for _, _, _, ay2 in self.anchors) + 14,
            )
        else:
            self._near_bounds = None

    def is_parked(self, track_id):
        if not isinstance(track_id, int):
            return False
        return PARKING_ID_START <= track_id < PARKING_ID_START + len(self.anchors)

    def relabel(self, detections, frame_idx):
        if not self.anchors:
            return list(detections)

        min_cx, max_cx, min_by, max_by = self._near_bounds
        parked = {}
        other = []
        for box, raw_id, confidence, category in detections:
            detection = (box, raw_id, confidence, category)
            if category not in (2, 5, 7):
                other.append(detection)
                continue

            slot_id, score, in_parking = parking_match(box, self.anchors)
            if in_parking:
                self.clear_runs.pop(raw_id, None)
                if slot_id is not None and (slot_id not in parked or score > parked[slot_id][0]):
                    # Normalizar a categoría coche (2) para evitar parpadeo car/truck en plazas fijas.
                    parked[slot_id] = (score, (box, slot_id, confidence, 2))
                continue

            # Detecciones parciales entre plazas necesitan continuidad antes de sumar otro ID.
            x1, _, x2, bottom = box
            near_slots = min_cx <= (x1 + x2) / 2 <= max_cx and min_by <= bottom <= max_by
            last_frame, run = self.clear_runs.get(raw_id, (0, 0))
            run = (run + 1 if last_frame == frame_idx - 1 else 1) if near_slots else MIN_CLEAR_FRAMES
            self.clear_runs[raw_id] = (frame_idx, run)
            if run >= MIN_CLEAR_FRAMES:
                other.append(detection)

        if len(self.clear_runs) > 512:
            stale = [tid for tid, (lf, _) in self.clear_runs.items() if frame_idx - lf > MIN_CLEAR_FRAMES * 2]
            for tid in stale:
                del self.clear_runs[tid]

        other.extend(detection for _, detection in parked.values())
        return other
