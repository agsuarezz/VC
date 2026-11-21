"""Suavizado y grabación independientes de la webcam."""
from collections import Counter, deque
from pathlib import Path
import cv2


class PredictionSmoother:
    def __init__(self, window=7, min_votes=4):
        self.history = deque(maxlen=window)
        self.min_votes = min_votes
        self.label = None

    def reset(self):
        self.history.clear()
        self.label = None

    def update(self, label, confidence):
        self.history.append((label, confidence))
        winner, votes = Counter(x[0] for x in self.history).most_common(1)[0]
        if votes >= self.min_votes:
            self.label = winner
        values = [c for x, c in self.history if x == self.label and c is not None]
        return self.label, sum(values) / len(values) if values else None


class ExpressionCue:
    """Dispara un sonido una vez por cada cambio de expresión estable."""
    def __init__(self):
        self.label = None

    def reset(self):
        self.label = None

    def update(self, label):
        if label == self.label:
            return None
        self.label = label
        return label if label != 'neutral' else None


class DemoRecorder:
    """Muestreo a FPS fijos: tiempo del vídeo coincide con tiempo real (máx 30s)."""
    def __init__(self, path, size, now, fps=20, seconds=30):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps, self.seconds, self.start = fps, min(30, seconds), now
        self.frames = 0
        self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
        if not self.writer.isOpened():
            self.writer.release()
            raise RuntimeError(f'No se pudo abrir la grabación: {path}')

    def write(self, frame, now):
        if self.writer is None:
            return False
        elapsed = max(0, now - self.start)
        target = min(int(elapsed * self.fps) + 1, int(self.seconds * self.fps))
        while self.frames < target:
            self.writer.write(frame)
            self.frames += 1
        if elapsed >= self.seconds:
            self.close()
            return False
        return True

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
