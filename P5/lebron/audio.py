"""Original synthesized WAV effects; optional asynchronous Windows playback."""
from pathlib import Path
import math
import struct
import wave

EXPRESSION_SOUNDS = {
    'happy': ('Alegria', 'you are my sunshine'),
    'sad': ('Tristeza', 'lebron lebron'),
    'angry': ('Enfado', 'bo bo boom'),
    'surprise': ('Sorpresa', 'taco tuesday'),
}

CUSTOM_CLIPS = {
    'happy': 'sunshine.wav',
    'surprise': 'taco_tuesday.wav',
    'sad': 'lebron_lebron.wav',
    'angry': 'bo_bo_boom.wav',
}


def _write_effect(path, event):
    rate = 22050
    duration = {'shot': .38, 'chalk': .55, 'happy': .58, 'sad': .70, 'angry': .34, 'surprise': .44}[event]
    frames = bytearray()
    for i in range(int(rate * duration)):
        t = i / rate
        phase = t / duration
        envelope = math.sin(math.pi * phase) ** 2
        # Ascending arcade chirp for shot, gentle deterministic airy texture for chalk.
        if event == "shot":
            value = math.sin(2 * math.pi * (390 * t + 600 * t * t))
        elif event == 'chalk':
            value = (math.sin(2 * math.pi * 1577 * t) +
                     math.sin(2 * math.pi * 2341 * t) +
                     math.sin(2 * math.pi * (850 * t + 140 * t * t))) / 3
        elif event in ('happy', 'sad'):
            notes = (523.25, 659.25, 783.99) if event == 'happy' else (392., 311.13, 261.63)
            step = min(int(phase * 3), 2)
            note_phase = phase * 3 - step
            time_in_note = note_phase * duration / 3
            angle = 2 * math.pi * notes[step] * time_in_note
            value = (.76 * math.sin(angle) + .24 * math.sin(2 * angle)) * math.sin(math.pi * note_phase)
        elif event == 'angry':
            value = (.65 * math.sin(2 * math.pi * 92 * t) + .35 * math.sin(2 * math.pi * 177 * t)) * math.exp(-4 * t)
        else:  # Sorpresa: un barrido claramente separado del tiro.
            value = math.sin(2 * math.pi * (210 * t + 2900 * t * t)) * (.7 + .3 * math.sin(2 * math.pi * 18 * t))
        frames.extend(struct.pack("<h", int(11000 * envelope * value)))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(frames)


class AudioPlayer:
    def __init__(self, asset_dir):
        self.muted = False
        self.available = False
        self.error = None
        self._winsound = None
        self._paths = {}
        try:
            directory = Path(asset_dir) / "audio"
            directory.mkdir(parents=True, exist_ok=True)
            for event in ("shot", "chalk", *EXPRESSION_SOUNDS):
                clip_name = CUSTOM_CLIPS.get(event)
                clip = directory / clip_name if clip_name else None
                if clip is not None and clip.is_file():
                    self._paths[event] = clip
                    continue
                path = directory / f"{event}.wav"
                if not path.is_file():
                    _write_effect(path, event)
                self._paths[event] = path
            try:
                import winsound
                self._winsound = winsound
                self.available = True
            except ImportError:
                self.error = "Audio opcional disponible solo en Windows (winsound)."
        except OSError as exc:
            self.error = f"No se pudieron preparar los sonidos: {exc}"

    def play(self, event):
        if self.muted or not self.available or event not in self._paths:
            return False
        try:
            self._winsound.PlaySound(str(self._paths[event]),
                self._winsound.SND_FILENAME | self._winsound.SND_ASYNC | self._winsound.SND_NODEFAULT)
            return True
        except RuntimeError as exc:
            self.error = str(exc)
            return False

    def toggle(self):
        self.muted = not self.muted
        if self.muted:
            self.close()
        return self.muted

    def close(self):
        if self._winsound is not None:
            try:
                self._winsound.PlaySound(None, 0)
            except RuntimeError:
                pass
