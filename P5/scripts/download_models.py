"""Descarga reproducible de los dos modelos oficiales, sin webcam."""
from pathlib import Path
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URLS = {
    'gesture_recognizer.task': 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task',
    'face_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
    'hand_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
}


def main():
    directory = ROOT / 'models'
    directory.mkdir(exist_ok=True)
    manifest = {}
    for name, url in URLS.items():
        path = directory / name
        if not path.exists():
            temporary = path.with_suffix('.part')
            try:
                with urllib.request.urlopen(url, timeout=90) as source, temporary.open('wb') as target:
                    while block := source.read(1024 * 1024):
                        target.write(block)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        manifest[name] = {'url': url, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        print(name, manifest[name]['sha256'])
    (directory / 'sources.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
