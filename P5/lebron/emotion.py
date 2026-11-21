"""Shared HOG representation and trained expression classifier (BGR input)."""
from pathlib import Path

import cv2
import joblib
import numpy as np
from skimage.feature import hog

LABELS = ("happy", "sad", "angry", "surprise", "neutral")
IMAGE_SIZE = (64, 64)
SCHEMA = 1


def blendshape_vector(categories, names=None):
    """Stable category-name order shared by dataset and runtime."""
    values = {c.category_name: c.score for c in categories}
    order = sorted(values) if names is None else list(names)
    if len(order) != 52 or set(order) != set(values):
        raise ValueError('Se esperaban los mismos 52 coeficientes faciales')
    return np.asarray([values[name] for name in order], dtype=np.float32)


def geometry_vector(landmarks, aspect_ratio=1.0):
    """Expression regions aligned to the eye axis; translation/scale removed."""
    pts = np.asarray(landmarks, dtype=np.float32)[:468, :2].copy()
    pts[:,0] *= aspect_ratio
    axis = pts[263] - pts[33]
    scale = max(float(np.linalg.norm(axis)), 1e-6)
    u = axis / scale
    v = np.asarray([-u[1], u[0]])
    center = (pts[33] + pts[263]) / 2
    aligned = (pts - center) @ np.stack((u, v), axis=1) / scale
    # Eyes, eyebrows and lips, excluding contour/hair/iris and depth estimates.
    ids = [33,133,159,145,158,153,263,362,386,374,385,380,
           70,63,105,66,107,336,296,334,293,300,
           61,291,0,17,13,14,78,308,82,312,87,317,40,270,91,321]
    return aligned[ids].reshape(-1).astype(np.float32)


class BlendshapeExtractor:
    def __init__(self, model_path):
        import mediapipe as mp
        self.mp = mp
        self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(
            mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                output_face_blendshapes=True, num_faces=1))
        self.names = None

    def extract(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB if image.ndim == 2 else cv2.COLOR_BGR2RGB)
        result = self.detector.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb))
        if not result.face_blendshapes:
            return None
        self.last_landmarks = np.asarray([(p.x,p.y,p.z) for p in result.face_landmarks[0]], dtype=np.float32)
        self.last_aspect_ratio = image.shape[1]/image.shape[0]
        categories = result.face_blendshapes[0]
        if self.names is None:
            self.names = sorted(c.category_name for c in categories)
        return blendshape_vector(categories, self.names)

    def close(self):
        self.detector.close()


class ExpressionFaceCropper:
    """Identical Haar crop selection for JAFFE preparation and live mode 1."""
    def __init__(self):
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if self.detector.empty():
            raise RuntimeError("No se pudo cargar el detector Haar de OpenCV")

    def detect(self, frame_bgr):
        if frame_bgr is None or frame_bgr.size == 0:
            return None
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
        boxes = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if not len(boxes):
            return None
        return tuple(int(v) for v in max(boxes, key=lambda b: int(b[2]) * int(b[3])))


def normalize_face(image):
    if image is None or image.size == 0:
        raise ValueError("El recorte facial está vacío")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(np.asarray(gray, dtype=np.uint8))


def extract_hog(image):
    """Exactly the same fixed transform at training and webcam inference."""
    return hog(normalize_face(image), orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), block_norm="L2-Hys").astype(np.float32)


class EmotionModel:
    def __init__(self, path):
        # joblib uses pickle: load only artifacts produced locally by train.py.
        artifact = joblib.load(Path(path))
        if artifact.get("schema") not in (SCHEMA, 2):
            raise ValueError("Versión de modelo incompatible; vuelve a entrenar")
        self.pipeline = artifact["pipeline"]
        self.metadata = artifact.get("metadata", {})
        self.feature_type = artifact.get('feature_type', 'hog')
        self.needs_observation = self.feature_type != 'hog'
        self.feature_names = artifact.get('feature_names', [])
        self._extractor = None
        self._model_path = Path(path).parent / 'face_landmarker.task'
        self.classes = tuple(self.pipeline.classes_)
        if set(self.classes) != set(LABELS):
            raise ValueError("El modelo no contiene las cinco clases esperadas")

    def predict(self, face_bgr):
        if self.feature_type != 'hog':
            if self._extractor is None:
                self._extractor = BlendshapeExtractor(self._model_path)
            values = self._extractor.extract(face_bgr)
            if values is None:
                return None, None
            if self.feature_type == 'geometry':
                values = geometry_vector(self._extractor.last_landmarks,self._extractor.last_aspect_ratio)
            elif self.feature_type == 'combined':
                values = np.concatenate((values,geometry_vector(self._extractor.last_landmarks,self._extractor.last_aspect_ratio)))
            features = values.reshape(1, -1)
        else:
            features = extract_hog(face_bgr).reshape(1, -1)
        # Expose the SVM decision, not an uncalibrated pseudo-probability.
        return str(self.pipeline.predict(features)[0]), None

    def predict_observation(self, observation):
        if observation is None or getattr(observation, 'blendshapes', None) is None:
            return None, None
        features = np.asarray(observation.blendshapes, dtype=np.float32)
        if features.shape != (52,) or not np.isfinite(features).all():
            raise ValueError('Coeficientes faciales inválidos')
        size = getattr(observation,'frame_size',None)
        aspect = size[0]/size[1] if size is not None else 1.0
        if self.feature_type == 'geometry':
            features = geometry_vector(observation.landmarks,aspect)
        elif self.feature_type == 'combined':
            features = np.concatenate((features,geometry_vector(observation.landmarks,aspect)))
        elif self.feature_type != 'blendshapes':
            raise ValueError('El modelo HOG necesita predict(recorte)')
        return str(self.pipeline.predict(features.reshape(1, -1))[0]), None

    def close(self):
        if self._extractor is not None:
            self._extractor.close()
            self._extractor = None
