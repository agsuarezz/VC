"""Checks for representation consistency and train artifact integration."""
from pathlib import Path
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from lebron.emotion import EmotionModel, ExpressionFaceCropper, extract_hog, geometry_vector


class EmotionTests(unittest.TestCase):
    def test_bgr_and_gray_representation_agree(self):
        gray = np.random.default_rng(9).integers(0, 256, (83, 71), dtype=np.uint8)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        np.testing.assert_allclose(extract_hog(gray), extract_hog(bgr))
        self.assertEqual(extract_hog(gray).shape, (1764,))
        self.assertTrue(np.isfinite(extract_hog(gray)).all())

    def test_empty_and_absent_faces(self):
        with self.assertRaises(ValueError):
            extract_hog(np.empty((0, 0), dtype=np.uint8))
        cropper = ExpressionFaceCropper()
        self.assertIsNone(cropper.detect(None))
        self.assertIsNone(cropper.detect(np.zeros((240, 320, 3), dtype=np.uint8)))

    def test_delivered_model_predicts_and_has_train_only_scaler(self):
        model_path = Path(__file__).resolve().parents[1] / "models" / "emotion.joblib"
        model = EmotionModel(model_path)
        landmarks = np.random.default_rng(10).uniform(.2,.8,(478,3)).astype(np.float32)
        observation = SimpleNamespace(landmarks=landmarks, blendshapes=np.zeros(52,dtype=np.float32))
        label, confidence = model.predict_observation(observation)
        self.assertIn(label, model.classes)
        self.assertIsNone(confidence)
        expected_train = model.metadata['training_samples']
        self.assertEqual(int(model.pipeline[0].n_samples_seen_), expected_train)
        self.assertEqual(set(model.metadata['training_subjects']) & set(model.metadata['test_subjects']),set())
        self.assertEqual(model.predict_observation(None),(None,None))

    def test_geometry_invariant_to_translation_rotation_and_scale(self):
        landmarks = np.random.default_rng(10).uniform(.2,.8,(478,3)).astype(np.float32)
        angle=.3
        rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        transformed=landmarks.copy()
        transformed[:,:2] = landmarks[:,:2] @ rotation * 1.4 + .2
        np.testing.assert_allclose(geometry_vector(landmarks),geometry_vector(transformed),rtol=1e-5,atol=1e-5)
        self.assertEqual(geometry_vector(landmarks).shape,(76,))
        widescreen=landmarks.copy()
        widescreen[:,0] /= 16/9
        np.testing.assert_allclose(geometry_vector(landmarks),geometry_vector(widescreen,16/9),rtol=1e-5,atol=1e-5)


if __name__ == "__main__":
    unittest.main()
