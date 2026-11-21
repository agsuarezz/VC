"""El filtro conserva la webcam y deja la boca libre."""
from pathlib import Path
import unittest

import cv2
import numpy as np
from skimage.data import astronaut

from lebron.visuals import FaceTracker, draw_lebron_filter


class VisualTests(unittest.TestCase):
    def test_beard_preserves_mouth_and_crown_follows_face(self):
        frame = cv2.cvtColor(astronaut(), cv2.COLOR_RGB2BGR)
        model = Path(__file__).resolve().parents[1] / 'models/face_landmarker.task'
        tracker = FaceTracker(model)
        try:
            face = tracker.detect(frame)
        finally:
            tracker.close()
        self.assertIsNotNone(face)
        drawn = draw_lebron_filter(frame.copy(), face)
        lip_x, lip_y = np.rint(face.landmarks[[13, 14], :2].mean(axis=0)*512).astype(int)
        self.assertTrue(np.array_equal(drawn[lip_y, lip_x], frame[lip_y, lip_x]))
        changed = np.any(drawn != frame, axis=2)
        self.assertGreater(changed.sum(), 300)
        self.assertEqual(changed[:, :60].sum(), 0)
        forehead_y = round(float(face.landmarks[10,1]*512))
        self.assertGreater(changed[:forehead_y].sum(), 50)
        self.assertGreater(changed[lip_y+2:lip_y+25].sum(), 50)
        self.assertTrue(np.array_equal(draw_lebron_filter(frame.copy(), None), frame))


if __name__ == '__main__':
    unittest.main()
