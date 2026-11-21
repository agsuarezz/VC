import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from lebron.runtime import DemoRecorder, ExpressionCue, PredictionSmoother


class RuntimeTests(unittest.TestCase):
    def test_expression_plays_once_and_rearms_with_neutral(self):
        cue = ExpressionCue()
        self.assertIsNone(cue.update('neutral'))
        self.assertEqual(cue.update('happy'), 'happy')
        self.assertIsNone(cue.update('happy'))
        self.assertEqual(cue.update('sad'), 'sad')
        self.assertIsNone(cue.update('neutral'))
        self.assertEqual(cue.update('sad'), 'sad')
        self.assertIsNone(cue.update(None))

    def test_flickering_and_loss_reset(self):
        smooth = PredictionSmoother()
        for label in ('happy', 'sad', 'happy', 'sad', 'happy'):
            self.assertIsNone(smooth.update(label, None)[0])
        self.assertEqual(smooth.update('happy', None), ('happy', None))
        self.assertEqual(smooth.update('sad', None)[0], 'happy')
        smooth.reset()
        self.assertIsNone(smooth.update('sad', None)[0])

    def test_recording_caps_duration_even_with_sparse_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'demo.avi'
            frame = np.zeros((120, 160, 3), np.uint8)
            rec = DemoRecorder(path, (160, 120), 10, fps=10)
            self.assertTrue(rec.write(frame, 10))
            self.assertFalse(rec.write(frame, 41))
            self.assertEqual(rec.frames, 300)
            video = cv2.VideoCapture(str(path))
            self.assertTrue(video.isOpened())
            self.assertEqual(video.get(cv2.CAP_PROP_FRAME_COUNT), 300)
            self.assertEqual(video.get(cv2.CAP_PROP_FPS), 10)
            video.release()


if __name__ == '__main__':
    unittest.main()
