"""Tests de efectos de LeBron Lab: blanco y negro, lluvia de tacos y etiqueta mínima."""
import unittest

import numpy as np

from app import draw_predominant_emotion
from lebron.visuals import (
    FaceObservation,
    TacoRain,
    apply_sad_grayscale,
    draw_taco_tuesday,
)


def _synthetic_observation():
    lm = np.full((468, 3), 0.5, dtype=np.float32)
    lm[33, :2] = (0.43, 0.48)
    lm[263, :2] = (0.57, 0.48)
    lm[10, :2] = (0.50, 0.34)
    lm[152, :2] = (0.50, 0.72)
    return FaceObservation(bbox=(380, 180, 200, 220), landmarks=lm, frame_size=(960, 540))


class LebronEffectsTests(unittest.TestCase):
    def test_sad_grayscale_turns_color_frame_monochrome(self):
        frame = np.zeros((180, 240, 3), dtype=np.uint8)
        frame[:, :80] = (20, 40, 220)
        frame[:, 80:160] = (40, 200, 60)
        frame[:, 160:] = (230, 80, 30)
        out = apply_sad_grayscale(frame)
        self.assertEqual(out.shape, (180, 240, 3))
        self.assertTrue(np.array_equal(out[:, :, 0], out[:, :, 1]))
        self.assertTrue(np.array_equal(out[:, :, 1], out[:, :, 2]))
        self.assertGreater(int(out.max()), 50)

    def test_taco_rain_spawns_moves_across_screen_and_clears_when_inactive(self):
        rain = TacoRain(count=25, seed=7)
        base = np.full((540, 960, 3), (40, 50, 60), dtype=np.uint8)

        inactive = rain.draw(base.copy(), 1.0, active=False)
        self.assertTrue(np.array_equal(inactive, base))

        frame1 = rain.draw(base.copy(), 1.0, active=True)
        diff1 = np.any(frame1 != base, axis=2)
        self.assertGreater(diff1.sum(), 5000)
        # Repartidos por toda la anchura (izquierda, centro y derecha)
        self.assertGreater(diff1[:, :320].sum(), 500)
        self.assertGreater(diff1[:, 320:640].sum(), 500)
        self.assertGreater(diff1[:, 640:].sum(), 500)

        frame2 = rain.draw(base.copy(), 1.25, active=True)
        self.assertFalse(np.array_equal(frame1, frame2))

        cleared = rain.draw(base.copy(), 1.3, active=False)
        self.assertTrue(np.array_equal(cleared, base))

    def test_taco_tuesday_draws_above_head_and_stays_in_frame(self):
        base = np.full((540, 960, 3), (30, 35, 45), dtype=np.uint8)
        obs = _synthetic_observation()
        drawn = draw_taco_tuesday(base.copy(), obs, now=0.5)
        changed = np.any(drawn != base, axis=2)
        ys, xs = np.where(changed)
        self.assertGreater(len(ys), 500)
        forehead_y = int(obs.landmarks[10, 1] * 540)
        self.assertLess(ys.mean(), forehead_y)
        self.assertGreaterEqual(int(xs.min()), 0)
        self.assertLess(int(xs.max()), 960)

        # Tambien funciona con cara pegada al borde superior o sin observacion
        top_obs = _synthetic_observation()
        top_obs.landmarks[:, 1] *= 0.25
        top_obs.bbox = (380, 10, 200, 120)
        clamped = draw_taco_tuesday(base.copy(), top_obs, now=0.0)
        self.assertGreater(np.any(clamped != base, axis=2).sum(), 400)
        no_face = draw_taco_tuesday(base.copy(), None, now=0.0)
        self.assertGreater(np.any(no_face != base, axis=2).sum(), 400)

    def test_predominant_emotion_only_draws_corner_label_when_active(self):
        names = {'happy': 'Alegria', 'sad': 'Tristeza', 'surprise': 'Sorpresa', 'neutral': 'Neutral'}
        base = np.full((540, 960, 3), (50, 60, 70), dtype=np.uint8)

        empty = base.copy()
        draw_predominant_emotion(empty, None, names)
        self.assertTrue(np.array_equal(empty, base))

        labeled = base.copy()
        draw_predominant_emotion(labeled, 'surprise', names)
        changed = np.any(labeled != base, axis=2)
        self.assertGreater(changed.sum(), 100)
        # Sin paneles abajo ni a la derecha
        self.assertEqual(changed[100:, :].sum(), 0)
        self.assertEqual(changed[:, 320:].sum(), 0)


if __name__ == '__main__':
    unittest.main()
