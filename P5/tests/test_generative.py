"""Pruebas unitarias e integración para el filtro de arte generativo facial."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from generative_face import GenerativeFaceFilter, MODE_NAMES, SignalExtractor
from lebron.visuals import FaceObservation


def _synthetic_observation(
    mouth_open_gap: float = 0.01,
    blendshape_map: dict[str, float] | None = None,
    shift_x: float = 0.0,
) -> FaceObservation:
    """Genera 478 landmarks sintéticos realistas en coordenadas normalizadas [0, 1]."""
    rng = np.random.default_rng(7)
    lm = np.zeros((478, 3), dtype=np.float32)
    angles = np.linspace(0.0, 2.0 * np.pi, 478, endpoint=False)
    lm[:, 0] = 0.50 + shift_x + 0.16 * np.cos(angles) + rng.normal(0, 0.002, 478)
    lm[:, 1] = 0.50 + 0.22 * np.sin(angles) + rng.normal(0, 0.002, 478)
    lm[:, 2] = -0.05 * np.cos(angles)

    # Puntos clave usados por SignalExtractor y los renderizadores.
    lm[1] = (0.50 + shift_x, 0.52, -0.08)   # Punta de la nariz
    lm[10] = (0.50 + shift_x, 0.28, -0.01)  # Frente
    lm[152] = (0.50 + shift_x, 0.72, -0.01) # Mentón
    lm[33] = (0.40 + shift_x, 0.45, -0.03)  # Ojo izq exterior
    lm[133] = (0.46 + shift_x, 0.45, -0.04) # Ojo izq interior
    lm[159] = (0.43 + shift_x, 0.44, -0.04)
    lm[145] = (0.43 + shift_x, 0.46, -0.04)
    lm[263] = (0.60 + shift_x, 0.45, -0.03) # Ojo der exterior
    lm[362] = (0.54 + shift_x, 0.45, -0.04) # Ojo der interior
    lm[386] = (0.57 + shift_x, 0.44, -0.04)
    lm[374] = (0.57 + shift_x, 0.46, -0.04)
    lm[105] = (0.43 + shift_x, 0.39, -0.03) # Ceja izq
    lm[334] = (0.57 + shift_x, 0.39, -0.03) # Ceja der
    lm[78] = (0.44 + shift_x, 0.61, -0.04)  # Comisura izq
    lm[308] = (0.56 + shift_x, 0.61, -0.04) # Comisura der
    lm[13] = (0.50 + shift_x, 0.60 - mouth_open_gap / 2.0, -0.05)
    lm[14] = (0.50 + shift_x, 0.60 + mouth_open_gap / 2.0, -0.05)
    lm[468] = (0.43 + shift_x, 0.45, -0.04) # Iris izq
    lm[473] = (0.57 + shift_x, 0.45, -0.04) # Iris der

    return FaceObservation(
        bbox=(160, 120, 200, 240),
        landmarks=lm,
        mouth_open=mouth_open_gap > 0.04,
        blendshapes=None,
        frame_size=(640, 480),
        blendshape_map=blendshape_map,
    )


class GenerativeFilterTests(unittest.TestCase):
    def test_wink_triggers_only_on_unilateral_blink_with_cooldown(self):
        extractor = SignalExtractor(wink_hold=0.14, wink_cooldown=0.90)
        shape = (480, 640, 3)

        # Parpadeo bilateral no debe disparar guiño.
        both_closed = _synthetic_observation(blendshape_map={"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.88})
        for t in (0.0, 0.10, 0.25):
            sig = extractor.extract(both_closed, shape, t)
            self.assertIsNotNone(sig)
            self.assertFalse(sig.wink_triggered)

        # Guiño unilateral mantenido >= 0.14s sí dispara una sola vez.
        left_wink = _synthetic_observation(blendshape_map={"eyeBlinkLeft": 0.85, "eyeBlinkRight": 0.05})
        self.assertFalse(extractor.extract(left_wink, shape, 0.40).wink_triggered)
        self.assertTrue(extractor.extract(left_wink, shape, 0.56).wink_triggered)
        # Mantener el mismo ojo cerrado no vuelve a disparar.
        self.assertFalse(extractor.extract(left_wink, shape, 0.80).wink_triggered)

        # Reabrir ojos y volver a guiñar dentro del cooldown (0.90s desde 0.56 -> 1.46s) no dispara.
        both_open = _synthetic_observation(blendshape_map={"eyeBlinkLeft": 0.05, "eyeBlinkRight": 0.05})
        self.assertFalse(extractor.extract(both_open, shape, 0.90).wink_triggered)
        self.assertFalse(extractor.extract(left_wink, shape, 1.00).wink_triggered)
        self.assertFalse(extractor.extract(left_wink, shape, 1.20).wink_triggered)

        # Pasado el cooldown, un nuevo guiño vuelve a disparar.
        self.assertFalse(extractor.extract(both_open, shape, 1.40).wink_triggered)
        self.assertFalse(extractor.extract(left_wink, shape, 1.50).wink_triggered)
        self.assertTrue(extractor.extract(left_wink, shape, 1.66).wink_triggered)

    def test_mouth_waves_and_brow_sparks_spawn_and_expire(self):
        art = GenerativeFaceFilter(mode=0, dark_background=True, seed=1)
        frame = np.full((360, 480, 3), 80, dtype=np.uint8)
        obs_open = _synthetic_observation(
            mouth_open_gap=0.09,
            blendshape_map={"jawOpen": 0.85, "browInnerUp": 0.85, "mouthSmileLeft": 0.5, "mouthSmileRight": 0.5},
        )
        # Varios frames seguidos para superar el suavizado y generar ondas y chispas.
        for t in (0.0, 0.12, 0.25):
            art.render(frame, obs_open, t)
        self.assertGreater(len(art._waves), 0)
        self.assertGreater(len(art._sparks), 0)

        # Al cerrar boca/relajar cejas y avanzar el tiempo, las partículas expiran.
        obs_rest = _synthetic_observation(
            mouth_open_gap=0.005,
            blendshape_map={"jawOpen": 0.0, "browInnerUp": 0.0, "mouthSmileLeft": 0.0, "mouthSmileRight": 0.0},
        )
        art.render(frame, obs_rest, 4.0)
        self.assertEqual(len(art._waves), 0)
        self.assertEqual(len(art._sparks), 0)

    def test_all_modes_render_synthetic_face_and_handle_none(self):
        frame = np.full((360, 480, 3), 100, dtype=np.uint8)
        obs_a = _synthetic_observation(mouth_open_gap=0.07, blendshape_map={"jawOpen": 0.65, "browInnerUp": 0.5})
        obs_b = _synthetic_observation(
            mouth_open_gap=0.08,
            blendshape_map={"jawOpen": 0.75, "browInnerUp": 0.6},
            shift_x=0.04,
        )
        for mode in range(len(MODE_NAMES)):
            for dark in (False, True):
                art = GenerativeFaceFilter(mode=mode, dark_background=dark, seed=3)
                empty = art.render(frame, None, 0.0)
                self.assertEqual(empty.shape, frame.shape)
                self.assertEqual(empty.dtype, np.uint8)

                out1 = art.render(frame, obs_a, 0.1)
                out2 = art.render(frame, obs_b, 0.25)
                self.assertEqual(out2.shape, frame.shape)
                # Debe haber contenido dibujado distinto del lienzo vacío.
                diff = np.any(out2 != empty, axis=2)
                self.assertGreater(int(diff.sum()), 400)
                self.assertFalse(np.array_equal(out1, out2))

    def test_wink_switches_mode_inside_filter(self):
        art = GenerativeFaceFilter(mode=0, dark_background=True, seed=2)
        frame = np.zeros((320, 420, 3), dtype=np.uint8)
        wink = _synthetic_observation(blendshape_map={"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.05})
        art.render(frame, wink, 0.0)
        self.assertEqual(art.mode, 0)
        art.render(frame, wink, 0.20)
        self.assertEqual(art.mode, 1)

    @unittest.skipUnless(
        importlib.util.find_spec("mediapipe") is not None and importlib.util.find_spec("skimage") is not None,
        "Requiere mediapipe y scikit-image instalados",
    )
    def test_real_face_tracker_populates_blendshape_map_and_renders(self):
        from skimage.data import astronaut
        from lebron.visuals import FaceTracker

        frame = cv2.cvtColor(astronaut(), cv2.COLOR_RGB2BGR)
        model = Path(__file__).resolve().parents[1] / "models/face_landmarker.task"
        tracker = FaceTracker(model)
        try:
            obs = tracker.detect(frame)
        finally:
            tracker.close()
        self.assertIsNotNone(obs)
        self.assertIsNotNone(obs.blendshape_map)
        self.assertIn("jawOpen", obs.blendshape_map)
        self.assertIn("eyeBlinkLeft", obs.blendshape_map)

        for mode in range(len(MODE_NAMES)):
            art = GenerativeFaceFilter(mode=mode, dark_background=True, seed=5)
            empty = art.render(frame, None, 0.0)
            rendered = art.render(frame, obs, 0.1)
            self.assertGreater(int(np.any(rendered != empty, axis=2).sum()), 500)

    def test_headless_video_pipeline_writes_avi(self):
        """Comprueba que app_arte.py procesa un vídeo temporal en modo headless."""
        import subprocess
        import sys

        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            in_video = tmp_dir / "in.avi"
            out_video = tmp_dir / "out.avi"
            writer = cv2.VideoWriter(str(in_video), cv2.VideoWriter_fourcc(*"MJPG"), 15, (320, 240))
            for i in range(6):
                img = np.full((240, 320, 3), 40 + i * 10, dtype=np.uint8)
                writer.write(img)
            writer.release()

            proc = subprocess.run(
                [
                    sys.executable,
                    str(root / "app_arte.py"),
                    "--video",
                    str(in_video),
                    "--headless",
                    "--max-frames",
                    "5",
                    "--record",
                    str(out_video),
                ],
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(out_video.is_file())
            self.assertGreater(out_video.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
