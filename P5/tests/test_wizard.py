"""Unit tests for wizard finger tracking, smoothing, and spell loop detection."""
import math
import unittest

import numpy as np

from lebron.visuals import FaceObservation
from wizard_filter import (
    AirSpell,
    LEFT_BROW,
    LEFT_EYE_RING,
    OneEuroFilter,
    RIGHT_BROW,
    RIGHT_EYE_RING,
    draw_costume,
    draw_magic,
    pointing_tip,
)


def make_hand(tip_x=0.5, tip_y=0.4, extended=(True, False, False, False),
              fold_at_knuckle=False, index_bend=0.0):
    """Build synthetic 21x3 MediaPipe hand landmarks with tip of index at (tip_x, tip_y)."""
    pts = np.zeros((21, 3), dtype=np.float32)
    # Place index MCP below index tip, wrist below MCP.
    palm = 0.14
    seg = 0.045
    index_mcp = np.array((tip_x, tip_y + 3 * seg, 0.0), dtype=np.float32)
    wrist = np.array((tip_x + 0.03, index_mcp[1] + palm, 0.0), dtype=np.float32)
    pts[0] = wrist
    # Thumb out to the side (should not interfere with pointing).
    pts[1] = wrist + (-0.04, -0.04, 0.0)
    pts[2] = wrist + (-0.07, -0.08, 0.0)
    pts[3] = wrist + (-0.09, -0.11, 0.0)
    pts[4] = wrist + (-0.11, -0.13, 0.0)

    for idx, (base, is_up) in enumerate(zip((5, 9, 13, 17), extended)):
        mcp = index_mcp + np.array((idx * 0.028, idx * 0.005, 0.0), dtype=np.float32)
        pts[base] = mcp
        if is_up:
            bend = index_bend if idx == 0 else 0.0
            pts[base + 1] = mcp + np.array((bend, -seg, 0.0), dtype=np.float32)
            pts[base + 2] = mcp + np.array((bend * 1.2, -2 * seg, 0.0), dtype=np.float32)
            pts[base + 3] = mcp + np.array((0.0, -3 * seg, 0.0), dtype=np.float32)
        elif fold_at_knuckle:
            # Straight phalanges folded downward at the MCP joint toward the wrist.
            pts[base + 1] = mcp + np.array((0.0, 0.025, 0.01), dtype=np.float32)
            pts[base + 2] = mcp + np.array((0.0, 0.050, 0.02), dtype=np.float32)
            pts[base + 3] = mcp + np.array((0.0, 0.075, 0.03), dtype=np.float32)
        else:
            # Curled into the palm.
            pts[base + 1] = mcp + np.array((0.0, -0.02, 0.0), dtype=np.float32)
            pts[base + 2] = mcp + np.array((0.0, 0.01, 0.01), dtype=np.float32)
            pts[base + 3] = mcp + np.array((0.0, 0.035, 0.015), dtype=np.float32)
    return pts


class WizardTrackingTests(unittest.TestCase):
    def test_pointing_tip_accepts_single_index_and_knuckle_folded_others(self):
        size = (640, 480)
        tip = pointing_tip(make_hand(0.4, 0.35), size, tracing=False)
        self.assertIsNotNone(tip)
        self.assertAlmostEqual(float(tip[0]), 0.4 * 640, delta=1.0)
        self.assertAlmostEqual(float(tip[1]), 0.35 * 480, delta=1.0)

        folded = make_hand(0.5, 0.35, fold_at_knuckle=True)
        self.assertIsNotNone(pointing_tip(folded, size, tracing=False))

        bent = make_hand(0.5, 0.35, index_bend=0.032)
        self.assertIsNotNone(pointing_tip(bent, size, tracing=True))

    def test_pointing_tip_rejects_open_palm_fist_and_victory_start(self):
        size = (640, 480)
        open_palm = make_hand(extended=(True, True, True, True))
        fist = make_hand(extended=(False, False, False, False))
        victory = make_hand(extended=(True, True, False, False))
        self.assertIsNone(pointing_tip(open_palm, size, tracing=False))
        self.assertIsNone(pointing_tip(open_palm, size, tracing=True))
        self.assertIsNone(pointing_tip(fist, size, tracing=False))
        self.assertIsNone(pointing_tip(fist, size, tracing=True))
        self.assertIsNone(pointing_tip(victory, size, tracing=False))

    def test_one_euro_filter_smooths_jitter_and_tracks_fast_motion(self):
        filt = OneEuroFilter()
        rng = np.random.default_rng(7)
        raw_pts, smooth_pts = [], []
        for i in range(40):
            noisy = np.array((300.0, 240.0)) + rng.normal(0, 3.0, size=2)
            raw_pts.append(noisy)
            smooth_pts.append(filt.update(noisy, i / 30.0, scale=480))
        self.assertLess(np.std(smooth_pts[10:], axis=0).mean(),
                        0.65 * np.std(raw_pts[10:], axis=0).mean())

        # Fast ramp should stay close to target without excessive lag.
        t = 40 / 30.0
        for step in range(1, 16):
            target = np.array((300.0 + step * 20.0, 240.0))
            out = filt.update(target, t + step / 30.0, scale=480)
        self.assertLess(abs(out[0] - target[0]), 25.0)

    def test_air_spell_ignores_single_frame_glitch_and_tracks_circle_with_dropout(self):
        air = AirSpell()
        size = (640, 480)
        # Single isolated frame should not start a stroke.
        air.update([make_hand(0.2, 0.2)], size, 0.0)
        self.assertEqual(len(air.points), 0)
        air.update([], size, 0.25)
        self.assertEqual(len(air.points), 0)

        # Trace a circle with a brief 3-frame dropout and 1 teleport glitch mid-way.
        t = 1.0
        steps = 48
        for i in range(steps + 4):
            angle = 2 * math.pi * (i / steps)
            x = 0.5 + 0.16 * math.cos(angle)
            y = 0.45 + 0.20 * math.sin(angle)
            t += 1.0 / 30.0
            if i in (15, 16, 17):
                air.update([], size, t)
                continue
            if i == 25:
                air.update([make_hand(0.05, 0.05)], size, t)
                continue
            air.update([make_hand(x, y)], size, t)
            if air.spell is not None:
                break
        self.assertEqual(air.spell, 'circle')
        self.assertTrue(air.locked)

        # Lowering finger for >= 0.22s unlocks cleanly.
        air.update([], size, t + 0.10)
        self.assertTrue(air.locked)
        air.update([], size, t + 0.26)
        self.assertFalse(air.locked)

    def test_air_spell_trims_lead_in_tail_on_triangle(self):
        air = AirSpell()
        size = (640, 480)
        t = 0.0
        # Short lead-in before triangle vertices.
        lead = [(0.35 + 0.02 * i, 0.60 - 0.01 * i) for i in range(6)]
        v0, v1, v2 = (0.50, 0.25), (0.68, 0.62), (0.32, 0.62)
        path = lead
        for a, b in ((lead[-1], v0), (v0, v1), (v1, v2), (v2, v0)):
            for k in range(1, 15):
                frac = k / 14.0
                path.append((a[0] + (b[0] - a[0]) * frac, a[1] + (b[1] - a[1]) * frac))
        for x, y in path:
            t += 1.0 / 30.0
            air.update([make_hand(x, y)], size, t)
            if air.spell is not None:
                break
        self.assertEqual(air.spell, 'triangle')

    def test_air_spell_square_and_open_stroke_fail_and_fast_move_recovery(self):
        air = AirSpell()
        size = (640, 480)
        t = 0.0
        verts = [(0.35, 0.28), (0.65, 0.28), (0.65, 0.68), (0.35, 0.68), (0.35, 0.28)]
        for a, b in zip(verts, verts[1:]):
            for k in range(1, 13):
                frac = k / 12.0
                t += 1.0 / 30.0
                air.update([make_hand(a[0] + (b[0] - a[0]) * frac,
                                      a[1] + (b[1] - a[1]) * frac)], size, t)
                if air.spell is not None:
                    break
        self.assertEqual(air.spell, 'square')

        # Unlock and trace an open line -> should finish as 'failed' after grace timeout.
        air.update([], size, t + 0.30)
        self.assertFalse(air.locked)
        t += 0.35
        for k in range(10):
            t += 1.0 / 30.0
            air.update([make_hand(0.25 + 0.04 * k, 0.4)], size, t)
        self.assertGreater(len(air.points), 4)
        # Fast jump confirmed across 2 consecutive frames should be accepted, not dropped.
        t += 1.0 / 30.0
        air.update([make_hand(0.85, 0.4)], size, t)
        t += 1.0 / 30.0
        air.update([make_hand(0.86, 0.4)], size, t)
        self.assertGreater(air.cursor[0], 0.6 * size[0])
        # Lower finger for >= 0.45s to finish open stroke as failed.
        air.update([], size, t + 0.50)
        self.assertEqual(air.spell, 'failed')

    def test_draw_magic_renders_for_all_states(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        air = AirSpell()
        air.update([make_hand(0.5, 0.4)], (640, 480), 0.0)
        draw_magic(frame, air, 0.0)
        air.update([make_hand(0.52, 0.4)], (640, 480), 0.033)
        draw_magic(frame, air, 0.033)
        self.assertGreater(int(frame.sum()), 0)

    def test_draw_costume_renders_all_powerups_and_animates(self):
        lm = np.full((478, 3), 0.5, dtype=np.float32)
        lm[10, :2] = (0.50, 0.28)
        lm[152, :2] = (0.50, 0.72)
        lm[234, :2] = (0.37, 0.50)
        lm[454, :2] = (0.63, 0.50)
        lm[61, :2] = (0.45, 0.62)
        lm[291, :2] = (0.55, 0.62)
        lm[33, :2] = (0.41, 0.46)
        lm[263, :2] = (0.59, 0.46)
        lm[468, :2] = (0.44, 0.46)
        lm[473, :2] = (0.56, 0.46)
        for idx, ang in zip(LEFT_EYE_RING, np.linspace(0, 2 * math.pi, len(LEFT_EYE_RING), endpoint=False)):
            lm[idx, :2] = (0.44 + 0.032 * math.cos(ang), 0.46 + 0.018 * math.sin(ang))
        for idx, ang in zip(RIGHT_EYE_RING, np.linspace(0, 2 * math.pi, len(RIGHT_EYE_RING), endpoint=False)):
            lm[idx, :2] = (0.56 + 0.032 * math.cos(ang), 0.46 + 0.018 * math.sin(ang))
        for i, idx in enumerate(LEFT_BROW):
            lm[idx, :2] = (0.40 + 0.02 * i, 0.41 - 0.008 * math.sin(i * 0.8))
        for i, idx in enumerate(RIGHT_BROW):
            lm[idx, :2] = (0.60 - 0.02 * i, 0.41 - 0.008 * math.sin(i * 0.8))
        obs = FaceObservation(bbox=(240, 120, 180, 220), landmarks=lm, frame_size=(640, 480))
        base = np.full((480, 640, 3), (65, 70, 80), dtype=np.uint8)

        for spell in (None, 'circle', 'square', 'triangle', 'failed'):
            f1 = base.copy()
            f2 = base.copy()
            draw_costume(f1, obs, spell, now=0.2)
            draw_costume(f2, obs, spell, now=0.9)
            self.assertGreater(int(np.any(f1 != base, axis=2).sum()), 300, msg=f'spell={spell}')
            if spell is not None:
                self.assertFalse(np.array_equal(f1, f2), msg=f'spell={spell} should animate')


if __name__ == '__main__':
    unittest.main()
