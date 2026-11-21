"""Synthetic landmark sequences test temporal behavior, not detector accuracy."""
import unittest
import tempfile
import wave
from pathlib import Path
import numpy as np
from lebron.gestures import GestureEngine, ParticleSystem, open_palm
from lebron.audio import AudioPlayer, EXPRESSION_SOUNDS


def hand(x=0.5, y=0.7, opened=True):
    points = np.zeros((21, 3), dtype=np.float32)
    points[:, :2] = (x, y)
    points[0, 1] = y + 0.04
    for column, mcp in enumerate((5, 9, 13, 17)):
        dx = (column - 1.5) * 0.025
        points[mcp, :2] = (x + dx, y)
        points[mcp + 1, :2] = (x + dx, y - 0.04)
        points[mcp + 2, :2] = (x + dx, y - 0.07)
        points[mcp + 3, :2] = (x + dx, y - 0.11 if opened else y + 0.01)
    return points


class GestureTests(unittest.TestCase):
    def test_static_raised_hand_does_not_fire(self):
        engine = GestureEngine()
        for t in np.arange(0, 2, 0.1):
            self.assertEqual(engine.update([hand(y=0.2)], 0.5, float(t)), [])

    def test_shot_requires_low_to_high_and_respects_cooldown(self):
        engine = GestureEngine()
        self.assertEqual(engine.update([hand(y=0.72)], 0.5, 0), [])
        self.assertEqual(engine.update([hand(y=0.4)], 0.5, 0.2), ["shot"])
        self.assertEqual(engine.update([hand(y=0.72)], 0.5, 0.3), [])
        self.assertEqual(engine.update([hand(y=0.4)], 0.5, 0.4), [])

    def test_two_palms_prioritize_chalk_and_order_is_irrelevant(self):
        engine = GestureEngine()
        engine.update([hand(0.25), hand(0.75)], 0.5, 0)
        for t in (0.1, 0.2):
            self.assertEqual(engine.update([hand(0.75, 0.3), hand(0.25, 0.3)], 0.5, t), [])
        self.assertEqual(engine.update([hand(0.75, 0.3), hand(0.25, 0.3)], 0.5, 0.3), ["chalk"])

    def test_tracking_loss_and_long_gap_cancel_sequence(self):
        for loss in ("hands", "face", "gap"):
            engine = GestureEngine()
            engine.update([hand()], 0.5, 0)
            if loss == "hands":
                engine.update([], 0.5, 0.1)
            elif loss == "face":
                engine.update([hand()], None, 0.1)
            self.assertEqual(engine.update([hand(y=0.3)], 0.5, 0.6 if loss == "gap" else 0.2), [])

    def test_closed_palms_do_not_release(self):
        self.assertTrue(open_palm(hand()))
        self.assertFalse(open_palm(hand(opened=False)))
        engine = GestureEngine()
        engine.update([hand()], 0.5, 0)
        self.assertEqual(engine.update([hand(y=0.3, opened=False)], 0.5, 0.2), [])

    def test_two_static_high_palms_do_not_trigger_chalk(self):
        engine = GestureEngine()
        hands = [hand(.25, .36), hand(.75, .36)]
        for t in np.arange(0, 1, .1):
            self.assertEqual(engine.update(hands, .5, float(t)), [])
        self.assertEqual((engine.hand_count, engine.open_count), (2, 2))

    def test_chalk_requires_both_to_rise_and_rearms_from_low_position(self):
        engine = GestureEngine()
        low = [hand(.25,.66), hand(.75,.66)]
        high = [hand(.25,.40), hand(.75,.40)]
        engine.update(low, .5, 0)
        self.assertEqual(engine.update([high[0], low[1]], .5, .1), [])
        engine.update(high, .5, .2)
        self.assertEqual(engine.update(high, .5, .35), ["chalk"])
        for t in np.arange(.5, 3, .2):
            self.assertEqual(engine.update(high, .5, float(t)), [])
        engine.update(low, .5, 3)
        engine.update(high, .5, 3.1)
        self.assertEqual(engine.update(high, .5, 3.3), ["chalk"])

    def test_rise_survives_brief_dropout(self):
        engine = GestureEngine()
        high = [hand(.25,.40), hand(.75,.40)]
        engine.update([hand(.25,.65), hand(.75,.65)], .5, 0)
        self.assertEqual(engine.update([high[0]], .5, .1), [])
        engine.update(high, .5, .2)
        self.assertEqual(engine.update(high, .5, .35), ["chalk"])

    def test_long_dropout_cancels_readiness(self):
        engine = GestureEngine()
        engine.update([hand(.25,.65), hand(.75,.65)], .5, 0)
        engine.update([], .5, .2)
        engine.update([], .5, .4)
        high = [hand(.25,.40), hand(.75,.40)]
        for t in (.5,.65,.8):
            self.assertEqual(engine.update(high, .5, t), [])

    def test_victory_and_fist_have_distinct_effects(self):
        victory = hand(y=.4, opened=False)
        victory[[8,12]] = hand(y=.4)[[8,12]]
        for points, event in ((victory, 'victory'), (hand(y=.4, opened=False), 'fist')):
            engine = GestureEngine()
            events = []
            for t in np.arange(0, 3, .1):
                events.extend(engine.update([points], .5, float(t)))
            self.assertEqual(events, [event])

    def test_recognized_pose_and_shot_with_second_hand_visible(self):
        engine = GestureEngine()
        for t in (0, .1, .25):
            events = engine.update([hand(.25, .75), hand(.75, .4)], .5, t,
                                   [('None', .9), ('Victory', .8)])
        self.assertEqual(events, ['victory'])
        self.assertAlmostEqual(engine.event_origin[0], .75)
        engine.reset()
        engine.update([hand(.25, .64), hand(.75, .74)], .5, 0)
        self.assertEqual(engine.update([hand(.25, .52), hand(.75, .74)], .5, .12), [])
        self.assertEqual(engine.update([hand(.25, .50), hand(.75, .74)], .5, .25), ['shot'])

    def test_particles_are_bounded(self):
        particles = ParticleSystem(max_particles=100, seed=1)
        for t in range(5):
            particles.trigger([hand(), hand(0.7)], t)
        self.assertLessEqual(len(particles._particles), 100)

    def test_audio_generates_valid_wav_and_mute_prevents_playback(self):
        with tempfile.TemporaryDirectory() as temporary:
            player = AudioPlayer(temporary)
            clips = set()
            for event in ("shot", "chalk", *EXPRESSION_SOUNDS):
                with wave.open(str(Path(temporary) / "audio" / f"{event}.wav")) as sound:
                    self.assertEqual(sound.getnchannels(), 1)
                    self.assertEqual(sound.getsampwidth(), 2)
                    self.assertGreater(sound.getnframes(), 1000)
                    clips.add(sound.readframes(sound.getnframes()))
            self.assertEqual(len(clips), 6)
            player.toggle()
            self.assertTrue(player.muted)
            self.assertFalse(player.play("shot"))
            player.close()

    def test_lebron_variant_ignores_v_and_fist_effects(self):
        for posture in ('Victory', 'Closed_Fist'):
            engine = GestureEngine(posture_effects=False)
            for t in (0, .12, .24, .36, .48):
                events = engine.update([hand(y=.4, opened=False)], .5, t, [(posture, .9)])
                self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
