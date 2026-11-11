import unittest
from collections import Counter

from robojepa_worlds.rgb_schedule import FrameSchedule, lr_scale


class FrameScheduleTest(unittest.TestCase):
    def test_full_coverage_no_padding_or_dropped_tail(self):
        counts = [50, 50, 50, 50, 13]
        schedule = FrameSchedule(counts)
        for epoch in [0, 1]:
            plan = schedule._plan(epoch)
            seen = Counter((clip, frame) for step in plan for clip, frames in step for frame in frames)
            self.assertEqual(seen, Counter((i, f) for i, n in enumerate(counts) for f in range(n)))
            self.assertTrue(all(0 < len(frames) <= 4 for step in plan for _, frames in step))

    def test_resume_and_epoch_boundaries(self):
        counts = [50] * 332
        live = FrameSchedule(counts)
        self.assertEqual(len(live._plan(0)), 1079)
        for position in [0, 199, 200, 1078, 1079, 2158, 4999, 50]:
            self.assertEqual(live.at(position), FrameSchedule(counts).at(position))
        self.assertNotEqual(live._plan(0), live._plan(1))

    def test_learning_rate_phase_boundary(self):
        cfg = dict(adapter_warmup_steps=200, lr_warmup_steps=20, max_steps=5000)
        self.assertAlmostEqual(lr_scale(0, cfg), .05)
        self.assertEqual(lr_scale(199, cfg), 1.)
        self.assertEqual(lr_scale(200, cfg), 1.)
        self.assertAlmostEqual(lr_scale(5000, cfg), .1)
