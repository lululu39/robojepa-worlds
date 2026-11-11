import unittest

from robojepa_worlds.control_protocol import controls, training_yaw


class ControlProtocolTest(unittest.TestCase):
    def test_incoming_chunks_units_and_recorded_tail(self):
        recorded = [[float(i)]*36+[0.]*28 for i in (2, 3)]
        commands = controls(recorded, 3, 0.5)
        self.assertEqual(commands['recorded_then_stop'], recorded+[[0.]*64])
        self.assertEqual(commands['zero'], [[0.]*64]*3)
        for name, channel, value in [('forward', 0, 1), ('backward', 1, 1),
                                     ('yaw_negative', 8, -.5), ('yaw_positive', 8, .5)]:
            row = [0.]*9; row[channel] = value
            self.assertEqual(commands[name], [row*4+[0.]*28]*3)
        self.assertEqual(recorded[0][0], 2)

    def test_yaw_uses_only_training_nonzero_source_rows(self):
        def rows(values):
            return [[0.]*8+[x] for x in values]
        index = {'clips': [{'split': 'train', 'frame_actions': rows([99, 0, -1, 3])},
                           {'split': 'test', 'frame_actions': rows([0, 1000])}]}
        result = training_yaw(index)
        self.assertEqual(result['degrees_per_source_frame'], 2)
        self.assertEqual(result['nonzero_frames'], 2)


if __name__ == '__main__':
    unittest.main()
