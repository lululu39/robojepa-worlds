"""Only the documented decord metadata defect can bypass uv's platform check."""
import unittest

from cluster.decoder_environment import known_decord_tag_error


class DecoderEnvironment(unittest.TestCase):
    def test_known_published_wheel_defect(self):
        output = 'Checked 63 packages in 105ms\nFound 1 incompatibility\nThe package `decord` was built for a different platform\n'
        wheel = 'Wheel-Version: 1.0\nTag: cp36-cp36m-manylinux2010_x86_64\n'
        self.assertTrue(known_decord_tag_error(1, output, wheel, '0.6.0', 'x86_64'))
        for code, text, tag, version, machine in [
            (1, output + 'numpy is incompatible\n', wheel, '0.6.0', 'x86_64'),
            (1, output, wheel, '0.6.0', 'aarch64'),
            (1, output, wheel, '0.7.0', 'x86_64'),
            (1, output, 'Tag: cp311-cp311-linux_x86_64\n', '0.6.0', 'x86_64'),
            (2, output, wheel, '0.6.0', 'x86_64'),
        ]:
            with self.subTest(code=code, text=text, tag=tag, version=version, machine=machine):
                self.assertFalse(known_decord_tag_error(code, text, tag, version, machine))
