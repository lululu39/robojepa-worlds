import json
import unittest

from cluster.watch_decoder_train import first_update


class DecoderLaunchTest(unittest.TestCase):
    def test_only_finite_formal_updates_count(self):
        metrics = {'step':1, 'train/loss':1., 'train/mse':.02, 'train/lpips':.8, 'train/grad_norm':2.}
        line = json.dumps(metrics)
        self.assertIsNone(first_update(line, 'run'))
        self.assertIsNone(first_update('FORMAL_DECODER_START other resume_step 0\n'+line, 'run'))
        self.assertIsNone(first_update('FORMAL_DECODER_START run resume_step 1\n'+line, 'run'))
        self.assertEqual(first_update('FORMAL_DECODER_START run resume_step 0\n'+line, 'run'), metrics)
        with self.assertRaises(RuntimeError):
            first_update('FORMAL_DECODER_START run resume_step 0\n'+json.dumps({**metrics,'train/loss':float('nan')}), 'run')
