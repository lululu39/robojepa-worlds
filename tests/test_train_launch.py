import json
import unittest
from unittest.mock import patch

from cluster import submit
from cluster.watch_train import first_update


class TrainLaunchContracts(unittest.TestCase):
    def test_execution_metadata_reuses_existing_gpu_validation(self):
        before={'height':176,'formal':False,'auto_train':False,'milestone':'preflight'}
        after={**before,'formal':True,'milestone':'predictor_only'}
        with patch.object(submit,'command',side_effect=[submit.CONFIG_PATH,json.dumps(before),json.dumps(after)]):
            self.assertEqual(submit.changed_inputs('old','new',submit.INPUT_CODE),[])

    def test_numerical_changes_still_require_validation(self):
        for key,value in [('height',256),('action_learning_rate',.002),('max_steps',4000)]:
            before={'height':176,'action_learning_rate':.0001,'max_steps':2000,'formal':False}
            after={**before,'formal':True,key:value}
            with patch.object(submit,'command',side_effect=[submit.CONFIG_PATH,json.dumps(before),json.dumps(after)]):
                self.assertEqual(submit.changed_inputs('old','new',submit.INPUT_CODE),[submit.CONFIG_PATH])

    def test_model_change_cannot_be_hidden_by_metadata(self):
        before={'formal':False}; after={'formal':True}
        with patch.object(submit,'command',side_effect=[submit.CONFIG_PATH+'\nrobojepa_worlds/model.py',json.dumps(before),json.dumps(after)]):
            self.assertEqual(submit.changed_inputs('old','new',submit.INPUT_CODE),['robojepa_worlds/model.py'])

    def test_only_finite_post_marker_update_means_started(self):
        metrics={'step':1,'train/loss':.2,'train/teacher':.1,'train/rollout':.3,'train/grad_norm':1.}
        line=json.dumps(metrics)
        self.assertIsNone(first_update(line,'abc'))
        self.assertIsNone(first_update('FORMAL_TRAIN_START other resume_step 0\n'+line,'abc'))
        self.assertIsNone(first_update('FORMAL_TRAIN_START abc resume_step 1\n'+line,'abc'))
        self.assertEqual(first_update('FORMAL_TRAIN_START abc resume_step 0\n'+line,'abc'),metrics)
        with self.assertRaises(RuntimeError):
            first_update('FORMAL_TRAIN_START abc resume_step 0\n'+json.dumps({**metrics,'train/loss':float('nan')}),'abc')


if __name__=='__main__': unittest.main()
