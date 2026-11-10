"""Configuration regression without importing Torch on the submission node."""
import ast
import json
from pathlib import Path
import unittest


class PredictorConfiguration(unittest.TestCase):
    def test_single_camera_keeps_pretrained_view_geometry(self):
        root=Path(__file__).resolve().parents[1]
        tree=ast.parse((root/'robojepa_worlds/model.py').read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='predictor_config')
        scope={'ROOT':root,'json':json}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'model_config','exec'),scope)
        actual=scope['predictor_config']()
        released=json.loads((root/'vendor/robo_jepa/configs/released/wm-100m/config.json').read_text())
        for key in ['num_views','view_groups_v','view_map','use_strong_view_conditioning','use_rope',
                    'predictor_embed_dim','num_heads','depth']:
            self.assertEqual(actual[key],released[key],key)
        self.assertGreater(actual['num_views'],1)
        self.assertIn(1,actual['view_groups_v'])


if __name__=='__main__': unittest.main()
