"""Execute the upstream scalar parsing body on AP without importing torch.

Only its final torch.tensor constructor is replaced with an identity list factory.
The actual Torch parser is also called for every clip on the preparation worker.
"""
import ast
import copy
import json
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest

from robojepa_worlds.capsaicin import corrected_metadata,frame_actions

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'vendor/micro_world'


def scalar_reference():
    path=REFERENCE/'microworld/data/dataset_game_video.py'
    tree=ast.parse(path.read_text())
    selected=[node for node in tree.body if isinstance(node,ast.FunctionDef)
              and node.name in ['keyboard_to_onehot','process_collisions']]
    dataset=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='VideoGameDataset')
    selected += [node for node in dataset.body if isinstance(node,ast.FunctionDef) and node.name=='_actions_to_tensors']
    namespace={'torch':SimpleNamespace(tensor=lambda value,dtype=None:value,float32=None)}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    return namespace['_actions_to_tensors']


class UpstreamParity(unittest.TestCase):
    def test_exact_upstream_scalar_parser(self):
        parse=scalar_reference()
        for source,factors in [('Capsaicin',None),('Blender',None),('Minecraft',None),
                               ('DL3DV',{'pitch_factor':-1/15,'yaw_factor':1/15}),
                               ('Capsaicin',{'pitch_factor':1/15,'yaw_factor':1/15})]:
            data={'source':source,'actions':{str(i):dict(ws=i%3,ad=(i+1)%3,scs=i%4,
                      pitch_delta=i*.2,yaw_delta=-i*.3,collision=int(i%5==0),jump_invalid=int(i%7==0))
                      for i in range(37)}}
            if factors is not None: data['mouse_normalized_to_mc']=factors
            fixed=corrected_metadata(data)
            for start in [0,4,12]:
                mouse,keyboard=parse(SimpleNamespace(process_collisions=True),copy.deepcopy(fixed),start,30,31-start)
                expected=[k+m for k,m in zip(keyboard,mouse)]
                actual=frame_actions(fixed)[start:31]; actual[0]=[0.]*9
                self.assertEqual(actual,expected)

    def test_ws_repair_matches_actual_upstream_tool(self):
        process=runpy.run_path(str(REFERENCE/'tools/fix_ws_swap.py'))['process_file']
        raw={'actions':{str(i):{'ws':i%3} for i in range(9)}}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'metadata.json'; path.write_text(json.dumps(raw))
            process(str(path)); actual=json.loads(path.read_text())
            expected=corrected_metadata(raw)
            self.assertEqual(actual['actions'],expected['actions'])
            self.assertEqual(actual['action_encoding']['_sprite_ws_refixed'],True)
            self.assertEqual(process(str(path)),'skipped')


if __name__=='__main__': unittest.main()
