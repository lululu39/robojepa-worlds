import copy
from collections import Counter
import json
from pathlib import Path
import unittest
from robojepa_worlds.capsaicin import action_chunks,frame_actions,make_splits,scene_id,corrected_metadata


class DataContracts(unittest.TestCase):
    def metadata(self):
        return {'source':'Capsaicin','actions':{str(i):dict(ws=1,ad=2,scs=3,pitch_delta=i,yaw_delta=2*i,
                                                      collision=0,jump_invalid=0) for i in range(12)}}

    def test_collision_policy_matches_upstream(self):
        a=self.metadata(); b=copy.deepcopy(a)
        for row in b['actions'].values(): row.update(collision=1,jump_invalid=1)
        self.assertEqual(frame_actions(b)[1][:7],[0.]*7)
        self.assertEqual(frame_actions(a),frame_actions(b,process_collisions=False))

    def test_ws_fix_idempotent_and_nonmutating(self):
        raw=self.metadata(); fixed=corrected_metadata(raw)
        self.assertEqual(raw['actions']['1']['ws'],1)
        self.assertEqual(fixed['actions']['1']['ws'],2)
        self.assertEqual(corrected_metadata(fixed),fixed)
        self.assertEqual(frame_actions(raw),frame_actions(fixed))
        raw['action_encoding']={'_sprite_ws_refixed':True}
        self.assertEqual(frame_actions(raw)[1][:2],[1.,0.])

    def test_incoming_actions_skip_first_frame(self):
        idx,chunks=action_chunks(self.metadata(),12,4)
        self.assertEqual(idx,[0,4,8]); self.assertEqual(len(chunks),2)
        self.assertEqual(frame_actions(self.metadata())[0],[0.]*9)
        self.assertEqual(chunks[0][7::9][:4],[1.,2.,3.,4.])
        self.assertEqual(chunks[1][7::9][:4],[5.,6.,7.,8.])
        self.assertEqual(chunks[0][36:],[0.]*28)

    def test_mouse_conversion_is_degrees(self):
        a=self.metadata(); a['mouse_normalized_to_mc']={'pitch_factor':1/15,'yaw_factor':1/15}
        self.assertEqual(frame_actions(a)[3][7:],[3.,6.])
        a['mouse_normalized_to_mc']['pitch_factor']=-1/15
        self.assertEqual(frame_actions(a)[3][7:],[-3.,6.])

    def test_all_manifest_clips_have_scene_isolation(self):
        manifest=json.loads((Path(__file__).resolve().parents[1]/'experiments/capsaicin/dataset-manifest.json').read_text())
        stems=[Path(i['path']).stem for i in manifest['files'] if i['path'].endswith('.mp4')]
        split,scenes=make_splits(stems); reversed_split,_=make_splits(stems[::-1])
        self.assertEqual(split,reversed_split); self.assertEqual(len(stems),440); self.assertEqual(len(scenes),8)
        self.assertEqual(Counter(split.values()),{'train':332,'val':20,'test':88})
        self.assertEqual(scene_id('cap_sp_pitch_sponza_c01_cameramain_v00'),'sponza')
        self.assertEqual(scene_id('allow_sponza_obj_pitch_allow_a008_v00'),'sponza')
        for stem in stems: self.assertEqual(split[stem],scenes[scene_id(stem)])

    def test_unknown_scene_and_missing_actions_fail(self):
        with self.assertRaises(ValueError): scene_id('unknown_asset')
        a=self.metadata(); del a['actions']['3']
        with self.assertRaises(ValueError): frame_actions(a)
        with self.assertRaises(ValueError): action_chunks(self.metadata(),13)


if __name__=='__main__': unittest.main()
