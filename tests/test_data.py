import copy
import unittest
from robojepa_worlds.capsaicin import action_chunks, frame_actions, make_splits, scene_id


class DataContracts(unittest.TestCase):
    def metadata(self):
        return {'actions':{str(i):dict(ws=1,ad=2,scs=3,pitch_delta=i,yaw_delta=2*i,collision=0,jump_invalid=0) for i in range(12)}}

    def test_command_outcomes_do_not_change_input(self):
        a=self.metadata(); b=copy.deepcopy(a)
        for row in b['actions'].values(): row.update(collision=1,jump_invalid=1)
        self.assertEqual(frame_actions(a),frame_actions(b))

    def test_ws_fix_and_explicit_fixed_flag(self):
        a=self.metadata(); rows=frame_actions(a)
        self.assertEqual(rows[0][:2],[0.,1.])
        a['action_encoding']={'_sprite_ws_refixed':True}
        self.assertEqual(frame_actions(a)[0][:2],[1.,0.])

    def test_ordered_chunks_and_padding(self):
        idx,chunks=action_chunks(self.metadata(),12,4)
        self.assertEqual(idx,[0,4,8]); self.assertEqual(len(chunks),2)
        self.assertEqual(chunks[0][7],0); self.assertAlmostEqual(chunks[0][16],1/15)
        self.assertAlmostEqual(chunks[1][7],4/15); self.assertEqual(chunks[0][36:],[0.]*28)

    def test_scene_disjoint_and_deterministic(self):
        stems=[f'{s}_a{i:03}_v{j:02}' for s in ['gallery','forest','street','house','room'] for i in range(3) for j in range(2)]
        a,scenes=make_splits(stems); b,_=make_splits(stems[::-1]); self.assertEqual(a,b)
        self.assertEqual(set(a.values()),{'train','val','test'})
        for stem in stems: self.assertEqual(a[stem],scenes[scene_id(stem)])

    def test_missing_actions_fail(self):
        a=self.metadata(); del a['actions']['3']
        with self.assertRaises(ValueError): frame_actions(a)


if __name__=='__main__': unittest.main()
