import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from cluster import workflow,submit as submit_module


class WorkflowContracts(unittest.TestCase):
    def test_held_and_nonzero_jobs_stop(self):
        for ad in [dict(JobStatus=5),dict(JobStatus=4,ExitCode=1),dict(JobStatus=3),dict(JobStatus=4,ExitCode=0,ExitBySignal=True)]:
            with patch.object(workflow,'command',return_value=json.dumps([ad])):
                with self.assertRaises(RuntimeError): workflow.query({'job':'123.0'})

    def test_completed_job_requires_zero_exit(self):
        with patch.object(workflow,'command',side_effect=['[]',json.dumps([dict(JobStatus=4,ExitCode=0)])]):
            status,_=workflow.query({'job':'123.0'})
            self.assertEqual(status,4)

    def test_incomplete_report_cannot_advance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'gate').mkdir()
            with tarfile.open(root/'gate/results.tar','w') as tar:
                payload=b'{"complete": false}'; item=tarfile.TarInfo('results/gate.json'); item.size=len(payload)
                tar.addfile(item,io.BytesIO(payload))
            with patch.object(workflow,'JOBS',root),patch.object(workflow,'REPORTS',root/'reports'):
                with self.assertRaises(RuntimeError): workflow.collect({'key':'gate','stage':'gate'})
                self.assertFalse((root/'reports/gate.json').exists())

    def test_stop_marker_prevents_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'STOPPED').touch()
            with patch.object(workflow,'JOBS',root),patch.object(workflow,'submit') as submit:
                self.assertEqual(workflow.tick(),'stopped'); submit.assert_not_called()

    def test_gate_success_submits_four_encoding_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); reports=root/'reports'; reports.mkdir()
            for stage in ['prepare','gate']:
                (root/stage).mkdir()
                (root/stage/'state.json').write_text(json.dumps({'key':stage,'stage':stage,'job':'123.0'}))
                (reports/(stage+'.json')).write_text('{"complete":true}')
            with patch.object(workflow,'JOBS',root),patch.object(workflow,'REPORTS',reports), \
                 patch.object(workflow,'command',return_value=''), \
                 patch.object(workflow,'query',return_value=(4,{'NumJobStarts':1})), \
                 patch.object(workflow,'submit') as submit:
                self.assertEqual(workflow.tick(),'waiting')
                self.assertEqual(submit.call_count,4)
                self.assertEqual(submit.call_args_list[0].args,('encode','--shard','0'))
                self.assertEqual(submit.call_args_list[-1].args,('encode','--shard','3'))

    def test_only_explicit_dependency_holds_are_expected(self):
        state={'job':'123.0','waiting_dependencies':True}
        ad=dict(JobStatus=5,NumJobStarts=0,HoldReasonCode=15,RoboJepaWaitingDependencies=True)
        with patch.object(workflow,'command',return_value=json.dumps([ad])):
            self.assertEqual(workflow.query(state)[0],5)
        for changed in [dict(ad,NumJobStarts=1),dict(ad,HoldReasonCode=1),dict(ad,HoldReasonCode=12),dict(ad,RoboJepaWaitingDependencies=False)]:
            with patch.object(workflow,'command',return_value=json.dumps([changed])):
                with self.assertRaises(RuntimeError): workflow.query(state)

    def test_encoding_completion_does_not_start_formal_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); reports=root/'reports'; reports.mkdir()
            for key in ['prepare','gate',*[f'encode-{i}' for i in range(4)]]:
                stage='encode' if key.startswith('encode-') else key
                (root/key).mkdir()
                (root/key/'state.json').write_text(json.dumps({'key':key,'stage':stage,'job':'123.0'}))
                (reports/(key+'.json')).write_text('{"complete":true}')
            with patch.object(workflow,'JOBS',root),patch.object(workflow,'REPORTS',reports), \
                 patch.object(workflow,'command',return_value=''), \
                 patch.object(workflow,'query',return_value=(4,{'NumJobStarts':1})), \
                 patch.object(workflow,'submit') as submit:
                self.assertEqual(workflow.tick(),'encoding_complete'); submit.assert_not_called()

    def test_failed_dependency_validation_never_releases_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'encode-0').mkdir()
            state={'job':'123.0','key':'encode-0','stage':'encode','source':{'commit':'abc'},'waiting_dependencies':True}
            (root/'encode-0/state.json').write_text(json.dumps(state))
            ad=dict(JobStatus=5,NumJobStarts=0,HoldReasonCode=15,RoboJepaWaitingDependencies=True)
            with patch.object(submit_module,'JOBS',root), \
                 patch.object(submit_module,'command',return_value=json.dumps([ad])) as command, \
                 patch.object(submit_module,'validate_inputs',side_effect=RuntimeError('GPU gate required')):
                with self.assertRaises(RuntimeError): submit_module.release_deferred(state)
                self.assertTrue(all(call.args[0][0]!='condor_release' for call in command.call_args_list))
            self.assertTrue(json.loads((root/'encode-0/state.json').read_text())['waiting_dependencies'])

    def test_pending_parent_does_not_release_children(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); reports=root/'reports'; reports.mkdir()
            for key,stage,dependencies in [('prepare','prepare',[]),('gate','gate',['prepare']),
                                           ('encode-0','encode',['prepare','gate'])]:
                (root/key).mkdir()
                state={'key':key,'stage':stage,'job':'123.0','waiting_dependencies':bool(dependencies),'dependencies':dependencies}
                (root/key/'state.json').write_text(json.dumps(state))
            def query(state): return (5 if state['waiting_dependencies'] else 1),{'NumJobStarts':0}
            with patch.object(workflow,'JOBS',root),patch.object(workflow,'REPORTS',reports), \
                 patch.object(workflow,'command',return_value=''),patch.object(workflow,'query',side_effect=query), \
                 patch.object(workflow,'release_deferred') as release,patch.object(workflow,'submit') as submit:
                self.assertEqual(workflow.tick(),'waiting'); release.assert_not_called(); submit.assert_not_called()


if __name__=='__main__': unittest.main()
