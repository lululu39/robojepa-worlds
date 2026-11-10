import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from cluster import workflow


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


if __name__=='__main__': unittest.main()
