import csv
import tempfile
from pathlib import Path
from unittest import TestCase,mock
import torch
from paper_study.common import artifact,read,write
from paper_study.report import report,METRICS
from paper_study.train import optimize,setup

class ReportingAndResume(TestCase):
    def test_29_columns_units_three_seed_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'eval';source=root/'source';run.mkdir();source.mkdir()
            checkpoint=root/'ckpt';checkpoint.write_bytes(b'synthetic-test-checkpoint');ck=artifact(checkpoint)
            policies=['scalar_v3','D1_seed0','D1_seed1','D1_seed2']
            members=[dict(scene_id='s0',origin_log='log0'),dict(scene_id='s1',origin_log='log1')]
            write(run/'inputs.json',dict(models={p:ck for p in policies},cohorts=dict(full=members,common=members)))
            write(source/'run_contract.json',{})
            write(run/'run_contract.json',dict(source_evaluation=artifact(source/'run_contract.json'),native_exports={},model_configs={p:{} for p in policies}))
            def values(policy):
                score=.5 if policy=='scalar_v3' else .6+.1*int(policy[-1])
                return {m['scene_id']:{**{k:score for k in METRICS},'success':score} for m in members}
            for p in policies:
                for b in ('open_navtest','open_rare'):
                    write(run/'openloop'/p/b/'audit.json',dict(metrics=values(p),ade_fde=dict(ade_4s=1.2,fde_4s=2.4)))
                for b in ('full_NR','full_R','common_NR','common_R'):
                    folder=source/'collections'/f'{b}_{p}'
                    write(folder/'audit.json',dict(metrics=values(p)))
                    write(folder/'accepted.json',dict(status='PASS',audit=artifact(folder/'audit.json')))
                    write(folder/'collection.json',dict(checkpoint=ck,noise='formal_navtest_seed0'))
            with mock.patch('paper_study.report.WORKSPACE',root),mock.patch('paper_study.storage.WORKSPACE',root),mock.patch('paper_study.storage.settings',return_value=None),mock.patch('paper_study.report.check_open',lambda r,p,b:read(r/'openloop'/p/b/'audit.json')):
                result=report(run);self.assertTrue(result['formal_table_ready'])
                rows=list(csv.reader((root/'reports/eval/formal_29cols_percent.tsv').open(),delimiter='\t'))
                self.assertTrue(all(len(r)==29 for r in rows));self.assertEqual(float(rows[1][2]),1.2)
                mean=next(r for r in rows if r[0]=='D1_3seed_mean');self.assertAlmostEqual(float(mean[4]),70)
                (run/'openloop/D1_seed2/open_rare/audit.json').unlink()
                result=report(run);self.assertFalse(result['formal_table_ready'])
                rows=list(csv.reader((root/'reports/eval/formal_29cols.tsv').open(),delimiter='\t'))
                self.assertFalse(any(r[0]=='D1_3seed_mean' for r in rows))
    def test_optimizer_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);write(run/'manifest.json',dict(test=True))
            config={};settings=dict(steps=4,lr=.01,weight_decay=.01,grad_clip=10)
            def train(arm,stop=None):
                setup(2,'cpu');model=torch.nn.Linear(3,1)
                def loss(step):
                    x=torch.randn(5,3);v=(model(x)-x.sum(-1,keepdim=True)).square().mean()
                    return v,dict(mse=v)
                def save(path,provenance):torch.save(model.state_dict(),path)
                return optimize(run,arm,0,model,config,settings,loss,save,'cpu',True,stop)
            train('full');train('resumed',2);train('resumed')
            a=torch.load(run/'smoke/full_seed0/selector.pt')
            b=torch.load(run/'smoke/resumed_seed0/selector.pt')
            for k in a:torch.testing.assert_close(a[k],b[k],atol=0,rtol=0)
