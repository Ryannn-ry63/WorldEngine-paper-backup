import tempfile
from pathlib import Path
import unittest
import torch
from paper_study.common import artifact,verify,write,safe_name
from paper_study.data import split_logs
from paper_study.mechanism import summarize
from paper_study.models import DiagonalSetEncoder
from paper_study.statistics import paired_log_interval

class Contracts(unittest.TestCase):
    def test_artifact_and_immutable_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'m.json';write(p,dict(a=1));item=artifact(p);verify(item)
            with self.assertRaises(RuntimeError):write(p,dict(a=2))
            p.write_text('{}')
            with self.assertRaises(RuntimeError):verify(item)
        with self.assertRaises(ValueError):safe_name('../old')
    def test_log_split_no_overlap_and_exclusions(self):
        rows=[dict(scene_id=f'{i}_{j}',origin_log=str(i)) for i in range(100) for j in range(3)]
        groups=split_logs(rows,['0','1']);logs=[{r['origin_log'] for r in group} for group in groups.values()]
        self.assertTrue(all(not a&b for i,a in enumerate(logs) for b in logs[i+1:]))
        self.assertFalse(set.union(*logs)&{'0','1'})
    def test_unary_same_parameters_and_no_candidate_crosstalk(self):
        torch.manual_seed(3);torch.set_num_threads(2)
        full=torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(16,4,32,dropout=0,batch_first=True),1).eval()
        unary=DiagonalSetEncoder(full);x=torch.randn(2,20,16)
        self.assertEqual(sum(p.numel() for p in full.parameters()),sum(p.numel() for p in unary.parameters()))
        y=unary(x);x[:,1:]+=torch.randn_like(x[:,1:])*3
        torch.testing.assert_close(y[:,0],unary(x)[:,0])
        perm=torch.randperm(20);torch.testing.assert_close(unary(x)[:,perm],unary(x[:,perm]))
    def test_full20_no_partial_oracle(self):
        target=dict(row_id='x',mode='R',scene_id='s',visiting_index=0,candidate_indices=[0,1],continuation_policy='fixed')
        labels={'x':{i:dict(score=i/20,no_at_fault_collisions=1,drivable_area_compliance=1) for i in range(20)}}
        a=summarize(labels,[target]);self.assertEqual(a['unqueried_better_fraction'],1)
        self.assertEqual(a['details'][0]['subsets']['20']['probability_cover_best'],1)
        del labels['x'][19]
        with self.assertRaises(ValueError):summarize(labels,[target])
    def test_paired_cluster_constant_delta(self):
        ref={str(i):{'score':0.2} for i in range(6)};cur={i:{'score':.5} for i in ref};logs={str(i):str(i//2) for i in range(6)}
        result=paired_log_interval(cur,ref,logs,'score')
        self.assertAlmostEqual(result['delta'],.3)
        for endpoint in result['ci95']:self.assertAlmostEqual(endpoint,.3)
        with self.assertRaises(ValueError):paired_log_interval({},ref,logs,'score')
if __name__=='__main__':unittest.main()
