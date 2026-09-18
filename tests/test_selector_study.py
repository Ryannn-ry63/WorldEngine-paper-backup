import unittest
from selector_study.data import make_schedule
from selector_study.results import table_rows,BLOCKS,METRICS

class Contracts(unittest.TestCase):
    def test_sampling_preserves_common_feedback_and_source_restriction(self):
        records=[dict(kind='real_rare',common=[['r',0]],hard=[['r',1]]),
                 dict(kind='synthetic_rollout',common=[['r',2]],hard=[['s',0]])]
        rows=[dict(mode=m) for m in ('NR','NR','R','R')]
        a=make_schedule(records,rows,0,100)
        self.assertEqual(a,make_schedule(records,rows,0,100))
        self.assertNotEqual(a,make_schedule(records,rows,1,100))
        self.assertTrue(all(ri==0 for s in a for ri,vi in s['real']))
        self.assertTrue(any(ri==1 for s in a for ri,vi in s['mixed']))
        for s in a:
            self.assertEqual(len(s['common']),8)
            self.assertEqual([rows[i]['mode'] for i in s['feedback']],['NR']*8+['R']*8)

    def test_export_mean_sd_and_reactive_success(self):
        data=dict(ade_fde={},blocks={})
        for b in BLOCKS:
            data['blocks'][b]=dict(arms={})
            for arm in ('scalar_v3','P1','CL_NLL','MID'):
                n=1 if arm=='scalar_v3' else 3;ms=[]
                for i in range(n):
                    ms.append(dict({k:.5+i*.1 for k in METRICS},success=.2+i*.1 if b=='full_R' else .9))
                    data['ade_fde'][arm if n==1 else f'{arm}_seed{i}']=dict(ade_4s=1.+i,fde_4s=2.+i)
                data['blocks'][b]['arms'][arm]=dict(seed_means=ms)
        rows=table_rows(data)
        self.assertEqual(len(rows),16);self.assertTrue(all(len(r)==29 for r in rows))
        mean=next(r for r in rows if r[0]=='V3rarelog_to_mixed_P1_3optseed_mean')
        sd=next(r for r in rows if r[0]=='V3rarelog_to_mixed_P1_3optseed_std')
        self.assertAlmostEqual(mean[2],2.);self.assertAlmostEqual(mean[-1],.3)
        self.assertAlmostEqual(sd[2],1.);self.assertAlmostEqual(sd[-1],.1)

if __name__=='__main__':unittest.main()
