import unittest
import torch
from paper_study.objectives import feedback_terms,masked_group,teacher_kl,supervised_selector
from paper_study.specs import FEEDBACK

class Objectives(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.z=torch.randn(3,20,dtype=torch.double,requires_grad=True)
        self.ref=torch.randn(3,20,dtype=torch.double)
        self.pairs=[(2,7,.3),None,(18,0,.8)]
    def test_p1_nll_identity_and_gradients(self):
        for arm,alpha in [('nll',0),('pair_nll',1),('mid',.5)]:
            actual,_=feedback_terms(self.z,self.ref,self.pairs,[],FEEDBACK[arm])
            expected=sum(g*(-self.z[i].log_softmax(0)[w]+alpha*torch.nn.functional.softplus(self.z[i,l]-self.z[i,w])) for i,p in enumerate(self.pairs) if p is not None for w,l,g in [p])/3
            torch.testing.assert_close(actual,expected)
            torch.testing.assert_close(torch.autograd.grad(actual,self.z,retain_graph=True)[0],torch.autograd.grad(expected,self.z,retain_graph=True)[0])
    def test_unknown_rewards_and_empty_groups(self):
        r=torch.full_like(self.z,float('nan'));v=torch.zeros_like(self.z,dtype=torch.bool)
        v[0,[1,4]]=True;r[0,1]=2;r[0,4]=1
        v[2,[3,8]]=True;r[2,[3,8]]=1
        loss=masked_group(self.z,r,v);grad=torch.autograd.grad(loss,self.z)[0]
        self.assertTrue(torch.isfinite(grad).all());self.assertTrue((grad[~v]==0).all())
        self.assertTrue((grad[1:]==0).all())
        torch.testing.assert_close(loss,-(self.z[0,[1,4]].softmax(0)*torch.tensor([1.,-1.])).sum()/3)
    def test_shift_permutation_and_dpo_reference(self):
        for name in ['pair','set_mass','nll','dpo','pair_nll_mass_anchor']:
            spec=FEEDBACK[name];a,_=feedback_terms(self.z,self.ref,self.pairs,[],spec)
            b,_=feedback_terms(self.z+50,self.ref-12,self.pairs,[],spec);torch.testing.assert_close(a,b)
        z=self.ref.clone().requires_grad_();l,t=feedback_terms(z,self.ref,self.pairs,[],FEEDBACK['pair_nll_mass_anchor'])
        self.assertLess(abs(float(t['mass_anchor'])),1e-12)
        _,t=feedback_terms(z,self.ref,self.pairs,[],FEEDBACK['dpo'])
        torch.testing.assert_close(t['dpo'],torch.tensor(1.1/3,dtype=torch.double)*torch.tensor(2.,dtype=torch.double).log())
    def test_curriculum_integral(self):
        standard,_=feedback_terms(self.z,self.ref,self.pairs,[],FEEDBACK['pair_nll'])
        for name in ['pair_nll_interleaved','pair_nll_concentrated']:
            total=sum(feedback_terms(self.z,self.ref,self.pairs,[],FEEDBACK[name],s,10)[0] for s in range(1,11))/10
            torch.testing.assert_close(total,standard)
    def test_sft_masked_expert(self):
        trajectories=torch.zeros(3,20,8,3,dtype=torch.double);target=torch.ones(3,8,3,dtype=torch.double)
        trajectories[:,5]=target;mask=torch.ones(3,8,dtype=torch.double)
        loss=supervised_selector(self.z,trajectories,target,mask)
        torch.testing.assert_close(loss,torch.nn.functional.cross_entropy(self.z,torch.full((3,),5)))
        with self.assertRaises(ValueError):supervised_selector(self.z,trajectories,target,mask*0)
    def test_teacher_kl_direction(self):
        expected=(self.ref.softmax(-1)*(self.ref.log_softmax(-1)-self.z.log_softmax(-1))).sum(-1).mean()
        torch.testing.assert_close(teacher_kl(self.z,self.ref),expected)

if __name__=='__main__':unittest.main()
