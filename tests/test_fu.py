import unittest
import copy
import numpy as np
import torch
from src.data.prepare import connected_groups, balanced_group_split, overlaps
from src.data.federated import partition, forget_request, retained_partition
from src.fl.fedavg_core import model, state, aggregate, ClientWorker, fedavg
from src.fu.conpfu_inspired import forget_step, kd_loss, distill
from src.eval.metrics import similarity, matched_indices, mia, calibrate_mia


class FUTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_transitive_leakage_components(self):
        g=connected_groups(['a','a','b','c','d'],['x','y','y','z','w'])
        self.assertEqual(g[0],g[2])
        self.assertNotEqual(g[2],g[3])

    def test_partition_nested_forget_retained_weights(self):
        y=np.tile([0,1],10000)
        clients=partition(y,42)
        self.assertEqual(len(clients),5)
        np.testing.assert_array_equal(np.sort(np.concatenate(clients)),np.arange(len(y)))
        target=int(np.argmax([len(x) for x in clients]))
        f10,_=forget_request(y,clients[target],.1,43)
        f50,r=forget_request(y,clients[target],.5,43)
        self.assertTrue(np.isin(f10,f50).all())
        retained=retained_partition(clients,target,f50)
        self.assertFalse(np.isin(np.concatenate(retained),f50).any())
        counts=[len(x) for x in retained]
        updates=[{'w':torch.tensor([float(i)])} for i in range(5)]
        actual=aggregate(updates,counts)['w'].item()
        self.assertAlmostEqual(actual,sum(i*n for i,n in enumerate(counts))/sum(counts),places=6)
        with self.assertRaises(ValueError): retained_partition(clients,target,np.array([-1]))

    def test_ascent_sign_noop_and_fresh_initialization(self):
        m=model(2,[4,3],42)
        x=torch.tensor([[1.,0.],[0.,1.]])
        y=torch.tensor([0,1])
        old=copy.deepcopy(m)
        before=torch.nn.functional.cross_entropy(m(x),y).item()
        forget_step(m,x,y,[p.detach().clone() for p in m.parameters()],.001,0,100)
        self.assertGreater(torch.nn.functional.cross_entropy(m(x),y).item(),before)
        noop,logs=fedavg(old,[],{},0,42)
        self.assertEqual(logs,[])
        fresh=model(2,[4,3],42)
        for k,v in state(old).items():
            self.assertTrue(torch.equal(v,state(noop)[k]))
            self.assertTrue(torch.equal(v,state(fresh)[k]))

    def test_kd_teacher_frozen_temperature(self):
        s=torch.tensor([[1.,2.]],requires_grad=True)
        t=torch.tensor([[2.,1.]],requires_grad=True)
        y=torch.tensor([1]); temp=3.
        loss=kd_loss(s,t,y,temp,.3)
        reference=.3*torch.nn.functional.cross_entropy(s,y)+.7*temp**2*torch.nn.functional.kl_div(
            torch.log_softmax(s/temp,1),torch.softmax(t.detach()/temp,1),reduction='batchmean')
        self.assertAlmostEqual(loss.item(),reference.item())
        loss.backward()
        self.assertIsNone(t.grad); self.assertIsNotNone(s.grad)

    def test_similarity_and_mia_controls(self):
        p=np.array([.1,.9,.2,.8]); y=np.array([0,1,0,1])
        self.assertEqual(similarity(p,p)['prediction_mae'],0)
        mi,ni=matched_indices(y,y,42)
        np.testing.assert_array_equal(y[mi],y[ni])
        thresholds=calibrate_mia(y,np.array([.4,.6,.3,.7]))
        out=mia(y,np.array([.01,.99,.02,.98]),y,np.array([.4,.6,.3,.7]),thresholds)
        self.assertEqual(out['mia_auc'],1.)
        self.assertIsNone(out['mia_tpr_at_fpr_0_01'])


if __name__=='__main__': unittest.main()
