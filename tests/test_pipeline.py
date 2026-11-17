import unittest,sys,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from train import load,knn_proba,perturb,scores
class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x,cls.y,cls.xt,cls.yt=load();cls.r=json.loads((ROOT/'reports/evaluation.json').read_text(encoding='utf-8'));cls.p=json.loads((ROOT/'reports/parity-reference.json').read_text(encoding='utf-8'))
    def test_official_shape_range_and_no_identical_cross_split_images(self):
        self.assertEqual(self.x.shape,(3823,64));self.assertEqual(self.xt.shape,(1797,64))
        self.assertTrue(np.all((self.x>=0)&(self.x<=16)));self.assertEqual(set(self.y),set(range(10)))
        self.assertFalse(set(map(tuple,self.x))&set(map(tuple,self.xt)))
    def test_checksums(self):
        for name,digest in self.r['hashes'].items():self.assertEqual(hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest(),digest)
    def test_validation_partition_comes_only_from_official_training(self):
        from sklearn.model_selection import train_test_split
        tr,va=train_test_split(np.arange(len(self.y)),test_size=.2,stratify=self.y,random_state=42)
        self.assertFalse(set(tr)&set(va));self.assertEqual(len(tr)+len(va),3823);self.assertEqual(len(va),765)
    def test_model_selection_and_review_policy_use_validation(self):
        chosen=max(self.r['models'],key=lambda k:(self.r['models'][k]['validation']['accuracy'],self.r['models'][k]['validation']['macroF1']))
        self.assertEqual(chosen,self.r['selected'])
        eligible=[v for v in self.r['validationCoverage'] if v['accuracy'] is not None and v['accuracy']>=.99 and v['coverage']>=.5]
        threshold=min(eligible,key=lambda v:v['threshold'])['threshold'] if eligible else .8
        self.assertEqual(threshold,self.r['reviewThreshold'])
    def test_metrics_recompute_from_full_reference_predictions(self):
        for k,prob in self.p.items():
            p=np.array(prob);self.assertTrue(np.isfinite(p).all());np.testing.assert_allclose(p.sum(1),1,atol=1e-12)
            m=scores(self.yt,p)
            for key in ['accuracy','macroF1','brier','logLoss']:self.assertAlmostEqual(m[key],self.r['models'][k]['test'][key])
    def test_knn_tie_policy(self):
        p=knn_proba(np.zeros((1,64)),np.zeros((3,64)),np.array([4,2,9]),k=2)
        self.assertEqual(p.argmax(1)[0],2);self.assertEqual(p[0,4],.5)
    def test_transforms_and_clipping(self):
        p=np.zeros(64);p[7]=16;q=perturb(p,dx=1);self.assertEqual(q.sum(),0)
        np.testing.assert_array_equal(perturb(p),p)
        a=perturb(p,noise=4,seed=7);np.testing.assert_array_equal(a,perturb(p,noise=4,seed=7));self.assertTrue(np.all((a>=0)&(a<=16)))
if __name__=='__main__':unittest.main()
