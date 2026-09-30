"""Independent reference and invariance checks for the added Cox benchmark."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lib'))
from lib_est import cox_rmst_contrast


class CoxRMSTTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(528)
        self.X=rng.normal(size=(400,2))
        self.A=(rng.random(400)<.5).astype(int)
        T=rng.exponential(1/(.001*np.exp(.3*self.X[:,0]-.2*self.X[:,1]-.4*self.A)))
        C=rng.exponential(2400,size=400)
        self.t=np.minimum(T,C)
        self.e=(T<=C).astype(int)

    def test_unpenalized_against_lifelines_survival_integration(self):
        from lifelines import CoxPHFitter
        data=pd.DataFrame(self.X,columns=['x0','x1'])
        data['A'],data['t'],data['event']=self.A,self.t,self.e
        fit=CoxPHFitter().fit(data,'t','event')
        grid=np.unique(np.r_[0.,self.t[self.t<1095],1095.])
        curves=[]
        for a in [0,1]:
            new=data[['x0','x1','A']].copy();new['A']=a
            # lifelines interpolates at 0; supply exact baseline knot values
            # and explicitly set survival before the first observation to one.
            s=fit.predict_survival_function(new,times=grid).to_numpy()
            s[grid<self.t.min()]=1.
            curves.append(float(np.mean(np.diff(grid)@s[:-1])))
        ours=cox_rmst_contrast(self.t,self.e,self.A,self.X,1095,ridge=0)
        self.assertAlmostEqual(ours['estimate'],curves[1]-curves[0],places=3)

    def test_treatment_relabeling_reverses_contrast(self):
        one=cox_rmst_contrast(self.t,self.e,self.A,self.X,1095)['estimate']
        other=cox_rmst_contrast(self.t,self.e,1-self.A,self.X,1095)['estimate']
        self.assertAlmostEqual(one,-other,places=5)

    def test_constant_covariate_does_not_change_estimate(self):
        one=cox_rmst_contrast(self.t,self.e,self.A,self.X,1095)['estimate']
        extra=np.column_stack([self.X,np.ones(len(self.A))])
        other=cox_rmst_contrast(self.t,self.e,self.A,extra,1095)['estimate']
        self.assertAlmostEqual(one,other,places=8)

    def test_contrast_is_within_rmst_bounds(self):
        for tau in [0,365,1095]:
            result=cox_rmst_contrast(self.t,self.e,self.A,self.X,tau)
            self.assertGreaterEqual(result['rmst_control'],0)
            self.assertLessEqual(result['rmst_treated'],tau)
            self.assertLessEqual(abs(result['estimate']),tau)

    def test_missing_treatment_group_is_an_error(self):
        with self.assertRaises(ValueError):
            cox_rmst_contrast(self.t,self.e,np.ones(len(self.A)),self.X,1095)


if __name__=='__main__':unittest.main()
