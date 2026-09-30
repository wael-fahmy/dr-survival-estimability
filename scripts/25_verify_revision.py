"""Check preservation of original primary-horizon points and added output integrity."""
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'lib'))
from lib_io import SEED,barcode12
from lib_est import aipw_rmst,ipcw_km_rmst_contrast
spec=importlib.util.spec_from_file_location('stage7',ROOT/'scripts/07_estimate.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
frame=pd.read_csv(ROOT/'output/05_cohort_landmark90.csv',index_col=0)
frame.index=barcode12(frame.index)
E=pd.read_csv(ROOT/'output/07_estimates.csv')
rows=[]
for population in ['TNBC_chemo','ALL_chemo']:
    for adjustment in ['CLINICAL','EXISTING','UNION']:
        t,e,a,x,n=m._primary_inputs(frame,population,adjustment)
        for label,fn in [('AIPW (cross-fitted)',aipw_rmst),('IPCW-KM contrast',ipcw_km_rmst_contrast)]:
            got=(fn(t,e,a,x,1095,seed=SEED) if label.startswith('AIPW') else fn(t,e,a,1095,X=x,seed=SEED))['estimate']
            saved=E[(E.arm==population)&(E.adjustment==adjustment)&(E.tau==1095)&(E.estimator==label)].iloc[0].estimate_days
            rows.append(dict(population=population,adjustment=adjustment,estimator=label,
                             recomputed=got,saved=saved,abs_diff=abs(got-saved)))
            assert np.isclose(got,saved,atol=1e-8,rtol=0),(population,adjustment,label,got,saved)
C=pd.read_csv(ROOT/'output/20_comparisons.csv')
assert len(C)==8 and np.isfinite(C.estimate_days).all()
S=pd.read_csv(ROOT/'output/21_stress_summary.csv')
R=pd.read_csv(ROOT/'output/21_stress_replicates.csv')
assert len(S)==24 and len(R)==4800
assert (S.replicates==200).all() and (S.inner_bootstraps==199).all()
assert (S.valid_replicates+S.failed_replicates==200).all()
assert ((S.coverage>=0)&(S.coverage<=1)).all()
assert (S.coverage_mc_lo<=S.coverage).all() and (S.coverage_mc_hi>=S.coverage).all()
B=pd.read_csv(ROOT/'output/20_balance.csv')
assert (B.mean_abs_smd>=0).all() and (B.max_abs_smd>=B.mean_abs_smd).all()
report=dict(primary_points=rows,point_checks_passed=len(rows),stress_rows=len(R),
            stress_cells=6,methods=4,failed_estimator_replicates=int(S.failed_replicates.sum()),
            original_bootstrap_and_calibration='Preserved existing outputs; not rerun for this revision')
(ROOT/'output/25_revision_verification.json').write_text(json.dumps(report,indent=2))
print(f'Passed {len(rows)} original point-estimate checks; 4800 stress method/replicate records and 8 clinical comparisons checked.')
