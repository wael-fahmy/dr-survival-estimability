"""New independent stress scenarios: four estimators, known RMST truth.

Six cells: n=136/988 crossed with correctly specified PH/censoring, delayed
treatment benefit, and heavy-tailed nonlinear nuisance misspecification.
200 Monte Carlo replicates and 199 inner patient bootstraps per cell. These
are separate from the original 32-cell calibration and its 500-replicate
operating-point validation. --smoke writes to a separate directory.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'lib'))
from lib_io import OUTPUT,SEED
from lib_est import aipw_rmst,ipcw_km_rmst_contrast,cox_rmst_contrast,km_rmst,restricted_outcome

TAU=1095.
METHODS=['Unadjusted KM','IPCW-KM','AIPW','Cox standardized RMST']


def sample_covariates(rng,n,p,regime):
    return (rng.standard_t(3,size=(n,p))/np.sqrt(3) if regime=='nonlinear_heavy_tail'
            else rng.normal(size=(n,p)))


def hazards(X,regime,base):
    lp=.7*X[:,0]-.5*X[:,1]+.3*X[:,2]
    fx=.5*X[:,0]+.3*X[:,1]
    fc=.5*X[:,0]+.5*X[:,2]
    if regime=='nonlinear_heavy_tail':
        lp += .6*(X[:,0]>1)
        fx=.5*X[:,0]+.5*np.sin(X[:,1])+.3*(X[:,2]**2-1)
        fc += .35*(X[:,1]**2-1)
    ps=1/(1+np.exp(-np.clip(lp,-20,20)))
    frail=np.exp(np.clip(fx,-4,4))
    censor=np.log(2)/1100*np.exp(np.clip(fc,-4,4))
    h0=base*frail
    early,late=(1.2,.45) if regime=='delayed_benefit' else (.65,.65)
    return ps,h0,h0*early,h0*late,censor


def expected_events(h1,h2,c):
    cut=365.
    return h1/(h1+c)*(-np.expm1(-(h1+c)*cut))+np.exp(-(h1+c)*cut)*h2/(h2+c)*(-np.expm1(-(h2+c)*(TAU-cut)))


def calibrate_base(regime,target_rate):
    X=sample_covariates(np.random.default_rng(SEED+908),30000,3,regime)
    lo,hi=1e-7,.003
    for _ in range(45):
        mid=(lo+hi)/2
        ps,h0,h1,h2,c=hazards(X,regime,mid)
        avg=np.mean(ps*expected_events(h1,h2,c)+(1-ps)*expected_events(h0,h0,c))
        if avg<target_rate:lo=mid
        else:hi=mid
    return (lo+hi)/2


def simulate(rng,sc):
    n,p=sc['n'],sc['p']
    X=sample_covariates(rng,n,p,sc['regime'])
    ps,h0,h1,h2,c=hazards(X,sc['regime'],sc['base'])
    A=(rng.random(n)<ps).astype(int)
    first=np.where(A==1,h1,h0);second=np.where(A==1,h2,h0)
    E=rng.exponential(size=n)
    T=np.where(E<=first*365,E/first,365+(E-first*365)/second)
    C=rng.exponential(1/c)
    t,e=np.minimum(T,C),(T<=C).astype(int)
    rm0=-np.expm1(-h0*TAU)/h0
    rm1=-np.expm1(-h1*365)/h1+np.exp(-h1*365)*(-np.expm1(-h2*(TAU-365)))/h2
    truth=float(np.mean(rm1-rm0))
    return X,A,t,e,truth


def estimate_all(X,A,t,e,seed):
    vals=np.full(4,np.nan)
    if len(np.unique(A))<2:return vals
    functions=[lambda:km_rmst(t[A==1],e[A==1],TAU)-km_rmst(t[A==0],e[A==0],TAU),
               lambda:ipcw_km_rmst_contrast(t,e,A,TAU,X=X)['estimate'],
               lambda:aipw_rmst(t,e,A,X,TAU,seed=seed)['estimate'],
               lambda:cox_rmst_contrast(t,e,A,X,TAU)['estimate']]
    for j,fn in enumerate(functions):
        try:vals[j]=fn()
        except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError):pass
    return vals


def replicate(task):
    sc,r,B=task
    seed=SEED+100000+sc['id']*10000+r
    rng=np.random.default_rng(seed)
    X,A,t,e,truth=simulate(rng,sc)
    point=estimate_all(X,A,t,e,seed)
    _,complete=restricted_outcome(t,e,TAU)
    cmin=min(int(complete[A==a].sum()) for a in (0,1))
    rho=(sc['p']+1)/max(cmin*.8,1)
    sets=[3,max(4,sc['p']//2),sc['p']]
    ranges=np.array([estimate_all(X[:,:p],A,t,e,seed) for p in sets])
    movement=np.ptp(ranges,axis=0)
    boot=[]
    for b in range(B):
        idx=rng.integers(0,len(A),len(A))
        boot.append(estimate_all(X[idx],A[idx],t[idx],e[idx],seed))
    boot=np.array(boot)
    rows=[]
    for j,name in enumerate(METHODS):
        values=boot[:,j];ok=np.isfinite(values)
        valid=np.isfinite(point[j]) and ok.mean()>=.95
        lo,hi=np.quantile(values[ok],[.025,.975]) if valid else (np.nan,np.nan)
        rows.append(dict(cell=sc['id'],regime=sc['regime'],n=sc['n'],p=sc['p'],
                         replicate=r,estimator=name,truth=truth,estimate=point[j],
                         error=point[j]-truth,ci_lo=lo,ci_hi=hi,
                         bootstrap_se=float(np.std(values[ok],ddof=1)) if valid else np.nan,
                         covered=bool(lo<=truth<=hi) if valid else np.nan,
                         valid=bool(valid),bootstrap_failures=int(sum(~ok)),
                         events=int(((e==1)&(t<=TAU)).sum()),
                         complete_min=cmin,rho=rho,adjustment_movement=movement[j],
                         inner_bootstraps=B))
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--workers',type=int,default=min(8,os.cpu_count() or 1))
    args=parser.parse_args()
    reps,B=(3,9) if args.smoke else (200,199)
    out=OUTPUT/'smoke_biometrics' if args.smoke else OUTPUT
    out.mkdir(exist_ok=True)
    scenarios=[]
    for regime in ['ph_conditional_censoring','delayed_benefit','nonlinear_heavy_tail']:
        for n,p,ev in [(136,24,16),(988,27,72)]:
            scenarios.append(dict(id=len(scenarios),regime=regime,n=n,p=p,
                                  target_events=ev,base=calibrate_base(regime,ev/n)))
    # Persist only complete per-replicate records; a code hash prevents stale
    # checkpoints from silently surviving an implementation change.
    import hashlib
    digest=hashlib.sha256(Path(__file__).read_bytes()+(ROOT/'lib/lib_est.py').read_bytes()).hexdigest()
    checkpoint=out/'21_stress_checkpoint.json'
    rows=[]
    if checkpoint.exists():
        prior=json.loads(checkpoint.read_text())
        if prior['code_sha256']==digest and prior['replicates']==reps and prior['bootstrap']==B:
            rows=prior['rows']
    completed={(r['cell'],r['replicate']) for r in rows}
    tasks=[(sc,r,B) for sc in scenarios for r in range(reps) if (sc['id'],r) not in completed]
    total=len(scenarios)*reps
    done=len(completed)
    print(f'Stress validation: {total} replicates x {B} patient bootstraps; {done} resumed.',flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(replicate,task):task[:2] for task in tasks}
        for future in as_completed(futures):
            rows.extend(future.result());done+=1
            if done%20==0 or done==total:
                print(f'Stress validation {done}/{total}',flush=True)
                checkpoint.write_text(json.dumps(dict(code_sha256=digest,replicates=reps,
                                                      bootstrap=B,rows=rows)))
    data=pd.DataFrame(rows).sort_values(['cell','replicate','estimator'])
    data.to_csv(out/'21_stress_replicates.csv',index=False)
    summaries=[]
    for (cell,estimator),d in data.groupby(['cell','estimator']):
        valid=d[d.valid]
        errors=valid.error.to_numpy()
        R=len(valid)
        coverage=float(valid.covered.astype(float).mean())
        sderr=float(np.std(errors,ddof=1))
        z=1.959963984540054
        centre=(coverage+z*z/(2*R))/(1+z*z/R)
        half=z*np.sqrt(coverage*(1-coverage)/R+z*z/(4*R*R))/(1+z*z/R)
        # Match the original diagnostic: movement / sampling SD of the full-set estimate.
        sampling_sd=float(np.std(valid.estimate,ddof=1))
        ratio=float(valid.adjustment_movement.mean()/sampling_sd)
        summaries.append(dict(cell=cell,regime=d.regime.iloc[0],n=d.n.iloc[0],p=d.p.iloc[0],
                              estimator=estimator,replicates=len(d),valid_replicates=R,
                              failed_replicates=len(d)-R,truth=d.truth.mean(),bias=np.mean(errors),
                              bias_mcse=sderr/np.sqrt(R),rmse=np.sqrt(np.mean(errors**2)),
                              error_sd=sderr,mean_bootstrap_se=valid.bootstrap_se.mean(),
                              se_ratio=valid.bootstrap_se.mean()/sderr,
                              coverage=coverage,coverage_mcse=np.sqrt(coverage*(1-coverage)/R),
                              coverage_mc_lo=centre-half,coverage_mc_hi=centre+half,
                              events=d.events.mean(),rho=d.rho.mean(),
                              sensitivity_ratio=ratio,inner_bootstraps=B))
    pd.DataFrame(summaries).to_csv(out/'21_stress_summary.csv',index=False)
    (out/'21_stress_design.json').write_text(json.dumps(dict(scenarios=scenarios,
        seed=SEED,replicates_per_cell=reps,inner_bootstrap_replicates=B,code_sha256=digest,
        truth='analytic conditional RMST averaged over each replicate covariate sample',
        bootstrap='patient resampling; fixed seed within each fitted AIPW procedure',
        coverage='conditional on successful point estimate and >=95% successful inner fits; failures tabulated'),indent=2))
    print('Stress summaries complete.',flush=True)


if __name__=='__main__':main()
