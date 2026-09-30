"""Same-horizon KM/Cox benchmarks and estimator-specific diagnostics.

Default: 2000 patient bootstraps. --smoke uses 20 in a separate directory.
Existing treatment estimates are not replaced; this extends the fixed analysis.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import importlib.util
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib'))
from lib_io import OUTPUT, SEED, barcode12
from lib_est import (aipw_rmst, ipcw_km_rmst_contrast, cox_rmst_contrast,
                     km_rmst, smd, drop_constant, standardize, adaptive_ridge)


def stage7():
    spec = importlib.util.spec_from_file_location('stage7', ROOT / 'scripts/07_estimate.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def init_worker(inputs):
    global SAMPLE
    SAMPLE = inputs


def boot_bench(b):
    time, event, A, X, tau = SAMPLE
    idx = np.random.default_rng(SEED + 70000 + b).integers(0, len(A), len(A))
    t, e, a, x = time[idx], event[idx], A[idx], X[idx]
    if len(np.unique(a)) < 2:
        return (np.nan, np.nan)
    km = km_rmst(t[a == 1], e[a == 1], tau) - km_rmst(t[a == 0], e[a == 0], tau)
    try:
        cox = cox_rmst_contrast(t, e, a, x, tau)['estimate']
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        cox = np.nan
    return km, cox


def ph_diagnostics(time, event, A, X, columns, population):
    """Schoenfeld screens using lifelines Efron-tie reference refits.

    These are approximate screens for penalized fits, not proof of the PH or
    independent-censoring assumptions. Adjust p-values by Holm within fit.
    """
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test
    xc, keep = drop_constant(X)
    xs, _, _ = standardize(xc)
    names = [str(c) for c, k in zip(columns, keep) if k]
    lam = adaptive_ridge(len(A), xs.shape[1] + 1)
    rows = []
    for process in ('death', 'censoring'):
        data = pd.DataFrame(xs, columns=names)
        if process == 'death':
            data.insert(0, 'chemotherapy_by_landmark', A)
        covnames = list(data.columns)
        # lifelines penalizes coefficients after sample-SD normalization.
        penalties = lam / (len(A) * data.std(ddof=1).to_numpy() ** 2)
        if process == 'death':
            penalties[0] = 0.0
        data['duration'] = time
        data['observed'] = event if process == 'death' else 1 - event
        if process == 'censoring':
            data['treatment_stratum'] = A
        model = CoxPHFitter(penalizer=penalties)
        model.fit(data, 'duration', 'observed',
                  strata=['treatment_stratum'] if process == 'censoring' else None)
        test = proportional_hazard_test(model, data, time_transform='rank').summary
        pvalues = test.p.to_numpy()
        order = np.argsort(pvalues)
        adjusted = np.empty(len(order))
        adjusted[order] = np.minimum(1.0, np.maximum.accumulate(
            pvalues[order] * np.arange(len(order), 0, -1)))
        for (term, row), padj in zip(test.iterrows(), adjusted):
            rows.append(dict(population=population, process=process, term=term,
                             statistic=row.test_statistic, p=row.p, p_holm=padj,
                             reference_ties='Efron', n_covariates=len(covnames),
                             interpretation='exploratory penalized-model PH screen'))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    out = OUTPUT / 'smoke_biometrics' if args.smoke else OUTPUT
    out.mkdir(exist_ok=True)
    B = 20 if args.smoke else 2000
    m = stage7()
    frame = pd.read_csv(OUTPUT / '05_cohort_landmark90.csv', index_col=0)
    frame.index = barcode12(frame.index)
    estimates = pd.read_csv(OUTPUT / '07_estimates.csv')
    comparisons, fits, diagnostics, ph, stabilizers, support = [], [], [], [], [], []
    for key in ('TNBC_chemo', 'ALL_chemo'):
        t, e, a, x, n = m._primary_inputs(frame, key)
        inputs = t, e, a, x, 1095
        nominal = dict(population=key, n=n, events=int(((e == 1) & (t <= 1095)).sum()))
        point_km = km_rmst(t[a == 1], e[a == 1], 1095) - km_rmst(t[a == 0], e[a == 0], 1095)
        point_cox = cox_rmst_contrast(t, e, a, x, 1095)
        print(f'{key}: {B} bootstraps for unadjusted KM and Cox RMST', flush=True)
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                                 initargs=(inputs,)) as pool:
            boot = np.asarray(list(pool.map(boot_bench, range(B), chunksize=10)))
        for j, (name, estimate) in enumerate((('Unadjusted KM', point_km),
                                            ('Cox standardized RMST', point_cox['estimate']))):
            values = boot[:, j]
            ok = np.isfinite(values)
            if ok.mean() < 0.99:
                raise RuntimeError(f'{key} {name}: too many failed bootstraps {sum(~ok)}/{B}')
            lo, hi = np.quantile(values[ok], [.025, .975])
            comparisons.append(dict(**nominal, estimator=name, estimate_days=estimate,
                                    ci_lo=lo, ci_hi=hi, boot_se=np.std(values[ok], ddof=1),
                                    bootstrap_replicates=B, bootstrap_failed=int(sum(~ok)),
                                    target='unadjusted descriptive' if j == 0 else 'standardized landmark'))
        old = estimates[(estimates.arm == key) & (estimates.adjustment == 'CLINICAL') &
                        (estimates.tau == 1095)]
        for _, row in old.iterrows():
            comparisons.append(dict(**nominal, estimator=row.estimator,
                                    estimate_days=row.estimate_days, ci_lo=row.ci_lo,
                                    ci_hi=row.ci_hi, boot_se=row.boot_se,
                                    bootstrap_replicates=row.boot_ok,
                                    bootstrap_failed=row.boot_failed, target='standardized landmark'))
        ap = aipw_rmst(t, e, a, x, 1095, seed=SEED, return_psi=True)
        km = ipcw_km_rmst_contrast(t, e, a, 1095, X=x, return_weights=True)
        for row in ap['outcome_fits']:
            fits.append(dict(population=key, adjustment='CLINICAL', **row))
        xc, _ = drop_constant(x)
        for label, weights in [('unweighted', np.ones(n)),
                               ('primary KM treatment weights', km['treatment_weights']),
                               ('AIPW treatment weights', np.where(a == 1, a.mean() / ap['ps'],
                                                                  (1-a.mean()) / (1-ap['ps'])))]:
            s = np.abs(smd(xc, a, weights))
            diagnostics.append(dict(population=key, weights=label, mean_abs_smd=float(np.mean(s)),
                                    max_abs_smd=float(np.max(s)), n_smd_over_01=int(sum(s > .1))))
        mask = m.arms(frame)[key]['mask']
        cols = m.adjustment_sets(frame, mask, key)['CLINICAL'].columns
        ph.extend(ph_diagnostics(t, e, a, x, cols, key))
        for group in (0, 1):
            for horizon in (365, 730, 1095, 1825):
                support.append(dict(population=key, treatment_arm=group, horizon=horizon,
                                    at_risk=int(((a == group) & (t >= horizon)).sum()),
                                    observed_deaths=int(((a == group)&(t <= horizon)&(e == 1)).sum())))
        variants = [
            ('specified', {}),
            ('propensity clip 0.01/0.99', dict(ps_clip=(.01,.99))),
            ('propensity clip 0.001/0.999', dict(ps_clip=(.001,.999))),
            ('censoring floor 0.02', dict(g_floor=.02)),
            ('censoring floor 0.10', dict(g_floor=.10)),
            ('combined weight cap 0.999', dict(weight_cap_q=.999)),
            ('combined weight cap 0.95', dict(weight_cap_q=.95)),
            ('outcome prediction clipping off', dict(clip_m=False)),
            ('contrast clipping off', dict(clip_estimate=False)),
            ('all five devices relaxed', dict(ps_clip=(.001,.999),g_floor=.01,
                                             weight_cap_q=1.,clip_m=False,clip_estimate=False)),
            ('arm-only censoring', dict(censoring='km'))]
        for label, kw in variants:
            av = aipw_rmst(t,e,a,x,1095,seed=SEED,**kw)['estimate']
            kk = {k:v for k,v in kw.items() if k in ('ps_clip','g_floor','censoring')}
            kv = ipcw_km_rmst_contrast(t,e,a,1095,X=x,**kk)['estimate']
            stabilizers.append(dict(population=key, variant=label, aipw=av, ipcw_km=kv,
                                    aipw_shift=av-ap['estimate'], km_shift=kv-km['estimate']))
        print(f"{key}: Cox {point_cox['estimate']:+.2f}, KM {point_km:+.2f}", flush=True)
    for name, rows in [('20_comparisons',comparisons),('20_outcome_fold_df',fits),
                       ('20_balance',diagnostics),('20_ph_diagnostics',ph),
                       ('20_stabilizers',stabilizers),('20_horizon_support',support)]:
        pd.DataFrame(rows).to_csv(out / (name+'.csv'),index=False)
    print('Wrote Biometrics benchmark and diagnostic tables.', flush=True)


if __name__ == '__main__':
    main()
