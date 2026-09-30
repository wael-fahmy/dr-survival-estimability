"""Generate the matched Biometrics article, Web supplement and review notes.

All empirical quantities are loaded from generated CSV/JSON outputs. Existing
analysis output is kept distinct from added benchmarks and independent stress
simulations. No point estimates are hand-entered in the manuscript template.
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'lib'))
from lib_io import OUTPUT,PAPER,SEED,Results

TITLE='Design and outcome-model estimability in doubly robust survival analysis: a breast cancer reanalysis'


def read(name):return pd.read_csv(OUTPUT/(name+'.csv'))
def num(x,d=1):return f'{float(x):.{d}f}'
def esc(x):
    return str(x).replace('&',r'\&').replace('%',r'\%').replace('_',r'\_').replace('#',r'\#')
def arm(x):return {'TNBC_chemo':'TNBC','ALL_chemo':'Whole cohort'}.get(str(x),str(x))
def method(x):
    return {'IPCW-KM contrast':'Weighted KM','IPCW-KM':'Weighted KM',
            'AIPW (cross-fitted)':'AIPW','Cox standardized RMST':'Cox RMST'}.get(str(x),str(x))
def interval(row,d=1):return f"{num(row.ci_lo,d)}, {num(row.ci_hi,d)}"
def estimate_ci(row,d=1):return f"{num(row.estimate_days,d)} ({interval(row,d)})"


BIB=r'''
\begin{thebibliography}{99}
\bibitem[Bai et al.(2013)]{bai} Bai, X., Tsiatis, A. A., and O'Brien, S. M. (2013).
Doubly-robust estimators of treatment-specific survival distributions in observational studies with stratified sampling.
\textit{Biometrics} \textbf{69}, 830--839.
\bibitem[Cancer Genome Atlas Network(2012)]{tcga} Cancer Genome Atlas Network (2012).
Comprehensive molecular portraits of human breast tumours. \textit{Nature} \textbf{490}, 61--70.
\bibitem[Chernozhukov et al.(2018)]{chernozhukov} Chernozhukov, V., Chetverikov, D., Demirer, M.,
Duflo, E., Hansen, C., Newey, W., et al. (2018). Double/debiased machine learning for treatment
and structural parameters. \textit{The Econometrics Journal} \textbf{21}, C1--C68.
\bibitem[Conner et al.(2019)]{conner} Conner, S. C., Sullivan, L. M., Benjamin, E. J., LaValley, M. P.,
Galea, S., and Trinquart, L. (2019). Adjusted restricted mean survival times in observational studies.
\textit{Statistics in Medicine} \textbf{38}, 3832--3860.
\bibitem[Cox(1972)]{cox} Cox, D. R. (1972). Regression models and life-tables.
\textit{Journal of the Royal Statistical Society, Series B} \textbf{34}, 187--220.
\bibitem[Dafni(2011)]{dafni} Dafni, U. (2011). Landmark analysis at the 25-year landmark point.
\textit{Circulation: Cardiovascular Quality and Outcomes} \textbf{4}, 363--371.
\bibitem[Hern\'an and Robins(2020)]{hernanbook} Hern\'an, M. A. and Robins, J. M. (2020).
\textit{Causal Inference: What If}. Chapman and Hall/CRC, Boca Raton.
\bibitem[Hern\'an et al.(2016)]{hernantrial} Hern\'an, M. A., Sauer, B. C., Hern\'andez-D\'iaz, S.,
Platt, R., and Shrier, I. (2016). Specifying a target trial prevents immortal time bias and
other self-inflicted injuries in observational analyses. \textit{Journal of Clinical Epidemiology} \textbf{79}, 70--75.
\bibitem[Krikun and Alkhateeb(2026)]{krikun} Krikun, E. and Alkhateeb, A. (2026).
A Markov Blanket-based framework for dependency-aware feature selection in multimodal breast cancer data.
\textit{Network Modeling Analysis in Health Informatics and Bioinformatics} \textbf{15}, 158.
\bibitem[Meng and Huang(2026)]{refine} Meng, X. and Huang, J. Y. (2026).
REFINE2: a simplified simulation tool to help epidemiologists evaluate the suitability and sensitivity
of effect estimation within user-specified data. \textit{American Journal of Epidemiology} \textbf{195}, 533--542.
\bibitem[Peduzzi et al.(1996)]{peduzzi} Peduzzi, P., Concato, J., Kemper, E., Holford, T. R.,
and Feinstein, A. R. (1996). A simulation study of the number of events per variable in logistic regression
analysis. \textit{Journal of Clinical Epidemiology} \textbf{49}, 1373--1379.
\bibitem[Ray(2003)]{ray} Ray, W. A. (2003). Evaluating medication effects outside of clinical trials:
new-user designs. \textit{American Journal of Epidemiology} \textbf{158}, 915--920.
\bibitem[Riley et al.(2019)]{riley} Riley, R. D., Snell, K. I. E., Ensor, J., Burke, D. L.,
Harrell, F. E., Jr., Moons, K. G. M., et al. (2019). Minimum sample size for developing a multivariable
prediction model: Part II---binary and time-to-event outcomes. \textit{Statistics in Medicine} \textbf{38}, 1276--1296.
\bibitem[Robins and Finkelstein(2000)]{robins} Robins, J. M. and Finkelstein, D. M. (2000).
Correcting for noncompliance and dependent censoring in an AIDS clinical trial with inverse probability
of censoring weighted (IPCW) log-rank tests. \textit{Biometrics} \textbf{56}, 779--788.
\bibitem[van Smeden et al.(2016)]{vansmeden} van Smeden, M., de Groot, J. A. H., Moons, K. G. M.,
Collins, G. S., Altman, D. G., Eijkemans, M. J. C., et al. (2016).
No rationale for 1 variable per 10 events criterion for binary logistic regression analysis.
\textit{BMC Medical Research Methodology} \textbf{16}, 163.
\bibitem[VanderWeele(2019)]{vanderweele} VanderWeele, T. J. (2019). Principles of confounder selection.
\textit{European Journal of Epidemiology} \textbf{34}, 211--219.
\bibitem[Zhang and Schaubel(2012)]{zhang} Zhang, M. and Schaubel, D. E. (2012).
Double-robust semiparametric estimator for differences in restricted mean lifetimes in observational studies.
\textit{Biometrics} \textbf{68}, 999--1009.
\end{thebibliography}
'''


def values():
    R=Results()
    E=read('07_estimates'); C=read('20_comparisons'); D=read('07_diagnostics')
    N=read('07_nuisance_budget'); F=read('20_outcome_fold_df'); B=read('20_balance')
    P=read('20_ph_diagnostics'); ST=read('20_stabilizers'); SP=read('07_split_repeats')
    fit=read('16_estimability_fit').iloc[0]; L=read('16_level_sensitivity'); PN=read('16_penalty_sensitivity')
    S=read('21_stress_summary'); V0=read('00_estimator_validation'); LAD=read('08_ladder')
    ledger=read('05_cohort_ledger'); v={}
    mapping=dict(discordant=None,n_original='survival_n',tnbc_measured='tnbc_measured_n',
                 tnbc_imputed='tnbc_published_n',n_censored='binary5y_censored_before_coded_survivor',
                 n_all='landmark_cohort_n',n_all_treated='landmark_treated',later_starts='landmark_started_later',
                 observed_alive='binary5y_observed_alive',deaths_5y='binary5y_died_within',
                 deaths_later='binary5y_died_after_coded_survivor',n_sites='n_sites_ge20')
    for k,key in mapping.items():
        if key:v[k]=str(int(R.get(key)))
    v['discordant']=str(int(R.get('T_vs_chemo_misclassified_control')+R.get('T_vs_chemo_misclassified_treated')))
    v['censored_pct']=num(100*int(v['n_censored'])/int(v['n_original']))
    v['median_fu']=num(R.get('survival_median_followup_reverseKM'),0)
    for key,step in [('excluded_landmark',2),('excluded_missing_date',3),('excluded_prediagnosis',4)]:
        v[key]=str(int(ledger[ledger.step==step].iloc[0].n_dropped))
    for prefix,key in [('tn','TNBC_chemo'),('all','ALL_chemo')]:
        for suffix,est in [('km','IPCW-KM contrast'),('aipw','AIPW (cross-fitted)'),
                           ('cox','Cox standardized RMST'),('unadj','Unadjusted KM')]:
            row=C[(C.population==key)&(C.estimator==est)].iloc[0]
            v[f'{prefix}_{suffix}']=num(row.estimate_days)
            v[f'{prefix}_{suffix}_round']=num(row.estimate_days,0)
            v[f'{prefix}_{suffix}_ci']=interval(row)
            v[f'{prefix}_{suffix}_ci_round']=interval(row,0)
        er=E[(E.arm==key)&(E.adjustment=='CLINICAL')&(E.tau==1095)&E.estimator.str.startswith('AIPW')].iloc[0]
        v[f'events_{prefix}']=str(int(er.events))
        if prefix=='tn':
            v['n_tn']=str(int(er.n));v['n_tn_treated']=str(int(er.n_treated));v['n_tn_control']=str(int(er.n_control))
        ns=N[(N.arm==key)&(N.adjustment=='CLINICAL')&(N.component=='outcome model m_a')]
        for a in [0,1]:v[f'{prefix}_complete{a}']=str(int(ns[ns.treatment_arm==a].iloc[0].n_available))
        v[f'{prefix}_p']=str(int(ns.iloc[0].n_parameters))
        v[f'{prefix}_rho']=num(ns.iloc[0].n_parameters/(ns.n_available.min()*.8),2)
        smaller=int(ns.loc[ns.n_available.idxmin()].treatment_arm)
        df=F[(F.population==key)&(F.treatment_arm==smaller)].effective_df
        v[f'{prefix}_edf_range']=f'{num(df.min())}--{num(df.max())}'
        bs=B[(B.population==key)&(B.weights=='primary KM treatment weights')].iloc[0]
        bu=B[(B.population==key)&(B.weights=='unweighted')].iloc[0]
        assert bs.mean_abs_smd>=0 and bu.mean_abs_smd>=0
        v[f'{prefix}_smd_before']=num(bu.mean_abs_smd,3)
        v[f'{prefix}_smd_after']=num(bs.mean_abs_smd,3)
        v[f'{prefix}_smd_max']=num(bs.max_abs_smd,3)
        dr=D[(D.arm==key)&(D.adjustment=='CLINICAL')].iloc[0]
        v[f'{prefix}_ess']=f'{num(dr.ess_est_treated,0)}/{num(dr.ess_est_control,0)}'
        v[f'{prefix}_km_floor']=str(int(dr.km_cens_n_below_floor))
        v[f'{prefix}_ap_floor']=str(int(dr.aipw_cens_n_below_floor))
        sr=SP[SP.arm==key].iloc[0]
        v[f'{prefix}_split_range']=f'{num(sr["min"],0)} to {num(sr["max"],0)} days'
    for prefix,est in [('tn_km','IPCW'),('tn_aipw','AIPW')]:
        subset=E[(E.arm=='TNBC_chemo')&(E.tau==1095)&E.estimator.str.startswith(est)]
        v[prefix+'_min']=num(subset.estimate_days.min());v[prefix+'_max']=num(subset.estimate_days.max())
    for suffix,est in [('km','IPCW'),('aipw','AIPW')]:
        row=E[(E.arm=='TNBC_chemo')&(E.tau==1825)&(E.adjustment=='CLINICAL')&E.estimator.str.startswith(est)].iloc[0]
        v['tn_long_'+suffix]=estimate_ci(row,0)+' days'
    lm=read('07_landmark_sensitivity')
    v['all_day180']=num(lm[(lm.arm=='ALL_chemo')&lm.estimator.str.startswith('AIPW')].iloc[0].estimate_days)
    cohort=read('05_cohort_landmark90')
    v['targeted_n']=str(int(((cohort.HER2_pos==True)&(cohort.start_targeted>=0)&(cohort.start_targeted<=90)).sum()))
    site=read('06_site_summary');site=site[site.n>=20]
    v['site_fu_range']=f'{num(site.median_fu.min(),0)} to {num(site.median_fu.max(),0)}'
    ph=P[P.p_holm<.05]
    v['ph_holm_count']=str(len(ph))
    v['ph_finding']=('; '.join(f"{esc(r.term)} in the {arm(r.population).lower()} {r.process} model (adjusted $p={r.p_holm:.4f}$)" for _,r in ph.iterrows())
                     if len(ph) else 'none crosses the adjusted 0.05 level')
    v['max_stab_shift']=num(ST.aipw_shift.abs().max())
    v['max_km_stab_shift']=num(ST.km_shift.abs().max())
    mc=read('18_montecarlo_summary')
    core=mc[mc.quantity.str.startswith('ALL_chemo')]
    v['seed_reading']='; '.join(f"{('AIPW' if '/ AIPW /' in r.quantity else 'weighted-KM')} {r.conclusion}" for _,r in core.iterrows())
    for name,column in [('fit_intercept','fit_intercept'),('fit_slope','fit_slope'),('fit_r2','fit_r2')]:v[name]=num(fit[column],3)
    v['rho_threshold']=num(fit.threshold_rho_fitted,2)
    v['rho_ci']=f'{num(fit.threshold_rho_lo,2)}--{num(fit.threshold_rho_hi,2)}'
    for key,lev in [('rho_level_low',.25),('rho_level_high',1)]:v[key]=num(L[L.level==lev].iloc[0].threshold_rho,2)
    for key,scale in [('rho_pen_low',.3),('rho_pen_high',3)]:v[key]=num(PN[PN.ridge_scale==scale].iloc[0].threshold_rho,2)
    v['km_grid_max']=num(fit.ipcwkm_max_ratio_anywhere,2);v['movement_ratio']=num(fit.median_excess_aipw_over_km)
    v['stress_reps']=str(int(S.replicates.iloc[0]));v['stress_boot']=str(int(S.inner_bootstraps.iloc[0]))
    adj=S[S.estimator!='Unadjusted KM']
    summaries=[]
    for name,label in [('ph_conditional_censoring','Under proportional hazards with covariate-dependent censoring'),
                       ('delayed_benefit','Under delayed benefit'),('nonlinear_heavy_tail','Under nonlinear heavy-tailed misspecification')]:
        sub=adj[adj.regime==name]
        summaries.append(f"{label}, coverage across adjusted methods and the two sample sizes ranges from {num(100*sub.coverage.min())} to {num(100*sub.coverage.max())}\\%, with absolute bias up to {num(sub.bias.abs().max())} days.")
    worst=adj.loc[adj.coverage.idxmin()]
    summaries.append(f"The lowest adjusted-method coverage occurs for {method(worst.estimator)} at $n={int(worst.n)}$ in the {worst.regime.replace('_',' ')} scenario: {num(100*worst.coverage)}\\% (95\\% Monte Carlo interval {num(100*worst.coverage_mc_lo)}--{num(100*worst.coverage_mc_hi)}\\%).")
    failures=int(S.failed_replicates.sum())
    summaries.append(f"There are {failures} failed estimator--replicate evaluations across the stress study; coverage is conditional on successful evaluations, with failures reported separately.")
    summaries.append('Web Table 10 reports all four methods, including the confounded unadjusted benchmark, with RMSE, bias Monte Carlo errors and coverage uncertainty. The additional sensitivity points in Figure~\\ref{fig:calibration} assess transport of the warning relationship; they do not recalibrate its crossing.')
    v['stress_results']='\n\n'.join(summaries)
    v['original_coverage']=f'{num(100*V0.coverage_95.min())}--{num(100*V0.coverage_95.max())}\\%'
    v['original_bias']=num(V0.bias.abs().max())
    lad=LAD[LAD.rung!=3]
    v['ladder_sequence']=', '.join(num(x) for x in lad.rmst_estimate)
    v['ladder_shift']=num(lad.rmst_estimate.iloc[-1]-lad.rmst_estimate.iloc[0])
    v['estimator_gap']=num(float(v['tn_km'])-float(v['tn_aipw']))
    v['binary_sequence']=', '.join(num(x) for x in LAD[LAD.rung<3].estimate)
    comparisons=[]
    for key in ['TNBC_chemo','ALL_chemo']:
        for name in ['Unadjusted KM','IPCW-KM contrast','AIPW (cross-fitted)','Cox standardized RMST']:
            r=C[(C.population==key)&(C.estimator==name)].iloc[0]
            comparisons.append(f'{arm(key)} & {method(name)} & {estimate_ci(r)} \\\\')
    v['comparison_rows']='\n'.join(comparisons)
    labels=['Original coding and imputed TNBC','Structured chemotherapy field','Measured receptors only','Day-90 landmark','Clinical adjustment']
    v['ladder_rows']='\n'.join(f'{j} & {labels[j]} & {int(r.n)} & {int(r.rmst_events)} & {num(r.rmst_estimate)} ({num(r.rmst_ci_lo)}, {num(r.rmst_ci_hi)}) \\\\' for j,(_,r) in enumerate(lad.iterrows()))
    v['bibliography']=BIB
    return v


def table(caption,headers,rows,widths):
    spec=''.join('p{'+str(w)+'in}' for w in widths)
    head=' & '.join(headers)+r' \\'
    return '\n'.join([r'\begin{longtable}{'+spec+'}',r'\caption{'+caption+r'}\\',r'\toprule',head,
                       r'\midrule\endfirsthead',r'\toprule',head,r'\midrule\endhead',r'\bottomrule\endfoot',
                       *[' & '.join(map(str,row))+r' \\' for row in rows],r'\end{longtable}'])


def supplement(v):
    text=[r'''\documentclass[12pt,letterpaper]{article}
\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb,booktabs,longtable,array,graphicx,setspace}
\usepackage[hidelinks]{hyperref}
\setstretch{1.85}
\setlength{\parskip}{0pt}
\setlength{\emergencystretch}{1em}
\setlength{\tabcolsep}{3pt}
\renewcommand{\thetable}{Web \arabic{table}}
\renewcommand{\thefigure}{Web \arabic{figure}}
\renewcommand{\thesection}{\Alph{section}}
\begin{document}
\begin{center}\textbf{Supplementary Material}\\'''+esc(TITLE)+r'''\end{center}
This supplement and the main article are generated together from the supplied numerical outputs.
Clinical specification denotes the clinical adjustment set, covariate-conditional censoring and a 1095-day
post-landmark horizon unless stated otherwise. The older reference validation and calibration are separated
from the added Cox comparisons and independent stress scenarios.

\section{Estimators, assumptions and fixed preprocessing}
All expectations in this appendix refer to an eligible landmark population. Let $R=\min(T,\tau)$,
$Y=\min(\widetilde T,\tau)$, and $D$ indicate an observed death by $\tau$ or follow-up reaching $\tau$.
Write $m_a(X)=E(R\mid A=a,X)$ and $e(X)=\Pr(A=1\mid X)$.
Before stabilization, the complete-outcome AIPW contribution is
\begin{align*}
\psi(O)&=m_1(X)-m_0(X)\\
&\quad+\frac{AD}{e(X)G(Y^-\mid 1,X)}\{Y-m_1(X)\}\\
&\quad-\frac{(1-A)D}{\{1-e(X)\}G(Y^-\mid 0,X)}\{Y-m_0(X)\}.
\end{align*}
Under conditional independent censoring, $E(D\mid T,A,X)=G(R^-\mid A,X)$, with the stated
tie convention. For correct $G$, conditioning on $A,X$ gives
$E[D\{Y-m_a(X)\}/G(Y^-\mid a,X)\mid A=a,X]=m_a^0(X)-m_a(X)$,
where superscript 0 denotes the true conditional mean. Consequently,
\begin{align*}
E\{\psi(O)\mid X\}&=m_1(X)-m_0(X)\\
&\quad+\frac{e_0(X)}{e(X)}\{m_1^0(X)-m_1(X)\}\\
&\quad-\frac{1-e_0(X)}{1-e(X)}\{m_0^0(X)-m_0(X)\}.
\end{align*}
This recovers $m_1^0-m_0^0$ if $e=e_0$ or both outcome means are correct, conditional on correct $G$.
It does not provide robustness to arbitrary misspecification of $G$, and fixed weight truncation need not
retain this identity. It is not the full censoring-augmented efficient influence function.

Outcome regressions use complete cases weighted by $1/\widehat G$ inside treatment arms.
The propensity is ridge logistic; the censoring process uses ridge Cox with treatment strata and a
Breslow baseline. The unpenalized intercept is included in outcome/propensity parameter counts.
The adaptive penalty is $\lambda=\max\{1,p\max(1,10p/n)\}$ in the objective conventions in the code.
For AIPW propensity and outcome regressions, $p,n$ refer to the full input matrix and sample;
censoring fits compute their own training-sample penalty. Coefficients are trained on four of five folds.
Folds are stratified jointly by treatment and observed death within the horizon.

Clinical stage and TNM are ordinally encoded. Numeric missing values use full-cohort medians with
missing indicators. Sites with fewer than 20 patients in the landmark cohort are pooled; site coding is
fixed before restricting to a subgroup. Receptors defining TNBC are never imputed. The archived multimodal
matrix retains its previous processing; residual nonfinite entries are set to zero. These encodings and
clinical imputations are fixed for resampling. Within an AIPW input matrix, propensity/outcome scaling
also precedes fold splitting, whereas censoring-model scaling uses training observations. Bootstrap
refits therefore condition on the constructed feature matrix and do not reproduce original feature discovery.
No claim of fully nested model/feature selection is made.

Weighted KM clips propensity at 0.01/0.99; AIPW uses 0.02/0.98. Both floor $G$ at 0.05.
AIPW caps positive combined weights at the pooled 99th percentile, clips outcome means to $[0,\tau]$
and clips the contrast to $[-\tau,\tau]$. The Cox outcome comparator instead has one unpenalized treatment
coefficient, ridge-penalized standardized covariates, and a Breslow death-hazard baseline. Its arm-specific
survival predictions are averaged over the same observed covariate distribution and integrated exactly
between baseline-hazard knots. It does not use a separate censoring regression, relying on conditional
independent censoring and correct specification of the outcome model. Treatment contrast and hazard ratio
are distinct; only standardized RMST enters the comparison.

\section{Cohort audit and observed follow-up}
The starting frame is the supplied earlier cohort, not every patient ever enrolled in TCGA-BRCA.
The following ledger is sequential: each exclusion count refers to the preceding row's survivors.
Measured-receptor subgroup construction and original follow-up are shown in Web Figures 1--2.
The treatment-field comparison establishes discordance, not an externally validated error rate.
''']
    ledger=read('05_cohort_ledger')
    text.append(table('Day-90 analysis cohort ledger.', ['Rule','Excluded','Remaining'],
        [[esc(r.rule),'--' if pd.isna(r.n_dropped) else str(int(r.n_dropped)),str(int(r.n_after))] for _,r in ledger.iterrows()],[4.3,.7,.8]))
    text.append(r'''\section{Complete effect estimates and proportional-hazards diagnostics}
Web Table 2 reports the original adjustment-set and horizon sensitivity estimates.
The new unadjusted KM and Cox clinical benchmarks and their bootstrap intervals are in main Table 1.
They use 2000 new patient bootstrap draws, paired with each other. Earlier KM intervals for a correction
ladder rung use their original seed, so Monte Carlo differences in endpoints do not imply different point estimates.
The complete proportional-hazards screens appear later in Web Table 11.
''')
    est=read('07_estimates')
    text.append(table('All original adjusted RMST estimates. Intervals are patient-bootstrap percentile intervals.',
        ['Population/set','Horizon','Estimator','RMST (95\% interval)'],
        [[arm(r.arm)+' / '+str(r.adjustment).title(),str(int(r.tau)),method(r.estimator),estimate_ci(r)] for _,r in est.iterrows()],[1.6,.65,1.1,2.45]))
    bal=read('20_balance')
    text.append(table('Estimator-specific balance. Values are absolute standardized mean differences (SMD). Treatment weights alone assess baseline balance; complete-outcome censoring weights define a different diagnostic.',
        ['Population','Weight definition','Mean SMD','Max SMD','No. $>0.1$'],
        [[arm(r.population),esc(r.weights),num(r.mean_abs_smd,3),num(r.max_abs_smd,3),str(int(r.n_smd_over_01))] for _,r in bal.iterrows()],[1.0,2.1,.8,.8,.9]))
    text.append(r'''\section{Censoring, horizon support and five-device stabilization}
Censoring diagnostics are evaluated at each complete case's restricted time. They differ between AIPW's
training-fold censoring fits and the primary estimator's full-sample fit. The combined-weight ESS values
reported in the article refer to capped AIPW complete-outcome residual weights; they are not an effective
sample size for the time-varying KM risk sets. Selected-horizon risk counts and estimator weight diagnostics
are supplied as CSV files. Sites with short follow-up motivate, but do not validate, conditional censoring models.
''')
    ce=read('07_censoring_sensitivity')
    text.append(table('Original censoring-model sensitivity. Floor counts are complete cases with fitted censoring survival below 0.05.',
        ['Population/method','Model','RMST (95\% interval)','$G_{\min}$','Floor'],
        [[arm(r.arm)+' / '+method(r.estimator),'Cox' if str(r.censoring_model).startswith('Cox') else 'Arm KM',estimate_ci(r),num(r.cens_G_min,3),str(int(r.cens_n_below_floor))] for _,r in ce.iterrows()],[1.7,.65,2.25,.65,.55]))
    st=read('20_stabilizers')
    text.append(table('Updated stabilizer sensitivity, in RMST days. The joint relaxation now disables both prediction and contrast clipping, sets the propensity limits to 0.001/0.999 and censoring floor to 0.01, and removes the combined-weight cap. These remain near-untruncated rather than literally unbounded weights.',
        ['Population','Variant','AIPW','Weighted KM'],
        [[arm(r.population),esc(str(r.variant).replace('_',' ')),num(r.aipw),num(r.ipcw_km)] for _,r in st.iterrows()],[1.0,3.0,.8,1.0]))
    text.append(r'''\section{Landmark, fold information and seed stability}
Changing to day 180 changes both eligibility and exposure status. It is a sensitivity analysis for a different
landmark population, not a bound on selection bias. The outcome effective degrees of freedom are computed
from the actual training-fit matrix and Cox-derived weights as
$\operatorname{tr}\{D(D^\top WD+P)^{-1}D^\top W\}$, where the intercept entry of $P$ is zero.
Fold-specific complete counts, dimension, penalty and effective degrees of freedom are supplied in
\texttt{20\_outcome\_fold\_df.csv}. Nominal rho counts use complete outcomes, not deaths alone.
''')
    lm=read('07_landmark_sensitivity')
    text.append(table('Day-180 landmark sensitivity at a 1095-day horizon.', ['Population','$n$/deaths','Estimator','RMST (95\% interval)'],
        [[arm(r.arm),f'{int(r.n)}/{int(r.events)}',method(r.estimator),estimate_ci(r)] for _,r in lm.iterrows()],[1.2,.8,1.1,2.7]))
    mc=read('18_montecarlo_summary')
    text.append(table('Original six-seed repeat. This separates computational reproducibility from partition and simulation Monte Carlo variability.',
        ['Quantity','MC SD','Mean interval width','Interval reading'],
        [[esc(str(r.quantity).replace('TNBC_chemo / ','TNBC ').replace('ALL_chemo / ','Whole cohort ').replace(' / RMST difference at 1095 d','')),num(r.mc_sd,2),num(r.mean_ci_width,2),esc(r.conclusion)] for _,r in mc.iterrows()],[1.85,.65,1.0,2.2]))
    text.append(r'''\section{Original diagnostic calibration}
The 32-cell calibration uses four sample sizes (150, 300, 600, 1000), four covariate dimensions
(10, 25, 50, 75) and two baseline event-rate settings (0.12, 0.30). Each cell has 60 replicates.
Covariates are independent standard Gaussian. Treatment follows a logistic model with coefficients
$(0.7,-0.5,0.3)$ on the first three covariates and a centered linear predictor plus log odds for 0.6.
Survival is exponential with hazard $b\exp(0.5X_1+0.3X_2)0.65^A$, where
$b=-\log(1-r)/1095$. Censoring is exponential with mean $1.4\times1095$ and independent of covariates.
Adjustment sets are nested prefixes of size $\max(3,p/4)$, $\max(4,p/2)$ and $p$, with integer division.
All sets contain the assignment predictors. Sensitivity is the mean within-replicate range divided by
the full-set estimator's Monte Carlo sampling SD. The reported rho is averaged over replicates in a cell.
The log-linear fit and its crossing use 2000 bootstrap resamples of grid cells. These intervals reflect
variation across the selected cells and fit, not uncertainty over all possible data-generating regimes.
They also do not propagate every source of within-cell Monte Carlo uncertainty.
Penalty scales 0.3 and 3 use 30 replicates per cell; the specified scale uses 60.
The nested sets do not establish a lower bound on sensitivity under arbitrary real adjustment sets.
''')
    lev=read('16_level_sensitivity');pen=read('16_penalty_sensitivity')
    rows=[['Crossing',num(r.level,2),num(r.threshold_rho,2),f'{num(r.threshold_rho_lo,2)}, {num(r.threshold_rho_hi,2)}'] for _,r in lev.iterrows()]
    rows += [['Penalty multiplier',num(r.ridge_scale,1),num(r.threshold_rho,2),f'{num(r.threshold_rho_lo,2)}, {num(r.threshold_rho_hi,2)}'] for _,r in pen.iterrows()]
    text.append(table('Original warning-level sensitivities. The specified crossing is 0.50; the threshold is not an estimator-validity criterion.', ['Varied','Value','Threshold','95\% cell-bootstrap'],rows,[1.9,.65,1.0,2.15]))
    text.append(r'''\section{Validation against known truth and independent stress scenarios}
The original operating-point validation uses 500 replicates per scenario and 500 inner patient bootstrap
replicates. It simulates Gaussian covariates and exponential survival, logistic treatment and censoring
depending on a survival predictor. The input covariate counts are 25 and 28 before adding an intercept;
these approximate, rather than exactly match, the clinical outcome-model dimensions. Its event targets
and treated fractions are application-motivated. Empirical coverage cannot be assigned to the clinical
intervals merely because sample size and event counts are similar.
''')
    va=read('00_estimator_validation')
    text.append(table('Original operating-point validation. SD is empirical estimate SD; SE is mean bootstrap SE. Coverage refers only to these simulated scenarios.',
        ['Scenario/method','Truth','Bias','SD/SE','Coverage'],
        [[str(r.scenario)+' / '+method(r.estimator),num(r.truth),num(r.bias),f'{num(r.empirical_sd)}/{num(r.mean_bootstrap_se)}',num(100*r.coverage_95)+r'\%'] for _,r in va.iterrows()],[1.85,.7,.7,1.1,1.1]))
    text.append(r'''The independent stress study has six cells: $n=136,p=24$ or $n=988,p=27$, crossed with
three mechanisms. There are 200 replicates and 199 inner patient bootstraps per cell. New simulation seeds
are distinct from the original calibration. Gaussian scenarios use independent standard-normal covariates;
the nonlinear scenario uses independent $t_3/\sqrt{3}$ covariates. The default propensity is
$\operatorname{expit}(0.7X_1-0.5X_2+0.3X_3)$. The control death hazard is
$h_0=b\exp(0.5X_1+0.3X_2)$, and the censoring hazard is
$(\log 2/1100)\exp(0.5X_1+0.5X_3)$. Thus censoring depends on measured survival predictors.
The PH treatment hazard ratio is 0.65. The delayed-benefit scenario instead uses treatment hazard ratio
1.2 through day 365 and 0.45 thereafter; it violates a constant treatment hazard ratio in the Cox outcome model.

In the nonlinear heavy-tail scenario, add $0.6\ind(X_1>1)$ to the propensity linear predictor,
replace the death predictor by $0.5X_1+0.5\sin(X_2)+0.3(X_3^2-1)$, and add $0.35(X_2^2-1)$
to the censoring predictor. Linear nuisance regressions are then misspecified. To keep the generated
hazards finite, death/censoring predictors are clipped to $[-4,4]$ and the propensity predictor to $[-20,20]$.
The baseline $b$ is calibrated by deterministic-seed numerical averaging over 30000 reference covariates
to expected observed deaths near 16 or 72. The exact fitted baseline constants are saved in
\texttt{21\_stress\_design.json}; achieved mean deaths are in the replicate and summary CSV files.

For hazards $h_1$ before $u=365$ and $h_2$ afterward, the restricted mean is
$(1-e^{-h_1u})/h_1+e^{-h_1u}(1-e^{-h_2(\tau-u)})/h_2$.
The treated-minus-control difference is averaged over each replicate's covariates to obtain its exact
conditional standardized truth. Four estimators are compared against this truth; the unadjusted KM
benchmark is expected to retain confounding. All estimators use their stated penalties and stabilization.
The AIPW fold seed is fixed within a replicate's bootstrap procedure. A replicate is evaluable for a method
only if its point estimate is finite and at least 95\% of its inner fits succeed. Failures are reported;
coverage is conditional on evaluability. Bias Monte Carlo SE is the SD of estimation errors divided by
the square root of evaluable replicates. Wilson intervals describe binomial Monte Carlo uncertainty in coverage.
RMSE is relative to each replicate's conditional truth. The full table also reports unadjusted KM;
the main coverage figure shows the three adjusted methods.
'''.replace(r'\ind',r'\mathbf{1}'))
    st=read('21_stress_summary')
    regimes={'ph_conditional_censoring':'PH / conditional censoring','delayed_benefit':'Delayed benefit','nonlinear_heavy_tail':'Nonlinear / heavy tail'}
    text.append(table('Independent stress validation. Bias is accompanied by its Monte Carlo SE. Coverage includes a 95\% Wilson Monte Carlo interval. Failures are out of 200 replicates per cell and estimator; all individual results are supplied.',
        ['Scenario / $n$ / method','Bias (MC SE)','RMSE','Coverage (MC interval)','Failed'],
        [[regimes[r.regime]+' / '+str(int(r.n))+' / '+method(r.estimator),f'{num(r.bias)} ({num(r.bias_mcse)})',num(r.rmse),f'{num(100*r.coverage)} ({num(100*r.coverage_mc_lo)}, {num(100*r.coverage_mc_hi)})',str(int(r.failed_replicates))] for _,r in st.iterrows()],[2.0,1.0,.6,1.7,.55]))
    text.append(r'''Schoenfeld rank-time screens use lifelines Efron-tie reference refits for the death and censoring
processes. Penalties are converted to the code's scale, the treatment coefficient is unpenalized in the
death model, and censoring is stratified by treatment. Holm adjustment is within each model over covariates.
These exploratory checks use penalized fits, are not exact finite-sample tests, and cannot verify independent
censoring. They are reported as potential model-fit concerns, not automatic model acceptance/rejection.
''')
    ph=read('20_ph_diagnostics')
    text.append(table('Complete proportional-hazards screens. Raw and within-model Holm-adjusted p-values; Efron reference refits.',
        ['Population/process','Covariate','Raw $p$','Holm $p$'],
        [[arm(r.population)+' / '+str(r.process),esc(r.term),f'{r.p:.4g}',f'{r.p_holm:.4g}'] for _,r in ph.iterrows()],[1.55,2.25,.9,1.1]))
    text.append(r'''\section{Exploratory heterogeneity and scope of validation}
The earlier analysis gated heterogeneity at 50 deaths, a convention rather than a guarantee of usable
power. Only whole-cohort estimates are retained here. Their AIPW scores are cross-fitted for nuisance
estimation, but the later linear ranking was fitted and evaluated on the same cohort. The old subgroup,
BLP and RATE intervals resampled already computed scores rather than the full estimation procedure.
They are therefore omitted from this submission's heterogeneity table. This avoids presenting conditional
fixed-score intervals as if they reflected all nuisance, ranking and selection uncertainty.
The following estimates are descriptive only and do not establish targeting ability or effect modification.
Confirmatory RATE evaluation would require an independently learned ranking or an appropriate additional
cross-fitting scheme; nominal significance of the old interval is not used as evidence.
''')
    het=read('07_heterogeneity')
    mde=read('07_mde')
    text.append('For context, the earlier approximate minimum-detectable subgroup differences are '+
                '; '.join(arm(r.arm)+': '+num(r.mde_subgroup_difference_days)+' days' for _,r in mde.iterrows())+
                r'. These use $2(z_{0.975}+z_{0.80})s/\sqrt{nq(1-q)}$, where $q$ is the treated fraction and $s$ is the restricted-time SD among complete cases. They are rough complete-data calculations, not power calculations for the weighted survival estimator, censoring or a prespecified interaction test. The 50-death gate and this approximation do not establish adequate power.')
    text.append(table('Whole-cohort descriptive heterogeneity summaries. No confirmatory intervals are assigned. BLP denotes best linear projection; AUTOC is an in-sample ranking summary.',
        ['Analysis','Term','$n$','Estimate, days'],
        [[esc(r.analysis),esc(r.term).replace('>=',r'$\ge$').replace('<',r'$<$'),str(int(r.n)),num(r.estimate)] for _,r in het.iterrows()],[1.45,2.55,.6,1.15]))
    text.append(r'''Classification and clustering metrics do not evaluate the RMST contrast. Biological gene-set
enrichment would require a defined gene universe and molecular-selection target; no new signature is
proposed here. Statistical validation instead examines the specified estimators against known RMST truth,
alongside support, fit and sensitivity diagnostics. The related prognostic study addresses transport and
incremental predictive utility separately.

\section{Reproducibility, software and files}
The code-and-data supplement provides the clinical and derived inputs, generated CSV outputs, software
requirements, seed and source hashes, and a single pipeline entry point. The analysis uses NumPy/Pandas;
figures use Matplotlib, and the comparator cross-checks and PH screens use lifelines/SciPy. LaTeX compilation
is included in the documented build command. The original Kaplan--Meier, RMST and Cox-censoring cross-checks
are retained in \texttt{15\_lifelines\_crosscheck.csv}. The new Cox RMST implementation is independently
checked against lifelines on untied unpenalized data, under treatment relabeling, constant-column addition
and feasible-range constraints. Agreement of a numerical component does not validate a clinical model.

The supplied raw clinical exports and derived survival/feature tables suffice for the analyses in this paper.
The original raw omics matrices and Xena source export are not reconstructed. Input SHA-256 digests and
the distinction between supplied inputs and upstream provenance are documented in the plain-text README.
The bootstrap and simulation replication counts are recorded in their output tables. Reduced smoke runs
write into a separate directory and are never used to generate the submission PDFs.

\section{Related work, archived implementation and scope}
The earlier covariate list was intended as an input to a published feature-selection framework's downstream
analysis. The archived implementation accepted a feature-list argument but substituted a univariate
penalized-Cox screen. The source location is in the feature-extraction tree. This observation concerns that
archived routine; it does not establish which routine generated the published framework's results, and
no claim is made that those published results are affected or unaffected. The archived source file is
available from the authors on request. The feature-selection framework itself
is not re-evaluated in the current manuscript.

@@related_work@@

Generative AI assistance with coding, documentation and editorial preparation is disclosed in the article
and cover letter. No grant supported this reanalysis. All authors must approve the final manuscript and
contributor statements before submission.
''')
    for path,caption in [
        ('fig1_cohort_flow','Cohort construction and measured TNBC definition. Each exclusion is counted.'),
        ('fig2_censoring','Original-cohort survival and follow-up, and ascertainment of the five-year binary endpoint.'),
        ('fig4_overlap_balance','Earlier union-set TNBC propensity and balance display, retained as a sensitivity diagnostic. Main-text clinical-primary balance is separately recomputed from the primary estimator weights.'),
        ('fig5_site_confounding','Site-level treatment and follow-up summaries. Association alone does not establish that site is a confounder.')]:
        text.append(r'\clearpage\begin{figure}[p]\setstretch{1.85}\centering\includegraphics[width=\textwidth,height=0.70\textheight,keepaspectratio]{figures/'+path+r'.pdf}\caption{'+caption+r'}\end{figure}')
    text.append(r'\end{document}')
    return '\n'.join(text)


def main():
    v=values()
    # Related-work metadata can be corrected without hand-editing generated LaTeX.
    metadata=json.loads((ROOT/'submission_metadata.json').read_text(encoding='utf-8'))
    v['related_work']=esc(metadata['related_work_text'])
    def fill(s):
        def repl(m):
            if m.group(1) not in v:raise KeyError('Missing manuscript value '+m.group(1))
            return str(v[m.group(1)])
        return re.sub(r'@@([A-Za-z0-9_]+)@@',repl,s)
    article=fill((PAPER/'templates/biometrics-article.tex.in').read_text(encoding='utf-8'))
    si=fill(supplement(v))
    for name,content in [('sn-article.tex',article),('supplement.tex',si)]:
        assert '@@' not in content
        (PAPER/name).write_text(content,encoding='utf-8')
    # A machine-readable record makes manuscript substitution auditable.
    (OUTPUT/'22_manuscript_values.json').write_text(json.dumps(v,indent=2),encoding='utf-8')
    print('Generated matched Biometrics article and Web Appendices A-J / Tables 1-12 / Figures 1-4.')


if __name__=='__main__':main()
