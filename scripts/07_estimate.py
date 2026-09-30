"""
07_estimate.py — Phase 3 estimation.

Estimand
    Difference in restricted mean survival time at 1095 days after the day-90
    landmark (secondary: 1825 days), among patients alive and uncensored at the
    landmark, comparing chemotherapy initiated by day 90 with not initiated by
    day 90. Reported in days of life gained or lost over the horizon.

    RMST is used instead of the earlier binary five-year mortality because
    67.4% of the cohort is censored before day 1825; a binary indicator has to
    guess at those patients and the audited pipeline guessed "survived".

Censoring
    The identification assumption is C _||_ T | A, X. The censoring model
    implements it: a ridge-penalised Cox model for the censoring hazard,
    stratified by arm, with the adjustment covariates as regressors, fitted
    within the cross-fitting scheme for AIPW and on the full sample for the
    weighting-only estimator. The arm-only Kaplan-Meier censoring model, which
    assumes C _||_ T | A, is re-run as a sensitivity analysis so the effect of
    the weaker assumption is visible (07_censoring_sensitivity.csv).

Estimators
    * IPTW- and IPCW-weighted Kaplan-Meier contrast  (primary)
    * cross-fitted AIPW  (sensitivity)
    Both are reported for every arm and adjustment set. They are not averaged.

    The ordering is justified by a parameter count, not by which estimator
    produced the narrower interval: AIPW carries two outcome regressions the
    TNBC arm cannot estimate, and IPCW-KM does not carry them. See
    nuisance_budget() below. The ordering was fixed AFTER the estimates were
    computed and is not a prespecification; that is disclosed in the report and
    the manuscript.

Adjustment sets (side by side; the clinical set is primary)
    CLINICAL   age, stage, T, N, M, calendar era, contributing site
               (grade and tumour size are unavailable -- see 05_landmark.py).
               Primary, on causal grounds: each variable is a plausible common
               cause of treatment and survival, or a proxy for one.
    EXISTING   the 24 variables of the earlier analysis, with ER/PR/HER2
               dropped inside TNBC where they are constant by construction.
               Sensitivity: outcome-predictive features are prognostic, not
               thereby confounders.
    UNION      both. Sensitivity.

Uncertainty
    Patient-level nonparametric bootstrap, 2000 replicates, for every reported
    estimate. No analytic interval is quoted anywhere. Where an interval appears
    it is a bootstrap percentile interval for the stated estimand.

Produces:
    07_estimates.csv               ->  Table 3 and Figure 3
    07_diagnostics.csv             ->  Table 4 and Figure 4
    07_trimming.csv                ->  Diagnostics text
    07_mde.csv                     ->  Table 5
    07_heterogeneity.csv           ->  Table S5
    07_stabiliser_sensitivity.csv  ->  Table S1
    07_landmark_sensitivity.csv    ->  Table S3
    07_nuisance_budget.csv         ->  Section 5 text
    07_censoring_sensitivity.csv   ->  Table S6
    07_split_repeats.csv           ->  Split-instability subsection
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from lib_io import DATA, OUTPUT, SEED, Results, banner, barcode12, require  # noqa: E402
from lib_est import (aipw_rmst, bootstrap, ipcw_components,  # noqa: E402
                     ipcw_km_rmst_contrast, km_rmst, mde_heterogeneity,
                     mde_rmst, smd, effective_sample_size, wls_fit,
                     add_intercept, standardize, drop_constant,
                     adaptive_ridge, ridge_effective_df, restricted_outcome)

# REVISION_BOOT exists only so the pipeline can be smoke-tested quickly. The
# reported analysis uses 2000 and the manuscript records the value actually
# used, so a reduced run cannot masquerade as the real one.
B_BOOT = int(os.environ.get("REVISION_BOOT") or 2000)
# Under stage 18 (REVISION_OUTPUT set) this stage computes only what stage 18
# harvests: the primary adjustment set at the primary horizon, both estimators,
# each with the full bootstrap. The sensitivity analyses, the diagnostics table,
# the split repeats and the heterogeneity block are not read by stage 18 and
# are not recomputed there; the scratch copy keeps the canonical versions.
MINIMAL = bool(os.environ.get("REVISION_OUTPUT"))
TAUS = [1095] if MINIMAL else [1095, 1825]
TRIMS = [0.0, 0.01, 0.05, 0.10]
HETEROGENEITY_EVENT_FLOOR = 50
PRIMARY_SET = "CLINICAL"
N_SPLIT_REPEATS = 50
K_FOLDS = 5

STAGE_MAP = {"Stage I": 1, "Stage IA": 1, "Stage IB": 1,
             "Stage II": 2, "Stage IIA": 2, "Stage IIB": 2,
             "Stage III": 3, "Stage IIIA": 3, "Stage IIIB": 3, "Stage IIIC": 3,
             "Stage IV": 4}


# ── Covariate construction ────────────────────────────────────────────────────

def ordinal(series, mapping, prefix):
    """Ordinal code plus an explicit missingness indicator. Nothing is silently
    filled: the indicator carries the information that the value was absent."""
    v = series.map(mapping)
    miss = v.isna().astype(float)
    filled = v.fillna(v.median() if v.notna().any() else 0.0).astype(float)
    return pd.DataFrame({f"{prefix}": filled, f"{prefix}_missing": miss},
                        index=series.index)


def tnm_code(series, prefix):
    s = series.astype(str).str.upper().str.extract(r"^[TNM](\d)")[0]
    v = pd.to_numeric(s, errors="coerce")
    miss = v.isna().astype(float)
    filled = v.fillna(v.median() if v.notna().any() else 0.0).astype(float)
    return pd.DataFrame({prefix: filled, f"{prefix}_missing": miss},
                        index=series.index)


def build_clinical(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    age = pd.to_numeric(df["age"], errors="coerce")
    out.append(pd.DataFrame({
        "age": age.fillna(age.median()),
        "age_missing": age.isna().astype(float),
    }, index=df.index))
    out.append(ordinal(df["stage"], STAGE_MAP, "stage"))
    out.append(tnm_code(df["t_stage"], "t_stage"))
    out.append(tnm_code(df["n_stage"], "n_stage"))
    out.append(tnm_code(df["m_stage"], "m_stage"))
    yr = pd.to_numeric(df["year_dx"], errors="coerce")
    out.append(pd.DataFrame({
        "year_dx": yr.fillna(yr.median()),
        "year_dx_missing": yr.isna().astype(float),
    }, index=df.index))
    # Contributing site: one-hot for sites with >= 20 patients, rest pooled.
    counts = df["site"].value_counts()
    big = counts[counts >= 20].index
    site = df["site"].where(df["site"].isin(big), "OTHER")
    out.append(pd.get_dummies(site, prefix="site").astype(float))
    return pd.concat(out, axis=1)


def build_existing24(df: pd.DataFrame) -> pd.DataFrame:
    v2 = pd.read_csv(require(DATA / "ite_ready_dataset_v2.csv",
                             "the earlier analysis's 24-variable adjustment set"))
    v2["patient_id"] = barcode12(v2["patient_id"])
    v2 = v2.drop_duplicates("patient_id").set_index("patient_id")
    drop = ["T", "Y", "propensity_score", "T_hormone", "T_chemo",
            "T_targeted", "T_radiation", "T_hormone_excl"]
    X = v2.drop(columns=[c for c in drop if c in v2.columns], errors="ignore")
    return X.reindex(df.index)


def adjustment_sets(df: pd.DataFrame, mask, arm_key: str) -> dict:
    """The three adjustment sets for one arm, in reporting order."""
    ex = build_existing24(df).loc[mask].copy()
    if arm_key == "TNBC_chemo":
        ex = ex.drop(columns=[c for c in ("ER_status", "PR_status", "HER2_status")
                              if c in ex.columns])
    cl = build_clinical(df).loc[mask].copy()
    un = pd.concat([ex, cl], axis=1)
    un = un.loc[:, ~un.columns.duplicated()]
    return {"CLINICAL": cl, "EXISTING": ex, "UNION": un}


def as_matrix(Xdf: pd.DataFrame) -> np.ndarray:
    return np.nan_to_num(Xdf.to_numpy(dtype=float), nan=0.0, posinf=0.0,
                         neginf=0.0)


# ── Arm definitions ───────────────────────────────────────────────────────────

def arms(df: pd.DataFrame) -> dict:
    def treated_at_landmark(col):
        s = df[f"start_{col}"]
        return ((s.notna()) & (s <= 90) & (s >= 0)).astype(int)

    # The hormone-therapy arm in HR+/HER2- disease is deliberately NOT analysed
    # here. That population is the subject of a separate manuscript in
    # preparation by a co-author, using a different landmark, horizon and
    # design; running it here would duplicate the population and create a
    # self-overlap problem at submission. It is excluded by design, not because
    # of anything the data showed.
    return {
        "TNBC_chemo": dict(
            label="Chemotherapy in TNBC",
            mask=df["TNBC"].fillna(False).astype(bool).to_numpy(),
            A=treated_at_landmark("chemo").to_numpy()),
        "HER2pos_targeted": dict(
            label="Targeted therapy in HER2+",
            mask=df["HER2_pos"].fillna(False).astype(bool).to_numpy(),
            A=treated_at_landmark("targeted").to_numpy()),
        "ALL_chemo": dict(
            label="Chemotherapy, whole cohort",
            mask=np.ones(len(df), dtype=bool),
            A=treated_at_landmark("chemo").to_numpy()),
    }


def _boot_pair(time, event, A, Xv, tau, censoring="cox"):
    """Bootstrap closures for the two estimators on one configuration."""
    def boot_aipw(idx):
        if len(np.unique(A[idx])) < 2:
            return np.nan
        return aipw_rmst(time[idx], event[idx], A[idx], Xv[idx], tau,
                         seed=SEED, censoring=censoring)["estimate"]

    def boot_km(idx):
        if len(np.unique(A[idx])) < 2:
            return np.nan
        return ipcw_km_rmst_contrast(time[idx], event[idx], A[idx], tau,
                                     X=Xv[idx], seed=SEED,
                                     censoring=censoring)["estimate"]
    return boot_aipw, boot_km


CENS_KEYS = ("cens_G_min", "cens_G_p05", "cens_n_below_floor",
             "cens_ipc_weight_max", "cens_ipc_ess", "cens_converged",
             "cens_beta_max")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    banner("07 — Estimation (RMST difference, IPCW, cross-fitted AIPW)")
    res = Results()

    df = pd.read_csv(OUTPUT / "05_cohort_landmark90.csv", index_col=0)
    df.index = barcode12(df.index)

    X_clin = build_clinical(df)
    X_exist = build_existing24(df)
    print(f"  cohort {len(df):,} patients, {int(df['event'].sum()):,} deaths")
    print(f"  CLINICAL adjustment set : {X_clin.shape[1]} variables "
          f"(primary; grade and tumour size unavailable)")
    print(f"  EXISTING adjustment set : {X_exist.shape[1]} variables")
    print(f"  bootstrap replicates    : {B_BOOT}")
    if MINIMAL:
        print("  MINIMAL mode (stage 18): primary set, primary horizon, both "
              "estimators with full bootstrap; sensitivity blocks skipped")

    ARMS = arms(df)
    rows, diag_rows, trim_rows = [], [], []
    het_rows, mde_rows, cens_rows, split_rows = [], [], [], []

    for arm_key, arm in ARMS.items():
        m = arm["mask"]
        A = arm["A"][m]
        time = df["time"].to_numpy()[m]
        event = df["event"].to_numpy()[m].astype(int)
        n = int(m.sum())

        print(f"\n{'-' * 74}\n  {arm['label']}  (n={n:,})")
        if len(np.unique(A)) < 2 or min((A == 1).sum(), (A == 0).sum()) < 10:
            print(f"    SKIPPED: n treated={int((A==1).sum())}, "
                  f"n control={int((A==0).sum())} — one arm too small to estimate.")
            continue

        sets = adjustment_sets(df, m, arm_key)
        if MINIMAL:
            sets = {PRIMARY_SET: sets[PRIMARY_SET]}
        if arm_key == "TNBC_chemo" and "EXISTING" in sets:
            print(f"    receptor variables dropped inside TNBC "
                  f"-> EXISTING has {sets['EXISTING'].shape[1]} variables")

        for tau in TAUS:
            ev = int(((event == 1) & (time <= tau)).sum())
            ev1 = int(((event == 1) & (time <= tau) & (A == 1)).sum())
            ev0 = int(((event == 1) & (time <= tau) & (A == 0)).sum())
            print(f"\n    tau={tau}d   events={ev} "
                  f"(treated {ev1}, control {ev0})   "
                  f"n treated={int((A==1).sum())}, control={int((A==0).sum())}")

            for set_name, Xdf in sets.items():
                Xv = as_matrix(Xdf)

                a = aipw_rmst(time, event, A, Xv, tau, seed=SEED, return_psi=True)
                k = ipcw_km_rmst_contrast(time, event, A, tau, X=Xv, seed=SEED)
                boot_aipw, boot_km = _boot_pair(time, event, A, Xv, tau)
                ci_a = bootstrap(boot_aipw, n, B=B_BOOT, seed=SEED)
                ci_k = bootstrap(boot_km, n, B=B_BOOT, seed=SEED + 1)

                for est_name, pt, ci, extra in (
                    ("AIPW (cross-fitted)", a["estimate"], ci_a, a),
                    ("IPCW-KM contrast", k["estimate"], ci_k, k),
                ):
                    rows.append(dict(
                        arm=arm_key, arm_label=arm["label"], adjustment=set_name,
                        primary_set=(set_name == PRIMARY_SET),
                        estimator=est_name, tau=tau, n=n,
                        n_treated=int((A == 1).sum()), n_control=int((A == 0).sum()),
                        events=ev, events_treated=ev1, events_control=ev0,
                        estimate_days=pt, ci_lo=ci["lo"], ci_hi=ci["hi"],
                        boot_se=ci["se"], boot_ok=ci["B_ok"], boot_failed=ci["B_failed"],
                        weight_cap=extra.get("weight_cap"),
                        n_weights_capped=extra.get("n_weights_capped"),
                        out_of_range=extra.get("out_of_range"),
                        ess_est_treated=extra.get("ess_treated"),
                        ess_est_control=extra.get("ess_control"),
                        censoring_model=extra.get("censoring_model"),
                        **{key: extra.get(key) for key in CENS_KEYS},
                    ))
                    print(f"      {set_name:9s} {est_name:22s} "
                          f"{pt:+8.1f} d  [{ci['lo']:+8.1f}, {ci['hi']:+8.1f}]"
                          f"   (bootstrap {ci['B_ok']}/{B_BOOT}; "
                          f"G_min {extra['cens_G_min']:.3f}, "
                          f"{extra['cens_n_below_floor']} below floor)")

                # ── Diagnostics, primary horizon only ────────────────────────
                if tau == TAUS[0] and not MINIMAL:
                    ps = a["ps"]
                    Xc, keep = drop_constant(Xv)
                    Xs, _, _ = standardize(Xc)
                    p1 = A.mean()
                    w = np.where(A == 1, p1 / ps, (1 - p1) / (1 - ps))
                    s_un = smd(Xs, A)
                    s_w = smd(Xs, A, w)
                    _, _, _, g_km = ipcw_components(time, event, tau, strata=A)
                    diag_rows.append(dict(
                        arm=arm_key, adjustment=set_name,
                        primary_set=(set_name == PRIMARY_SET),
                        ridge=a["ridge"],
                        fallback_propensity=a["fallback_propensity"],
                        fallback_outcome=a["fallback_outcome"],
                        min_complete_per_arm_fold=a["min_complete_per_arm_fold"],
                        n_outcome_model_params=int(Xc.shape[1] + 1),
                        weight_cap=a["weight_cap"],
                        n_weights_capped=a["n_weights_capped"],
                        # arm-only Kaplan-Meier censoring curve, for comparison
                        censoring_G_min_armonly=float(np.min(g_km)),
                        n_G_below_floor_armonly=int((g_km < 0.05).sum()),
                        # covariate-conditional censoring model, as used:
                        # cross-fitted inside AIPW and full-sample for IPCW-KM
                        **{"aipw_" + key: a[key] for key in CENS_KEYS},
                        **{"km_" + key: k[key] for key in CENS_KEYS},
                        ps_min=float(ps.min()), ps_max=float(ps.max()),
                        ps_p01=float(np.percentile(ps, 1)),
                        ps_p99=float(np.percentile(ps, 99)),
                        ps_treated_mean=float(ps[A == 1].mean()),
                        ps_control_mean=float(ps[A == 0].mean()),
                        n_ps_below_01=int((ps < 0.01).sum()),
                        n_ps_above_99=int((ps > 0.99).sum()),
                        smd_mean_unweighted=float(np.abs(s_un).mean()),
                        smd_max_unweighted=float(np.abs(s_un).max()),
                        smd_mean_weighted=float(np.abs(s_w).mean()),
                        smd_max_weighted=float(np.abs(s_w).max()),
                        n_smd_gt_01_unweighted=int((np.abs(s_un) > 0.1).sum()),
                        n_smd_gt_01_weighted=int((np.abs(s_w) > 0.1).sum()),
                        # ESS of the stabilised IPTW weights: a balance and
                        # overlap diagnostic.
                        ess_iptw_treated=effective_sample_size(w[A == 1]),
                        ess_iptw_control=effective_sample_size(w[A == 0]),
                        # ESS of the combined weight actually used in the
                        # estimating equation (IPTW x IPCW). This is the one
                        # that governs precision.
                        ess_est_treated=a["ess_treated"],
                        ess_est_control=a["ess_control"],
                        weight_max=float(w.max()), weight_p99=float(np.percentile(w, 99)),
                    ))
                    for tr in TRIMS:
                        at = aipw_rmst(time, event, A, Xv, tau, seed=SEED, trim=tr)
                        trim_rows.append(dict(
                            arm=arm_key, adjustment=set_name, trim=tr,
                            estimate_days=at["estimate"], n_kept=at["n_kept"],
                            n_trimmed=at["n_trimmed"]))

                    # ── Split instability: the same estimate under repeated
                    # fold partitions. Reported as a first-class diagnostic
                    # because the whole-cohort AIPW interval's relationship to
                    # zero depends on the seed (stage 18).
                    split_rows.append(split_repeats(arm_key, arm["label"],
                                                    set_name, time, event, A,
                                                    Xv, tau, a["estimate"]))

                # ── Censoring-model sensitivity, primary set only ────────────
                if tau == TAUS[0] and set_name == PRIMARY_SET and not MINIMAL:
                    cens_rows.extend(censoring_sensitivity(
                        arm_key, arm["label"], set_name, time, event, A, Xv,
                        tau, n, a, ci_a, k, ci_k))

            # ── Heterogeneity gate ───────────────────────────────────────────
            if tau == TAUS[0] and not MINIMAL:
                Xv = as_matrix(sets[PRIMARY_SET])
                Y, complete = restricted_outcome(time, event, tau)
                sd_out = float(np.std(Y[complete == 1])) if complete.any() else float("nan")
                mde = mde_rmst(sd_out, (A == 1).sum(), (A == 0).sum())
                mde_het = mde_heterogeneity(ev, sd_out, (A == 1).sum(), (A == 0).sum())
                mde_rows.append(dict(
                    arm=arm_key, arm_label=arm["label"], tau=tau, events=ev,
                    sd_restricted_time=sd_out,
                    mde_main_effect_days=mde,
                    mde_subgroup_difference_days=mde_het,
                    meets_event_floor=(ev >= HETEROGENEITY_EVENT_FLOOR),
                ))
                print(f"\n      events at tau={tau}: {ev}  "
                      f"(floor for heterogeneity: {HETEROGENEITY_EVENT_FLOOR})")
                if ev < HETEROGENEITY_EVENT_FLOOR:
                    print(f"      HETEROGENEITY NOT ESTIMABLE in this arm.")
                    print(f"        With {ev} events the smallest detectable "
                          f"difference between two equal-sized subgroups' effects")
                    print(f"        is {mde_het:,.0f} days of restricted mean "
                          f"survival at 80% power -- larger than the whole")
                    print(f"        plausible effect. No BLP, RATE, quintile "
                          f"calibration or subgroup point estimate is reported.")
                else:
                    print(f"      events clear the floor — running heterogeneity.")
                    het_rows.extend(
                        heterogeneity(arm_key, arm["label"], df, m, A, time,
                                      event, Xv, sets[PRIMARY_SET].columns, tau, n))

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT / "07_estimates.csv", index=False)
    if MINIMAL:
        res.update({"estimates": out.to_dict(orient="records"),
                    "n_bootstrap": B_BOOT, "taus": TAUS,
                    "primary_adjustment_set": PRIMARY_SET,
                    "minimal_mode": True})
        print(f"\n  wrote output/07_estimates.csv ({len(out)} rows), minimal mode")
        return

    sens_rows = landmark_sensitivity()
    stab_rows = stabiliser_sensitivity(df)
    nuis_rows = nuisance_budget(df)
    pd.DataFrame(diag_rows).to_csv(OUTPUT / "07_diagnostics.csv", index=False)
    pd.DataFrame(trim_rows).to_csv(OUTPUT / "07_trimming.csv", index=False)
    pd.DataFrame(mde_rows).to_csv(OUTPUT / "07_mde.csv", index=False)
    pd.DataFrame(sens_rows).to_csv(OUTPUT / "07_landmark_sensitivity.csv",
                                   index=False)
    pd.DataFrame(stab_rows).to_csv(OUTPUT / "07_stabiliser_sensitivity.csv",
                                   index=False)
    pd.DataFrame(nuis_rows).to_csv(OUTPUT / "07_nuisance_budget.csv",
                                   index=False)
    pd.DataFrame(cens_rows).to_csv(OUTPUT / "07_censoring_sensitivity.csv",
                                   index=False)
    pd.DataFrame(split_rows).to_csv(OUTPUT / "07_split_repeats.csv", index=False)
    if het_rows:
        pd.DataFrame(het_rows).to_csv(OUTPUT / "07_heterogeneity.csv", index=False)

    res.update({
        "estimates": out.to_dict(orient="records"),
        "diagnostics": diag_rows, "trimming": trim_rows,
        "mde": mde_rows, "heterogeneity": het_rows,
        "landmark_sensitivity": sens_rows,
        "stabiliser_sensitivity": stab_rows,
        "nuisance_budget": nuis_rows,
        "censoring_sensitivity": cens_rows,
        "split_repeats": split_rows,
        "n_bootstrap": B_BOOT, "taus": TAUS,
        "primary_adjustment_set": PRIMARY_SET,
        "n_split_repeats": N_SPLIT_REPEATS,
        "heterogeneity_event_floor": HETEROGENEITY_EVENT_FLOOR,
        "existing_set_size": int(X_exist.shape[1]),
        "clinical_set_size": int(X_clin.shape[1]),
    })
    print(f"\n  wrote output/07_estimates.csv ({len(out)} rows)")


def censoring_sensitivity(arm_key, label, set_name, time, event, A, Xv, tau, n,
                          a_cox, ci_a_cox, k_cox, ci_k_cox):
    """The same configuration under the arm-only censoring model.

    The specified model conditions G on A and X. The arm-only model conditions
    on A alone, which is the stronger assumption the earlier version of this
    analysis made without stating it. Both are bootstrapped so the change in
    the interval, not only the point, is visible.
    """
    print(f"\n      censoring-model sensitivity ({set_name}, tau={tau}): "
          f"arm-only Kaplan-Meier")
    a_km = aipw_rmst(time, event, A, Xv, tau, seed=SEED, censoring="km")
    k_km = ipcw_km_rmst_contrast(time, event, A, tau, X=Xv, seed=SEED,
                                 censoring="km")
    boot_a, boot_k = _boot_pair(time, event, A, Xv, tau, censoring="km")
    ci_a_km = bootstrap(boot_a, n, B=B_BOOT, seed=SEED)
    ci_k_km = bootstrap(boot_k, n, B=B_BOOT, seed=SEED + 1)
    rows = []
    for est_name, model_label, r, ci in (
            ("AIPW (cross-fitted)", "Cox on A and X (specified)", a_cox, ci_a_cox),
            ("AIPW (cross-fitted)", "arm-only Kaplan-Meier", a_km, ci_a_km),
            ("IPCW-KM contrast", "Cox on A and X (specified)", k_cox, ci_k_cox),
            ("IPCW-KM contrast", "arm-only Kaplan-Meier", k_km, ci_k_km)):
        rows.append(dict(
            arm=arm_key, arm_label=label, adjustment=set_name, tau=tau,
            estimator=est_name, censoring_model=model_label,
            estimate_days=r["estimate"], ci_lo=ci["lo"], ci_hi=ci["hi"],
            **{key: r[key] for key in CENS_KEYS}))
        print(f"        {est_name:22s} {model_label:28s} {r['estimate']:+8.1f} d  "
              f"[{ci['lo']:+8.1f}, {ci['hi']:+8.1f}]   G_min {r['cens_G_min']:.3f} "
              f"below floor {r['cens_n_below_floor']}  max IPC weight "
              f"{r['cens_ipc_weight_max']:.1f}")
    return rows


def split_repeats(arm_key, label, set_name, time, event, A, Xv, tau, fixed):
    """Cross-fitted AIPW under N_SPLIT_REPEATS fold partitions.

    A fixed seed gives computational reproducibility, not statistical
    stability. The median across partitions is the aggregation Chernozhukov et
    al. (2018) recommend; the spread is the diagnostic. Point estimates only:
    the bootstrap interval is reported for the fixed-seed estimate, and stage 18
    repeats the bootstrap under alternate seeds.
    """
    ests = np.array([aipw_rmst(time, event, A, Xv, tau, seed=SEED + 1000 * s)["estimate"]
                     for s in range(1, N_SPLIT_REPEATS + 1)])
    row = dict(arm=arm_key, arm_label=label, adjustment=set_name, tau=tau,
               n_splits=int(len(ests)), fixed_seed_estimate=float(fixed),
               median=float(np.median(ests)), mean=float(ests.mean()),
               sd=float(ests.std(ddof=1)), min=float(ests.min()),
               max=float(ests.max()), q25=float(np.percentile(ests, 25)),
               q75=float(np.percentile(ests, 75)),
               n_positive=int((ests > 0).sum()), n_negative=int((ests < 0).sum()))
    print(f"      split repeats ({set_name}): fixed-seed {fixed:+.1f}, median of "
          f"{len(ests)} partitions {row['median']:+.1f}, range "
          f"[{row['min']:+.1f}, {row['max']:+.1f}], sd {row['sd']:.1f}")
    return row


def _primary_inputs(df, arm_key, set_name=PRIMARY_SET):
    """Return (time, event, A, X, n) for one arm under one adjustment set."""
    ARMS = arms(df)
    arm = ARMS[arm_key]
    m = arm["mask"]
    X = adjustment_sets(df, m, arm_key)[set_name]
    return (df["time"].to_numpy()[m], df["event"].to_numpy()[m].astype(int),
            arm["A"][m], as_matrix(X), int(m.sum()))


def stabiliser_sensitivity(df, tau=1095):
    """Vary every shrinkage device, one at a time and then all together.

    Five devices are applied in the primary specification: propensity clipping,
    the censoring-survival floor, the combined-weight cap, clipping the outcome
    models to [0, tau], and clipping the estimate to [-tau, tau]. Each is
    documented individually elsewhere; what matters here is their combined
    effect, which no single diagnostic shows. The censoring model itself is
    varied last, so its effect can be read against the others.
    """
    print(f"\n{'-' * 74}\n  Stabiliser sensitivity ({PRIMARY_SET} set, tau={tau})")
    rows = []
    for arm_key, label in (("TNBC_chemo", "Chemotherapy in TNBC"),
                           ("ALL_chemo", "Chemotherapy, whole cohort")):
        time, event, A, X, n = _primary_inputs(df, arm_key)
        variants = [
            ("as specified", dict()),
            ("propensity clip 0.01/0.99", dict(ps_clip=(0.01, 0.99))),
            ("propensity clip off (0.001/0.999)", dict(ps_clip=(0.001, 0.999))),
            ("censoring floor 0.02", dict(g_floor=0.02)),
            ("censoring floor 0.10", dict(g_floor=0.10)),
            ("weight cap at 0.999 quantile", dict(weight_cap_q=0.999)),
            ("weight cap at 0.95 quantile", dict(weight_cap_q=0.95)),
            ("all relaxed together", dict(ps_clip=(0.001, 0.999), g_floor=0.01,
                                          weight_cap_q=1.0)),
            ("censoring model: arm-only Kaplan-Meier", dict(censoring="km")),
        ]
        base = None
        for name, kw in variants:
            r = aipw_rmst(time, event, A, X, tau, seed=SEED, **kw)
            k = ipcw_km_rmst_contrast(time, event, A, tau, X=X, seed=SEED,
                                      ps_clip=kw.get("ps_clip", (0.02, 0.98)),
                                      g_floor=kw.get("g_floor", 0.05),
                                      censoring=kw.get("censoring", "cox"))
            if base is None:
                base = r["estimate"]
            rows.append(dict(
                arm=arm_key, arm_label=label, variant=name,
                aipw=r["estimate"], ipcw_km=k["estimate"],
                shift_vs_specified=r["estimate"] - base,
                n_weights_capped=r["n_weights_capped"],
                weight_max=r["weight_max"],
                out_of_range=r["out_of_range"]))
            print(f"    {label[:26]:26s} {name:40s} AIPW {r['estimate']:+8.1f}  "
                  f"IPCW-KM {k['estimate']:+8.1f}  "
                  f"(shift {r['estimate'] - base:+.1f})")
    return rows


def nuisance_budget(df, tau=1095):
    """How much information each nuisance model has, per arm and adjustment set.

    This is the evidence for which estimator is primary, and it is computable
    without looking at any effect estimate: it counts parameters against the
    complete cases available to fit them. The effective degrees of freedom of
    the ridge fit are reported beside the nominal count, because the penalty
    makes the nominal count an overstatement; rho as defined uses the nominal
    count, and the effective version is recorded for the reader.
    """
    print(f"\n{'-' * 74}\n  Nuisance-model budget (tau={tau})")
    rows = []
    for arm_key, label in (("TNBC_chemo", "Chemotherapy in TNBC"),
                           ("ALL_chemo", "Chemotherapy, whole cohort")):
        for set_name in ("CLINICAL", "EXISTING", "UNION"):
            time, event, A, X, n = _primary_inputs(df, arm_key, set_name)
            Y, complete, w_ipc, _ = ipcw_components(time, event, tau, strata=A)
            Xc, _ = drop_constant(X)
            Xs, _, _ = standardize(Xc)
            D = add_intercept(Xs)
            p = D.shape[1]
            lam = adaptive_ridge(n, p)
            ev = int(((event == 1) & (time <= tau)).sum())
            for a in (0, 1):
                sel = (A == a) & (complete == 1)
                cc = int(sel.sum())
                per_fold = cc * (1 - 1 / K_FOLDS)
                df_eff = ridge_effective_df(D[sel], w_ipc[sel], lam) if cc >= 2 else float("nan")
                rows.append(dict(
                    arm=arm_key, arm_label=label, adjustment=set_name,
                    primary_set=(set_name == PRIMARY_SET),
                    component="outcome model m_a",
                    treatment_arm=a, n_parameters=p, n_available=cc,
                    n_available_per_training_fold=int(per_fold),
                    params_per_observation=round(p / max(per_fold, 1), 2),
                    ridge=lam, effective_df=df_eff,
                    effective_df_per_observation=round(df_eff / max(per_fold, 1), 2)))
            rows.append(dict(
                arm=arm_key, arm_label=label, adjustment=set_name,
                primary_set=(set_name == PRIMARY_SET),
                component="propensity model e(X)",
                treatment_arm=-1, n_parameters=p, n_available=n,
                n_available_per_training_fold=int(n * (1 - 1 / K_FOLDS)),
                params_per_observation=round(p / max(n * (1 - 1 / K_FOLDS), 1), 2),
                ridge=lam, effective_df=float("nan"),
                effective_df_per_observation=float("nan")))
            rows.append(dict(
                arm=arm_key, arm_label=label, adjustment=set_name,
                primary_set=(set_name == PRIMARY_SET),
                component="censoring Cox on X (per arm baseline)",
                treatment_arm=-1, n_parameters=p - 1,
                n_available=int((event == 0).sum()),
                n_available_per_training_fold=int((event == 0).sum() * (1 - 1 / K_FOLDS)),
                params_per_observation=round((p - 1) / max((event == 0).sum() * (1 - 1 / K_FOLDS), 1), 2),
                ridge=lam, effective_df=float("nan"),
                effective_df_per_observation=float("nan")))
            worst = max(rows[-4]["params_per_observation"],
                        rows[-3]["params_per_observation"])
            print(f"    {label[:26]:26s} {set_name:8s} events={ev:3d}  p={p:3d}  "
                  f"complete cases per arm="
                  f"{int(((A==0)&(complete==1)).sum())}/"
                  f"{int(((A==1)&(complete==1)).sum())}  rho={worst:.2f}")
    return rows


def landmark_sensitivity(tau=1095):
    """Repeat the primary specification at a day-180 landmark.

    26% of dated chemotherapy starts fall after day 90, so where the landmark is
    placed changes who counts as treated. Moving it to day 180 reclassifies most
    of those patients as treated and costs a further slice of the cohort. If the
    conclusion depends on that choice, it should be visible.
    """
    print(f"\n{'-' * 74}\n  Landmark sensitivity: day 180 vs day 90 ({PRIMARY_SET} "
          f"set, tau={tau})")
    path = OUTPUT / "05_cohort_landmark180.csv"
    if not path.exists():
        print("    day-180 cohort not found — skipped")
        return []

    df = pd.read_csv(path, index_col=0)
    df.index = barcode12(df.index)

    rows = []
    for arm_key, mask_col, start_col, label in (
            ("TNBC_chemo", "TNBC", "start_chemo", "Chemotherapy in TNBC"),
            ("ALL_chemo", None, "start_chemo", "Chemotherapy, whole cohort")):
        m = (df[mask_col].fillna(False).astype(bool).to_numpy()
             if mask_col else np.ones(len(df), dtype=bool))
        st = df[start_col]
        A = ((st.notna()) & (st <= 180) & (st >= 0)).astype(int).to_numpy()[m]
        time = df["time"].to_numpy()[m]
        event = df["event"].to_numpy()[m].astype(int)
        Xa = as_matrix(adjustment_sets(df, m, arm_key)[PRIMARY_SET])
        n = int(m.sum())
        if min((A == 1).sum(), (A == 0).sum()) < 10:
            print(f"    {label}: SKIPPED, n1={int((A==1).sum())} "
                  f"n0={int((A==0).sum())}")
            continue

        a = aipw_rmst(time, event, A, Xa, tau, seed=SEED)
        k = ipcw_km_rmst_contrast(time, event, A, tau, X=Xa, seed=SEED)
        boot_a, boot_k = _boot_pair(time, event, A, Xa, tau)
        ci_a = bootstrap(boot_a, n, B=B_BOOT, seed=SEED)
        ci_k = bootstrap(boot_k, n, B=B_BOOT, seed=SEED + 1)
        ev = int(((event == 1) & (time <= tau)).sum())
        for est_name, pt, ci in (("AIPW (cross-fitted)", a["estimate"], ci_a),
                                 ("IPCW-KM contrast", k["estimate"], ci_k)):
            rows.append(dict(landmark=180, arm=arm_key, arm_label=label,
                             adjustment=PRIMARY_SET, estimator=est_name, tau=tau,
                             n=n, n_treated=int((A == 1).sum()),
                             n_control=int((A == 0).sum()), events=ev,
                             estimate_days=pt, ci_lo=ci["lo"], ci_hi=ci["hi"]))
            print(f"    {label:30s} {est_name:22s} {pt:+8.1f} d  "
                  f"[{ci['lo']:+8.1f}, {ci['hi']:+8.1f}]  "
                  f"n={n} ({int((A==1).sum())}/{int((A==0).sum())}) ev={ev}")
    return rows


def heterogeneity(arm_key, label, df, m, A, time, event, Xv, colnames, tau, n):
    """Best linear predictor of the CATE, prespecified group effects, quintile
    calibration and RATE/AUTOC. Only called for arms clearing the event floor.

    The event floor of 50 is a prespecified convention: ten events per
    parameter (Peduzzi et al., 1996) for a heterogeneity model carrying up to
    five terms, a rule of thumb with known weaknesses (van Smeden et al., 2016)
    that is used here as a gate, not as a guarantee.
    """
    rows = []
    a = aipw_rmst(time, event, A, Xv, tau, seed=SEED, return_psi=True)
    psi = a["psi"]

    sub = df.loc[m]
    age = pd.to_numeric(sub["age"], errors="coerce").to_numpy()
    n_stage = sub["n_stage"].astype(str).str.upper().str.extract(r"^N(\d)")[0]
    n_stage = pd.to_numeric(n_stage, errors="coerce").to_numpy()
    stage = sub["stage"].map(STAGE_MAP).to_numpy(dtype=float)

    groups = {
        "age < 50": age < 50,
        "age >= 50": age >= 50,
        "node negative (N0)": n_stage == 0,
        "node positive (N1-3)": n_stage >= 1,
        "stage I-II": stage <= 2,
        "stage III-IV": stage >= 3,
    }
    for gname, g in groups.items():
        g = np.asarray(g, dtype=bool) & np.isfinite(psi)
        if g.sum() < 20:
            rows.append(dict(arm=arm_key, analysis="group effect", term=gname,
                             n=int(g.sum()), estimate=np.nan, ci_lo=np.nan,
                             ci_hi=np.nan, note="group too small (<20)"))
            continue
        ci = bootstrap(lambda idx, _g=g, _p=psi: _p[idx][_g[idx]].mean()
                       if _g[idx].sum() > 5 else np.nan, n, B=B_BOOT, seed=SEED)
        rows.append(dict(arm=arm_key, analysis="group effect", term=gname,
                         n=int(g.sum()), estimate=float(psi[g].mean()),
                         ci_lo=ci["lo"], ci_hi=ci["hi"], note=""))

    # Best linear predictor of the CATE on the prespecified modifiers.
    M = np.column_stack([
        np.nan_to_num(age, nan=np.nanmedian(age)),
        np.nan_to_num(n_stage, nan=0.0) >= 1,
        np.nan_to_num(stage, nan=2.0) >= 3,
    ]).astype(float)
    M = M - M.mean(axis=0)
    D = add_intercept(M)
    coef = wls_fit(D, psi, ridge=1e-6)
    terms = ["intercept (ATE)", "age (centred)", "node positive", "stage III-IV"]
    for j, t in enumerate(terms):
        ci = bootstrap(
            lambda idx, _D=D, _p=psi, _j=j: wls_fit(_D[idx], _p[idx], ridge=1e-6)[_j],
            n, B=B_BOOT, seed=SEED + 7)
        rows.append(dict(arm=arm_key, analysis="BLP of CATE", term=t, n=n,
                         estimate=float(coef[j]), ci_lo=ci["lo"], ci_hi=ci["hi"],
                         note=""))

    # Quintile calibration and RATE/AUTOC on the fitted linear CATE. The
    # quintile means are point estimates only; the quintile membership is
    # itself fitted on the same data, so a bootstrap interval for them would
    # understate the uncertainty. They are reported in the Supporting
    # Information, marked as such.
    tau_hat = D @ coef
    q = pd.qcut(pd.Series(tau_hat), 5, labels=False, duplicates="drop")
    for k in sorted(pd.unique(q.dropna())):
        sel = (q == k).to_numpy()
        rows.append(dict(arm=arm_key, analysis="CATE calibration",
                         term=f"quintile {int(k) + 1}", n=int(sel.sum()),
                         estimate=float(psi[sel].mean()),
                         ci_lo=np.nan, ci_hi=np.nan,
                         note="predicted %.1f d" % tau_hat[sel].mean()))

    def autoc(idx=None):
        p = psi if idx is None else psi[idx]
        th = tau_hat if idx is None else tau_hat[idx]
        order = np.argsort(-th)
        pv = p[order]
        cum = np.cumsum(pv) / np.arange(1, len(pv) + 1)
        overall = pv.mean()
        qs = np.arange(1, len(pv) + 1) / len(pv)
        toc = cum - overall
        return float(np.trapezoid(toc, qs))

    ci = bootstrap(lambda idx: autoc(idx), n, B=B_BOOT, seed=SEED + 11)
    rows.append(dict(arm=arm_key, analysis="RATE (AUTOC)", term="AUTOC", n=n,
                     estimate=autoc(), ci_lo=ci["lo"], ci_hi=ci["hi"], note=""))
    return rows


if __name__ == "__main__":
    main()
