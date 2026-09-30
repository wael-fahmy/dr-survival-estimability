"""
16_estimability_diagnostic.py — when is the outcome model of a doubly robust
estimator starved, and can you tell before you estimate anything?

Motivation, from the applied analysis. In the triple-negative arm the
cross-fitted AIPW estimate moves across three adjustment sets by an amount many
times larger than the IPCW-weighted Kaplan-Meier contrast on the same data, and
its sign is decided by where the weight cap sits. In the whole-cohort arm the
AIPW estimate is far steadier across adjustment sets. The difference between the
two arms is not the estimator and not the effect size; it is how much
information the outcome regressions have to work with.

"Use a doubly robust estimator" is standard advice, and it comes with no
guidance on when the outcome-model half of it is too starved to contribute.
This script proposes a diagnostic for that, computable before any estimate
exists, and calibrates it by simulation.

The diagnostic
--------------
For a cross-fitted doubly robust estimator with K folds, an outcome model
carrying p parameters (including the intercept, after constant columns are
dropped), and c_min complete cases in the smaller treatment arm in the full
sample (patients whose restricted time is fully observed), define

    rho = p / (c_min * (1 - 1/K))

the number of outcome-model parameters per usable observation in the arm that
constrains the fit: c_min * (1 - 1/K) is the number of complete cases that arm
contributes to one training fold. It needs no outcome values, no treatment
effect and no fitted model: p is a property of the design matrix, c_min of the
follow-up.

What is calibrated
------------------
Across a grid of sample size, event rate and covariate dimension, we measure how
far each estimator moves when the adjustment set is changed among three nested
choices, and compare that movement to the estimator's own sampling variability.
An estimator whose adjustment-set sensitivity is small relative to its standard
error is reporting the data; one whose sensitivity dominates is reporting the
covariate list.

The crossing level (movement equal to 50% of the estimator's own sampling
standard deviation) is a calibration convention. It is not tied to a
conventional operating characteristic such as size or coverage, and the
threshold it produces is a simulation-calibrated warning level for the studied
regime, not a general cutoff. Two sensitivity analyses make the dependence
visible: the threshold at other crossing levels (from the same fit), and the
threshold when the ridge penalty in every nuisance model is scaled down or up.

Output
------
    output/16_estimability_grid.csv        one row per grid cell and estimator
    output/16_estimability_fit.csv         the calibrated rho-to-sensitivity map
    output/16_level_sensitivity.csv        threshold at other crossing levels
    output/16_penalty_sensitivity.csv      threshold under other penalties

Produces:
    16_estimability_grid.csv     ->  Section 5.2
    16_estimability_fit.csv      ->  Section 5.2 and Equation (3)
    16_level_sensitivity.csv     ->  Table S7
    16_penalty_sensitivity.csv   ->  Table S7
"""

from __future__ import annotations

import os
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from lib_io import OUTPUT, SEED, Results, banner  # noqa: E402
from lib_est import (aipw_rmst, ipcw_components,  # noqa: E402
                     ipcw_km_rmst_contrast)

TAU = 1095
K_FOLDS = 5
# REVISION_SIM_REPS exists only for smoke-testing the pipeline; the reported
# calibration uses 60 replicates per cell and the output records the value.
N_REP = int(os.environ.get("REVISION_SIM_REPS") or 60)
LEVEL = 0.5                       # the calibration convention
OTHER_LEVELS = [0.25, 0.5, 0.75, 1.0]
PENALTY_SCALES = [0.3, 3.0]       # the specified penalty is scale 1
N_REP_PENALTY = min(30, N_REP)

# Grid. n and the censoring rate together set the event count; p sets the
# outcome-model dimension. Chosen to bracket the two real arms: the
# triple-negative arm sits near (n=136, p=25 to 42) and the whole cohort near
# (n=988, p=28 to 52).
GRID_N = [150, 300, 600, 1000]
GRID_P = [10, 25, 50, 75]
GRID_EVENT_RATE = [0.12, 0.30]     # approximate fraction with an event by tau

# Under stage 18 (REVISION_OUTPUT set) only the quantities stage 18 harvests
# are recomputed; the penalty sweep is a one-off sensitivity analysis.
UNDER_MONTE_CARLO = bool(os.environ.get("REVISION_OUTPUT"))


def simulate(rng, n, p, event_rate, hr=0.65, p_treated=0.6, confound=1.0):
    X = rng.normal(size=(n, p))
    lp = confound * (0.7 * X[:, 0] - 0.5 * X[:, 1] + 0.3 * X[:, 2])
    lp = lp - lp.mean() + np.log(p_treated / (1 - p_treated))
    A = (rng.random(n) < 1 / (1 + np.exp(-lp))).astype(int)
    frail = np.exp(0.5 * X[:, 0] + 0.3 * X[:, 1])
    base = -np.log(1 - event_rate) / TAU
    hz = base * frail * np.where(A == 1, hr, 1.0)
    T = rng.exponential(1 / hz)
    C = rng.exponential(TAU * 1.4, n)
    time, event = np.minimum(T, C), (T <= C).astype(int)
    h1, h0 = base * frail * hr, base * frail
    truth = float(np.mean((1 - np.exp(-h1 * TAU)) / h1)
                  - np.mean((1 - np.exp(-h0 * TAU)) / h0))
    return X, A, time, event, truth


def rho_of(X, A, time, event, p_used, k=K_FOLDS):
    """The diagnostic. Uses only the design and the follow-up."""
    _, complete, _, _ = ipcw_components(time, event, TAU, strata=A)
    c_min = min(int(((A == 0) & (complete == 1)).sum()),
                int(((A == 1) & (complete == 1)).sum()))
    denom = max(c_min * (1 - 1 / k), 1.0)
    return (p_used + 1) / denom, c_min


def run_grid(rng, n_rep, ridge_scale=1.0, verbose=True, t0=None):
    """One pass over the grid. Returns one row per cell and estimator."""
    rows = []
    total = len(GRID_N) * len(GRID_P) * len(GRID_EVENT_RATE)
    done = 0
    t0 = t0 or _time.time()
    for n in GRID_N:
        for p in GRID_P:
            for er in GRID_EVENT_RATE:
                # three nested adjustment sets, as an applied analyst would face
                sets = [max(3, p // 4), max(4, p // 2), p]
                a_est = {s: [] for s in sets}
                k_est = {s: [] for s in sets}
                rhos, evs, cmins, truths = [], [], [], []

                for r in range(n_rep):
                    X, A, time, event, truth = simulate(rng, n, p, er)
                    if min((A == 1).sum(), (A == 0).sum()) < 15:
                        continue
                    ev = int(((event == 1) & (time <= TAU)).sum())
                    rho, c_min = rho_of(X, A, time, event, p)
                    rhos.append(rho); evs.append(ev); cmins.append(c_min)
                    truths.append(truth)
                    for s in sets:
                        Xs = X[:, :s]
                        a_est[s].append(
                            aipw_rmst(time, event, A, Xs, TAU, seed=r,
                                      ridge_scale=ridge_scale)["estimate"])
                        k_est[s].append(
                            ipcw_km_rmst_contrast(time, event, A, TAU, X=Xs,
                                                  seed=r,
                                                  ridge_scale=ridge_scale)["estimate"])
                if not rhos:
                    continue

                # adjustment-set sensitivity: within a replicate, the range of
                # the estimate across the three sets, averaged over replicates
                A_mat = np.column_stack([a_est[s] for s in sets])
                K_mat = np.column_stack([k_est[s] for s in sets])
                a_sens = float(np.mean(A_mat.max(1) - A_mat.min(1)))
                k_sens = float(np.mean(K_mat.max(1) - K_mat.min(1)))
                # sampling variability of each estimator on the full set
                a_se = float(np.std(A_mat[:, -1], ddof=1))
                k_se = float(np.std(K_mat[:, -1], ddof=1))
                truth_mean = float(np.mean(truths))

                for name, sens, se, mat in (("AIPW", a_sens, a_se, A_mat),
                                            ("IPCW-KM", k_sens, k_se, K_mat)):
                    rows.append(dict(
                        n=n, p=p, event_rate=er, estimator=name,
                        ridge_scale=ridge_scale,
                        rho=float(np.mean(rhos)),
                        events=float(np.mean(evs)),
                        complete_min=float(np.mean(cmins)),
                        truth=truth_mean,
                        bias=float(np.mean(mat[:, -1]) - truth_mean),
                        adjustment_sensitivity=sens,
                        sampling_sd=se,
                        sensitivity_ratio=sens / se if se > 0 else np.nan,
                        n_replicates=int(len(rhos))))
                done += 1
                if verbose:
                    el = _time.time() - t0
                    print(f"  [{done}/{total}] n={n:5d} p={p:3d} rate={er:.2f}  "
                          f"rho={np.mean(rhos):5.2f}  events={np.mean(evs):5.1f}  "
                          f"AIPW sens={a_sens:6.1f}d  IPCW-KM sens={k_sens:6.1f}d  "
                          f"({el:.0f}s)")
    return pd.DataFrame(rows)


def calibrate(a: pd.DataFrame, level=LEVEL, bootstrap_cells=True):
    """Log-linear fit of the sensitivity ratio on log rho and the crossing."""
    a_sorted = a.sort_values("rho").reset_index(drop=True)
    lr = np.log(a_sorted["rho"].to_numpy())
    rt = a_sorted["sensitivity_ratio"].to_numpy()
    b1, b0 = np.polyfit(lr, rt, 1)
    thresh = float(np.exp((level - b0) / b1)) if b1 > 0 else float("nan")
    resid = rt - (b0 + b1 * lr)
    r2 = 1 - float(np.var(resid) / np.var(rt))
    crossing = a_sorted[a_sorted["sensitivity_ratio"] >= level]
    empirical = float(crossing["rho"].iloc[0]) if len(crossing) else float("nan")
    lo = hi = float("nan")
    n_boot = 0
    if bootstrap_cells:
        # Bootstrap the crossing point over grid cells. The threshold is an
        # estimate like any other and the applied conclusion for the
        # whole-cohort arm sits close enough to it that reporting a point value
        # alone would overstate what the grid supports.
        rng_b = np.random.default_rng(SEED + 99)
        n_cells = len(a_sorted)
        boots = []
        for _ in range(2000):
            idx = rng_b.integers(0, n_cells, n_cells)
            lrb, rtb = lr[idx], rt[idx]
            if np.ptp(lrb) < 1e-9:
                continue
            bb1, bb0 = np.polyfit(lrb, rtb, 1)
            if bb1 > 0:
                boots.append(float(np.exp((level - bb0) / bb1)))
        boots = np.asarray(boots)
        if len(boots):
            lo, hi = (float(np.percentile(boots, 2.5)),
                      float(np.percentile(boots, 97.5)))
        n_boot = int(len(boots))
    return dict(b0=b0, b1=b1, r2=r2, thresh=thresh, empirical=empirical,
                lo=lo, hi=hi, n_boot=n_boot)


def main() -> None:
    banner("16 — Pre-estimation estimability diagnostic")
    rng = np.random.default_rng(SEED)
    t0 = _time.time()

    out = run_grid(rng, N_REP, ridge_scale=1.0, t0=t0)
    out.to_csv(OUTPUT / "16_estimability_grid.csv", index=False)

    # ── calibrate rho against the sensitivity ratio ─────────────────────────
    a = out[out.estimator == "AIPW"].copy()
    k = out[out.estimator == "IPCW-KM"].copy()
    merged = a.merge(k, on=["n", "p", "event_rate"], suffixes=("_aipw", "_km"))
    merged["excess"] = (merged["adjustment_sensitivity_aipw"]
                        / merged["adjustment_sensitivity_km"].clip(lower=1e-9))

    print("\n" + "=" * 74)
    print("  How the diagnostic tracks adjustment-set sensitivity")
    print("=" * 74)
    show = merged[["n", "p", "event_rate", "rho_aipw", "events_aipw",
                   "adjustment_sensitivity_aipw", "adjustment_sensitivity_km",
                   "sensitivity_ratio_aipw", "sensitivity_ratio_km", "excess"]]
    show = show.rename(columns={"rho_aipw": "rho", "events_aipw": "events",
                                "adjustment_sensitivity_aipw": "sens_AIPW",
                                "adjustment_sensitivity_km": "sens_KM",
                                "sensitivity_ratio_aipw": "ratio_AIPW",
                                "sensitivity_ratio_km": "ratio_KM"})
    print(show.sort_values("rho").round(3).to_string(index=False))

    # The sensitivity ratio rises monotonically with rho but does not reach 1
    # on this grid, so "first exceeds its own sd" is not a usable crossing.
    # Calibrate instead at a stated level: the rho at which adjustment-set
    # movement reaches half the estimator's own sampling sd. This is a
    # convention, and the level sensitivity below shows what it costs.
    cal = calibrate(a, LEVEL)
    b0, b1, r2 = cal["b0"], cal["b1"], cal["r2"]
    thresh, thresh_lo, thresh_hi = cal["thresh"], cal["lo"], cal["hi"]
    empirical = cal["empirical"]
    km_max = float(k["sensitivity_ratio"].max())
    excess_med = float(merged["excess"].median())

    print(f"\n  Calibration")
    print(f"    log-linear fit: ratio = {b0:+.3f} {b1:+.3f} * log(rho)   "
          f"R^2 = {r2:.3f}")
    print(f"    AIPW adjustment-set movement reaches {LEVEL:.0%} of its own "
          f"sampling sd at rho = {thresh:.2f} "
          f"(95% bootstrap interval {thresh_lo:.2f} to {thresh_hi:.2f}; "
          f"{empirical:.2f} at the first grid cell)")
    print(f"    IPCW-KM never exceeds {km_max:.2f} anywhere on the grid")
    print(f"    across every cell, AIPW moves {excess_med:.1f}x further than "
          f"IPCW-KM when the adjustment set changes")
    print(f"\n    Note: the simulated adjustment sets are nested prefixes of "
          f"the same covariates,")
    print(f"    so they differ in dimension but not in content. Real "
          f"adjustment sets differ in both,")
    print(f"    so this calibration is conservative: observed sensitivity at a "
          f"given rho should")
    print(f"    be treated as a lower bound.")

    # ── level sensitivity: the same fit, other crossing conventions ────────
    lev_rows = []
    for lv in OTHER_LEVELS:
        c = calibrate(a, lv, bootstrap_cells=True)
        lev_rows.append(dict(level=lv, threshold_rho=c["thresh"],
                             threshold_rho_lo=c["lo"], threshold_rho_hi=c["hi"],
                             specified=(lv == LEVEL)))
    lev = pd.DataFrame(lev_rows)
    lev.to_csv(OUTPUT / "16_level_sensitivity.csv", index=False)
    print(f"\n  Crossing level sensitivity (same grid, same fit):")
    for _, r in lev.iterrows():
        print(f"    level {r['level']:.2f}: rho = {r['threshold_rho']:.2f} "
              f"[{r['threshold_rho_lo']:.2f}, {r['threshold_rho_hi']:.2f}]"
              f"{'   <- specified' if r['specified'] else ''}")

    # ── penalty sensitivity: the grid re-run with the ridge scaled ─────────
    pen_rows = [dict(ridge_scale=1.0, n_replicates=N_REP, threshold_rho=thresh,
                     threshold_rho_lo=thresh_lo, threshold_rho_hi=thresh_hi,
                     fit_slope=b1, fit_intercept=b0, fit_r2=r2,
                     ipcwkm_max_ratio=km_max, specified=True)]
    if UNDER_MONTE_CARLO:
        print("\n  penalty sensitivity skipped under stage 18 (not harvested)")
    else:
        for sc in PENALTY_SCALES:
            print(f"\n  Penalty sensitivity: ridge x{sc:g} ({N_REP_PENALTY} "
                  f"replicates per cell)")
            g = run_grid(np.random.default_rng(SEED + int(100 * sc)),
                         N_REP_PENALTY, ridge_scale=sc, verbose=False, t0=t0)
            ga = g[g.estimator == "AIPW"]
            gk = g[g.estimator == "IPCW-KM"]
            c = calibrate(ga, LEVEL, bootstrap_cells=True)
            pen_rows.append(dict(ridge_scale=sc, n_replicates=N_REP_PENALTY,
                                 threshold_rho=c["thresh"], threshold_rho_lo=c["lo"],
                                 threshold_rho_hi=c["hi"], fit_slope=c["b1"],
                                 fit_intercept=c["b0"], fit_r2=c["r2"],
                                 ipcwkm_max_ratio=float(gk["sensitivity_ratio"].max()),
                                 specified=False))
            print(f"    ridge x{sc:g}: threshold rho = {c['thresh']:.2f} "
                  f"[{c['lo']:.2f}, {c['hi']:.2f}]  R^2 {c['r2']:.3f}  "
                  f"({_time.time() - t0:.0f}s)")
    pen = pd.DataFrame(pen_rows).sort_values("ridge_scale")
    pen.to_csv(OUTPUT / "16_penalty_sensitivity.csv", index=False)

    # ── apply to the real arms, every adjustment set ───────────────────────
    real = []
    nb = OUTPUT / "07_nuisance_budget.csv"
    if nb.exists() and np.isfinite(thresh):
        n_df = pd.read_csv(nb)
        if "adjustment" not in n_df.columns:
            n_df["adjustment"] = "UNION"
            n_df["primary_set"] = True
        for (arm, adj), s in n_df[n_df.component == "outcome model m_a"].groupby(
                ["arm", "adjustment"], sort=False):
            worst = s.loc[s["params_per_observation"].idxmax()]
            r = float(worst["n_parameters"]) / max(
                worst["n_available"] * (1 - 1 / K_FOLDS), 1)
            r_eff = (float(worst["effective_df"]) / max(
                worst["n_available"] * (1 - 1 / K_FOLDS), 1)
                if "effective_df" in worst and np.isfinite(worst["effective_df"])
                else float("nan"))
            # Classify against the interval, not the point estimate. An arm
            # inside the interval is indeterminate, not passing.
            if r >= thresh_hi:
                verdict = ("outcome model starved: prefer a weighting-only "
                           "estimator")
            elif r >= thresh_lo:
                verdict = ("indeterminate: rho lies inside the threshold "
                           "interval")
            else:
                verdict = "below the threshold interval"
            real.append(dict(arm=arm, arm_label=worst["arm_label"],
                             adjustment=adj,
                             primary_set=bool(worst.get("primary_set", adj == "UNION")),
                             rho=r, rho_effective_df=r_eff,
                             n_parameters=int(worst["n_parameters"]),
                             effective_df=float(worst.get("effective_df", np.nan)),
                             c_min=int(worst["n_available"]),
                             threshold=thresh, threshold_lo=thresh_lo,
                             threshold_hi=thresh_hi, verdict=verdict))
    elif not np.isfinite(thresh):
        print("\n    THRESHOLD UNDEFINED on this grid; no arm is classified.")

    if real:
        print(f"\n  Applied to the real arms (threshold rho = "
              f"{thresh:.2f}, 95% interval {thresh_lo:.2f} to {thresh_hi:.2f}):")
        for r in real:
            print(f"    {r['arm_label']:28s} {r['adjustment']:8s}"
                  f"{' (primary)' if r['primary_set'] else '          '} "
                  f"rho = {r['rho']:5.2f}  (effective-df version "
                  f"{r['rho_effective_df']:5.2f})   {r['verdict']}")

    fit = pd.DataFrame([dict(
        level=LEVEL, fit_intercept=b0, fit_slope=b1, fit_r2=r2,
        threshold_rho_fitted=thresh, threshold_rho_empirical=empirical,
        threshold_rho_lo=thresh_lo, threshold_rho_hi=thresh_hi,
        n_threshold_bootstrap=int(cal["n_boot"]),
        ipcwkm_max_ratio_anywhere=km_max, median_excess_aipw_over_km=excess_med,
        n_grid_cells=int(len(a)), n_replicates=N_REP, k_folds=K_FOLDS,
        tau=TAU)])
    fit.to_csv(OUTPUT / "16_estimability_fit.csv", index=False)

    Results().update({
        "estimability_grid": out.to_dict(orient="records"),
        "estimability_level": LEVEL,
        "estimability_fit": dict(intercept=b0, slope=b1, r2=r2),
        "estimability_threshold_rho": thresh,
        "estimability_threshold_lo": thresh_lo,
        "estimability_threshold_hi": thresh_hi,
        "estimability_threshold_empirical": empirical,
        "estimability_km_max_ratio": km_max,
        "estimability_median_excess": excess_med,
        "estimability_level_sensitivity": lev.to_dict(orient="records"),
        "estimability_penalty_sensitivity": pen.to_dict(orient="records"),
        "estimability_real_arms": real,
    })
    print(f"\n  wrote output/16_estimability_grid.csv, 16_estimability_fit.csv, "
          f"16_level_sensitivity.csv and 16_penalty_sensitivity.csv  "
          f"({_time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
