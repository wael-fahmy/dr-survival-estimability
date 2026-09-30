"""
00_validate_estimators.py — validate the estimators AT THE OPERATING POINT.

An earlier version of this script simulated n=600 with 12 replicates and
reported bias only. That was not adequate. It ran at a sample size the analysis
never uses, with too few replicates to characterise anything, and it never
checked the property every claim in the paper depends on: whether the bootstrap
intervals cover the truth.

This version:

  * simulates at the actual operating points -- n=136 with ~16 events and n=988
    with ~72 events at tau=1095, matching the real treated fractions and
    censoring distributions
  * runs N_REP replicates, not 12
  * reports bootstrap interval COVERAGE, the fraction of 95% intervals that
    contain the truth
  * reports the standard deviation of the estimate across replicates alongside
    the mean bootstrap standard error, so a bootstrap that misstates precision
    is visible
  * reports bias as a fraction of the real primary estimate, not in isolation
  * makes censoring depend on a covariate that also drives survival, so the
    covariate-conditional censoring model (C _||_ T | A, X) is exercised and
    the arm-only model (C _||_ T | A) is misspecified, as in the real cohort
    where follow-up differs by an order of magnitude across sites

Bootstrap replicates inside the simulation are set by B_SIM, below the B=2000
used for the reported analysis, for tractability. On the real TNBC data the two
give interval widths within 2% of each other (176.6 vs 173.3 days), so coverage
estimated at B_SIM is a fair guide to coverage at B=2000.

Also retained: exact checks of weighted Kaplan-Meier against a naive O(n^2)
reference, and of RMST against a closed form.

Produces:
    00_estimator_validation.csv  ->  Table 8
"""

from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from lib_io import OUTPUT, SEED, Results, banner  # noqa: E402
from lib_est import (aipw_rmst, bootstrap, ipcw_km_rmst_contrast,  # noqa: E402
                     km_estimate, km_estimate_tv, km_rmst, median_followup,
                     cox_fit, standardize)

import os  # noqa: E402
# REVISION_SIM_REPS exists only for smoke-testing the pipeline; the reported
# validation uses 500 replicates and the output records the value.
N_REP = int(os.environ.get("REVISION_SIM_REPS") or 500)
B_SIM = min(500, max(20, N_REP))
TAU = 1095

# Calibrated to the day-90 landmark cohort under the primary (clinical)
# adjustment set. See 05_event_counts.csv and 07_diagnostics.csv for the values
# being matched; p is the clinical set's dimension after constant columns are
# dropped, 25 in the triple-negative subgroup and 28 in the whole cohort.
SCENARIOS = [
    dict(key="TNBC-like", n=136, p=25, p_treated=90 / 136, target_events=16,
         median_fu=1135.0,
         label="TNBC arm operating point (n=136, ~16 events)"),
    dict(key="cohort-like", n=988, p=28, p_treated=400 / 988, target_events=72,
         median_fu=1045.0,
         label="Whole-cohort operating point (n=988, ~72 events)"),
]
CENS_DEPENDENCE = 0.5     # log-hazard of censoring per unit of X[:, 0]


def km_reference(time, event, w=None):
    """Deliberately naive O(n^2) Kaplan-Meier, used only as a cross-check."""
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    w = np.ones(len(time)) if w is None else np.asarray(w, float)
    o = np.argsort(time, kind="mergesort")
    t, e, w = time[o], event[o], w[o]
    out_t, out_s, s = [0.0], [1.0], 1.0
    for u in np.unique(t):
        ar = w[t >= u].sum()
        d = w[(t == u) & (e == 1)].sum()
        if ar > 0 and d > 0:
            s *= 1 - d / ar
        out_t.append(float(u))
        out_s.append(float(s))
    return np.array(out_t), np.array(out_s)


def simulate(rng, n, p, p_treated, base_hazard, cens_rate, hr=0.65,
             confound=1.0, tau=TAU):
    """One replicate at a given operating point.

    Covariate dimension p matches the primary adjustment set, because the
    behaviour being validated depends on the ratio of nuisance parameters to
    events, not only on n. Censoring depends on X[:, 0], which also drives
    survival: censoring is independent of survival given X and A, and not
    otherwise.
    """
    X = rng.normal(size=(n, p))
    lp = confound * (0.7 * X[:, 0] - 0.5 * X[:, 1] + 0.3 * X[:, 2])
    lp = lp - lp.mean() + np.log(p_treated / (1 - p_treated))
    A = (rng.random(n) < 1 / (1 + np.exp(-lp))).astype(int)
    frail = np.exp(0.5 * X[:, 0] + 0.3 * X[:, 1])
    hz = base_hazard * frail * np.where(A == 1, hr, 1.0)
    T = rng.exponential(1 / hz)
    c_rate = cens_rate * np.exp(CENS_DEPENDENCE * X[:, 0])
    C = rng.exponential(1 / c_rate)
    time, event = np.minimum(T, C), (T <= C).astype(int)
    h1, h0 = base_hazard * frail * hr, base_hazard * frail
    truth = float(np.mean((1 - np.exp(-h1 * tau)) / h1)
                  - np.mean((1 - np.exp(-h0 * tau)) / h0))
    return X, A, time, event, truth


def calibrate(rng, sc):
    """Find the baseline hazard and censoring rate that reproduce the target
    event count and median follow-up at this operating point."""
    cens_rate = np.log(2) / sc["median_fu"]
    lo, hi = 1e-6, 1e-2
    for _ in range(40):
        mid = (lo + hi) / 2
        ev = np.mean([
            simulate(rng, sc["n"], 4, sc["p_treated"], mid, cens_rate)[3]
            [simulate(rng, sc["n"], 4, sc["p_treated"], mid, cens_rate)[2] <= TAU].sum()
            for _ in range(3)])
        # recompute cleanly: count events observed within tau
        evs = []
        for _ in range(5):
            _, _, t, e, _ = simulate(rng, sc["n"], 4, sc["p_treated"], mid,
                                     cens_rate)
            evs.append(int(((e == 1) & (t <= TAU)).sum()))
        ev = float(np.mean(evs))
        if ev < sc["target_events"]:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2, cens_rate


def run_scenario(sc, primary_estimate, res_rows, rng):
    banner(sc["label"])
    base_hazard, cens_rate = calibrate(rng, sc)

    # confirm the calibration
    chk = [simulate(rng, sc["n"], sc["p"], sc["p_treated"], base_hazard,
                    cens_rate) for _ in range(30)]
    ev = np.mean([int(((e == 1) & (t <= TAU)).sum()) for _, _, t, e, _ in chk])
    fu = np.mean([median_followup(t, e) for _, _, t, e, _ in chk])
    tr = np.mean([A.mean() for _, A, _, _, _ in chk])
    print(f"  calibrated: base hazard {base_hazard:.3e}, censoring rate "
          f"{cens_rate:.3e}")
    print(f"  realised  : {ev:.1f} events (target {sc['target_events']}), "
          f"median follow-up {fu:,.0f} d (target {sc['median_fu']:,.0f}), "
          f"treated fraction {tr:.2f} (target {sc['p_treated']:.2f})")
    print(f"  running {N_REP} replicates with B={B_SIM} bootstrap each ...")

    recs = {"AIPW": [], "IPCW-KM": []}
    t0 = _time.time()
    for r in range(N_REP):
        X, A, time, event, truth = simulate(rng, sc["n"], sc["p"],
                                            sc["p_treated"], base_hazard,
                                            cens_rate)
        if len(np.unique(A)) < 2:
            continue
        a = aipw_rmst(time, event, A, X, TAU, seed=r)["estimate"]
        k = ipcw_km_rmst_contrast(time, event, A, TAU, X=X, seed=r)["estimate"]
        ci_a = bootstrap(
            lambda idx: (aipw_rmst(time[idx], event[idx], A[idx], X[idx], TAU,
                                   seed=r)["estimate"]
                         if len(np.unique(A[idx])) > 1 else np.nan),
            sc["n"], B=B_SIM, seed=10_000 + r)
        ci_k = bootstrap(
            lambda idx: (ipcw_km_rmst_contrast(time[idx], event[idx], A[idx],
                                               TAU, X=X[idx], seed=r)["estimate"]
                         if len(np.unique(A[idx])) > 1 else np.nan),
            sc["n"], B=B_SIM, seed=20_000 + r)
        recs["AIPW"].append((a, truth, ci_a["lo"], ci_a["hi"], ci_a["se"]))
        recs["IPCW-KM"].append((k, truth, ci_k["lo"], ci_k["hi"], ci_k["se"]))
        if (r + 1) % 50 == 0:
            el = _time.time() - t0
            print(f"    {r + 1}/{N_REP}  ({el:.0f}s elapsed, "
                  f"{el / (r + 1) * (N_REP - r - 1):.0f}s remaining)")

    for est, v in recs.items():
        v = np.asarray(v, dtype=float)
        pt, truth, lo, hi, se = v[:, 0], v[:, 1], v[:, 2], v[:, 3], v[:, 4]
        bias = float(pt.mean() - truth.mean())
        emp_sd = float(pt.std(ddof=1))
        mean_se = float(np.nanmean(se))
        cov = float(np.mean((lo <= truth) & (truth <= hi)))
        row = dict(
            scenario=sc["key"], scenario_label=sc["label"], estimator=est,
            n=sc["n"], target_events=sc["target_events"],
            realised_events=float(ev), n_replicates=int(len(v)),
            b_bootstrap=B_SIM,
            truth=float(truth.mean()), mean_estimate=float(pt.mean()),
            bias=bias,
            empirical_sd=emp_sd, mean_bootstrap_se=mean_se,
            se_ratio=mean_se / emp_sd if emp_sd else float("nan"),
            coverage_95=cov, mean_ci_width=float(np.nanmean(hi - lo)),
        )
        res_rows.append(row)
        print(f"\n  {est}")
        print(f"    truth {row['truth']:+.1f} d, mean estimate "
              f"{row['mean_estimate']:+.1f} d, bias {bias:+.1f} d "
              f"({100 * bias / primary_estimate:+.0f}% of the primary estimate)")
        print(f"    empirical SD {emp_sd:.1f} d vs mean bootstrap SE "
              f"{mean_se:.1f} d  (ratio {row['se_ratio']:.2f})")
        print(f"    95% interval COVERAGE {cov * 100:.1f}%  "
              f"(mean width {row['mean_ci_width']:.0f} d)")


def main() -> None:
    banner("00 — Estimator validation at the operating point")
    rng = np.random.default_rng(SEED)

    # ── exact checks that do not depend on the operating point ──────────────
    t = rng.exponential(800, 400)
    e = (rng.random(400) < 0.4).astype(int)
    w = rng.random(400) + 0.5
    for label, args in (("no ties", (t, e, None)), ("weighted", (t, e, w)),
                        ("heavy ties", (np.round(t / 50) * 50, e, None))):
        a, b = km_estimate(*args), km_reference(*args)
        if not (np.allclose(a[1], b[1]) and np.allclose(a[0], b[0])):
            raise AssertionError(f"Kaplan-Meier mismatch ({label})")
        print(f"  Kaplan-Meier vs naive reference, {label:12s}: match")

    # Time-varying-weight Kaplan-Meier (used by the primary estimator under
    # the covariate-conditional censoring model) must reduce exactly to the
    # ordinary weighted Kaplan-Meier when the censoring weights do not vary
    # across subjects at a given time.
    grid = np.unique(t)
    G_const = np.tile(np.linspace(1.0, 0.3, len(grid)), (len(t), 1))
    a, b = km_estimate(t, e, w), km_estimate_tv(t, e, w, grid, G_const)
    if not (np.allclose(a[1], b[1]) and np.allclose(a[0], b[0])):
        raise AssertionError("time-varying-weight Kaplan-Meier does not reduce "
                             "to the weighted Kaplan-Meier")
    print("  IPCW Kaplan-Meier (time-varying weights) vs weighted KM: match")

    # Cox partial likelihood: the Newton-Raphson fit must recover a known
    # coefficient vector at large n (exact agreement with lifelines is checked
    # in stage 15 where that library is installed).
    Xc = rng.normal(size=(20_000, 3))
    b_true = np.array([0.5, -0.3, 0.2])
    Tc = rng.exponential(1 / (0.001 * np.exp(Xc @ b_true)))
    Cc = rng.exponential(1500, 20_000)
    tc, ec = np.minimum(Tc, Cc), (Tc <= Cc).astype(int)
    b_hat = cox_fit(standardize(Xc)[0], tc, ec, ridge=1e-8)["beta"]
    b_ref = b_true * Xc.std(axis=0)
    err = float(np.max(np.abs(b_hat - b_ref)))
    print(f"  Cox fit vs generating coefficients (n=20,000): max |error| {err:.3f}")
    if err > 0.05:
        raise AssertionError("Cox fit does not recover the generating coefficients")

    lam = 1 / 900.0
    big = rng.exponential(1 / lam, 200_000)
    got, want = km_rmst(big, np.ones(len(big), int), TAU), (1 - np.exp(-lam * TAU)) / lam
    err = abs(got - want) / want
    print(f"  RMST vs closed form: {got:.1f} vs {want:.1f} ({err * 100:.2f}% error)")
    if err > 0.02:
        raise AssertionError("RMST integration off by more than 2%")

    # ── the real primary estimate, so bias can be expressed against it ──────
    primary = None
    est_p = OUTPUT / "07_estimates.csv"
    if est_p.exists():
        e_df = pd.read_csv(est_p)
        sel = e_df[(e_df.arm == "TNBC_chemo") & (e_df.adjustment == "UNION")
                   & (e_df.tau == TAU) & (e_df.estimator.str.startswith("AIPW"))]
        if len(sel):
            primary = float(sel.iloc[0]["estimate_days"])
    if primary is None:
        primary = 46.86
        print(f"\n  (07_estimates.csv not yet written; expressing bias against "
              f"the last recorded primary estimate, {primary:+.2f} d)")
    else:
        print(f"\n  bias will be expressed against the primary estimate "
              f"{primary:+.2f} d")

    rows: list[dict] = []
    for sc in SCENARIOS:
        run_scenario(sc, primary, rows, rng)

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT / "00_estimator_validation.csv", index=False)
    Results().put("estimator_validation", rows)

    print("\n" + "=" * 74)
    print("  SUMMARY")
    print("=" * 74)
    print(out[["scenario", "estimator", "truth", "mean_estimate", "bias",
               "empirical_sd", "mean_bootstrap_se", "se_ratio",
               "coverage_95"]].round(3).to_string(index=False))

    worst = out["coverage_95"].min()
    print(f"\n  worst 95% interval coverage across all scenarios: "
          f"{worst * 100:.1f}%")
    if worst < 0.90:
        print("  WARNING: coverage is materially below nominal. Every interval "
              "in the report is wider in truth than it appears.")
    print("  This table is reported in the manuscript, not just here.")


if __name__ == "__main__":
    main()
