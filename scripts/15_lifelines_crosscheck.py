"""
15_lifelines_crosscheck.py — reproduce the core quantities with lifelines.

Why. lib_est.py implements Kaplan-Meier, RMST and inverse probability of
censoring weighting from first principles, because the analysis environment has
neither lifelines nor scikit-survival. A reviewer running this code is entitled
to ask why established implementations were not used. The answer is
availability, and the way to close it is to show the hand-written code agrees
with a maintained library wherever the library can be installed.

This script recomputes, on the real day-90 landmark cohort:

    * the Kaplan-Meier survival curve in each treatment arm
    * restricted mean survival time at 1095 and 1825 days in each arm
    * the arm-only censoring survival function
    * the Cox censoring model (coefficients and Breslow baseline) against
      lifelines' CoxPHFitter, unpenalised, on the clinical covariates
    * the unadjusted RMST difference

using lifelines, and reports the maximum absolute discrepancy against
lib_est.py. It exits 0 and prints a clear notice if lifelines is absent, so
run_all.py is unaffected on a machine without it.

    pip install lifelines
    python scripts/15_lifelines_crosscheck.py

Produces:
    15_lifelines_crosscheck.csv  ->  the Reproducibility statement
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from lib_io import OUTPUT, Results, banner, barcode12  # noqa: E402
from lib_est import (censoring_survival, cox_fit, km_estimate,  # noqa: E402
                     km_rmst, standardize)

TAUS = [1095, 1825]
TOL = 1e-6          # agreement we expect for KM; RMST integration is coarser
TOL_RMST = 0.5      # days


def main() -> None:
    banner("15 — Cross-check against lifelines")
    res = Results()

    try:
        from lifelines import CoxPHFitter, KaplanMeierFitter
        from lifelines.utils import restricted_mean_survival_time
    except ImportError:
        print("  lifelines is not installed in this environment.")
        print("  This check is optional and was skipped; run_all.py is "
              "unaffected.")
        print("\n  To run it:  pip install lifelines && python "
              "scripts/15_lifelines_crosscheck.py")
        print("\n  lib_est.py is validated independently in three other ways:")
        print("    * weighted Kaplan-Meier against a direct O(n^2) reference, "
              "exact including ties (stage 00)")
        print("    * RMST against the closed form for an exponential, 0.07% "
              "error (stage 00)")
        print("    * bias and interval coverage against simulated data with a "
              "known answer at the operating point (stage 00)")
        res.put("lifelines_crosscheck",
                dict(status="skipped: lifelines not installed"))
        return

    df = pd.read_csv(OUTPUT / "05_cohort_landmark90.csv", index_col=0)
    df.index = barcode12(df.index)
    time = df["time"].to_numpy(float)
    event = df["event"].to_numpy(int)
    A = ((df["start_chemo"].notna()) & (df["start_chemo"] <= 90)
         & (df["start_chemo"] >= 0)).astype(int).to_numpy()

    rows = []

    # ── Kaplan-Meier, per arm ───────────────────────────────────────────────
    for a in (0, 1):
        m = A == a
        kmf = KaplanMeierFitter().fit(time[m], event[m])
        t_ours, s_ours = km_estimate(time[m], event[m])
        grid = np.unique(np.concatenate([[0.0], time[m]]))
        theirs = kmf.predict(grid).to_numpy()
        idx = np.clip(np.searchsorted(t_ours, grid, side="right") - 1,
                      0, len(s_ours) - 1)
        ours = s_ours[idx]
        d = float(np.max(np.abs(ours - theirs)))
        rows.append(dict(quantity=f"Kaplan-Meier survival, arm {a}",
                         ours=np.nan, lifelines=np.nan, max_abs_diff=d,
                         tolerance=TOL, agrees=bool(d <= TOL)))
        print(f"  KM curve, arm {a}: max |difference| = {d:.2e}")

    # ── RMST, per arm and horizon ───────────────────────────────────────────
    for tau in TAUS:
        for a in (0, 1):
            m = A == a
            kmf = KaplanMeierFitter().fit(time[m], event[m])
            theirs = float(restricted_mean_survival_time(kmf, t=tau))
            ours = km_rmst(time[m], event[m], tau)
            d = abs(ours - theirs)
            rows.append(dict(quantity=f"RMST at {tau} d, arm {a}",
                             ours=ours, lifelines=theirs, max_abs_diff=d,
                             tolerance=TOL_RMST, agrees=bool(d <= TOL_RMST)))
            print(f"  RMST tau={tau}, arm {a}: ours {ours:8.2f}  "
                  f"lifelines {theirs:8.2f}  diff {d:.4f} d")

        r1o = km_rmst(time[A == 1], event[A == 1], tau)
        r0o = km_rmst(time[A == 0], event[A == 0], tau)
        k1 = KaplanMeierFitter().fit(time[A == 1], event[A == 1])
        k0 = KaplanMeierFitter().fit(time[A == 0], event[A == 0])
        theirs = (float(restricted_mean_survival_time(k1, t=tau))
                  - float(restricted_mean_survival_time(k0, t=tau)))
        ours = r1o - r0o
        d = abs(ours - theirs)
        rows.append(dict(quantity=f"unadjusted RMST difference at {tau} d",
                         ours=ours, lifelines=theirs, max_abs_diff=d,
                         tolerance=TOL_RMST, agrees=bool(d <= TOL_RMST)))
        print(f"  unadjusted RMST difference tau={tau}: ours {ours:+8.2f}  "
              f"lifelines {theirs:+8.2f}  diff {d:.4f} d")

    # ── censoring survival function, the basis of the IPC weights ───────────
    #
    # G(t) here is the LEFT limit, P(not yet censored at t-), which is what the
    # IPC weight 1/G(Y) requires. lifelines' predict() is right-continuous, so
    # the comparison evaluates it a hair below each time.
    #
    # t = 0 must be excluded from the numerical comparison, and the reason is
    # not a defect. G(0-) = 1 by definition: nobody can be censored strictly
    # before time zero. lifelines cannot be asked for S(0-) -- predict(-1e-9)
    # returns NaN, below its support -- so an earlier version of this check
    # clamped the query to max(t - 1e-9, 0) and compared our G(0-) = 1 against
    # their S(0) = 0.9983, which already applies a censoring AT zero. That is a
    # comparison artefact. The single patient at t = 0 in this cohort is
    # censored at the landmark, is therefore not a complete case, and carries
    # IPC weight exactly 0, so no estimate depends on G(0) at all.
    for a in (0, 1):
        m = A == a
        kmf = KaplanMeierFitter().fit(time[m], 1 - event[m])
        G = censoring_survival(time[m], event[m])
        grid = np.unique(time[m])
        n_zero = int((grid == 0).sum())
        grid = grid[grid > 0]
        theirs = kmf.predict(grid - 1e-9).to_numpy()
        ours = G(grid)
        # our G is floored at 1e-3 inside censoring_survival; compare above it
        keep = theirs > 1e-3
        d = float(np.max(np.abs(ours[keep] - theirs[keep]))) if keep.any() else 0.0
        rows.append(dict(quantity=f"censoring survival G(t-), arm {a}",
                         ours=np.nan, lifelines=np.nan, max_abs_diff=d,
                         tolerance=1e-6, agrees=bool(d <= 1e-6)))
        note = f"  ({n_zero} time at t=0 excluded: G(0-)=1 by definition)" if n_zero else ""
        print(f"  censoring G(t-), arm {a}: max |difference| = {d:.2e}"
              f"  over {int(keep.sum())} times{note}")

    # ── Cox censoring model, unpenalised, against lifelines ─────────────────
    #
    # The censoring model used by both estimators is a ridge-penalised Cox
    # model for the censoring hazard, stratified by arm. lifelines scales its
    # penalty differently, so the comparison is made unpenalised (ridge 1e-8
    # here, penalizer 0 there) on the continuous clinical covariates, where an
    # unpenalised fit is well defined. lifelines handles tied times by Efron's
    # method and lib_est.py by Breslow's; the two coincide when there are no
    # ties, so the comparison breaks the ties in days with a deterministic
    # jitter far below a day (the same jittered times go to both fits). On the
    # tied data the two conventions differ by about 2e-3 in the coefficients,
    # which is the size of the Efron-Breslow gap, not a defect. Agreement of
    # the coefficients and of the Breslow baseline establishes that the
    # partial-likelihood code is right; the penalty is then a matter of
    # arithmetic.
    cov = pd.DataFrame({
        "age": pd.to_numeric(df["age"], errors="coerce"),
        "year_dx": pd.to_numeric(df["year_dx"], errors="coerce"),
    }).fillna(pd.DataFrame({"age": pd.to_numeric(df["age"], errors="coerce"),
                            "year_dx": pd.to_numeric(df["year_dx"], errors="coerce")
                            }).median())
    Xs, _, _ = standardize(cov.to_numpy(float))
    cens = 1 - event
    t_jit = time + np.random.default_rng(0).uniform(0, 1e-3, len(time))
    ours = cox_fit(Xs, t_jit, cens, ridge=1e-8, strata=A)
    dd = pd.DataFrame(Xs, columns=["age", "year_dx"])
    dd["t"], dd["c"], dd["arm"] = t_jit, cens, A
    cph = CoxPHFitter(penalizer=0.0).fit(dd, "t", "c", strata=["arm"])
    d_beta = float(np.max(np.abs(ours["beta"] - cph.params_.to_numpy())))
    rows.append(dict(quantity="Cox censoring model, coefficients (arm-stratified)",
                     ours=float(ours["beta"][0]),
                     lifelines=float(cph.params_.iloc[0]),
                     max_abs_diff=d_beta, tolerance=1e-4,
                     agrees=bool(d_beta <= 1e-4)))
    print(f"  Cox censoring coefficients: ours {np.round(ours['beta'], 5)}  "
          f"lifelines {np.round(cph.params_.to_numpy(), 5)}  "
          f"max |difference| = {d_beta:.2e}")
    # Baseline cumulative hazard, compared on a relative scale because the
    # late-time baseline is large when a coefficient is large (a standardized
    # coefficient of 2 makes the baseline at the covariate mean many times the
    # hazard of a typical patient).
    bch = cph.baseline_cumulative_hazard_
    worst_h = 0.0
    for a in (0, 1):
        uniq, H0 = ours["base"][a]
        col = [c for c in bch.columns if str(c) in (str(a), f"({a},)", str((a,)))]
        theirs = bch[col[0]] if col else bch.iloc[:, a]
        theirs_at = np.interp(uniq, theirs.index.to_numpy(float),
                              theirs.to_numpy(float))
        rel = np.abs(H0 - theirs_at) / np.maximum(np.abs(theirs_at), 1e-8)
        worst_h = max(worst_h, float(np.max(rel[theirs_at > 1e-6])))
    rows.append(dict(quantity="Cox censoring model, Breslow baseline cumulative hazard (relative)",
                     ours=np.nan, lifelines=np.nan, max_abs_diff=worst_h,
                     tolerance=1e-4, agrees=bool(worst_h <= 1e-4)))
    print(f"  Cox censoring baseline cumulative hazard: max relative "
          f"|difference| = {worst_h:.2e}")

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT / "15_lifelines_crosscheck.csv", index=False)

    n_ok = int(out["agrees"].sum())
    worst_km = out[out["quantity"].str.startswith("Kaplan")]["max_abs_diff"].max()
    worst_rmst = out[out["quantity"].str.contains("RMST")]["max_abs_diff"].max()
    print(f"\n  {n_ok} of {len(out)} quantities agree within tolerance")
    print(f"  worst Kaplan-Meier discrepancy : {worst_km:.2e}")
    print(f"  worst RMST discrepancy         : {worst_rmst:.4f} days")

    import lifelines as _ll
    res.put("lifelines_crosscheck", dict(
        status="run", lifelines_version=_ll.__version__,
        n_quantities=int(len(out)), n_agree=n_ok,
        worst_km_diff=float(worst_km), worst_rmst_diff=float(worst_rmst),
        all_agree=bool(out["agrees"].all()),
        rows=out.to_dict(orient="records")))

    if not out["agrees"].all():
        bad = out[~out["agrees"]]["quantity"].tolist()
        print(f"\n  DISAGREEMENT in: {bad}")
        print("  This is a defect in lib_est.py and must be resolved before "
              "submission.")
        sys.exit(1)
    print("\n  lib_est.py reproduces lifelines on every quantity checked.")


if __name__ == "__main__":
    main()
