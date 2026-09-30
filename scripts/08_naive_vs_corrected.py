"""
08_naive_vs_corrected.py — the correction ladder for the headline result.

The headline is chemotherapy in TNBC. This script rebuilds it one correction
at a time so the contribution of each design decision is visible.

    rung 0  the original analysis: dummy-T, imputed-receptor TNBC, binary
            five-year mortality, time zero at diagnosis
    rung 1  treatment fixed: actual chemotherapy receipt from treatment_type
    rung 2  subgroup fixed: TNBC from directly measured receptors only
    rung 3  outcome fixed: RMST at 1095 d with IPCW instead of a binary that
            treats censored patients as survivors
    rung 4  time zero fixed: day-90 landmark, treatment status as of day 90
    rung 5  confounding addressed: the primary specification, an IPTW- and
            IPCW-weighted Kaplan-Meier contrast on the clinical adjustment set

Two scales are carried. Rungs 0-2 are risk differences (probability of death
within five years) on the original analysis's own scale. In addition, every
rung is expressed as an RMST difference at 1095 days, so that the movement
produced by each design correction can be compared in days throughout: for
rungs 0-2 this is the unadjusted IPCW Kaplan-Meier contrast on the same cohort
and treatment coding, with time zero at diagnosis; rung 3 is by construction
identical to rung 2 on this scale, because rung 3 changes only the scale. The
risk differences and the RMST differences are never differenced against each
other.

Every rung carries a 2000-replicate patient-level bootstrap interval.

Produces:
    08_ladder.csv  ->  Table 7 and Figure 6
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from lib_io import DATA, OUTPUT, SEED, TCGA_NA, Results, banner, barcode12  # noqa: E402
from lib_est import bootstrap, ipcw_km_rmst_contrast  # noqa: E402

B_BOOT = int(os.environ.get("REVISION_BOOT") or 2000)
TAU = 1095
PRIMARY_SET = "CLINICAL"
rows = []


def add(rung, label, scale, est, ci, n, n1, n0, events, rmst, note=""):
    rows.append(dict(rung=rung, label=label, scale=scale, estimate=est,
                     ci_lo=ci["lo"], ci_hi=ci["hi"], n=n, n_treated=n1,
                     n_control=n0, events=events,
                     rmst_estimate=rmst["estimate"], rmst_ci_lo=rmst["lo"],
                     rmst_ci_hi=rmst["hi"], rmst_events=rmst["events"],
                     note=note))
    unit = "pp" if scale.startswith("risk") else "d"
    print(f"  {rung}  {label:52s} {est:+8.3f} {unit}  "
          f"[{ci['lo']:+8.3f}, {ci['hi']:+8.3f}]   n={n:,} ({n1}/{n0}) ev={events}"
          f"   | RMST {rmst['estimate']:+7.1f} d [{rmst['lo']:+7.1f}, {rmst['hi']:+7.1f}]")


def risk_diff_boot(Y, A, n):
    def f(idx):
        y, a = Y[idx], A[idx]
        if a.sum() < 3 or (a == 0).sum() < 3:
            return np.nan
        return y[a == 1].mean() - y[a == 0].mean()
    return f


def rmst_contrast(t, e, A):
    """Unadjusted IPCW Kaplan-Meier RMST contrast with a bootstrap interval."""
    n = len(A)
    pt = ipcw_km_rmst_contrast(t, e, A, TAU, X=None, seed=SEED)["estimate"]
    ci = bootstrap(lambda i: (ipcw_km_rmst_contrast(t[i], e[i], A[i], TAU,
                                                    X=None, seed=SEED)["estimate"]
                              if len(np.unique(A[i])) > 1 else np.nan),
                   n, B=B_BOOT, seed=SEED)
    return dict(estimate=pt, lo=ci["lo"], hi=ci["hi"],
                events=int(((e == 1) & (t <= TAU)).sum()))


def main() -> None:
    banner("08 — Naive versus corrected: the chemotherapy-in-TNBC ladder")
    res = Results()

    surv = pd.read_csv(OUTPUT / "03_survival.csv", index_col=0)
    subs = pd.read_csv(OUTPUT / "02_subgroups.csv", index_col=0)
    v2 = pd.read_csv(DATA / "ite_ready_dataset_v2.csv")
    v2["patient_id"] = barcode12(v2["patient_id"])
    v2 = v2.drop_duplicates("patient_id").set_index("patient_id")

    clin = pd.read_csv(DATA / "clinical.tsv", sep="\t", na_values=TCGA_NA,
                       low_memory=False)
    clin["patient_id"] = barcode12(clin["cases.submitter_id"])
    chemo_pts = set(clin.loc[clin["treatments.treatment_type"]
                             .str.lower().str.contains("chemo", na=False),
                             "patient_id"])

    # The original analysis's imputation-based TNBC
    er = v2["ER_status"].fillna(0) > 0.5
    pr = v2["PR_status"].fillna(0) > 0.5
    h2 = v2["HER2_status"].fillna(0) > 0.5
    tnbc_orig = v2.index[(~er) & (~pr) & (~h2)]
    tnbc_dir = subs.index[subs["TNBC"].fillna(False).astype(bool)]

    base = surv.join(v2[["T"]], how="left")
    base["chemo"] = base.index.isin(chemo_pts).astype(int)
    base["Y5"] = ((base["OS"] == 1) & (base["OS.time"] <= 1825)).astype(int)

    print(f"\n  scale: risk difference in five-year mortality "
          f"(percentage points; negative = fewer deaths in treated),")
    print(f"  with the same cohort and coding as an RMST difference at {TAU} "
          f"days on the right\n")

    def pp(ci):
        return {k: v * 100 if k in ("lo", "hi") else v for k, v in ci.items()}

    # rung 0 — the original analysis
    d = base.loc[base.index.isin(tnbc_orig)].dropna(subset=["T"])
    Y, A = d["Y5"].to_numpy(float), d["T"].to_numpy(int)
    t, e = d["OS.time"].to_numpy(float), d["OS"].to_numpy(int)
    ci = bootstrap(risk_diff_boot(Y, A, len(d)), len(d), B=B_BOOT, seed=SEED)
    add(0, "original analysis (dummy T, imputed TNBC, binary 5y)",
        "risk difference (pp)",
        float((Y[A == 1].mean() - Y[A == 0].mean()) * 100), pp(ci),
        len(d), int(A.sum()), int((A == 0).sum()), int(Y.sum()),
        rmst_contrast(t, e, A), "the original analysis's design")

    # rung 1 — treatment fixed
    Y, A = d["Y5"].to_numpy(float), d["chemo"].to_numpy(int)
    ci = bootstrap(risk_diff_boot(Y, A, len(d)), len(d), B=B_BOOT, seed=SEED)
    add(1, "+ real chemotherapy receipt (treatment_type)", "risk difference (pp)",
        float((Y[A == 1].mean() - Y[A == 0].mean()) * 100), pp(ci),
        len(d), int(A.sum()), int((A == 0).sum()), int(Y.sum()),
        rmst_contrast(t, e, A), "T replaced by therapy actually recorded")

    # rung 2 — subgroup fixed
    d2 = base.loc[base.index.isin(tnbc_dir)]
    Y, A = d2["Y5"].to_numpy(float), d2["chemo"].to_numpy(int)
    t2, e2 = d2["OS.time"].to_numpy(float), d2["OS"].to_numpy(int)
    ci = bootstrap(risk_diff_boot(Y, A, len(d2)), len(d2), B=B_BOOT, seed=SEED)
    r2 = rmst_contrast(t2, e2, A)
    add(2, "+ TNBC from measured receptors only", "risk difference (pp)",
        float((Y[A == 1].mean() - Y[A == 0].mean()) * 100), pp(ci),
        len(d2), int(A.sum()), int((A == 0).sum()), int(Y.sum()),
        r2, "70 imputation-derived patients removed")

    print(f"\n  scale: RMST difference at {TAU} days "
          f"(days; positive = longer survival in treated)")
    print(f"  NOT comparable with the risk differences above -- different "
          f"scale. The right-hand column is comparable throughout.\n")

    # rung 3 — outcome fixed (still time zero at diagnosis, no landmark).
    # On the RMST scale this rung is rung 2: the correction is the scale.
    add(3, "+ RMST with IPCW instead of censoring-blind binary", "RMST (days)",
        r2["estimate"], dict(lo=r2["lo"], hi=r2["hi"]), len(d2), int(A.sum()),
        int((A == 0).sum()), r2["events"], r2,
        "no landmark yet: time zero still at diagnosis")

    # rung 4 — time zero fixed
    lm = pd.read_csv(OUTPUT / "05_cohort_landmark90.csv", index_col=0)
    lm.index = barcode12(lm.index)
    L = lm[lm["TNBC"].fillna(False).astype(bool)]
    t, e = L["time"].to_numpy(float), L["event"].to_numpy(int)
    A = L["A_chemo"].to_numpy(int)
    r4 = rmst_contrast(t, e, A)
    add(4, "+ day-90 landmark, treatment fixed at day 90", "RMST (days)",
        r4["estimate"], dict(lo=r4["lo"], hi=r4["hi"]), len(L), int(A.sum()),
        int((A == 0).sum()), r4["events"], r4, "immortal time removed")

    # rung 5 — confounding addressed
    _est_p = OUTPUT / "07_estimates.csv"
    est = pd.read_csv(_est_p) if _est_p.exists() else None
    if est is not None:
        f = est[(est["arm"] == "TNBC_chemo") & (est["adjustment"] == PRIMARY_SET)
                & (est["tau"] == TAU)
                & (est["estimator"] == "IPCW-KM contrast")]
        if len(f):
            r = f.iloc[0]
            r5 = dict(estimate=float(r["estimate_days"]), lo=float(r["ci_lo"]),
                      hi=float(r["ci_hi"]), events=int(r["events"]))
            add(5, "+ confounding adjustment (clinical set, primary estimator)",
                "RMST (days)", r5["estimate"], dict(lo=r5["lo"], hi=r5["hi"]),
                int(r["n"]), int(r["n_treated"]), int(r["n_control"]),
                int(r["events"]), r5, "primary specification")
    else:
        print("  5  (run 07_estimate.py first for the adjusted rung)")

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT / "08_ladder.csv", index=False)
    res.put("ladder", out.to_dict(orient="records"))

    print(f"\n  Reading the ladder")
    r0, r2_ = out.iloc[0], out.iloc[2]
    print(f"    The original analysis reports {r0['estimate']:+.1f} pp. Replacing "
          f"the treatment dummy with actual chemotherapy receipt and the imputed")
    print(f"    TNBC subgroup with measured receptors moves it to "
          f"{r2_['estimate']:+.1f} pp on the same scale, with the cohort falling "
          f"from {int(r0['n'])} to {int(r2_['n'])} patients.")
    print(f"    On the RMST scale the same corrections move the estimate from "
          f"{r0['rmst_estimate']:+.1f} to {r2_['rmst_estimate']:+.1f} days, the "
          f"landmark to {out.iloc[4]['rmst_estimate']:+.1f}, and adjustment to "
          f"{out.iloc[-1]['rmst_estimate']:+.1f}.")
    print(f"    Every rung's interval is a {B_BOOT}-replicate patient-level "
          f"bootstrap for that rung's own estimand.")


if __name__ == "__main__":
    main()
