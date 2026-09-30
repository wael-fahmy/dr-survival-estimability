"""
06_site.py — contributing site (tissue source site), and whether it matters.

Site = characters 6 and 7 of the TCGA barcode (TCGA-XX-YYYY -> XX). It encodes
which institution contributed the case, and therefore local practice, referral
pattern, and how completely follow-up was recorded. The audited pipeline never
adjusted for it.

Reports treatment-annotation completeness by site and tests the site-survival
association by permutation, since no chi-square / log-rank distribution is
available in this environment.

Produces:
    06_site_summary.csv  ->  Section 6.7 and Figure 5
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from lib_io import OUTPUT, SEED, Results, banner  # noqa: E402
from lib_est import km_estimate, km_rmst, median_followup  # noqa: E402

N_PERM = 2000
TAU = 1095


def logrank_like_stat(time, event, groups):
    """Sum of squared standardized deviations of observed vs expected deaths.

    A permutation reference distribution is built by shuffling the group labels,
    so no chi-square quantile is needed.
    """
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    groups = np.asarray(groups)
    levels = np.unique(groups)
    o = np.zeros(len(levels))
    e = np.zeros(len(levels))
    for t in np.unique(time[event == 1]):
        at_risk = time >= t
        n = at_risk.sum()
        d = ((time == t) & (event == 1)).sum()
        if n == 0:
            continue
        for i, g in enumerate(levels):
            ng = (at_risk & (groups == g)).sum()
            o[i] += ((time == t) & (event == 1) & (groups == g)).sum()
            e[i] += d * ng / n
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = np.nansum((o - e) ** 2 / np.where(e > 0, e, np.nan))
    return float(stat)


def main() -> None:
    banner("06 — Contributing site")
    res = Results()

    df = pd.read_csv(OUTPUT / "05_cohort_landmark90.csv", index_col=0)
    df["site"] = df["site"].astype(str)

    recs = []
    for site, s in df.groupby("site"):
        recs.append(dict(
            site=site,
            n=len(s),
            deaths=int(s["event"].sum()),
            death_rate=float(s["event"].mean()),
            chemo_rate=float(s["A_chemo"].mean()),
            any_chemo_rate=float(s["chemo_ever"].mean()),
            treat_annotated=float(s["any_surgery"].fillna(0).gt(0).mean()),
            median_fu=median_followup(s["time"].to_numpy(), s["event"].to_numpy()),
            median_time=float(s["time"].median()),
        ))
    g = (pd.DataFrame(recs).set_index("site")
         .sort_values("n", ascending=False))

    print(f"  sites: {len(g)}   cohort: {len(df):,} patients, "
          f"{int(df['event'].sum()):,} deaths")
    print(f"\n  sites with n >= 20:")
    print(g[g["n"] >= 20].round(3).to_string())
    g.to_csv(OUTPUT / "06_site_summary.csv")

    big = g[g["n"] >= 20]
    print(f"\n  across sites with n >= 20:")
    print(f"    chemotherapy-at-landmark rate ranges "
          f"{big['chemo_rate'].min():.3f} to {big['chemo_rate'].max():.3f}")
    print(f"    death rate ranges "
          f"{big['death_rate'].min():.3f} to {big['death_rate'].max():.3f}")
    print(f"    median follow-up ranges "
          f"{big['median_fu'].min():,.0f} to {big['median_fu'].max():,.0f} days")

    top = g.nlargest(3, "deaths")
    share = top["deaths"].sum() / g["deaths"].sum()
    print(f"\n    the 3 sites contributing the most deaths hold "
          f"{top['n'].sum():,} of {len(df):,} patients "
          f"({top['n'].sum() / len(df) * 100:.1f}%) but "
          f"{int(top['deaths'].sum()):,} of {int(g['deaths'].sum()):,} deaths "
          f"({share * 100:.1f}%)")

    # Permutation test of the site-survival association, restricted to sites
    # with enough patients to carry information.
    sub = df[df["site"].isin(big.index)]
    obs = logrank_like_stat(sub["time"], sub["event"], sub["site"])
    rng = np.random.default_rng(SEED)
    null = np.empty(N_PERM)
    grp = sub["site"].to_numpy()
    for i in range(N_PERM):
        null[i] = logrank_like_stat(sub["time"], sub["event"], rng.permutation(grp))
    p = float((np.sum(null >= obs) + 1) / (N_PERM + 1))
    print(f"\n  site vs survival, permutation log-rank-like test "
          f"({len(big)} sites, n={len(sub):,}, {N_PERM:,} permutations)")
    print(f"    observed statistic {obs:.2f}   permutation p = {p:.4f}")
    verdict = ("site is associated with survival" if p < 0.05
               else "no detectable site-survival association at this size")
    print(f"    -> {verdict}")

    # Association between site and treatment: permutation on chemo rate spread.
    obs_t = float(np.var(big["chemo_rate"], ddof=0))
    null_t = np.empty(N_PERM)
    a = sub["A_chemo"].to_numpy()
    for i in range(N_PERM):
        perm = pd.Series(rng.permutation(a), index=sub.index).groupby(sub["site"]).mean()
        null_t[i] = float(np.var(perm, ddof=0))
    p_t = float((np.sum(null_t >= obs_t) + 1) / (N_PERM + 1))
    print(f"\n  site vs treatment, permutation test on between-site variance of "
          f"the chemotherapy rate")
    print(f"    observed variance {obs_t:.5f}   permutation p = {p_t:.4f}")
    print(f"    -> site predicts treatment "
          f"{'and survival both: it is a confounder' if (p < .05 and p_t < .05) else ''}")

    res.update({
        "n_sites": int(len(g)),
        "n_sites_ge20": int(len(big)),
        "site_chemo_rate_min": float(big["chemo_rate"].min()),
        "site_chemo_rate_max": float(big["chemo_rate"].max()),
        "site_death_rate_min": float(big["death_rate"].min()),
        "site_death_rate_max": float(big["death_rate"].max()),
        "site_medfu_min": float(big["median_fu"].min()),
        "site_medfu_max": float(big["median_fu"].max()),
        "site_top3_death_share": float(share),
        "site_top3_patient_share": float(top["n"].sum() / len(df)),
        "site_survival_stat": obs, "site_survival_p": p,
        "site_treatment_var": obs_t, "site_treatment_p": p_t,
        "site_table": g.reset_index().to_dict(orient="records"),
    })


if __name__ == "__main__":
    main()
