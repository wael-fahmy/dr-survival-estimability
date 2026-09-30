"""Stage 18 — Monte Carlo error of the RNG-dependent results.

The Biometrical Journal Checklist for Code and Data Supplements (2025) requires
that code depending on random number generation be re-run several times and the
Monte Carlo error shown to be negligible, in the sense that it does not affect
the substantive conclusions.

Three things in this pipeline consume randomness:

  the patient-level bootstrap behind every reported interval (stage 07),
  the fold assignment inside cross-fitted AIPW (stage 07),
  the simulation grid and the threshold bootstrap that calibrate rho
  (stage 16).

This stage re-runs stages 07 and 16 under alternate seeds and reports, for each
headline quantity, the spread across seeds against the width of the quantity's
own bootstrap interval. A Monte Carlo error that is small relative to the
interval it sits inside is not affecting any conclusion drawn from that
interval.

Under REVISION_OUTPUT, stage 07 computes only the quantities harvested here
(primary adjustment set, primary horizon, both estimators, each with the full
bootstrap) and stage 16 skips its penalty sweep; neither is read by this stage.
The canonical output/ is never written to. Each replicate runs against its own
copy of output/ in a scratch directory, with REVISION_SEED and REVISION_OUTPUT
set; lib_io reads both from the environment and falls back to the canonical
values when they are unset, so an ordinary run is unaffected.

Produces:  output/18_montecarlo_error.csv
           output/18_montecarlo_summary.csv
Feeds:     AUDIT_REPORT.md (stage 11), SUPPLEMENT_CHECKLIST.md
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from lib_io import OUTPUT, SEED, Results, banner  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
REVISION = SCRIPTS.parent

# Five alternate seeds plus the canonical one. Chosen as fixed offsets rather
# than drawn at random so this stage is itself reproducible.
ALT_SEEDS = [SEED + off for off in (101, 202, 303, 404, 505)]

# The quantities a reader would draw a conclusion from.
ARMS = [("TNBC_chemo", "IPCW-KM"), ("TNBC_chemo", "AIPW"),
        ("ALL_chemo", "IPCW-KM"), ("ALL_chemo", "AIPW")]
TAU = 1095


def _estimator_key(name: str) -> str:
    return "AIPW" if str(name).startswith("AIPW") else "IPCW-KM"


def harvest(out_dir: Path) -> dict:
    """Pull the headline numbers out of one replicate's output directory."""
    got = {}
    est = pd.read_csv(out_dir / "07_estimates.csv")
    primary = Results().get("primary_adjustment_set") or "CLINICAL"
    est = est[(est["tau"] == TAU) & (est["adjustment"] == primary)]
    est["ekey"] = est["estimator"].map(_estimator_key)
    for arm, ek in ARMS:
        row = est[(est["arm"] == arm) & (est["ekey"] == ek)]
        if len(row) != 1:
            continue
        r = row.iloc[0]
        got[f"{arm}|{ek}|estimate"] = float(r["estimate_days"])
        got[f"{arm}|{ek}|ci_lo"] = float(r["ci_lo"])
        got[f"{arm}|{ek}|ci_hi"] = float(r["ci_hi"])
    fit = pd.read_csv(out_dir / "16_estimability_fit.csv").iloc[0]
    got["rho|threshold"] = float(fit["threshold_rho_fitted"])
    got["rho|threshold_lo"] = float(fit["threshold_rho_lo"])
    got["rho|threshold_hi"] = float(fit["threshold_rho_hi"])
    return got


def run_replicate(seed: int, workdir: Path) -> dict:
    """Copy the canonical output, re-run the RNG-dependent stages, harvest."""
    out = workdir / f"seed_{seed}"
    shutil.copytree(OUTPUT, out)
    env = dict(os.environ, REVISION_SEED=str(seed), REVISION_OUTPUT=str(out),
               PYTHONDONTWRITEBYTECODE="1")
    for stage in ("07_estimate.py", "16_estimability_diagnostic.py"):
        r = subprocess.run([sys.executable, "-B", "-X", "utf8",
                            str(SCRIPTS / stage)],
                           cwd=str(REVISION), env=env,
                           capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            raise SystemExit(f"seed {seed}: {stage} failed\n"
                             f"{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return harvest(out)


def main() -> None:
    banner("18  Monte Carlo error across seeds")

    base = harvest(OUTPUT)
    rows = [dict(seed=SEED, role="canonical", **base)]

    with tempfile.TemporaryDirectory(prefix="mcerr_") as tmp:
        workdir = Path(tmp)
        for i, s in enumerate(ALT_SEEDS, 1):
            print(f"  replicate {i}/{len(ALT_SEEDS)}  seed {s} ...", flush=True)
            rows.append(dict(seed=s, role="alternate", **run_replicate(s, workdir)))

    wide = pd.DataFrame(rows)
    wide.to_csv(OUTPUT / "18_montecarlo_error.csv", index=False)

    # ── summarise: MC spread against the quantity's own interval width ───────
    summary = []
    for arm, ek in ARMS:
        e = wide[f"{arm}|{ek}|estimate"].to_numpy(float)
        lo = wide[f"{arm}|{ek}|ci_lo"].to_numpy(float)
        hi = wide[f"{arm}|{ek}|ci_hi"].to_numpy(float)
        width = float(np.mean(hi - lo))
        # Does the qualitative reading of the interval change across seeds?
        spans_zero = (lo < 0) & (hi > 0)
        summary.append(dict(
            quantity=f"{arm} / {ek} / RMST difference at {TAU} d",
            mean=float(e.mean()), mc_sd=float(e.std(ddof=1)),
            mc_range=float(e.max() - e.min()),
            mean_ci_width=width,
            mc_sd_pct_of_ci_width=100.0 * float(e.std(ddof=1)) / width,
            ci_lo_mc_sd=float(lo.std(ddof=1)), ci_hi_mc_sd=float(hi.std(ddof=1)),
            conclusion="interval spans zero" if spans_zero.all()
            else ("interval excludes zero" if (~spans_zero).all()
                  else "CHANGES ACROSS SEEDS"),
            conclusion_stable=bool(spans_zero.all() or (~spans_zero).all())))

    th = wide["rho|threshold"].to_numpy(float)
    tl = wide["rho|threshold_lo"].to_numpy(float)
    thi = wide["rho|threshold_hi"].to_numpy(float)
    width = float(np.mean(thi - tl))
    # The two arms are classified against the threshold; does either flip?
    # rho for the primary adjustment set, as recorded by stage 16.
    _arms = {a["arm"]: float(a["rho"]) for a in
             (Results().get("estimability_real_arms") or [])
             if a.get("primary_set", True)}
    rho_tn = _arms.get("TNBC_chemo", 3.12)
    rho_all = _arms.get("ALL_chemo", 0.35)
    tn_above = th < rho_tn
    all_below = th > rho_all
    summary.append(dict(
        quantity="rho threshold (calibration)",
        mean=float(th.mean()), mc_sd=float(th.std(ddof=1)),
        mc_range=float(th.max() - th.min()),
        mean_ci_width=width,
        mc_sd_pct_of_ci_width=100.0 * float(th.std(ddof=1)) / width,
        ci_lo_mc_sd=float(tl.std(ddof=1)), ci_hi_mc_sd=float(thi.std(ddof=1)),
        conclusion="triple-negative above, whole cohort below"
        if (tn_above.all() and all_below.all()) else "CHANGES ACROSS SEEDS",
        conclusion_stable=bool(tn_above.all() and all_below.all())))

    s = pd.DataFrame(summary)
    s.to_csv(OUTPUT / "18_montecarlo_summary.csv", index=False)

    worst = float(s["mc_sd_pct_of_ci_width"].max())
    stable = bool(s["conclusion_stable"].all())
    print(f"\n  {len(ALT_SEEDS) + 1} seeds; "
          f"largest Monte Carlo SD is {worst:.1f}% of the corresponding "
          f"interval width")
    for _, r in s.iterrows():
        print(f"    {r['quantity']:<52} MC SD {r['mc_sd']:>7.2f}  "
              f"({r['mc_sd_pct_of_ci_width']:>4.1f}% of CI)  {r['conclusion']}")
    print(f"\n  every substantive conclusion stable across seeds: {stable}")
    if not stable:
        print("  WARNING: a conclusion changes with the seed. This must be "
              "reported, not smoothed over.")

    Results().put("montecarlo", dict(
        n_seeds=len(ALT_SEEDS) + 1, seeds=[SEED] + ALT_SEEDS,
        worst_mc_sd_pct_of_ci_width=worst,
        all_conclusions_stable=stable,
        quantities=s.to_dict(orient="records")))


if __name__ == "__main__":
    main()
