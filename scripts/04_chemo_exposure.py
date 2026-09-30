"""
04_chemo_exposure.py — chemotherapy exposure and its timing.

Source: clinical.tsv, treatments.treatment_type and
treatments.days_to_treatment_start (days from diagnosis).

The audited pipeline used treatment_type for the arm indicators but never read
days_to_treatment_start, so therapy that began months after diagnosis was
treated as if it were present at baseline. 26% of dated chemotherapy starts fall
after day 90; carrying those backwards to time zero is immortal time bias.

It also derived its *primary* treatment variable T from a one-hot dummy of the
serialized string "['yes', 'yes']" in treatments.treatment_or_therapy rather
than from treatment_type at all. That column is reconstructed here so the
disagreement can be quantified.

Produces:
    04_exposure.csv           ->  Section 6.1 text
    04_T_vs_chemo.csv         ->  Introduction
    04_chemo_first_start.csv  ->  Section 6.1 text
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from lib_io import (DATA, OUTPUT, TCGA_NA, Results, banner,  # noqa: E402
                    barcode12, require)

KEYWORDS = {
    "chemo": "chemo",
    "hormone": "hormone",
    "targeted": "targeted",
    "radiation": "radiation",
    "immunotherapy": "immunotherapy",
    "pharmaceutical_nos": "pharmaceutical",
    "surgery": "surgery",
}


def main() -> None:
    banner("04 — Chemotherapy exposure and start timing")
    res = Results()

    clin = pd.read_csv(require(DATA / "clinical.tsv", "treatment exposure"),
                       sep="\t", na_values=TCGA_NA, low_memory=False)
    clin["patient_id"] = barcode12(clin["cases.submitter_id"])
    tt = clin["treatments.treatment_type"]
    ds = pd.to_numeric(clin["treatments.days_to_treatment_start"], errors="coerce")
    clin["start"] = ds

    print(f"  clinical.tsv rows            : {len(clin):,}")
    print(f"  unique patients              : {clin['patient_id'].nunique():,}")
    print(f"  rows with a treatment_type   : {int(tt.notna().sum()):,}")
    print(f"  rows with a start date       : {int(ds.notna().sum()):,} "
          f"({ds.notna().mean() * 100:.1f}%)")

    print("\n  treatments.treatment_type:")
    print(tt.value_counts(dropna=False).to_string())
    tt.value_counts(dropna=False).to_csv(OUTPUT / "04_treatment_type_counts.csv")

    # ── Per-patient exposure flags and first start date per modality ─────────
    rows = {}
    first_start = {}
    for name, kw in KEYWORDS.items():
        m = tt.str.lower().str.contains(kw, na=False)
        pts = set(clin.loc[m, "patient_id"])
        rows[name] = pts
        fs = (clin.loc[m & clin["start"].notna()]
                  .groupby("patient_id")["start"].min())
        first_start[name] = fs
        print(f"    {name:20s} patients={len(pts):5,d}  dated={len(fs):5,d} "
              f"({len(fs) / max(len(pts), 1) * 100:.1f}%)")

    all_pts = sorted(set(clin["patient_id"]))
    exp = pd.DataFrame(index=pd.Index(all_pts, name="patient_id"))
    for name, pts in rows.items():
        exp[f"any_{name}"] = exp.index.isin(pts).astype(int)
        exp[f"start_{name}"] = first_start[name].reindex(exp.index)

    # ── Completeness and plausibility of the chemotherapy start date ─────────
    chemo_pts = rows["chemo"]
    fs = first_start["chemo"]
    n_chemo, n_dated = len(chemo_pts), len(fs)
    print(f"\n  Chemotherapy start date")
    print(f"    patients with chemotherapy   : {n_chemo:,}")
    print(f"    with >=1 dated record        : {n_dated:,} "
          f"({n_dated / n_chemo * 100:.1f}%)")
    print(f"    with NO date at all          : {n_chemo - n_dated:,}")
    q = fs.quantile([0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    print(f"\n    first chemotherapy start, days from diagnosis:")
    print(f"      min {fs.min():,.0f}   p05 {q[0.05]:,.0f}   p25 {q[0.25]:,.0f}   "
          f"median {q[0.50]:,.0f}   p75 {q[0.75]:,.0f}   p95 {q[0.95]:,.0f}   "
          f"max {fs.max():,.0f}")
    neg = int((fs < 0).sum())
    zero = int((fs == 0).sum())
    gt90 = int((fs > 90).sum())
    gt365 = int((fs > 365).sum())
    print(f"\n    negative (before diagnosis, implausible) : {neg:,}  "
          f"(min {fs.min():,.0f})")
    print(f"    exactly zero                             : {zero:,}")
    print(f"    later than day 90                        : {gt90:,} "
          f"({gt90 / n_dated * 100:.1f}%)  <- excluded by a day-90 landmark")
    print(f"    later than day 365                       : {gt365:,}")

    fs.to_csv(OUTPUT / "04_chemo_first_start.csv")
    exp.to_csv(OUTPUT / "04_exposure.csv")

    # ── Reconstruct the audited pipeline's primary treatment variable ────────
    tot = clin["treatments.treatment_or_therapy"]
    print(f"\n  treatments.treatment_or_therapy (source of the published T):")
    print(tot.value_counts(dropna=False).to_string())

    cf = pd.read_csv(require(DATA / "causal_features_dataset.csv",
                             "published treatment variable"), index_col=0)
    cf.index = barcode12(cf.index)
    tcol = [c for c in cf.columns if "treatment_or_therapy" in c and "yes" in c]
    published_T = cf[tcol[0]].astype(int)
    print(f"\n  published T column : {tcol[0]}")
    print(f"    T=1 {int(published_T.sum()):,}   T=0 {int((published_T == 0).sum()):,}")

    aligned = pd.DataFrame({"T": published_T})
    aligned["chemo"] = aligned.index.isin(chemo_pts).astype(int)
    aligned["any_therapy"] = aligned.index.isin(
        set().union(*[rows[k] for k in ("chemo", "hormone", "targeted",
                                        "radiation", "immunotherapy",
                                        "pharmaceutical_nos")])
    ).astype(int)

    ct = pd.crosstab(aligned["T"], aligned["chemo"], margins=True)
    print(f"\n  published T  x  actual chemotherapy receipt:")
    print(ct.to_string())
    ct.to_csv(OUTPUT / "04_T_vs_chemo.csv")
    mis_ctrl = int(((aligned["T"] == 0) & (aligned["chemo"] == 1)).sum())
    mis_trt = int(((aligned["T"] == 1) & (aligned["chemo"] == 0)).sum())
    print(f"\n    chemotherapy recipients placed in the CONTROL arm : {mis_ctrl:,}")
    print(f"    non-recipients placed in the TREATED arm          : {mis_trt:,}")

    ct2 = pd.crosstab(aligned["T"], aligned["any_therapy"], margins=True)
    print(f"\n  published T  x  any recorded therapy:")
    print(ct2.to_string())
    treated_controls = int(((aligned["T"] == 0) & (aligned["any_therapy"] == 1)).sum())
    n_ctrl = int((aligned["T"] == 0).sum())
    print(f"\n    'controls' who received some recorded therapy : "
          f"{treated_controls:,} / {n_ctrl:,} "
          f"({treated_controls / n_ctrl * 100:.1f}%)")
    print(f"    the n=1,076 'any treatment' arm is therefore a "
          f"treated-versus-treated comparison.")

    res.update({
        "chemo_patients": n_chemo,
        "chemo_dated": n_dated,
        "chemo_undated": n_chemo - n_dated,
        "chemo_start_median": float(fs.median()),
        "chemo_start_p25": float(q[0.25]), "chemo_start_p75": float(q[0.75]),
        "chemo_start_min": float(fs.min()), "chemo_start_max": float(fs.max()),
        "chemo_start_negative": neg, "chemo_start_zero": zero,
        "chemo_start_gt90": gt90, "chemo_start_gt365": gt365,
        "published_T_col": tcol[0],
        "published_T_1": int(published_T.sum()),
        "published_T_0": int((published_T == 0).sum()),
        "T_vs_chemo_misclassified_control": mis_ctrl,
        "T_vs_chemo_misclassified_treated": mis_trt,
        "controls_receiving_therapy": treated_controls,
        "n_controls": n_ctrl,
        "exposure_counts": {k: len(v) for k, v in rows.items()},
    })


if __name__ == "__main__":
    main()
