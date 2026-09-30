"""
02_tnbc.py — TNBC and the other receptor subgroups, from directly measured results.

Rule: a receptor is negative only when a test directly reported it negative.
Equivocal, Conflicting and Not-tested are *indeterminate* and exclude the patient
from any subgroup whose definition depends on that receptor. Cohort-defining
variables are never imputed.

The audited pipeline instead MICE-imputed ER/PR/HER2, z-scaled them, and then
thresholded at > 0.5. Because the imputed values all land between the two modes,
that threshold selects exactly the "positive mode" rows and sweeps every imputed
patient into "negative" -- inflating TNBC from 156 to 226.

Also cross-checks against brca_pam50.txt, which covers 74 patients. That is far
too few to validate a subgroup definition at cohort scale; the raw contingency
counts are reported rather than a bare agreement rate, and PAM50 is not treated
as ground truth.

Produces:
    02_subgroups.csv       ->  cohort frame, consumed by stage 05
    02_tnbc_ledger.csv     ->  Table 1
    02_pam50_crosstab.csv  ->  Section 6.1 text
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from lib_io import DATA, OUTPUT, Ledger, Results, banner, barcode12, require  # noqa: E402

DETERMINATE = ("Positive", "Negative")


def main() -> None:
    banner("02 — TNBC and receptor subgroups (directly measured only)")
    res = Results()

    st = pd.read_csv(OUTPUT / "01_receptor_status.csv", index_col=0)

    # Cohort frame = the 1,076 patients carried by the audited analysis, so the
    # rebuilt subgroups are directly comparable to the published ones.
    cf = pd.read_csv(require(DATA / "causal_features_dataset.csv",
                             "cohort frame for subgroup construction"), index_col=0)
    cf.index = barcode12(cf.index)
    cohort = pd.DataFrame(index=cf.index)
    cohort = cohort.join(st)
    for c in ("ER", "PR", "HER2"):
        cohort[c] = cohort[c].fillna("Not tested")

    n_total = len(cohort)
    print(f"  cohort frame                    : {n_total:,} patients")

    det = cohort[["ER", "PR", "HER2"]].isin(DETERMINATE)
    n_all_det = int(det.all(axis=1).sum())
    print(f"  all three receptors determinate : {n_all_det:,}")
    print(f"  >=1 indeterminate or missing    : {n_total - n_all_det:,}")

    # ── TNBC with a full exclusion ledger ────────────────────────────────────
    led = Ledger("TNBC (directly measured)")
    df = led.start(cohort.reset_index().rename(columns={"index": "patient_id"}),
                   "cohort frame (patients in causal_features_dataset.csv)")
    df = led.exclude(df, df["ER"].isin(DETERMINATE),
                     "ER status not directly measured (equivocal/conflicting/not tested)")
    df = led.exclude(df, df["PR"].isin(DETERMINATE),
                     "PR status not directly measured")
    df = led.exclude(df, df["HER2"].isin(DETERMINATE),
                     "HER2 status not directly measured")
    df = led.exclude(df, df["ER"].eq("Negative"), "ER positive")
    df = led.exclude(df, df["PR"].eq("Negative"), "PR positive")
    df = led.exclude(df, df["HER2"].eq("Negative"), "HER2 positive")
    led.check()
    tnbc_ids = set(df["patient_id"])
    print(f"\n{led.to_markdown()}")
    (led.frame()).to_csv(OUTPUT / "02_tnbc_ledger.csv", index=False)

    # ── The other two prespecified subgroups, same rule ──────────────────────
    er, pr, h2 = cohort["ER"], cohort["PR"], cohort["HER2"]
    hr_pos = (
        er.isin(DETERMINATE) & pr.isin(DETERMINATE) & h2.eq("Negative")
        & (er.eq("Positive") | pr.eq("Positive"))
    )
    her2_pos = h2.eq("Positive")

    subgroups = pd.DataFrame({
        "TNBC": cohort.index.isin(tnbc_ids),
        "HR_pos_HER2_neg": hr_pos.to_numpy(),
        "HER2_pos": her2_pos.to_numpy(),
    }, index=cohort.index)
    out = cohort.join(subgroups)
    out.to_csv(OUTPUT / "02_subgroups.csv")

    print("\n  subgroup sizes (directly measured only):")
    for c in subgroups.columns:
        print(f"    {c:20s} {int(subgroups[c].sum()):5,d}")

    # ── Comparison with the published, imputation-based TNBC ─────────────────
    v2 = pd.read_csv(require(DATA / "ite_ready_dataset_v2.csv",
                             "published TNBC definition for comparison"))
    v2["patient_id"] = barcode12(v2["patient_id"])
    pub_er = v2["ER_status"].fillna(0) > 0.5
    pub_pr = v2["PR_status"].fillna(0) > 0.5
    pub_h2 = v2["HER2_status"].fillna(0) > 0.5
    v2["TNBC_published"] = (~pub_er) & (~pub_pr) & (~pub_h2)
    pub_ids = set(v2.loc[v2["TNBC_published"], "patient_id"])

    only_pub = pub_ids - tnbc_ids
    only_new = tnbc_ids - pub_ids
    print(f"\n  published TNBC (imputed+scaled) : {len(pub_ids):,}")
    print(f"  rebuilt TNBC (measured only)    : {len(tnbc_ids):,}")
    print(f"  in published but not rebuilt    : {len(only_pub):,}")
    print(f"  in rebuilt but not published    : {len(only_new):,}")

    comp = cohort.loc[sorted(only_pub), ["ER", "PR", "HER2"]]
    breakdown = {c: comp[c].value_counts().to_dict() for c in comp.columns}
    print("\n  receptor states of the patients the published definition added:")
    print(pd.DataFrame(breakdown).fillna(0).astype(int).to_string())
    comp.to_csv(OUTPUT / "02_published_only_tnbc.csv")

    # ── PAM50 cross-check (74 patients total; not ground truth) ──────────────
    pam = pd.read_csv(require(DATA / "brca_pam50.txt", "PAM50 cross-check"), sep="\t")
    pam["patient_id"] = barcode12(pam["sample"])
    pam = pam.drop_duplicates("patient_id").set_index("patient_id")
    overlap = cohort.index.intersection(pam.index)
    print(f"\n  PAM50 file: {len(pam):,} unique patients; "
          f"{len(overlap):,} overlap the cohort")

    px = cohort.loc[overlap].join(pam[["PAM50"]])
    px["TNBC"] = px.index.isin(tnbc_ids)
    px["determinate"] = px[["ER", "PR", "HER2"]].isin(DETERMINATE).all(axis=1)
    usable = px[px["determinate"] & px["PAM50"].ne("Unknown")]
    ct = pd.crosstab(usable["TNBC"], usable["PAM50"].eq("Basal"))
    ct.index = ["not TNBC", "TNBC"]
    ct.columns = ["not Basal", "Basal"]
    print(f"\n  usable for the cross-check (determinate receptors, PAM50 called): "
          f"{len(usable):,}")
    print("\n  TNBC x PAM50-Basal contingency (raw counts):")
    print(ct.to_string())
    ct.to_csv(OUTPUT / "02_pam50_crosstab.csv")

    agree = float(np.trace(ct.to_numpy()) / ct.to_numpy().sum()) if len(usable) else float("nan")
    print(f"\n  concordance on {len(usable)} patients: {agree:.3f}")
    print("  NOTE: 74 PAM50 calls cannot validate a subgroup definition across "
          "1,076 patients. PAM50 is a transcriptional subtype, not an IHC assay;\n"
          "        disagreement is expected and is not evidence either way.")

    res.update({
        "cohort_frame_n": n_total,
        "receptors_all_determinate_n": n_all_det,
        "receptors_indeterminate_n": n_total - n_all_det,
        "tnbc_measured_n": len(tnbc_ids),
        "tnbc_published_n": len(pub_ids),
        "tnbc_published_only_n": len(only_pub),
        "tnbc_measured_only_n": len(only_new),
        "tnbc_published_only_states": breakdown,
        "hr_pos_her2_neg_n": int(hr_pos.sum()),
        "her2_pos_n": int(her2_pos.sum()),
        "pam50_patients": int(len(pam)),
        "pam50_overlap": int(len(overlap)),
        "pam50_usable": int(len(usable)),
        "pam50_crosstab": ct.to_dict(),
        "pam50_concordance": agree,
    })


if __name__ == "__main__":
    main()
