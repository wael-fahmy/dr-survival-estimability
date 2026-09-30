"""
03_survival.py — overall survival, one documented derivation.

Primary source: data/outcome.csv.

Provenance. The brief specifies TCGA-BRCA.survival.tsv. That file is not present
in this repository (checked on disk, inside every archive, and in the Thesis_v3
git history). outcome.csv is not a substitute for it -- it is its recorded
product: 01_Causal_feature_extraction/RNA/01_survival_preprocessing.ipynb reads
TCGA-BRCA.survival.tsv (1,232 samples, columns OS.time / OS / _PATIENT), keeps
primary-tumour barcodes, and its stored execution output shows
`Saved: data/outcome.csv  Shape: (1076, 2)`. The file supplied here matches that
shape exactly. The derivation chain is therefore documented end to end even
though the first link is absent, and this script states that plainly rather than
pretending the Xena file was read.

Cross-check: GDC clinical.tsv carries an independent survival record
(demographic.vital_status, demographic.days_to_death,
diagnoses.days_to_last_follow_up). It is compared, not merged. Differences are
reported, not reconciled away.

Produces:
    03_survival.csv         ->  cohort frame, consumed by stage 05
    03_survival_ledger.csv  ->  Table 1
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from lib_io import (DATA, OUTPUT, TCGA_NA, Ledger, Results, banner,  # noqa: E402
                    barcode12, require)
from lib_est import median_followup  # noqa: E402

XENA_SURVIVAL = DATA / "TCGA-BRCA.survival.tsv"


def main() -> None:
    banner("03 — Overall survival")
    res = Results()

    if XENA_SURVIVAL.exists():
        print(f"  TCGA-BRCA.survival.tsv : PRESENT — deriving directly from it")
        surv = pd.read_csv(XENA_SURVIVAL, sep="\t", index_col=0)
        surv = surv[["OS.time", "OS"]].dropna()
        surv = surv[surv["OS.time"] > 0]
        vial = surv.index.str[13:15]
        surv = surv[vial.isin(["01", "06"])]
        surv.index = barcode12(surv.index)
        surv = surv[~surv.index.duplicated(keep="first")]
        derivation = ("TCGA-BRCA.survival.tsv -> drop missing/zero OS.time -> "
                      "keep primary tumour (01) and metastatic (06) barcodes -> "
                      "first record per patient")
    else:
        print(f"  TCGA-BRCA.survival.tsv : ABSENT")
        print(f"  Using its recorded product data/outcome.csv "
              f"(see 01_survival_preprocessing.ipynb, saved shape (1076, 2))")
        o = pd.read_csv(require(DATA / "outcome.csv", "overall survival"), index_col=0)
        surv = o[["OS.time", "OS"]].copy()
        surv.index = barcode12(surv.index)
        surv = surv[~surv.index.duplicated(keep="first")]
        derivation = ("data/outcome.csv, itself derived from TCGA-BRCA.survival.tsv "
                      "by 01_survival_preprocessing.ipynb (1,232 samples -> "
                      "primary-tumour barcodes -> 1,076)")

    surv.index.name = "patient_id"
    led = Ledger("Survival records")
    s = led.start(surv.reset_index(), "records in the survival source")
    s = led.exclude(s, s["OS.time"].notna() & s["OS"].notna(),
                    "missing OS or OS.time")
    s = led.exclude(s, s["OS.time"] > 0, "non-positive follow-up time")
    led.check()
    surv = s.set_index("patient_id")

    n = len(surv)
    ev = int(surv["OS"].sum())
    med_ost = float(surv["OS.time"].median())
    med_fu = median_followup(surv["OS.time"].to_numpy(), surv["OS"].to_numpy())

    print(f"\n  derivation      : {derivation}")
    print(f"  n patients      : {n:,}")
    print(f"  events (deaths) : {ev:,}  ({ev / n * 100:.1f}%)")
    print(f"  median OS.time  : {med_ost:,.0f} days")
    print(f"  median follow-up (reverse KM) : {med_fu:,.0f} days")

    # What the five-year binary outcome does to this cohort.
    tau5 = 1825
    died5 = int(((surv["OS"] == 1) & (surv["OS.time"] <= tau5)).sum())
    died_after = int(((surv["OS"] == 1) & (surv["OS.time"] > tau5)).sum())
    cens_before = int(((surv["OS"] == 0) & (surv["OS.time"] < tau5)).sum())
    known_alive = int(((surv["OS"] == 0) & (surv["OS.time"] >= tau5)).sum())
    print(f"\n  Five-year binary outcome as defined in the audited pipeline")
    print(f"    Y=1  died within 1825 d            : {died5:,}")
    print(f"    Y=0  died after 1825 d             : {died_after:,}")
    print(f"    Y=0  CENSORED before 1825 d        : {cens_before:,} "
          f"({cens_before / n * 100:.1f}%)  <- five-year status unknown")
    print(f"    Y=0  observed alive at 1825 d      : {known_alive:,} "
          f"({known_alive / n * 100:.1f}%)")

    # Reconciliation 1: causal_features_dataset.csv
    cf = pd.read_csv(require(DATA / "causal_features_dataset.csv",
                             "survival reconciliation"), index_col=0)
    cf.index = barcode12(cf.index)
    common = surv.index.intersection(cf.index)
    t_match = int(np.isclose(surv.loc[common, "OS.time"],
                             cf.loc[common, "OS.time"]).sum())
    e_match = int((surv.loc[common, "OS"] == cf.loc[common, "OS"]).sum())
    print(f"\n  Reconciliation vs causal_features_dataset.csv")
    print(f"    rows in analysis dataset : {len(cf):,} (events {int(cf['OS'].sum()):,})")
    print(f"    overlapping patients     : {len(common):,}")
    print(f"    OS.time identical        : {t_match:,} / {len(common):,}")
    print(f"    OS identical             : {e_match:,} / {len(common):,}")

    # Reconciliation 2: GDC clinical.tsv, an independent record
    clin = pd.read_csv(require(DATA / "clinical.tsv", "GDC survival cross-check"),
                       sep="\t", na_values=TCGA_NA, low_memory=False)
    clin["patient_id"] = barcode12(clin["cases.submitter_id"])
    g = clin.groupby("patient_id").agg(
        vital=("demographic.vital_status", "first"),
        days_to_death=("demographic.days_to_death", "max"),
        days_last_fu=("diagnoses.days_to_last_follow_up", "max"),
    )
    g["gdc_event"] = (g["vital"] == "Dead").astype(int)
    g["gdc_time"] = np.where(g["gdc_event"] == 1, g["days_to_death"], g["days_last_fu"])
    print(f"\n  Reconciliation vs GDC clinical.tsv (independent record)")
    print(f"    patients   : {len(g):,}   deaths : {int(g['gdc_event'].sum()):,}")
    both = surv.index.intersection(g.index)
    ev_agree = int((surv.loc[both, "OS"] == g.loc[both, "gdc_event"]).sum())
    dt = (surv.loc[both, "OS.time"].to_numpy()
          - pd.to_numeric(g.loc[both, "gdc_time"], errors="coerce").to_numpy())
    dt = dt[np.isfinite(dt)]
    print(f"    overlapping patients          : {len(both):,}")
    print(f"    death indicator agrees        : {ev_agree:,} / {len(both):,} "
          f"({ev_agree / len(both) * 100:.1f}%)")
    print(f"    follow-up time identical      : {int((dt == 0).sum()):,} / {len(dt):,}")
    print(f"    median |difference| in days   : {np.median(np.abs(dt)):,.0f}")
    print(f"    NOTE: the two sources disagree on {len(both) - ev_agree} death "
          f"indicators. Reported, not reconciled.")

    surv.to_csv(OUTPUT / "03_survival.csv")
    led.frame().to_csv(OUTPUT / "03_survival_ledger.csv", index=False)

    res.update({
        "survival_source": derivation,
        "survival_xena_present": XENA_SURVIVAL.exists(),
        "survival_n": n, "survival_events": ev,
        "survival_median_os_time": med_ost,
        "survival_median_followup_reverseKM": med_fu,
        "binary5y_died_within": died5,
        "binary5y_died_after_coded_survivor": died_after,
        "binary5y_censored_before_coded_survivor": cens_before,
        "binary5y_observed_alive": known_alive,
        "recon_cf_n": int(len(cf)), "recon_cf_events": int(cf["OS"].sum()),
        "recon_cf_overlap": int(len(common)),
        "recon_cf_time_identical": t_match, "recon_cf_event_identical": e_match,
        "recon_gdc_n": int(len(g)), "recon_gdc_deaths": int(g["gdc_event"].sum()),
        "recon_gdc_overlap": int(len(both)),
        "recon_gdc_event_agree": ev_agree,
        "recon_gdc_time_identical": int((dt == 0).sum()),
    })


if __name__ == "__main__":
    main()
