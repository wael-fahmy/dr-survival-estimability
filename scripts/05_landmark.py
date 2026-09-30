"""
05_landmark.py — day-90 landmark and the analysis cohort.

Two things the audited pipeline did not do:

  1. Patients who died or were censored before day 90 cannot contribute a
     meaningful comparison of therapy begun around that time. They are excluded
     and counted.
  2. Treatment status is fixed *as of day 90*. A chemotherapy course that began
     on day 200 is not treatment at the landmark; classifying it as such and
     then measuring survival from diagnosis credits the treated arm with the
     time a patient had to survive in order to receive the treatment at all.
     26.4% of dated chemotherapy starts fall after day 90.

The clock is reset to t - 90 so that all follow-up is measured forward from the
landmark. A day-180 landmark is written alongside as a prespecified sensitivity.

Every exclusion is recorded in a ledger; nothing is dropped silently.

Produces:
    05_cohort_landmark90.csv   ->  the analysis cohort for stages 06-08
    05_cohort_landmark180.csv  ->  Table 9
    05_cohort_ledger.csv       ->  Table 1 and Figure 1
    05_event_counts.csv        ->  Table 2
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from lib_io import (DATA, OUTPUT, TCGA_NA, Ledger, Results, banner,  # noqa: E402
                    barcode12, require, site_code)

LANDMARKS = [90, 180]
PRIMARY_LANDMARK = 90


def build(landmark: int, res: Results | None = None) -> pd.DataFrame:
    surv = pd.read_csv(OUTPUT / "03_survival.csv", index_col=0)
    subs = pd.read_csv(OUTPUT / "02_subgroups.csv", index_col=0)
    exp = pd.read_csv(OUTPUT / "04_exposure.csv", index_col=0)

    clin = pd.read_csv(DATA / "clinical.tsv", sep="\t", na_values=TCGA_NA,
                       low_memory=False)
    clin["patient_id"] = barcode12(clin["cases.submitter_id"])
    covars = clin.groupby("patient_id").agg(
        age=("demographic.age_at_index", "first"),
        stage=("diagnoses.ajcc_pathologic_stage", "first"),
        t_stage=("diagnoses.ajcc_pathologic_t", "first"),
        n_stage=("diagnoses.ajcc_pathologic_n", "first"),
        m_stage=("diagnoses.ajcc_pathologic_m", "first"),
        year_dx=("diagnoses.year_of_diagnosis", "first"),
        grade=("diagnoses.tumor_grade", "first"),
    )

    led = Ledger(f"Analysis cohort — day-{landmark} landmark")
    df = surv.join(subs, how="left").join(exp, how="left").join(covars, how="left")
    df["site"] = site_code(df.index)
    df.index.name = "patient_id"
    df = led.start(df.reset_index(), "patients with an overall-survival record")

    df = led.exclude(df, df["OS.time"].notna() & df["OS"].notna(),
                     "missing survival data")
    df = led.exclude(df, df["OS.time"] >= landmark,
                     f"died or censored before day {landmark} (landmark not reached)")

    # Treatment as of the landmark: a dated start at or before it, and no credit
    # for undated records, which cannot be placed in time.
    start = df["start_chemo"]
    df["A_chemo"] = ((start.notna()) & (start <= landmark) & (start >= 0)).astype(int)
    df["chemo_ever"] = df["any_chemo"].fillna(0).astype(int)
    df["chemo_after_landmark"] = (
        (start.notna()) & (start > landmark)).astype(int)
    df["chemo_undated"] = (
        (df["chemo_ever"] == 1) & (start.isna())).astype(int)

    # Chemotherapy recipients whose start date is missing cannot be placed
    # relative to the landmark, so their exposure is undefined. Excluded and
    # counted rather than assumed untreated.
    df = led.exclude(df, df["chemo_undated"] == 0,
                     "received chemotherapy but start date missing "
                     "(exposure undefined at the landmark)")
    # Implausible negative start dates.
    df = led.exclude(df, ~((start.reindex(df.index).notna()) & (start.reindex(df.index) < 0)),
                     "chemotherapy start date before diagnosis (implausible)")

    df["time"] = df["OS.time"] - landmark
    df["event"] = df["OS"].astype(int)
    led.check()

    if res is not None and landmark == PRIMARY_LANDMARK:
        led.frame().to_csv(OUTPUT / "05_cohort_ledger.csv", index=False)
        res.put("landmark_ledger_md", led.to_markdown())
    print(f"\n{led.to_markdown()}")
    return df.set_index("patient_id")


def main() -> None:
    banner("05 — Day-90 landmark and analysis cohort")
    res = Results()

    for lm in LANDMARKS:
        df = build(lm, res if lm == PRIMARY_LANDMARK else None)
        df.to_csv(OUTPUT / f"05_cohort_landmark{lm}.csv")

        n = len(df)
        print(f"\n  Day-{lm} cohort: {n:,} patients, {int(df['event'].sum()):,} deaths")
        print(f"    treated at landmark (A_chemo=1) : {int(df['A_chemo'].sum()):,}")
        print(f"    chemotherapy started later      : "
              f"{int(df['chemo_after_landmark'].sum()):,} "
              f"(untreated at the landmark, by design)")

        rows = []
        for name in ("TNBC", "HR_pos_HER2_neg", "HER2_pos"):
            s = df[df[name].fillna(False).astype(bool)]
            for tau in (1095, 1825):
                ev = int(((s["event"] == 1) & (s["time"] <= tau)).sum())
                rows.append(dict(
                    landmark=lm, subgroup=name, n=len(s), tau=tau, events=ev,
                    n_treated=int(s["A_chemo"].sum()),
                    n_control=int((s["A_chemo"] == 0).sum()),
                    events_treated=int(((s["event"] == 1) & (s["time"] <= tau)
                                        & (s["A_chemo"] == 1)).sum()),
                    events_control=int(((s["event"] == 1) & (s["time"] <= tau)
                                        & (s["A_chemo"] == 0)).sum()),
                ))
            s2 = df
            rows.append(dict(landmark=lm, subgroup="ALL", n=len(s2), tau=1095,
                             events=int(((s2["event"] == 1) & (s2["time"] <= 1095)).sum()),
                             n_treated=int(s2["A_chemo"].sum()),
                             n_control=int((s2["A_chemo"] == 0).sum()),
                             events_treated=int(((s2["event"] == 1) & (s2["time"] <= 1095)
                                                 & (s2["A_chemo"] == 1)).sum()),
                             events_control=int(((s2["event"] == 1) & (s2["time"] <= 1095)
                                                 & (s2["A_chemo"] == 0)).sum())))
        ec = pd.DataFrame(rows).drop_duplicates()
        print("\n  event counts by subgroup and horizon:")
        print(ec.to_string(index=False))
        if lm == PRIMARY_LANDMARK:
            ec.to_csv(OUTPUT / "05_event_counts.csv", index=False)
            res.update({
                "landmark": lm,
                "landmark_cohort_n": n,
                "landmark_cohort_events": int(df["event"].sum()),
                "landmark_treated": int(df["A_chemo"].sum()),
                "landmark_started_later": int(df["chemo_after_landmark"].sum()),
                "event_counts": ec.to_dict(orient="records"),
            })

    print("\n  Grade is unavailable: diagnoses.tumor_grade is null for every "
          "patient in clinical.tsv.")
    print("  Tumour size is unavailable: clinical.tsv carries no size or "
          "dimension field. T stage is used as the closest available proxy.")


if __name__ == "__main__":
    main()
