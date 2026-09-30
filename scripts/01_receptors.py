"""
01_receptors.py — receptor status as four (plus one) distinct states.

The audited pipeline collapsed receptor status to a binary with
    int((x.str.lower() == 'positive').any())
which codes Equivocal, "Test Value Reported", "Copy Number Reported", Unknown
and never-tested all as *negative*, and silently resolves patients with both a
Positive and a Negative result across visits to Positive. It then MICE-imputed
the missing values and z-scaled them, so a variable that defines cohort
membership became a continuous imputed quantity.

This script keeps the states apart and imputes nothing:

    Positive | Negative | Equivocal | Conflicting | Not tested

`Conflicting` is a fifth state, not folded into any other: 18 patients have both
a Positive and a Negative ERBB2 result across follow-up visits (ESR1 8, PGR 5).
The audited code resolved all of them to Positive without recording it.

Source: TCGA-BRCA.clinical.tsv is probed first for IHC receptor fields, per the
revision brief. When it is absent the brief's stated fallback is used --
follow_up.tsv molecular tests. Which branch ran is recorded in the output.

Produces:
    01_receptor_status.csv         ->  cohort frame, consumed by stage 02
    01_receptor_coverage.csv       ->  Section 6.1 text
    01_repeat_and_conflict.csv     ->  Section 6.1 text
    01_molecular_test_results.csv  ->  Section 6.1 text
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from lib_io import DATA, OUTPUT, Results, banner, barcode12, require  # noqa: E402

GENE_MAP = {"ESR1": "ER", "PGR": "PR", "ERBB2": "HER2"}
STATES = ["Positive", "Negative", "Equivocal", "Conflicting", "Not tested"]


def resolve_state(results: set[str]) -> str:
    """Collapse one patient's results for one gene into a single state."""
    low = {str(r).strip().lower() for r in results}
    has_p = "positive" in low
    has_n = "negative" in low
    if has_p and has_n:
        return "Conflicting"
    if has_p:
        return "Positive"
    if has_n:
        return "Negative"
    if "equivocal" in low:
        return "Equivocal"
    return "Not tested"          # indeterminate assay output only


def probe_xena_ihc() -> Path | None:
    """Return TCGA-BRCA.clinical.tsv if present AND carrying IHC receptor fields."""
    cand = DATA / "TCGA-BRCA.clinical.tsv"
    if not cand.exists():
        return None
    head = pd.read_csv(cand, sep="\t", nrows=0)
    wanted = [c for c in head.columns
              if any(k in c.lower() for k in
                     ("er_status", "pr_status", "her2_status",
                      "estrogen_receptor", "progesterone_receptor",
                      "her2_immunohistochemistry"))]
    return cand if wanted else None


def main() -> None:
    banner("01 — Receptor status (four states, no imputation)")
    res = Results()

    xena = probe_xena_ihc()
    if xena is not None:
        source = "TCGA-BRCA.clinical.tsv (IHC fields present)"
        raise SystemExit(
            "TCGA-BRCA.clinical.tsv is present and carries IHC fields. The "
            "IHC branch is not implemented because that file was absent when "
            "this pipeline was written and its column names could not be "
            "verified. Inspect it and extend this script rather than letting "
            "the fallback run silently."
        )

    source = "follow_up.tsv molecular tests (fallback)"
    print(f"  TCGA-BRCA.clinical.tsv : ABSENT — using the brief's stated fallback")
    print(f"  Source                 : {source}")

    fu_path = require(DATA / "follow_up.tsv", "receptor status fallback")
    fu = pd.read_csv(fu_path, sep="\t", low_memory=False)
    fu["patient_id"] = barcode12(fu["cases.submitter_id"])

    gcol, rcol = "molecular_tests.gene_symbol", "molecular_tests.test_result"
    m = fu.loc[fu[gcol].isin(GENE_MAP), ["patient_id", gcol, rcol]].copy()
    m = m[m[rcol].notna()]

    print(f"\n  follow_up rows                : {len(fu):,}")
    print(f"  rows for ESR1 / PGR / ERBB2   : {len(m):,}")

    # Tabulate every distinct result value, and how many are neither P nor N.
    tab = (m.groupby([gcol, rcol]).size()
             .rename("n").reset_index()
             .sort_values("n", ascending=False))
    tab.to_csv(OUTPUT / "01_molecular_test_results.csv", index=False)
    print("\n  molecular_tests.test_result tabulation:")
    print(tab.to_string(index=False))

    non_pn = ~m[rcol].str.strip().str.lower().isin(["positive", "negative"])
    n_non_pn = int(non_pn.sum())
    print(f"\n  rows neither Positive nor Negative "
          f"(coded negative by the audited code): {n_non_pn:,}")
    print(m.loc[non_pn, rcol].value_counts().to_string())

    # Repeat testing and cross-visit conflicts, per gene.
    repeat_rows = []
    for gene, short in GENE_MAP.items():
        s = m[m[gcol] == gene]
        per = s.groupby("patient_id").size()
        pn = s[s[rcol].str.strip().str.lower().isin(["positive", "negative"])]
        conflict = pn.groupby("patient_id")[rcol].nunique()
        no_pn = set(s["patient_id"]) - set(pn["patient_id"])
        repeat_rows.append(dict(
            gene=gene, receptor=short,
            patients=int(per.size), rows=int(len(s)),
            patients_multi_test=int((per > 1).sum()),
            max_tests_one_patient=int(per.max()),
            patients_conflicting=int((conflict > 1).sum()),
            patients_tested_no_PN_result=int(len(no_pn)),
        ))
    rep = pd.DataFrame(repeat_rows)
    rep.to_csv(OUTPUT / "01_repeat_and_conflict.csv", index=False)
    print("\n  repeat testing and cross-visit conflicts:")
    print(rep.to_string(index=False))

    # Build the five-state receptor table.
    st = (m.groupby(["patient_id", gcol])[rcol]
            .apply(lambda x: resolve_state(set(x)))
            .unstack())
    st.columns = [GENE_MAP[c] for c in st.columns]
    for c in ("ER", "PR", "HER2"):
        if c not in st.columns:
            st[c] = "Not tested"
    st = st[["ER", "PR", "HER2"]].fillna("Not tested")

    st.to_csv(OUTPUT / "01_receptor_status.csv")
    print(f"\n  patients with >=1 receptor test : {len(st):,}")
    print("\n  state coverage per receptor:")
    cov = pd.DataFrame({c: st[c].value_counts() for c in st.columns})
    cov = cov.reindex(STATES).fillna(0).astype(int)
    print(cov.to_string())
    cov.to_csv(OUTPUT / "01_receptor_coverage.csv")

    res.update({
        "receptor_source": source,
        "receptor_xena_clinical_present": xena is not None,
        "receptor_rows_3genes": int(len(m)),
        "receptor_rows_non_PN": n_non_pn,
        "receptor_non_PN_breakdown": m.loc[non_pn, rcol].value_counts().to_dict(),
        "receptor_patients_any_test": int(len(st)),
        "receptor_coverage": cov.to_dict(),
        "receptor_repeat_conflict": rep.to_dict(orient="records"),
    })
    print(f"\n  wrote output/01_receptor_status.csv  ({len(st):,} patients)")


if __name__ == "__main__":
    main()
