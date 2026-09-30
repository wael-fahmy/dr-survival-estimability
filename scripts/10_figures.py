"""
10_figures.py — export every figure's underlying data, then draw fallbacks.

Why the split. The figures were originally drawn by lib_plot.py, a hand-written
PDF backend built because matplotlib is unavailable in the analysis sandbox.
That backend has no font metrics: it cannot measure a string, so it truncates
labels at a fixed character count and places annotations at fixed offsets. Every
figure ended up with clipped text or collisions.

The fix is to stop treating the sandbox as the place figures are finished. This
script writes each figure's data to output/figdata/*.csv, and
revision/figures/make_figures.py redraws all six with matplotlib wherever the
manuscript is actually built, where font metrics exist. The lib_plot versions
are still emitted so the document compiles in the sandbox, but they are drafts;
make_figures.py overwrites them with the publication versions.

Produces:
    output/figdata/*.csv  ->  the inputs to every figure, drawn by stage 17
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from lib_io import OUTPUT, PAPER, Results, banner, barcode12  # noqa: E402
from lib_est import (add_intercept, adaptive_ridge, drop_constant,  # noqa: E402
                     km_estimate, logit_fit, logit_predict, smd, standardize)
from lib_plot import Axes, Canvas, PALETTE, nice_ticks  # noqa: E402

FIGS = PAPER / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
FIGDATA = OUTPUT / "figdata"
FIGDATA.mkdir(parents=True, exist_ok=True)
P = PALETTE

# Covariate display names. Figure 4 previously sliced names to 26 characters,
# which produced "CLIN_age_at_index.demograp". An explicit lookup is the only
# way to abbreviate these correctly.
COVAR_LABELS = {
    "CLIN_age_at_index.demographic": "Age at diagnosis",
    "CLIN_ajcc_pathologic_stage.diagnoses_Stage IV": "AJCC stage IV",
    "CLIN_ajcc_pathologic_stage.diagnoses_Stage III": "AJCC stage III",
    "CLIN_ajcc_pathologic_m.diagnoses_M1": "M1",
    "CLIN_ajcc_pathologic_n.diagnoses_N1b": "N1b",
    "CLIN_ajcc_pathologic_n.diagnoses_N0 (i-)": "N0(i-)",
    "CLIN_ajcc_pathologic_n.diagnoses_NX": "NX",
    "CLIN_ajcc_pathologic_t.diagnoses_T4b": "T4b",
    "CLIN_ajcc_staging_system_edition.diagnoses_4th": "AJCC 4th edition",
    "CLIN_ajcc_staging_system_edition.diagnoses_5th": "AJCC 5th edition",
    "CLIN_ajcc_staging_system_edition.diagnoses_6th": "AJCC 6th edition",
    "CLIN_tissue_or_organ_of_origin.diagnoses_Breast, NOS": "Site: breast NOS",
    "CLIN_tissue_or_organ_of_origin.diagnoses_Lower-inner quadrant of breast":
        "Site: lower-inner quadrant",
    "PROT_4EBP1": "Protein 4E-BP1",
    "PROT_ZAP-70": "Protein ZAP-70",
    "METH_cg00101629": "Methylation cg00101629",
    "METH_cg19851563": "Methylation cg19851563",
    "RNA_ENSG00000264589.4": "RNA ENSG00000264589",
    "pathology_details.lymph_nodes_positive": "Lymph nodes positive",
    "pathology_details.lymph_nodes_tested": "Lymph nodes examined",
    "pathology_details.consistent_pathology_review": "Consistent path. review",
    "ER_status": "ER status", "PR_status": "PR status",
    "HER2_status": "HER2 status",
    "age": "Age", "age_missing": "Age missing",
    "stage": "Stage", "stage_missing": "Stage missing",
    "t_stage": "T category", "t_stage_missing": "T missing",
    "n_stage": "N category", "n_stage_missing": "N missing",
    "m_stage": "M category", "m_stage_missing": "M missing",
    "year_dx": "Year of diagnosis", "year_dx_missing": "Year missing",
}


def label_for(name: str) -> str:
    if name in COVAR_LABELS:
        return COVAR_LABELS[name]
    if name.startswith("site_"):
        return f"Site {name[5:]}"
    return name.replace("_", " ")


def save(df: pd.DataFrame, name: str) -> None:
    df.to_csv(FIGDATA / name, index=False)


# ── Figure data exports ───────────────────────────────────────────────────────

def data_cohort_flow():
    led = pd.read_csv(OUTPUT / "05_cohort_ledger.csv")
    tn = pd.read_csv(OUTPUT / "02_tnbc_ledger.csv")
    led["panel"] = "Analysis cohort (day-90 landmark)"
    tn["panel"] = "TNBC, directly measured receptors"
    out = pd.concat([led, tn], ignore_index=True)
    save(out, "fig1_cohort_flow.csv")
    return led, tn


def data_censoring():
    surv = pd.read_csv(OUTPUT / "03_survival.csv", index_col=0)
    res = Results()
    t, s = km_estimate(surv["OS.time"].to_numpy(), surv["OS"].to_numpy())
    tc, sc = km_estimate(surv["OS.time"].to_numpy(), 1 - surv["OS"].to_numpy())
    save(pd.DataFrame({"time": t, "survival": s}), "fig2_km_overall.csv")
    save(pd.DataFrame({"time": tc, "still_followed": sc}), "fig2_km_followup.csv")
    cats = pd.DataFrame([
        dict(category="observed alive at 5 y",
             n=res.get("binary5y_observed_alive"), evidence="observed"),
        dict(category="died within 5 y",
             n=res.get("binary5y_died_within"), evidence="observed"),
        # A patient who died after day 1825 was observed alive at day 1825,
        # so coding them a five-year survivor is supported by the data. Only
        # patients censored before the horizon have unknown status.
        dict(category="died after 5 y, coded survivor (observed alive at 5 y)",
             n=res.get("binary5y_died_after_coded_survivor"),
             evidence="observed"),
        dict(category="censored before 5 y, coded survivor",
             n=res.get("binary5y_censored_before_coded_survivor"),
             evidence="assigned without evidence"),
    ])
    save(cats, "fig2_classification.csv")
    return (t, s), (tc, sc), cats


def data_forest():
    e = pd.read_csv(OUTPUT / "07_estimates.csv")
    e = e[e["tau"] == 1095].copy()
    labels = {"TNBC_chemo": "Chemotherapy in TNBC",
              "ALL_chemo": "Chemotherapy, whole cohort"}
    e["arm_label"] = e["arm"].map(labels)
    save(e[["arm", "arm_label", "adjustment", "estimator", "n", "n_treated",
            "n_control", "events", "estimate_days", "ci_lo", "ci_hi"]],
         "fig3_forest.csv")
    return e, labels


def data_overlap():
    df = pd.read_csv(OUTPUT / "05_cohort_landmark90.csv", index_col=0)
    df.index = barcode12(df.index)
    d = df[df["TNBC"].fillna(False).astype(bool)]
    A = d["A_chemo"].to_numpy(int)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from importlib import import_module
    est = import_module("07_estimate")
    X = pd.concat([est.build_existing24(df).loc[d.index]
                   .drop(columns=["ER_status", "PR_status", "HER2_status"],
                         errors="ignore"),
                   est.build_clinical(df).loc[d.index]], axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    Xv = np.nan_to_num(X.to_numpy(float), nan=0.0)
    Xc, keep = drop_constant(Xv)
    Xs, _, _ = standardize(Xc)
    D = add_intercept(Xs)
    ps = np.clip(logit_predict(D, logit_fit(D, A, ridge=adaptive_ridge(len(A), D.shape[1]))),
                 0.02, 0.98)
    p1 = A.mean()
    w = np.where(A == 1, p1 / ps, (1 - p1) / (1 - ps))
    s_un, s_w = smd(Xs, A), smd(Xs, A, w)
    names = [label_for(c) for c in X.columns[keep]]

    save(pd.DataFrame({"propensity": ps, "treated": A}), "fig4_propensity.csv")
    save(pd.DataFrame({"covariate": names,
                       "smd_unweighted": np.abs(s_un),
                       "smd_weighted": np.abs(s_w)})
         .sort_values("smd_unweighted", ascending=False),
         "fig4_balance.csv")
    return ps, A, names, s_un, s_w


def data_ladder():
    L = pd.read_csv(OUTPUT / "08_ladder.csv")
    save(L, "fig5_ladder.csv")
    return L


def data_site():
    g = pd.read_csv(OUTPUT / "06_site_summary.csv")
    save(g, "fig6_site.csv")
    return g[g["n"] >= 20].sort_values("chemo_rate")


# ── Draft renderings (superseded by figures/make_figures.py) ─────────────────

def draft_cohort_flow(led, tn):
    cv = Canvas(468, 320)
    cv.text(0, 308, "Cohort construction", size=10, bold=True)
    for col, (d, title) in enumerate(((led, "Analysis cohort (day-90 landmark)"),
                                      (tn, "TNBC, directly measured receptors"))):
        x0 = 22 + col * 236
        cv.text(x0, 288, title, size=8, bold=True)
        rows = d.to_dict("records")
        top = float(rows[0]["n_after"])
        y = 268
        for r in rows:
            n = float(r["n_after"])
            w = 130 * n / top
            cv.rect(x0, y - 10, w, 10, fill=P["light"], stroke=P["blue"], lw=0.5)
            cv.text(x0 + w + 4, y - 7, f"{int(n):,}", size=6.5, bold=True)
            for k, line in enumerate(_wrap(str(r["rule"]), 46)[:2]):
                cv.text(x0, y + 3 + 7 * (1 - k), line, size=5.4,
                        color=(0.3, 0.3, 0.35))
            if not pd.isna(r["n_dropped"]) and r["n_dropped"] > 0:
                cv.text(x0 + w + 34, y - 7, f"-{int(r['n_dropped']):,}",
                        size=6.5, color=P["red"])
            y -= 32
    cv.text(22, 12, "Draft rendering. Publication version: figures/make_figures.py",
            size=5.6, color=(0.55, 0.55, 0.6))
    return cv.save(FIGS / "fig1_cohort_flow.pdf")


def _wrap(text, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= width:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def draft_censoring(km, fu, cats):
    t, s = km
    tc, sc = fu
    cv = Canvas(468, 250)
    cv.text(0, 238, "Why a binary five-year outcome fails on this cohort",
            size=10, bold=True)
    ax = Axes(cv, 48, 56, 195, 150, (0, 3000), (0, 1),
              xlabel="days from diagnosis", ylabel="probability")
    ax.frame(xticks=[0, 1000, 1825, 3000], yticks=[0, .25, .5, .75, 1], yfmt="{:.2f}")
    ax.step(t, s, color=P["blue"], w=1.4)
    ax.step(tc, sc, color=P["orange"], w=1.4, dash="3 2")
    cv.line(ax.px(1825), ax.y, ax.px(1825), ax.y + ax.h, w=0.8, color=P["red"],
            dash="2 2")
    cv.text(58, 196, "overall survival", size=6.5, color=P["blue"])
    cv.text(58, 187, "still under follow-up", size=6.5, color=P["orange"])
    tot = cats["n"].sum()
    y = 196
    cv.text(285, 212, "How each patient is classified", size=8, bold=True)
    cols = [P["green"], P["red"], P["orange"], P["grey"]]
    for (_, r), c in zip(cats.iterrows(), cols):
        cv.rect(285, y, 110 * r["n"] / tot, 10, fill=c)
        for k, line in enumerate(_wrap(f"{r['category']} ({r['n']:,})", 34)[:2]):
            cv.text(285, y + 13 + 7 * (1 - k), line, size=5.8,
                    color=(0.3, 0.3, 0.35))
        y -= 32
    cv.text(285, 20, "Draft rendering. See figures/make_figures.py",
            size=5.6, color=(0.55, 0.55, 0.6))
    return cv.save(FIGS / "fig2_censoring.pdf")


def draft_forest(e, labels):
    order = [a for a in ("TNBC_chemo", "ALL_chemo") if (e["arm"] == a).any()]
    slots = sum(1 + len(e[e["arm"] == a]) + 1 for a in order)
    cv = Canvas(468, 340)
    cv.text(0, 328, "RMST difference at 1095 days after the day-90 landmark",
            size=10, bold=True)
    lo = min(e["ci_lo"].min(), -60) * 1.08
    hi = max(e["ci_hi"].max(), 60) * 1.08
    ax = Axes(cv, 190, 56, 245, 240, (lo, hi), (-0.5, slots - 0.5),
              xlabel="difference in restricted mean survival (days)")
    ax.frame(xticks=nice_ticks(lo, hi, 5), yticks=None, grid=False)
    cv.line(ax.px(0), ax.y, ax.px(0), ax.y + ax.h, w=0.9,
            color=(0.55, 0.55, 0.6), dash="2 2")
    row = slots - 1
    for arm in order:
        s = e[e["arm"] == arm]
        cv.text(8, ax.py(row) - 2, labels[arm], size=7.5, bold=True)
        cv.text(8, ax.py(row) - 10,
                f"n={int(s['n'].iloc[0]):,}, {int(s['events'].iloc[0])} events",
                size=6, color=(0.45, 0.45, 0.5))
        row -= 1
        for _, r in s.iterrows():
            col = P["blue"] if r["estimator"].startswith("AIPW") else P["orange"]
            ax.errbar_h(row, r["ci_lo"], r["ci_hi"], r["estimate_days"], color=col)
            cv.text(184, ax.py(row) - 2.2,
                    f"{r['adjustment']}  {r['estimator'].split()[0]}",
                    size=5.8, anchor="end", color=(0.35, 0.35, 0.4))
            row -= 1
        row -= 1
    cv.circle(200, 320, 2.2, fill=P["blue"])
    cv.text(206, 318, "AIPW", size=6.5, color=P["blue"])
    cv.circle(255, 320, 2.2, fill=P["orange"])
    cv.text(261, 318, "IPCW-weighted Kaplan-Meier", size=6.5, color=P["orange"])
    cv.text(0, 10, "Draft rendering. See figures/make_figures.py", size=5.6,
            color=(0.55, 0.55, 0.6))
    return cv.save(FIGS / "fig3_effects_forest.pdf")


def draft_overlap(ps, A, names, s_un, s_w):
    cv = Canvas(468, 260)
    cv.text(0, 248, "Overlap and covariate balance, chemotherapy in TNBC",
            size=10, bold=True)
    ax = Axes(cv, 44, 66, 165, 145, (0, 1), (0, 1),
              xlabel="estimated propensity score", ylabel="density (scaled)")
    bins = np.linspace(0, 1, 16)
    h1, _ = np.histogram(ps[A == 1], bins=bins)
    h0, _ = np.histogram(ps[A == 0], bins=bins)
    mx = max(h1.max(), h0.max()) or 1
    ax.frame(xticks=[0, .25, .5, .75, 1], yticks=[0, .5, 1], yfmt="{:.1f}")
    bw = bins[1] - bins[0]
    for i in range(len(bins) - 1):
        xa = ax.px(bins[i])
        wpx = ax.px(bins[i] + bw) - xa
        cv.rect(xa, ax.y, wpx / 2, ax.h * h1[i] / mx, fill=P["blue"])
        cv.rect(xa + wpx / 2, ax.y, wpx / 2, ax.h * h0[i] / mx, fill=P["orange"])
    # legend above the plot, never near the axis title
    cv.text(52, 220, "treated", size=6.5, color=P["blue"])
    cv.text(95, 220, "control", size=6.5, color=P["orange"])

    order = np.argsort(-np.abs(s_un))[:12]
    ax2 = Axes(cv, 320, 66, 110, 145,
               (0, max(0.6, float(np.abs(s_un).max()) * 1.1)), (0, 13),
               xlabel="|standardized mean difference|")
    ax2.frame(xticks=[0, .2, .4, .6], yticks=None, grid=False, xfmt="{:.1f}")
    cv.line(ax2.px(0.1), ax2.y, ax2.px(0.1), ax2.y + ax2.h, w=0.8,
            color=P["red"], dash="2 2")
    for k, j in enumerate(order):
        yy = 12 - k
        cv.circle(ax2.px(abs(s_un[j])), ax2.py(yy), 1.8, fill=P["grey"])
        cv.circle(ax2.px(abs(s_w[j])), ax2.py(yy), 1.8, fill=P["blue"])
        cv.text(316, ax2.py(yy) - 2, names[j][:24], size=5.2, anchor="end",
                color=(0.4, 0.4, 0.45))
    cv.text(320, 220, "before", size=6.2, color=P["grey"])
    cv.text(360, 220, "after weighting", size=6.2, color=P["blue"])
    cv.text(0, 10, "Draft rendering. See figures/make_figures.py", size=5.6,
            color=(0.55, 0.55, 0.6))
    return cv.save(FIGS / "fig4_overlap_balance.pdf")


def draft_ladder(L):
    cv = Canvas(468, 280)
    cv.text(0, 268, "Effect of each design correction, chemotherapy in TNBC",
            size=10, bold=True)
    rd = L[L["scale"].str.startswith("risk")]
    rm = L[L["scale"].str.startswith("RMST")]
    for panel, (sub, xlab, base_y, col) in enumerate((
            (rd, "risk difference, 5-y mortality (pp)", 168, P["blue"]),
            (rm, "RMST difference at 1095 d (days)", 46, P["green"]))):
        lo = min(sub["ci_lo"].min(), -1) * 1.1
        hi = max(sub["ci_hi"].max(), 1) * 1.1
        ax = Axes(cv, 210, base_y, 120, 62, (lo, hi), (-0.5, 2.5), xlabel=xlab)
        ax.frame(xticks=nice_ticks(lo, hi, 4), yticks=None, grid=False)
        cv.line(ax.px(0), ax.y, ax.px(0), ax.y + ax.h, w=0.9,
                color=(0.55, 0.55, 0.6), dash="2 2")
        for _, r in sub.iterrows():
            yy = 2 - (r["rung"] - (0 if panel == 0 else 3))
            ax.errbar_h(yy, r["ci_lo"], r["ci_hi"], r["estimate"],
                        color=P["red"] if r["rung"] == 0 else col)
            for k, line in enumerate(_wrap(f"{int(r['rung'])}. {r['label']}", 34)[:2]):
                cv.text(206, ax.py(yy) - 2 + 6 * (1 - k) - 3, line, size=5.4,
                        anchor="end", color=(0.3, 0.3, 0.35))
    cv.text(0, 10, "Draft rendering. See figures/make_figures.py", size=5.6,
            color=(0.55, 0.55, 0.6))
    return cv.save(FIGS / "fig6_correction_ladder.pdf")


def draft_site(g):
    res = Results()
    cv = Canvas(468, 250)
    cv.text(0, 238, "Contributing site is associated with treatment and survival",
            size=10, bold=True)
    ax = Axes(cv, 44, 56, 165, 148, (0, 0.8), (0, 0.6),
              xlabel="chemotherapy rate at the landmark",
              ylabel="proportion who died")
    ax.frame(xticks=[0, .2, .4, .6, .8], yticks=[0, .2, .4, .6],
             yfmt="{:.1f}", xfmt="{:.1f}")
    for _, r in g.iterrows():
        rad = 1.4 + 3.2 * (r["n"] / g["n"].max())
        cv.circle(ax.px(min(r["chemo_rate"], .8)), ax.py(min(r["death_rate"], .6)),
                  rad, fill=P["blue"], stroke=(1, 1, 1), lw=0.4)
    mx = float(g["median_fu"].max())
    ax2 = Axes(cv, 300, 56, 135, 148, (0, mx * 1.05), (-0.5, len(g) - 0.5),
               xlabel="median follow-up (days)")
    ax2.frame(xticks=nice_ticks(0, mx, 4), yticks=None, grid=False)
    for i, (_, r) in enumerate(g.sort_values("median_fu").iterrows()):
        cv.rect(ax2.x, ax2.py(i) - 3.0, ax2.px(r["median_fu"]) - ax2.x, 6.0,
                fill=P["light"], stroke=P["blue"], lw=0.4)
        cv.text(ax2.x - 4, ax2.py(i) - 2, str(r["site"]), size=5.6, anchor="end",
                color=(0.35, 0.35, 0.4))
    cv.text(0, 10, "Draft rendering. See figures/make_figures.py", size=5.6,
            color=(0.55, 0.55, 0.6))
    return cv.save(FIGS / "fig5_site_confounding.pdf")


def main() -> None:
    banner("10 — Figure data export and draft rendering")

    led, tn = data_cohort_flow()
    km, fu, cats = data_censoring()
    e, labels = data_forest()
    ps, A, names, s_un, s_w = data_overlap()
    L = data_ladder()
    g = data_site()

    files = sorted(p.name for p in FIGDATA.glob("*.csv"))
    print(f"  wrote {len(files)} data files to output/figdata/")
    for f in files:
        print(f"    {f}")

    made = [draft_cohort_flow(led, tn), draft_censoring(km, fu, cats),
            draft_forest(e, labels), draft_overlap(ps, A, names, s_un, s_w),
            draft_ladder(L), draft_site(g)]
    print(f"\n  wrote {len(made)} draft PDFs to paper/figures/")
    print("  These are drafts. Run figures/make_figures.py where matplotlib is")
    print("  available to replace them with the publication versions.")

    Results().update({"figdata_files": files,
                      "figures": [p.name for p in made]})


if __name__ == "__main__":
    main()
