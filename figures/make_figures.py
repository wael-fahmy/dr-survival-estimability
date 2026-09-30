"""
make_figures.py — build the six publication figures with matplotlib.

Run this wherever matplotlib is installed. It reads only
../output/figdata/*.csv, which run_all.py produces, and writes PDFs into
../paper/figures/, overwriting the draft versions.

    cd revision/figures && python make_figures.py

Why this exists. The analysis sandbox has no matplotlib, so the pipeline draws
its figures with a hand-written PDF backend that has no font metrics. It cannot
measure a string, so it truncates labels and places annotations at fixed
offsets, which produced clipped text and collisions in every figure. Rather than
patch a backend that cannot know how wide its own text is, the pipeline exports
the data and this script draws it properly.

Requires: matplotlib >= 3.5, pandas, numpy. No other dependency.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
except ImportError:
    sys.exit("matplotlib is required. pip install matplotlib, then re-run.")

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "output" / "figdata"
OUT = HERE.parent / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

if not DATA.exists():
    sys.exit(f"{DATA} not found. Run run_all.py first.")

BLUE, ORANGE, RED = "#335C99", "#D9852E", "#B83D3D"
GREEN, GREY, LIGHT = "#408C61", "#73737F", "#CCD6E5"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,          # embed TrueType so the PDF is text-searchable
})


def wrap(s, n):
    return "\n".join(textwrap.wrap(str(s), n))


def fig1_cohort_flow():
    d = pd.read_csv(DATA / "fig1_cohort_flow.csv")
    panels = list(dict.fromkeys(d["panel"]))
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.6))
    for ax, panel in zip(axes, panels):
        s = d[d["panel"] == panel].reset_index(drop=True)
        top = s["n_after"].iloc[0]
        y = np.arange(len(s))[::-1]
        ax.barh(y, s["n_after"] / top, height=0.42, color=LIGHT,
                edgecolor=BLUE, linewidth=0.7)
        for yi, (_, r) in zip(y, s.iterrows()):
            ax.text(r["n_after"] / top + 0.02, yi, f"{int(r['n_after']):,}",
                    va="center", fontsize=7, fontweight="bold")
            if pd.notna(r["n_dropped"]) and r["n_dropped"] > 0:
                ax.text(r["n_after"] / top + 0.20, yi,
                        f"−{int(r['n_dropped']):,}", va="center",
                        fontsize=7, color=RED)
        ax.set_yticks(y)
        ax.set_yticklabels([wrap(r, 44) for r in s["rule"]], fontsize=6)
        ax.set_xlim(0, 1.45)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_title(panel, fontsize=8.5, loc="left", fontweight="bold")
    fig.suptitle("Cohort construction", fontsize=10, fontweight="bold",
                 x=0.02, ha="left")
    fig.text(0.02, 0.005, "Bar length is proportional to cohort size. Every "
             "exclusion is counted against a named rule.", fontsize=6.2,
             color="#666")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(OUT / "fig1_cohort_flow.pdf")
    plt.close(fig)


def fig2_censoring():
    km = pd.read_csv(DATA / "fig2_km_overall.csv")
    fu = pd.read_csv(DATA / "fig2_km_followup.csv")
    cats = pd.read_csv(DATA / "fig2_classification.csv")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.5, 3.3),
                                 gridspec_kw={"width_ratios": [1.15, 1]})
    a1.step(km["time"], km["survival"], where="post", color=BLUE, lw=1.4,
            label="overall survival")
    a1.step(fu["time"], fu["still_followed"], where="post", color=ORANGE,
            lw=1.4, ls="--", label="still under follow-up")
    a1.axvline(1825, color=RED, ls=":", lw=1.0)
    a1.text(1825, 1.02, "5 y", color=RED, fontsize=6.5, ha="center")
    a1.set_xlim(0, 3000)
    a1.set_ylim(0, 1.05)
    a1.set_xlabel("days from diagnosis")
    a1.set_ylabel("probability")
    a1.legend(loc="lower left", frameon=False)
    a1.set_title("Survival and follow-up", loc="left")

    tot = cats["n"].sum()
    colours = {"observed": GREEN, "assigned without evidence": GREY}
    y = np.arange(len(cats))[::-1]
    a2.barh(y, cats["n"] / tot, height=0.5,
            color=[colours[e] for e in cats["evidence"]])
    for yi, (_, r) in zip(y, cats.iterrows()):
        a2.text(r["n"] / tot + 0.012, yi, f"{int(r['n']):,} "
                f"({r['n'] / tot * 100:.1f}%)", va="center", fontsize=6.8)
    a2.set_yticks(y)
    a2.set_yticklabels([wrap(c, 34) for c in cats["category"]], fontsize=6.2)
    a2.set_xlim(0, 0.95)
    a2.set_xticks([])
    a2.spines["bottom"].set_visible(False)
    a2.spines["left"].set_visible(False)
    a2.set_title("How each patient is classified", loc="left")
    share = cats.loc[cats["evidence"] == "assigned without evidence", "n"].sum() / tot
    a2.text(0, -0.9, f"{share * 100:.1f}% of patients are assigned an outcome "
            f"without evidence.", fontsize=6.8, color=RED, fontweight="bold")
    fig.suptitle("Why a binary five-year outcome fails on this cohort",
                 fontsize=10, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(OUT / "fig2_censoring.pdf")
    plt.close(fig)


def fig3_forest():
    e = pd.read_csv(DATA / "fig3_forest.csv")
    arms = list(dict.fromkeys(e["arm"]))
    rows, labels, colours = [], [], []
    for arm in arms:
        s = e[e["arm"] == arm]
        rows.append(None)
        labels.append(f"{s['arm_label'].iloc[0]}  "
                      f"(n={int(s['n'].iloc[0]):,}, "
                      f"{int(s['events'].iloc[0])} events)")
        colours.append(None)
        for _, r in s.iterrows():
            rows.append(r)
            labels.append(f"    {r['adjustment']}, "
                          f"{'AIPW' if r['estimator'].startswith('AIPW') else 'IPCW-KM'}")
            colours.append(BLUE if r["estimator"].startswith("AIPW") else ORANGE)

    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.28 * len(rows) + 1.5))
    for yi, r, c in zip(y, rows, colours):
        if r is None:
            continue
        ax.plot([r["ci_lo"], r["ci_hi"]], [yi, yi], color=c, lw=1.2)
        ax.plot([r["ci_lo"], r["ci_lo"]], [yi - .18, yi + .18], color=c, lw=1.2)
        ax.plot([r["ci_hi"], r["ci_hi"]], [yi - .18, yi + .18], color=c, lw=1.2)
        ax.plot(r["estimate_days"], yi, "o", color=c, ms=4)
    ax.axvline(0, color="#8C8C99", ls="--", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    for tick, r in zip(ax.get_yticklabels(), rows):
        if r is None:
            tick.set_fontweight("bold")
    ax.set_xlabel("difference in restricted mean survival (days)")
    ax.set_title("RMST difference at 1095 days after the day-90 landmark",
                 fontsize=10, fontweight="bold", loc="left")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=BLUE, marker="o", ls="-",
                              label="AIPW (cross-fitted)"),
                       Line2D([], [], color=ORANGE, marker="o", ls="-",
                              label="IPCW-weighted Kaplan-Meier")],
              loc="lower right", frameon=False)
    fig.text(0.01, 0.005, "Targeted therapy in HER2-positive disease is absent: "
             "only six patients had started it by the landmark, so the arm is "
             "not estimable.", fontsize=6.2, color=RED)
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    fig.savefig(OUT / "fig3_effects_forest.pdf")
    plt.close(fig)


def fig4_overlap():
    ps = pd.read_csv(DATA / "fig4_propensity.csv")
    bal = pd.read_csv(DATA / "fig4_balance.csv").head(14)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.5, 3.6),
                                 gridspec_kw={"width_ratios": [1, 1.1]})
    bins = np.linspace(0, 1, 16)
    a1.hist([ps.loc[ps["treated"] == 1, "propensity"],
             ps.loc[ps["treated"] == 0, "propensity"]], bins=bins,
            color=[BLUE, ORANGE], label=["treated", "control"])
    a1.set_xlabel("estimated propensity score")
    a1.set_ylabel("patients")
    a1.legend(frameon=False, loc="upper left")
    a1.set_title("Propensity distribution by arm", loc="left")

    y = np.arange(len(bal))[::-1]
    a2.scatter(bal["smd_unweighted"], y, s=18, color=GREY, label="before weighting")
    a2.scatter(bal["smd_weighted"], y, s=18, color=BLUE, label="after weighting")
    a2.axvline(0.1, color=RED, ls=":", lw=1.0)
    a2.set_yticks(y)
    a2.set_yticklabels(bal["covariate"], fontsize=6.2)
    a2.set_xlabel("|standardized mean difference|")
    a2.legend(frameon=False, loc="lower right")
    a2.set_title("Balance, 14 worst-balanced covariates", loc="left")
    fig.suptitle("Overlap and covariate balance, chemotherapy in TNBC",
                 fontsize=10, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "fig4_overlap_balance.pdf")
    plt.close(fig)


def fig5_ladder():
    L = pd.read_csv(DATA / "fig5_ladder.csv")
    rd = L[L["scale"].str.startswith("risk")].copy()
    # Lower panel: every rung on one scale, the RMST difference at 1095 days,
    # so that the movement produced by each correction is comparable in days.
    rm = L.copy()
    rm["estimate"], rm["ci_lo"], rm["ci_hi"] = (rm["rmst_estimate"],
                                                rm["rmst_ci_lo"], rm["rmst_ci_hi"])
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.0),
                             gridspec_kw={"height_ratios": [3, 6]})
    for ax, sub, xlab, col in ((axes[0], rd,
                                "risk difference in 5-year mortality "
                                "(percentage points), rungs 0 to 2 only", BLUE),
                               (axes[1], rm,
                                "RMST difference at 1095 days (days), "
                                "every rung on one scale", GREEN)):
        y = np.arange(len(sub))[::-1]
        for yi, (_, r) in zip(y, sub.iterrows()):
            c = RED if r["rung"] == 0 else col
            ax.plot([r["ci_lo"], r["ci_hi"]], [yi, yi], color=c, lw=1.2)
            ax.plot([r["ci_lo"], r["ci_lo"]], [yi - .15, yi + .15], color=c, lw=1.2)
            ax.plot([r["ci_hi"], r["ci_hi"]], [yi - .15, yi + .15], color=c, lw=1.2)
            ax.plot(r["estimate"], yi, "o", color=c, ms=4)
        ax.axvline(0, color="#8C8C99", ls="--", lw=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels([wrap(f"{int(r['rung'])}. {r['label']}", 46)
                            for _, r in sub.iterrows()], fontsize=6.4)
        ax.set_xlabel(xlab)
    axes[0].set_title("Effect of each design correction, chemotherapy in TNBC",
                      fontsize=10, fontweight="bold", loc="left")
    fig.text(0.01, 0.005, "Upper panel: the original analysis's scale. Lower "
             "panel: the same rungs as RMST differences, comparable in days "
             "throughout; rung 3 equals rung 2 on this scale because it changes "
             "only the scale. Rung 0 is the original analysis.",
             fontsize=6.2, color="#666")
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    fig.savefig(OUT / "fig6_correction_ladder.pdf")
    plt.close(fig)


def fig6_site():
    g = pd.read_csv(DATA / "fig6_site.csv")
    g = g[g["n"] >= 20]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.5, 3.5))
    a1.scatter(g["chemo_rate"], g["death_rate"],
               s=12 + 120 * g["n"] / g["n"].max(), color=BLUE, alpha=0.75,
               edgecolor="white", linewidth=0.6)
    for _, r in g.iterrows():
        if r["death_rate"] > 0.25 or r["chemo_rate"] > 0.6:
            a1.annotate(r["site"], (r["chemo_rate"], r["death_rate"]),
                        textcoords="offset points", xytext=(6, 2), fontsize=6.2)
    a1.set_xlabel("chemotherapy rate at the landmark")
    a1.set_ylabel("proportion who died")
    a1.set_title("Each point is one site\n(marker area is site size)", loc="left")

    s = g.sort_values("median_fu")
    y = np.arange(len(s))
    a2.barh(y, s["median_fu"], height=0.6, color=LIGHT, edgecolor=BLUE, lw=0.6)
    a2.set_yticks(y)
    a2.set_yticklabels(s["site"], fontsize=6.2)
    a2.set_xlabel("median follow-up (days)")
    a2.set_title("Follow-up differs by an order of magnitude", loc="left")
    fig.suptitle("Contributing site is associated with treatment and survival",
                 fontsize=10, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "fig5_site_confounding.pdf")
    plt.close(fig)


def main():
    made = []
    for fn in (fig1_cohort_flow, fig2_censoring, fig3_forest, fig4_overlap,
               fig5_ladder, fig6_site):
        fn()
        made.append(fn.__name__)
        print(f"  {fn.__name__}")
    print(f"\nwrote {len(made)} figures to {OUT}")
    print("These replace the draft renderings produced inside the analysis "
          "sandbox.")


if __name__ == "__main__":
    main()
