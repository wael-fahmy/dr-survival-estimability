"""Generate vector workflow and simulation figures from the recorded outputs."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output'
FIG=ROOT/'paper/figures'
plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'ps.fonttype':42})

def workflow():
    fig,ax=plt.subplots(figsize=(6.5,7))
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    boxes=[('1  Audit the inputs','Receptors, treatment fields, initiation dates,\nsurvival and censoring provenance'),
           ('2  Specify the landmark contrast','Day 90 eligibility and fixed initiation status;\n1095-day RMST; clinical adjustment'),
           ('3  Assess support','Events, complete outcomes, parameter burden,\nbalance, weights and follow-up'),
           ('4  Compare on the same RMST scale','Unadjusted KM, weighted KM, cross-fitted AIPW\nand standardized Cox outcome model'),
           ('5  Challenge assumptions and stability','Adjustment, censoring, horizon, stabilizers, folds;\nknown-truth simulation and stress scenarios'),
           ('6  Interpret with limitations','Separate design changes from estimator dependence;\nreport uncertainty and remaining identification limits')]
    for i,(title,detail) in enumerate(boxes):
        y=.93-i*.160
        ax.add_patch(FancyBboxPatch((.025,y-.105),.95,.13,boxstyle='round,pad=0.006',
                                  facecolor='#edf3f8',edgecolor='#345776',linewidth=1.1))
        ax.text(.5,y-.005,title,ha='center',va='center',weight='bold',fontsize=11)
        ax.text(.5,y-.062,detail,ha='center',va='center',fontsize=10.5,linespacing=1.4)
        if i<5:ax.annotate('',xy=(.5,y-.139),xytext=(.5,y-.112),
                           arrowprops=dict(arrowstyle='-|>',color='#345776',lw=1.2))
    fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.98)
    fig.savefig(FIG/'fig7_workflow.pdf');plt.close(fig)

def validation():
    g=pd.read_csv(OUT/'16_estimability_grid.csv')
    s=pd.read_csv(OUT/'21_stress_summary.csv')
    f=pd.read_csv(OUT/'16_estimability_fit.csv').iloc[0]
    fig,(ax,bx)=plt.subplots(2,1,figsize=(6.5,6.1),gridspec_kw={'height_ratios':[1,1.2]})
    colors={'AIPW':'#9b3e0c','IPCW-KM':'#006a82','Cox standardized RMST':'#725393'}
    labels={'AIPW':'AIPW','IPCW-KM':'Weighted KM','Cox standardized RMST':'Cox RMST'}
    a=g[g.estimator=='AIPW']
    ax.scatter(a.rho,a.sensitivity_ratio,s=18,color='#a6a6a6',label='Original AIPW cells')
    xx=np.geomspace(.08,10,100)
    ax.plot(xx,f.fit_intercept+f.fit_slope*np.log(xx),color='#555555',lw=1,label='Original log-linear fit')
    for j,name in enumerate(colors):
        d=s[s.estimator==name]
        ax.scatter(d.rho,d.sensitivity_ratio,marker=['^','s','D'][j],s=38,
                   color=colors[name],label=labels[name]+' stress cells')
    ax.axhline(.5,color='#888888',ls='--',lw=.8)
    ax.axvline(f.threshold_rho_fitted,color='#888888',ls=':',lw=.8)
    ax.set(xscale='log',xlabel=r'Parameter-to-information ratio $\rho$',
           ylabel='Adjustment movement / SD',ylim=(-.05,1.6))
    ax.set_title('A. Original calibration and new stress cells',loc='left',fontsize=11,weight='bold')
    ax.legend(fontsize=8,frameon=False,ncol=2,loc='upper left')
    cell_labels=['PH / 136','PH / 988','Delayed / 136','Delayed / 988','Nonlinear / 136','Nonlinear / 988']
    for j,name in enumerate(colors):
        d=s[s.estimator==name].sort_values('cell');x=np.arange(6)+(j-1)*.18
        yy=d.coverage.to_numpy()*100
        bx.errorbar(x,yy,yerr=np.vstack((yy-100*d.coverage_mc_lo,100*d.coverage_mc_hi-yy)),
                    fmt=['^','s','D'][j],ms=5,color=colors[name],capsize=2,lw=1,label=labels[name])
    bx.axhline(95,color='#555555',ls='--',lw=.8)
    bx.set(xticks=np.arange(6),xticklabels=cell_labels,ylabel='Interval coverage (%)',ylim=(50,102))
    bx.tick_params(axis='x',labelrotation=35,labelsize=9)
    bx.set_title('B. Stress coverage with Monte Carlo intervals',loc='left',fontsize=11,weight='bold')
    bx.legend(fontsize=9,loc='lower left',frameon=False,ncol=3)
    for a in (ax,bx):
        a.spines[['top','right']].set_visible(False)
        a.grid(axis='y',alpha=.18)
    fig.tight_layout(h_pad=2)
    fig.savefig(FIG/'fig8_calibration_stress.pdf');plt.close(fig)

if __name__=='__main__':
    FIG.mkdir(exist_ok=True,parents=True)
    workflow();validation()
    (ROOT/'paper/figure_alt_text.txt').write_text(
        'Figure 1. Six sequential workflow boxes: audit inputs; specify the day-90 landmark RMST contrast; '
        'assess information and weight support; compare four estimators; challenge assumptions using sensitivity '
        'analysis and simulation; interpret uncertainty and identification limits.\n\n'
        'Figure 2. Top: original AIPW calibration cells and fitted sensitivity curve against the parameter-to-information '
        'ratio, with new AIPW, weighted KM and Cox stress cells. Bottom: coverage in six stress scenarios with '
        'Monte Carlo confidence bars. Cox coverage is lowest in the larger delayed-benefit scenario. '
        'The horizontal reference is nominal 95 percent coverage.\n',encoding='utf-8')
    print('Generated workflow and calibration/stress figures.')
