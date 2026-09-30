"""Reproduce the Biometrics reanalysis from supplied clinical and derived inputs."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent
STAGES=['00_validate_estimators.py','01_receptors.py','02_tnbc.py','03_survival.py',
        '04_chemo_exposure.py','05_landmark.py','06_site.py','07_estimate.py',
        '08_naive_vs_corrected.py','10_figures.py','15_lifelines_crosscheck.py',
        '16_estimability_diagnostic.py','17_publication_figures.py','18_montecarlo_error.py',
        '20_biometrics_comparisons.py','21_stress_validation.py']
RENDER=['23_biometrics_figures.py','22_write_biometrics.py']
REQUIRED=['clinical.tsv','follow_up.tsv','outcome.csv','brca_pam50.txt',
          'causal_features_dataset.csv','ite_ready_dataset_v2.csv']
def run(args):
    subprocess.run([sys.executable,'-B','-X','utf8','-u',*args],cwd=ROOT,check=True)
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-only',action='store_true')
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--compile',action='store_true')
    args=parser.parse_args()
    for name in REQUIRED:
        if not (ROOT/'data'/name).is_file():raise FileNotFoundError('Missing input: data/'+name)
    if args.verify:
        run(['-m','unittest','discover','-s','tests','-p','test_biometrics_estimators.py'])
        run(['scripts/25_verify_revision.py'])
        return
    started=time.time()
    if not args.render_only:
        # Every stage executes; a failure stops rendering. Existing outputs are retained for inspection.
        for stage in STAGES:
            print('Running '+stage,flush=True)
            run(['scripts/'+stage])
    for stage in RENDER:run(['scripts/'+stage])
    inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT/'data').rglob('*')) if p.is_file()}
    (ROOT/'output/input_sha256.json').write_text(json.dumps(inputs,indent=2))
    if args.compile:
        spec=importlib.util.spec_from_file_location('build',ROOT/'scripts/24_build_submission.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        mod.compile_documents();mod.qa()
    print(f'Completed in {time.time()-started:.1f} seconds.')
if __name__=='__main__':main()
