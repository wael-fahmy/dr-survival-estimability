"""Compile the article and supplement, then run layout checks on the PDFs."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess
import sys
import fitz

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'paper'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def compile_documents():
    for name in ['sn-article','supplement']:
        for _ in range(3):
            result=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',name+'.tex'],
                                  cwd=PAPER,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            (PAPER/(name+'.build-output.txt')).write_text(result.stdout,encoding='utf-8')
            if result.returncode:raise RuntimeError('LaTeX failed: '+name+'; see paper/'+name+'.build-output.txt')
        log=(PAPER/(name+'.log')).read_text(errors='replace')
        if re.search(r'undefined references|undefined citations|Citation .* undefined|Reference .* undefined',log):
            raise RuntimeError('Unresolved LaTeX references in '+name)
        print(name+': compiled')

def qa():
    (PAPER/'validation').mkdir(exist_ok=True,parents=True)
    report={}
    for name in ['sn-article','supplement']:
        doc=fitz.open(PAPER/(name+'.pdf'))
        texts=[p.get_text() for p in doc]
        # Headings identify counted text; title, summary and display pages are excluded.
        table_page=next((i for i,t in enumerate(texts) if re.search(r'^Table 1:',t,re.M)),None) if name=='sn-article' else None
        intro=next((i for i,t in enumerate(texts) if 'Introduction' in t),None) if name=='sn-article' else None
        report[name]={'physical_pages':len(doc),'counted_pages':table_page-intro if table_page is not None and intro is not None else None,
                      'unresolved_question_marks':any('??' in t for t in texts)}
        if name=='sn-article':
            summary=texts[1].split('Summary.',1)[1].split('Key words:',1)[0]
            report[name]['summary_words']=len(summary.split())
            assert report[name]['summary_words']<=225
            assert report[name]['counted_pages'] is not None and report[name]['counted_pages']<=25
        assert not report[name]['unresolved_question_marks']
        log=(PAPER/(name+'.log')).read_text(errors='replace')
        report[name]['overfull_boxes']=re.findall(r'Overfull \\[hv]box[^\n]*',log)
        report[name]['known_longtable_toolchain_warnings']=log.count('Infinite glue shrinkage found in box being split')
        # TeX glyph bboxes extend slightly beyond their mathematical box; report rather than
        # treating tiny font sidebearings as layout failure. Inspect flagged large deviations.
        bounds=[]
        for i,p in enumerate(doc):
            for b in p.get_text('dict')['blocks']:
                if b['type']!=0:continue
                for line in b['lines']:
                    for s in line['spans']:
                        x0,y0,x1,y1=s['bbox']
                        if x0<69 or x1>543:bounds.append({'page':i+1,'text':s['text'],'bbox':s['bbox']})
        report[name]['horizontal_margin_flags']=bounds
        assert not report[name]['overfull_boxes'],name+' has overfull layout boxes'
        assert not bounds,name+' exceeds horizontal text margins'
        (PAPER/'validation'/('revised_'+name+'.txt')).write_text('\n\f\n'.join(texts),encoding='utf-8')
    (ROOT/'output/24_document_qa.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--qa-only',action='store_true')
    args=parser.parse_args()
    if not args.qa_only:compile_documents()
    qa()
