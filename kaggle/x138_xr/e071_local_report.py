"""Summarize locally replayed candidates without claiming Kaggle readiness."""
import json
from pathlib import Path
from datetime import datetime, timezone
from e071_local_replay import ROOT, OUT, RES, ARMS, sha, save
from finalize_e071 import read_and_validate, graph_signature


def report():
    shapes=json.loads((ROOT/'results/E054_CONTROL_completed.json').read_text())['result']['shapes']
    reference=read_and_validate(ROOT/'outputs/e071/baseline_output/submission.csv',shapes)
    base={stem:graph_signature(*g) for stem,g in reference.items()}
    rows=[]
    for name in ARMS:
        p=OUT/name/'verified.json'
        if not p.exists():continue
        rec=json.loads(p.read_text());csv=OUT/name/'submission.csv'
        assert rec['stage']=='LOCAL_VERIFIED_ONLY' and rec['csv_sha256']==sha(csv)
        if name=='C3_D04_R094':assert all(rec['remote_parity'].values())
        details={}
        for stem,g in read_and_validate(csv,shapes).items():
            n,e,div,h=graph_signature(*g);bn,be,bd,bh=base[stem]
            details[stem]=dict(nodes=sum(n.values()),edges=sum(e.values()),divisions=div,
                nodes_added=sum((n-bn).values()),nodes_removed=sum((bn-n).values()),
                edges_added=sum((e-be).values()),edges_removed=sum((be-e).values()),
                same_as_scored_c3=h==bh,canonical_sha256=h)
        rows.append(dict(candidate=name,division=ARMS[name][0],readmission=ARMS[name][1],
            csv_sha256=rec['csv_sha256'],file=str(csv),details=details,
            remote_complete=name=='C3_D04_R094',public_score=None))
    candidates=[r for r in rows if r['candidate']!='C3_CONTROL']
    fingerprints={}
    for r in candidates:
        fp=tuple((s,d['canonical_sha256']) for s,d in sorted(r['details'].items()))
        fingerprints.setdefault(fp,[]).append(r['candidate'])
    duplicates=[v for v in fingerprints.values() if len(v)>1]
    result=dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),stage='LOCAL_ONLY',
        candidate_count=len(candidates),kaggle_complete_count=1,candidates=rows,
        duplicate_graphs=duplicates,auto_submit=False,reminder_enabled=False,
        warning='Local replay verification is not a new public score or a completed Kaggle notebook.')
    save(RES/'local_candidate_comparison.json',result)
    print(json.dumps(dict(local_candidates=len(candidates),duplicates=duplicates,
                         controls=len(rows)-len(candidates),kaggle_complete=1)))
    if len(candidates)==5:
        assert not duplicates,'Do not prepare redundant final submissions'
        assert not any(all(d['same_as_scored_c3'] for d in r['details'].values()) for r in candidates)
        ledger=json.loads((RES/'ledger.json').read_text())
        ledger['local_execution']['status']='FIVE_LOCAL_VERIFIED'
        ledger['local_execution']['comparison']='results/E071/local_candidate_comparison.json'
        save(RES/'ledger.json',ledger)


if __name__=='__main__':report()
