"""Compare final graphs by coordinates and freeze only five verified, distinct runs."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / 'results/E071'
sys.path.insert(0, str(ROOT / 'src'))
from biohub_lab.submission import read_and_validate


def load(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, obj):
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(p)


def graph_signature(nodes, edges):
    positions = {i: tuple(v[k] for k in ('t','z','y','x')) for i,v in nodes.items()}
    nc = Counter(positions.values())
    ec = Counter((positions[a], positions[b]) for a,b in edges)
    outgoing = Counter(a for a,b in edges)
    divisions = sum(v == 2 for v in outgoing.values())
    # Canonical representation ignores arbitrary node/row numbering, retains multiplicity.
    canonical = json.dumps([sorted(nc.items()), sorted(ec.items())], separators=(',',':'))
    return nc, ec, divisions, hashlib.sha256(canonical.encode()).hexdigest()


def compare():
    plans = load(RES / 'candidate_plan.json')['candidates']
    shapes = load(ROOT / 'results/E054_CONTROL_completed.json')['result']['shapes']
    base_path = ROOT / 'outputs/e071/baseline_output/submission.csv'
    base = {s: graph_signature(n,e) for s,(n,e) in read_and_validate(base_path,shapes).items()}
    output = []
    for row in plans:
        name = row['candidate']; receipt = RES / f'{name}_verified.json'
        if not receipt.exists(): continue
        verified = load(receipt); csv = ROOT / 'outputs/e071' / name / 'submission.csv'
        assert sha(csv) == verified['csv_sha256']
        assert sha(Path(row['folder']) / 'notebook.ipynb') == verified['notebook_sha256']
        assert verified['gate'] == verified['csv_bounds_and_graph'] == 'PASS'
        graphs = read_and_validate(csv,shapes)
        details = {}
        for stem, (nodes, edges) in graphs.items():
            n,e,div,h = graph_signature(nodes,edges)
            bn,be,bd,bh = base[stem]
            details[stem] = dict(nodes=len(nodes), edges=len(edges), divisions=div,
                node_count_delta=len(nodes)-sum(bn.values()),
                node_count_delta_fraction=(len(nodes)-sum(bn.values()))/sum(bn.values()),
                nodes_added=sum((n-bn).values()), nodes_removed=sum((bn-n).values()),
                edges_added=sum((e-be).values()), edges_removed=sum((be-e).values()),
                division_count_delta=div-bd, canonical_sha256=h, same_as_c3=h==bh)
        combined = hashlib.sha256(json.dumps({k:v['canonical_sha256'] for k,v in sorted(details.items())}, sort_keys=True).encode()).hexdigest()
        output.append(dict(**verified, details=details, canonical_sha256=combined,
            same_as_c3=all(v['same_as_c3'] for v in details.values()),
            local_file=str(csv), file_name='submission.csv', folder=row['folder'],
            verification_receipt=str(receipt), overrides=row['overrides'],
            evidence=row['evidence'], notebook_url='https://www.kaggle.com/code/'+row['kernel']))
    groups = {}
    for r in output: groups.setdefault(r['canonical_sha256'],[]).append(r['candidate'])
    duplicates = [g for g in groups.values() if len(g)>1]
    summary = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),
        verified_count=len(output), candidates=output, duplicate_graphs=duplicates,
        no_op_candidates=[r['candidate'] for r in output if r['same_as_c3']],
        reference_csv_sha256=sha(base_path))
    save(RES / 'visible_comparison.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='candidates'}))
    for r in output:
        print(json.dumps(dict(candidate=r['candidate'], changes={k:{key:v[key] for key in ['node_count_delta','nodes_added','nodes_removed','edges_added','edges_removed','division_count_delta']} for k,v in r['details'].items()})))
    if len(output)==5:
        assert not duplicates and not summary['no_op_candidates'], 'Do not freeze redundant submissions'
        assert not (RES/'submission_queue.json').exists(), 'Queue already frozen; do not overwrite'
        for i,r in enumerate(output,1): r['order']=i
        save(RES/'submission_queue.json', dict(experiment='E071', status='READY',
            ready_at_utc=summary['checked_at_utc'], competition='biohub-cell-tracking-during-development',
            reset_utc='2026-09-29T00:00:00+00:00', reset_lima='2026-09-28T19:00:00-05:00',
            auto_submit=False, reminder_enabled=False, leaderboard_submissions_made=0,
            base_public_score=.955, candidates=output,
            policy='Prepared for final reset; no leaderboard submission from preparation. Check live quota, kernel version and receipts before any later authorized send.',
            warning='Distinct visible graphs are a technical check, not proof of leaderboard improvement. No clean independent validation exists for these public all-train weights.'))
        state=load(RES/'ledger.json');state.update(status='FIVE_READY',ready_count=5,
            completed_at_utc=summary['checked_at_utc'],auto_submit=False,reminder_enabled=False)
        save(RES/'ledger.json',state)


if __name__=='__main__': compare()
