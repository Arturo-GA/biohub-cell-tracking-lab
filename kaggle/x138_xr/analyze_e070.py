"""Rank complete E070 labs and inventory truly verified candidate artifacts."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from analyze_cpu4 import analyze, score

ROOT=Path(__file__).resolve().parents[2]
RES=ROOT/'results/E070'


def load(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    combined={}; results={};diagnostics={}
    for part in ('a','b'):
        assert load(RES/f'lab_{part}_completed.json')['manifest_verified']
        folder=ROOT/f'outputs/e070/lab_{part}/e070_{part}'
        analyze(folder,'C3_public0955',199)
        for path in folder.glob('*.rows.json'):
            name=path.name.removesuffix('.rows.json');rows=load(path)
            if name in combined:
                keys=('edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn','num_pred_nodes','adj_edge_jaccard')
                assert set(rows)==set(combined[name])
                assert all(abs(rows[s][k]-combined[name][s][k])<1e-10 for s in rows for k in keys),('Controls differ between labs',name)
            combined[name]=rows
        for row in load(folder/'results.json'):results[row['name']]=row
        diagnostics.update(load(folder/'graph_diagnostics.json'))
    stems=sorted(combined['C3_public0955'])
    def matrix(rows):
        return np.array([[rows[s]['adj_edge_jaccard']*(rows[s]['edge_tp']+rows[s]['edge_fp']+rows[s]['edge_fn']),
                          rows[s]['edge_tp']+rows[s]['edge_fp']+rows[s]['edge_fn'],rows[s]['division_tp'],
                          rows[s]['division_tp']+rows[s]['division_fp']+rows[s]['division_fn']] for s in stems])
    arrays={k:matrix(v) for k,v in combined.items()};control=arrays['C3_fork8']
    rng=np.random.default_rng(20260927);weights=rng.multinomial(199,np.full(199,1/199),size=1500)
    ranked=[]
    for name,a in arrays.items():
        delta=float(score(a)-score(control));loo=np.array([delta-(score(np.delete(a,i,0))-score(np.delete(control,i,0))) for i in range(199)])
        keep=np.ones(199,bool);keep[np.argsort(loo)[-5:]]=False
        b=score(a,weights)-score(control,weights)
        row=dict(results[name],increment_over_fork8=delta,increment_ci90=list(np.quantile(b,[.05,.95])),
                 increment_remove_top5=float(score(a[keep])-score(control[keep])),
                 changed_graphs_vs_fork8=sum(diagnostics[name][s]['final_graph_hash']!=diagnostics['C3_fork8'][s]['final_graph_hash'] for s in stems))
        ranked.append(row)
    ranked.sort(key=lambda r:r['score'],reverse=True)
    (RES/'ranking.json').write_text(json.dumps(dict(videos=199,ranking=ranked,note='Training/tuning diagnostics; not a public-score prediction or private holdout.'),indent=2)+'\n')
    with (RES/'rows.csv').open('w',newline='') as f:
        keys=['config']+list(next(iter(combined['C3_public0955'].values())).keys());writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for name,rows in combined.items():
            for s in stems:writer.writerow(dict(config=name,**rows[s]))
    for row in ranked:print(f"{row['name']:24} {row['score']:.9f} vsC3 {row['delta']:+.9f} vsF8 {row['increment_over_fork8']:+.9f} changed {row['changed_graphs_vs_fork8']}")
    verified=[load(ROOT/'results/E068/c3_fork8_verified.json')]+[load(p) for p in RES.glob('*_verified.json')]
    for r in verified:assert r['gate']=='PASS' and r['csv_bounds_and_graph']=='PASS'
    hashes={};duplicates=[]
    for r in verified:
        if r['csv_sha256'] in hashes:duplicates.append([hashes[r['csv_sha256']],r['candidate']])
        else:hashes[r['csv_sha256']]=r['candidate']
    inventory=dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),verified_candidates=verified,
                   distinct_visible_csv_count=len(hashes),duplicate_visible_outputs=duplicates,
                   note='Ready for manual selection. Do not equate technical validity with demonstrated score improvement. No automatic leaderboard submission.')
    (RES/'verified_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print('Verified candidates:',len(verified),'Distinct visible CSVs:',len(hashes),'Duplicates:',duplicates)


if __name__=='__main__':main()
