"""Summarize completed CPU4 output; diagnostics, not private-score guarantees."""
import argparse
import json
from pathlib import Path
import numpy as np


def score(matrix, weights=None):
    # Columns: adjusted-edge weighted numerator, edge union, div TP, div union.
    totals = matrix.sum(axis=0) if weights is None else weights @ matrix
    return totals[..., 0] / np.where(totals[..., 1] > 0, totals[..., 1], 1) + 0.1 * totals[..., 2] / np.where(totals[..., 3] > 0, totals[..., 3], 1)


def analyze(folder, baseline='base(x138)', expected_videos=199):
    all_rows = {p.name.removesuffix('.rows.json'): json.loads(p.read_text()) for p in folder.glob('*.rows.json')}
    base = all_rows[baseline]
    stems = sorted(base)
    assert len(stems) == expected_videos
    assert all(set(r) == set(stems) for r in all_rows.values())
    def array(rows):
        out = []
        for stem in stems:
            r = rows[stem]
            eu = r['edge_tp'] + r['edge_fp'] + r['edge_fn']
            du = r['division_tp'] + r['division_fp'] + r['division_fn']
            out.append([r['adj_edge_jaccard'] * eu, eu, r['division_tp'], du])
        return np.array(out, dtype=float)
    arrays = {k: array(v) for k, v in all_rows.items()}
    control = arrays[baseline]
    rng = np.random.default_rng(20260926)
    weights = rng.multinomial(len(stems), np.full(len(stems), 1/len(stems)), size=1500)
    g44 = np.array([s.startswith('44b6') for s in stems])
    output = []
    for name, a in arrays.items():
        delta = float(score(a) - score(control))
        # Removal sensitivity uses contribution to aggregate delta, not raw video score.
        loo = np.array([delta - (score(np.delete(a,i,0))-score(np.delete(control,i,0))) for i in range(len(stems))])
        order = np.argsort(loo)[::-1]
        sensitivity = {}
        for n in (1,3,5):
            keep = np.ones(len(stems),dtype=bool); keep[order[:n]]=False
            sensitivity[str(n)] = float(score(a[keep])-score(control[keep]))
        boot = score(a,weights)-score(control,weights)
        mixed = {}
        for share in (0.25,0.5,0.75):
            w = np.where(g44, share/g44.sum(), (1-share)/(~g44).sum()) * len(stems)
            mixed[str(share)] = float(score(a,w)-score(control,w))
        individual = np.array([score(a[i:i+1])-score(control[i:i+1]) for i in range(len(stems))])
        item = dict(name=name, score=float(score(a)), delta=delta,
                    ci90=list(np.quantile(boot,[0.05,0.95])),
                    d44=float(score(a[g44])-score(control[g44])),
                    d6b=float(score(a[~g44])-score(control[~g44])),
                    remove_top_gain_videos=sensitivity, mixtures=mixed,
                    worst_videos=[dict(stem=stems[i],delta=float(individual[i])) for i in np.argsort(individual)[:5]])
        if 'D_defensive_P8' in arrays:
            item['delta_vs_D'] = float(score(a)-score(arrays['D_defensive_P8']))
        output.append(item)
    output.sort(key=lambda r:r['score'],reverse=True)
    report = dict(baseline=baseline,note='Reused training/tuning videos; bootstrap is descriptive, not a private improvement probability.',
                  videos=len(stems), ranking=output)
    (folder/'robustness.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for r in output:
        print(f"{r['name']:25} score={r['score']:.6f} delta={r['delta']:+.6f} 44={r['d44']:+.6f} 6b={r['d6b']:+.6f} remove5={r['remove_top_gain_videos']['5']:+.6f}")


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--base',default='base(x138)');p.add_argument('--videos',type=int,default=199)
    args=p.parse_args();analyze(args.folder,args.base,args.videos)
