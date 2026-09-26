"""Paired stem-bootstrap of lab rule deltas (stdlib only).
usage: python boot.py rows.csv [rows2.csv ...] [--ref L7]"""
import csv
import random
import sys
from collections import defaultdict

args = [a for a in sys.argv[1:] if not a.startswith("--")]
ref = sys.argv[sys.argv.index("--ref") + 1] if "--ref" in sys.argv else "L7"
if ref in args:
    args.remove(ref)
rows = defaultdict(dict)  # rule -> stem -> row
for path in args:
    for r in csv.DictReader(open(path, encoding="utf-8")):
        rows[r["rule"]][r["stem"]] = {k: float(r[k]) for k in ("adjusted_edge_jaccard", "weight", "div_tp", "div_fp", "div_fn")}


def score(rule, stems):
    R = rows[rule]
    w = sum(R[s]["weight"] for s in stems) or 1.0
    adj = sum(R[s]["adjusted_edge_jaccard"] * R[s]["weight"] for s in stems) / w
    tp = sum(R[s]["div_tp"] for s in stems); fp = sum(R[s]["div_fp"] for s in stems); fn = sum(R[s]["div_fn"] for s in stems)
    dj = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return adj + 0.1 * dj


stems = sorted(rows[ref])
rng = random.Random(0)
B = 2000
boots = [[rng.choice(stems) for _ in stems] for _ in range(B)]
print(f"n_stems={len(stems)} ref={ref} ref_score={score(ref, stems):.5f}")
out = []
for rule in rows:
    if set(rows[rule]) != set(stems):
        continue
    d = score(rule, stems) - score(ref, stems)
    ds = sorted(score(rule, b) - score(ref, b) for b in boots)
    p = sum(x > 0 for x in ds) / B
    win = sum(1 for s in stems if score(rule, [s]) > score(ref, [s]) + 1e-12)
    lose = sum(1 for s in stems if score(rule, [s]) < score(ref, [s]) - 1e-12)
    out.append((d, rule, ds[int(0.05 * B)], ds[int(0.95 * B)], p, win, lose))
for d, rule, lo, hi, p, win, lose in sorted(out, reverse=True):
    print(f"{rule:<24} delta={d:+.5f}  90%CI=[{lo:+.5f},{hi:+.5f}]  P(>0)={p:.2f}  stems +{win}/-{lose}")
