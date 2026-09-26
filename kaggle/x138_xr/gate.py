"""Gate a finished kernel run: parse its log, check visible counts vs targets, no degrade/fallback/errors.
usage: python gate.py LOGFILE 'ds=nodes/edges,ds=nodes/edges,...' [tol_frac]
exit 0 = pass, 1 = fail
"""
import json
import re
import sys

log, targets = sys.argv[1], sys.argv[2]
tol = float(sys.argv[3]) if len(sys.argv) > 3 else 0.004
t = open(log, encoding="utf-8", errors="replace").read()
try:
    s = "".join(e.get("data", "") for e in json.loads(t))
except Exception:
    s = t
bad = [l for l in s.splitlines() if re.search(r"DEADLINE:|REPAIR FAILED|Traceback|repair_fallback\"?: ?1|Error:", l)]
final = {m.group(1): (int(m.group(2)), int(m.group(3)))
         for m in re.finditer(r"^\s+(\S+): nodes=(\d+) edges=(\d+) divisions=\d+", s, re.M)}
ok = not bad and len(final) == 4
print("final:", final)
for item in targets.split(","):
    ds, ne = item.split("=")
    n, e = (int(x) for x in ne.split("/"))
    got = final.get(ds)
    if got is None:
        print("MISSING", ds); ok = False; continue
    dn, de = abs(got[0] - n) / n, abs(got[1] - e) / e
    flag = "ok" if dn <= tol and de <= tol else "MISMATCH"
    if flag != "ok":
        ok = False
    print(f"{ds}: got {got} target {(n, e)} dn={dn:.4f} de={de:.4f} {flag}")
rows = re.findall(r"Final submission.csv rows=(\d+)\s+config=(\S+)", s)
print("rows/config:", rows)
if rows and rows[-1][1] != "base":
    ok = False; print("NOT base config")
if bad:
    print("BAD LINES:", bad[:5])
print("GATE", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
