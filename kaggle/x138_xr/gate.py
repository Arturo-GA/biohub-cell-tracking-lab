"""Gate a finished kernel run: parse its log, check visible counts vs targets, no degrade/fallback/errors.
usage: python gate.py LOGFILE 'ds=nodes/edges,ds=nodes/edges,...' [tol_frac]
exit 0 = pass, 1 = fail
"""
import json
import re
import sys
import math

log, targets = sys.argv[1], sys.argv[2]
tol = float(sys.argv[3]) if len(sys.argv) > 3 else 0.004
if not math.isfinite(tol) or not 0 <= tol <= 0.05:
    raise SystemExit("GATE FAIL: tolerance must be between 0 and 0.05; disabled count checks are forbidden")
t = open(log, encoding="utf-8", errors="replace").read()
try:
    s = "".join(e.get("data", "") for e in json.loads(t))
except Exception:
    s = t
bad = [l for l in s.splitlines() if re.search(
    r"DEADLINE:|REPAIR FAILED|Traceback|repair_fallback[\"']?\s*:\s*1|deadline_degraded[\"']?\s*:\s*1|Error:|XR.*(?:FAILED|ERROR)|xr_\w+ skipped", l)]
final = {m.group(1): (int(m.group(2)), int(m.group(3)))
         for m in re.finditer(r"^\s+(\S+): nodes=(\d+) edges=(\d+) divisions=\d+", s, re.M)}
ok = not bad and len(final) == 4
expected_names = {item.split("=")[0] for item in targets.split(",")}
if set(final) != expected_names:
    ok = False
    print("DATASET MISMATCH", sorted(final), sorted(expected_names))
print("final:", final)
for item in targets.split(","):
    ds, ne = item.split("=")
    n, e = (int(x) for x in ne.split("/"))
    if n <= 1 or e <= 1:
        raise SystemExit("GATE FAIL: supply actual candidate counts, not placeholder targets")
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
if len(rows) != 1 or rows[0][1] != "base":
    ok = False; print("MISSING/REPEATED marker or NOT base config")
elif int(rows[0][0]) != sum(n + e for n, e in final.values()):
    ok = False; print("FINAL ROW COUNT MISMATCH")
if bad:
    print("BAD LINES:", bad[:5])
print("GATE", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
