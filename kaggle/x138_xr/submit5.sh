#!/usr/bin/env bash
# Timed submitter: waits until the UTC date reaches SUBMIT_DAY (00:00 UTC reset), re-gates each kernel's
# latest log, then submits in priority order. No score polling. usage: bash submit5.sh 2026-09-27 slug:version:"msg" ...
# This legacy path does not bind the downloaded log to the submitted version.
# Refuse until replaced by artifact/hash/version verification; gate.py alone cannot do that.
echo "DISABLED: use a verified, pinned kernel version and candidate-specific CSV counts."
exit 2
K=/c/Users/Arturo/Diplomado/Scripts/kaggle
COMP=biohub-cell-tracking-during-development
SUBMIT_DAY=$1; shift
BASE="44b6_0113de3b=1/1,44b6_0b24845f=1/1,6bba_05b6850b=1/1,6bba_05db0fb1=1/1"
while [ "$(date -u +%Y-%m-%d)" \< "$SUBMIT_DAY" ]; do sleep 60; done
sleep 30
for spec in "$@"; do
  slug="${spec%%:*}"; rest="${spec#*:}"; ver="${rest%%:*}"; msg="${rest#*:}"
  st=$($K kernels status jarturo/$slug 2>&1)
  case "$st" in *COMPLETE*) ;; *) echo "$(date -u +%H:%M) SKIP $slug: $st"; continue;; esac
  $K kernels logs jarturo/$slug > "log_final_$slug.json" 2>/dev/null
  if python gate.py "log_final_$slug.json" "$BASE" 1e9 > "gate_final_$slug.txt"; then
    r=$($K competitions submit $COMP -k jarturo/$slug -v "$ver" -f submission.csv -m "$msg" 2>&1 | tail -1)
    echo "$(date -u +%H:%M) SUBMIT $slug v$ver: $r"
  else
    echo "$(date -u +%H:%M) GATE FAIL $slug"; grep -E "BAD|MISMATCH|NOT base" "gate_final_$slug.txt"
  fi
  sleep 20
done
echo "submit5 done"
