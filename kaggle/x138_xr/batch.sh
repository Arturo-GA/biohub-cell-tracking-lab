#!/usr/bin/env bash
# Push kernels (retrying on the 2-GPU-session limit), gate each finished run, then submit in priority
# order once the UTC day has rolled over to SUBMIT_DAY. usage: bash batch.sh SUBMIT_DAY folder:slug:"message" ...
K=/c/Users/Arturo/Diplomado/Scripts/kaggle
COMP=biohub-cell-tracking-during-development
SUBMIT_DAY=$1; shift
BASE="44b6_0113de3b=1/1,44b6_0b24845f=1/1,6bba_05b6850b=1/1,6bba_05db0fb1=1/1"
declare -a FOLDERS SLUGS MSGS STATE
i=0
for spec in "$@"; do
  FOLDERS[$i]="${spec%%:*}"; rest="${spec#*:}"; SLUGS[$i]="${rest%%:*}"; MSGS[$i]="${rest#*:}"; STATE[$i]=new; i=$((i+1))
done
n=$i
while :; do
  alldone=1
  for ((j=0; j<n; j++)); do
    f=${FOLDERS[$j]}; s=${SLUGS[$j]}
    case ${STATE[$j]} in
      new)
        if [ "$f" = "-" ]; then STATE[$j]=running; else
          out=$($K kernels push -p "$f" 2>&1 | tail -1)
          case "$out" in *successfully*) echo "$(date -u +%H:%M) pushed $s"; STATE[$j]=running;; *) :;; esac
        fi; alldone=0;;
      running)
        st=$($K kernels status jarturo/$s 2>&1)
        case "$st" in
          *COMPLETE*) $K kernels logs jarturo/$s > "log_$s.json" 2>/dev/null
            if python gate.py "log_$s.json" "$BASE" 1e9 > "gate_$s.txt"; then STATE[$j]=ready; else STATE[$j]=failed; fi
            echo "$(date -u +%H:%M) $s -> ${STATE[$j]}"; grep -E "final:|rows/config|BAD" "gate_$s.txt";;
          *ERROR*|*CANCEL*) STATE[$j]=failed; echo "$(date -u +%H:%M) $s FAILED: $st";;
        esac; alldone=0;;
      ready)
        if [ "$(date -u +%Y-%m-%d)" \> "$SUBMIT_DAY" ] || [ "$(date -u +%Y-%m-%d)" = "$SUBMIT_DAY" ]; then
          # submit strictly in priority order: all earlier items must be submitted or failed
          ok=1; for ((q=0; q<j; q++)); do case ${STATE[$q]} in submitted|failed|skipped) ;; *) ok=0;; esac; done
          if [ $ok = 1 ]; then
            r=$($K competitions submit $COMP -k jarturo/$s -f submission.csv -v 1 -m "${MSGS[$j]}" 2>&1 | tail -1)
            echo "$(date -u +%H:%M) submit $s: $r"
            case "$r" in *remaining*|*uccess*) STATE[$j]=submitted;; *"0 submissions"*|*limit*|*exceed*) STATE[$j]=skipped;; *) STATE[$j]=skipped;; esac
          fi
        fi; alldone=0;;
    esac
  done
  fin=1; for ((j=0; j<n; j++)); do case ${STATE[$j]} in submitted|failed|skipped) ;; *) fin=0;; esac; done
  [ $fin = 1 ] && break
  sleep 60
done
echo "FINAL:"; for ((j=0; j<n; j++)); do echo "  ${SLUGS[$j]} ${STATE[$j]}"; done
