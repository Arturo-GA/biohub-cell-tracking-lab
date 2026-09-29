# E070 — cinco submissions completas, sin mejora pública

**Cierre confirmado 28-09, 19:56 UTC:** fork8 **0.953**, DC030 **0.947**, prune12 **0.953**, prune12_joint **0.951** y joint **0.951**. Ninguna supera c3 **0.955**. Evidencia: `results/E070/all_submission_status.json` y `public_decision.json`. Los recibos de envío son definitivos; no repetir envíos. El seguimiento está desactivado por petición de Arturo. E071 volvió a c3 y sus cinco resultados completos están en [E071_EXECUTION.es.md](E071_EXECUTION.es.md).

## Historial anterior al cierre

Snapshot 2026-09-28T12:11:34.504328+00:00. The following states and instructions are historical and superseded by the closure above.

## Public results and authorized decision

C3_fork8 v1 (56622884) completed at 0.953. F8_dc030 v1 (56622918) completed at 0.947. Both regress versus the confirmed c3 reference 0.955. The local improvements did not transfer to public. Do not treat DC030 as an improved base.

Arturo explicitly authorized the other three if neither first candidate improved 0.955. That condition was rechecked on Kaggle before each call. The three were accepted on 28-09-2026 at 12:08-12:09 UTC / 07:08-07:09 Lima:

| Candidate | Kernel version | Submission | Last status |
|---|---|---|---|
| F8_prune12 | jarturo/biohub-e070-f8-prune12 v1 | 56640207 | PENDING |
| F8_prune12_joint | jarturo/biohub-e070-f8-prune12-joint v1 | 56640212 | PENDING |
| F8_joint | jarturo/biohub-e070-f8-joint v1 | 56640230 | PENDING |

**All five daily slots are consumed. Do not send anything again.** No new inference was launched: these were frozen, previously verified notebooks. Durable receipts are results/E070/*_submission_attempt.json; each has the exact version, hash and Kaggle ref.

## Remaining follow-up

The heartbeat biohub-completar-cinco-submissions remains ACTIVE every 25 minutes, now read-only. Run:

    C:/dev/rsna-knee/.venv/Scripts/python.exe kaggle/x138_xr/e070_submit.py status-all

Inspect results/E070/all_submission_status.json and update submission_queue.json and ledger.json for meaningful state changes. Notify only a completed score, error or required decision. Compare results against c3 0.955 and fork8 0.953. Do not reuse the earlier unfavorable-branch instructions to submit the same candidates. Pause this heartbeat when the three final refs are terminal and report the outcome.

The authoritative decision is results/E070/public_decision.json; the original conditional authorization is conditional_next_steps.json. The first two results were actually verified at 12:07 UTC; the triggering heartbeat timestamp was earlier and is not the verification time. No more GPU work or submissions are authorized from this monitoring stage.

## Seguimiento desactivado por Arturo

2026-09-28T15:00:26.283106+00:00: automatizacion `biohub-completar-cinco-submissions` confirmada PAUSED. Arturo avisara cuando terminen los notebooks. Esta indicacion sustituye el seguimiento automatico descrito arriba; no modifica los envios aceptados ni sus resultados.
