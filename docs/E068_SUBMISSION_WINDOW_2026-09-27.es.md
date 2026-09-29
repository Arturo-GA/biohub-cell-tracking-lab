# Ventana de envíos del 27-09 UTC

A las 00:06 UTC, al atender la petición de enviar B/A, la API ya mostraba cinco submissions nuevas PENDING:

- 56591883 S1, 00:01:16 UTC.
- 56591970 S5, 00:01:45 UTC.
- 56592042 S2, 00:02:13 UTC.
- 56592094 c6, 00:02:43 UTC.
- 56592151 c3, 00:03:13 UTC.

Son los candidatos de la tanda anterior, no A/B. La API confirma `maxDailySubmissions=5`. Se intentó una sola vez enviar B (`jarturo/biohub-exact-b`, v1, submission.csv); Kaggle devolvió HTTP 400. No se conservó el cuerpo detallado de la respuesta, por lo que no se atribuye al error un mensaje específico. La consulta posterior confirmó que el intento no creó otra submission. Los cinco slots diarios están ocupados.

No se intentó A ni se repitieron peticiones de envío. Ambos siguen COMPLETE y con artefactos verificados. La siguiente ventana prevista es 28-09 00:00 UTC = 27-09 19:00 Lima.

La consulta de solo lectura de procesos Python/bash/node relacionados con Biohub/submit5 y tareas programadas identificables como Biohub/Kaggle no encontró coincidencias activas. El origen de los cinco envíos no se pudo determinar; no afirmar que proceden de Claude, de otro agente o de un script concreto.

Recibos: `results/E068/submission_attempt_2026-09-27_B.json` y `results/E068/submissions_2026-09-27_opening.json`. No se creó una automatización nueva.
