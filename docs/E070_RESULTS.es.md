# E070 — resultados públicos de los cinco candidatos

> **Cierre 28-09, 19:56 UTC:** los cinco envíos están COMPLETE. Fork8 **0.953**, DC030 **0.947**, prune12 **0.953**, prune12_joint **0.951**, joint **0.951**, todos por debajo de c3 **0.955**. Las mejoras locales no se trasladaron al público. No promover estos cambios ni reenviar candidatos. El reminder está apagado. Evidencia: `results/E070/all_submission_status.json` y `public_decision.json`; continuación y cierre de la última tanda: [E071_EXECUTION.es.md](E071_EXECUTION.es.md). Las instrucciones y estados pendientes que siguen son históricos.

> **Resultados publicos confirmados 28-09 12:07 UTC:** fork8 **0.953**, DC030 **0.947**, ambos inferiores a c3 **0.955**. Se activo la rama autorizada y se enviaron poda12 (**56640207**), poda12_joint (**56640212**) y joint (**56640230**), aceptados y PENDING. Quedan **cero cupos hoy**. El seguimiento sigue cada 25 minutos solo para sus resultados. Esta actualizacion sustituye las estimaciones y planes de envio anteriores. Detalle: `results/E070/public_decision.json`.

> **Autorizacion condicional nueva de Arturo (28-09):** esperar los dos resultados iniciales. Si alguno supera el 0.955 publico, ejecutar experimentos complementarios y dejar sus submissions listas; no enviarlas automaticamente en esa rama. Si ambos tienen scores validos y ninguno supera 0.955, enviar una vez F8_prune12, F8_prune12_joint y F8_joint v1. Esta autorizacion sustituye la reserva anterior del quinto cupo en la rama sin mejora. Un ERROR no es un score cero: diagnosticar antes de gastar cupos. Primera consulta realizada a las 05:08 UTC: ambos PENDING; seguimiento activo cada 25 minutos. Politica y hashes: `results/E070/conditional_next_steps.json`. `e070_submit.py` verifica la condicion en Kaggle antes de cada envio; `status-all` consulta los cinco. Cuatro pruebas del control de envios pasan.

> **Envios realizados 28-09 00:04 UTC (27-09 19:04 Lima):** C3_fork8 v1 = **56622884**; F8_dc030 v1 = **56622918**. Ambos aceptados por Kaggle y PENDING en la ultima consulta. Cuota confirmada: dos consumidos, **tres disponibles**. Seguimiento cada 25 minutos exclusivamente para consultar estos resultados; no volver a enviar ni consumir otros cupos automaticamente. Recibos en `results/E070/*_submission_attempt.json`; estado en `first_two_status.json`.

> **Autorizacion posterior de Arturo:** programado el envio de C3_fork8 v1 y F8_dc030 v1 para el 27-09 a las 19:00 Lima / 28-09 00:00 UTC. Solo estos dos candidatos pueden enviarse automaticamente; los otros tres cupos quedan disponibles. Esta instruccion sustituye las notas anteriores de no envio automatico, exclusivamente para estos dos. Recibo: `results/E070/scheduled_first_two.json`. El seguimiento fue reactivado con ejecucion unica y despues revisara sus resultados.

> Estrategia actualizada con Arturo: **2 + hasta 2 + 1 reservado**. Primero C3_fork8 y DC030; esperar sus scores. Después evaluar el envío de prune12 y prune12_joint. Mantener libre el quinto cupo para una combinación validada según los resultados. Joint queda preparado pero sin envío inicial previsto. Ver `results/E070/submission_strategy.json`; esta estrategia prevalece sobre el orden anterior de cinco envíos. No hay envío automático.

Cierre: 2026-09-27T21:45:29.498446+00:00. Cinco candidatos elegidos entre seis inferencias completas, todas distintas y verificadas. Ningún envío al leaderboard realizado por E070. Reset objetivo: **28-09-2026 00:00 UTC / 27-09 19:00 Lima**.

## Orden propuesto

| Orden | Candidato | Score local | Diferencia local frente a c3 | Evidencia |
|---|---|---:|---:|---|
| 1 | F8_dc030 | 0.928908911 | +0.001395526 | Exploratorio prioritario |
| 2 | C3_fork8 | 0.928395837 | +0.000882452 | Referencia con evidencia local mas amplia |
| 3 | F8_prune12_joint | 0.928569343 | +0.001055958 | Exploratorio combinado |
| 4 | F8_prune12 | 0.928470195 | +0.000956810 | Conservador; incremento exploratorio |
| 5 | F8_joint | 0.928494988 | +0.000981604 | Exploratorio de baja prioridad |

La cola ejecutable está en `results/E070/submission_queue.json`: kernel y versión exactos, archivo, hashes de código/CSV, recibos y justificación. El orden prioriza DC030 y luego mide fork8, el cambio común a toda la tanda. Las inferencias ya están ejecutadas; no hace falta relanzarlas para enviar su `submission.csv` v1.

## Qué encontramos

Los dos laboratorios CPU terminaron: ocho variantes y dos controles, estos últimos repetidos en ambos laboratorios. Cada configuración cubre 199 videos. Los controles coinciden por video entre laboratorios y con la referencia anterior. La caché solo reutiliza grafos finales idénticos, incluidas coordenadas redondeadas y enlaces.

DC030 sube de 0.928395837 (fork8) a 0.928908911 local: +0.000513074. Mejora en ambos grupos, con 36 videos que ganan y siete que pierden. Sus divisiones TP/FP/FN cambian de 24/49/127 a 23/33/128: elimina 16 falsas y pierde una verdadera. Su intervalo bootstrap incremental del 90% es [-0.000663242, +0.001417651], así que es una hipótesis prometedora, todavía incierta.

La descomposición de la métrica atribuye +0.000500000 de esa ganancia a divisiones y solo +0.000013074 a enlaces. DC030 mejora principalmente las decisiones de división; no demuestra una mejora amplia del seguimiento. Joint también depende de divisiones: +0.000121212 en ese componente y -0.000022061 en enlaces. Poda 12 aporta +0.000074358 en enlaces y no cambia las divisiones. Esta diferencia justifica conservar la combinación, además de las variantes individuales. Evidencia: `results/E070/increment_components.json`.

Fork8 tiene una señal local más extendida frente a c3: 100 videos mejoran y tres empeoran; conserva las 24 divisiones verdaderas y reduce falsas de 62 a 49. Poda 12 agrega una ganancia pequeña, con 84 mejoras y cuatro pérdidas sobre fork8. La combinación con joint queda segunda por score local, pero su intervalo incremental también cruza cero.

Joint solo mejora un video y empeora tres frente a fork8; queda último en prioridad. En el test visible cambia coordenadas de 125 nodos, sin cambiar enlaces. La combinación con poda 12 permite probar ese mecanismo con un filtrado distinto. No son cinco modelos independientes: comparten la misma base y sus errores estarán correlacionados.

## Descartes y reserva

- P8 está ejecutado y verificado como sexta reserva. Mejora solo +0.000039410 sobre fork8; 23 videos mejoran y 99 empeoran. Al retirar los cinco mayores aportes, el incremento cae a -0.000302281. DC030 ocupa su plaza.
- Curvatura: -0.000007659 frente a fork8. No promover.
- Corrección de posiciones atípicas: -0.000162693 frente a fork8; intervalo incremental completamente negativo. No promover.
- Corte final 0.4: mejora agregada mínima, con 175 videos que empeoran y 14 que mejoran frente a fork8; tampoco merece otra inferencia.

## Validación y límites

Los seis CSV tienen hashes diferentes. Los cinco seleccionados pasan manifiesto, identidad del notebook congelado, hash de fuente, validación estructural y límites físicos del CSV, conteos por video y GATE del log. Las métricas corrieron en CPU; las inferencias en GPU, con máximo dos sesiones simultáneas.

Estos 199 videos se reutilizaron para desarrollo y selección. Los intervalos y las pruebas retirando videos son diagnósticos, no validación independiente ni probabilidades de subir en Kaggle. El **0.955 público corresponde a c3**; ningún score E070 local debe sumarse a él para prometer un resultado público.

Antes de enviar en una acción autorizada, comprobar cuota y envíos anteriores, y usar kernel/versión/archivo de la cola. No enviar hashes duplicados. El heartbeat de preparación quedó desactivado (PAUSED) al cerrar la cola; recibo en `results/E070/automation_paused.json`. No se envió automáticamente al leaderboard.

Evidencia: `results/E070/ranking.json`, `verified_inventory.json`, `submission_queue.json`, recibos `*_verified.json`, `lab_*_completed.json` y `outputs/e070/lab_*/e070_*/robustness.json`.
