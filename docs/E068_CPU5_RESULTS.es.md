# CPU5 terminado: se mantienen A y B — 26-09-2026 22:12 UTC

CPU5 v1 terminó COMPLETE y evaluó siete configuraciones en 199 videos. Cero discrepancias de safe-division y en las 12 comprobaciones de suavizado de referencia. La celda del laboratorio consumió 4117 segundos (~69 minutos), CPU.

Las cuatro variantes de continuidad de ramas hijas no mejoraron: A_pre4, A_post4 y A_pre5 dieron exactamente el score de A (0.9315734571657013); B_pre4 dio el de B (0.9315971466733369). También coinciden las métricas guardadas por video con sus controles respectivos. Esto no demuestra identidad del CSV o ausencia de cambios en enlaces no evaluados; no se capturaron hashes de todos los grafos candidatos. No gastar GPU ni una submission adicional en estas variantes.

A y B v1 siguen COMPLETE, con archivos ya verificados y recibos en `results/E068/ab_verified_outputs.json`. Están listos para enviar, todavía no enviados. Prioridad prevista B y A, por su ganancia corregida; diferencia local entre ambos muy pequeña. Se conservan los restantes cupos para candidatos distintos o correcciones, en vez de llenar la tanda con CPU5.

Próxima ventana prevista: 27-09 00:00 UTC = 26-09 19:00 Lima; al consultar el reloj faltaban 1 h 48 min. No se ha programado una tarea de envío automático.

Resultados archivados en `results/E068/cpu5_results.json`, `cpu5_rows.csv` y `cpu5_robustness.json`.
