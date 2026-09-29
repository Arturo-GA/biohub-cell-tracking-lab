# CPU4: resultados corregidos y promoción — 26-09-2026

Kaggle `jarturo/biohub-exact-replay-cpu4` v1 terminó COMPLETE. Nueve configuraciones con 199/199 videos cada una; cero discrepancias de reproducción de safe-division. Las 12 comparaciones de suavizado/redondeo contra la función original dieron cero diferencias. Tiempo de la celda del laboratorio: 4452 segundos (~74 minutos), sin GPU.

| Candidato | Score local | Delta frente a base | Divisiones TP/FP/FN |
|---|---:|---:|---|
| Base x138 L7 | 0.926257 | — | 24/65/127 |
| B: tau 0.9 + poda E + cortes/fork | **0.931597** | **+0.005340** | 33/66/118 |
| A: tau 0.9 + P8 + cortes/fork | **0.931573** | **+0.005317** | 33/66/118 |
| Tau 0.95 + P8 | 0.931441 | +0.005184 | 33/68/118 |
| E: tau 1.0 + P8 | 0.931142 | +0.004885 | 33/72/118 |
| Tau 0.85 + P8 | 0.930851 | +0.004594 | 31/63/120 |
| C: tau 0.9 + P8 sin fork | 0.930246 | +0.003990 | 33/84/118 |
| D: P8 conservador | 0.928435 | +0.002179 | 24/49/127 |
| F: E conservador | 0.928396 | +0.002139 | 24/49/127 |

A y B mejoran ambos grupos: 44b6 +0.001288; 6bba +0.006180/+0.006209. Al retirar los cinco videos que más contribuyen quedan +0.003082/+0.003119. Las tres mezclas de grupos también son positivas. El rango tau 0.85–0.95 mantiene ganancia; 0.9 vence a 1.0 con las mismas divisiones verdaderas y seis falsas menos. Eliminar la poda de bifurcaciones empeora el resultado (84 FP frente a 66).

La corrección del replay reduce algo la estimación anterior, pero no elimina la ganancia. Los datos son reutilizados para desarrollo y los raw capturados son float32; no representan un test independiente. No sumar estos deltas al público 0.954 ni interpretar el bootstrap como probabilidad de mejora.

## Decisión ejecutada

Se promovieron A y B a inferencia GPU privada. Ambas subidas aceptadas a las ~18:44 UTC del 26-09:

- `jarturo/biohub-exact-a`, **v1**, kernelId 136009679. Notebook SHA256 `5a091d89c9b2168fe12b00e22f7c719f60ea6feafcc981bd1189bd9a3d432eba`.
- `jarturo/biohub-exact-b`, **v1**, kernelId 136009686. Notebook SHA256 `a66880a46d83cc193fb77bf9d1a715fa618b576d9d192d4b43759c6af7204fe7`.

Se mantienen L7, una sola pasada y deadline 34200; validator desactivado, manifiesto del código y parámetros al terminar. A usa una poda uniforme; B la específica de grupo. Su diferencia local es demasiado pequeña para afirmar superioridad estadística, por lo que las dos variantes permiten comprobar transferencia pública. No se lanzan C ni un nuevo E: el laboratorio no los favorece. D queda como opción conservadora, sin ocupar todavía otra sesión GPU.

Todavía no se ha enviado una nueva submission. Los cinco cupos del 26 estaban consumidos; la siguiente ventana prevista es 27-09 00:00 UTC / 26-09 19:00 Lima. Antes del envío: comprobar COMPLETE, manifiesto, versión v1, integridad del CSV y logs sin fallback o deadline. No usar `submit5.sh`/`batch.sh` antiguos.

Resultados reproducibles: `results/E068/cpu4_exact_results.json`, `cpu4_exact_rows.csv`, `cpu4_exact_robustness.json` y `cpu4_ab_push_receipts.json`.
