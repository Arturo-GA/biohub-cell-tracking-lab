# CPU7 y C3_fork8 — resultados comprobados el 27-09-2026

Ambos kernels terminaron correctamente. Consulta de estado: 18:54 UTC. CPU7 evaluó nueve configuraciones sobre 199 videos; el laboratorio tardó 4.665 segundos (77,8 minutos) en CPU. C3_fork8 produjo el archivo de inferencia previsto.

## Comparación local

| Variante | Score local | Diferencia frente a c3 | Videos mejor / peor |
|---|---:|---:|---:|
| C3_public0955 | 0.927513 | referencia | 0 / 0 |
| C3_fork8 | **0.928396** | **+0.000882** | **100 / 3** |
| C3_fork8_boundary | 0.928396 | +0.000882 | 100 / 3 |
| C3_cut06_fork8 | 0.928200 | +0.000687 | 155 / 44 |
| C3_fork6 | 0.927948 | +0.000435 | 82 / 2 |
| C3_prune12 | 0.927588 | +0.000074 | 84 / 4 |
| C3_prune14 | 0.927569 | +0.000056 | 97 / 17 |
| C3_prune10_uniform | 0.927487 | -0.000026 | 73 / 96 |
| C3_cut06 | 0.927318 | -0.000195 | 154 / 45 |

Fork8 confirma exactamente el resultado ya observado en CPU4; no constituye una nueva evaluación independiente. El score público de 0.955 pertenece a c3, no a este nuevo candidato.

Fork8 reduce las divisiones falsas de 62 a 49, mantiene 24 divisiones correctas y 127 omitidas. Mejora en ambos grupos de videos (+0.000628 en 44b6 y +0.000929 en 6bba). Al retirar los cinco videos con mayor contribución favorable, el incremento agregado sigue siendo +0.000505. Estos diagnósticos proceden de datos reutilizados para entrenamiento y ajuste: no permiten pronosticar 0.956 público ni calcular una probabilidad de mejora.

La variante `boundary` tiene exactamente los mismos resultados y hashes de nodos, enlaces y coordenadas en los 199 videos. No merece otro envío. Los cortes a 0.6 empeoran frente a fork8; la poda más agresiva aporta ganancias pequeñas y frágiles. No promover el resto de esta rejilla a nuevos notebooks GPU.

## Verificaciones realizadas

- Reproducción de safe-division: cero discrepancias en 199 videos.
- Suavizado y redondeo: 18 comprobaciones de paridad, cero discrepancias.
- La referencia c3 reproduce 0.9275133844979068.
- Notebook local CPU7 coincide con el SHA256 del recibo de lanzamiento.
- C3_fork8 coincide con el notebook y manifiesto congelados al lanzar v1.
- CSV: **232.347 filas**, cuatro videos, coordenadas y estructura del grafo válidas; conteos del log coinciden exactamente con el CSV. `GATE PASS`.
- Validador interno desactivado; el archivo final usa una única configuración fija. La frase heredada «public 0.939 configuration» del log no es un nuevo resultado: la configuración efectiva consta en `RUN_MANIFEST`.

## Próximo envío

Prioridad única de esta tanda: [jarturo/biohub-c3-fork8](https://www.kaggle.com/code/jarturo/biohub-c3-fork8), **versión 1**, archivo `submission.csv`. Está listo, **todavía no enviado al leaderboard**.

La API consultada a las 18:56 UTC confirma cinco submissions con fecha UTC 27-09, todas COMPLETE; c3 sigue siendo la mejor con 0.955. Los cupos se renuevan el **28-09 a las 00:00 UTC = 27-09 a las 19:00 Lima**. No se hizo un intento condenado por el límite ni se creó un envío automático.

La siguiente investigación prioritaria sigue siendo el linker independiente de hengck23 sobre detecciones de c3, descrito en [la revisión pública](E068_PUBLIC_REVIEW_2026-09-27_PM.es.md). Esta sesión revisó y dejó listo el candidato existente; no entrenó ni lanzó ese linker.

## Artefactos

- `results/E068/cpu7_results.json`, `cpu7_rows.csv`, `cpu7_robustness.json`, `cpu7_graph_diagnostics.json` y `cpu7_completed.json`.
- `results/E068/c3_fork8_verified.json`: hashes, conteos y verificaciones del candidato.
- `results/E068/submission_readiness_2026-09-27_pm.json`: comprobación de cupos y submissions.
- `results/E068/next_submission_candidate.json`: candidato preparado, sin programación automática.
- `outputs/e068_review/cpu7/` y `outputs/e068_review/final_c3_fork8/`: descargas y logs.
