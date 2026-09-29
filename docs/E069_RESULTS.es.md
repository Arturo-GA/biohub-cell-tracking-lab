# E069 — resultados y decisión

El modelo independiente de hengck23 se ejecutó sobre imágenes reales en la RTX 3050 de la laptop. La evaluación oficial posterior se ejecutó en Kaggle CPU. No se entrenaron nuevos pesos ni se consumió GPU de Kaggle en E069. El intento inicial de lanzar una sesión GPU fue rechazado por capacidad y se sustituyó por exportación CPU, inferencia local y evaluación CPU.

## Linker: once configuraciones, panel fijo de 32 videos

Notebook privado `jarturo/biohub-e069-local-linker-cpu`, versión 1, terminado y descargado con manifiesto verificado. El control reproduce exactamente las métricas de c3 en los 32 videos. El tiempo de evaluación fue 615 segundos, además de la preparación y reproducción de los grafos.

| Variante | Score local del panel | Incremento sobre su control | Videos mejor / peor |
|---|---:|---:|---:|
| c3 | 0.944102596 | Referencia | — |
| Intercambios geométricos | 0.944263148 | +0.000160552 sobre c3 | 1 / 0 |
| Linker estricto | 0.944263148 | +0.000160552 sobre c3 | 1 / 0 |
| Linker moderado | 0.944263148 | +0.000160552 sobre c3 | 1 / 0 |
| Reconexión geométrica | 0.944102596 | 0 | 0 / 0 |
| Reconexión neuronal | 0.944102596 | 0 | 0 / 0 |
| Veto neuronal de ramas heurísticas | 0.944102596 | 0 | 0 / 0 |
| fork8 | 0.944452475 | +0.000349879 sobre c3 | 14 / 0 |
| fork8 + linker estricto | 0.944613058 | +0.000160582 sobre **fork8** | 1 / 0 |
| fork8 + linker + reconexión | 0.944613058 | +0.000160582 sobre **fork8** | 1 / 0 |
| fork8 + veto neuronal | 0.944452475 | 0 sobre **fork8** | 0 / 0 |

El panel contiene 16 videos de desarrollo y 16 de confirmación, fijados por hash antes de medir los resultados. El incremento propio del linker en confirmación fue cero. Como el detector base fue entrenado con estos datos, el panel no es una validación independiente del detector.

La ganancia de los intercambios se concentra en `6bba_d2b9fc0c`. Sus métricas pasan de 917/5/4 a 918/3/3 enlaces TP/FP/FN. Es exactamente el resultado que ya obtenía CPU6 con intercambios geométricos estrictos; no es una corrección nueva descubierta por el segundo modelo. La igualdad de métricas no implica que todos sus grafos sean idénticos.

Los criterios neuronales aceptaron cuatro intercambios estrictos y cinco moderados de 235.803 candidatos. Solo un intercambio aceptado queda en la región anotada a 7 µm. No hubo candidatos de reconexión; de 383 ramas heurísticas, 375 tenían imágenes disponibles y dos fueron vetadas, ambas fuera de la cobertura anotada y sin efecto en el score. La falta de ganancia medible aquí no demuestra que las correcciones fuera de anotaciones sean incorrectas, pero tampoco proporciona evidencia para promoverlas.

## Duplicados: 199 videos

Notebook `jarturo/biohub-e069-duplicates-cpu`, versión 1, completado y descargado con manifiesto verificado. Las seis configuraciones cubren los 199 videos; la fase de comparación duró 3.351 segundos, además de la preparación. Ambas reglas de duplicados, tanto solas como combinadas con fork8, aplicaron **cero eliminaciones** y reprodujeron las métricas de su control.

| Variante | Score local, 199 videos | Incremento sobre su control |
|---|---:|---:|
| c3 | 0.927513384 | Referencia |
| Duplicados estrictos | 0.927513384 | 0 |
| Duplicados moderados | 0.927513384 | 0 |
| fork8 | 0.928395837 | +0.000882452 sobre c3 |
| fork8 + duplicados estrictos | 0.928395837 | 0 sobre fork8 |
| fork8 + duplicados moderados | 0.928395837 | 0 sobre fork8 |

Fork8 vuelve a reproducir el resultado anterior: 100 videos mejoran y tres empeoran; 24 divisiones verdaderas conservadas, 49 falsas frente a 62 del control, y 127 omitidas en ambos. **No comparar directamente** estos scores con los de la tabla de 32 videos: corresponden a poblaciones distintas.

El diagnóstico adicional del panel encuentra 254 pares de componentes con proximidad ocasional a 3 µm, pero ninguno la mantiene ocho fotogramas. Relajar umbrales para forzar operaciones no queda justificado por ese resultado.

## Decisión para las submissions

- Mantener **C3_fork8 v1** como candidato prioritario, ya ejecutado y verificado. El CSV tiene 232.347 filas y pasó las verificaciones de grafo, coordenadas, conteos y código congelado.
- No promover el linker ni su combinación a una nueva submission por ahora: la mejora adicional es pequeña, depende de un solo video y ya existía con geometría. Agrega otra red y tiempo de inferencia sin demostrar una corrección adicional.
- No preparar submissions de reconexión o veto: las primeras no actúan y el segundo no presenta mejora evaluada. No llenar cinco cupos con variantes inactivas o métricamente redundantes.
- El mejor score público confirmado sigue siendo **0.955 de c3**. Ninguno de los scores de las tablas anteriores es un score público ni debe sumarse a 0.955.

C3_fork8 todavía no se ha enviado al leaderboard. La próxima ventana de cuota registrada es **27-09 a las 19:00 Lima (28-09 00:00 UTC)**. No hay envío automático programado. Antes de enviarlo, comprobar de nuevo las submissions existentes para evitar duplicados y respetar el límite diario.

## Evidencia reproducible

- `results/E069/local_eval_completed.json`: once configuraciones completas y manifiesto verificado.
- `results/E069/duplicates_completed.json`: seis configuraciones completas sobre 199 videos y manifiesto verificado.
- `results/E069/comparison_summary.json`: resultados de los dos laboratorios con referencias correctas.
- `results/E069/local_eval_comparisons.json`: comparación de las combinaciones contra fork8, grupos y subconjuntos.
- `results/E069/prior_cpu6_overlap.json`: igualdad exacta de las métricas con la corrección anterior de CPU6.
- `results/E069/duplicate_candidate_diagnostic.json`: diagnóstico sin etiquetas de candidatos de duplicados.
- `results/E069/code_audit.json`, recibos de lanzamiento y manifiestos congelados: código y pesos utilizados.
- `results/E068/next_submission_candidate.json` y `c3_fork8_verified.json`: candidato listo y sus verificaciones.
- `kaggle/x138_xr/summarize_e069.py`: resumen verificable de ambos laboratorios completos. No envía ni programa submissions.

Diez pruebas de código pasaron con `python tests/test_e069_independent.py`. No se instalaron dependencias adicionales en la laptop.
