# E069 — linker independiente y duplicados persistentes

> **Completado el 27-09-2026, verificado a las 20:15 UTC.** Los tres notebooks CPU terminaron; se ejecutaron seis configuraciones sobre 199 videos y once sobre un panel fijo de 32. Resultados y decisión: [E069_RESULTS.es.md](E069_RESULTS.es.md). No repetir lanzamientos ni transferencias de la secuencia histórica de abajo. E069 no aporta un nuevo candidato prioritario; C3_fork8 v1 permanece listo para el reset y no se programó un envío automático.

## Protocolo

Dos líneas distintas sobre las detecciones de c3 (mejor público confirmado: 0.955):

1. **Duplicados persistentes, 199 videos, CPU.** Suprimir solamente un componente lineal débil cubierto durante varios fotogramas por otro componente más confiable, con proximidad y movimiento compatibles. Se protegen todos los componentes con divisiones. Comparaciones: c3, duplicados estrictos, duplicados moderados, fork8, fork8 + cada regla de duplicados.
2. **Linker independiente de hengck23, 32 videos.** Conservar detecciones y grados del grafo. Comparar swaps geométricos con swaps respaldados por logits neuronales, bajo los mismos candidatos. Proteger bifurcaciones y contextos incompletos. Usar consenso de ambas direcciones y margen de logits; no promediar probabilidades incompatibles.

El panel del linker se fija por hash dentro de dos grupos de videos y cuatro estratos de cantidad de nodos, cuatro videos por estrato. Se divide de antemano en 16 de desarrollo y 16 de confirmación. No se elige por score ni por éxitos publicados del donante. Como el detector base fue entrenado con estos videos, confirmación **no significa** validación independiente del detector ni garantiza mejora pública.

## Implementación y comprobaciones

- `kaggle/x138_xr/independent_rules.py`: duplicados y swaps, sin acceso a anotaciones.
- `independent_linker.py`: extracción `[d0,d1,e2]`, coordenadas `(z,y/4,x/4)`, carga estricta del checkpoint con `weights_only=True` y solo la clase `DotDict` auditada permitida.
- `hengck_model_v12.py`: definiciones del modelo público, con atribución; sin ejecución de la demo.
- `build_e069.py` y `e069_runner.py`: notebooks privados con fuentes, configuración y manifiestos fijados.
- `run_e069_local.py`: inferencia en GPU local, sin cargar anotaciones de los bundles.
- `run_e069_local_joins.py`: segunda comparación que reconecta fragmentos con contexto de movimiento y margen neuronal en ambas direcciones, incluyendo la opción de no conectar. Se añadió antes de evaluar la métrica al observar pocos intercambios propuestos. Informa explícitamente los candidatos sin imágenes disponibles en la exportación; el control geométrico usa exactamente el mismo subconjunto.
- `run_e069_local_forks.py`: veto independiente de una rama heurística cuando su logit es menor que -2 y la rama conservada es la favorita del modelo con logit mayor que 2. No modifica bifurcaciones con ambos enlaces respaldados por el modelo base ni elimina detecciones. Se mide por separado y combinado con fork8. Los umbrales se fijaron antes de evaluar estos resultados con anotaciones.
- `e069_ops.py`: operaciones explícitas de consulta, descarga y transferencia de logits; no programa tareas ni envía submissions.
- Diez pruebas unitarias pasan: preservación de grados, protección de divisiones, contexto temporal, duplicados, confianza/movimiento, indexación de logits, conversión de coordenadas, reconexión de fragmentos y veto de ramas heurísticas.
- Prueba local del modelo: checkpoint completo, logits finitos y GPU RTX 3050 de 4 GB. Es una prueba técnica con imágenes sintéticas, no evidencia de score.

Checkpoint: `ecd8869de9cf405c1a93a563b4810faad5fd2378f57341e18c4e6b5a87201552`.

## Ejecución y uso de recursos

- Lanzado CPU: `jarturo/biohub-e069-duplicates-cpu`, v1.
- El lanzamiento de `jarturo/biohub-e069-linker-cache` en Kaggle GPU fue **rechazado por capacidad**, no ejecutado. La API indicó dos sesiones GPU ocupadas; aparecían RUNNING `rsna-knee-silver-01-readers` y `rsna-knee-silver-04-both`. No se interrumpieron.
- Alternativa lanzada CPU: `jarturo/biohub-e069-images-cpu`, v1. Exporta únicamente las imágenes reducidas necesarias y grafos de laboratorio de los 32 videos.
- Exportación CPU completada y descargada: 32 bundles verificados en `outputs/e069/image_fast/e069_image_cache`. Recibo: `results/E069/image_cache_completed.json`.
- Inferencia real en GPU de la laptop completada: 235.803 candidatos de intercambio, cuatro aceptados por el criterio estricto y cinco por el moderado. Reconexión de fragmentos: cero candidatos. Veto de ramas: 383 candidatas, 375 con imágenes disponibles y dos aceptadas. No se entrenaron pesos nuevos.
- Logits/candidatos transferidos al dataset **privado** `jarturo/biohub-e069-local-linker-cache`; no se subieron imágenes, anotaciones ni pesos. Recibo: `results/E069/local_dataset_created.json`.
- Evaluación CPU lanzada a las 19:47:52 UTC: `jarturo/biohub-e069-local-linker-cpu` v1, once configuraciones. Recibo: `results/E069/local_eval_launch.json`. Los notebooks de inferencia para test no incluyen anotaciones.
- E069 no consumió GPU de Kaggle. El uso de GPU fue local en la RTX 3050; no se interrumpieron trabajos de RSNA.

## Continuación reproducible

Con Python del SDK (`C:/dev/rsna-knee/.venv/Scripts/python.exe`), ejecutar `kaggle/x138_xr/e069_ops.py check`. Las descargas solo proceden si el kernel está COMPLETE y verifican notebook y manifiesto.

1. `e069_download_images.py`: descarga paralela verificable a `outputs/e069/image_fast/e069_image_cache`, reutilizando archivos de la descarga secuencial inicial.
2. Con Python ML (`C:/dev/biohub/.venv/Scripts/python.exe`): `kaggle/x138_xr/run_e069_local.py outputs/e069/image_fast/e069_image_cache outputs/e069/local_cache`, después `kaggle/x138_xr/run_e069_local_joins.py` y `kaggle/x138_xr/run_e069_local_forks.py`.
3. Con Python SDK: `e069_ops.py upload_local`, comprobar que el dataset se procesa correctamente y `e069_ops.py launch_local_eval`.
4. `e069_ops.py download_duplicates` y `e069_ops.py download_evaluation` cuando completen.

Antes de promover un candidato, exigir reproducción exacta de c3, operaciones efectivas, ganancia en la métrica completa y un balance razonable entre errores corregidos y añadidos. Para el linker, mirar también los dos subconjuntos y comparar contra geometría sola. No interpretar bootstrap como probabilidad de mejora pública. No enviar variantes duplicadas o inactivas solo para llenar los cinco cupos.

## Diagnósticos adicionales

- `results/E069/duplicate_candidate_diagnostic.json`: en el panel hay 21.354 componentes lineales de al menos ocho nodos, 16.586 de longitud 8–40, y 4.272 de estos con probabilidad media ≤0,75. Existen 254 pares con proximidad ocasional de 3 µm, pero **ninguno** mantiene esa proximidad durante ocho fotogramas. La falta de operaciones de duplicados no proviene simplemente de un umbral de confianza demasiado estricto.
- `results/E069/swap_gt_coverage_diagnostic.json` y `fork_nearest_gt_diagnostic.json`: solo uno de los intercambios neuronales aceptados queda dentro de la cobertura de las anotaciones a 7 µm; ambos vetos quedan fuera. Las distancias se calcularon después del suavizado y redondeo de c3. Este diagnóstico no reemplaza la métrica oficial ni determina por sí solo si una arista es correcta.
- El intercambio cubierto pertenece a `6bba_d2b9fc0c`, que también concentraba la ganancia de intercambios estrictos de CPU6. Comprobar sus métricas exactas antes de atribuir una mejora nueva al linker.
- `summarize_e069.py` exige los dos recibos de finalización verificados, las seis/once configuraciones completas y todos los videos. Compara cada combinación con fork8 directamente para no atribuir al nuevo modelo la ganancia ya conocida de fork8.
