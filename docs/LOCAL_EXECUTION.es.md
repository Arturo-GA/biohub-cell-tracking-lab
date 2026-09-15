# Ejecución local y Kaggle CPU

Implementación del 15 de septiembre de 2026. El entorno aislado `C:\dev\biohub\.venv` tiene **PyTorch 2.5.1+cu121** y ejecuta operaciones en la RTX 3050 con el driver existente. No se modificó RSNA ni el driver. Se eligió la distribución oficial CUDA 12.1 y se comprobó su funcionamiento en este equipo. [Distribuciones oficiales de PyTorch](https://pytorch.org/get-started/previous-versions/), [compatibilidad de CUDA](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

## Comprobaciones completadas

- **96 pruebas distintas** pasaron en el entorno nuevo: batería de 93 y tres comprobaciones adicionales de política CPU e importación de paquetes; después se repitieron las cinco de entrenamiento portable por una protección adicional. Cubren equivalencia de atención por bloques, gradientes con recomputación de activaciones, carga por video, detección de cambios de datos y corrección del CSV.
- La prueba de capacidad ejecutó **100 pasos sobre una ventana real de 17.300 centros**, con características y etiquetas sintéticas: mediana **0,704 s/paso**, máximo **300,8 MiB asignados y 368 MiB reservados por PyTorch**. El proceso llegó a unos **4,0 GiB de RAM**. Es memoria del asignador de PyTorch, no toda la ocupación del dispositivo, que también incluye contexto y escritorio.
- El codificador por bloques procesó un grafo completo de **165.029 centros** en **9,06 segundos**, manteniendo estados entre capas en CPU. El proceso llegó a unos **4,3 GiB de RAM**. Esta cifra no incluye puntuar todas las parejas ni ejecutar el solver.
- En una prueba CUDA determinista, pausar al paso 2 y continuar hasta el 4 reprodujo exactamente pesos, optimizador, scaler y muestras de la ejecución continua. La ejecución normal puede tener diferencias numéricas propias de CUDA; el checkpoint conserva todo su estado reanudable.
- La primera ejecución con imágenes reales completó CELLECT, Harmonic, el grafo y las características de los dos UNet en la RTX 3050. Se corrigió una ruta fija de registro de Kaggle mediante una sustitución comprobada de una sola constante al cargar el módulo, manteniendo intacto su archivo y sus instrucciones de inferencia. Pasaron **dos pruebas adicionales** de esa adaptación: **98 pruebas distintas acumuladas**, sin repetir toda la batería previa.

Recibos: [capacidad](../results/LOCAL_CUDA_BENCHMARK.json), [reanudación CUDA](../results/LOCAL_CUDA_RESUME_CHECK.json), [paquetes CPU](../results/CPU_WORKFLOW_PREFLIGHT.json), [comprobaciones de implementación](../results/LOCAL_IMPLEMENTATION_CHECKS.json). La extracción secuencial también reprodujo exactamente las características de ambos detectores juntos usando los pesos públicos reales sobre imágenes sintéticas. [Comprobación](../results/LOCAL_FEATURE_EXTRACTION_CHECK.json). Estas pruebas no entrenaron un candidato científico ni produjeron una estimación de score. No se extrapola el tiempo por paso al tiempo total: faltan carga de datos, características, calibración y selección.

## Qué cambia en el programa

`event_local_train.py` usa una caché compartida de un video para ajuste/calibración; el muestreo conserva la misma distribución y no depende de la caché. `event_portable.py` permite arrays NPY con lectura mapeada y verifica hashes al iniciar. El entrenamiento mantiene la arquitectura, las ventanas de diez fotogramas, los 6.000 pasos y la partición 48/16/48.

`EventGraphNet.encode_streamed` conserva todos los vecinos de cada capa. Divide filas de consultas y mantiene estados completos en CPU, sin cortar el contexto temporal. Los parámetros y el formato de pesos del modelo no cambian. La recomputación de activaciones y el tamaño de bloque son opciones de ejecución.

`resume.pt` guarda pesos, optimizador, scheduler, scaler, paso, estados aleatorios, muestreador, historial e identidad de datos/configuración. Se escribe de forma atómica, cada 100 pasos o aproximadamente cinco minutos, con copia anterior. `best.pt` se selecciona solo por calibración. Los pesos antiguos de E013 no contienen todo el estado requerido para reanudar; se conservan y no se sobrescriben al iniciar un entrenamiento nuevo.

## Entrada por video

La versión 1 del notebook privado [CPU Image Preparation](https://www.kaggle.com/code/jarturo/biohub-lab-cpu-image-preparation) terminó correctamente, con GPU y TPU desactivadas. Empaquetó el primer video de ajuste (`44b6_144b256d`, unos 544 MB) y calculó sus propuestas gaussianas en **304,13 segundos**, sin leer GEFF. Es el primer bloque de la preparación de los 64 videos de ajuste/calibración; no sustituye la partición por un experimento de un video. Se consultó su estado tras el aviso de Arturo, sin monitor automático.

Las salidas recuperadas se guardan en `outputs/cpu_finished_check/prepare/local_inputs`. Desde la raíz del proyecto:

```powershell
.\.venv\Scripts\python.exe scripts/run_local_prepared_video.py --cpu-package outputs/cpu_finished_check/prepare/local_inputs
```

El importador verifica el ZIP y cada archivo, rechaza rutas fuera del destino y conserva una reserva de 10 GiB de disco y una caché de imágenes de hasta 6 GiB. El ejecutor verifica los pesos públicos y el código del detector, reutiliza las gaussianas CPU y completa las detecciones/características que falten. Extrae las características de cada UNet por separado. La detección combinada carga ambos modelos y ya completó este primer video real; queda por comprobar la capacidad con otros tamaños. Si falla una etapa, los productos anteriores con recibo válido se conservan.

Se verificaron los **102 archivos** importados del video y el hash de sus gaussianas. La laptop completó sus **100 fotogramas** de tamaño `64 × 256 × 256`: **107.746 centros**, características finitas de **64 canales** y **25 vecinos** por centro. La comprobación posterior reprodujo todos los vecinos y la concatenación de características; también verificó los límites espaciales y los hashes finales. [Preparación CPU verificada](../results/CPU_IMAGE_PREPARE_v1_completed.json), [resultado local verificado](../results/LOCAL_REAL_VIDEO_PREPARATION.json).

El primer intento completó CELLECT y se detuvo al escribir el registro de Harmonic en `/kaggle/working`. Tras redirigir solo ese registro, la reanudación reutilizó CELLECT y las gaussianas. El proceso exitoso restante tardó **260,89 segundos**, con máximos de PyTorch de **693,3 MiB asignados y 884 MiB reservados**, y unos **5,18 GiB de memoria de proceso**. Estas mediciones excluyen la etapa CELLECT del intento anterior, la preparación CPU y la descarga; tampoco incluyen toda la memoria del dispositivo.

Hay **2 de 112 juegos completos de grafos/características** y están copiados y verificados los **48 grafos de E012** en `outputs/local_event_graph/videos`. No se ha iniciado el entrenamiento científico del selector local. La consulta única de continuación del 15 de septiembre a las 19:58 UTC todavía mostró el candidato E013 antiguo como `RUNNING`; sus resultados recuperables se incorporarán cuando estén disponibles, evitando repetir etapas completas.

El siguiente bloque se construye con `scripts/build_cpu_workflow.py --offset 1`, y así sucesivamente hasta 63. Cada envío usa un recibo nuevo con `scripts/launch_cpu_notebook.py`; un recibo existente impide repetir accidentalmente el lanzamiento. Se avanza cuando el bloque anterior está guardado y verificado, sin crear monitores.

El bloque de `offset 1`, video `44b6_1d530831`, terminó como **versión 2**. Se recuperaron y verificaron sus 102 archivos de imagen, unos 518 MB, y 4.533 propuestas gaussianas. La preparación CPU tardó **301,93 segundos**. [Resultado verificado](../results/CPU_IMAGE_PREPARE_v2_completed.json).

La laptop completó ese segundo video sin errores: **92.396 centros**, con 64 características finitas y 25 vecinos por centro. CELLECT, Harmonic, construcción del grafo y extracción visual terminaron en **352,67 segundos**. Se reutilizaron las gaussianas CPU; la descarga y su cálculo no forman parte de ese tiempo. Los máximos registrados por PyTorch fueron **1,32 GiB asignados y 1,50 GiB reservados**; la memoria máxima del proceso fue unos **5,15 GiB**. Se verificaron hashes, límites espaciales, concatenación visual y todos los vecinos. [Resultado local del segundo video](../results/LOCAL_REAL_VIDEO_02.json).

## Preparación CPU por lotes

La **versión 3** fue aceptada el 15 de septiembre a las **20:05 UTC**, privada y con GPU/TPU desactivadas. Prepara hasta cuatro videos desde `offset 2`, con un presupuesto total de **4 GiB de archivos de imagen**. El lote conserva un prefijo consecutivo de la lista: si el siguiente video excede el presupuesto, queda para el lote posterior. No se cambia la partición ni se omiten videos del experimento. [Lanzamiento](../results/CPU_IMAGE_PREPARE_v3_launch.json).

Cada video conserva su carpeta, archivo ZIP, gaussianas y recibos. `batch_result.json` registra los videos seleccionados, los completados y el siguiente índice. La laptop los procesa secuencialmente mediante:

```powershell
.\.venv\Scripts\python.exe scripts/run_local_prepared_batch.py --cpu-root outputs/cpu_prepare_v3/local_inputs
```

El importador mantiene sus límites de caché y reserva de disco. Este comando se usa después de descargar y verificar las salidas, cuando el lote haya terminado; no consulta Kaggle ni espera a notebooks remotos. Se puede reconstruir un lote con `scripts/build_cpu_workflow.py --offset 2 --count 4 --prepare-only`. La opción `--prepare-only` conserva intacto el notebook del control ya evaluado. No se ha consultado automáticamente el estado de la versión 3.

Pasaron **tres pruebas de regresión** del lote: presupuesto y orden consecutivo, rechazo de índices/entradas inválidos y rechazo de paquetes hijos incompletos. Los dos paquetes CPU compilan y se importan, y se comprobó que el notebook del control conserva exactamente su hash de lanzamiento. Son **101 pruebas distintas acumuladas** con las anteriores, sin repetir toda la batería histórica. [Verificación del lote](../results/CPU_BATCH_PREFLIGHT.json).

## Entrenamiento y evaluación por etapas

El ejecutor portable es `scripts/event_stages.py`. Admite `inventory`, `freeze`, `labels`, `pack`, `train`, `calibrate`, `score`, `solve`, `csv` y `evaluate`. Las etapas CUDA se rechazan cuando detectan Kaggle; este flujo tampoco se ejecuta en Colab. El lanzador de notebooks admite exclusivamente metadatos privados con GPU/TPU desactivadas.

Después de completar y congelar los 112 juegos de grafos/características, preparar etiquetas de ajuste y calibración en CPU. El ejecutor impide leer etiquetas antes del congelado y no genera etiquetas de evaluación. Después:

```powershell
.\.venv\Scripts\python.exe scripts/event_stages.py pack --root outputs/local_event_graph
.\.venv\Scripts\python.exe scripts/event_stages.py train --root outputs/local_event_graph --device cuda
```

Para continuar desde el último estado completo:

```powershell
.\.venv\Scripts\python.exe scripts/event_stages.py train --root outputs/local_event_graph --device cuda --resume
```

`--stop-after` limita el paso absoluto de esta sesión sin cambiar el presupuesto del scheduler. `--time-budget` permite sesiones de duración acotada. Crear `outputs/local_event_graph/STOP` solicita una pausa al terminar el paso; retirar ese archivo antes de continuar. Tras una interrupción abrupta, se retoma el último checkpoint íntegro.

`calibrate` y `score --role evaluation` usan el codificador por bloques. Las tablas compactas de puntuaciones, grafos, checkpoint y calibración permiten ejecutar `solve --role evaluation`, `csv --role evaluation` y `evaluate` en Kaggle CPU. La entrada de evaluación debe contener exactamente los 48 videos. Se fija el CSV antes de calcular la métrica oficial; no hay envío automático al leaderboard. La viabilidad temporal de una submission final exclusivamente CPU sigue pendiente de medir.

## Control recuperado

Se reprodujo el suavizado del punto problemático a partir de sus predicciones originales: el redondeo posterior dejó una coordenada Z fuera del volumen. La exportación corregida aplica la proyección al entero válido más cercano solo a ese caso de borde; rechaza otras excursiones. Conserva **1.072.577 nodos y 1.031.579 enlaces**, y el CSV original permanece intacto. La validación completa pasó. [Recibo de corrección](../results/E013_control_export_repair.json).

Kaggle rechazó adjuntar directamente las salidas del notebook fallido. Arturo autorizó expresamente la subida del CSV recuperado; se creó el dataset [Control CPU Cache](https://www.kaggle.com/datasets/jarturo/biohub-lab-control-cpu-cache), versión 1, y la API confirmó que es privado y contiene el archivo completo. El notebook exige su hash original antes de corregirlo y evaluarlo. [Recibo del dataset](../results/CONTROL_CPU_DATASET.json).

El 15 de septiembre de 2026 a las 19:23 UTC, Kaggle aceptó la **versión 2** de [CPU Control Recovery](https://www.kaggle.com/code/jarturo/biohub-lab-cpu-control-recovery), con GPU y TPU desactivadas. [Recibo del lanzamiento](../results/E013_cpu_control_launch.json). La confirmación específica resolvió el rechazo previo de permisos.

Tras el aviso de Arturo, la API confirmó **COMPLETE**. Se descargó el CSV y se verificó su identidad exacta con la corrección local (`503b20ac…9aa366`), los 20 archivos fuente del paquete y la cobertura íntegra de los 48 videos. El agregado se recalculó a partir de las métricas por video. La evaluación tardó **130,63 segundos**, con score **0,909257**, Jaccard de enlaces ajustado **0,903802** y Jaccard de divisiones **0,054545**: 3 correctas, 22 falsas y 30 omitidas. [Resultado y verificaciones](../results/E013_cpu_control_completed.json).

Este valor es una referencia de desarrollo condicional para E013; no se compara directamente con el **0,946** público propio ni con el **0,947** comunicado para Harmonic, porque pertenecen a conjuntos distintos. Todavía no demuestra una mejora del candidato. No hubo monitor ni envío al leaderboard. La comprobación se reproduce con `scripts/audit_cpu_control.py`; las métricas detalladas y el CSV permanecen en `outputs/cpu_finished_check/control` fuera de Git.
