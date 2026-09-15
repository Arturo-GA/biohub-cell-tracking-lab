# Ejecución local y Kaggle CPU

Implementación del 15 de septiembre de 2026. El entorno aislado `C:\dev\biohub\.venv` tiene **PyTorch 2.5.1+cu121** y ejecuta operaciones en la RTX 3050 con el driver existente. No se modificó RSNA ni el driver. Se eligió la distribución oficial CUDA 12.1 y se comprobó su funcionamiento en este equipo. [Distribuciones oficiales de PyTorch](https://pytorch.org/get-started/previous-versions/), [compatibilidad de CUDA](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

## Comprobaciones completadas

- **96 pruebas distintas** pasaron en el entorno nuevo: batería de 93 y tres comprobaciones adicionales de política CPU e importación de paquetes; después se repitieron las cinco de entrenamiento portable por una protección adicional. Cubren equivalencia de atención por bloques, gradientes con recomputación de activaciones, carga por video, detección de cambios de datos y corrección del CSV.
- La prueba de capacidad ejecutó **100 pasos sobre una ventana real de 17.300 centros**, con características y etiquetas sintéticas: mediana **0,704 s/paso**, máximo **300,8 MiB asignados y 368 MiB reservados por PyTorch**. El proceso llegó a unos **4,0 GiB de RAM**. Es memoria del asignador de PyTorch, no toda la ocupación del dispositivo, que también incluye contexto y escritorio.
- El codificador por bloques procesó un grafo completo de **165.029 centros** en **9,06 segundos**, manteniendo estados entre capas en CPU. El proceso llegó a unos **4,3 GiB de RAM**. Esta cifra no incluye puntuar todas las parejas ni ejecutar el solver.
- En una prueba CUDA determinista, pausar al paso 2 y continuar hasta el 4 reprodujo exactamente pesos, optimizador, scaler y muestras de la ejecución continua. La ejecución normal puede tener diferencias numéricas propias de CUDA; el checkpoint conserva todo su estado reanudable.

Recibos: [capacidad](../results/LOCAL_CUDA_BENCHMARK.json), [reanudación CUDA](../results/LOCAL_CUDA_RESUME_CHECK.json), [paquetes CPU](../results/CPU_WORKFLOW_PREFLIGHT.json), [comprobaciones de implementación](../results/LOCAL_IMPLEMENTATION_CHECKS.json). La extracción secuencial también reprodujo exactamente las características de ambos detectores juntos usando los pesos públicos reales sobre imágenes sintéticas. [Comprobación](../results/LOCAL_FEATURE_EXTRACTION_CHECK.json). Estas pruebas no entrenaron un candidato científico ni produjeron una estimación de score. No se extrapola el tiempo por paso al tiempo total: faltan carga de datos, características, calibración y selección.

## Qué cambia en el programa

`event_local_train.py` usa una caché compartida de un video para ajuste/calibración; el muestreo conserva la misma distribución y no depende de la caché. `event_portable.py` permite arrays NPY con lectura mapeada y verifica hashes al iniciar. El entrenamiento mantiene la arquitectura, las ventanas de diez fotogramas, los 6.000 pasos y la partición 48/16/48.

`EventGraphNet.encode_streamed` conserva todos los vecinos de cada capa. Divide filas de consultas y mantiene estados completos en CPU, sin cortar el contexto temporal. Los parámetros y el formato de pesos del modelo no cambian. La recomputación de activaciones y el tamaño de bloque son opciones de ejecución.

`resume.pt` guarda pesos, optimizador, scheduler, scaler, paso, estados aleatorios, muestreador, historial e identidad de datos/configuración. Se escribe de forma atómica, cada 100 pasos o aproximadamente cinco minutos, con copia anterior. `best.pt` se selecciona solo por calibración. Los pesos antiguos de E013 no contienen todo el estado requerido para reanudar; se conservan y no se sobrescriben al iniciar un entrenamiento nuevo.

## Entrada por video

Se inició el notebook privado [CPU Image Preparation](https://www.kaggle.com/code/jarturo/biohub-lab-cpu-image-preparation), con GPU y TPU desactivadas. Empaqueta el primer video de ajuste y calcula sus propuestas gaussianas. No lee GEFF. Es el primer bloque de la preparación de los 64 videos de ajuste/calibración; no sustituye la partición por un experimento de un video. No se consulta su progreso automáticamente.

Cuando sus salidas estén disponibles, descargar `local_inputs` a una carpeta de `outputs/`. Desde la raíz del proyecto:

```powershell
.\.venv\Scripts\python.exe scripts/run_local_prepared_video.py --cpu-package outputs/cpu_image_prepare/local_inputs
```

El importador verifica el ZIP y cada archivo, rechaza rutas fuera del destino y conserva una reserva de 10 GiB de disco y una caché de imágenes de hasta 6 GiB. El ejecutor verifica los pesos públicos y el código del detector, reutiliza las gaussianas CPU y completa las detecciones/características que falten. Extrae las características de cada UNet por separado. La detección combinada todavía carga ambos modelos; su capacidad con imágenes reales grandes debe medirse. Si falla una etapa, los productos anteriores con recibo válido se conservan.

Ya están copiados y verificados los **48 grafos de E012** en `outputs/local_event_graph/videos`. Todavía faltan las características visuales reales. Los resultados recuperables del candidato E013 se incorporarán cuando Arturo avise, evitando repetir etapas completas. Hasta entonces, la preparación CPU puede avanzar de forma independiente.

El siguiente bloque se construye con `scripts/build_cpu_workflow.py --offset 1`, y así sucesivamente hasta 63. Cada envío usa un recibo nuevo con `scripts/launch_cpu_notebook.py`; un recibo existente impide repetir accidentalmente el lanzamiento. Se avanza cuando el bloque anterior está guardado y verificado, sin crear monitores.

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

Kaggle rechazó adjuntar directamente las salidas del notebook fallido. El notebook `cpu_control` está preparado para leer una copia del CSV desde un dataset privado. Su subida está pendiente de la confirmación específica solicitada por la revisión automática de permisos; la autorización de código anterior no cubría ese archivo de predicciones. No se ha calculado todavía la métrica oficial de este control corregido.
