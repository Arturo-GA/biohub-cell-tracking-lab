# E014 completado y evaluación en CPU

## Resultado final de la evaluación, 19 de septiembre

La ejecución CPU terminó en **9.445,98 segundos (2 h 37 min 26 s)**. El score de desarrollo es **0,324351**, frente a **0,909257** del control, una diferencia de **−0,584906**. El Jaccard de enlaces ajustado empeoró en **48 de 48 videos**. Se descarta E014 como sustituto completo del tracker y no se envía al leaderboard. No se inició otra ejecución ni entrenamiento al revisar este resultado.

En calibración, continuaciones sin divisiones obtuvo 0,302565, frente a 0,301225 al permitir divisiones; por eso se eligió la primera opción antes de evaluar. El candidato final no predice divisiones y omite las 33 anotadas. Su recuperación de nodos es 0,994846, pero su Jaccard de enlaces sin ajuste es apenas 0,374725: localizar muchas células no equivale a reconstruir bien sus trayectorias.

El candidato conserva **2.622.332 nodos**, frente a **1.072.577** del control, una razón de **2,44489**. Esto justifica revisar la selección de detecciones y su correspondencia uno a uno. No prueba por sí solo que todo el exceso sean duplicados. Hubo fallback en **302/1.200 ventanas** de evaluación; sin embargo, **28 videos no necesitaron fallback y todos empeoraron también**. Aumentar únicamente el tiempo del optimizador no tiene respaldo suficiente.

Se verificaron los 16 archivos fuente contra el paquete del lanzamiento, la configuración congelada, la elección del brazo, las cohortes completas y los agregados oficiales recalculados desde los informes por video. Los hashes del CSV coinciden entre los recibos remotos; el CSV grande no se descargó ni se volvió a evaluar localmente. [Auditoría completa](../results/E014_CPU_completed.json). Reproducción: `python scripts/audit_e014_cpu_completed.py`.

La precisión promedio 0,998211 de candidatos conocidos no anticipó el comportamiento de trayectorias completas. Las etiquetas de E013 pueden asignar varias detecciones cercanas a una misma célula anotada, mientras que la métrica oficial exige correspondencias de grafos; hace falta auditar esa diferencia con los resultados existentes antes de atribuirle toda la caída. Se conserva Harmonic como referencia y se detiene la repetición de E013/E014 sin cambios. Cualquier siguiente propuesta debe demostrar primero una mejora de trayectorias completas en calibración; no basta otra mejora de clasificación. Esta revisión no consume GPU.

E014 completó 6.000 pasos. Se verificaron el paquete fuente original, los hashes de entrada y checkpoints y las métricas de calibración recalculadas desde sus puntuaciones. El entrenamiento registrado duró **156,65 segundos** y el proceso completo **357,85 segundos**; estas duraciones no son necesariamente iguales al tiempo facturado o descontado de cuota por Kaggle.

El checkpoint elegido de enlaces es el paso 3.000 y el de divisiones el paso 4.000. En los candidatos conocidos de calibración, la precisión promedio agrupada de enlaces es **0,998211**, mientras que la de divisiones es **0,006851**. No son scores del concurso. La buena métrica promedio por video del entrenamiento de divisiones ocultaba dificultades al agrupar las puntuaciones entre videos; no se considera resuelta esta tarea. Solo cuatro videos aportan ambas clases de divisiones.

Se fijaron umbrales mediante F1 para enlaces y F0,5 para divisiones, usando exclusivamente los 16 videos de calibración. El de divisiones acepta un positivo y tres negativos y omite 54 positivos: esto confirma que no basta con presentar la métrica promedio de entrenamiento como éxito. [Auditoría](../results/E014_completed.json).

## Ejecución CPU enviada

El notebook privado [Association CPU Evaluation](https://www.kaggle.com/code/jarturo/biohub-lab-association-cpu-evaluation), versión 1, fue aceptado por Kaggle con **GPU y TPU desactivadas**. [Recibo](../results/E014_CPU_launch.json). Reutiliza los pesos E014 y los grafos/características E013, verificando sus hashes. No entrena ni recalcula detectores.

1. Puntúa las asociaciones en CPU. Conserva cuatro continuaciones y dos divisiones que superen el umbral por madre.
2. En los 16 videos de calibración compara continuaciones sin divisiones con continuaciones y divisiones aprendidas. Usa el optimizador temporal existente y registra cada fallback. La calidad de nodo se fija en 0,5 porque E014 no entrena esa tarea; esto forma parte de la nueva configuración, no de una comparación aislada entre clasificadores.
3. Congela ambos CSV antes de acceder a sus anotaciones para calcular la métrica oficial. Elige el mejor score de calibración; los empates favorecen el modelo sin divisiones.
4. Fija esa selección antes de predecir y evaluar los 48 videos de comparación. Reporta la diferencia contra el control **0,909257**. No envía el CSV al leaderboard.

La ejecución puede tardar horas. No consume GPU. La evaluación sigue siendo de desarrollo condicional y no permite prometer el resultado público. La validación previa incluyó tres pruebas de umbrales y restricciones del grafo, además de cargar ambos checkpoints reales y comprobar inferencia finita en CPU. [Preflight](../results/E014_CPU_preflight.json).

## Política de cómputo solicitada por Arturo

CPU por defecto para preparación, inferencia de estos modelos pequeños, asociación, métricas y futuros entrenamientos pequeños. Utilizar GPU solo cuando el trabajo realmente la requiera, justificándolo por el modelo, memoria o una medición de viabilidad; una espera de varias horas en CPU es aceptable. Mantener fuera del plan Colab y cómputo pagado. No activar monitores de ejecución: Arturo avisa cuando termina.
