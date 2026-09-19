# E014: aprendizaje de asociaciones con el caché E013

Arturo autorizó de nuevo usar GPU Kaggle el 18 de septiembre de 2026. La API informó 6 horas disponibles, ninguna consumida ni reservada y pago por uso desactivado. Esta autorización sustituye la restricción anterior de no iniciar nuevas GPU Kaggle; Colab y cómputo pagado siguen fuera del plan.

Se lanzó la versión 1 del notebook privado [Biohub Lab Association Ranking](https://www.kaggle.com/code/jarturo/biohub-lab-association-ranking). Kaggle aceptó la ejecución y su primera respuesta fue **QUEUED**. [Recibo de lanzamiento](../results/E014_launch.json). Sesión GPU T4 con límite de 7.200 segundos; sin seguimiento automático ni submission al leaderboard.

## Qué cambia

E013 terminó, pero su candidato obtuvo 0,314681 frente a 0,909257 del control. E014 reemplaza la atención sobre el grafo por dos modelos independientes de apariencia y geometría: uno para enlaces y otro para divisiones. Evita que las dos tareas compartan gradientes y omite posición absoluta y tiempo como entradas. El modelo de divisiones es simétrico respecto al orden de las hijas.

La pérdida combina clasificación equilibrada entre casos conocidos y comparación entre eventos positivos y negativos de una misma madre. Se muestrea primero el video y después sus casos; así, los videos con muchas detecciones no dominan por volumen. Los casos sin etiqueta no pasan a ser negativos. Los videos con una sola clase siguen aportando casos a su grupo positivo o negativo; la comparación entre competidores usa solamente madres que tienen ambas clases conocidas.

Se mantienen los **48 videos de ajuste y 16 de calibración**, con 6.000 pasos para cada modelo. Cada 500 pasos se selecciona un checkpoint independiente por tarea mediante precisión promedio por video, normalizada respecto a la prevalencia de positivos. Se guardan los mejores pesos, el último estado de los modelos, optimizadores, schedulers y generadores aleatorios, junto con los diagnósticos de calibración. El ejecutor aún no implementa un comando para reanudar ese estado; conservarlo permite añadir una recuperación sin perder el estado del optimizador.

## Reutilización y límites de la evidencia

El notebook adjunta la salida original de E013 y exige el hash del manifiesto congelado `1c8f7d3d…6fb5c0`. Verifica los grafos y características de los 64 videos usados, y que las etiquetas correspondan a cada grafo; registra sus hashes antes de entrenar. No recalcula imágenes ni vuelve a ejecutar los detectores. No accede a etiquetas de los 48 videos de evaluación.

Esta ejecución entrena y mide el orden de preferencia entre candidatos conocidos. **No produce aún un score oficial ni un CSV para submission**. Después habrá que comprobar las trayectorias completas con asociación estructurada y comparar con el control. Las métricas de calibración y los ejemplos duplicados de detecciones no equivalen a eventos biológicos independientes. Las 48 secuencias de evaluación ya influyeron en el diseño, y el detector público secundario vio los 199 videos; la evidencia sigue siendo de desarrollo condicional.

## Verificación antes del lanzamiento

Pasaron seis pruebas: exclusión de etiquetas desconocidas, competidores de la misma madre, simetría de las hijas, sentido de los gradientes, tratamiento correcto de empates en precisión promedio y aprendizaje de enlaces sintéticos. Una ejecución integrada de tres pasos comprobó entrenamiento, selección, validación y escritura/lectura de checkpoints; es solo una prueba técnica, no un resultado del concurso. El ZIP publicado contiene cinco archivos revisados, idénticos a los locales tras normalizar finales de línea. [Preflight](../results/E014_preflight.json).

Código: `src/biohub_lab/association_rank.py`, `scripts/association_rank_runner.py`, `scripts/build_association_rank_notebook.py` y `scripts/launch_association_rank.py`. El lanzador rechaza repetir el envío si ya existe su recibo.
