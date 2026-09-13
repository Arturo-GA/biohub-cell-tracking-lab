# Auditoría técnica — 2026-09-13

## Notebook entregado

Fuente identificada mediante Kaggle API: `amanatar/improved-metric-hack-last-call`, última ejecución pública consultada: **20 de julio de 2026**. Fue el primer resultado del orden por score de la API, pero ese orden incluye resultados históricos: no demuestra que siga siendo el mejor modelo válido.

La arquitectura combina TemporalUNet3D, transformer de nodos, TTA y un solver ILP. Después añade cierre de huecos de un frame y filtrado de tracks cortos. La celda 6 inserta un nodo hub en `t=-1000`, coordenadas `-10000`, conecta hasta 3000 componentes y agrega 20 divisiones ficticias. La celda 7 evalúa archivos GEFF anteriores a esa transformación del CSV: su métrica local no mide lo que envía.

El mismo mecanismo fue descrito por participantes y reconocido para investigación por un organizador en el [foro oficial](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/714101). No usamos ese mecanismo como mejora de tracking.

Otros problemas operativos observados: asume dos GPUs, importa `Path` condicionalmente en la instalación, y su supuesto conjunto local no aporta evidencia de exclusión del entrenamiento. Estos puntos se derivan del código descargado, no de una ejecución propia del notebook original.

## La métrica cambió

El [código oficial actual](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/metrics.md) describe evidencia local para divisiones, en lugar de conectividad global. En la implementación inspeccionada se deduplican enlaces, se filtran enlaces no consecutivos y se limita el número de hijos.

La puntuación combina adjusted edge Jaccard y 0.1 × division Jaccard. La penalización usa una estimación del número total de células: **el número de anotaciones escasas no es ese total**. La agregación pondera el término ajustado por TP+FP+FN por video y agrega los conteos de divisiones. Este repositorio llama las funciones oficiales sin reescribir esas reglas.

La copia publicada por los organizadores es la referencia verificable. No tenemos acceso al binario privado de scoring de Kaggle; la identidad exacta de su versión debe contrastarse con feedback de una submission control.

## Referencias recientes

- [Harmonic Fusion](https://www.kaggle.com/code/flexonafft/biohub-harmonic-fusion), última ejecución consultada 2026-09-10: dos seeds, asociación inversa, TTA de features y DeepCenter. Es nuestra base de inferencia congelada. Su sección posterior selecciona muestras de train y las llama held-out, pero no verifica la membresía de entrenamiento de los checkpoints; además utiliza una métrica proxy de divisiones basada en componentes. Esa sección se excluye.
- [Biohub Cell Tracking: 0.947 LB](https://www.kaggle.com/code/reyhanksatria/biohub-cell-tracking-0-947-lb), consultado el 2026-09-13: añade referencias a TTA del segundo modelo. El 0.947 es una afirmación del autor/título, no un resultado propio reproducido. Su metadata descargada tenía tres referencias de dataset vacías; no se usa como base ejecutable sin resolverlas.

## Transferencia desde RSNA

Se leyó la tarea **Evalúa estado del proyecto Kaggle** y el contexto local de RSNA. Ideas metodológicas transferibles: conservar un control público fuerte; cachear trabajo costoso; cambiar un mecanismo por experimento; separar diagnósticos circulares de evaluación independiente; registrar descartes y correlación/solapamiento de errores. Ni sus pesos de resonancia ni sus umbrales AUC se transfieren a Biohub.

La hipótesis de división no equivale a garantizar una ganancia. El diagnóstico actual usa muestras de train con exclusión de los nombres visibles de test; hasta verificar el split de los pesos, no se etiqueta como validación independiente.
