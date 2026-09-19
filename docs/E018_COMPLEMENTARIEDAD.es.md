# E018: enlaces correctos complementarios, solo CPU

Lanzado el 19 de septiembre de 2026: [Biohub Lab Edge Complementarity CPU](https://www.kaggle.com/code/jarturo/biohub-lab-edge-complementarity-cpu), privado, versión 1.

El objetivo es saber si E017 recupera suficientes asociaciones que Harmonic pierde como para justificar otra arquitectura. No se vuelve a entrenar ni a procesar imágenes. Se reutilizan los CSV congelados de Harmonic validado y de E017, y los informes de sus métricas. Los seis hashes de esos archivos y sus resultados se fijan en `baseline/e018_pins.json`.

Para cada uno de los 16 videos de calibración, se construyen los grafos exactamente como en la evaluación existente. Se usa el emparejamiento de nodos y las máscaras de enlaces del código oficial fijado, no nuestra asignación diagnóstica alternativa de E015. Antes de aceptar resultados se exige que los conteos TP, FP y FN de cada sistema coincidan con los informes anteriores.

El análisis cuenta enlaces anotados correctos compartidos, exclusivos de Harmonic, exclusivos de E017 y no recuperados por ninguno. Para los exclusivos de E017 distingue si Harmonic tiene ambos extremos emparejados con GT o si le falta al menos uno. Esta distinción informa sobre representación/asociación, pero no demuestra causalidad: cada sistema tiene su propio emparejamiento oficial de nodos.

También resume características de los enlaces de E017, calculadas sin etiquetas: longitud física, contexto anterior y posterior único, y cambio de desplazamiento. Los intervalos se fijaron antes de observar los resultados: distancia hasta 3 µm, de 3 a 6 y superior a 6; cambio de desplazamiento hasta 2 µm o superior. Los histogramas separan aciertos exclusivos, aciertos compartidos, falsos positivos y enlaces sin evidencia suficiente. Estos últimos no se consideran falsos automáticamente.

La unión ideal de enlaces anotados correctos es un diagnóstico que depende de GT. **No es un grafo fusionado realizable ni un score de competición.** No se generan predicciones nuevas, no se seleccionan umbrales y no se entrena un selector sobre estos 16 videos. Si aparece señal útil, cualquier modelo posterior debe aprender con los videos de fit y validarse por separado. Las divisiones no se fusionan en este análisis.

Cuatro pruebas locales superadas: contabilidad de conjuntos con extremos ausentes, recuperación con ambos extremos presentes, rechazo de falsos TP fuera de GT, y características geométricas sin anotaciones. Fuentes y notebook compilados. La compatibilidad completa del matcher se verificará al ejecutar y reproducir los conteos de los 16 videos.

Salidas: `edge_complementarity/result.json`, `overlap_by_video.json` y `feature_histograms.json`. No se configura monitoreo automático ni submission. Git conserva código, configuración e informes agregados; datos e identificadores detallados de GT no se exportan en los informes.
