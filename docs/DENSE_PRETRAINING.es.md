# E006 — Preentrenamiento con linajes densos

Se implementó tras el resultado público negativo de E005 (**0.913**, control **0.946**). La hipótesis es que aprender primero con trayectorias completamente conocidas y miles de divisiones sintéticas aporta una representación más útil que entrenar exclusivamente con las pocas divisiones anotadas de Biohub.

## Resultado completado — 14 de septiembre de 2026

La versión 1 terminó correctamente. **No se promueve a submission:** el preentrenamiento mejoró el ranking de divisiones frente a E004, pero sustituir todo el enlazador sigue dando un resultado inferior a Harmonic. [Recibo completo y verificación](../results/E006_completed.json).

| Comparación oficial condicional, mismos cuatro videos y detecciones | Score | Aristas ajustadas | División Jaccard | Divisiones TP / FP / FN |
|---|---:|---:|---:|---:|
| Harmonic E000 | 0.9666951 | 0.9416951 | 0.2500 | 2 / 1 / 5 |
| Temporal desde cero E004 | 0.9412437 | 0.9412437 | 0.0000 | 0 / 1 / 7 |
| Preentrenamiento denso E006 | 0.9436645 | 0.9374145 | 0.0625 | 1 / 9 / 6 |

E006 gana **0.0024208** frente a E004, pero pierde **0.0230306** frente a Harmonic. La mejora de división respecto a E004 aporta +0.00625 al score; la pérdida de aristas ajustadas resta 0.0038292. Las detecciones y su recall son idénticos. Estas cifras **no son leaderboard ni validación independiente del pipeline completo**: el detector público fue entrenado con esos videos.

| Embrión reservado al modelo temporal | AP división E004 → E006 | Exactitud padre E004 → E006 | Vecino más cercano |
|---|---:|---:|---:|
| 44b6: 71 videos, 26 divisiones anotadas | 0.1872 → 0.3819 | 0.994855 → 0.996419 | 0.997932 |
| 6bba: 128 videos, 125 divisiones anotadas | 0.0527 → 0.0820 | 0.988153 → 0.987025 | 0.991821 |

La AP mejora en ambas separaciones, pero la elección de padres permanece por debajo del vecino más cercano. La mejora del ranking no garantiza que las probabilidades ni la selección de eventos estén bien calibradas. No se realizaron nuevas selecciones de umbral sobre estos resultados.

Cada separación completó 2.048 escenas, **4.185 eventos sintéticos**, 4.000 pasos de preentrenamiento y 3.000 de adaptación. Las dos usan las mismas semillas de geometría con texturas de embriones distintos; sus eventos no se deben sumar como 8.370 geometrías independientes. Ambos checkpoints de adaptación fueron seleccionados en el paso 500 por el conjunto de desarrollo. La generación, ambos entrenamientos y el diagnóstico tardaron **416,91 segundos**, excluyendo bootstrap y exportación del notebook.

El CSV final contiene **74.980 nodos y 72.647 enlaces**. Frente a Harmonic agrega 454 enlaces y retira 302; sus bifurcaciones predichas aumentan de **73 a 340**. Esos 340 eventos no equivalen a 340 falsos positivos evaluados, porque la anotación es escasa. La métrica oficial sí registra nueve divisiones falsas y una correcta. Los 396 problemas de optimización terminaron con gap cero: resolver exactamente el objetivo aprendido no implica que ese objetivo describa correctamente el linaje.

Se verificaron el payload del notebook descargado, los hashes y contenido de los cuatro checkpoints, los manifiestos y la exclusión de desarrollo y del embrión reservado del banco de texturas. Se cargaron ambos modelos finales estrictamente y se validó el CSV contra las dimensiones reales. No se lanzó inferencia de test ni se hizo una nueva submission. Los pesos se conservan como componente de investigación; esta versión se cierra como sustituto completo del enlazador.

## Cambio y comparación

Se mantiene la arquitectura de E004, su preparación real, las separaciones por embrión, el muestreo de adaptación, el criterio de selección del checkpoint y los umbrales de inferencia. **El cambio es la inicialización obtenida mediante un preentrenamiento denso.** Esto permite estudiar la aportación de esa supervisión sin cambiar simultáneamente el solver o calibrar umbrales con el leaderboard. No se presenta como una nueva arquitectura ni como reproducción de un paper.

Cada separación genera **2.048 películas 3D de cinco frames**. Se preentrena durante **4.000 pasos**, y después se ejecutan los mismos **3.000 pasos de adaptación real** que E004. El checkpoint de preentrenamiento se toma al final del presupuesto fijo; la adaptación selecciona checkpoint únicamente con desarrollo del embrión de entrenamiento. El segundo embrión queda fuera de ambos entrenamientos y del banco de texturas.

## Generación y etiquetas

El renderer propio simula movimiento colectivo e individual, núcleos elipsoidales, cambios de intensidad, ruido Poisson y gaussiano, desenfoque, fondo suave y error de localización. Una madre puede dividirse; sus hijas heredan la textura, reducen su volumen y se separan. La conservación aproximada de fluorescencia se aplica antes de la normalización de cada frame. Un 25 % de escenas incluye dos células independientes con trayectorias de cruce, que no se etiquetan como una división.

Todos los núcleos renderizados tienen centro y relaciones padre-hija conocidas. Los negativos de pares incluyen candidatos vecinos incorrectos de esos linajes densos. No se considera fondo a una célula real sin anotación: la adaptación mantiene la supervisión escasa de E004.

Las texturas proceden de recortes centrales de hasta 256 núcleos anotados de los videos de entrenamiento de cada separación, con sustracción de fondo y máscara espacial. Se registran video, nodo y hash del banco. Los fondos sintéticos son procedurales. No se usan texturas de desarrollo ni del embrión reservado, ni se descarga el dataset sintético público. El guardado de pesos registra la procedencia; cargar un preentrenamiento de la separación incorrecta falla explícitamente.

## Evaluación automática

El notebook [Biohub Lab Dense Lineage Pretraining](https://www.kaggle.com/code/jarturo/biohub-lab-dense-lineage-pretraining) encadena generación, preentrenamiento, adaptación y evaluación. Adjunta nuestros outputs completos de preparación temporal, E004 y el diagnóstico Harmonic.

- Evalúa elección de padres y AP de pares de división en todos los videos del embrión reservado de cada modelo.
- Compara esas métricas con E004, verificando los hashes de sus pesos de referencia.
- Sustituye asociaciones sobre las mismas detecciones de los cuatro videos del diagnóstico y evalúa el CSV con la métrica oficial.
- Conserva el alcance de esa última comparación: el modelo temporal no vio el embrión reservado, pero el detector público sí estuvo entrenado con esos videos.

La decisión requiere recuperar divisiones anotadas y mejorar el score completo, revisando falsos positivos y aristas. Una mejora en AP o en una métrica sintética no basta para afirmar mejora de leaderboard. Este notebook **no emite una submission de test ni la envía al concurso**; produce pesos y los resultados necesarios para decidir.

## Recursos y verificación

Las escenas y sus recortes se escriben en `/tmp/biohub_dense` dentro de Kaggle; se conservan manifiestos, semillas y cuatro ejemplos por separación en la salida. La preparación real de los 199 videos se reutiliza. Las dos separaciones usan GPUs distintas cuando están disponibles. El conjunto sintético se carga en memoria por separación para evitar miles de archivos mmap abiertos.

La suite de 39 pruebas verifica, entre otros componentes, topología binaria completa, enlaces consecutivos, correspondencia de etiquetas, cruces negativos, reproducción por semilla, exclusión del embrión reservado y transferencia de pesos mediante una ejecución reducida de ambos entrenamientos. Los conteos y tiempos completos se recuperaron del notebook; la prueba local de funcionamiento no se interpreta como evidencia de calidad en Biohub.

```text
python scripts/build_temporal_notebooks.py dense
python -m kaggle kernels push -p kaggle/temporal_dense
```

## Fundamento y límites

[SynCellFactory](https://arxiv.org/abs/2404.16421) estudia vídeos sintéticos para ampliar la supervisión de tracking mediante ControlNet. E006 usa un renderer procedural 3D propio y no implementa ese generador. El [generador público de Biohub](https://www.kaggle.com/code/josefreitasalvesneto/biohub-synthetic-dataset) motivó revisar la procedencia de las texturas; su código se había leído, pero no se ejecuta ni se copia aquí.

Persisten riesgos de diferencias entre imágenes sintéticas y reales, texturas con señal de vecinos, mitosis simplificada y olvido durante la adaptación. La tasa de divisiones sintéticas es una elección de preentrenamiento, no una estimación de la tasa real. La corrección de prior de inferencia sigue siendo la de E004 y podría continuar siendo inadecuada. Este experimento mide si cambiar la supervisión inicial ayuda antes de atribuir el fallo exclusivamente a calibración.
