# E005 — Submission de los modelos temporales existentes

El 14 de septiembre de 2026 Arturo autorizó enviar los modelos ya entrenados para medir su resultado real en Kaggle, aunque E004 había resultado negativo en la evaluación local. Este experimento reutiliza esos pesos y no realiza otro entrenamiento. El control de leaderboard a comparar es **0.946**; el **0.947** de Harmonic Fusion completo sigue siendo una referencia reportada por Arturo.

## Resultado confirmado

**Score público: 0.913**, estado **COMPLETE**, submission **56223367**, script version **349689941**, verificados mediante la API autenticada de Kaggle. Queda **0.033 por debajo** de nuestro control y **0.034 por debajo** de la referencia reportada por Arturo, a la precisión publicada. El envío ya existía al revisar la cuenta y esta tarea no creó otro. Se cierra esta versión como resultado negativo; E000 sigue como control.

Se descargaron y verificaron el código empaquetado, los hashes de los checkpoints y el CSV de la ejecución visible. El detector produjo exactamente el mismo archivo que E000, incluido su SHA256. El candidato conserva los **122.745 nodos**, agrega 226 enlaces y retira 272: quedan **118.398 enlaces**. Las bifurcaciones predichas pasan de **98 a 0** en los cuatro videos visibles. Son conteos de predicciones, no 98 divisiones anotadas ni una descomposición de la métrica del test oculto. La ausencia de bifurcaciones es una limitación clara de este ensemble y resulta coherente con los problemas de divisiones observados en E004.

La ejecución del control dentro del notebook tardó **1.380,32 s (23,01 min)** y la etapa temporal **51,73 s**, excluyendo la carga inicial de sus checkpoints. Los 396 problemas del solver registrados terminaron con gap cero. El CSV descargado pasó la validación de datasets, coordenadas, IDs y linaje; su SHA256 es `d5e198051d459d7ef16e6c92fd7d2b0f52e95ab190dd16188de7e06a21d102bd`.

[Recibo de ejecución y comparación visible](../results/E005_test_completed.json), [submission confirmada](../results/E005_submission.json).

## Qué se envía

Notebook: [Biohub Lab Temporal Submission](https://www.kaggle.com/code/jarturo/biohub-lab-temporal-submission), generado en `kaggle/temporal_test`. Adjunta la salida completa de **Temporal Train, versión 2**. Los dos checkpoints se verifican por SHA256 antes de cargarse; los hashes se encuentran en `src/biohub_lab/temporal_submission.py`.

1. Se ejecuta la configuración congelada de Harmonic Control sobre **todos los videos de test disponibles en esa ejecución**. Se conservan sus detecciones y se sustituyen sus enlaces.
2. Ambos modelos temporales procesan todos los videos, incluidos los de embriones nuevos. No se selecciona un modelo según el nombre del video.
3. Se promedian por igual las probabilidades de cada padre candidato, incluida la opción sin padre. Sobre esa media se construyen las hipótesis comunes de división.
4. Cada modelo evalúa las mismas hipótesis de división, corrige sus logits con su proporción de clases de entrenamiento y convierte el resultado en probabilidades. Se promedian esas probabilidades y se convierten de nuevo en logits.
5. Un único optimizador reconstruye el linaje con las probabilidades combinadas. Se mantienen el umbral de división de 0,5, los candidatos, distancias y restricciones de E004.
6. El CSV completo se valida contra los volúmenes de esta ejecución antes de publicarse como `submission.csv`. Se registran hashes, conteos, cambios de enlaces, estados y gaps del solver.

La inferencia no necesita etiquetas ni acceso a la carpeta `train`, y no adjunta predicciones fijas del test visible. Kaggle puede repetir el procedimiento con otros videos en el test oculto.

## Qué puede concluirse

E004 obtuvo **0.9412437** frente a **0.9666951** del control en cuatro videos de entrenamiento con detecciones fijas. Esa comparación usaba un único modelo que excluyó el embrión correspondiente. **E005 usa un ensemble de dos modelos y no tiene esa misma separación de validación.** El resultado del test visible no constituye una nueva validación por embrión, porque parte de sus nombres corresponde a datos públicos de entrenamiento.

La medición externa confirmó que este candidato queda por debajo del control. La corrección por proporciones de clases no garantiza calibración en un embrión nuevo. El score público se registra separado de las métricas locales y no permite atribuir por sí solo cuánto de la caída corresponde a aristas o divisiones en el test oculto.

## Verificación

La suite cubre la invariancia al orden de los modelos, la equivalencia al duplicar un modelo, la generación completa del CSV con un nombre de embrión nuevo sin carpeta de entrenamiento, los IDs de nodo no consecutivos del detector y el rechazo de pesos con hash incorrecto. La comparación de grafos utiliza las coordenadas conservadas y una correspondencia explícita entre IDs.

Ejecución reproducible:

```text
python scripts/build_temporal_notebooks.py test
python -m kaggle kernels push -p kaggle/temporal_test
```

La revisión local del archivo final se completó después de que el usuario avisara que había terminado la submission. No se repitió el envío existente.
