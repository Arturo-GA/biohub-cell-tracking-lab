# E007 — Especialista temporal de mitosis sobre Harmonic

E006 mejoró el ranking de divisiones, pero sustituir todas las asociaciones redujo el score condicional de 0.9666951 a 0.9436645. **E007 implementa una red nueva para verificar eventos y una reparación aditiva del grafo de Harmonic.** No es una nueva selección de umbral de E006 ni una reproducción de CELLECT u OrganoidTracker.

## Representación del evento

La entrada son tres recortes: madre en `t`, primera hija en `t+1` y posible segunda hija en `t+1`. Cada recorte contiene cinco frames, con el preprocesamiento ya disponible. Juntos abarcan `t−2 ... t+3`. La nueva rama procesa cada frame con un CNN 3D compartido y alinea las tres vistas sobre esos seis tiempos. Dos indicadores marcan la ausencia de la vista de madre al final y de hijas al principio. En los límites del video se conserva el padding temporal del preprocesamiento original.

Las hijas se combinan mediante media, diferencia absoluta y producto; el resultado es invariante al intercambio de sus identidades. Una GRU bidireccional representa el orden temporal. Se incorporan además geometría física y las representaciones y puntuación de división de **E006 congelado**, verificando sus hashes. No se reutiliza su corrección global de prior para decidir reparaciones. Cada especialista tiene **365.891 parámetros**, de los que **168.385 se entrenan**; al terminar se comprueba que todos los pesos de E006 permanecen idénticos.

## Supervisión y selección

Se entrenan dos especialistas, reservando un embrión completo para cada uno y manteniendo las listas de entrenamiento/desarrollo de E004 y E006.

1. Generar **2.048 películas sintéticas de nueve frames por separación**. Las divisiones dejan suficiente contexto anterior y posterior. Texturas exclusivamente de los videos de entrenamiento de esa separación, con el renderer propio y cruces negativos de E006.
2. Entrenar la nueva red durante **4.000 pasos** con clases balanceadas. E006 permanece congelado. El checkpoint sintético es el último del presupuesto fijo.
3. Adaptar durante **3.000 pasos**. Cada lote contiene 16 positivos y 16 negativos reales conocidos, más ocho positivos y ocho negativos sintéticos para mantener exposición a linajes densos. La pérdida combina BCE y ordenamiento positivo/negativo por pares. Se aplican reflexiones espaciales e intensidad compartida entre vistas y tiempos del evento.
4. Seleccionar el checkpoint por AP de todos los triples de desarrollo; desempatar por BCE balanceada. Elegir un único umbral de score por F0.5 en ese mismo conjunto, priorizando precisión. Los empates de puntuación se tratan como un grupo. No se escoge ningún umbral con el embrión reservado o con el leaderboard.

Los negativos reales siguen exigiendo una hija cuyo padre anotado sea otro: una célula sin anotación no se convierte automáticamente en negativo. El F0.5 se mide en ese conjunto de candidatos etiquetados; **no certifica precisión ni calibración sobre los candidatos desconocidos del detector**. Una separación tiene pocas divisiones de desarrollo, por lo que esa elección puede transferirse mal. Se guardan scores, etiquetas, videos y filas de desarrollo y holdout para auditarlo.

## Reparación del grafo

El punto de partida es el CSV Harmonic congelado. Una propuesta requiere una madre con exactamente una hija y una posible segunda hija, en el siguiente frame, sin padre en el control. Ambas hijas deben estar dentro del radio físico de 20 µm usado en el experimento temporal. Se consideran hasta las cuatro huérfanas más cercanas por madre; no se consulta GT para proponerlas.

Tras aplicar el umbral de desarrollo, una asignación bipartita de máximo peso elige propuestas compatibles, usando el margen sobre el umbral y permitiendo abstención. Una madre recibe como máximo una nueva hija y una huérfana como máximo un padre. **Todos los nodos y enlaces de Harmonic se conservan.** El código comprueba esta propiedad y valida el CSV final, incluidos límites físicos, frames consecutivos y grados del linaje.

Este alcance no permite corregir un padre equivocado ya asignado, eliminar una falsa división del control, crear detecciones ausentes o saltar frames. Añadir falsas divisiones todavía puede bajar la métrica: preservar enlaces no garantiza preservar el score.

## Evaluación y criterio de decisión

El notebook encadena ambos entrenamientos, AP y conteos al umbral fijado en los 199 videos reservados entre los dos modelos, y la métrica oficial del CSV reparado sobre los cuatro videos del diagnóstico. No reemplaza los pesos de E006 ni recalcula Harmonic; reutiliza sus salidas verificadas. La comparación completa sigue siendo **condicional** porque el detector y enlazador públicos de Harmonic vieron esos videos durante su entrenamiento.

Se registran propuestas evaluadas, propuestas sobre el umbral, enlaces agregados, enlaces conservados, divisiones TP/FP/FN y score oficial. Para promoverlo debe mejorar el tracking frente a Harmonic y revisar los falsos positivos; una mayor AP aislada no basta. No hay selección automática de otro umbral en esos cuatro videos. Este notebook **no produce una submission de test ni la envía**.

El inventario local encontró 17.948 propuestas entre los cuatro videos, con un máximo de 151,9 MB de recortes uint8 por video. Es una comprobación de capacidad y oportunidades geométricas, no una medida de cuántas mitosis verdaderas puede recuperar. [Inventario](../results/E007_candidate_inventory.json).

## Ejecución y trazabilidad

```text
python scripts/build_temporal_notebooks.py specialist
python -m kaggle kernels push -p kaggle/temporal_specialist
```

Notebook: [Biohub Lab Mitosis Specialist](https://www.kaggle.com/code/jarturo/biohub-lab-mitosis-specialist). Se adjuntan las salidas completas de Temporal Prepare, Official Metric AB y Dense Lineage Pretraining. Las separaciones usan GPUs distintas cuando hay dos disponibles. Las escenas permanecen en `/tmp`; se exportan manifiestos, pesos y recibos.

Las pruebas incluyen intercambio de hijas, dependencia del orden temporal, gradientes en la nueva red y ausencia de gradientes en E006, contexto completo de las mitosis sintéticas, umbrales con empates, asignación con conflicto entre madres y un ciclo reducido de entrenamiento, recarga e inferencia. Las pruebas de funcionamiento no demuestran mejora en Biohub. El lanzamiento y los resultados se registran en `results/STATUS.json`; no se crea un monitor ni una espera de finalización.
