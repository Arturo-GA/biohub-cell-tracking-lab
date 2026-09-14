# E007 — Especialista temporal de mitosis sobre Harmonic

E006 mejoró el ranking de divisiones, pero sustituir todas las asociaciones redujo el score condicional de 0.9666951 a 0.9436645. **E007 implementa una red nueva para verificar eventos y una reparación aditiva del grafo de Harmonic.** No es una nueva selección de umbral de E006 ni una reproducción de CELLECT u OrganoidTracker.

## Resultado completado — 14 de septiembre de 2026

La versión 1 terminó correctamente. **No se envía a submission:** el score oficial condicional fue **0.9558180**, frente a **0.9666951** de Harmonic, una pérdida de **0.0108771**. Se conservaron todos los nodos y enlaces del control, pero las adiciones no aumentaron los aciertos de división. [Recibo y verificación](../results/E007_completed.json).

| Métrica sobre los mismos cuatro videos | Harmonic | E007 |
|---|---:|---:|
| Score oficial condicional | 0.9666951 | 0.9558180 |
| Aristas ajustadas | 0.9416951 | 0.9404334 |
| División Jaccard | 0.2500000 | 0.1538462 |
| Divisiones TP / FP / FN | 2 / 1 / 5 | 2 / 6 / 5 |
| Aristas evaluadas TP / FP / FN | 2051 / 70 / 65 | 2052 / 74 / 64 |

Se añadieron **61 enlaces**, se retiraron **cero** y las bifurcaciones predichas pasaron de **73 a 134**. De las nuevas aristas, la evaluación aumentó en una los aciertos y en cuatro los falsos positivos; las demás no deben clasificarse automáticamente como correctas o falsas con anotación escasa. El CSV contiene **74.980 nodos y 72.556 enlaces**. Se reconstruyó exactamente su grafo a partir de las puntuaciones guardadas y el umbral original, sin volver a seleccionar umbrales. La pérdida en la contribución de divisiones fue 0.0096154 y la de aristas ajustadas, 0.0012617.

| Embrión reservado | Paso seleccionado | Divisiones de desarrollo | AP desarrollo | AP embrión reservado | TP / FP / FN al umbral de desarrollo |
|---|---:|---:|---:|---:|---:|
| 44b6 | 500 | 31 | 0.3626 | 0.3522 | 7 / 4 / 19 |
| 6bba | 2500 | 3 | 1.0000 | 0.1774 | 12 / 27 / 113 |

El segundo caso evidencia mala transferencia: un resultado perfecto con tres positivos de desarrollo no predijo buen rendimiento en el otro embrión. Estas métricas de triples etiquetados son distintas de las divisiones evaluadas en el CSV. Ambos modelos completaron 4.000 pasos sintéticos y 3.000 de adaptación. Cada separación generó 2.048 películas de nueve frames con 4.185 eventos; sus geometrías usan las mismas semillas, por lo que no son 8.370 eventos independientes. El pipeline tardó **516,59 segundos** (8,61 minutos), excluyendo bootstrap y exportación. La inferencia del especialista en los cuatro videos tomó 29,42 segundos.

Se verificaron el código descargado, hashes y contenido de los checkpoints, separación por embrión, procedencia de las texturas y conservación exacta de los pesos E006. Se recalcularon la AP, el umbral de desarrollo y los conteos de holdout desde los scores descargados. Las métricas oficiales proceden de la ejecución Kaggle con el código fijado; no se volvió a ejecutar localmente la evaluación de imágenes/GEFF. Esta comparación sigue siendo **condicional, no leaderboard**, porque Harmonic se entrenó con esos videos.

### Auditoría de oportunidades y cambio de prioridad

Se recuperaron los cuatro grafos de GT de la preparación temporal y se verificaron sus hashes de configuración. Para cada una de las siete divisiones se estudiaron las detecciones más cercanas y las propuestas disponibles **en el frame exacto**, con distancia física de 7 µm:

- Dos eventos ya tienen la bifurcación correspondiente en el control bajo esta correspondencia de vecinos.
- En dos eventos de `6bba_afb141ff` falta una hija dentro de 7 µm: las detecciones más cercanas están a 10,12 y 12,23 µm.
- En otros dos eventos de ese video, ambas hijas tienen la misma detección más cercana.
- Solo un evento presenta una propuesta geométricamente compatible de E007 en el frame exacto: `6bba_337b1b3a`, madre en frame 38. Su score fue −8,8672, frente al umbral de desarrollo 6,9336; fue rechazada.

Esta auditoría **no es la correspondencia oficial ni un máximo alcanzable formal**: la evaluación oficial considera contexto de linaje y tolerancia temporal, y la proximidad geométrica por sí sola no demuestra una división correcta. Aun así, cambia la prioridad de trabajo: estudiar detecciones de hijas ausentes o insuficientemente separadas durante mitosis antes de otra reparación exclusiva de enlaces. Se cierra E007, sin nueva submission ni otro entrenamiento iniciado en esta revisión. [Auditoría por evento](../results/E007_opportunity_audit.json).

Para reproducir ambas verificaciones con los outputs descargados:

```text
python scripts/verify_mitosis_result.py
python scripts/audit_mitosis_opportunity.py
```

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
