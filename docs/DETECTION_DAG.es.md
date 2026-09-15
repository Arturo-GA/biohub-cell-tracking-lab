# E011: trayectorias desde detecciones y cobertura de desarrollo

E010 dependía de trayectorias de Harmonic que ya podían estar rotas o mal asociadas. Las cinco divisiones pendientes no entraban en sus ventanas en el fotograma anotado. E011 elimina ese requisito: construye las hipótesis desde las detecciones, sin leer el grafo ni los modelos de asociación de Harmonic.

## Generador implementado

Se ejecutan los detectores CELLECT y de separación gaussiana sobre las imágenes crudas. Se conservan sus configuraciones de E008/E009 y se combinan sus propuestas con NMS a 1,2 µm, sin incorporar centros de las anotaciones. Los archivos anteriores de preparación temporal contienen centros GT y **no se usan como detecciones ni se adjuntan a este notebook**.

El generador representa explícitamente dos relaciones entre centros de fotogramas consecutivos:

- Una madre puede considerar hasta 16 centros del siguiente fotograma dentro de 20 µm. Se representan todas las parejas que satisfacen separación entre 1,5 y 20 µm y distancia del punto medio a la madre de hasta 10 µm.
- Cada centro puede continuar hacia hasta ocho centros del fotograma siguiente, dentro de 10 µm.

Las parejas se guardan como bits y las continuaciones como un grafo dirigido sin ciclos. Así quedan representadas las opciones de trayectorias sin enumerar su crecimiento combinatorio. No hay exigencia de antecesores, madre previamente enlazada, hija huérfana o ancla perteneciente a un track de Harmonic. Tampoco se aplican los filtros de contraste y masa de E010.

Para consultar una pareja, un flujo de dos unidades con capacidad de una por centro obtiene dos trayectorias que no comparten detecciones. Puede reorganizar una primera elección cuando esta bloquea a la otra rama; no usa un enlace voraz irrevocable. El destino es un fotograma, no una célula de un track existente. El horizonte es cuatro fotogramas después de la madre, acortado al final del video. Las trayectorias necesitan una observación por fotograma: no se inventan centros ni se interpolan detecciones ausentes.

Esta búsqueda es exacta **dentro del grafo ya limitado por radios, máximos de vecinos y condiciones de las parejas**. Esos filtros todavía pueden excluir la solución; por eso se audita la cobertura de cada etapa. Los puntajes de CELLECT y gaussianas se usan para deduplicar propuestas, no como probabilidades de asociación calibradas. Una muestra reproducible de 128 consultas por video materializa trayectorias antes de acceder a anotaciones; no es una selección final de predicciones.

## Videos y separación de etapas

`baseline/detection_dag_dev.json` fija **48 videos, 24 por cada grupo de adquisición**, elegidos ordenando un SHA256 de sus nombres. Se excluyen los cuatro diagnósticos anteriores y los cuatro videos visibles de test que también figuran en el inventario de train. Los otros 143 videos de train quedan fuera de esta auditoría. No se usan recuentos de divisiones ni imágenes para elegir la muestra.

Son archivos separados de los cuatro diagnósticos, **no embriones independientes ni un nuevo holdout intacto de todo el proyecto**: ambos grupos y estos videos participaron en trabajos previos. E011 no utiliza aquellos modelos entrenados, pero la pertenencia al entrenamiento público original de CELLECT sigue sin verificarse. La muestra se considera desarrollo; no se presentará como validación independiente ni resultado del leaderboard.

La ejecución tiene tres etapas:

1. Dos procesos, uno por GPU disponible hasta dos, generan nuevas detecciones y grafos de los 48 videos. Cada proceso recibe solamente rutas `.zarr` y pesos públicos.
2. Se verifican los hashes de propuestas y grafos y se guarda `frozen_inputs.json` para **todos** los videos.
3. Solo entonces se abren los GEFF y se comprueba cobertura. Los grafos se verifican otra vez después de la lectura; las anotaciones no modifican propuestas ni hipótesis.

No se entrena una red ni se produce una submission en esta ejecución. El código de bootstrap reutiliza únicamente la instalación de dependencias de Harmonic; no ejecuta su inferencia. No se adjuntan salidas de ningún notebook anterior como fuente de detecciones.

## Qué mide la auditoría

Para cada división anotada se buscan candidatos dentro de 7 µm en el fotograma exacto. Se cuentan, separadamente:

1. Madre y dos hijas distintas presentes entre las detecciones.
2. Existencia de una pareja madre/hijas en las hipótesis representadas.
3. Existencia de dos caminos sin compartir centros durante el horizonte disponible.
4. Existencia de dos caminos que además permanezcan cerca de **cada rama anotada** durante cuatro fotogramas. Esta última fracción usa únicamente eventos con continuación anotada completa de ambas hijas; se informa también su denominador.

La cuarta comprobación usa una búsqueda de estados de ambas ramas condicionada a vecindarios de las anotaciones. Evita confundir dos caminos geométricamente posibles con la posibilidad de seguir las identidades correctas. Todos estos valores son **techos optimistas de cobertura**, no precisión: el evaluador puede encontrar la mejor opción dentro de las hipótesis. Sus testigos condicionados por anotaciones se guardan como auditoría y nunca deben usarse como predicciones o submission. No se calcula la métrica oficial en E011.

## Criterio previo al entrenamiento

La configuración se fija antes de ejecutar la cobertura, sin sweep en los siete eventos inspeccionados. Para pasar el filtro se requieren:

- Al menos 20 divisiones en total y 20 con continuación anotada completa; al menos cinco de cada tipo en cada grupo.
- Al menos 90 % de cobertura de parejas y de dos caminos, globalmente y en cada grupo.
- Al menos 85 % de cobertura de los caminos consistentes con las anotaciones completas, globalmente y en cada grupo.

Estos mínimos permiten detectar un problema grande de cobertura, pero no ofrecen por sí solos una garantía estadística de transferencia. Si el filtro falla, se informa la etapa y el grupo afectados. Si pasa, únicamente permite continuar investigando una puntuación de eventos, sus negativos reales y su calibración: **no lanza entrenamiento automáticamente ni demuestra que se puedan elegir las hipótesis correctas**.

## Archivos y comprobaciones

El notebook deja `detection_dag_experiment/result.json`, `readiness.json`, `frozen_inputs.json` y recibos por proceso. Por video conserva propuestas, `dag/graph.npz`, `dag/trajectory_samples.npz` y recuentos. Los testigos de cobertura quedan separados en `coverage/`. Los datos, grafos y correspondencias detalladas permanecen en Kaggle o `outputs/`; Git solo recibe código, configuración y resultados agregados.

Las pruebas nuevas comprueban madres sin historial, hijas sin enlaces, reorganización de caminos ante conflictos, equivalencia con enumeración exhaustiva en grafos pequeños, errores de detección y poda distinguibles, cambios de identidad no contabilizados como continuidad correcta, selección por nombres, deduplicación y rechazo de muestras insuficientes. La integración usa Zarr y anotaciones sintéticas, detector gaussiano real y un sustituto sintético de CELLECT; comprueba que todos los grafos estén fijados antes de leer GEFF y que una alteración del archivo impida evaluarlo.

Implementación propia. El flujo con capacidades y el almacenamiento de parejas son herramientas del generador; no se presenta como reproducción de un tracker publicado ni como un modelo ya competitivo.

**Ejecución completada:** [Biohub Lab Detection DAG Coverage](https://www.kaggle.com/code/jarturo/biohub-lab-detection-dag-coverage), versión 1, lanzada el 14 de septiembre de 2026 y comprobada completa el 15 de septiembre a las 02:31 UTC. Pasaron 72 pruebas locales antes del lanzamiento, incluidas las nueve nuevas. Recibos: `results/E011_launch.json`, `results/E011_preflight.json`, `results/E011_technical_smoke.json`.

## Resultado: cobertura insuficiente para entrenar

Terminaron los **48 videos** en 7.602,83 segundos, aproximadamente **2 h 7 min**, excluyendo instalación y cola. Se representaron 1.646.262 centros, 23.657.173 enlaces posibles madre/hija, 7.168.158 continuaciones y 85.542.826 parejas de hijas. Estos tamaños describen las hipótesis disponibles; no son detecciones correctas ni divisiones predichas.

| Etapa de cobertura | Total | Grupo 44b6 | Grupo 6bba |
|---|---:|---:|---:|
| Madre y dos hijas distintas disponibles | 18/33 (54,5 %) | 5/8 (62,5 %) | 13/25 (52,0 %) |
| Pareja inicial representada | 17/33 (51,5 %) | 5/8 (62,5 %) | 12/25 (48,0 %) |
| Dos trayectorias sin compartir centros | 16/33 (48,5 %) | 5/8 (62,5 %) | 11/25 (44,0 %) |
| Trayectorias compatibles con las ramas anotadas | 11/25 (44,0 %) | 5/7 (71,4 %) | 6/18 (33,3 %) |

Hay suficientes eventos para aplicar el criterio previo: 33 divisiones y 25 con contexto completo, con los mínimos satisfechos en cada grupo. **Fallaron los mínimos de cobertura globales y de ambos grupos.** El último renglón usa un denominador diferente porque ocho divisiones carecen del contexto anotado completo exigido; no se interpreta como 11 de 33.

De las 33 divisiones, **15 se pierden antes de formar parejas por falta de centros compatibles**; una más se pierde al restringir las parejas y otra al exigir dos continuaciones. Por tanto, el principal problema inicial está en el conjunto de detecciones que recibe el generador. La existencia de trayectorias cualesquiera tampoco basta: solo 11 de los 25 eventos con contexto completo tienen dos trayectorias compatibles con sus ramas anotadas.

E011 no entrenó un modelo, no generó una submission y no calculó la métrica oficial. **Se cierra esta configuración sin entrenamiento ni envío:** un clasificador de eventos no puede seleccionar centros ausentes de sus candidatos. Nuestro mejor score público verificado sigue siendo 0.946; no se volvió a consultar el leaderboard para esta auditoría.

## Auditoría de las pérdidas y siguiente dirección

Se reconstruyeron **exactamente los 48 grafos** desde los archivos de propuestas descargados. Coincidieron las matrices de centros, los vecinos, las máscaras de parejas, las 6.144 consultas muestreadas y sus trayectorias. Se comprobaron el payload lanzado, código, selección de videos, pesos según los recibos, archivos congelados y checksums de las anotaciones preparadas. Se recalcularon todos los recuentos y testigos de cobertura, reproduciendo también el criterio fallido. **No se volvió a ejecutar localmente la inferencia de imágenes de los detectores.**

Los 15 eventos sin centros iniciales suficientes se distribuyen así:

- **10:** hay madre cercana, pero faltan dos detecciones distintas compatibles con las hijas.
- **2:** hay dos hijas compatibles, pero falta la madre.
- **3:** faltan tanto la madre como dos hijas compatibles.

Al comprobar madre y dos hijas, CELLECT solo cubre **17/33**, gaussianas **3/33** y la unión antes de la deduplicación **18/33**. El conjunto combinado conserva esos **18/33**: la deduplicación final a 1,2 µm no explica la pérdida de cobertura inicial. Estos recuentos no miden falsos positivos y se refieren a las propuestas ya filtradas por cada detector.

Entre los **14 de 25** eventos que no permiten seguir ambas ramas anotadas durante todo el contexto, **13** carecen de centros compatibles en algún momento de ese contexto; en **uno** hay centros por fotograma, pero las restricciones de hipótesis impiden conectarlos de la forma requerida. Esto refuerza el diagnóstico de detecciones insuficientes, sin demostrar que las asociaciones restantes sean fáciles de elegir.

La decisión de E011 eliminó también los centros principales de Harmonic, además de sus asociaciones. CELLECT y gaussianas habían sido usados como **complementos** en E008/E009; su cobertura como conjunto completo de detecciones no estaba demostrada. Por ello, E011 no permite concluir que construir trayectorias desde detecciones sea una mala arquitectura, ni constituye una comparación controlada con E010: cambiaron tanto los videos como las fuentes de detección.

La siguiente dirección propuesta es **recuperar los centros de los detectores de Harmonic como base, conservar las propuestas complementarias y construir las asociaciones desde cero con el generador nuevo**. Primero habría que medir cobertura en los mismos 48 videos con el protocolo ya fijado. Es una hipótesis de trabajo: todavía no se ha medido la cobertura de esos centros en esta muestra. No se relajaron umbrales ni se inició un nuevo notebook durante esta revisión.

Resultados agregados: [`E011_completed.json`](../results/E011_completed.json) y [`E011_bottleneck_audit.json`](../results/E011_bottleneck_audit.json). Verificador: [`verify_detection_dag_result.py`](../scripts/verify_detection_dag_result.py), ejecutado con `src` y `scripts` en `PYTHONPATH`. Las correspondencias detalladas con anotaciones permanecen en `outputs/E011_bottleneck_details.json`, excluido de Git.
