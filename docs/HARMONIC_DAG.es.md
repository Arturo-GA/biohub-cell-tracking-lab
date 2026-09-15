# E012: detecciones principales de Harmonic y trayectorias nuevas

E011 representó trayectorias sin exigir tracks anteriores, pero sus detectores complementarios solo ofrecieron centros compatibles para 18 de 33 divisiones. Quince eventos quedaban fuera antes de formar parejas. E012 recupera el detector principal de Harmonic y mantiene la construcción de asociaciones del generador nuevo.

## Comparación fijada

Se usan los **mismos 48 videos de desarrollo**, el manifiesto de E011 y sus criterios: radio de coincidencia de 7 µm, mínimo 90 % de cobertura de parejas y caminos, y 85 % de caminos compatibles con las ramas anotadas. Los mínimos deben cumplirse globalmente y en cada grupo. Se mantienen radios, número de vecinos, geometría de parejas y horizonte de cuatro fotogramas del DAG. No se hace una búsqueda de umbrales.

| Variante | Detecciones | Asociaciones posibles |
|---|---|---|
| E011 | CELLECT y gaussianas ya calculados | Grafo E011 fijado, verificado por hash |
| Harmonic solo | Centros nuevos de los dos detectores temporales de Harmonic | Grafo nuevo con la configuración de E011 |
| Combinada | Todos los centros de Harmonic más centros complementarios distintos | Grafo nuevo con la configuración de E011 |

Reutilizar E011 evita repetir su inferencia de aproximadamente dos horas. `baseline/harmonic_dag_inputs.json` fija los hashes de su manifiesto anterior a la lectura de anotaciones y de **240 archivos** de propuestas, recibos y grafos. La generación abre solo las propuestas y ese manifiesto; no abre sus resultados de cobertura ni los testigos condicionados por anotaciones. El notebook adjunto puede contener esas otras salidas, pero no se usan como entradas del generador.

## Qué se toma de Harmonic

Se extraen los centros de su detector temporal de dos modelos, **antes de cualquier asociación, ILP o filtrado de tracks**. Se conservan el umbral de detección 0,965, TTA de ocho vistas XY, mezcla del segundo detector de 0,80 y guardia de retención de candidatos de 0,90. La supresión de máximos usa el valor efectivo del predictor, 3 µm; el campo 5 µm del archivo de configuración de entrenamiento no era utilizado por ese punto de entrada de inferencia.

El código de inicialización original verifica los archivos públicos y aplica sus parches. Luego el adaptador extrae las funciones de detección, conserva las instrucciones de cada ventana que producen centros y elimina el bucle de predicción de enlaces. Utiliza un lector propio de metadatos Zarr que no consulta GEFF. El promedio de características destinado al enlazador se desactiva; el TTA de logits de detección se conserva. No se ejecutan las reparaciones geométricas ni DeepCenter como filtro o fuente adicional de centros.

Los dos checkpoints se cargan una vez por proceso GPU. La ejecución usa hasta dos GPU disponibles. El paquete no contiene los pesos; adjunta los datasets públicos ya empleados por el control. El dataset DeepCenter sigue adjunto porque la inicialización original comprueba su integridad, aunque esta auditoría no ejecuta ese modelo.

## Combinación y alcance

Primero se reproduce la unión CELLECT/gaussianas de E011. Se conservan **todos** los centros principales de Harmonic y se añaden los centros de esa unión situados a más de 1,2 µm de cualquier centro principal en el mismo fotograma. Esta precedencia explícita evita comparar como probabilidades los scores de detectores diferentes. No hay límite de adiciones ni requisito de un track previo. Las coordenadas se guardan en la rejilla nativa TZYX; el origen de cada centro queda registrado.

Los centros principales no se eliminan ni se desplazan al combinar. La nueva unión puede cambiar los vecinos más cercanos y sustituir propuestas complementarias cercanas: **no se presupone que su cobertura de parejas o trayectorias tenga que mejorar**. Las tres variantes permiten medir ese efecto.

Se fijan todos los grafos nuevos antes de leer anotaciones. Después se evalúan las tres variantes sobre las mismas divisiones, incluyendo el denominador separado de eventos con ambas ramas anotadas durante cuatro fotogramas. Se conserva la cobertura por video y grupo, las diferencias respecto a E011 y los testigos de auditoría fuera de Git.

Esta muestra sigue siendo desarrollo: los grupos se han usado antes y el detector público secundario fue entrenado con estos videos. Las cifras serán **techos de cobertura condicionados por anotaciones**, no precisión, validación independiente ni score oficial. El experimento no entrena ni produce una submission. Pasar el criterio permitiría investigar cómo puntuar y seleccionar eventos; no demostraría que esas decisiones sean correctas.

## Verificación antes de ejecutar

Las cinco pruebas nuevas comprueban preservación de centros principales, escala y forma de imagen sin GEFF, extracción del código sin llamadas de asociación, comparación de tres variantes tras fijar los grafos y rechazo de entradas o grafos alterados.

Además, `scripts/verify_harmonic_dag_preflight.py` descargó/verificó ambos checkpoints públicos y sus arquitecturas, reprodujo la secuencia real de parches e hizo inferencia CPU sobre un video sintético de cinco fotogramas. Produjo diez centros principales y once combinados, bloqueando expresamente las llamadas de asociación y la lectura de GEFF. Comprobó equivalencia de las instrucciones de detección conservadas y los 240 archivos de E011. Es una prueba técnica de integración, **no evidencia de cobertura real**.

Implementación: `src/biohub_lab/harmonic_centers.py` y `scripts/harmonic_dag_runner.py`. Paquete: `kaggle/harmonic_dag`. Recibo técnico: [`E012_detector_smoke.json`](../results/E012_detector_smoke.json). El estado de lanzamiento se registra en `results/STATUS.json`; no hay monitor local ni entrenamiento automático.

**Completado:** [Biohub Lab Harmonic Detection DAG](https://www.kaggle.com/code/jarturo/biohub-lab-harmonic-detection-dag), versión 1, lanzada el 15 de septiembre de 2026 a las 03:09 UTC y comprobada completa a las 13:47 UTC. Pasaron **77 pruebas locales** antes de lanzarla, incluidas las cinco nuevas. Recibos: [`E012_launch.json`](../results/E012_launch.json) y [`E012_preflight.json`](../results/E012_preflight.json).

## Resultado: la combinación pasa el criterio de cobertura

Terminaron los 48 videos en **5.179,81 segundos, aproximadamente 1 h 26 min**, excluyendo inicialización y cola. Se conservaron los **1.237.567 centros de Harmonic** y se añadieron **1.462.003** centros complementarios, para un total de **2.699.570**. La comparación utilizó los mismos eventos, configuraciones y radios fijados en E011.

| Cobertura | E011 | Harmonic solo | Combinación |
|---|---:|---:|---:|
| Madre y dos hijas disponibles | 18/33 | 31/33 | **33/33** |
| Pareja inicial representada | 17/33 | 31/33 | **33/33** |
| Dos trayectorias sin compartir centros | 16/33 | 31/33 | **33/33** |
| Trayectorias compatibles con las ramas anotadas completas | 11/25 (44 %) | 21/25 (84 %) | **24/25 (96 %)** |
| Criterio previo al entrenamiento | No pasa | No pasa | **Pasa** |

El último renglón de cobertura tiene denominador 25 porque solo esos eventos tienen el contexto anotado completo de cuatro fotogramas. Los otros ocho no se cuentan como aciertos ni fallos de esa comprobación.

La combinación pasa también por grupo: **44b6 cubre 8/8 divisiones y 7/7 continuaciones completas**; **6bba cubre 25/25 y 17/18**, respectivamente. Harmonic solo queda por debajo del mínimo en 44b6: 7/8 divisiones y 4/7 continuaciones completas. Los complementos aportan cobertura más allá de recuperar el detector principal.

Estos resultados demuestran que existen candidatos compatibles con casi todos los eventos anotados en esta muestra. **No demuestran que un modelo pueda elegirlos entre las alternativas, ni representan un score de 0.96 en Kaggle.** La combinación representa 194.294.999 parejas posibles frente a 33.928.772 con Harmonic solo; todavía falta distinguir eventos válidos de hipótesis falsas o duplicadas. No se produjo una submission y el mejor score público propio previamente verificado sigue siendo 0.946.

## Verificación y aportación de cada fuente

Se descargaron 576 archivos de grafos, detecciones y cobertura. Se verificaron los **432 archivos nuevos fijados por hash**, los **240 archivos reutilizados**, el payload, la selección de videos, los checkpoints y configuraciones públicos y el código efectivo del detector. Se reconstruyeron **exactamente los 96 grafos nuevos** (48 por variante), incluidas matrices de vecinos, máscaras de parejas y las 12.288 consultas de trayectorias muestreadas. También se comprobó que todos los centros principales permanecieran intactos en la combinación.

Se recalculó la cobertura de las tres variantes con anotaciones cuyos checksums estaban fijados. Coincidieron todos los recuentos, testigos de cobertura y decisiones del criterio. La variante E011 reprodujo su resultado anterior. No se volvió a ejecutar la inferencia completa de imágenes en el equipo local.

La comparación evento por evento confirmó **cero pérdidas de cobertura** respecto a E011 o a Harmonic solo en las etapas auditadas. La combinación conserva las 11 continuaciones completas cubiertas por E011 y añade 13; conserva las 21 de Harmonic solo y añade tres. Los dos eventos que Harmonic solo perdía inicialmente carecían de dos detecciones distintas compatibles con las hijas; la combinación los recupera.

El único evento que sigue sin continuación completa compatible tiene **centros ausentes en algún fotograma posterior de las ramas anotadas**. Su madre y sus hijas iniciales sí están representadas y hay dos caminos geométricamente posibles; esos caminos no bastan para seguir ambas ramas anotadas durante todo el horizonte. Se mantiene la configuración fijada para pasar al trabajo de selección.

Recibos agregados: [`E012_completed.json`](../results/E012_completed.json) y [`E012_coverage_audit.json`](../results/E012_coverage_audit.json). Verificador: [`verify_harmonic_dag_result.py`](../scripts/verify_harmonic_dag_result.py). Las correspondencias detalladas permanecen en `outputs/E012_coverage_details.json`, excluido de Git.

## Decisión y siguiente implementación

Se conserva la configuración combinada de E012 como base para desarrollar la **puntuación aprendida de centros, parejas de hijas y continuaciones temporales**, seguida de una selección global que evite reutilizar detecciones y asigne como máximo un padre por célula y dos hijas por madre. El objetivo siguiente es elegir un grafo correcto entre las hipótesis que ahora sí están disponibles.

El entrenamiento del selector debe separar videos de ajuste, calibración y evaluación. Los 48 videos de E011/E012 ya han guiado decisiones de diseño: se conservan como diagnóstico de desarrollo y no se presentan como prueba intacta. Los 143 videos fuera de este diagnóstico permiten preparar una partición del selector, pero tampoco constituyen por sí solos una validación independiente de los detectores públicos. Las anotaciones son parciales; los negativos deben derivarse de asociaciones anotadas incompatibles, sin tratar automáticamente cada hipótesis no anotada como falsa.

Antes de otra submission habrá que generar un CSV final mediante el selector, medir aciertos, falsos positivos y omisiones de divisiones con la métrica oficial, y compararlo con Harmonic en los mismos videos. Superar el criterio de cobertura habilita esta investigación; no autoriza a interpretar sus testigos condicionados por anotaciones como predicciones. **Durante esta revisión no se inició otro entrenamiento ni notebook.**
