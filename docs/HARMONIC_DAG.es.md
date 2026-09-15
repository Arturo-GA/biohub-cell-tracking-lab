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

**Lanzado:** [Biohub Lab Harmonic Detection DAG](https://www.kaggle.com/code/jarturo/biohub-lab-harmonic-detection-dag), versión 1, el 15 de septiembre de 2026 a las 03:09 UTC (14 de septiembre en Lima). Estado inicial observado: **QUEUED**. Pasaron **77 pruebas locales**, incluidas las cinco nuevas. Recibos: [`E012_launch.json`](../results/E012_launch.json) y [`E012_preflight.json`](../results/E012_preflight.json). La cobertura real está pendiente; se revisará cuando Arturo avise que terminó.
