# E010: selección conjunta de centros y divisiones

E008 y E009 añadieron propuestas de detección, pero ambos empeoraron la métrica y no recuperaron divisiones oficiales adicionales. E010 cambia la unidad de decisión: selecciona un evento con dos trayectorias y sus centros durante varios fotogramas. Puede retirar centros fusionados del control, recuperar propuestas descartadas y conectar las dos hijas en una misma operación.

## Implementación

Se parte del grafo final congelado de Harmonic E001. Las alternativas de centros provienen de los archivos de propuestas de E008 v2 y E009 v2, anteriores a su inyección en Harmonic. Esos archivos ya incluyen la NMS y corroboración temporal propias de cada detector; no son todas sus activaciones crudas. `baseline/joint_inputs.json` fija por SHA256 los recibos, CSV de control y ocho archivos de propuestas.

1. Se enumeran madres con una continuación y dos antecesores. Se buscan trayectorias huérfanas que comiencen en los siguientes cuatro fotogramas. Ambas ramas deben alcanzar el cuarto fotograma y persistir dos más. Se consideran hasta cuatro huérfanas por madre, ordenadas por distancia física.
2. Los prefijos de ambas ramas pasan a ser reemplazables. Se exploran centros del control y propuestas alternativas con un beam de ocho caminos por rama, seis candidatos por fotograma y restricciones de desplazamiento. Los dos centros del cuarto fotograma son anclas fijas.
3. La imagen se normaliza por fotograma y se resta un fondo gaussiano, a media resolución XY. Se exige evidencia de dos picos desde el primer fotograma de hijas, un cambio respecto a los dos fotogramas anteriores, señal de la madre, balance aproximado de fluorescencia, separación y movimiento compatibles. Son priors analíticos iniciales, no probabilidades calibradas ni segmentación completa.
4. Se conservan hasta tres hipótesis positivas por ventana. Un programa entero maximiza su puntuación conjunta, excluyendo eventos que reutilicen recursos del grafo o el mismo centro físico. La ausencia de cambios es la alternativa de ganancia cero. El solver registra si alcanzó optimalidad o solo un incumbente factible.
5. Se reconstruye el grafo completo, se exporta un CSV de coordenadas enteras y se valida. Se ejecuta la métrica oficial sobre ese CSV congelado. No se vuelve a ejecutar el postprocesamiento de Harmonic después de la reconstrucción conjunta.

Los valores iniciales incluyen horizonte 4, persistencia 2, separación mínima de hijas 2,6 µm, paso máximo de continuación 6 µm y cambio mínimo de contraste 0,18. La configuración completa queda en `JointConfig` y en cada recibo. Se corrigió un fallo descubierto con imágenes sintéticas: usar solo el contraste mediano posterior permitía que un camino comenzara en el centro fusionado y saltara después a una segunda célula ya existente. Ahora deben existir dos picos desde el primer fotograma declarado de hijas.

## Alcance de la comparación

El inventario geométrico, calculado sin anotaciones sobre el control, contiene **12.110 ventanas** en los mismos cuatro videos de diagnóstico. No se restringe la inferencia a tiempos o coordenadas de divisiones anotadas. El módulo de reconstrucción recibe el grafo base, las propuestas y la imagen; no recibe GEFF ni etiquetas. El evaluador abre las anotaciones después de congelar las predicciones.

La comparación sigue siendo **un diagnóstico sobre entrenamiento**, no validación independiente: los cuatro videos pertenecen al entrenamiento del detector secundario público y ya se usaron en análisis anteriores. Una mejora aquí justificaría investigar transferencia; no demostraría mejora en el leaderboard. E010 no entrena una red nueva: implementa un optimizador nuevo que reutiliza las inferencias costosas ya terminadas. Se ejecuta en CPU.

Este diseño exige una trayectoria huérfana persistente cercana. No recuperará eventos sin esa ancla, ni detectará centros ausentes de ambos archivos de propuestas. La búsqueda beam y la preselección de ventanas son aproximadas; la optimalidad del solver se refiere únicamente al conjunto de hipótesis generado. Tampoco garantiza que las heurísticas de fluorescencia se transfieran entre embriones o etapas.

## Evidencia y archivos

- `joint_experiment/result.json`: estados y errores por etapa, hashes, configuración, recuentos, tiempos, métricas y diferencia frente al control.
- `joint_experiment/predictions.csv`: grafo final evaluado.
- Por video: `candidate_pool.npz`, `window_audit.jsonl`, `event_audit.jsonl`, `final_graph.npz` y `receipt.json`. Permiten reconstruir la selección, sus descartes y los conflictos.
- Los grafos y detalles espaciales quedan en outputs de Kaggle o en la carpeta local ignorada `outputs/`. En Git se guardan código, configuración, procedencia y resultados agregados.

Las pruebas cubren reemplazo de centros fusionados, rechazo de dos células preexistentes, conservación del grafo sin alternativas, persistencia, optimalidad de un pequeño problema frente a enumeración exhaustiva, conflictos espaciales, remapeo de identificadores, integridad de entradas y un recorrido de integración con imágenes Zarr sintéticas. En la prueba de integración se sustituye únicamente la métrica final y el descubrimiento del dataset; el evaluador oficial se ejecuta realmente en Kaggle.

## Referencia conceptual

[Ultrack, Nature Methods, 2025](https://www.nature.com/articles/s41592-025-02778-0) formula conjuntamente la selección de hipótesis de segmentación y sus enlaces temporales. E010 toma esa idea como orientación. El código aquí es propio y trabaja con centros, trayectorias huérfanas y eventos; no reproduce el árbol de segmentaciones ni la implementación completa de Ultrack, y no utiliza sus pesos.

Lanzado en [Kaggle](https://www.kaggle.com/code/jarturo/biohub-lab-joint-lineage-selection), versión 1, el 14 de septiembre de 2026. Estado inicial observado: **QUEUED**. Resultados pendientes. Pasaron las **63 pruebas locales**, incluidas las nueve nuevas. Recibos: `results/E010_launch.json` y `results/E010_preflight.json`. No se publica automáticamente una submission ni se mantiene un monitor de ejecución.
