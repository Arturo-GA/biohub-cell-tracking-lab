# E013: modelo de eventos sobre detecciones reales

**Actualización de cómputo, 15 de septiembre:** la ejecución local usa la GPU de la laptop y Kaggle CPU, siguiendo el [plan de migración](LOCAL_KAGGLE_CPU.es.md). La implementación y los notebooks versión 1 descritos aquí preceden esa adaptación. El control terminó la inferencia y falló en la validación de una coordenada; su exportación corregida ya se evaluó en CPU sin repetir la inferencia: **0,909257 en los 48 videos**, con 3 divisiones correctas, 22 falsas y 30 omitidas. El CSV descargado coincide exactamente con la corrección local y se reprodujo el agregado. Es la referencia de desarrollo para la comparación emparejada, no un score del leaderboard. [Resultado verificado](../results/E013_cpu_control_completed.json).

E012 encontró candidatos compatibles con 33/33 divisiones y 24/25 continuaciones anotadas completas, pero todavía no eligió un grafo ni produjo un score. E013 implementa y entrena el selector que faltaba. El mejor resultado público propio sigue siendo **0.946** hasta una nueva submission evaluada.

## Partición y alcance

La partición se fija por SHA256 del nombre del video dentro de cada grupo de adquisición, con semilla `biohub-events-v1`. Se utilizan **48 videos para ajuste**, **16 para calibración** y **los 48 de E012 para evaluación del selector**. Quedan 79 fuera de esas etapas y se conservan las ocho exclusiones anteriores. Manifiesto: [`event_graph_split.json`](../baseline/event_graph_split.json).

Los videos de ajuste y calibración son nuevos respecto al diagnóstico E011/E012. No son imágenes inéditas para los detectores públicos: el segundo detector fue entrenado sobre los 199 videos. Además, los 48 de evaluación ya orientaron el diseño del generador. La comparación es **desarrollo condicional**, no validación independiente de todo el sistema. No se seleccionan checkpoints ni umbrales con las anotaciones de esos 48 videos.

## Implementación

Se generan las detecciones combinadas de E012, con sus parámetros conservados, en los 64 videos de ajuste/calibración. Para evaluar se reutilizan los 48 grafos de E012, fijados por checksum. No se inyectan centros anotados como candidatos.

Cada detección recibe 64 características visuales congeladas: 32 canales de cada UNet público. Se usa una sola vista de ventanas de dos fotogramas y muestreo trilineal en la posición nativa del centro. Los vecinos, los grafos y las características de los 112 videos se guardan y se fijan por hash antes de leer anotaciones para este experimento.

La red nueva tiene cuatro capas de atención local, anchura 64, y vecinos espaciales del fotograma actual, anterior y posterior. Aprende tres salidas: asociación madre-hija, división explícita en dos hijas y calidad de localización. La cabeza de división es simétrica al intercambiar hijas. El entrenamiento usa ventanas de diez fotogramas, 6.000 pasos AdamW, precisión mixta en GPU y muestreo alternado de ventanas de división y ventanas con enlaces anotados. Los detectores permanecen congelados.

Las correspondencias supervisadas requieren una anotación a un máximo de 7 µm y una ventaja de al menos 1 µm respecto a la segunda anotación más cercana. Un enlace negativo necesita un progenitor anotado incompatible; una división negativa necesita al menos una hija con progenitor anotado incompatible. La ausencia de una segunda hija anotada no demuestra ausencia de división. Los candidatos sin correspondencia no son negativos automáticos. La calidad de localización solo se supervisa donde existe correspondencia. Se registran por separado ejemplos y divisiones biológicas únicas representadas.

El checkpoint se elige por pérdida en ventanas fijas de los 16 videos de calibración. Después se fijan umbrales de logits por F1 para enlaces y F0.5 para divisiones, usando todos sus candidatos con etiqueta conocida. Estos valores describen candidatos supervisados correlacionados, no una precisión poblacional sobre células sin anotación.

## Selección del grafo

En evaluación se puntúan **todas las parejas elegibles** de E012. Para limitar el tamaño de la optimización se retienen hasta cuatro continuaciones por madre y dos divisiones por madre que superen el umbral. Esta poda ocurre después de puntuar; la cobertura de E012 no se atribuye al conjunto podado ni al grafo final.

La selección resuelve programas lineales enteros sobre ocho transiciones con solapamiento: compromete cuatro transiciones por avance y conserva las entradas ya elegidas en la frontera. Hay variables de activación de centros, nacimiento, muerte, continuación y división binaria. Cada célula tiene como máximo un progenitor y cada madre elige una continuación, una división o muerte. Se penalizan centros simultáneos a menos de 3 µm sin prohibirlos rígidamente. No se permiten nodos aislados.

La ganancia de una división combina el promedio de sus dos logits de enlace, relativo al umbral, con dos veces el margen de su cabeza de división. Así, añadir una segunda arista no duplica automáticamente la recompensa del enlace. Los costes de nacimiento/muerte son 0.7 y la penalización por duplicado cercano es 2.0. Son decisiones fijadas antes de evaluar; no se ajustan con los 48 videos.

Cada ventana tiene un límite de cinco segundos y una tolerancia relativa de 2 %. Se comprueba la factibilidad entera antes de aceptar una solución. Si no existe una solución factible disponible, se registra expresamente el uso de una selección voraz factible. Se guardan estado del solver, brecha, candidatos y decisiones por ventana. El procedimiento no certifica un óptimo global del video completo.

## Dos ejecuciones de Kaggle

Kaggle aceptó ambos notebooks privados, **versión 1**, el 15 de septiembre de 2026 a las **14:35 UTC**, con estado inicial **QUEUED**. Tras un rechazo inicial de la revisión automática, Arturo autorizó expresamente este envío y la subida se completó. No se consultará nuevamente su progreso hasta su aviso. Recibos: [`E013_launch.json`](../results/E013_launch.json) y [`E013-control_launch.json`](../results/E013-control_launch.json). El control ya se recuperó y evaluó mediante CPU Control Recovery v2; los resultados del candidato siguen pendientes.

Para evitar acumular preparación, entrenamiento e inferencia de ambos sistemas en una sola sesión, el experimento tiene dos notebooks privados:

- [Modelo nuevo](https://www.kaggle.com/code/jarturo/biohub-lab-learned-event-graph): prepara características, entrena, calibra y genera/evalúa el CSV de los 48 videos.
- [Control Harmonic](https://www.kaggle.com/code/jarturo/biohub-lab-event-graph-control): ejecuta el Harmonic completo congelado sobre exactamente los mismos 48 videos y calcula la misma métrica oficial.

El CSV del candidato y sus predicciones se fijan por hash antes de acceder a las anotaciones de evaluación. Se usa la métrica oficial conservada en el commit `075fc5f5a52d11077f9dc2b074644618f26939e2`, con los estimados de población del GEFF, no con el número de anotaciones parciales como sustituto. La comparación emparejada se hará cuando estén disponibles ambas salidas. Ninguno envía automáticamente una submission.

Se guardan características por video, etiquetas, `last.pt`, `best.pt`, historial, calibración, logits podados, grafos finales y métricas. La generación admite reutilización de videos completos tras verificar sus recibos; el entrenamiento guarda checkpoints, aunque no implementa reanudación automática del estado del optimizador. Datos y pesos permanecen fuera de Git. No se crea monitor: Arturo avisará cuando terminen.

## Verificación previa

Pasaron **86 pruebas**, incluidas nueve nuevas: etiquetas desconocidas/ambiguas, partición reproducible, simetría de hijas y propagación de contexto, decisión temporal distinta de la mejor arista local, conservación de restricciones entre ventanas, alternativa factible explícita, puntuación de todas las parejas y entrenamiento/calibración/CSV completos en un caso sintético.

La comprobación adicional usa los dos checkpoints públicos reales sobre cinco imágenes sintéticas. Verifica 64 canales visuales, sensibilidad al desplazamiento de un píxel nativo, bloqueo de asociaciones públicas y de lectura de anotaciones, selección temporal factible y CSV entero válido. También verifica los hashes de los 48 grafos reutilizados y de todos los archivos empaquetados. Es evidencia técnica, **no una estimación de rendimiento**. Recibo: [`E013_preflight.json`](../results/E013_preflight.json).

Código: `event_data.py`, `event_model.py`, `event_train.py`, `event_inference.py` y `event_solver.py` en `src/biohub_lab`; ejecución y empaquetado en `scripts/event_graph_runner.py`, `event_control_runner.py` y `build_event_graph_notebook.py`.
