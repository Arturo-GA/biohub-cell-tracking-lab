# E063: selección aprendida de trayectorias complementarias

## Pregunta y separación de datos

E061 localizó más centros, pero la selección manual de adiciones no mejoró el grafo. E062 recuperó dos enlaces correctos netos y agregó seis incorrectos. E063 aprende cuándo admitir una trayectoria nueva, combinando confianza del detector, probabilidades y competencia entre asociaciones, velocidad, aceleración, densidad local, distancia a los nodos originales y anclajes temporales.

El protocolo usa 40 videos para ajustar el selector y 16 diferentes para elegir su umbral. Ninguno pertenece a los 24 videos de evaluación E061/E062. Estos últimos son una evaluación de investigación reutilizada, no un conjunto independiente del proyecto. El detector E061 ya fue entrenado con los 40 videos de ajuste; los 16 de desarrollo son nuevos para ese detector. No se conoce la pertenencia de videos al preentrenamiento del checkpoint público.

Para obtener ejemplos sin ejecutar de nuevo todas las transformaciones del ensamble, el entrenamiento y desarrollo usan detecciones y asociaciones del modelo público primario, sin TTA ni la segunda semilla. Es una referencia aproximada, no una reproducción de Harmonic. La evaluación final usa las coordenadas y probabilidades reales guardadas en E062, sobre el control Harmonic/visual del proyecto. Esta diferencia de distribución se debe considerar al interpretar el resultado.

## Etiquetas y modelo

Los candidatos se generan sin anotaciones y sin los filtros manuales anteriores de confianza o aceleración. Se conservan la asignación uno a uno, la continuidad temporal y la exclusión espacial de candidatos a menos de 3 µm de un nodo original. No se sustituyen los enlaces originales ni se ocupan extremos que ya tienen enlace.

En ajuste, se integran los segmentos factibles y se utiliza el emparejamiento oficial. Solo los enlaces marcados `pred_valid` por la métrica aportan etiquetas. Una trayectoria con algún enlace evaluable incorrecto es negativa. Para ser positiva, todos sus enlaces evaluables deben ser correctos y debe recuperar al menos un enlace GT que el grafo original no acertaba. Una trayectoria que solo reproduce aciertos ya presentes es negativa para la **utilidad de añadirla**, no una célula clasificada como falsa. Si no tiene enlaces evaluables, se excluye. No se afirma que sus partes sin anotación sean correctas. Una prueba ejecutada con la métrica oficial verifica explícitamente la distinción entre un enlace correcto, uno incorrecto desde una célula anotada y otro sin supervisión.

El selector es una regresión logística regularizada con 39 características, términos cuadráticos e interacciones por pares. Se ajusta en CPU mediante L-BFGS, con pesos normalizados por video. La regularización se fija antes de evaluar. Sus salidas son puntuaciones de selección, no probabilidades de medalla ni una garantía de precisión calibrada.

Las etiquetas se obtienen en el contexto de todas las propuestas factibles y no identifican por sí solas cada perjuicio causado al emparejamiento original. Por eso la selección del umbral se hace con la métrica del grafo completo, incluyendo penalización por nodos y divisiones.

## Decisión predefinida

Se comparan en los 16 videos de desarrollo el control, las reglas E062 y nueve umbrales prefijados: 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 0.95 y 1.01 (no añadir). Se añadieron umbrales bajos al plan de selección antes de obtener etiquetas o métricas, porque recuperar un acierto nuevo es un evento poco frecuente; no cambió la inferencia neuronal ya lanzada. Se mantiene el presupuesto máximo de nodos nuevos del 1 %. El umbral se elige antes de abrir las métricas de los 24 videos finales. Debe mejorar el puntaje y los aciertos, no aumentar falsas divisiones y no empeorar ninguno de los dos embriones.

Si ningún umbral pasa desarrollo, se conserva el control. Se permite medir, como diagnóstico declarado, el mejor umbral de desarrollo que produjo cambios; no puede promocionarse a submission si falló la condición de desarrollo. En la evaluación final, la promoción exige al menos +0.001, más enlaces correctos, sin incremento de falsas divisiones y sin regresión en los cuatro grupos predefinidos.

Tras observar las primeras mejoras agregadas de desarrollo, y antes de ejecutar la evaluación final E063, se añadió una variante de selección condicionada por embrión. Usa el mismo modelo entrenado y los mismos umbrales, eligiendo por separado con los ocho videos de desarrollo de cada embrión. Exige mejorar puntaje y aciertos sin aumentar falsas divisiones; si ninguno pasa, desactiva las adiciones para ese embrión. A igual puntaje prefiere el umbral mayor. Un embrión desconocido conserva el control. Se combinan las métricas de desarrollo por video para verificar la variante antes de leer las etiquetas de los 24 videos finales. Es una extensión basada en desarrollo, no un protocolo completamente fijado antes de todas las métricas; puede sobreajustar los dos embriones conocidos. La variante global y la condicionada se comparan en la evaluación reutilizada y se someten al mismo criterio de promoción.

## Cómputo y estado

GPU únicamente para inferencia del detector y del asociador. Preparación de la unión, emparejamiento oficial, ajuste del selector, selección del umbral y evaluación se ejecutan en CPU. Los manifiestos, pesos de entrada, modelo ajustado y CSV previos a las métricas se fijan mediante SHA-256.

El ajuste convergió en 343 iteraciones. Los 40 videos aportaron 758 trayectorias supervisadas (116 útiles, 642 negativas para utilidad); 36 videos tenían al menos un ejemplo supervisado. Se excluyeron 58.556 trayectorias sin enlaces evaluables. El modelo tiene 39 características y 820 coeficientes contando interacciones e intercepto. Sus coeficientes están en `baseline/e063_selector_model.json`, verificados contra el SHA-256 del entrenamiento Kaggle.

### Resultados de desarrollo: referencia primaria simplificada

| Variante | Score | TP | FP |
|---|---:|---:|---:|
| Control | 0,753329 | 7.994 | 1.230 |
| Reglas E062 | 0,755111 | 8.017 | 1.228 |
| Selector global, umbral 0,95 | 0,755638 | 8.025 | 1.228 |
| Selector condicionado por embrión | 0,755645 | 8.023 | 1.228 |

La variante global mejoró el agregado, pero no pasó la condición por embrión: 44b6 pasó de 0,816399952 a 0,816367944. En 6bba pasó de 0,736379032 a 0,739314196. La selección condicionada dejó 44b6 intacto y usó umbral 0,95 en 6bba.

### Resultados en los 24 videos del ensamble real

| Variante | Score local | TP | FP | Nodos añadidos |
|---|---:|---:|---:|---:|
| Control Harmonic/visual | 0,919878 | 11.389 | 638 | 0 |
| Reglas E062 reproducidas | 0,919413 | 11.391 | 644 | 1.984 |
| Selector global | 0,918536 | 11.387 | 643 | 5.121 |
| Selector condicionado | 0,918708 | 11.387 | 643 | 2.129 |

Las divisiones no cambian: 3 TP, 6 FP y 8 FN. Ambas variantes aprendidas añaden un FP en `6bba_9a41d029`; en `6bba_2646afc7` pierden dos TP netos y añaden cuatro FP. Los demás videos conservan sus conteos TP/FP/FN, aunque las adiciones pueden penalizar el puntaje por cantidad de nodos. Los conteos netos no bastan para descartar recuperaciones compensadas por pérdidas; se lanzó una auditoría adicional del conjunto exacto de enlaces GT acertados, en CPU y sin modificar predicciones.

La mejora de desarrollo **no se transfirió al ensamble real**. Ninguna variante pasa promoción. No se envía una nueva submission y se conserva el leaderboard previo de 0,946. Esto rechaza este selector y esta integración, no demuestra un techo de la competencia. La diferencia entre la referencia simplificada de entrenamiento y Harmonic es una limitación del experimento; estos resultados no aíslan por sí solos su efecto causal.

Cinco pruebas locales del selector y siete de compatibilidad E061/E062 pasan. La prueba de etiquetas dispersas con la métrica oficial pasó en Kaggle. Detección y asociación consumieron 1.104,095 segundos de proceso GPU (18,40 minutos); no es una lectura de facturación/cuota de Kaggle. Preparación, ajuste, selección y evaluación fueron CPU. La auditoría final también terminó; no quedan trabajos pendientes. El total CPU de las cuatro etapas es de 849,553 segundos (14,16 minutos de proceso).


### Auditoría exacta de complementariedad

La auditoría oficial compara los conjuntos de enlaces GT acertados, no solo sus conteos. Las reglas E062 recuperan cinco enlaces y pierden tres (saldo +2). El selector global y el condicionado recuperan un enlace y pierden tres (saldo −2). Ese único enlace recuperado no lo acertaban ni el control ni las reglas E062, y aparece en `6bba_9a41d029`; en ese mismo video se pierde otro acierto. Los otros dos aciertos perdidos corresponden a `6bba_2646afc7`. Existe por tanto una señal pequeña de complementariedad, pero no una política de selección validada que la convierta en mejora total. No se exporta ninguna selección basada en GT para inferencia.

Se verificó que el control y las reglas E062 reproducen exactamente sus métricas previas, que la variante condicionada deja 44b6 intacto y que el modelo descargado conserva el hash del entrenamiento. Resultado consolidado: `results/E063_decision.json`. No se usó Colab.
