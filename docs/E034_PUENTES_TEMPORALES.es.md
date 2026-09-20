# E034: reconstruir trayectorias con evidencia temporal

E033 corrigió 93 decisiones de asociación en la prueba de candidatos, pero su integración mantuvo el puntaje del grafo completo. Aquella integración conservaba todos los nodos y grados: no podía recuperar células ausentes ni unir fragmentos.

E034 permite unir un final de trayectoria con un inicio posterior mediante detecciones complementarias de E031. El protocolo se fija antes de evaluar: separación de 2 a 4 fotogramas; una detección en cada fotograma intermedio; distancia a la interpolación de hasta 4 µm; desplazamiento máximo de 8 µm por fotograma; distancia superior a 3 µm a los nodos originales del mismo fotograma. Se exige al menos 1 µm de diferencia frente al segundo candidato espacial y se rechazan extremos con varias propuestas. Ningún donante puede reutilizarse. No se eliminan conexiones originales.

Se comparan control, reconstrucción geométrica y reconstrucción con filtro visual. Este último exige coseno mínimo de 0.8 entre todos los pares consecutivos del puente, incluidos los extremos. Se utiliza el encoder E033 entrenado en el otro embrión. El filtro es una hipótesis fijada, no un umbral optimizado con estos resultados. No se consultan anotaciones para generar propuestas.

La preparación y la evaluación oficial se ejecutan en CPU. GPU solo calcula las representaciones de los donantes seleccionados; las representaciones de los extremos ya existen. No hay entrenamiento nuevo. Si no hay propuestas, no se lanza GPU.

Se evaluarán los grafos completos de los 16 videos reutilizados, contando conexiones correctas, incorrectas y ausentes, además de nodos añadidos y puntaje. Esto sigue siendo calibración: la procedencia de entrenamiento del detector público no está verificada. Una mejora local no equivale a una mejora del leaderboard.

Limitaciones: los puentes solo reparan fragmentos con ambos extremos presentes. No crean linajes sin anclas ni corrigen directamente divisiones. Los tres tests cubren la reconstrucción válida, rechazo visual, fotogramas ausentes, ambigüedad y protección de extremos ya conectados.

La preparación v1 terminó con 343 propuestas y 792 donantes únicos. El proceso CPU tardó 372.58 segundos, incluyendo búsqueda de archivos montados. Antes de empaquetar la evaluación se acotó esa búsqueda a los directorios de montaje para evitar recorrer los fragmentos de imagen; no cambió el algoritmo ni se reconstruyeron los notebooks ya lanzados. Los payloads y recibos preservan el código exacto de cada etapa.

## Resultado completo

| Variante | Puntaje local | Puentes aceptados | Nodos añadidos | Cambio enlaces correctos | Cambio enlaces incorrectos |
|---|---:|---:|---:|---:|---:|
| Harmonic | 0,900752960 | 0 | 0 | 0 | 0 |
| Geometría | 0,901013522 | 342 | 791 | +4 | +1 |
| Filtro visual | 0,900971013 | 145 | 359 | +3 | +1 |

El control tiene 6.674 enlaces correctos, 485 incorrectos y 501 omitidos según la métrica oficial. Los enlaces correctos adicionales aparecen únicamente en el embrión 6bba. En 44b6 no cambian los recuentos de enlaces correctos/incorrectos, aunque sí la cantidad de nodos. Las divisiones permanecen en 3 correctas, 4 falsas y 3 omitidas en todas las variantes. Las anotaciones son escasas: los nodos nuevos sin correspondencia anotada no deben interpretarse automáticamente como falsos positivos.

El filtro visual acepta menos puentes, pero también pierde uno de los aciertos adicionales y no elimina el error adicional medido. Por tanto, no demuestra ventaja sobre la reconstrucción geométrica. El cambio es demasiado pequeño para promover esta variante o atribuirle progreso hacia la medalla de plata. No se hará un barrido de umbrales con este conjunto reutilizado y no se envió submission.

La inferencia GPU terminó en 17,14 segundos de proceso; la evaluación CPU en 107,81 segundos. Los tiempos no incluyen todo el arranque/exportación facturable de Kaggle. Los tres notebooks v1 terminaron correctamente y no quedan ejecuciones pendientes. [Recibo y recuentos](../results/E034_completed.json).

Esta prueba cierra la hipótesis de puentes cortos con filtro visual fijo. La siguiente línea propuesta es evaluar secuencias de división con continuidad de ambas hijas; ni E033 ni E034 corrigieron explícitamente esos eventos. Esa línea todavía no está implementada ni lanzada.
