# E036: hipótesis completas de división y recuperación

E035 encontró errores fuera de su espacio de candidatos: una hija sin detección y otra con un padre distinto bajo el emparejamiento diagnóstico. Sus clasificadores no superaron la condición de desarrollo y se mantienen rechazados. E036 prueba un decodificador cinemático distinto; no cambia sus umbrales ni reutiliza esas cabezas para forzar resultados.

Se comparan el control Harmonic, reasignación de padres y reasignación con recuperación de una hija. Todas las propuestas de los 16 videos se calculan y guardan antes de cargar anotaciones. La evaluación usa después la métrica oficial congelada. No se seleccionan videos, madres ni instantes a partir de los errores conocidos.

Cada hipótesis conserva una hija ya conectada y propone una segunda con tres posiciones consecutivas. Puede retirar el enlace del padre anterior de la segunda hija, siempre que ese enlace no pertenezca a una división existente. Para recuperar una hija ausente, utiliza una detección batch_bn de E031 en un fotograma y la conecta a dos nodos posteriores de Harmonic. Ese anclaje posterior también puede requerir retirar un enlace previo. Solo se permite recuperar un fotograma ausente.

La detección recuperada debe estar a más de 3 µm de los nodos originales del mismo instante, a un máximo de 4 µm de la extrapolación hacia atrás del anclaje, con una diferencia de al menos 1 µm frente a la segunda alternativa y un desplazamiento de hasta 8 µm hacia el anclaje. Las hijas originales se buscan entre las cinco más cercanas a 20 µm. Se exige historia de velocidad de la madre, hijas en lados opuestos de su posición prevista (coseno ≤ −0,25), separación inicial de 2–16 µm y aumento de separación de al menos 0,5 µm en dos fotogramas.

La decisión compara el coste de los enlaces actuales con la posición y velocidad del centro de las dos hijas bajo la hipótesis de división. La escala de movimiento es la mediana del residuo de las continuaciones de Harmonic, acotada entre 1 y 4 µm. El coste cuadrático de cada enlace antiguo se limita a 16. Se fijan penalizaciones adimensionales de 3 por división, 4 por terminación desplazada y 2 por detección añadida, más el residuo de anclaje de esta última. Son penalizaciones heurísticas, no probabilidades calibradas. Se exige ganancia de coste superior a 1, sin barrido de parámetros.

Un programa entero selecciona simultáneamente las hipótesis que más reducen ese coste. Los recursos compartidos incluyen nodos afectados y las colas de ambas hijas; dos reparaciones que los comparten no pueden aceptarse juntas. El límite de optimización es de 60 segundos por video y variante, y se informa del estado y de la brecha de optimalidad. Se verifica la solución y la validez completa del CSV.

Se conservan todas las divisiones originales y todos los nodos originales. Por tanto, esta versión puede recuperar divisiones y modificar continuaciones, pero no suprimir falsas divisiones existentes ni reconstruir hijas ausentes durante muchos fotogramas. El radio y los criterios geométricos también pueden excluir divisiones reales; un resultado negativo se limitará a estas hipótesis.

Cuatro tests verifican la reasignación de padres, conflicto entre eventos, recuperación y anclaje de un nodo, exclusión de la rama de donantes en el control de reasignación, generación de hipótesis con movimiento de separación y ausencia de propuestas sin historia. Ejecución íntegra en CPU, sin entrenamiento nuevo ni GPU. La validación sigue siendo calibración reutilizada y la procedencia de entrenamiento de los detectores externos sigue sin verificarse. No se envía submission automáticamente.

## Resultado final

| Variante | Puntaje local | Eventos seleccionados | Nodos añadidos | Cambio enlaces correctos | Cambio enlaces incorrectos |
|---|---:|---:|---:|---:|---:|
| Control | 0,900752960 | 0 | 0 | 0 | 0 |
| Solo padres | 0,900503770 | 47 | 0 | −1 | +1 |
| Conjunta | 0,900503622 | 50 | 3 | −1 | +1 |

La rama de padres retiró 44 enlaces y añadió 47. La conjunta retiró los mismos 44 y añadió 53, incluyendo tres detecciones intermedias nuevas. Los cambios medidos de aciertos y errores se concentran en `44b6_9bfa6a0a`; las métricas de los otros 15 videos permanecen iguales. No debe interpretarse que todas las reparaciones restantes sean correctas: la mayoría no cuenta con anotaciones verificables.

Las divisiones siguen en 3 verdaderos positivos, 4 falsos positivos y 3 falsos negativos. Los totales de enlaces pasan de 6.674 TP, 485 FP y 501 FN a 6.673 TP, 486 FP y 502 FN en ambas variantes. La pequeña diferencia entre ellas proviene de los tres nodos añadidos y su efecto en la métrica ajustada.

Todas las optimizaciones no vacías terminaron con estado óptimo y brecha cero. Por tanto, el resultado no se explica por haber alcanzado el límite de cálculo: las hipótesis escogidas mejoran el coste heurístico fijado, pero empeoran el tracking medido. Se rechaza este criterio cinemático y no se barren sus penalizaciones con la calibración.

El notebook privado v1 terminó correctamente en 212,36 segundos de proceso CPU, además del arranque/exportación. GPU y entrenamiento nuevo: cero. No se envió submission y no quedan trabajos pendientes. [Recibo, cambios por video y métricas](../results/E036_completed.json).

Este resultado no prueba que toda reconstrucción conjunta fracase: descarta esta formulación basada en posiciones, velocidad y penalizaciones fijas. Cualquier continuación debería aportar evidencia diferente para distinguir divisiones de cruces o errores de seguimiento, y validar esa evidencia antes de otra integración completa. No hay un experimento posterior lanzado en este cierre.
