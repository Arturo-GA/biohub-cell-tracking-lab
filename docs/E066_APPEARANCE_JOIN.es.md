# E066: unión de fragmentos confirmada por apariencia

Tras E065, que no cambia el test visible, se investiga otra operación: añadir un enlace entre el final de una trayectoria y el comienzo de otra en el fotograma inmediatamente siguiente. No se alteran posiciones ni se eliminan enlaces originales. No se añaden detecciones ni divisiones.

## Método

Se reutiliza el descriptor de imagen nativa descrito en E065. Cada extremo debe disponer de dos enlaces de contexto sin bifurcaciones: tres fotogramas anteriores para el final y tres posteriores para el inicio. La velocidad se estima entre los dos extremos de cada tramo, y la apariencia por mediana de sus tres descriptores.

Una unión debe cumplir distancia máxima, compatibilidad de apariencia y errores de extrapolación tanto hacia adelante como hacia atrás. Su coste suma distancia, error de movimiento y diferencia de apariencia, cada uno normalizado por su límite. Se exige que ambos extremos se elijan mutuamente como mejor opción, con margen respecto a su segunda opción. No se cambian enlaces existentes ni se conectan extremos que ya tengan el grado necesario ocupado.

Tres variantes se fijaron antes de sus métricas:

| Variante | Radio máximo | Residuo máximo de movimiento | Diferencia cuadrática de apariencia | Margen de coste |
| --- | ---: | ---: | ---: | ---: |
| strict | 8 µm | 3 µm | 0,5 | 0,20 |
| balanced | 12 µm | 5 µm | 1,0 | 0,15 |
| recall | 14 µm | 8 µm | 1,5 | 0,10 |

El contexto completo abarca seis fotogramas. Todas las aristas añadidas son consecutivas; no se añaden enlaces que salten fotogramas. Los límites son reglas exploratorias, no confianzas calibradas.

## Evaluación y selección

Se evalúa en CPU sobre los mismos 24 videos reutilizados, con el control E061 fijado por hash. La vista previa CPU usa solo imágenes test visibles y el control previamente generado para comprobar si cambia algo, sin etiquetas ni submission. Antes de observar sus métricas se fijó excluir del envío las variantes sin cambios en el test visible y elegir la mejor puntuación de evaluación entre las restantes. A igualdad se respeta el orden strict, balanced, recall. Esta exclusión evita un envío duplicado; no selecciona usando el score público.

El usuario autorizó el experimento y pidió expresamente una submission al terminar. El envío será exploratorio si no hay mejora local; el resultado se comunicará sin atribuirle una mejora no demostrada. El conjunto de 24 videos no es independiente y mezcla control Harmonic con ocho grafos de asociación visual; la submission parte de Harmonic puro.

El notebook de envío recalcula Harmonic con GPU desde todas las imágenes test disponibles, incluidas las ocultas en Kaggle. Los descriptores y uniones usan CPU. Las cachés de vista previa no forman parte del notebook de envío. Se validan hashes, coordenadas idénticas, conservación de cada enlace original, nuevas aristas y estructura temporal antes del envío único.

Cuatro pruebas pasan: unir fragmentos coherentes, rechazar apariencias incompatibles, conservar enlaces ya existentes y aceptar un grafo vacío. Evaluación y vista previa en ejecución.

## Resultado

| Variante | Score | TP | FP | FN | Enlaces añadidos |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 0.919878413 | 11389 | 638 | 651 | 0 |
| strict | 0.919878413 | 11389 | 638 | 651 | 7 |
| balanced | 0.919958706 | 11390 | 638 | 650 | 22 |
| recall | 0.920118085 | 11392 | 638 | 648 | 54 |

La variante recall gana tres TP sin FP adicionales: delta +0,000239672. Es una señal positiva en datos reutilizados, sin demostrar generalización. Las tres variantes dejan idéntico el control del test visible. No se ejecutó GPU ni se envió E066 por separado; se conserva como posible complemento. Un test visible idéntico no implica necesariamente test oculto idéntico. Evaluación 332,95 s CPU; vista previa 33,39 s CPU.
