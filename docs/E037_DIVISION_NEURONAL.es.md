# E037: representaciones neuronales para divisiones

Se prueba si el encoder volumétrico temporal entrenado en E033 aporta información que no capturan las seis medidas locales de intensidad usadas en E035. E036 mostró que reducir un coste de movimiento puede empeorar el tracking. Por eso esta prueba primero exige evidencia de clasificación en desarrollo, antes de integrar un nuevo criterio al grafo.

Se reutilizan exactamente los ejemplos y etiquetas de E035: 40 videos de entrenamiento con 20 positivos y 9.372 negativos, y ocho de desarrollo con cuatro positivos y 1.116 negativos. No se seleccionan otros videos, instantes ni divisiones. Los centros perturbados de E035 y sus nueve nodos de contexto se mantienen. La preparación solo genera recortes de esos nodos; cada recorte contiene tres tiempos y 16³ vóxeles a resolución isotrópica, con la misma normalización por cuantiles y log1p de E033.

Cada uno de los dos encoders E033 produce su propio conjunto de representaciones. Para cada encoder se entrenan cabezas independientes sobre los 40 videos y se elige el criterio exclusivamente en desarrollo. Los videos usados para entrenar los encoders son distintos de esos 48 videos. En una evaluación posterior del grafo se usará cada pareja encoder/cabeza exclusivamente en el embrión que el encoder no utilizó al entrenarse. No se mezclan las representaciones de ambos encoders dentro de una cabeza.

Se comparan geometría (22 dimensiones), apariencia neuronal inicial (22 + 192) y secuencia neuronal (22 + 576). Las características son simétricas al intercambiar las hijas. Las cabezas ExtraTrees conservan exactamente los parámetros de E035: 256 árboles, mínimo tres muestras por hoja, fracción de características 0,8, ponderación equilibrada y semilla 350920. La referencia geométrica debe reproducir su AP anterior.

“Inicial” utiliza las representaciones de la madre y las dos primeras posiciones de las hijas; “secuencia” utiliza nueve posiciones a lo largo de las tres trayectorias. Cada representación ya incorpora tres imágenes consecutivas por diseño del encoder E033. Por tanto, la variante inicial no es una ablación de imagen estrictamente estática; la comparación mide el contexto adicional de las trayectorias.

Se conserva el umbral de desarrollo de E035: al menos dos verdaderos positivos aceptados y ningún falso positivo conocido. Adicionalmente, antes de correr se exige que la AP neuronal supere en al menos 0,05 la referencia de apariencia local (0,4425595), y en 0,05 la AP del mismo modelo con las representaciones de nodos mezcladas dentro de cada video de desarrollo. Si varias variantes cumplen, se elige la de mayor AP por encoder; en empate se prefiere la secuencia. La mezcla visual es un control diagnóstico, no una fuente de entrenamiento.

Con cuatro positivos en desarrollo la incertidumbre sigue siendo grande: superar la condición no equivale a demostrar generalización ni mejora del leaderboard. El encoder original usa candidatos de un detector público cuya procedencia de entrenamiento no está completamente verificada. La calibración posterior seguiría siendo reutilizada y condicionada a esa base.

La preparación y las cabezas se ejecutan en CPU. GPU solo extrae las representaciones con los encoders congelados; no hay entrenamiento neuronal nuevo. No se cargan imágenes ni etiquetas de calibración en estas tres etapas. Los tres tests verifican simetría, uso de historia temporal, casos vacíos y que empates con negativos o un único positivo no superen la condición.

La cuarta etapa queda preparada, pero solo se lanza si alguna pareja encoder/cabeza supera la condición. Para aislar la aportación visual mantiene los candidatos y la integración de E035: añade hijas sin padre y conserva divisiones existentes. Los embriones sin una cabeza aceptada mantienen el control. Reutiliza las representaciones ya calculadas en E033 en los nodos de Harmonic, sin nueva GPU. No es todavía una integración con las reasignaciones y donantes de E036. Si la representación no demuestra utilidad, no se justifica ampliar esa integración.

## Resultado final

| Representación | AP desarrollo con encoder entrenado en 6bba | AP desarrollo con encoder entrenado en 44b6 |
|---|---:|---:|
| Geometría, referencia repetida | 0,405893 | 0,405893 |
| Neuronal inicial | 0,228227 | 0,324741 |
| Neuronal de secuencia | 0,143437 | 0,281595 |

La referencia anterior de apariencia local E035 es 0,442560. Las cuatro variantes neuronales quedan por debajo de ambas referencias. Ninguna permite aceptar al menos dos positivos de desarrollo sin aceptar algún negativo conocido, por lo que todas fallan también el umbral original, independientemente de las condiciones adicionales de AP.

Con representaciones de nodos mezcladas dentro de cada video, las AP de inicial/secuencia fueron 0,144039/0,060599 para el encoder entrenado en 6bba y 0,050034/0,051749 para el entrenado en 44b6. La mezcla reduce el rendimiento en este conjunto, pero las representaciones originales no aportan una mejora sobre las referencias. Con solo cuatro positivos de desarrollo no corresponde convertir estas diferencias en una conclusión general sobre todos los encoders o todas las divisiones.

La prueba conserva las capacidades de asociación de E033 como un resultado distinto: que ese encoder mejorara decisiones padre–hijo no implicaba que clasificara mejor divisiones. Se rechaza esta transferencia mediante las cabezas probadas y no se realiza un barrido de parámetros. No se evaluó un nuevo CSV ni se obtuvo un nuevo puntaje local o de leaderboard en E037. La cuarta etapa permanece empaquetada, sin ejecución.

Se prepararon 12.698 recortes y se calcularon sus representaciones con ambos encoders congelados. Tiempos de proceso: preparación CPU 358,53 s; inferencia GPU 7,57 s; seis cabezas CPU 60,00 s. No incluyen todo el arranque/exportación de Kaggle. Los seis modelos se descargaron y sus hashes se verificaron frente al manifiesto. Los tres notebooks v1 terminaron correctamente, no hubo submission y no quedan trabajos pendientes. [Resultado estructurado y procedencia](../results/E037_completed.json).
