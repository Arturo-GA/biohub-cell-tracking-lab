# E032 — asociación temporal sobre detecciones reales

Prueba CPU que reutiliza las características neuronales obtenidas por E031. Entrena cuatro cabezas pequeñas: apariencia y geometría, cada una en las dos direcciones de separación por embrión. El detector permanece congelado; **no es todavía entrenamiento contrastivo del encoder volumétrico**.

## Diferencia respecto de E030

Los candidatos son máximos reales del detector, incluidos vecinos difíciles. No se crean desplazando centros anotados. Para cada célula anotada representada, se consideran hasta 12 progenitores detectados del fotograma anterior, a un máximo de 20 µm. La evaluación requiere al menos dos candidatos y que el progenitor verdadero esté representado; se informa también cuántos enlaces quedan fuera por fallos de detección/propuesta.

Un emparejamiento uno a uno a 7 µm conecta detecciones y anotaciones para construir supervisión. Las hipótesis alternativas dentro de 7 µm del progenitor anotado quedan sin etiqueta durante entrenamiento, porque pueden ser duplicados de localización. Otras hipótesis fuera de esa tolerancia pueden ser negativas de **esa relación parental**, sin clasificarse como células falsas o fondo. En evaluación se compara el ranking completo de candidatos, incluyendo las hipótesis ambiguas.

## Entrenamiento y comprobaciones

- Modo de detector fijado: `batch_bn`, como la semántica de la demo pública. No se selecciona usando el resultado de E031.
- Se entrena con los ocho videos `44b6` y se evalúan los ocho `6bba`; luego se invierte. Ninguna cabeza aprende con etiquetas del embrión donde se evalúa.
- Características: diferencia absoluta y producto de vectores neuronales de 64 dimensiones normalizados por vector, más desplazamiento físico y distancia. No se usa el progenitor GT para calcular movimiento ni se ajusta normalización con el conjunto evaluado.
- MLP de 64 y 32 unidades, 1.000 pasos CPU fijados; sin selección de checkpoint por validación. Control equivalente que recibe solo geometría.
- Se informa acierto al elegir progenitor frente a vecino más cercano y frente a características visuales permutadas entre ejemplos. Se guardan las cuatro cabezas.
- Dos pruebas de supervisión aprobadas: los duplicados cercanos permanecen sin etiqueta y no se inventa un progenitor verdadero cuando no fue detectado.

La evidencia sigue condicionada a la procedencia desconocida del detector público, a la reutilización de la cohorte de calibración y a los enlaces representados. Un ranking mejor no equivale a un score de grafo mejor: solo justificaría integrar la cabeza en el decodificador y evaluar el CSV completo. No se envía automáticamente al leaderboard.

## Resultado completo

Las cuatro cabezas terminaron sus 1.000 pasos en CPU. El script completo duró 26 segundos. Se descargaron los pesos y se verificaron sus cuatro hashes y la carga estricta de parámetros. Dos pruebas de supervisión aprobadas.

| Control | Aciertos sobre 6.725 decisiones |
|---|---:|
| Vecino más cercano | 6.118 |
| Cabeza geométrica aprendida | 6.021 |
| Cabeza con apariencia | 6.021 |
| Cabeza con apariencia permutada | 5.996 |

La apariencia gana 20 decisiones frente a la cabeza geométrica en un embrión y pierde 20 en el otro. Pierde frente al vecino más cercano en ambas direcciones. Hay sensibilidad pequeña a la imagen, pero no una ventaja útil demostrada.

**Decisión:** no integrar estas cabezas en un submission ni aumentar sus pasos como siguiente experimento. El resultado no descarta aprender representaciones temporales directamente desde los volúmenes: aquí solo se entrenó una cabeza sobre características tempranas de un detector congelado. Además, la supervisión excluye alternativas cercanas al progenitor que son ambiguas; resolver esos casos requiere mejores identidades/etiquetas o un objetivo temporal que aproveche secuencias reales, no etiquetarlos arbitrariamente como negativos.

Recibo completo: `results/E032_completed.json`. No utiliza GPU adicional ni Colab. Sin ejecuciones pendientes ni mejora de leaderboard demostrada.
