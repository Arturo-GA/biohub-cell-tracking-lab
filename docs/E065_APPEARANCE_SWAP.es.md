# E065: reasociación por apariencia de tramos de trayectoria

E064 terminó técnicamente bien, con score 0.946 y sin mejora. El usuario pidió otro experimento y autorizó expresamente enviarlo cuando terminara. E065 cambia asociaciones, sin reutilizar la corrección de centros de E064.

## Hipótesis

Un cruce de identidad puede dejar dos enlaces plausibles por distancia pero inconsistentes con el aspecto de las células en los fotogramas vecinos. E065 calcula descriptores directamente de la imagen 3D nativa y considera intercambiar los destinos de dos enlaces. No entrena una red nueva ni interpreta células no anotadas como negativas.

Cada descriptor toma 125 muestras trilineales en una cuadrícula física de 5×5×5 puntos, entre −2 y +2 µm por eje. Normaliza la textura por media y desviación local y añade dos componentes de brillo/contraste logarítmico. Estos dos componentes se centran por fotograma para reducir cambios generales de iluminación. El descriptor del lado izquierdo promedia célula y antecesora; el derecho promedia célula y sucesora.

Se consideran parejas de células fuente separadas hasta 12 µm, con enlaces a fotogramas consecutivos. Solo se cambian enlaces individuales con contexto anterior y posterior no ramificado. La longitud de cada enlace propuesto debe ser como máximo 14 µm. Se exige reducción absoluta de distancia de apariencia de al menos 0,10, una reducción relativa y límites de longitud y aceleración bidireccional. Se aceptan primero las mayores reducciones de apariencia, con vecindarios de cuatro fotogramas disjuntos. Se conserva exactamente cada coordenada, cantidad de enlaces y grados entrantes/salientes; las bifurcaciones no se modifican.

## Variantes fijadas antes de las métricas

| Variante | Reducción relativa mínima de apariencia | Aumento máximo de longitud total | Aumento máximo de discrepancia de movimiento |
| --- | ---: | ---: | ---: |
| strict | 30 % | 0,5 µm | 0 µm |
| balanced | 20 % | 1,5 µm | 0,5 µm |
| appearance | 40 % | 3 µm | 1 µm |

Son reglas geométricas y de apariencia, no probabilidades calibradas. No se debe interpretar una reducción de esta función como evidencia automática de una asociación correcta.

## Evaluación y envío

Se reutilizan 24 videos y el control E061, fijado por SHA-256. El control mezcla Harmonic y asociación visual en ocho videos. Las predicciones se congelan antes de leer anotaciones; las métricas oficiales se ejecutan en CPU. Este conjunto ya fue usado para investigar y no constituye un holdout independiente. La selección elige la mejor variante nueva que cambie el CSV para la submission exploratoria autorizada; no implica superar al control.

La submission recompone Harmonic desde imágenes test usando GPU y luego calcula los descriptores y reasociaciones en CPU. Repite el proceso en los videos ocultos que Kaggle suministre. No contiene etiquetas, credenciales ni predicciones test precalculadas. Antes del envío se descargan los CSV para verificar hashes, cambios reales, coordenadas idénticas y grados de los nodos conservados. Un comprobante exclusivo evita repetir el envío por accidente.

Cinco pruebas locales pasan: cruce artificial reparado, apariencia idéntica sin cambios, bifurcaciones intactas imagen constante con descriptor finito y nulo, y grafo vacío. El manejo de grafos vacíos se añadió antes de la submission; no altera los grafos no vacíos de evaluación. Evaluación en curso; los resultados y el envío se registrarán al terminar.

Se añadió una vista previa CPU de las tres variantes en el test visible, sin etiquetas, usando el control completo E064 solo como caché de investigación. Su único propósito es evitar gastar GPU en un CSV idéntico; no se permite construir la submission si la variante seleccionada no cambia enlaces. El envío real vuelve a calcular todas las predicciones desde las imágenes.

## Resultado

Las tres variantes empatan exactamente con el control: 0,9198784126424887, 11.389 TP, 638 FP y 651 FN. Cambian 24, 44 y 18 enlaces respectivamente sobre los 24 videos de investigación. La vista previa del test visible devuelve cero cambios para las tres. No se lanza GPU ni se envía E065. Evaluación: 377.52 s CPU; vista previa: 53,89 s CPU. Se continúa con E066.
