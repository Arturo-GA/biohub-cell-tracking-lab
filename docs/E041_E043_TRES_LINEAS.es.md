# E041–E043: resultados de las tres líneas

Las tres pruebas se ejecutaron en orden y terminaron. Ninguna cumplió todos sus
criterios de promoción. **La señal más útil fue la supervisión densa externa:**
mejoró la recuperación de centros con tolerancia estricta, pero perdió cobertura
frente al campo estático a 7 µm. No hubo nueva submission.

## Resumen

| Experimento | Resultado | Decisión |
|---|---|---|
| E041, supervisión densa NIS3D | 227 centros a 3 µm frente a 194 del mejor control a esa distancia; 321 a 7 µm frente a 354 del estático | Conservar la señal; no promover el detector completo |
| E042, detalle XY nativo | Error 3,795601 µm frente a 3,767027 µm con XY reducido | No promover esta adaptación de resolución |
| E043, hipótesis de linaje con incertidumbre | Score local 0,893002 / 0,891838 frente a 0,900753 del control | Rechazar ambas variantes |

Los resultados E041/E042 son de componentes. Solo E043 calcula el seguimiento
completo con la métrica oficial. No son puntajes del leaderboard.

## E041: supervisión densa externa

Fuente: [NIS3D, NeurIPS 2023](https://github.com/yu-lab-vt/NIS3D),
[registro oficial 11456029](https://zenodo.org/records/11456029), CC-BY-4.0.
El archivo tiene 3.296.003.909 bytes y MD5
`f229013526645c79d31bc58ba7f12be7`.

Se prepararon 512 recortes por cada uno de los seis volúmenes originales, sin
usar como volúmenes adicionales las particiones sugeridas del mismo dataset.
Las fuentes contienen 20.939 centros con confianza de al menos 2/3; no se midió
cuántos centros únicos quedaron cubiertos por los 3.072 recortes muestreados.
Las regiones indefinidas y de confianza 1/3 se enmascaran con margen y no se
convierten en fondo. Se normaliza la intensidad y el tamaño nuclear mediano.
El remuestreo conserva la forma física usando el espaciado de cada volumen.
Las coordenadas siguen la transformación exacta de `ndimage.zoom`, incluyendo
el redondeo de la forma de salida.

Modelo propio de 353.921 parámetros, mapas gaussianos de centros, pérdida BCE
ponderada por `1 + 20*target`, 4.000 pasos de lote 8. No se ajusta con etiquetas
de Biohub. Evaluación en cinco fotogramas de cada uno de los 16 videos de
desarrollo de E038: 80 imágenes y 488 centros anotados. Comparación con topes de
128, 256 y 512 candidatos y emparejamiento uno a uno.

Criterio fijado a 256 candidatos: superar a cada control E039 a 3 µm en ambos
embriones, sin perder recuperación a 7 µm frente a ninguno.

| Método, tope de 256 candidatos | A 3 µm | A 7 µm |
|---|---:|---:|
| NIS3D denso | 227 | 321 |
| Máximos de imagen | 194 | 319 |
| Campo estático E039 | 177 | 354 |
| Campo temporal E039 | 175 | 344 |

A 3 µm mejora en ambos embriones: 69 frente a 59 y 158 frente a 135 del mejor
control a esa distancia. A 7 µm pierde frente al estático: 84 frente a 93 y
237 frente a 261. A 512 candidatos obtiene 297/437, frente a 268/423 de máximos
de imagen; esto es diagnóstico y no cambia el criterio fallido.

Auditoría descriptiva posterior: gana a los máximos a 3 µm en 10 videos, empata
en 5 y pierde en 1. Bootstrap agrupado por video y estratificado por embrión:
intervalo percentil 95% de +11 a +53 centros para la suma en 16 videos. Es
material descriptivo sobre desarrollo reutilizado, no validación independiente.
No se comparó directamente con Harmonic en esta cohorte.

Se verificaron 4.000 actualizaciones y cambio L2 de primera capa 0,573890.
[Resultados](../results/E041_completed.json),
[auditoría por video](../results/E041_paired_audit.json),
[revisión visual puntual](../results/E041_target_visual_check.json).

## E042: resolución con referencia de la célula anterior

Se toman los ejemplos E038 que tienen padre consecutivo anotado y no tienen
una bifurcación anotada de ese padre: 4.803 pares de ajuste y 1.964 de desarrollo,
en 40 y 16 videos respectivamente. Tener una sola hija anotada no prueba que
no exista otra hija sin anotar.

Dos canales: recorte actual y consulta centrada en el padre anterior. Ambos
brazos reciben las mismas consultas, objetivos y campo de visión. Se compara
XY nativo en recortes 16×64×64 contra XY4 interpolado a esa misma forma, con
coordenadas alineadas. La preparación verificó igualdad exacta entre las
muestras decimadas y los recortes E038. Las consultas usan posiciones anotadas:
es una prueba condicional, no seguimiento libre con detecciones propias.

Misma arquitectura de 450.179 parámetros, semilla, 3.000 pasos de lote 16 y
pérdida SmoothL1 de desplazamiento. Sin reflejos, para evitar cambios de medio
vóxel como factor adicional. Criterio: error medio por video nativo al menos
10% menor que el reducido y mejora en cada embrión.

| Entrada | Error medio por video (µm) |
|---|---:|
| Reducida, consulta correcta | 3,767027 |
| Nativa, consulta correcta | 3,795601 |
| Reducida, consulta permutada | 5,938968 |
| Nativa, consulta permutada | 5,929244 |
| Sin corregir la propuesta inicial | 6,181716 |

La versión nativa empeora ligeramente en ambos embriones: 3,630717 frente a
3,599230 en 44b6; 3,960485 frente a 3,934823 en 6bba. La permutación dentro de
cada lote de inferencia indica dependencia de la referencia temporal. No es
una comparación con una red entrenada sin referencia. Tampoco permite
extrapolar el resultado a todas las arquitecturas o a detección completa.

Ambas redes realizaron 2.997 actualizaciones de 3.000 intentos con AMP; cambios
L2 de primera capa 0,403729 y 0,400710.
[Resultados y filas por video](../results/E042_completed.json).

## E043: selección de linajes con incertidumbre

Implementación propia inspirada en la línea de seguimiento con incertidumbre.
Se revisó [NabaviLab](https://github.com/NabaviLab/bayesian-transformer-cell-tracking),
pero no se copia su código ni se reproduce su Transformer bayesiano. Su
resolución codiciosa de divisiones no comparte la exclusión de hijas con las
asociaciones simples; aquí se impone esa exclusión en una optimización entera.

Se conserva el conjunto de nodos y las hijas con enlace entrante. Todos los
nodos del fotograma anterior pueden ser madres. Cada hija tiene un padre y
cada madre hasta dos hijas. Se permiten cambios en el número de divisiones.
El costo combina movimiento, apariencia E033 y contexto pasado/futuro del
grafo inicial. Cada par de fotogramas se resuelve por separado; no se optimiza
una trayectoria global en el tiempo.

Las nuevas divisiones reciben una penalización log-odds según la frecuencia
de bifurcación del grafo original del video, acotada entre 0,0001 y 0,2. Activar
una madre antes sin hijas se penaliza según la fracción de madres activas.
No se usan anotaciones para ajustar estas penalizaciones. La formulación
inicial con número fijo de divisiones se retiró antes de lanzar E043 porque
podía impedir recuperar mitosis omitidas.

Se calculan una solución directa y tres soluciones con perturbaciones Gumbel
de escala 0,5. El consenso usa log-odds de frecuencia frente al estado sin
evento de cada madre, con suavizado 0,1. Estas frecuencias no son probabilidades
bayesianas calibradas. Cada resolución tiene un límite de tres segundos; el
código contempla un respaldo si no prueba optimalidad.

Criterio previo: ganar al menos 0,002 de score, recuperar al menos una división
TP adicional y no aumentar divisiones FP.

| Variante | Score local | Divisiones TP/FP/FN | Enlaces TP/FP/FN |
|---|---:|---|---|
| Harmonic control | 0,900752960 | 3/4/3 | 6674/485/501 |
| Solución directa | 0,893001674 | 3/7/3 | 6672/490/503 |
| Consenso | 0,891838399 | 3/8/3 | 6674/488/501 |

Se cambiaron 211 y 198 enlaces. Las bifurcaciones predichas totales pasaron de
316 a 519 y 506; solo las contabilizadas por la métrica se reportan como FP.
No se recuperaron divisiones anotadas adicionales. Las 7.920 resoluciones
alcanzaron el óptimo: 1.584 directas, 4.752 perturbadas y 1.584 de consenso.
El fracaso observado no se explica por agotar el límite del solver.

[Resultados completos](../results/E043_completed.json).

## Verificación, recursos y límites

- Las ocho etapas finales terminaron como notebooks privados; se verificaron
  los hashes de los notebooks frente a sus recibos de lanzamiento.
- Siete pruebas locales pasaron. Los tres modelos se descargaron, verificaron
  por SHA256 y cargaron estrictamente; todos sus parámetros son finitos.
- GPU: 288,408 segundos de proceso, unos **4,8 minutos**, para entrenamiento e
  inferencia neuronal. Las etapas CPU finales suman 2.411,437 segundos.
- Esos tiempos no incluyen arranque/exportación, la preparación inicial fallida
  ni la corrección intermedia sustituida. No equivalen a cuota facturada.
- E041 preparación v1 falló por nombres distintos dentro del ZIP. La v2 terminó.
  Después se corrigieron anisotropía y registro exacto de coordenadas en CPU,
  antes de lanzar cualquier GPU; se conservan los recibos de las revisiones.
- Sin Colab, sin nueva submission y sin trabajos pendientes de estas tres líneas.

Las cohortes de desarrollo/calibración se han reutilizado. La pertenencia de
los datos al entrenamiento de modelos públicos anteriores no está plenamente
verificada. Ningún resultado prueba un techo del concurso ni cercanía a plata.
La evidencia nueva favorece investigar supervisión densa; no justifica enviar
las variantes actuales ni repetir ajustes de sus umbrales sobre estos resultados.

[Auditoría final](../results/E041_E043_audit.json),
[verificación de pesos](../results/E041_E043_weight_verification.json),
[resumen de las tres pruebas](../results/E041_E043_completed.json).
