# E058–E059: complementariedad real entre modelos

## E058: auditoría completada

Se compararon los enlaces correctos individuales con el emparejamiento oficial,
no solo los conteos TP/FP. Las variantes sintéticas E056 y E057 recuperan cero
enlaces correctos distintos de la asociación visual E023 y pierden cero. Los
CSV distintos no representan complementariedad evaluable en estos ocho videos.
Harmonic conserva cinco aciertos que visual pierde, pero pierde otros 21.

De los 150 enlaces anotados omitidos por visual, 83 tienen ambos extremos
emparejados y están en los candidatos guardados; 18 carecen de destino, 19 de
origen y 30 de ambos extremos. En los 83 alcanzables, 24 enlaces son la primera
opción de su columna, 42 la segunda y 17 opciones posteriores. Veinte tienen
probabilidad inferior a 0,05. Cinco son enlaces de división. Estas cifras
orientan el diagnóstico; no autorizan seleccionar enlaces utilizando etiquetas.

## E059: protocolo fijado antes de evaluar

Dos checkpoints públicos del dataset `bhpepper/biohub-synthetic-5fold-ensemble-v1`:
SWA y fold0, cada uno con 2.076.706 parámetros. Carga estricta, SHA-256 y prueba
de inferencia aprobados en la laptop. Se ejecutan sin el filtro de baja confianza
de E056 y sin el límite de peso sintético de 0,25. No se reentrena el detector ni
se repite Harmonic: únicamente se calculan probabilidades neuronales en GPU sobre
las detecciones originales de los ocho videos completos.

Se conservan todos los candidatos originales y se agregan los ocho mejores por
fila y columna de cada experto. Las probabilidades se remapean por coordenadas
al grafo original, sin suponer que IDs de diferentes ejecuciones coinciden.

Evaluación CPU: control visual, SWA solo, fold0 solo, mezcla geométrica 50/50
de cada experto con visual, y mezcla visual/SWA/fold0 de pesos 0,50/0,25/0,25.
Los expertos solos pueden proponer candidatos nuevos; las mezclas usan el soporte
original para no interpretar probabilidades ausentes como cero. Se mantienen
nodos y divisiones originales. Se congelan los CSV antes de calcular métricas.
Se exige reproducir exactamente el control local 0,9502832669356378 y se cuentan
aciertos recuperados y perdidos por video con el evaluador oficial.

Los ocho videos se han reutilizado muchas veces: resultados exploratorios,
sin holdout independiente. Se desconoce si el preentrenamiento público vio estos
datos. El último leaderboard confirmado sigue siendo 0,946. Los scores locales
no son estimaciones directas del leaderboard.

E059_CAPTURE completado, privado, versión 1: 147,73 segundos de proceso GPU
(la facturación de cuota puede diferir), 1.584 pares entre ambos modelos.
E059_EVALUATE lanzado en CPU, versión 1. Sin nueva submission.

## E059 completado

Evaluación CPU: 250,18 segundos. Todos los controles y hashes verificados.

| Variante | Score local | Aciertos nuevos | Aciertos perdidos |
| --- | ---: | ---: | ---: |
| Visual | 0,950283267 | 0 | 0 |
| SWA solo | 0,940964701 | 1 | 24 |
| Fold0 solo | 0,940020470 | 2 | 26 |
| Visual/SWA 50/50 | 0,949675287 | 0 | 3 |
| Visual/fold0 50/50 | 0,948898010 | 0 | 5 |
| Visual/SWA/fold0 | 0,950257969 | 0 | 2 |

Los expertos aportan dos enlaces únicos en total, porque el único recuperado
por SWA también lo recupera fold0. Uno corresponde a una columna donde el
modelo original tenía margen 0,737: limitar intervenciones a confianza baja lo
excluía. El otro es una segunda opción original. Esto muestra una complementariedad
pequeña, insuficiente en las mezclas uniformes probadas. No se promueven.

## E060: intervención selectiva, solo CPU

Cuatro variantes: SWA o fold0, con filtro de confianza o filtro conjunto de
confianza y trayectoria. Parten del grafo visual y aplican componentes completos
de desacuerdo para preservar las restricciones de asignación. Se exige ganancia
de evidencia del experto por enlace >= log(2); el filtro conjunto exige además
contexto temporal completo y curvatura <=4 micras y <= anterior+0,5 micras.
Umbrales heredados de E055 sin reajustarlos a los dos enlaces recuperados.
No hay entrenamiento del selector con las etiquetas de evaluación ni selección
por nombre de video. Lanzado privado en CPU, versión 1.

## E060 completado: no promover

| Variante selectiva | Score local | Aciertos nuevos | Aciertos perdidos |
| --- | ---: | ---: | ---: |
| SWA, confianza | 0,948523714 | 0 | 5 |
| SWA, confianza y trayectoria | 0,949315439 | 0 | 2 |
| Fold0, confianza | 0,948350772 | 0 | 4 |
| Fold0, confianza y trayectoria | 0,949703604 | 0 | 1 |

Las cuatro mezclas selectivas tampoco consiguen retener los aciertos nuevos.
Se descartan para producción. Nueve candidatos completos en E059–E060
(dos modelos y siete ensambles), sin nueva submission ni ejecuciones pendientes.
Seis pruebas unitarias existentes de mezcla y componentes pasan; carga estricta
e inferencia de ambos checkpoints verificadas; controles locales exactos.

Esta familia de expertos no demuestra una mejora desplegable. Los 67 enlaces
con extremos ausentes tampoco pueden arreglarse cambiando solo asociaciones.
La siguiente línea razonable es un detector complementario con evaluación
temporal de sus nuevas células, no otro barrido de pesos de estos checkpoints.
Eso queda como hipótesis de investigación, no como resultado obtenido.
