# E038–E040: tres líneas nuevas, ejecutadas en orden

Solicitud: probar SpatialDINO, un detector temporal y un modelo de divisiones
con supervisión ampliada. Ninguna etapa envía automáticamente al leaderboard.
Los estados efectivos y resultados están en `results/E0XX_*`.

## Cohortes y auditoría

La auditoría CPU de los 199 videos encontró 151 divisiones consecutivas
anotadas. De ellas, 110 pertenecen a 135 videos fuera del grupo de 64 usado por
E030–E037. Esto no demuestra que los modelos públicos nunca vieran esos videos.

Se fijaron 16 videos de desarrollo, ocho por embrión, antes de obtener resultados.
Se enriquecieron por presencia de divisiones: 15 eventos en total. La selección
es determinista mediante SHA256, sin consultar errores de ningún modelo.
Estos son videos de desarrollo, no una evaluación final independiente.

El inventario de 151 divisiones ya existía en la etapa E004, que trabajó con los
199 videos. Esta ampliación recupera supervisión excluida de E035–E037; no
descubre anotaciones nuevas ni videos nunca vistos en todo el historial.
E040 cambia la entrada a una consulta espacial conjunta de madre e hijas sobre
volúmenes de 32³; los experimentos antiguos empleaban otros recortes y modelos.

E038 y E039 entrenan sus componentes nuevos con los mismos 40 videos anteriores.
E040 amplía a 159 videos y 124 divisiones anotadas. Excluye del entrenamiento los
16 videos nuevos de desarrollo, los 16 de calibración original y los ocho de
desarrollo anteriores. Las 15 divisiones de desarrollo nunca se usan para el
gradiente. No se añaden etiquetas manuales inventadas ni se tratan células sin
anotar como fondo.

## E038: SpatialDINO

Fuente: https://github.com/kirchhausenlab/spatialdino, commit
`ca3ab86b34430d963f12a3909baaeb9343c63b7d`.
Checkpoint público `models/spatial_dino/step=249999/backbone.pth`, 86,068,439 bytes,
SHA256 `ce955465ea00333f3b17af6d5c6bd93627577069d19dffe314c7c48bbf649bdf`.

Adaptación de inferencia ViT-S/8 3D mediante atención SDPA de PyTorch, con el
registro cero de inferencia del repositorio. Se verificaron todas las claves
del checkpoint. La comparación CPU con los bloques publicados dio diferencia
absoluta máxima de 1.43e-6. No se incluyen pesos externos en Git.

Prueba de localización: hasta 128 centros anotados por video, con desplazamientos
uniformes fijados de ±4 vóxeles isotrópicos. Recorte 16³ interpolado a 48³,
características volumétricas agrupadas en 2³ regiones. Ridge con alpha100,
comparado con Ridge sobre intensidades 8³ y con no corregir el centro inicial.
Se exige reducir el error medio por video al menos 10% frente a ambos controles
y mejorar frente a intensidades en ambos embriones. La tarea parte de una
propuesta cercana a GT: no mide detección completa ni seguimiento.

## E039: detector temporal mediante campos hacia centros

Resultado E038: **no supera el criterio**. Error medio por video: centro sin
corrección 6,185612 µm, Ridge de intensidades 5,394559 µm,
SpatialDINO 5,435024 µm. Se usaron 4.948 ejemplos de ajuste y 2.034 de desarrollo.
Preparación CPU 450,924 s, inferencia GPU 18,717 s, evaluación CPU 3,898 s.
La memoria máxima asignada por PyTorch fue 0,144 GB; esto no representa toda la
memoria del proceso CUDA. Se descarta esta adaptación para integración, sin
afirmar que todas las aplicaciones de SpatialDINO estén agotadas.

Dos redes convolucionales 3D con la misma estructura, una con un fotograma y
otra con tres. Entrenamiento desde cero, 2,000 pasos por variante y lote 4.
La pérdida SmoothL1 solo supervisa el vector al centro dentro de un radio de
tres vóxeles de centros conocidos; el resto queda enmascarado. No hay pérdida
de clasificación de fondo sobre regiones desconocidas.

Inferencia densa en los fotogramas 10,30,50,70,90 de los 16 videos de desarrollo.
Los vectores emiten votos trilineales ponderados por contraste de imagen.
Se compara con máximos de diferencia de gaussianas a presupuestos iguales de
128, 256, 512 centros. Criterio principal fijado en 256: mejorar recuperación a
3µm frente a ambos controles en ambos embriones, sin perder recuperación a 7 µm.
Los otros presupuestos son diagnósticos. La anotación parcial permite medir
recuperación de centros conocidos; no permite declarar precisión ni puntaje
oficial de tracking.

## E040: división conjunta con más eventos reales

Resultado E039: **no supera el criterio**. En 80 imágenes con 488 centros
anotados, a presupuesto de 256:

| Variante | Centros a 3 µm | Centros a 7 µm |
|---|---:|---:|
| Máximos de imagen | 194 | 319 |
| Red de un fotograma | 177 | 354 |
| Red de tres fotogramas | 175 | 344 |

Con 512 candidatos, temporal obtiene 256/455 frente a 248/446 de la red estática,
pero los máximos de imagen aún localizan 268 a 3 µm. Esa señal secundaria no
revierte el criterio previamente fijado ni justifica un submission. El resultado
no mide diferencias respecto a Harmonic, que no se ejecutó en esta cohorte.
Ambas redes realizaron 1.998 actualizaciones efectivas de 2.000 intentos con AMP.
Pesos propios descargados y hashes verificados. Entrenamiento GPU 318,862 s,
inferencia GPU 28,471 s, evaluación CPU 55,024 s. Arquitecturas de 353.955/354.819 parámetros.

Se conservan todas las divisiones binarias consecutivas anotadas, sin exigir
las tres posiciones de historia/futuro que descartaban eventos antes. Las
imágenes faltantes fuera del intervalo se replican en el borde temporal. Los
ejemplos negativos requieren una hija con otro padre conocido; una madre con
una única hija anotada no establece por sí sola una división negativa.

Entrada: cinco volúmenes de 32³ alrededor de la madre y dos mapas de consulta,
uno para madre y otro para la unión simétrica de hijas. CNN3D entrenada de
extremo a extremo y cabeza con 11 características geométricas. Muestreo de
entrenamiento equilibrado entre ejemplos positivos y negativos, 2.000 pasos,
lote 16. CPU compara con ExtraTrees geométrico sobre exactamente los mismos
ejemplos. Ablación en inferencia repite el fotograma central.

Criterio: al menos 3 eventos verdaderos sin falsos conocidos al umbral de
desarrollo, y AP al menos 0,05 por encima de geometría y de la ablación estática.
La clasificación usa candidatos anotados; incluso superar este criterio no
demuestra mejora sobre el grafo completo y requeriría integración posterior.

## Verificación y recursos

Resultado E040: **no supera el criterio**. AP conjunta temporal 0,257212,
repetición del fotograma central 0,177604 y geometría 0,657862. Solo geometría
encuentra un umbral con al menos tres verdaderos positivos y ningún falso
conocido. En 44b6, ambas geometría y temporal alcanzan AP1,0 sobre seis eventos;
en 6bba, temporal baja a0,137980 frente a0,583116 de geometría sobre nueve.
La mejora frente a la repetición de imagen no establece valor añadido frente
a geometría ni mejora de seguimiento.

La sigmoid GPU generó empates de precisión media. Se auditó su efecto: incluso
ordenando todos los positivos primero dentro de cada empate, la AP temporal
solo alcanzaría0,371804, aún inferior a geometría. El máximo temporal es un
único falso evento conocido. Ese límite se refiere a desempatar los scores
guardados, no a demostrar equivalencia con una nueva inferencia FP32. No se
reentrenó ni se cambió el umbral para favorecer el resultado.
Detalle: `results/E040_score_precision_audit.json`.

E040 completó 1.996 actualizaciones efectivas. Pesos propios descargados y
verificados; entrenamiento y puntuación GPU87,479 s, evaluación CPU4,014 s.
**Las tres adaptaciones se cierran sin submission ni nuevos scores oficiales
de tracking.** No quedan trabajos de estas etapas pendientes. Los resultados
no prueban que los métodos completos estén agotados ni un techo del leaderboard.

Preparación E040 completada en 896,778 s de CPU. Recuentos comprobados:
124 positivos y 14.413 negativos de ajuste, 15 positivos y 6.525 negativos
de desarrollo. De las cohortes159/16, aportan ejemplos140/15 videos;
los demás no tienen pares etiquetables y se omiten del entrenamiento/evaluación.
Se guardaron13.139 y3.807 recortes únicos de madres, respectivamente.
El manifiesto queda fijado por `baseline/e040_input_pin.json` antes del GPU.

Para repetir las cinco comprobaciones locales, con `src` en `PYTHONPATH`:
`python -m unittest discover -s tests -p test_three_lines.py -v`.
La equivalencia de SpatialDINO se reproduce con
`scripts/fetch_spatial_reference.py` y `scripts/verify_spatial_adapter.py`.

Cinco comprobaciones locales verifican zonas sin supervisión, vectores en
regiones solapadas, votación al centro, gradientes y dimensiones de las redes,
preservación de divisiones sin historia completa, simetría de hijas y reflexión
espacial consistente con los mapas de consulta.

Kaggle CPU prepara imágenes, consulta anotaciones y calcula métricas. GPU se
usa para inferencia de SpatialDINO y entrenamiento/inferencia de redes nuevas.
Sin Colab. Los tiempos de proceso se reportan separados de arranque/exportación
de Kaggle. La ejecución es secuencial E038 → E039 → E040.
