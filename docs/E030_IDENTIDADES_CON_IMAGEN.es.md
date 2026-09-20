# E030: identidad y linaje con apariencia

Implementación de la propuesta posterior a E029. La unión por cercanía solo añadió un enlace correcto; ahora se aprende la identidad visual del núcleo, la selección del progenitor y la compatibilidad de una división completa.

## Implementación

- `physical_views`: secciones XY, XZ e YZ de 64×64, muestreadas a 0,40625 µm/píxel. Se respeta la separación axial de 1,625 µm y se normaliza cada volumen por percentiles 2/99,8.
- `extract_tokens`: captura la entrada de la capa `out` de Cellpose-DINO ViT-B, después de la normalización del encoder y de retirar CLS/storage. Concatena la media de los cuatro tokens centrales y la media del entorno en cada vista: 4608 características. Reutiliza exactamente los pesos y dependencias fijados en E028; comprueba repetibilidad. **No usa `styles`**, que es aleatorio en ese modelo.
- `ImageIdentityModel`: proyección compartida a 96 dimensiones, clasificación de identidad entre dos detecciones, competencia entre progenitores con opción ausente, y cabeza de división simétrica respecto de las dos hijas. Se usan desplazamientos físicos junto con la apariencia.
- `decode_events`: optimización conjunta entera. Cada madre puede continuar o dividirse en dos; cada hija tiene como máximo un progenitor. Identidades duplicadas de un mismo fotograma compiten por activarse. Se comprueba la factibilidad del resultado.
- `infer_lineage`: acepta detecciones de distintas fuentes y sus descriptores sin etiquetas. Exige proporcionar los puntos de operación de división e identidad; no los inventa a partir de este diagnóstico.

Referencias: [DINOv3](https://arxiv.org/abs/2508.10104) y [implementación oficial Cellpose](https://github.com/MouseLand/cellpose/blob/a54cb48849b7e225a81e8e43dcb042d42427f543/cellpose/vit.py). Esta es una adaptación experimental, no un resultado publicado de esos autores.

## Entrenamiento y cómputo

1. Kaggle **CPU** prepara imágenes reales de los 48 videos fit y los 16 videos de calibración separados. Escoge ocho transiciones repartidas por video; únicamente en fit añade las transiciones de divisiones anotadas para no desperdiciar su escasa supervisión. No selecciona errores del control.
2. Cada centro tiene dos perturbaciones independientes de hasta ±1 µm por eje para aprender consistencia frente a errores de localización. Se generan progenitores cercanos y pares de hijas. Una arista es negativa solo si contradice un progenitor anotado. Una división sin evidencia suficiente queda desconocida; no se convierte en negativa por ausencia de etiqueta.
3. Kaggle **GPU** extrae una sola vez los descriptores del encoder congelado. Límite de 2400 segundos comprobado entre lotes; no hay optimizador ni decodificación de máscaras en GPU.
4. Antes de entrenar se detectó que las transiciones uniformes de calibración no contienen divisiones. Se fijó `e030_head_protocol.json`: **40 videos para entrenamiento y ocho del conjunto fit original para desarrollo**, con seis divisiones; quedan 29 divisiones para aprendizaje. La separación se eligió por un orden SHA determinista y presencia de divisiones, sin consultar predicciones. Kaggle **CPU** entrena 600 pasos fijos de las cabezas pequeñas. Entrena también una ablación que recibe solo geometría, con el mismo protocolo. Elimina aleatoriamente algunos progenitores verdaderos solo durante fit para aprender la opción ausente. Congela ambos checkpoints antes de evaluar desarrollo y calibración.
5. Se comparan exactitud de progenitores, disponibilidad del progenitor verdadero, AP de divisiones e identidad, y un control que permuta los descriptores visuales. No hay selección de checkpoints por leaderboard.

## Qué permite concluir

Esta primera ejecución es un **currículo centrado en anotaciones con perturbación**, no una validación del detector. Los negativos son núcleos anotados próximos, no el conjunto completo de candidatos de Harmonic/NucVerse; la cabeza de identidad aún debe enfrentar duplicados y confusores reales. Un buen resultado aquí no prueba que el CSV mejore. Los 16 videos son calibración utilizada en investigación previa; tampoco constituyen un nuevo test ciego.

Antes de promover un submission deben fijarse puntos de operación usando detecciones reales de fit/desarrollo, comprobarse el grafo entero contra Harmonic con la métrica oficial, y confirmarse el resultado en videos reservados. La API de inferencia está implementada; esa validación completa no queda sustituida por la exactitud entre centros anotados.

## Verificación y estado

Siete pruebas locales pasaron (incluyen aprendizaje de identidad con distancias idénticas y rechazo de saltos temporales): geometría física, extracción repetible independiente de `styles`, simetría de identidad/división, competencia conjunta de duplicados y divisiones, y sensibilidad/gradientes de la apariencia. Paquetes privados con código y configuración; imágenes, etiquetas y pesos quedan fuera de Git.

Las tres etapas terminaron y fueron auditadas. Preparación CPU: **278,90 s**, 13 364 centros perturbados. Extracción GPU: **202,43 s** de proceso, con error máximo de repetición cero. Entrenamiento y evaluación CPU: **59,84 s**, 600 pasos por modelo, 518 500 parámetros por cabeza conjunta. El tiempo de proceso no equivale necesariamente al consumo facturado de cuota Kaggle.

Se verificaron los 4/6/5 archivos fuente de las tres etapas, los manifiestos encadenados, la separación 40/8/16 y los dos checkpoints descargados, sus hashes y su carga estricta. Pesos guardados en `outputs/e030_train/image_identity_train/` y en la salida privada de Kaggle; fuera de Git. Recibo consolidado: `results/E030_completed.json`.

## Resultado y decisión

| Diagnóstico | Imagen DINO | Sin imagen | Imagen permutada | Vecino más cercano |
|---|---:|---:|---:|---:|
| Enlaces correctos, 16 videos de calibración | 586/588 | 585/588 | 585/588 | 588/588 |
| Enlaces correctos, ocho videos de desarrollo | 367/368 | 367/368 | 366/368 | 367/368 |
| AP de división, seis eventos de desarrollo | 0,2193 | 0,2082 | 0,1676 | No medida |
| AP de identidad, calibración | 1,0000 | 1,0000 | 1,0000 | No medida |

Los 16 videos de calibración no contienen divisiones en las transiciones seleccionadas; allí la AP de división es indefinida, no cero. Todos los progenitores verdaderos están representados, así que este diagnóstico tampoco verifica detecciones omitidas ni la opción de progenitor ausente en evaluación.

**No se promueve ni se envía un submission.** La rama visual cambia algunas predicciones, pero no supera al vecino más cercano en enlaces; la diferencia de AP sobre seis divisiones es insuficiente para afirmar una mejora general. La identidad perfecta incluso sin imagen demuestra que los duplicados perturbados de este currículo son demasiado fáciles. No se afirma una mejora de la métrica oficial, de Harmonic ni del leaderboard.

La implementación y el primer entrenamiento quedan completados. No se justifica repetir más épocas sobre este currículo. Para continuar esta arquitectura hace falta supervisar candidatos reales de Harmonic/NucVerse: duplicados de distintas fuentes, vecinos confundibles, progenitores ausentes y divisiones completas. La calibración de esos candidatos y la evaluación del grafo entero siguen pendientes; no se ejecutaron en E030. No quedan trabajos activos.
