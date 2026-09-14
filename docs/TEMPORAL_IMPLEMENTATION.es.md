# E004 — Modelo temporal propio con evaluación por embrión

Implementación y entrenamiento completados el 14 de septiembre de 2026 UTC. El control de leaderboard permanece en **0.946** y la referencia aportada por Arturo es **0.947**. **Esta versión no mejora el control y no se envía al leaderboard.** Recibo completo: `results/E004_completed.json`.

## Resultado

Los dos modelos, de **197.506 parámetros** cada uno, completaron **3.000 pasos de optimización** desde cero en GPU. Se eligió el checkpoint del paso 1.000 en ambos casos usando únicamente el conjunto de desarrollo. La preparación de los 199 videos tardó **813,29 s** y produjo 266.529 recortes, incluyendo distractores. La cobertura de candidatos fue de **128.866/128.883 enlaces anotados**. Los pesos se descargaron y sus SHA256 se verificaron localmente.

| Embrión reservado | Videos completos | Exactitud de padre: modelo | Vecino más cercano | AP de pares de división |
|---|---:|---:|---:|---:|
| 44b6 | 71 | 99,4855 % | 99,7932 % | 0,1872 |
| 6bba | 128 | 98,8153 % | 99,1821 % | 0,0527 |

Estos porcentajes usan centros anotados y distractores DoG. **No son scores de Kaggle** y no prueban la calidad de un detector. El modelo no supera al control geométrico en ninguna de las dos separaciones. La AP se calcula sobre pares con relación conocida: hay 26 positivos en el primer embrión y 125 en el segundo; no debe interpretarse como precisión de todas las divisiones del video.

En los cuatro videos con detecciones Harmonic fijas, el score oficial pasó de **0.9666951 a 0.9412437**. La diferencia se descompone en **−0.0004514 de aristas ajustadas** y **−0.025 por divisiones**. Divisiones TP/FP/FN: **0/1/7**, frente a **2/1/5** del control. La inferencia y reconstrucción de los cuatro videos tardó **46,07 s**, excluyendo instalación, entrenamiento y lectura inicial de modelos. Todos los problemas del solver terminaron con gap cero y el CSV final pasó su validación.

Los tiempos registrados por trabajador fueron 147,14 s y 113,77 s, con ejecución concurrente; su suma no es el tiempo de pared del notebook. Se conservan los modelos y la implementación para estudiar errores. No se ajustarán umbrales del leaderboard para presentar este resultado como una mejora.

## Datos y separación

La auditoría recorrió los **199 videos**, con **133.318 nodos, 128.883 enlaces y 151 divisiones anotadas**. El embrión `44b6` aporta 71 videos y 26 divisiones; `6bba`, 128 videos y 125 divisiones. Los organizadores confirmaron que hay dos embriones de entrenamiento y que los de test son distintos: [respuesta del host](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/716793). Recibo: `results/E004_data_audit.json`.

Se entrenan dos modelos desde inicialización aleatoria. Cada uno excluye por completo un embrión. Dentro del otro embrión, un 20 % de videos seleccionado por hash sirve para elegir checkpoint; es un conjunto de desarrollo dentro del mismo embrión, no una evaluación independiente. El embrión excluido se evalúa al final y sus etiquetas no eligen checkpoint. La presencia de solo dos embriones limita las conclusiones de generalización.

El detector público secundario declara los 199 videos en entrenamiento. Por ello se distinguen dos evaluaciones:

1. **Modelo temporal independiente por embrión:** imágenes y centros anotados del embrión reservado, junto a distractores extraídos por DoG. Mide elección del padre y clasificación de pares de hijas. El uso de centros anotados facilita el problema y **no produce un score de competencia**.
2. **Comparación condicional con Harmonic:** conserva sus detecciones de cuatro videos completos ya disponibles y sustituye todas las asociaciones por el modelo que excluyó ese embrión. Usa la métrica oficial sobre el CSV final. El detector sí vio esos videos; **no es validación independiente del pipeline completo**.

## Arquitectura y entrenamiento

Cada nodo utiliza un recorte de **cinco frames**, centrado espacialmente en su posición. Las imágenes se reducen por `(1,4,4)` en Z/Y/X, resultando en vóxeles de 1,625 µm por eje. Cada recorte mide `9×17×17` y se normaliza a partir de la propia imagen. Un encoder 3D con convoluciones y GroupNorm produce 64 características por nodo. No se cargan pesos públicos en este encoder.

Para cada célula hija, se consideran hasta 12 padres del frame anterior dentro de 20 µm. Un transformer de dos capas compara sus imágenes y desplazamientos físicos, con una opción de aparición sin padre. Se aprende mediante entropía cruzada sobre enlaces anotados. Los targets sin padre conocido no reciben pérdida. Se elimina artificialmente el padre correcto en el 10 % de ejemplos para enseñar la opción de aparición; esa frecuencia es una elección de entrenamiento, no una tasa de apariciones medida en test.

Una cabeza adicional evalúa **madre y dos hijas juntas**. Su representación es invariante al intercambio de las hijas. Los positivos son divisiones anotadas; un negativo exige que al menos una hija tenga un padre anotado distinto. Tener un único hijo anotado no etiqueta automáticamente a una madre como no divisoria. Se muestrean positivos y negativos por igual; al inferir se corrige el logit por la proporción de clases de los ejemplos de entrenamiento. Esa corrección no garantiza calibración al cambiar de dominio o densidad.

Los distractores se extraen de máximos DoG en la imagen, a más de 5 µm de centros anotados y cerca de ellos. Su función es ofrecer candidatos competidores para una hija cuyo padre sí se conoce. No se etiquetan indiscriminadamente todas las células sin anotación como fondo.

Presupuesto inicial: **3.000 pasos por separación**, AdamW, reducción coseno de learning rate, recortes invertidos espacialmente con desplazamientos consistentes y precisión mixta en GPU. La pérdida combina elección de padre y clasificación de divisiones. Cada 500 pasos se evalúa el conjunto de desarrollo; el criterio suma entropía cruzada de padres y 0,25 veces BCE balanceada de pares de hijas. Se guardan curvas, split, checkpoint y SHA256.

## Reconstrucción del linaje

Se codifica cada detección una sola vez. Las opciones del optimizador son una continuación simple o una división explícita. Cada padre puede elegir una opción y cada hija puede tener como máximo un padre. Una división requiere evidencia positiva de su cabeza de tres células; dos enlaces fuertes por separado no bastan para crearla.

El problema se resuelve por frame mediante programación entera. Se registran estado y gap del solver; una solución con límite de tiempo solo se acepta si su asignación es entera y factible. No se afirma optimalidad cuando queda un gap. Los enlaces son consecutivos y los CSVs pasan la validación del proyecto.

## Ejecución y límites

- `scripts/build_temporal_notebooks.py prepare`: prepara recortes y ejemplos de todo train en CPU, sin pesos preentrenados.
- `scripts/build_temporal_notebooks.py train`: adjunta esa salida; entrena las dos separaciones en GPUs distintas cuando están disponibles y ejecuta después la comparación condicional con Harmonic.
- `src/biohub_lab/temporal_data.py`, `temporal_model.py`, `temporal_train.py` y `temporal_inference.py` contienen datos, red, entrenamiento e inferencia.
- `tests/test_temporal.py` verifica contexto temporal, separación, supervisión escasa, simetría, gradientes, máscaras de padding y optimización contra enumeración exhaustiva.

Los recortes y checkpoints permanecen en Kaggle o en directorios locales ignorados por Git. Las dos separaciones y la métrica oficial condicional ya se revisaron: **se descarta esta versión para submission y no se entrena una versión final sobre ambos embriones**.

## Siguiente hipótesis que merece una implementación distinta

La escasez de divisiones y el cambio de dominio son limitaciones visibles: un modelo entrenó con solo 23 divisiones positivas y el otro con 94 después de separar desarrollo. Esto no demuestra por sí solo la causa de cada error. Antes de ampliar la red o repetir ajustes, la siguiente hipótesis es preentrenar con linajes densos y divisiones variadas, conservando el test por embrión.

Se revisó el [generador sintético público de José Freitas](https://www.kaggle.com/code/josefreitasalvesneto/biohub-synthetic-dataset), anunciado con 165.267 divisiones en el [foro](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/732103). El código extrae templates y fondos de imágenes reales, además de calibrar distribuciones. Por ello no debe asumirse que su salida publicada es independiente de nuestro embrión reservado. Una adaptación para validación estricta tendría que regenerar esos recursos usando únicamente el embrión de entrenamiento de cada separación. **Solo se auditó el código; no se ejecutó ni se usaron sus datos para entrenar E004.**

La idea de aprender asociaciones con contexto temporal se apoya en [Trackastra](https://arxiv.org/abs/2405.15700). Esta arquitectura con recortes de imagen y cabeza de triples es una implementación propia; no se presenta como reproducción de ese paper ni como novedad científica demostrada.
