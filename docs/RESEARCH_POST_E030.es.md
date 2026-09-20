# Investigación posterior a E030 — 20 de septiembre de 2026

Sí hay líneas justificadas para avanzar. La prioridad es combinar **detecciones más completas con aprendizaje temporal sobre candidatos reales**. No hay evidencia pública suficiente para atribuir una técnica concreta a los líderes ni para prometer plata. Nuestro último resultado público sigue siendo 0,946; esta revisión no produjo un nuevo score.

Se consultaron listados autenticados de Kaggle por ejecución reciente y score, se descargaron siete notebooks/scripts antes ausentes del registro y se inspeccionaron sus componentes relevantes. Se revisaron papers y repositorios primarios. Las discusiones se recuperaron mediante páginas indexadas: pueden faltar comentarios recientes. El navegador integrado falló al inicializarse. Los títulos, posiciones de autores y scores históricos del índice no se consideran resultados actuales verificados. El inventario con hashes está en `results/RESEARCH_20260920_POST_E030.json`. No se ejecutó código externo ni se consumió GPU.

## 1. Detector denso: el recurso nuevo más directamente aprovechable

[Cell Point Detector, hengck23](https://www.kaggle.com/code/hengck23/cell-point-detector) publica una demo de inferencia con UNet3D y pesos. Su [discusión sobre FOCUS-3D](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/738217), ya conocida pero revisitada, describe entrenamiento con centros densos de FOCUS y anotaciones Kaggle. Lo nuevo revisado aquí es el código del detector disponible.

La demo usa `StrongUNet3D3Level`, canales 64/128/256, reduce XY por cuatro y extrae máximos 3D. El dataset contiene dos checkpoints y tres archivos Python; se descargaron únicamente estos últimos. El archivo llamado `loss_and_metric_v12.py` solo define `DotDict`: no constituye una receta de entrenamiento. No encontramos un manifiesto completo de videos usados para ajustar los pesos. **Es un detector disponible, no una solución de tracking validada fuera de entrenamiento.**

Aplicación propuesta: comparar su cobertura y localización con Harmonic en un conjunto fijado antes de ver resultados; identificar qué omisiones recupera y si conserva las células entre fotogramas. Después, entrenar un detector propio con supervisión densa de confianza y etiquetas dispersas, sin tratar todo centro no anotado como fondo. Evitar otra unión indiscriminada de puntos: E029 ya mostró que más nodos pueden aportar casi ningún enlace correcto.

La procedencia del entrenamiento debe resolverse antes de llamar «holdout» a esa comparación. Si no se conoce, el checkpoint sirve como diagnóstico/maestro condicionado, no como evidencia independiente de generalización. Licencia de pesos y datos también pendiente antes de incorporarlos al entrenamiento distribuido.

## 2. Aprender identidad a través del tiempo: cambio de señal de entrenamiento

[Zyss et al., Contrastive learning for cell division detection and tracking in live cell imaging data](https://link.springer.com/article/10.1186/s12859-025-06344-5), publicado el 27 de diciembre de 2025, volumen 2026, aprende representaciones usando correspondencias temporales débiles. También incorpora contexto temporal para reconocer madre y dos hijas. Su evidencia procede de cultivos y benchmarks, no de Biohub 3D. Además, los resultados CTC de tracking no emplearon las representaciones contrastivas específicas por limitaciones de datos: no atribuirles todos los resultados del artículo. El texto enlaza datos anotados; no localizamos allí un repositorio GitHub.

Nuestra adaptación propuesta: entrenar sobre recortes volumétricos de la misma célula en tiempos diferentes; contrastarlos con vecinos realmente ambiguos. Usar trayectorias de confianza solo en entrenamiento, con supervisión anotada cuando exista. Las divisiones requieren ejemplos madre–hijas y ventanas temporales, no únicamente distancias entre centros. Esto cambia el problema que aprende E030: su evaluación de centros anotados perturbados fue tan fácil que vecino más cercano acertó 588/588 enlaces.

**Protocolo:** separar videos antes de generar pseudoetiquetas o ajustar normalizaciones; construir los candidatos con exactamente la información disponible en inferencia; excluir relaciones desconocidas de las etiquetas negativas seguras; reservar eventos de división en desarrollo y evaluación. Comparar geometría, apariencia simple y encoder temporal bajo los mismos candidatos y decodificador. Medir grafo completo, errores por video y eventos, además de clasificación de pares. No promover un modelo solo por AP o por aciertos en centros GT.

## 3. SpatialDINO: alternativa 3D, con viabilidad pendiente

[SpatialDINO, preprint de enero de 2026](https://www.biorxiv.org/content/10.64898/2025.12.31.697247v2), tiene [repositorio de los autores](https://github.com/kirchhausenlab/spatialdino). Aprende representaciones directamente volumétricas; podría evitar perder contexto al convertir un núcleo en planos 2D. Su dominio incluye estructuras subcelulares y no garantiza transferencia a núcleos de pez cebra. El repositorio anuncia modelos en S3 y ejemplos con múltiples GPU; no verificamos aquí memoria ni latencia en T4. El código indica MIT, pero no debe confundirse con la licencia del artículo o de cada peso.

Se reserva como encoder alternativo preentrenado. No recomiendo preentrenarlo desde cero ni dedicarle la cuota semanal antes de comprobar tamaño, licencia y una inferencia volumétrica. Tampoco sustituye la corrección del conjunto de entrenamiento de E030.

## Auditoría de los siete notebooks nuevos

| Notebook | Hallazgo y decisión |
|---|---|
| [hengck23/cell-point-detector](https://www.kaggle.com/code/hengck23/cell-point-detector) | Recurso concreto para detección densa; falta receta completa y procedencia del split. Prioridad alta condicionada. |
| [arnav170/biohub-reid3](https://www.kaggle.com/code/arnav170/biohub-reid3) | Descriptores de intensidad, forma y contexto de 28 dimensiones, clasificador de enlaces sobre detecciones reales. Implementa exclusión por video al ajustar el clasificador. Sin embargo, ajusta normalización usando todos los videos y construye velocidad de entrenamiento mediante el progenitor GT. Hay que reconstruir esa característica desde tracks predichos y ajustar normalización dentro de cada fold. Útil como control CPU, no prueba de mejora actual. |
| [noisyislands/biohub-xgboost-division-events](https://www.kaggle.com/code/noisyislands/biohub-xgboost-division-events) | Trece características geométricas desde GT. Los negativos mezclan una hija con otro nodo del tiempo de la madre, no del tiempo de la hija. Imprime diagnóstico dentro de entrenamiento. No copiar ese muestreo ni interpretarlo como validación. |
| [binasalama/biohub-learned-unet-transformer-ilp-gap-recovery](https://www.kaggle.com/code/binasalama/biohub-learned-unet-transformer-ilp-gap-recovery) | Funciones conocidas de recuperación de huecos y posprocesamiento. No aporta en lo inspeccionado un entrenamiento nuevo que justifique otra línea. |
| [raunakdey07/biohub-harmonic-fusion-v3](https://www.kaggle.com/code/raunakdey07/biohub-harmonic-fusion-v3) | Mantiene las tres familias de pesos públicos de Harmonic. La versión y su posición en el listado no prueban una arquitectura nueva ni un score vigente. |
| [howonkang/biohub-0947-short5-prepp-r1](https://www.kaggle.com/code/howonkang/biohub-0947-short5-prepp-r1) | Variante de rescate y longitud de tracks. «short5» no es evidencia de un Transformer nuevo entrenado con cinco fotogramas. |
| [newwang12/biohub-v1-grouped](https://www.kaggle.com/code/newwang12/biohub-v1-grouped) | Calcula grupos de videos, pero los overrides de los tres grupos están vacíos y el código exige que sigan vacíos. Esta versión no aplica políticas diferentes por grupo. |

Las observaciones corresponden a los archivos descargados en esta fecha, no a futuras versiones. La revisión fue estática y dirigida a entrenamiento, candidatos, inferencia y validación; no una ejecución íntegra de cada notebook.

## Qué aportan las discusiones y otros proyectos

- [Problems with edje connection](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/739685): un participante describe mejoras CV de 0,01–0,03 sin traslado equivalente al leaderboard. Es un testimonio compatible con nuestra experiencia, no una demostración de que los enlaces llegaron a su techo.
- [Does anyone have a different design for divisions](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737438): el autor describe fallos simultáneos en varios filtros en regiones densas. Refuerza estudiar decisiones conjuntas y ejemplos difíciles. Nuestro decodificador conjunto de E030 ya existe; falta demostrar que sus puntuaciones aprendidas sirven en esos candidatos.
- [Auto data generator](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/739731): Sergio Alvarez relata que probó volúmenes sintéticos sin mejora clara y abandonó esa vía. El índice lo mostraba primero, pero no se verificó su puesto actual. No demuestra qué técnica explica su puntuación. La conversación enlaza datos reales de Zebrahub; sus tracks automáticos no deben confundirse con anotación manual ni asumirse independientes de Kaggle.
- [Public Notebook Rankings Need a Metric Refresh](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/736937): reporta scores antiguos inflados aún visibles tras corregirse la métrica. Ayuda a explicar el desacuerdo entre ordenar por score y abrir un notebook; sigue siendo un reporte de participante, no una comprobación de cada notebook listado.
- [Proyecto matt-ceran](https://github.com/matt-ceran/biohub-cell-tracking): documenta aprendizaje con positivos y muestras sin etiqueta, y una auditoría de divisiones que separa fallos de propuestas de fallos de ranking. Su README también reconoce resultados negativos. Es útil metodológicamente; no lo identificamos como solución de un líder ni como superior a Harmonic.

[ARGUS, julio de 2026](https://arxiv.org/html/2607.08297v1) combina flujo, asignación y reparación de trayectorias en 2D. Es reciente, pero transferirlo requiere adaptación 3D y se solapa con nuestras líneas E017/E019; prioridad baja. El trabajo MICCAI 2025 [Bayesian Transformers and Higher-Order Graph Matching](https://github.com/NabaviLab/bayesian-transformer-cell-tracking) ofrece otra pista sobre incertidumbre, pero solo se revisaron resumen/repositorio y no se considera implementación auditada.

FOCUS-3D, HOCT, NucVerse, Ultrack, CELLECT y Trackastra ya figuraban en la investigación. No se presentan como nuevos descubrimientos; tampoco se considera agotado un paper completo porque una adaptación parcial nuestra haya fallado.

## Decisión propuesta

La siguiente línea sustancial debe ser **detector denso + representación temporal entrenada con candidatos de inferencia**. Primero resolver procedencia y medir cobertura del detector disponible; después construir un conjunto temporal que incluya competencia entre vecinos y divisiones reales. La implementación de ReID sirve como comparación CPU corregida. SpatialDINO queda como alternativa si ese problema exige un encoder más fuerte.

Preparación, asociación y evaluación en laptop/Kaggle CPU. GPU únicamente para inferencia neuronal y entrenamiento del encoder/detector. No Colab. Esta revisión deja una dirección concreta, pero todavía no demuestra mejora de leaderboard ni activa un experimento nuevo.
