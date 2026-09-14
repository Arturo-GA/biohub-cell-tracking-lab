# Investigación para Biohub — 13 de septiembre de 2026

**Actualización al cierre de E004, 14 de septiembre:** el control tiene un score público confirmado de **0.946**. HOCT y el nuevo modelo temporal entrenado desde cero no superaron sus controles. E004 sí estableció una evaluación del modelo temporal dejando un embrión completo fuera del entrenamiento; su comparación con el detector público se declara condicional. Resultados en `TEMPORAL_IMPLEMENTATION.es.md`. El siguiente cambio de datos propuesto es supervisión densa de divisiones, preservando esa separación; no se ha ejecutado aún.

La oportunidad más interesante que encontré es **combinar un detector 3D nuevo y destilado con un linker que modele mejor las divisiones**. FOCUS-3D y HOCT son las dos líneas principales. Division Guard ya terminó sin recuperar ninguna arista anotada ni división adicional y se cerró. HOCT ya tiene implementación de inferencia propia y ejecución en Kaggle; ver `HOCT_IMPLEMENTATION.es.md`.

## Situación al inicio de la investigación, antes de las ejecuciones

La API del leaderboard devolvió 0.970, 0.968 y 0.966 en los tres primeros puestos durante la consulta de esta sesión. Son valores redondeados y temporales. No había submissions propias de Biohub en la cuenta consultada. La referencia pública con título 0.947 está aproximadamente 0.019 por debajo del tercer puesto en esa foto, pero su score no se ha reproducido aquí. [Leaderboard oficial](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/leaderboard).

La página oficial consultada fija entrada/fusión de equipos para el 22 de septiembre y cierre el 29 de septiembre de 2026 a las 23:59 UTC. Indica notebooks CPU/GPU de hasta 12 horas, sin internet, y permite datos/modelos externos de disponibilidad pública. En Lima, el cierre corresponde al 29 de septiembre a las 18:59. Las reglas y fechas deben volver a verificarse antes de la entrega final. [Competencia](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview).

## Papers y proyectos priorizados

| Prioridad | Trabajo y fecha | Hallazgo útil | Aplicación propuesta y límite |
|---|---|---|---|
| Alta | **FOCUS-3D**, preprint de agosto de 2026 | Segmentación volumétrica con modelos generalista, nuclear y de membrana | Generar targets de centros en train y destilarlos a UNet3D; las máscaras también habilitan HOCT. Los pesos de Hugging Face piden aceptar acceso y compartir contacto: no se descargaron ni se aceptaron condiciones. [Código oficial](https://github.com/yu-lab-vt/FOCUS-3D), [model card](https://huggingface.co/Qinghua-thu/FOCUS-3D), [preprint](https://doi.org/10.64898/2026.08.25.746907). |
| Alta | **Higher-Order Cell Tracking Transformer (HOCT)**, 13 jul 2026, preprint | Representa enlaces como tokens y usa su geometría; aborda problemas de asociación cerca de divisiones | Evaluar linker con detecciones/máscaras fijas y pesos públicos. Su mejora publicada no es una mejora de Kaggle. Los organizadores dicen que no lo habían probado en estos datos. [Paper](https://arxiv.org/abs/2607.11754), [código](https://github.com/royerlab/hoct). |
| Media | **Dense Embeddings from Self-Supervision and Foundation Models Improve Cell Linking Performance**, manuscrito MIDL 2026 | Explora features de SAM, µSAM y MAESTER para Trackastra | Cachear features por célula y entrenar una cabeza ligera de asociaciones, buscando errores distintos al baseline. Las comparaciones recuperadas son en datasets de HeLa, no una demostración en zebrafish 3D. El PDF completo devolvió verificación de navegador; no se confirmó su estado editorial final. [Manuscrito en OpenReview](https://openreview.net/attachment?id=4OLrHi9ROr&name=pdf). |
| Media | **Revisiting foundation models for cell instance segmentation**, MIDL 2026 | Compara modelos generales y de microscopía y propone generación automática de prompts | Usar prompts de nuestro detector y revisar solo regiones ambiguas. No asumir que SAM más reciente mejora automáticamente Biohub. [PMLR](https://proceedings.mlr.press/v315/archit26a.html), [preprint de marzo](https://arxiv.org/abs/2603.17845). |
| Media | **NucVerse3D**, Scientific Reports 2026 | Modelo 3D de máscara nuclear y campo de gradientes, con código y pesos | Detector alternativo para medir cobertura complementaria al UNet público; posible fuente de máscaras de HOCT. Adaptación de dominio y coste pendientes. [Código de autores](https://github.com/Segovia-lab/NucVerse3D), [paper](https://doi.org/10.1038/s41598-026-51994-x). |
| Media | **Ultrack**, Nature Methods 2025 | Múltiples hipótesis de segmentación y optimización global, diseñado también para zebrafish 3D | Evitar decidir demasiado pronto una única detección en zonas densas; probar hipótesis alternativas solo donde haya incertidumbre. Integración más costosa que un parche de postproceso. [Repositorio oficial](https://github.com/royerlab/ultrack), [documentación](https://royerlab.github.io/ultrack/), [paper](https://doi.org/10.1038/s41592-025-02778-0). |
| Media-baja | **Segment Anything for Cell Tracking**, 12 sep 2025, preprint | Adapta SAM2 a tracking 2D y 3D sin fine-tuning específico | Profesor de consistencia temporal o regiones difíciles; medir memoria/tiempo antes de usarlo en test completo. [Paper](https://arxiv.org/abs/2509.09943). |
| Referencia | **Trackastra**, ECCV 2024 | Aprende asociaciones entre células dentro de una ventana temporal | Control útil para un linker aprendido; su modelo general 2D no implica transferencia automática a 3D. El repo ofrece una variante con features SAM2. [Paper](https://arxiv.org/abs/2405.15700), [código](https://github.com/weigertlab/trackastra). |
| Idea secundaria | **CellNet**, 10 jun 2026, preprint | Detección y conteo desde anotaciones puntuales escasas y ruidosas | Inspiración para pérdidas que no traten zonas sin anotación como fondo. Trabaja con contraste de fase; no es un detector 3D listo para esta competencia. [Paper](https://arxiv.org/abs/2606.12286). |

## Ideas del foro que sí cambian la estrategia

En una [discusión reciente de Biohub](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/738217), hengck23 propone evaluar FOCUS-3D por cobertura de anotaciones y número total de nodos, entrenar el linker con sus centroides y destilar después el detector. Otro participante reporta timeout con inferencia directa. Eso favorece la estrategia de profesor offline + estudiante rápido. Son observaciones de participantes, no mediciones propias ni recomendaciones del host. No adoptamos la sugerencia de entrenar en datos ocultos sin estudiar primero reglas y validación.

En el [anuncio de HOCT](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/726521), Jordão Bragantini publica el modelo, advierte que depende de las máscaras y recomienda comparar CPU/GPU. El entrenamiento no estaba disponible en ese anuncio. Por eso el primer experimento razonable es de inferencia y asociaciones, no reentrenar el paper desde cero.

El [reporte del hack](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/714101) explica la discrepancia entre CSV y evaluación local del notebook entregado. La auditoría del repositorio oficial confirma que hay reglas nuevas; los rankings históricos de esos notebooks no son una base estable para decidir.

## Repositorios de otros participantes

La revisión técnica adicional de [NucVerse3D](https://github.com/Segovia-lab/NucVerse3D) confirmó pesos públicos en [Zenodo](https://doi.org/10.5281/zenodo.18517324), con modelos específicos y generalistas. Su implementación usa TensorFlow 2.16.2; el README describe pesos de unos 486 MB por modelo y predicción de máscara más campo de gradientes. Es una alternativa de segmentación sin el trámite de acceso de FOCUS-3D, pero integrar un segundo entorno CUDA y medir coste/dominio requiere trabajo adicional. No se ha ejecutado en este proyecto.

El [proyecto de matt-ceran](https://github.com/matt-ceran/biohub-cell-tracking) aporta una comparación distinta: propuestas DoG, filtrado CNN, min-cost-flow de toda la película y validación bloqueada. Lo útil es contrastar cuánto aportan detector y linker por separado. Sus resultados locales no equivalen a podio ni se reutilizaron como evidencia propia.

El [registro de JunhaoLiXD](https://github.com/JunhaoLiXD/Biohub_Cell_Tracking/blob/main/docs/experiments.md) documenta que añadir divisiones mediante reglas puede aumentar falsos positivos y empeorar LB. Es una razón para probar nuestra modificación con control y no interpretar más bifurcaciones como mejor tracking.

## Ruta hacia una mejora sustancial

Mi inferencia a partir de estas fuentes: pequeños ajustes sobre una base compartida pueden mejorar algo, pero una diferencia cercana a dos centésimas frente al podio probablemente exige una señal complementaria. No hay evidencia suficiente para prometer qué método cerrará esa distancia.

La combinación candidata es: **FOCUS-3D como profesor → detector UNet3D destilado → HOCT o cabeza de aristas con varias ventanas temporales → ILP → CSV evaluado con métrica actual**. Se debe medir cada cambio aisladamente. Si el detector actual ya encuentra casi todos los centroides, invertir antes en asociaciones; si falla en núcleos juntos o débiles, priorizar el maestro 3D.

Antes de ensemble, medir coincidencia de errores en nodos y aristas emparejados. Promediar modelos con fallos casi idénticos suele ofrecer menos que corregir el mecanismo que los produce. La comparación requiere videos independientes, procedencia de los pesos, escalas físicas correctas y tiempo medido en Kaggle. El plan ejecutable por etapas está en `EXPERIMENTS.es.md`.

## Ampliación del 14 de septiembre durante E005

La prueba E005 ya se lanzó con los pesos temporales existentes; estas fuentes no cambian esa submission. Tras localizar fallos de divisiones en E004, se añadieron tres referencias primarias:

| Trabajo | Señal nueva que aporta | Aplicación y límite para Biohub |
|---|---|---|
| **CELLECT**, Nature Methods, 20 oct 2025 | UNet3D de dos frames con mapa de centros, embeddings densos de 64 canales y probabilidad de división; aprendizaje contrastivo. | Candidato a profesor de apariencia/división y fuente de centros complementarios. El paper muestra transferencia entre modalidades y especies, no una mejora medida en Biohub. [Paper](https://www.nature.com/articles/s41592-025-02886-x), [código de autores y pesos referenciados](https://github.com/zzz333za/CELLECT). |
| **OrganoidTracker 2.0**, Nature Methods, 8 oct 2025 | Redes que predicen enlaces y divisiones más estimación de incertidumbre mediante alternativas de tracking. | Inspiración para comparar soluciones completas alrededor de una división. Los autores ofrecen modelos de C. elegans y organoides; su documentación actual indica migración de TensorFlow a PyTorch y conversión de pesos. El código mezcla MIT para redes y GPL para otros módulos; revisar archivos concretos antes de integrar. [Paper](https://www.nature.com/articles/s41592-025-02845-6), [repositorio](https://github.com/jvzonlab/OrganoidTracker), [documentación actual](https://jvzonlab.github.io/OrganoidTracker/INSTALLATION.html). |
| **Tui**, 24 abr 2026 | Optimización con correcciones de linaje que considera divisiones y fusiones. | Referencia para restricciones que miren varias generaciones. Las fusiones no se trasladan a nuestro CSV, que admite un padre por hija. Sus benchmarks publicados no son Biohub ni demuestran transferencia a nuestros volúmenes. [Artículo](https://pmc.ncbi.nlm.nih.gov/articles/PMC13106940/), [código de autores](https://github.com/hftsai/tui). |

Mi inferencia: la siguiente inversión sustancial debería aportar supervisión y apariencia de división adicionales, antes de ampliar el mismo clasificador entrenado con pocas divisiones. CELLECT y linajes sintéticos generados exclusivamente desde el embrión de entrenamiento son dos caminos a contrastar. Ninguno se ejecutó en esta ampliación; E005 es exclusivamente la medición de los pesos ya disponibles.

## Alcance de esta investigación

Se revisaron fuentes primarias, repositorios y discusiones hasta la fecha indicada. No se verificó exhaustivamente cada notebook público ni se reprodujeron los scores publicados por terceros. El acceso al texto de algunos papers falló; se identifica arriba cuando afectó el análisis. HOCT ya se evaluó con máscaras aproximadas y nodos fijos: 0.9195083 frente a 0.9666951 del control, un resultado negativo. FOCUS-3D y los demás detectores no se han entrenado/evaluado en nuestra cuenta.
