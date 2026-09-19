# Investigación y experimento E015 — 19 septiembre 2026

E014 queda rechazado: en los mismos 48 videos obtuvo 0.324351 frente a 0.909257 del control Harmonic. Produjo 2.445 veces más nodos y empeoró en los 48 videos, incluidos los 28 sin fallback del solver. Son resultados locales de desarrollo, no puntuaciones de leaderboard. Recibo: `results/E014_CPU_completed.json`.

La revisión del código encontró que `event_data.sparse_labels` permite asignar múltiples detecciones a una misma célula anotada. Así, varias asociaciones alternativas pueden etiquetarse simultáneamente como correctas. La AP de enlaces de 0.9982 no demuestra que se seleccione una trayectoria coherente. Esto es una discrepancia comprobada en la definición de etiquetas; todavía falta medir su contribución al fracaso real. El test sintético reproduce el problema, pero no establece causalidad en los videos.

## Discusiones y métrica

Consulté el contenido público accesible/indexado; algunas páginas de Kaggle no devuelven todos sus comentarios al abrirlas. No revisé contenido privado ni puedo afirmar haber leído todo el foro.

- En [FOCUS-3D, discusión 738217](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/738217), participantes proponen medir cobertura y exceso de detecciones. Se reporta un timeout de inferencia directa y se plantea usar segmentación como maestra para un detector ligero. Son experiencias de participantes, no resultados reproducidos aquí.
- [Very dim nodes?](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737896) muestra incertidumbre sobre anotaciones de células poco visibles. Las explicaciones propuestas son especulativas; no justifican eliminar automáticamente nodos tenues.
- La [documentación oficial actual](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md) exige correspondencia bipartita de nodos dentro de 7 µm y topología local dirigida para divisiones. Compartir un componente conectado no basta. Los nodos sin correspondencia no equivalen automáticamente a falsos positivos, dada la anotación incompleta. Nuestro archivo fijado `division_metrics.py` ya contiene las restricciones locales; E015 mantiene la misma implementación para comparabilidad.

## Papers y aplicación concreta

| Fuente primaria | Idea útil | Aplicación y límite |
| --- | --- | --- |
| [Ultrack, Nature Methods 2025](https://doi.org/10.1038/s41592-025-02778-0) | Seleccionar conjuntamente hipótesis de segmentación compatibles y enlaces temporales. | E015 prueba exclusión física de candidatos y asociación global. Es una ablación propia y limitada; no implementa Ultrack completo ni su jerarquía de segmentaciones. |
| [Trackastra, ECCV 2024](https://arxiv.org/abs/2405.15700) | Asociaciones contextualizadas en ventanas temporales y normalización parental que admite divisiones. | Próxima arquitectura: competencia entre padres sobre identidades depuradas, en lugar de puntuaciones independientes para duplicados. Sus pesos requieren entradas adecuadas, incluidas máscaras; no son un reemplazo directo del modelo actual. |
| [Attrackt, ICCV Workshops 2025; registro publicado en 2026](https://www.janelia.org/publication/an-investigation-of-unsupervised-cell-tracking-and-interactive-fine-tuning) | Aprendizaje sin etiquetas extensas y corrección de decisiones inciertas con pocas anotaciones. | Investigar consistencia temporal y priorizar enlaces ambiguos. No asumir que una detección sin etiqueta es fondo. El artículo evalúa otros datasets; falta demostrar transferencia a Biohub. |
| [FOCUS-3D, preprint 2026 y repositorio de autores](https://github.com/yu-lab-vt/FOCUS-3D) | Segmentación volumétrica generalizable; preprint DOI 10.64898/2026.08.25.746907. | Posible maestro en clips de entrenamiento para destilar un detector de puntos. Pendiente medir cobertura, memoria y tiempo; todavía no descargamos ni entrenamos sus pesos. GPU solo si esta etapa la necesita. |

La [nota del autor del baseline sobre diagnóstico estructural](https://pilkwangkim.github.io/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics/) refuerza preservar un control y medir cambios de grafo. Sus explicaciones antiguas de divisiones deben contrastarse con la documentación oficial actual. No adoptamos conectividad global como sustituto de una división biológica válida.

## Implementación lanzada

[Notebook privado E015](https://www.kaggle.com/code/jarturo/biohub-lab-detection-identity-audit), versión 1, solo CPU. Estado inicial observado: RUNNING. No hay monitoreo automático ni envío al leaderboard.

1. Reutiliza los grafos E013 y las predicciones E014 existentes; verifica hashes y reconstruye exactamente el CSV original.
2. Sobre los 16 videos de calibración, selecciona representantes con separación de 3 µm. Prioriza origen Harmonic, duración del componente y luego ID. Redirige los votos de enlaces y resuelve asignación bipartita global por par de frames, con opción de quedar sin enlace. No introduce divisiones.
3. Congela el CSV antes de leer anotaciones. No usa el número estimado de células para decidir detecciones ni barre radios.
4. Mide multiplicidad y discrepancia entre correspondencia de entrenamiento y asignación única diagnóstica. Esta asignación diagnóstica no se presenta como implementación exacta de tracksdata.
5. Evalúa el candidato con la métrica oficial fijada, contra el E014 de calibración (0.302565). No vuelve a utilizar el conjunto de 48 videos.

Pruebas: cuatro casos unitarios superados, compilación de fuentes y notebook, importación del runner y validación de notebook privado sin aceleradores. La ejecución completa y la evaluación real quedan pendientes en Kaggle.

El radio de 3 µm puede fusionar células verdaderamente distintas; esta ablación mide ese compromiso. Reducir nodos no prueba por sí solo una mejora: revisar también Jaccard de enlaces sin ajuste, recall y resultados por video. Incluso si supera E014, no demuestra superar Harmonic.

## Decisión tras los resultados

- Si la discrepancia de identidad es frecuente y la selección mejora enlaces sin una caída importante de cobertura, preparar supervisión por identidad y asociaciones competitivas sobre los 48 videos de entrenamiento. Mantener desconocidas las detecciones sin anotación.
- Si reduce nodos pero no recupera enlaces, abandonar esta proyección como solución y preservar las trayectorias Harmonic como ancla para un modelo de reparación de enlaces ambiguos con contexto temporal.
- Si el fallo dominante es cobertura/localización, probar FOCUS-3D como maestro en clips de entrenamiento antes de presupuestar inferencia extensa.

Las decisiones usarán calibración; los 48 videos ya consultados son desarrollo, no un test independiente. Además, el detector secundario tuvo acceso de entrenamiento a los 199 videos públicos. Ningún resultado de esta investigación garantiza una mejora en leaderboard ni un puesto de podio.
