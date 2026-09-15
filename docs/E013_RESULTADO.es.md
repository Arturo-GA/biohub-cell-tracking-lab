# E013: resultado y decisión, 15 de septiembre de 2026

El notebook original terminó los 6.000 pasos y evaluó los 48 videos fijados. El candidato se rechaza para submission: score **0,314681**, frente a **0,909257** del control en ese mismo conjunto. Son resultados de desarrollo condicional; no equivalen al leaderboard público 0,946/0,947.

| Métrica | Candidato | Control |
|---|---:|---:|
| Jaccard de enlaces ajustado | 0,314681 | 0,903802 |
| Divisiones correctas | 0 | 3 |
| Divisiones falsas | 270 | 22 |
| Divisiones omitidas | 33 | 30 |
| Recuperación de nodos | 0,988769 | 0,982333 |

Se recuperaron 523 archivos pequeños, incluidos ambos checkpoints, informes y código. Se verificaron los 29 archivos del paquete contra el ZIP original y su hash de lanzamiento, la partición, el checkpoint elegido y los manifiestos de los 112 juegos de características. Se recalcularon ambos agregados desde sus métricas por video. El CSV y las matrices grandes siguen en Kaggle; no se afirma que se haya repetido localmente la métrica oficial ni que las 112 matrices estén descargadas. Los archivos detallados están en `outputs/e013_candidate_recovery`, fuera de Git. Auditoría reproducible: `python scripts/audit_e013_completed.py`.

El mejor checkpoint fue el paso 500: pérdida de calibración 0,64055. Al paso 6.000 subió a 1,14045. En los ejemplos conocidos de calibración, el umbral de enlaces acepta todos los casos, incluidos 5.576 negativos; el F1 de 0,95439 refleja en gran parte una prevalencia positiva del 91,28 %. AUROC de enlaces: 0,64919. AUROC de divisiones: 0,30923; el umbral acepta 2 positivos y 631 negativos, y pierde 53 positivos. Estos son ejemplos de correspondencias entre detecciones, no divisiones biológicas independientes.

El optimizador usó fallback en 288 de 1.200 ventanas (24 %). Esto y la discriminación deficiente justifican estudiar por separado aprendizaje y asociación; todavía no permiten atribuir toda la caída a una causa única. Cambiar solo el número de pasos o el umbral de divisiones no justifica otro entrenamiento completo.

## Siguiente experimento sustancial

1. Reutilizar las características originales, descargando los bloques necesarios con sus hashes y reserva de disco. Mantener juntos cada grafo, características y etiquetas originales: las detecciones locales no tienen necesariamente los mismos índices.
2. Auditar en los 48 videos de ajuste y 16 de calibración las alternativas que compiten por una misma célula, la multiplicidad de detecciones y la correspondencia con etiquetas. Mantener enmascarados los casos desconocidos.
3. Diseñar un objetivo de selección entre enlaces competidores con restricciones de asociación, en lugar de depender solo de clasificación binaria aislada. Medir las trayectorias completas de calibración al seleccionar el modelo. La arquitectura y configuración se fijarán después del diagnóstico, antes de la nueva evaluación.
4. Separar el efecto del modelo del optimizador con una referencia de asociación determinista y revisar la viabilidad de las ventanas. Las 48 secuencias de evaluación ya informaron este diseño; no presentarlas como validación independiente.
5. Entrenar el nuevo selector en la RTX 3050 de la laptop; utilizar Kaggle CPU para asociación y evaluación. Conservar el control actual. No repetir E013 sin cambios ni enviar este candidato al leaderboard.

CPU Image v4 también terminó: cuatro entradas más, offset 6–9, siguiente offset 10. Se recuperaron sus informes; se pospone descargar los ~2 GB de imágenes porque E013 ya produjo esas características. El lote local anterior terminó y pasó su auditoría: **6 videos locales completos**. No hay un nuevo entrenamiento, lote remoto ni monitor ejecutándose por esta revisión.

Recibos: [comparación](../results/E013_completed_comparison.json), [calibración](../results/E013_calibration_diagnostic.json), [CPU v4](../results/CPU_IMAGE_PREPARE_v4_observed.json), [lote local](../results/LOCAL_REAL_BATCH_03.json).
