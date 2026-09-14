# Experimentos

## E000 — Control fijo y evaluación del CSV

Base: primeras siete celdas de Harmonic Fusion congeladas. Se retira el sweep posterior basado en su métrica proxy. Conserva detector, pesos, TTA, ILP y postproceso de la fase de inferencia. Este control no pretende reproducir un score histórico del notebook completo, que podía seleccionar otra configuración.

Medir: tiempo, nodos, enlaces, divisiones, hash del CSV, adjusted edge Jaccard y división oficial. Validar que todos los datasets estén presentes y el grafo use coordenadas reales. No confundir ejecución sobre cuatro muestras visibles con evaluación del test oculto.

**Resultado de leaderboard confirmado:** submission `56214656`, estado `COMPLETE`, score público **0.946** mediante la API autenticada de Kaggle. Queda **0.001 por debajo** del 0.947 de referencia aportado por Arturo, a la precisión de los valores disponibles. Es un resultado del control Harmonic congelado; HOCT no intervino en esta submission. La diferencia de configuración con el notebook completo está identificada, pero no se ha demostrado que explique por sí sola toda la diferencia de score. El notebook anterior `improved-metric-hack-last-call`, submission `56213231`, también terminó: **0.885**. Ambos recibos están en `results/kaggle_submissions.json`.

### Referencia de leaderboard aportada por Arturo — 2026-09-13

Arturo reporta **0.947 en el leaderboard** para el archivo `biohub-harmonic-fusion.ipynb` adjunto. Se conserva como referencia externa aportada por el usuario; no se dispone del identificador de esa submission para verificarla de forma independiente. Registro: `results/harmonic_user_reference.json`.

La comparación confirmó que las fuentes de las 12 celdas coinciden exactamente con el notebook público congelado; el archivo adjunto incluye salidas de ejecución. Esas salidas muestran que el sweep final seleccionó `tight55` (`MOTION_RELINK_TIGHT_UM=5.5`) y reescribió el CSV de 241,189 a 241,306 filas. Nuestro E000 ejecuta las primeras siete celdas y conserva `MOTION_RELINK_TIGHT_UM=6.0`, con 241,189 filas. Por tanto, **nuestro control no reproduce la configuración final del archivo asociado al 0.947**; la coincidencia de filas con la fase anterior tampoco prueba identidad del CSV.

El objetivo de comparación es superar ese 0.947 reportado con evidencia de leaderboard. El 0.966695 local sobre train no demuestra que ya se haya superado. Se mantiene la prioridad del usuario por nuevas implementaciones sustanciales; este dato no inicia otro barrido de pequeños ajustes.

## E001 — Division Guard (cerrado: sin mejora demostrada)

**Hipótesis:** una fusión simétrica de asociaciones hacia delante y hacia atrás puede debilitar asociaciones correctas alrededor de una división, donde las entradas invertidas difieren del patrón de entrenamiento. El predictor inverso no tiene por qué aportar la misma evidencia en ese caso.

Se conserva la fórmula armónica del control. Solo cambia el peso inverso por célula fuente, desde 0.15 hacia cero, si sus dos mejores candidatos a hija cumplen:

1. La probabilidad forward de ambos como hijos del mismo padre supera 0.55; la protección crece linealmente hasta 0.85.
2. Ambas hijas están a un máximo de 9 µm del padre y a 14 µm entre sí.
3. Todas las decisiones usan predicciones y coordenadas; no se leen etiquetas para la inferencia.

El módulo no crea nodos, enlaces ni divisiones. El ILP existente sigue decidiendo la topología. Las distancias se calculan en coordenadas físicas, tras deshacer el downsample.

**Posible fallo:** proteger una falsa división causada por células cercanas o por exceso de confianza. Aunque no cambie directamente la topología, al cambiar puntuaciones puede alterar muchos enlaces del solver. Vigilar el término de aristas, no solo la división. No se afirma que esta regla sea una novedad científica ni una implementación del paper HOCT.

**Resultado:** control 0.9666951095, candidato 0.9667049274; delta +0.000009818. Los conteos de aciertos/errores de aristas y divisiones son idénticos. Solo cambian cuatro nodos predichos y el ajuste de conteo. Los cuatro videos se encontraron en el entrenamiento del segundo detector. Se cierra sin mejora de tracking demostrada. Registro: `results/E001_completed.json`.

## E002 — FOCUS-3D como maestro de un detector rápido (diseñado)

Primero segmentar un subconjunto permitido de train; medir cobertura de los centroides anotados y número de detecciones frente a `estimated_number_of_nodes`. Si mejora esa frontera, destilar centros/heatmaps a un UNet3D rápido. Separar embriones/adquisiciones antes de generar targets. Evitar convertir automáticamente cualquier punto sin anotación en negativo.

Comparaciones: mismo linker con detector actual vs detector destilado; después combinación selectiva solo en huecos confirmados. Registrar estabilidad temporal de detecciones y precisión de localización. La inferencia completa de FOCUS-3D puede exceder el límite, según un participante; distilar es una propuesta para reducir ese coste, no una aceleración ya medida.

## E003 — HOCT sobre máscaras derivadas de intensidad (implementado)

Implementación completa y notebook Kaggle en `kaggle/hoct_diagnostic`; variante de test en `kaggle/hoct_test`. Consulta `docs/HOCT_IMPLEMENTATION.es.md` para conocer las diferencias con HOCT oficial. Sustituye todas las asociaciones, extrae morfología de imágenes y resuelve linajes con matching capacitado. Conserva detecciones para atribuir las diferencias al enlazador.

**Resultado:** 0.9195083 frente a 0.9666951 del control; delta −0.0471868. Divisiones TP/FP/FN 1/12/6 frente a 2/1/5. Se rechaza esta transferencia directa para submission. La implementación se ejecutó correctamente, pero no mejoró el tracking. No se iniciará un barrido de pequeños ajustes como continuación automática.

Usar un checkpoint público local y máscaras 3D derivadas de imágenes. Medir su linker con detecciones fijas antes de mezclar detectores. Probar CPU y GPU, memoria y tiempo por video. Reconstruir enlaces consecutivos válidos: HOCT admite gaps que no se deben exportar directamente como enlaces que salten frames.

Una vez verificadas las representaciones y las etiquetas utilizables, estudiar una cabeza ligera sobre features congeladas. No iniciar entrenamiento completo sin confirmar la disponibilidad del código y del protocolo. El código de entrenamiento aún no estaba publicado en el anuncio consultado.

## E004 — Encoder de cinco frames, atención de padres y divisiones explícitas (completado)

Se entrenaron dos modelos desde cero, cada uno reservando un embrión completo, sobre ejemplos procedentes de los 199 videos. La auditoría verificó 151 divisiones reales. Cada modelo completó 3.000 pasos y se eligió el paso 1.000 usando desarrollo del embrión de entrenamiento.

**Resultado negativo para promoción:** evaluación oficial condicional **0.9412437**, frente a **0.9666951** de Harmonic con las mismas detecciones. Delta de aristas ajustadas −0.0004514; delta de contribución de divisiones −0.025. Divisiones TP/FP/FN 0/1/7 frente a 2/1/5. En la elección de padres con centros anotados y distractores, el modelo también quedó por debajo del vecino más cercano en ambos embriones reservados. No se generó una nueva submission ni se realizó un barrido de umbrales.

El encoder y el modelo temporal sí se evaluaron fuera del embrión de entrenamiento; la comparación completa con Harmonic sigue condicionada por su detector entrenado con todos los videos. [Diseño, resultados y límites](TEMPORAL_IMPLEMENTATION.es.md). Registro: `results/E004_completed.json`.

## E005 — Ensemble temporal en leaderboard (completado)

Arturo autorizó medir los pesos de E004 en Kaggle pese al resultado local negativo. Se combinaron por igual las probabilidades de los dos modelos antes de reconstruir el linaje, conservando las detecciones y los umbrales. La submission **56223367** terminó con **0.913 público**, frente a **0.946** de E000: delta **−0.033**. El código y el CSV visible se descargaron y verificaron después del aviso del usuario; el envío ya existía y no se duplicó.

En el test visible el detector es idéntico a E000, incluidos su hash y sus 122.745 nodos. El ensemble agrega 226 enlaces, retira 272 y no produce bifurcaciones, frente a 98 bifurcaciones predichas por el control. Esto identifica una limitación del candidato, sin confundir esos conteos con eventos verdaderos anotados ni explicar por sí solo la descomposición del score oculto. Se cierra esta versión. [Resultado y límites](TEMPORAL_SUBMISSION.es.md), `results/E005_test_completed.json`.

## E006 — Supervisión densa antes de la adaptación real (completado)

Renderer procedural de películas 3D completamente etiquetadas, con divisiones y cruces negativos. Cada separación genera 2.048 escenas utilizando texturas exclusivamente de sus videos de entrenamiento. El modelo mantiene la arquitectura y adaptación de E004; cambia su inicialización mediante 4.000 pasos de preentrenamiento. El notebook encadena la adaptación de 3.000 pasos y ambas evaluaciones, con comparación automática frente a E004 y Harmonic. [Diseño y límites](DENSE_PRETRAINING.es.md).

**Resultado:** score oficial condicional **0.9436645**, una mejora de **0.0024208** sobre E004 pero una pérdida de **0.0230306** frente a Harmonic. Divisiones TP/FP/FN **1/9/6**, frente a **2/1/5** del control. La AP de división mejora de 0.1872 a 0.3819 en 44b6 y de 0.0527 a 0.0820 en 6bba; la exactitud al elegir padres sigue por debajo del vecino más cercano en ambos embriones. Se completaron ambos entrenamientos y la verificación de pesos, manifiestos y CSV. Se cierra este reemplazo completo del enlazador sin submission, conservando sus pesos para investigación. Registro: `results/E006_completed.json`.

## E007 — Especialista temporal y reparación aditiva (implementado)

Nueva rama CNN 3D + GRU bidireccional sobre seis tiempos de madre/hijas, combinada con representaciones congeladas de E006. Se entrena con 2.048 películas sintéticas de nueve frames por separación, 4.000 pasos de preentrenamiento y 3.000 de adaptación con replay. El checkpoint y el único umbral se seleccionan en desarrollo, sin consultar el embrión reservado. La inferencia agrega enlaces entre una madre con una hija y una huérfana; resuelve conflictos por asignación bipartita y conserva todas las asociaciones del control. [Protocolo, pruebas y limitaciones](MITOSIS_SPECIALIST.es.md). Aún no hay una mejora medida de este candidato.

## Orden y decisión actual

1. Conservar E000 y su 0.946 público como control. E001, E003 y E004 no demostraron mejora local; E005 confirmó un resultado público inferior con los pesos temporales. Se cierra el ensemble actual.
2. Mantener la separación por embrión de E004 para nuevos modelos; declarar por separado cualquier dependencia de un detector entrenado con el embrión reservado.
3. E006 demostró una mejora medida del ranking de divisiones, pero no del tracking completo frente a Harmonic. No promoverlo ni repetir automáticamente el mismo reemplazo con otro umbral. E007 implementa evidencia especializada de mitosis junto con las asociaciones del control. Revisar sus conteos de divisiones y métrica oficial antes de una inferencia de test.
4. E002 queda como ruta alternativa si una evaluación independiente identifica pérdidas de detección. Una mejora de clasificación o de un diagnóstico condicional no equivale por sí sola a una mejora de leaderboard.

No se transporta el umbral 0.01 de RSNA a esta métrica. No se promueve automáticamente una diferencia pequeña de un diagnóstico sobre train. Para análisis de incertidumbre, re-muestrear videos o adquisiciones, nunca miles de enlaces como si fueran independientes. Con pocos embriones, el intervalo también tiene limitaciones.

## Estado

Ver `results/STATUS.json`. E002 sigue pendiente. E003 usó inferencia HOCT real con pesos públicos; no se ha realizado entrenamiento HOCT o FOCUS-3D. E004 entrenó dos modelos propios desde cero; E005 reutilizó sus pesos y ya tiene resultado de leaderboard. E006 completó preentrenamiento sintético y adaptación real sin superar el control. E007 reutiliza sus representaciones como parte de un especialista nuevo; el estado remoto se registra por separado y no se mantiene un seguimiento local activo.
