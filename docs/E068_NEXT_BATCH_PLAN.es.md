# Próxima tanda: mejora y resistencia al cambio del conjunto privado

> **Actualización operativa 27-09:** este documento conserva el plan inicial. Los resultados públicos cambiaron la prioridad hacia c3 (0.955) con tau 0.6. CPU4–CPU6 ya terminaron. Véase [E068_CPU6_RESULTS.es.md](E068_CPU6_RESULTS.es.md) para la decisión actual, C3_fork8 y CPU7; no ejecutar automáticamente el orden A–E propuesto aquí.

Fecha: 26-09-2026. Estado: propuesta; no se han lanzado nuevos kernels ni envíos al preparar este plan. Base revisada: `main`, commit `1b4b4c4`.

**Actualización tras autorización:** el usuario pidió empezar y aclaró que la prioridad es mejorar el score público con una submission robusta, sin exigir garantías del privado. Se relajan los filtros privados de la sección 4: sus resultados se reportan como diagnóstico, no como vetos automáticos por perder en una mezcla simulada. Se mantienen reproducción correcta, evaluación completa y ejecución estable como requisitos. Se lanzó `jarturo/biohub-exact-replay-cpu4` v1, privado y sin GPU. La API confirmó el cierre 29-09-2026 23:59 UTC y cinco envíos realizados el 26; todavía no se ha enviado ninguna nueva submission durante esta ejecución.

## 1. Evidencia que cambia la decisión

El control competitivo es **x138 + V1284 + L7**, público confirmado **0.954**, envío `56540269`. Se conserva intacto. El puesto 177 y el corte aproximado de plata del traspaso son una fotografía, no el ranking actual ni una garantía del privado.

El laboratorio `jarturo/biohub-safediv-lab-cpu3` terminó. Sus salidas descargadas están en `outputs/e068_review/cpu3/`: 199 videos, 71 de 44b6 y 128 de 6bba; 20 configuraciones y cero discrepancias del replay de aristas de base. Resultados provisionales:

| Variante | Mejora local frente a base | Divisiones TP / FP |
|---|---:|---:|
| Base | 0 | 24 / 65 |
| XR: poda E + cortes + fork8-nanonly | +0.002161 | 24 / 49 |
| Tau 0.9 solo en 6bba | +0.003018 | 34 / 88 |
| Tau 1.0 solo en 6bba + XR | +0.005043 | 33 / 70 |
| **Tau 0.9 solo en 6bba + XR** | **+0.005433** | **33 / 65** |

La última combinación mejora provisionalmente ambos grupos: +0.001423 en 44b6 y +0.006304 en 6bba. Aumentar tau en 44b6 empeora ese grupo. Por ello se reemplaza la propuesta anterior de gastar un envío en tau 1.0 global. Tau 0.9 obtiene las mismas divisiones verdaderas que 1.0 y cinco falsas menos en los combos comparables.

**Estos deltas no se suman al 0.954 público.** CPU3 conserva posiciones suavizadas del grafo original aunque las variantes cambien enlaces. El score real de cada candidato exige recomputarlas. Además, estos 199 videos contienen datos usados en entrenamiento y selección; no constituyen un holdout independiente. Véase `docs/E068_CODEX_REVIEW.es.md`.

## 2. Primera fase: evaluación fiel, en CPU

1. Corregir el replay para aplicar exactamente el orden del notebook: cambios de grafo, filtro L7, reglas posteriores, suavizado con el grafo resultante y conversión final a coordenadas enteras. Reutilizar las capturas existentes; no repetir inferencia GPU.
2. Verificar equivalencia con la implementación de referencia en casos con cortes, divisiones, eliminación de componentes y límites de imagen. Exigir reproducción del control y de variantes; reproducir solo las aristas de base no basta.
3. Medir tiempo en cinco videos representativos y extrapolar antes de lanzar la comparación completa. Ejecutar los finalistas en los 199 videos, sin omitir errores ni comparar intersecciones de videos incompletas.
4. Reparar la verificación previa al envío: marcador final obligatorio, versión y hash exactos, videos y conteos coherentes con cada candidato, ausencia de fallos o activación del límite de reparación. Comprobar el CSV fuera del notebook de inferencia. No reutilizar las tolerancias `1e9` del script actual.

Si el replay corregido cambia las conclusiones, se actualiza la selección antes de gastar GPU o submissions. El primer producto de esta fase es una tabla reproducible por video, no un nuevo score público.

## 3. Candidatos: comparación limitada y con controles

Todos mantienen pesos V1284, L7, tau 0.6 en 44b6 y una sola pasada. Los nombres A–E son etiquetas de planificación, todavía no nuevos kernels.

- **P8:** poda uniforme de componentes con longitud <8 y probabilidad <0.7.
- **Poda E:** misma regla con longitud <8 en 44b6 y <11 en 6bba.
- **Cortes:** reglas existentes `cut_nan_dist_um=8`, `cut_end_prob=0.5`, `cut_end_mode=last`.
- **Fork:** `fork_min_branch=8`, `fork_nan_only=1`; no poda general de todas las ramas cortas.

| Candidato | Tau en 6bba | Poda | Cortes | Fork | Pregunta que resuelve |
|---|---:|---|---|---|---|
| **A: principal simple** | 0.9 | P8 | Sí | Sí | ¿Se conserva la recuperación de divisiones con una poda uniforme? |
| **B: mayor evidencia local actual** | 0.9 | E | Sí | Sí | ¿La poda por grupo aporta frente a A después de corregir el replay? |
| **C: protección de ramas** | 0.9 | P8 | Sí | No | ¿La poda de bifurcaciones está borrando divisiones verdaderas? |
| **D: defensivo** | 0.6 | P8 | Sí | Sí | ¿Se gana reduciendo errores sin relajar la generación de divisiones? |
| **E: comparador existente** | 1.0 | P8 | Sí | Sí | ¿0.9 mejora realmente frente a S5 v2, ya ejecutado? |

Evaluar primero control, D, A y B; después C y E. A es una hipótesis nueva sobre el combo, no el ganador ya demostrado. B sí corresponde al mejor combo provisional de CPU3. Añadir como referencia local XR con poda E y tau 0.6, ya medido por CPU3, para separar el efecto de la poda del de las divisiones.

Orden previsto de envío: A/B según evaluación corregida, D, el otro A/B, C y E. **Hasta cinco**, no cinco obligatorios: descartar candidatos dominados, idénticos en salida o que fallen los controles. S5 v2 puede reutilizarse solo después de verificar su versión y artefacto; no requiere otra inferencia por estar en esta tabla.

## 4. Pruebas específicas contra el shake-up

Congelar las configuraciones anteriores antes de mirar sus nuevos scores públicos. Para cada una, guardar métrica oficial, errores de enlace/división, conteos, tiempo y diferencias frente al control y frente a D.

1. **Grupos por separado:** verificar 44b6 y 6bba; repetir la comparación dando 25%, 50% y 75% de peso a 44b6, recalculando la agregación correspondiente. No elegir únicamente por la mezcla de entrenamiento. Dos grupos no permiten estimar la generalización a cualquier embrión nuevo.
2. **Concentración de ganancias:** quitar los 1, 3 y 5 videos que más contribuyen a la mejora y volver a comparar. Medir también el decil más perjudicado y las divisiones verdaderas perdidas. Una ganancia que desaparece al quitar un video es frágil.
3. **Condiciones difíciles:** desglosar por densidad de detecciones, movimiento y calidad de imagen, calculados sin etiquetas. Buscar pérdidas repetidas en un régimen concreto. Las regiones sin anotación no se tratan como negativos.
4. **Estabilidad de parámetros:** probar 0.85 y 0.95 alrededor de tau 0.9 solo en CPU, manteniendo el resto fijo. Son una prueba de estabilidad, no una invitación a escoger el decimal máximo. No abrir una nueva búsqueda grande.
5. **Incertidumbre pareada:** bootstrap por video y, si existen grupos de adquisición verificables, por esos grupos. Una cota inferior del 90% positiva sirve como criterio operativo complementario; no corrige reutilización de datos ni equivale a probabilidad de medalla.

Para ser finalista principal: mejora global reproducible, ausencia de pérdida media en cualquiera de los dos grupos, estabilidad bajo las mezclas anteriores y ganancia que siga positiva tras retirar los cinco mayores contribuyentes. Ante fallo de estos criterios, no maquillar el resultado cambiando el criterio: mantenerlo como candidato exploratorio o descartarlo. Si ninguno pasa, conservar el control 0.954.

Preferir la configuración más sencilla cuando las diferencias sean pequeñas o inciertas. No aprender un selector por video con sus scores de validación. Para grupos desconocidos, cualquier despacho por identidad debe tener una política explícita y conservadora, tau 0.6 y poda uniforme, validada técnicamente antes de usarse.

## 5. Combinación y segunda tanda

La primera tanda ya combina mecanismos complementarios: recuperación de divisiones y reducción de enlaces/ramas espurias. No promediar CSV ni unir aristas indiscriminadamente: los identificadores y la topología pueden ser incompatibles.

Solo preparar una segunda tanda si la primera deja un conflicto concreto: por ejemplo, A recupera divisiones pero C conserva ramas verdaderas que A elimina. En ese caso, probar **una regla de protección de poda basada en la confianza de las dos ramas**, usando las probabilidades ya disponibles, y compararla con A y C. La regla se congela y se evalúa en los 199 videos antes de enviarse. No decidir excepciones mirando etiquetas de cada video ni ajustar pesos según leaderboard.

Si no hay una señal complementaria clara, no consumir el día siguiente repitiendo variantes equivalentes. No priorizar ahora otro entrenamiento de V1284, una arquitectura nueva, `parent12` global, tau global elevado ni los umbrales agresivos ya negativos.

## 6. Recursos, calendario y finales

- **Laptop y Kaggle CPU:** replay, métricas, análisis y preparación. Trabajar con capturas, sin volver a cargar todo el flujo de imágenes para cada umbral.
- **GPU Kaggle:** únicamente las inferencias/exportaciones finales que realmente la requieren. Estimación del traspaso: 20–30 minutos visibles por variante; cuatro nuevas serían aproximadamente 80–120 minutos, sin garantía. Consultar cuota real antes y reservar capacidad para una reparación. Separar este coste del tiempo de scoring oculto.
- **Kernel final:** `BIOHUB_VALIDATOR_ENABLE=0`, `REPAIR_DEADLINE_S=34200`, una sola llamada de exportación. Nada de barridos o una segunda validación dentro del límite de ejecución.
- Según el calendario del traspaso, la siguiente renovación es **27-09 00:00 UTC = 26-09 19:00 Lima**. Si los controles no están terminados, no enviar solo por alcanzar esa hora.
- Reserva conservadora: último envío **29-09 10:00 UTC = 05:00 Lima**; seleccionar finales antes de **29-09 20:00 UTC = 15:00 Lima**. El traspaso sitúa el cierre a las 23:59 UTC del 29. Confirmar plazo, cupos y reglas vigentes en Kaggle antes de ejecutar; la página oficial de timeline no devolvió contenido legible durante esta revisión. El margen propuesto es operativo, no una afirmación de que Kaggle exija finalizar el scoring antes del cierre.

**Final 1:** mejor combinación que pase los controles de robustez y no muestre una degradación pública inexplicada. **Final 2:** variante defensiva D si se sostiene, o el control probado `56540269`. No elegir automáticamente dos variantes casi idénticas solo porque empaten arriba; comparar dónde pierden y qué errores comparten. La complementariedad se mide con errores/deltas por video, no con correlación de scores brutos dominada por la dificultad de los videos.

No hay evidencia suficiente para asignar una probabilidad de plata ni un score privado esperado. La meta de esta tanda es conseguir una mejora transferible sobre 0.954 y conservar una segunda opción menos dependiente de recuperar divisiones adicionales.
