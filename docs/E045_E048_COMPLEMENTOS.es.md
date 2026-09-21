# E045–E048: combinaciones de experimentos complementarios

Las cuatro pruebas terminaron. Ningún candidato superó su criterio de aprobación; no se hizo una nueva submission. Los scores siguientes son locales, sobre cohortes reutilizadas, no del leaderboard.

Se prueban mecanismos diferentes de combinación, con controles que permiten comprobar si la mezcla aporta algo más que cada componente. Las anotaciones parciales no se usan para generar propuestas ni se convierten en fondo. No se garantiza que exista una combinación que mejore el leaderboard.

## Revisión de aportes anteriores

| Componente | Evidencia útil | Limitación que debe resolver la combinación |
|---|---|---|
| E041, supervisión densa NIS3D | Mejor localización a 3 µm | Menor cobertura a 7 µm |
| E039, campos estático/temporal | Mayor cobertura a 7 µm, según presupuesto | Menor localización estricta |
| E033, encoder temporal aprendido | +93 decisiones correctas de progenitor | Su integración con nodos y grados fijos no cambió el score |
| E034, puentes con detecciones E031 | Recupera algunos enlaces entre fragmentos | Ganancia pequeña; el filtro visual pierde un acierto sin quitar el FP adicional |
| E027/E029, NucVerse | Recupera omisiones distintas de Harmonic | Ventanas escogidas por errores conocidos; no son una validación para submission |
| E042, resolución nativa | La referencia temporal contiene información | No supera su control de resolución; no se incluye como supuesto componente mejor |
| E035–E037/E040/E043, divisiones | Distintas propuestas de eventos | Sin mejora validada de divisiones; no se suman automáticamente a otra variante |

## E045: seis diseños de detección, CPU

Predicciones congeladas E041 y E039, mismas 80 imágenes y 488 centros anotados de desarrollo reutilizado. Se fija el protocolo antes de lanzar. Los seis diseños son: refinar cada uno de estático/temporal/máximos de imagen con centros densos a <=3 µm y correspondencia uno a uno; sumar mapas normalizados denso+estático; combinar denso+media de estático/temporal; conservar 75% de candidatos densos y añadir estáticos espacialmente distintos. No hay ajuste de pesos mediante etiquetas ni barrido de radios.

| Diseño | 256 candidatos: 3/7 µm | 512 candidatos: 3/7 µm |
|---|---:|---:|
| Denso, control | 227 / 321 | 297 / 437 |
| Estático, control | 177 / 354 | 248 / 446 |
| Temporal, control | 175 / 344 | 256 / 455 |
| Máximos de imagen, control | 194 / 319 | 268 / 423 |
| Estático refinado por denso | 197 / 348 | 269 / 445 |
| Temporal refinado por denso | 188 / 337 | 269 / 452 |
| Imagen refinada por denso | 193 / 319 | 266 / 426 |
| Consenso espacial | 227 / 336 | 307 / 449 |
| Consenso espaciotemporal | 224 / 332 | 305 / 452 |
| Denso con estáticos complementarios | 225 / 337 | 298 / 439 |

Ninguno superó el criterio principal a 256. El resultado de 512 es diagnóstico y no modifica retrospectivamente ese criterio. Se utiliza la señal del consenso espacial para definir una nueva prueba exploratoria E047 en otra cohorte. No supera al temporal a 7 µm; sí supera al denso y al estático a ambos radios en el agregado. E047 requiere mérito propio en seguimiento completo.

Los cuatro controles a 256 se reprodujeron exactamente. Proceso CPU: 112.8597 s. Cuatro pruebas nuevas de límites y correspondencias pasaron. La extracción del mapa completo E039 se refactorizó sin cambiar sus candidatos ni valores: comparación exacta con la revisión b28fa94, semilla 47, registrada en `results/E045_vote_regression.json`.

## E046: puentes y encoder sobre el grafo aumentado, CPU

Se reutilizan los donantes de E034, sus representaciones y las representaciones E033 de Harmonic. Primero se añaden los puentes; luego el encoder reasigna continuaciones sobre el grafo aumentado, incluyendo los nodos nuevos. Dos diseños: puente geométrico+encoder y puente filtrado visualmente+encoder. El predictor no consulta etiquetas. No se reentrena ni se recalculan características en GPU.

| Grafo | Score local | Enlaces TP/FP/FN |
|---|---:|---:|
| Harmonic | 0.900752960 | 6674 / 485 / 501 |
| Puentes geométricos | 0.901013522 | 6678 / 486 / 497 |
| Puentes geométricos + encoder | 0.901013522 | 6678 / 486 / 497 |
| Puentes visuales | 0.900971013 | 6677 / 486 / 498 |
| Puentes visuales + encoder | 0.900971013 | 6677 / 486 / 498 |

Las dos combinaciones no añaden ninguna ganancia a sus componentes de puente. Todas conservan divisiones TP/FP/FN=3/4/3. Proceso CPU: 183.3311 s. Se rechazan como mejora del ensamble.

El resumen oficial contiene score y divisiones, pero los recuentos de enlaces están en `samples`. Se auditaron explícitamente en `results/E046_counts_audit.json`. Tras terminar, se corrigió la lectura de esos recuentos en el runner para futuros usos: esta ejecución rechazó ambos brazos ya por el primer requisito de score, antes de consultar dichas claves; la corrección no altera su decisión ni sus métricas. El notebook versionado conserva el código realmente ejecutado.

## E047: seguimiento completo con mapas complementarios

Extensión exploratoria motivada por el diagnóstico E045 de 512, no una aprobación retrospectiva. Modelos E041 denso y E039 estático congelados; se calculan los 100 fotogramas de los otros 16 videos de calibración. Estos también se reutilizaron en experimentos anteriores y no son un holdout independiente. E031 suministra imágenes normalizadas, recortadas a [0,1] antes de las redes. Las diferencias de redondeo del float16 y del denominador de normalización frente a la preparación E038 se mantienen explícitas; no se ajustan etiquetas.

Inferencia GPU completada en 68.5013 segundos de proceso, máximo asignado 0.592 GB. Sin optimizador ni entrenamiento. Se exportan logits y campos sin comprimir para evitar dedicar GPU a compresión; todo voto espacial, detección, propuesta y evaluación se realiza en CPU.

Brazos fijados: Harmonic; puentes con denso solo (control del donante); puentes con consenso; refinamiento de Harmonic por consenso a <=3 µm; refinamiento más puentes con consenso. Presupuesto del detector 512 por imagen. Los puentes conservan la regla E034 (2–4 fotogramas, interpolación, exclusión de nodos existentes, sin reutilizar donantes). El refinamiento conserva el número de nodos e identidades y usa asignación uno a uno, revirtiendo colisiones. Las anotaciones solo se leen en la evaluación final del CSV.

Criterio: score >=Harmonic+0.001, ganancia de enlaces TP, sin aumento de enlaces FP ni divisiones FP, sin perder divisiones TP y sin caída de score en ninguno de los dos embriones. Si pasa, necesita confirmación adicional antes de una submission.

## Verificación local

### Resultado final E047

| Grafo | Score local | Enlaces TP/FP/FN |
|---|---:|---:|
| Harmonic | 0.900752960 | 6674 / 485 / 501 |
| Puentes con denso | 0.901408618 | 6682 / 487 / 493 |
| Puentes con consenso | 0.901065241 | 6682 / 490 / 493 |
| Refinamiento con consenso | 0.892901336 | 6635 / 510 / 540 |
| Refinamiento y puentes con consenso | 0.894262270 | 6644 / 507 / 531 |

Todas las divisiones permanecen en 3/4/3. El mejor candidato, denso con puentes, recupera ocho enlaces pero añade dos falsos; sube 0.000656 global y cae 0.000170 en 44b6. El consenso recupera los mismos ocho enlaces con cinco falsos adicionales y una caída mayor en 44b6. Refinar coordenadas perjudica a ambos embriones. Ninguno pasa el criterio fijado. No se seleccionan retrospectivamente reglas por embrión para forzar una ganancia.

La ejecución recuperada usó 1069.8345 segundos CPU, además del intento fallido registrado. Se reprodujo el SHA256 del control y se recalcularon los recuentos desde los resultados por video: `results/E047_counts_audit.json`. La señal de detección de E045 no se tradujo en un ensamble de seguimiento superior al componente denso. Todos los trabajos terminaron.

Las 21 pruebas de seguimiento pasaron al ejecutarse fuera del sandbox; dos habían fallado exclusivamente por permisos de creación y limpieza de directorios temporales de Windows, incluso dentro del workspace. No se modificó el algoritmo para eludirlas. Cuatro pruebas nuevas E045 pasaron. Los manifiestos, pesos y archivos de entrada se fijan mediante SHA256 en los protocolos y recibos privados.

## Corrección de ejecución E047

La primera evaluación CPU se detuvo antes de puntuar: el control Harmonic puede contener IDs diferentes con coordenadas redondeadas coincidentes. El refinador asumía que todas sus anclas tenían posiciones distintas. Se corrigió para preservar esas anclas coincidentes y excluir sus posiciones de los nuevos destinos; los demás nodos mantienen la asignación uno a uno. Se añadió una prueba específica, que pasó junto con las otras cuatro E045. La revisión no altera el objetivo ni usa etiquetas para corregirlo. Se relanzó como `E047_RECOVER` para conservar intactos los recibos del intento fallido. No se repitió la inferencia GPU.

## E048: DivNet público compatible como complemento de divisiones

Derivado de la revisión pública descrita en `REVISION_KAGGLE_20260920.es.md`. La adaptación pública de un notebook anunciado como 0.948 no carga sus pesos DivNet; se verificó y se implementó una arquitectura compatible que carga los 50 tensores estrictamente. No se ejecutó el pipeline público completo ni sus mecanismos inválidos de puntuación.

Prueba exploratoria CPU, sin entrenamiento, sobre los 316 forks originales de Harmonic. Dos interpretaciones de normalización fijadas antes de evaluar: imagen completa o recorte. Cuatro lags y marcador, XY4 fijo. Rechazar probabilidad <0.5; eliminar la arista a la hija más distante, conservar nodos y resto del grafo. No crea divisiones nuevas. Control y dos candidatos se evalúan con la métrica completa. Se requiere score +0.001, menos divisiones FP y no perder divisiones TP ni enlaces TP, sin aumentar enlaces FP.

El checkpoint público resume 199 videos; no se conoce la lista exacta usada en `best_overall`. La evaluación es condicional y no una validación independiente. Tampoco se afirma identidad exacta con el preprocesamiento original de entrenamiento. La implementación de recortes sí concuerda con la función pública sobre entradas idénticas, error máximo 2.3842e-7. Dos pruebas locales verifican marcador fraccional y bordes.

### E048: resultado

Completado en 113.6973 segundos CPU. Se puntuaron los 316 forks: probabilidades [0.514637,0.551911] con normalización de imagen y [0.516549,0.552025] con normalización de recorte. Ambas variantes quedan enteramente por encima del umbral publicado 0.5, no retiran ninguna arista y generan exactamente el mismo SHA256 de CSV que Harmonic. Score 0.9007529596, enlaces 6674/485/501 y divisiones 3/4/3 en todos los brazos. No hay mejora ni submission. No se barre el umbral sobre estas pocas divisiones para forzar una ganancia. Véanse `results/E048_completed.json` y `results/E048_audit.json`.
