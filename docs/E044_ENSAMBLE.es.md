# E044 — Ensamble de supervisión densa y campo estático

Pregunta: ¿pueden dos componentes que no superaron por sí solos su criterio completo complementarse al combinar sus detecciones?

Se reutilizan las predicciones congeladas de E041 (supervisión densa externa NIS3D) y E039 (campo estático aprendido con anotaciones parciales Biohub). No se entrenan modelos nuevos y todo el ensayo utiliza CPU. Los radios de 3 y 7 µm son tolerancias de evaluación, no resoluciones de entrenamiento diferentes.

## Protocolo fijado antes de ejecutar

- Mismos 16 videos de desarrollo, cinco imágenes por video y 488 centros anotados; desarrollo reutilizado, no prueba independiente.
- Entradas: hasta 512 máximos por modelo, ordenados por sus propios valores. No se promedian puntuaciones sin calibración.
- Mezcla principal: alternar candidatos por rango, primero denso en cada rango; descartar candidatos a distancia de un vóxel isotrópico o menor de uno ya aceptado. Detenerse en 256 candidatos por imagen. No utiliza etiquetas para seleccionar puntos.
- Controles: primeros 256 candidatos de cada componente. Deben reproducir exactamente 227/321 aciertos a 3/7 µm para E041 y 177/354 para E039.
- Diagnóstico de complementariedad: centros anotados emparejados solo por uno de los modelos, por ambos o por ninguno. Depende de la asignación uno a uno; también se evalúa la unión real de los candidatos.
- Unión diagnóstica: hasta 512 candidatos (256+256). Usa más presupuesto y no constituye una comparación justa ni un candidato aprobado.
- Criterio del componente: en cada embrión, igualar o superar al mejor componente a 3 µm y superarlo a 7 µm. Fijado antes de conocer resultados.

Los aciertos se obtienen mediante asignación húngara de cardinalidad máxima con distancia física de 1.625 µm por vóxel isotrópico. Los centros sin anotación no se clasifican como falsos positivos. Incluso un resultado favorable necesitaría evaluación posterior del grafo completo y de las divisiones antes de justificar una submission.

## Trazabilidad

Protocolo y hashes de las tres fuentes: `baseline/e044_protocol.json`. Código: `scripts/ensemble_evaluate_runner.py`. Recibo de lanzamiento: `results/E044_EVALUATE_launch.json`. Notebook privado: https://www.kaggle.com/code/jarturo/biohub-e044-evaluate-cpu .

Cuatro pruebas locales verifican presupuesto y supresión de coincidencias, emparejamiento uno a uno, unidades físicas y entradas vacías.

## Resultado completo

El ensamble simple no superó el criterio. Sí existe complementariedad entre componentes; no se ha demostrado una mejora del sistema ni del leaderboard.

| Candidatos por imagen | Modelo | Aciertos a 3 µm | Aciertos a 7 µm |
|---|---|---:|---:|
| 256 | Denso E041 | 227 | 321 |
| 256 | Estático E039 | 177 | 354 |
| 256 | Ensamble E044 | 211 | 329 |
| 512 | Unión diagnóstica E044 | 272 | 394 |
| 512 | Denso E041, evaluación anterior | 297 | 437 |
| 512 | Estático E039, evaluación anterior | 248 | 446 |

La unión tampoco supera a los mejores componentes con 512 candidatos: su ganancia frente a 256 no prueba una ventaja del ensamble. A presupuesto 256, en 44b6 la mezcla alcanza 71/88 frente a denso 69/84 y estático 56/93; en 6bba alcanza 140/241 frente a 158/237 y 121/261. Por tanto, no basta con la mejoría localizada del primer embrión.

En las asignaciones de los componentes de 256 candidatos, a 3 µm hay 132 centros comunes, 95 exclusivos del denso y 45 exclusivos del estático; a 7 µm hay 281 comunes, 40 exclusivos del denso y 73 exclusivos del estático. Son centros anotados recuperados, no una estimación de precisión. La unión real reproduce 272/394 aciertos.

Decisión: conservar los dos modelos como posibles componentes y descartar esta mezcla por alternancia de rango. Una hipótesis posterior sería preservar las propuestas del campo estático y usar el mapa denso para refinar su localización con correspondencias uno a uno; sería una prueba nueva, todavía no ejecutada, y podría empeorar centros correctos. Otra sería seleccionar propuestas por persistencia temporal. No hay evidencia para prometer que sumarán sus mejoras ni para enviar E044 al leaderboard.

Los controles se reprodujeron exactamente. Se verificaron SHA256 de manifiestos, predicciones, imágenes y etiquetas. Las cuatro pruebas locales pasaron. Ejecución del proceso: 49.6009 segundos, exclusivamente CPU; excluye inicio y exportación de Kaggle. Estado final COMPLETE, sin GPU, sin submission ni trabajos pendientes.
