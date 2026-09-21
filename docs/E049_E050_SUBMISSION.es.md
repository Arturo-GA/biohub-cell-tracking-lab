# E049–E050: pesos por candidato y submission exploratoria

El usuario aportó una captura ordenada por **Public Score**: Biohub 0.95 muestra 0.950; el notebook de proxy, Harmonic V3 y Harmonic muestran 0.947. Es evidencia de los valores del listado de Kaggle, no solo de los títulos. Se corrige la incertidumbre expresada en la respuesta anterior. Esto no demuestra que toda modificación interna del código sea responsable del score ni elimina los errores de carga DivNet auditados.

## Cambio aprovechado de los códigos públicos

Se descargó [Harmonic V3 de raunakdey07](https://www.kaggle.com/code/raunakdey07/biohub-harmonic-fusion-v3) y el archivo público `ppsweep_selected.json` de este y del [notebook de evgendvorkin](https://www.kaggle.com/code/evgendvorkin/biohub-0-942-lb-proxy-score-0-9417). Ambos seleccionan `tight55`, es decir, `MOTION_RELINK_TIGHT_UM=5.5`. El control local anterior conserva 6.0. Se fija el valor público antes de escribir el CSV de test; no se ejecuta el barrido con anotaciones durante una submission.

Los archivos publicados indican proxy 0.949038→0.951095 en ocho videos. Estos valores son del proxy; el score 0.947 procede de la captura del listado. Buena parte de la fusión de dos modelos y la protección de detecciones ya coincide con nuestra base. Los añadidos artificiales fuera del volumen del notebook 0.95 no se incorporan.

## E049: variantes fijadas antes de evaluar

Se reutilizan candidatos E047 y grafos de los mismos 16 videos. Sin GPU, sin reentrenamiento, sin nombres de embriones como variable de decisión y sin consultar etiquetas al proponer cambios.

- **Denso:** reproduce exactamente los puentes E047, como control.
- **Ponderado:** conserva al menos 75% del peso de localización del detector denso. El experto complementario recibe más peso cuando su máximo tiene mejor rango y concuerda espacialmente con el denso; recibe cero si dista más de 3 µm. Solo completa huecos, conserva los nodos originales.
- **Movimiento:** estima desplazamiento local con la mediana de hasta ocho trayectorias próximas. La historia propia recibe más peso si concuerda con el vecindario. Combina fiabilidad temporal y acuerdo entre detectores para aceptar puentes. Inspirado en la idea pública de movimiento local, implementado aquí sin copiar el pipeline.
- **Ponderado + movimiento:** aplica ambos mecanismos.

Son indicadores de fiabilidad observables, no una garantía de cuál modelo acierta. No se ajustan pesos por embrión después de mirar las anotaciones.

| Variante | Score local | Enlaces TP/FP/FN |
|---|---:|---:|
| Harmonic | 0.900752960 | 6674 / 485 / 501 |
| Denso | 0.901408618 | 6682 / 487 / 493 |
| Ponderado | **0.901409553** | 6682 / 487 / 493 |
| Movimiento | 0.900690271 | 6675 / 486 / 500 |
| Ponderado + movimiento | 0.900690240 | 6675 / 486 / 500 |

Las divisiones permanecen en 3/4/3. El movimiento rechaza demasiados enlaces útiles. El ponderado es el máximo según la selección fijada; la diferencia de menos de una millonésima frente a denso no es evidencia de superioridad relevante. La ganancia frente a Harmonic sigue acompañada de dos falsos adicionales. CPU: 171.918 s de proceso. Pasaron tres pruebas locales de conservación de nodos/topología, desacuerdo espacial y ausencia de evidencia de movimiento.

## E050: decisión y límites

Se prepara el mejor candidato E049 con el valor público `tight55`. El usuario pide expresamente medir el mejor candidato en Kaggle, aun con ganancias locales pequeñas; se cambia la política anterior de no enviar si aparecía cualquier FP adicional. No se presenta como aprobación del criterio antiguo ni como mejora segura.

Esta combinación con `tight55` no tiene evaluación local conjunta: es transferencia exploratoria. La calibración está reutilizada y no es una validación independiente. La inferencia final recalcula todas las imágenes de test, carga los pesos congelados estrictamente, no lee anotaciones y no consume predicciones cacheadas de los ejemplos visibles. Se validan coordenadas, IDs, aristas, grados y cobertura antes de enviar. La GPU se requiere para las redes volumétricas; los votos y grafos se calculan en CPU dentro del notebook final que debe poder ejecutarse sobre test oculto.

**Enviado:** notebook privado [jarturo/biohub-e050-weighted-complement](https://www.kaggle.com/code/jarturo/biohub-e050-weighted-complement), versión 1. Kaggle aceptó la submission **56409893**, el 2026-09-21 01:03:12 UTC. Resultado confirmado el 2026-09-21: **COMPLETE, 0.946**, igual al control y la submission visual anterior. Recibos: `results/E050_SUBMISSION_attempt.json` y `results/E050_SUBMISSION_status.json`.

El CSV descargado se validó localmente y su SHA256 coincide con el producido en Kaggle: `7a022f034c1b75ec591f4c424a0b788f19410aeb87eb9ed9699a3610f28a3697`. Total 122902 nodos y 118669 enlaces. Se completaron 49 huecos con 108 nodos y 157 enlaces nuevos; ningún nodo original se movió. El peso medio del experto complementario varió entre 0.0903 y 0.2117 por video. Esto confirma que el mecanismo está activo, no que todas esas adiciones sean correctas.

Tiempo del runner completo: 1767.07 s; etapa complementaria: 185.95 s. Son tiempos de proceso, no una medición de facturación GPU. Se detectaron búsquedas recursivas innecesarias de modelos en el código público; la primera pausa duró unos 302.5 s. Se conserva la ejecución y se registra la optimización pendiente de priorizar rutas verificadas antes de buscar en todo el dataset (`results/E050_lookup_latency.json`). No se reentrenaron modelos.
