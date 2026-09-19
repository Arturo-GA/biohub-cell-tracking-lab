# E017 completado: comparación final con Harmonic en CPU pendiente

19 de septiembre de 2026. Ambos notebooks anteriores terminaron correctamente y sus paquetes fueron verificados: 25 archivos del nuevo seguimiento CPU y 42 del control Harmonic.

| Métrica en los mismos 16 videos | Geométrico E016 | Trayectorias E017 |
| --- | ---: | ---: |
| Score oficial local | 0.788782 | 0.818006 |
| Jaccard de enlaces sin ajuste | 0.800147 | 0.829487 |
| Cobertura media de nodos | 99.3262 % | 99.2576 % |
| Enlaces TP | 6534 | 6470 |
| Enlaces FP | 991 | 625 |
| Enlaces FN | 641 | 705 |
| Nodos activos | 364674 | 363279 |
| Divisiones TP / FP / FN | 0 / 0 / 6 | 0 / 0 / 6 |

E017 mejora +0.0292235 y gana en 12 videos, con cuatro regresiones. Reduce 366 falsos enlaces, pero pierde 64 correctos. Su mayor continuidad temporal no significa recuperar todas las trayectorias: parte de la mejora viene de evitar asociaciones incorrectas, con un costo de cobertura de enlaces. No incorpora divisiones.

El proceso CPU del runner tardó 183.58 segundos. La inferencia completa de Harmonic tardó 3055.29 segundos (aproximadamente 50 min 55 s). Son tiempos de los respectivos procesos, no mediciones de cuota facturada. El control generó su CSV completo para los 16 videos; no ejecutó la evaluación oficial durante la sesión GPU.

## Comparación pendiente, ya lanzada

[Biohub Lab Full Calibration Compare CPU](https://www.kaggle.com/code/jarturo/biohub-lab-full-calibration-compare-cpu), versión 1, privado y solo CPU.

Verifica hashes de resultados, predicciones de Harmonic e informes previos. Valida su exportación y conserva el original. Solo permite corregir, con recibo explícito, una coordenada espacial entera exactamente igual al límite superior, proyectándola al último voxel; rechaza otras excursiones y valores no enteros. Este es el mismo tipo de redondeo identificado anteriormente en E013, no una corrección de trayectorias ni una optimización por GT.

Evalúa Harmonic con la métrica oficial fijada y compara con los informes congelados de E016 geométrico, E016 aprendido y E017. No repite entrenamiento, inferencia de imágenes ni selección de parámetros. La comparación por video incluye cambios en TP, FP, FN y Jaccard ajustado. No toca el conjunto de 48 videos de evaluación anterior.

## Decisión

Conservar la hipótesis de movimiento colectivo y contexto temporal para esta comparación completa: la mejora frente a geometría es mayor y más distribuida que en E016. Todavía no enviar al leaderboard ni afirmar que supera Harmonic. El control débil no basta para decidir una promoción.

Cuando termine la evaluación, decidir si existe evidencia para integrar razonamiento temporal con el sistema completo, preservando sus divisiones, o si debemos cambiar de representación. No ajustar retrospectivamente parámetros para los cuatro videos perdedores ni seleccionar sistemas por nombre de video.

## Verificación

`scripts/audit_e017_completed.py` compara código ejecutado con payloads, comprueba cohortes, selección sin GT entre las dos inicializaciones, mejora del objetivo interno y agregados oficiales recalculados desde informes por video. CSV y pesos no fueron descargados ni reejecutados localmente. Recibo: `results/E017_completed.json`.

Tres pruebas de exportación pasaron: CSV válido intacto, corrección exclusiva del límite superior y rechazo de otras excursiones/fracciones. El nuevo notebook y sus fuentes se compilaron. Recibos de preparación y lanzamiento: `results/E017_compare_preflight.json` y `results/E017_compare_launch.json`.

La referencia pública propia sigue siendo 0.946. Todos los scores de esta tabla son de desarrollo condicional en 16 videos; no son puntuaciones de leaderboard. No se configuró monitoreo ni envío automático.
