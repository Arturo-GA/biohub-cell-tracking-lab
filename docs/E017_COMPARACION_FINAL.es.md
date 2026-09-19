# Comparación final E017: Harmonic sigue siendo superior

19 de septiembre de 2026. Evaluación completada en CPU sobre exactamente los mismos 16 videos de calibración, con la misma versión fijada de la métrica. No hubo correcciones de coordenadas: el hash del CSV validado coincide con el original.

| Sistema | Score local | Jaccard de enlaces sin ajuste | TP / FP / FN de divisiones |
| --- | ---: | ---: | --- |
| Harmonic completo | **0.900753** | **0.871279** | 3 / 4 / 3 |
| E017 movimiento del tejido | 0.818006 | 0.829487 | 0 / 0 / 6 |
| E016 aprendido | 0.789741 | 0.801276 | 0 / 0 / 6 |
| E016 geométrico | 0.788782 | 0.800147 | 0 / 0 / 6 |

E017 pierde 0.0827473 de score frente a Harmonic. Solo supera su Jaccard ajustado en dos videos y pierde en catorce. Los otros dos sistemas pierden en los dieciséis. Las doce victorias de E017 reportadas antes eran frente al control geométrico, no frente a Harmonic.

La diferencia no se debe únicamente a las divisiones: Harmonic logra 6674 enlaces correctos y 485 falsos, frente a 6470 correctos y 625 falsos de E017. Harmonic tiene 501 enlaces anotados omitidos frente a 705. Además, utiliza 318094 nodos activos, frente a 363279. Su cobertura media de nodos es menor (97.80 % frente a 99.26 %), pero reconstruye mejor las asociaciones.

Del déficit de score, 0.03 corresponde al término de divisiones y aproximadamente 0.05275 al Jaccard ajustado de enlaces. Por tanto, añadir simplemente las divisiones de Harmonic no resolvería el problema de asociaciones. Tampoco se pueden combinar scores como si sus predicciones fueran compatibles automáticamente.

## Decisión

Rechazar E016 y E017 como reemplazos completos; conservar Harmonic. No hacer submission ni nuevos barridos de parámetros sobre estas variantes. La nueva idea temporal superó a geometría, pero no aportó suficiente información para superar al sistema con inferencia visual completa.

Para decidir otra arquitectura, falta medir qué enlaces correctos adicionales obtiene E017 donde Harmonic falla y qué evidencia visual/temporal permitiría reconocerlos sin usar GT en inferencia. Los conteos agregados actuales no identifican ese solapamiento. Ese análisis puede hacerse en CPU con predicciones existentes; no requiere repetir entrenamiento ni inferencia GPU. Solo si existe un conjunto útil de errores recuperables tendría sentido desarrollar una representación conjunta de apariencia, dinámica y divisiones. No elegir variantes por nombre de video ni mezclar retrospectivamente los resultados ganadores.

Esta revisión no lanzó nuevos experimentos. No hay ejecuciones pendientes registradas de E017 ni monitoreo automático. La referencia pública propia sigue siendo 0.946; estos scores son locales y no se comparan directamente con el leaderboard.

## Auditoría

`scripts/audit_e017_compare.py` verificó los 25 archivos del paquete de evaluación contra el payload congelado; cohortes y versión oficial; hashes de los informes previos; conteos y agregación de los cuatro sistemas; diferencias por video y ausencia de modificaciones de exportación. El proceso de comparación duró 45.19 segundos en CPU, sin incluir preparación de sesión.

La métrica se ejecutó en Kaggle. Localmente se recalcularon los agregados desde los informes; no se descargaron ni reejecutaron los CSV. Recibo: `results/E017_compare_completed.json`. Código e informes agregados guardados en GitHub privado.
