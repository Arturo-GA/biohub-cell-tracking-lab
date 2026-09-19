# E016 completado y auditado — 19 septiembre 2026

El entrenamiento terminó: 3000 pasos sobre 48 videos, seguido de evaluación sobre 16 videos de calibración. Todo se ejecutó en CPU. El runner tardó 157.18 segundos en total; esta cifra no incluye preparación de la sesión ni instalación de dependencias.

| Métrica, mismos 16 videos | Control geométrico | Modelo aprendido |
| --- | ---: | ---: |
| Score oficial local | 0.788782 | 0.789741 |
| Jaccard de enlaces sin ajuste | 0.800147 | 0.801276 |
| Cobertura media de nodos | 99.3262 % | 99.3318 % |
| Enlaces TP | 6534 | 6532 |
| Enlaces FP | 991 | 977 |
| Enlaces FN | 641 | 643 |
| Nodos activos predichos | 364674 | 365696 |
| Divisiones TP / FP / FN | 0 / 0 / 6 | 0 / 0 / 6 |

La diferencia de score es +0.0009585. El modelo reduce 14 falsos enlaces, pero pierde dos enlaces correctos. Mejora en 4 videos y empeora en 12 según el Jaccard ajustado. Los ocho videos del grupo 44b6 empeoran; los cuatro videos que mejoran pertenecen al grupo 6bba. En Jaccard sin ajuste hay cuatro empates adicionales. Esto muestra una respuesta desigual entre videos; no demuestra una mejora generalizable.

## Qué aprendimos

La comparación controlada atribuye al entrenamiento únicamente la pequeña diferencia entre 0.788782 y 0.789741. El salto respecto al score 0.484370 de E015 no se puede atribuir al aprendizaje: E016 también cambia el conjunto de detecciones a origen Harmonic y el mecanismo de asociación. El control sin aprendizaje ya consigue casi todo ese salto.

Se usaron 29104 ejemplos supervisados, de los cuales solo 45 tenían etiqueta nula (0.155 %). Por ello, la opción de no asociar tiene poca supervisión positiva. Además, las detecciones sin anotación se enmascaran durante entrenamiento, pero compiten durante inferencia. Ambos son límites concretos del diseño; no queda probado que expliquen por sí solos los errores observados.

La pérdida del último minibatch fue 0.01959. No es una métrica de generalización y no justifica aumentar pasos. El modelo continúa sin representar divisiones.

## Decisión y siguiente trabajo

No enviar E016 al leaderboard ni extender este entrenamiento. Conservar el checkpoint y los informes para diagnóstico. El sistema Harmonic completo sigue siendo la referencia, con score público propio 0.946; ninguna cifra local de esta tabla es un score público ni una comparación contra Harmonic completo.

El siguiente trabajo debe cerrar la comparación pendiente contra Harmonic completo en los mismos 16 videos y examinar sus desacuerdos con E016. Reutilizar predicciones o características existentes si están disponibles. Si hacen falta nuevas características visuales para reproducir fielmente Harmonic, estimar esa etapa antes de asignarle GPU; el análisis de grafos y la evaluación siguen en CPU.

La dirección de implementación preferida es preservar trayectorias fiables del control y aprender reparaciones en casos ambiguos con contexto temporal, en lugar de reemplazar todas las asociaciones con esta cabeza de pares. La selección de casos debe depender de imágenes, geometría o incertidumbre, no del nombre del video o de elegir retrospectivamente los cuatro videos ganadores. No se ha lanzado otro experimento en esta revisión.

## Verificación reproducible

`scripts/audit_e016_completed.py` verificó los 22 archivos del paquete ejecutado contra el payload congelado, las cohortes exactas, los 12 registros de progreso hasta el paso 3000, la versión fijada de la métrica y los hashes declarados de predicciones/checkpoint. Recalculó los scores a partir de los conteos por video y comprobó su igualdad con los informes del runner.

La evaluación oficial se ejecutó en Kaggle. No se descargaron ni verificaron localmente el contenido de los CSV o del checkpoint; los hashes se conservan como declarados por el runner verificado. Recibo: `results/E016_completed.json`. No hubo envío al leaderboard ni monitoreo automático.
