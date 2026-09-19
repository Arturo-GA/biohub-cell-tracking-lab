# E015 completado — 19 septiembre 2026

El notebook privado terminó correctamente, sin GPU. El proceso medido por el runner duró 92.05 segundos; no incluye preparación de sesión ni instalación de dependencias.

| Métrica, mismos 16 videos de calibración | E014 original | E015 |
| --- | ---: | ---: |
| Score oficial de desarrollo | 0.302565 | 0.484370 |
| Jaccard de enlaces sin ajuste | 0.348005 | 0.524962 |
| Cobertura de nodos | 98.6712 % | 98.4668 % |
| Nodos predichos | 774181 | 590382 |
| Divisiones TP / FP / FN | 0 / 0 / 6 | 0 / 0 / 6 |

Los 16 videos mejoraron tanto en Jaccard sin ajuste como ajustado. La mejora no se explica únicamente por penalizar menos el número de nodos. La proyección combina selección de representantes y reasignación global de enlaces: este experimento no separa la contribución causal de ambas operaciones.

## Qué demuestra el diagnóstico

En el grafo original, 5778 de las 7299 células anotadas cubiertas por la correspondencia de entrenamiento admitían varias detecciones (79.2 %). Se contabilizaban 15048 instancias de enlaces positivos para solo 6835 enlaces anotados distintos. Al exigir correspondencia única diagnóstica, solo 4230 enlaces permanecían positivos. Tras E015, estos últimos aumentaron a 5247.

La asignación diagnóstica es una herramienta de auditoría; no reproduce exactamente el matcher oficial. Sus conteos no deben sustituir los TP oficiales. El resultado respalda que recompensar alternativas simultáneamente oculta errores de identidad y asociación. No demuestra que ese sea el único problema.

## Decisión

Conservar E015 como evidencia para rediseñar el entrenamiento. No enviarlo al leaderboard: no hay evidencia de que supere nuestro control Harmonic. Su score tampoco debe compararse directamente con el 0.909257 del control, calculado sobre otros 48 videos. El 0.946 público continúa siendo nuestra referencia de leaderboard.

El siguiente experimento debe entrenar sobre los 48 videos de fit existentes, reutilizando features, con estas modificaciones estructurales:

1. Representar alternativas próximas como hipótesis competidoras de una identidad; impedir que todas reciban simultáneamente crédito positivo. La asignación de etiquetas usa únicamente GT del conjunto de entrenamiento.
2. Aprender elección de padre con competencia entre candidatos y opción de no asociar; las detecciones sin anotación permanecen desconocidas, no negativas automáticas. No usar una correspondencia por cercanía como prueba de fondo.
3. Mantener separación por video entre fit y calibración. Medir score completo, enlaces sin ajuste, cobertura y multiplicidad, no seleccionar por AP de pares aislados.
4. Comparar también un control Harmonic en esos mismos 16 videos antes de considerar reemplazarlo. Mantener CPU para las cabezas sobre features; no repetir inferencia visual ni gastar GPU sin necesidad.

Este entrenamiento siguiente todavía no está implementado ni lanzado. No se programó monitoreo automático.

## Verificación

Se compararon los 20 archivos del paquete ejecutado con el payload congelado y se verificó su hash. Se comprobó el conjunto exacto de 16 videos, la versión fijada de la métrica, la congelación declarada previa a lectura de GT y la igualdad de los informes con las sumas recalculadas por video. Los nodos del diagnóstico coinciden con los de la evaluación oficial.

La métrica oficial se ejecutó en Kaggle; localmente se recalcularon los agregados de sus informes. No se descargaron ni recalcularon los CSV localmente. Recibo: `results/E015_completed.json`; auditoría reproducible: `scripts/audit_e015_completed.py`.
