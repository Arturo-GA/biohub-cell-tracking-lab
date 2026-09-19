# E018 verificado y siguiente hipótesis E019

19 de septiembre de 2026. El usuario autorizó seguir automáticamente ejecuciones cortas y continuar con el siguiente paso justificado. Esto sustituye la preferencia anterior de esperar su aviso para cada notebook corto; no crea una automatización recurrente ni autoriza submissions automáticas.

## E018: evidencia de complementariedad

El diagnóstico terminó en 36.39 segundos CPU y reprodujo los conteos oficiales. De 7175 enlaces anotados, ambos sistemas recuperan 6301, solo Harmonic recupera 373, solo E017 recupera 169 y ninguno recupera 332. De los 169 exclusivos de E017, 62 tienen ambos extremos emparejados en Harmonic y 107 tienen al menos uno sin correspondencia en Harmonic.

Esto justifica estudiar segmentos ausentes, pero no una unión indiscriminada. Las características simples no separan limpiamente los aciertos exclusivos de los falsos positivos: por ejemplo, 82 de los 169 aciertos exclusivos tienen contexto temporal único a ambos lados, igual que 361 de los 625 falsos positivos. Los histogramas son marginales y no permiten reconstruir combinaciones ni medir la precisión de un selector.

La unión ideal de 6843 enlaces correctos depende de GT y no es una predicción posible ni un score. La auditoría `scripts/audit_e018_completed.py` verificó los 25 archivos ejecutados, la contabilidad por video y cada familia de histogramas. Recibo: `results/E018_completed.json`.

## E019: recuperación de segmentos anclados

Se conserva el grafo completo de Harmonic: coordenadas, nodos, enlaces y padres de divisiones. Se dividen las trayectorias sin ramificaciones de E017 en segmentos delimitados por coincidencias espaciales con Harmonic.

- Anclas: vecinos mutuamente más cercanos por frame, hasta 1 µm, correspondencia uno a uno.
- Detecciones de E017 sin ancla y a 3 µm o menos de una detección Harmonic quedan bloqueadas para evitar duplicados. También se controla la distancia a nuevos nodos aceptados.
- Se proponen puentes entre dos anclas y extensiones con una sola ancla. Las extensiones necesitan al menos tres enlaces; no se incorporan componentes totalmente aislados de Harmonic.
- Se acepta un segmento únicamente cuando los extremos ocupados no tienen un enlace que habría que reemplazar. Las divisiones existentes no reciben ramas adicionales ni pierden ramas.
- Primero se consideran puentes, luego segmentos más largos, con desempate por IDs. Es selección determinista de segmentos compatibles; no optimización global ni regla entrenada.

Los parámetros son supuestos iniciales fijados, no resultados de un barrido. La hipótesis fue motivada por resultados de calibración: su evaluación en esos mismos 16 videos es exploratoria, no validación independiente. No se usa GT para decidir qué segmento añadir.

El CSV completo se congela antes de evaluar con la métrica oficial. Conservar enlaces no garantiza conservar TP ni score: nodos añadidos pueden cambiar el emparejamiento, elevar la penalización de conteo o introducir enlaces erróneos. Esto se medirá explícitamente.

Cuatro pruebas locales pasaron: puente con células ausentes, preservación de divisiones, rechazo de extensión demasiado corta y rechazo de duplicado próximo. El notebook se ejecuta en CPU con predicciones existentes, sin imágenes nuevas, entrenamiento o submission. Su finalización se revisará dentro de esta tarea por la autorización actual.

## Resultado E019: descartado

Se siguió la ejecución corta hasta finalizar y se descargaron los informes sin pedir otro aviso al usuario. El proceso del runner duró 69.39 segundos, más el tiempo de preparación/publicación de Kaggle.

| Métrica | Harmonic | E019 |
| --- | ---: | ---: |
| Score local | 0.900753 | 0.899805 |
| Enlaces TP | 6674 | 6678 |
| Enlaces FP | 485 | 493 |
| Enlaces FN | 501 | 497 |
| Nodos activos | 318094 | 319590 |
| Divisiones TP / FP / FN | 3 / 4 / 3 | 3 / 4 / 3 |

Se añadieron 1496 nodos y 1512 enlaces conservando la referencia. El saldo fue de cuatro TP adicionales y ocho FP adicionales, con mayor penalización por nodos. Dos videos mejoraron y catorce empeoraron. Algunos videos incluso perdieron TP pese a conservar las aristas originales, coherente con que añadir nodos puede cambiar la correspondencia oficial.

Se descarta E019 y se conserva Harmonic. No se hará un barrido de radios/longitudes ni una selección retrospectiva por video. La complementariedad ideal detectada por E018 no se convirtió en una ganancia práctica con esta regla de recuperación. Una representación que incorpore evidencia visual adicional requerirá un diseño y validación propios; todavía no se ha implementado ni lanzado.

Auditoría: `scripts/audit_e019_completed.py` verificó los 25 archivos ejecutados, preservación declarada y comprobada por el runner, congelación antes de evaluación, cohortes, conteos y agregados. Recibo: `results/E019_completed.json`. La métrica se ejecutó en Kaggle; localmente se verificaron informes, no se descargó ni reejecutó el CSV. No hay otro experimento pendiente de esta secuencia ni submissions realizadas.
