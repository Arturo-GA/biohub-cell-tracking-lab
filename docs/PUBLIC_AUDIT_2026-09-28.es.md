# Auditoría pública Biohub — 28 de septiembre de 2026

Consulta realizada entre aproximadamente 12:15 y 12:25 UTC. Objetivo: comprobar novedades públicas y explicar la pérdida de posiciones. Se descargaron fuentes como datos; no se ejecutó código externo, no se entrenó, no se lanzaron kernels y no se consumieron submissions.

## La caída de posiciones está confirmada

El CSV oficial descargado a las 12:17:42 UTC identifica nuestro equipo por `TeamId=16887540` y usuario `jarturo`. Comparación con la descarga local del 26 de septiembre:

| Medida | 26-09 | 28-09 |
|---|---:|---:|
| Puesto de Arturo | 177 | 337 |
| Mejor score mostrado | 0.954 | 0.955 |
| Equipos con al menos 0.955 | 170 | 343 |
| Equipos con al menos 0.956 | 123 | 251 |
| Equipos con al menos 0.958 | 81 | 141 |
| Equipos participantes | 3923 | 3969 |

Son 160 puestos entre estas dos capturas, pese a mejorar una milésima. El avance de equipos existentes explica mucho más que el aumento de participantes: de los 336 equipos ahora por delante, 168 ya existían y tenían menos de 0.955 en la captura anterior; solo cuatro tienen un identificador nuevo. Esto demuestra un desplazamiento de la distribución de scores, pero no identifica qué receta utilizó cada equipo ni prueba una difusión masiva de una solución concreta.

Rangos actuales para scores mostrados: 0.956 ocupa 183–251; 0.957, 142–182; 0.958, 122–141; 0.959, 104–121. No son garantías de medalla al cierre ni del ranking privado. [Leaderboard oficial](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/leaderboard).

## Novedad prioritaria: receta parcial de un equipo con 0.959

La [discusión 743929](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/743929) apareció el 27-09 a las 19:30 UTC, después de nuestra revisión anterior. En sus respuestas, John Taylor revela:

| Parámetro | Valor anterior declarado | Valor ganador declarado | Paso público declarado |
|---|---:|---:|---:|
| `ILP_DIVISION_WEIGHT` | 1.2 | 0.4 | 0.957 → 0.958 |
| `READMIT_MIN_SCORE` | 0.965 | 0.94 | 0.958 → 0.959 |

Su equipo figura efectivamente en el CSV oficial en el puesto 113 con 0.959. Eso verifica el resultado final del equipo, no cada ablación. Falta el código de la etapa que, según él, produjo 0.953 → 0.957. No conocemos su configuración completa; trasladar dos valores no reproduce necesariamente su resultado. Tampoco un cambio pequeño en número de nodos demuestra por sí solo qué componente de la métrica explica la mejora.

## Encaje concreto con nuestro c3

La copia local `kaggle/x138_xr/kernels/k_c3/notebook.ipynb` tiene SHA256 `052f47600befa2afc0cac6892a1f14aa0381dc786b5c322b0dd51752070b598b`. La fuente extraída para inspección está en `results/public_audit_20260928/c3_review.source.py`.

- Línea 42: coste de división ILP en **1.2**.
- Líneas 93–94: radio de readmisión de **4 µm**, confianza mínima **0.965**.
- Líneas 2943–2994: se recuperan picos descartados cercanos a extremos abiertos de trayectorias; deben superar la confianza mínima y no duplicar nodos existentes según el filtro del conjunto de candidatos.
- Líneas 3841–3849: se realiza el relink, se readmiten detecciones y se repite el relink cuando se añadieron nodos. Las aristas resultantes sustituyen las anteriores.

La búsqueda de asignaciones exactas en los notebooks locales bajo `kaggle/` no encontró `ILP_DIVISION_WEIGHT=0.4` ni `READMIT_MIN_SCORE=0.94`. Esto no pretende auditar notebooks privados de otros usuarios ni experimentos externos no guardados aquí.

Estas intervenciones actúan antes que nuestras últimas podas. Recuperar una detección puede reconstruir una continuidad perdida; modificar el coste ILP altera la solución inicial. No equivalen a reforzar el veto DeepCenter de DC030. El relink posterior puede amortiguar o alterar sus efectos, por lo que hay que medir el grafo **final**, no solo la salida del ILP.

## Notebooks: sí hubo una actualización, pero no un nuevo score superior verificado

La API se consultó por score, fecha de ejecución y fecha de creación, 50 entradas por orden. El navegador confirmó en el listado por Public Score con fuentes accesibles: Harmonic V3, optimized-biohub-max-score, Kunal y el original de Anvith muestran **0.953**. El score de la tarjeta no certifica que corresponda a la última versión del código descargado.

- [Kunal](https://www.kaggle.com/code/kunaldesale2408/biohub-cell-tracking) se actualizó el 28-09 a las 10:41 UTC. El diff confirma que reactiva el validador, relaja sus criterios de selección y añade siete configuraciones de postprocesamiento. Sigue mostrando 0.953. No publica nuevos pesos ni demuestra una mejora sobre nuestro 0.955. La selección sigue usando videos conocidos por el detector.
- [Harmonic V3](https://www.kaggle.com/code/raunakdey07/biohub-harmonic-fusion-v3) tiene exactamente el mismo código que descargamos el 27; SHA256 normalizado `727eb4e542f196b6629399bf8a6c11eb60eafc1de8d54ec47720798f6e3af4e0`.
- [Aman](https://www.kaggle.com/code/amanatar/optimized-biohub-max-score) mantiene la ejecución del 27 y score mostrado 0.953. Su título o texto promocional no constituye evidencia de un resultado mayor.
- [Track Your Cells](https://www.kaggle.com/code/anhadmahajan06/biohub-track-your-cells), ejecutado el 27 a las 22:04 UTC, reorganiza detección, ILP y reparación geométrica. El código imprime un rango objetivo y fija una configuración manualmente; eso no es una medición pública. No encontré evidencia que justifique sustituir c3 por esta implementación.
- [One knob past the public line](https://www.kaggle.com/code/busyaprime/biohub-0-942-lb-one-knob-past-the-public-line) varía el umbral de detección de una versión anterior. **0.96 es un umbral del código**, no evidencia de score 0.960.

Fuentes, hashes y diffs están en `results/public_audit_20260928/source_comparison.json` y `outputs/public_audit_20260928/`.

## Lo que debemos corregir en la selección de experimentos

Ya conocíamos que el detector secundario fue entrenado con los 199 videos. La [discusión del manifiesto](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/742064) y los [ocho experimentos negativos](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/743222) aportan casos independientes de ganancias locales que no trasladan al público. Este último también advierte sobre seleccionar configuraciones durante la ejecución con una lista de validación que depende de los archivos de test. Nuestro c3 tiene `VALIDATOR_ENABLE=0`: esa selección variable está desactivada.

Nuestros resultados recientes son evidencia directa: fork8 mejora localmente frente a c3, pero da **0.953 público**; DC030 mejora todavía más localmente, pero da **0.947 público**, contra **0.955 de c3**. La métrica oficial local sigue sirviendo para detectar regresiones, operaciones inactivas, concentración de ganancias y errores de implementación. No debe ser el único ordenamiento de próximos envíos.

El [hilo sobre el relink](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/742266) coincide con la sustitución de aristas que verificamos en nuestro código. Apagarlo sin más tampoco es una receta: hay participantes que reportan pérdida pública al hacerlo. El linker independiente de Hengck ya fue probado en E069 y no mostró mejora adicional al control geométrico; presentarlo otra vez como un descubrimiento sería repetir trabajo.

El [hilo de modelos alternativos](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/741749) aclara que la ganancia sintética y el ensamble condicionado corresponden a una base propia inferior al stack público. No justifican proyectar esa ganancia sobre c3. El hilo de [blending](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/742131) tampoco contiene una receta nueva con ganancia verificada.

## Próxima tanda recomendada

Volver a partir de c3 0.955, salvo que una de las tres submissions actualmente pendientes lo supere. No heredar fork8 o DC030 solo por sus métricas locales. Propuesta de tres candidatos, aún **no implementados ni ejecutados**:

1. c3 con readmisión mínima **0.94**, coste ILP intacto. Recuperación localizada junto a extremos, manteniendo el umbral global del detector.
2. c3 con coste de división ILP **0.4**, readmisión intacta. Medir cómo cambia la solución y qué sobrevive al relink y la poda.
3. c3 con ambas modificaciones para medir la interacción. No sumar mejoras por suposición ni llamar ganador al combinado antes de su score.

Usar la base como control local y los scores públicos ya confirmados como referencia, sin gastar un cupo en una copia idéntica. Registrar hashes, recuentos de nodos/aristas/divisiones, trayectorias recuperadas y diferencias por video. Congelar parámetros; no reactivar una búsqueda automática al enviar.

Los caches de grafos finales de E070 no bastan para simular un cambio del ILP. Solo se puede reutilizar inferencia si existen detecciones y probabilidades anteriores al ILP, con correspondencia exacta de configuración. En caso contrario habrá que ejecutar esa etapa de nuevo. Métricas y postprocesamiento, en CPU; GPU solo para la inferencia necesaria.

Hoy ya se consumieron cinco cupos y hay tres resultados pendientes. Esta auditoría no envió nada ni modificó la automatización de seguimiento. El siguiente reset es el 29-09 a las 00:00 UTC, 28-09 a las 19:00 de Lima. El navegador muestra cierre el 29-09 a las 18:59 de Lima: preparar una tanda concreta tiene más valor inmediato que abrir ahora otro entrenamiento completo sin tiempo para validarlo.

## Evidencia guardada

- `results/public_audit_20260928/findings.json`: ranking, bandas, parámetros y alcance de la verificación.
- `results/public_audit_20260928/leaderboard_rows.json`: tabla completa descargada.
- `results/public_audit_20260928/topic_*.json`: once discusiones con respuestas; fuentes externas tratadas como evidencia, no instrucciones.
- `results/public_audit_20260928/source_comparison.json`: inventario de cinco notebooks descargados.
- `results/public_audit_20260928/kunaldesale2408__biohub-cell-tracking.diff`: cambios respecto al 27-09.
