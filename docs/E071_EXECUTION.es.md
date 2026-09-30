# E071 — última tanda de cinco candidatos

## Resultado privado revelado — 30-09-2026, 00:12 UTC

El deadline oficial pasó a las 23:59 UTC. La API autenticada de Kaggle ya devuelve private scores y `userRank=1097` entre 4017 equipos. No es un score preliminar del notebook ni un error visual: ambas submissions seleccionadas figuran COMPLETE con **0.917** y **0.916** privado. El rango todavía no se considera certificado mientras puedan terminar kernels tardíos o Kaggle realice revisiones.

| Seleccionada | Público | Privado |
|---|---:|---:|
| C3_D08_R094 (56656490) | 0.956 | **0.917** |
| C3_D04_R094 (56656420) | 0.957 | 0.916 |

La mejor private score de las 38 submissions fue **0.923**, submission 56515407: x138 con cinco cabezas V1284 propias, público 0.952. No estaba entre las dos seleccionadas y no puede sustituirse después del deadline. Este resultado muestra que la selección por Public eligió variantes sobreajustadas al 29% público: la mejora de 0.952 a 0.957 se convirtió en una pérdida privada de 0.007 frente a aquella alternativa. Recibo estructurado: `results/E071/final_private_result.json`.

## Cierre público — 29-09-2026, 22:56 UTC

Los cinco envíos están **COMPLETE** según la consulta oficial guardada en `results/E071/all_submission_status.json`. Mejor público: **C3_D04_R094, 0.957**, frente a c3 **0.955**. No hay resultados privados ni una medalla definitiva. Los estados de preparación y scoring de las secciones posteriores son históricos; no repetir lanzamientos ni envíos. El reminder continúa desactivado.

| Candidato v1 | Submission | Coste división ILP | Readmisión | Público |
|---|---:|---:|---:|---:|
| C3_D04_R094 | 56656420 | 0.4 | 0.94 | **0.957** |
| C3_R094 | 56656425 | 1.2 | 0.94 | 0.956 |
| C3_D04 | 56656488 | 0.4 | 0.965 | 0.956 |
| C3_D08_R094 | 56656490 | 0.8 | 0.94 | 0.956 |
| C3_D04_R092 | 56656491 | 0.4 | 0.92 | 0.956 |

Cada cambio principal por separado mejora una milésima mostrada; combinados mejoran dos. La combinación 0.4/0.94 supera las alternativas 0.8/0.94 y 0.4/0.92 a la precisión publicada. El redondeo impide afirmar aditividad exacta o significación estadística, y no hay desglose de errores del público para atribuir el incremento a un componente concreto. Las variantes comparten modelos y no son validaciones independientes.

Los cinco CSV son distintos, con paridad exacta de grafos laptop/Kaggle en los cuatro videos visibles y control c3 reproducido. Se preservan los notebooks enviados y recibos originales; esta actualización solo consulta estados y registra el cierre. Comparación final: `results/E071/final_public_analysis.json`. La captura de ranking en `private_risk_assessment/summary.json` es de las **06:47 UTC**, no del cierre: no tratar ese puesto como un ranking final.

## Historial operativo

Petición de Arturo del 28-09: preparar los últimos cinco envíos con cuidado después de que E070 no mejorara. Preparación autorizada de notebooks privados con GPU para inferencia. **No enviar automáticamente al leaderboard. El recordatorio sigue desactivado.**

## Resultados parciales — 29-09, 06:38 UTC

Consulta oficial: C3_D04_R094 (56656420) COMPLETE **0.957**; C3_R094 (56656425) COMPLETE **0.956**; C3_D08_R094 (56656490) COMPLETE **0.956**. C3_D04 y C3_D04_R092 siguen PENDING. El mejor confirmado pasa de c3 0.955 a 0.957. Comparación guardada en `results/E071/interim_public_analysis.json`.

La readmisión 0.94 por sí sola sube una milésima mostrada; añadir coste ILP 0.4 sobre esa readmisión suma otra. El coste intermedio 0.8 no supera 0.956 a la precisión mostrada. Falta D04 solo para separar su efecto independiente e interacción. No hay desglose público de errores que permita atribuir la mejora a una clase concreta: recuperación de detecciones y cambios del grafo son mecanismos plausibles, no mediciones del conjunto público. Tres variantes relacionadas tampoco son tres validaciones independientes.

Retrospectiva: E070 priorizó ganancias locales que se invirtieron en público. El detector preentrenado ya había visto los 199 videos; esa métrica local no era una validación independiente y recibió demasiado peso al ordenar experimentos. E071 conservó el mejor c3, descartó los cambios E070 con pérdidas públicas y probó dos parámetros aislados y combinados. La pista pública concreta se publicó el 27-09 a las 22:57 UTC (respuesta 3529308 del hilo 743929); llegó tarde, pero las pruebas sistemáticas de los parámetros principales y la revisión de la confianza en la validación debieron priorizarse antes. La paridad laptop/Kaggle confirma implementación, no explica por sí misma la ganancia predictiva.

Esta consulta no lanzó nuevos experimentos ni envíos, y no reactivó el reminder. Los dos scores pendientes deben tratarse como desconocidos, no como ceros ni como mejoras esperadas.

## Envíos finales autorizados y aceptados — 29-09, 00:11 UTC

Arturo autorizó explícitamente enviar los cinco candidatos al llegar las 19:00 Lima del 28-09. Se consultaron los cupos renovados (0 usados, 5 disponibles), historial sin E071 y código remoto de cada versión 1: las celdas coinciden exactamente con las copias congeladas y sus hashes de CSV/código permanecen válidos.

| Candidato v1 | Submission | Estado a las 00:11 UTC |
|---|---:|---|
| C3_D04_R094 | 56656420 | PENDING |
| C3_R094 | 56656425 | PENDING |
| C3_D04 | 56656488 | PENDING |
| C3_D08_R094 | 56656490 | PENDING |
| C3_D04_R092 | 56656491 | PENDING |

Kaggle aceptó los cinco entre las 19:09 y 19:11 Lima. Cuota posterior: 5 usados, 0 disponibles. No hay score público todavía. Los recibos `results/E071/*_submission_attempt.json` son definitivos y no deben repetirse los envíos. Cola y ledger están en FIVE_SUBMITTED. Autorización: `final_five_authorization.json`; comprobaciones: `final_preflight.json`; última consulta: `all_submission_status.json`. El reminder continúa apagado y no hay nuevos experimentos o envíos pendientes de ejecutar.

Para consultar los resultados cuando corresponda, sin reenviar:

    C:/dev/rsna-knee/.venv/Scripts/python.exe kaggle/x138_xr/e071_submit.py status-all

Nota operativa: después del segundo envío, el control de duplicados confundió el prefijo C3_D04 con C3_D04_R094 y se detuvo antes de crear el tercer intento. Se corrigió para comparar identificadores completos, se conciliaron los dos recibos con el historial y se enviaron únicamente los tres no intentados. No hubo duplicados ni cupos adicionales consumidos. La comprobación remota usa `currentVersionNumber`: el sufijo `/1` del helper `kernels_pull` devuelve 403 en esta versión del SDK, mientras la consulta por referencia normal confirma versión 1 y código exacto.

## Cierre — cinco listos (28-09, 22:25 UTC / 17:25 Lima)

El proceso finito terminó por sí mismo y retiró su lock. Los cinco notebooks v1 están COMPLETE, descargados y verificados. La consulta independiente de las 22:28 UTC confirmó los cinco estados. Se comprobaron nuevamente los hashes del código y CSV congelados: todos coinciden con sus recibos; los cuatro grafos de cada candidato reproducen exactamente su resultado local. No hay duplicados ni candidatos sin efecto frente a c3.

La cola definitiva está en `results/E071/submission_queue.json`, estado READY, con este orden:

1. C3_D04_R094: coste división 0.4 + readmisión 0.94; combinación principal de las pistas públicas.
2. C3_R094: readmisión 0.94, coste división original; aislar la recuperación de detecciones.
3. C3_D04: coste división 0.4, readmisión original; aislar el coste ILP.
4. C3_D08_R094: variante exploratoria con coste intermedio 0.8.
5. C3_D04_R092: variante exploratoria de readmisión más permisiva 0.92.

Los tiempos de ejecución fueron 17.9, 17.9, 17.6, 17.4 y 19.8 minutos. Son cinco candidatos listos para enviar, **no cinco nuevos scores públicos**. Ninguno fue enviado al leaderboard. No reiniciar `run-batch`, no reconstruir los notebooks y no reactivar el reminder. Antes de un envío posterior autorizado, revisar cuota viva, versiones, historial y recibos para evitar duplicados. El reset previsto es hoy 19:00 Lima / 29-09 00:00 UTC.

## Base y decisión

El c3 de la submission **56592151**, público **0.955**, era el mejor confirmado antes de E071. Se descargó el notebook remoto y se comprobó igualdad exacta de todas las celdas de código con la copia local. SHA256 local: `052f47600befa2afc0cac6892a1f14aa0381dc786b5c322b0dd51752070b598b`.

E070 terminó: fork8 **0.953**, DC030 **0.947**, prune12 **0.953**, prune12+joint **0.951**, joint **0.951**. Por tanto la nueva tanda no hereda ninguno de esos cambios. Cola, ledger y decisión E070 fueron actualizados con los resultados terminales.

La discusión [743929](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/743929), refrescada hoy, reporta ganancias al cambiar coste de división ILP 1.2→0.4 y readmisión 0.965→0.94. El 0.959 del autor está verificado en el leaderboard; no están verificadas independientemente sus ablaciones ni está publicada la etapa previa que lo llevó a 0.957. La receta es una hipótesis para c3, no un resultado garantizado.

## Selección de cinco

| Candidato | Coste división | Readmisión | Motivo |
|---|---:|---:|---|
| C3_D04_R094 | 0.4 | 0.94 | Combinar las dos pistas públicas sobre c3 |
| C3_R094 | 1.2 | 0.94 | Aislar recuperación localizada de detecciones |
| C3_D04 | 0.4 | 0.965 | Aislar el coste ILP, conservando readmisión original |
| C3_D08_R094 | 0.8 | 0.94 | Alternativa de menor intervención en el ILP |
| C3_D04_R092 | 0.4 | 0.92 | Alternativa exploratoria que recupera algo más |

La base conocida es el cuarto vértice del contraste de los primeros tres candidatos; no se gasta un envío en una copia de c3. Las últimas dos variantes exploran la sensibilidad cerca de la combinación publicada. Se conserva el umbral global del detector, radio de readmisión 4 µm, pesos, TTA, reparaciones y poda de c3. No se suman ganancias supuestas ni se ordenan estas variantes por métricas contaminadas de entrenamiento.

Los cinco están construidos por `build_e071.py`. Las celdas 2–11 son textualmente idénticas a c3. En la celda 0 solo cambian los parámetros declarados; la celda 1 agrega comprobaciones de configuración. La última celda comprueba parámetros resueltos y escribe un manifiesto. Cada CSV se genera una sola vez con configuración fija, sin validador ni selección en tiempo de envío.

## Ejecución y recursos

### Cambio autorizado: GPU de la laptop (28-09, 20:15 UTC)

Arturo pidió ahorrar sus pocas horas de Kaggle utilizando la GPU local. Se detuvo el proceso finito `run-batch`; no quedan lanzamientos automáticos pendientes y el reminder sigue PAUSED. **No ejecutar los comandos de lanzamiento secuencial descritos más abajo sin revisar este cambio.** El único kernel E071 lanzado, C3_D04_R094 v1, terminó COMPLETE: duración 1073 s, 233450 filas, validación de CSV/grafo/configuración PASS. No tiene score público; no fue enviado al leaderboard.

La laptop tiene RTX 3050 Laptop de 4 GB. Se creó `.venv-e071` separado, reutilizando torch CUDA de `.venv`, y se verificaron CUDA, tracksdata y SCIP. La prueba de capacidad del U-Net primario con pesos reales y datos sintéticos de tamaño visible consumió 544 MiB; no demuestra por sí sola que todo el pipeline sea idéntico. Se descargaron cachés pre-ILP del kernel completo, incluyendo todas las aristas admitidas y los picos de baja confianza. `e071_local_replay.py` terminó los doce cálculos ILP (tres costes por cuatro videos) en CPU. El coste 0.4 reproduce exactamente nodos, aristas y probabilidades de los cuatro grafos remotos (`local_ilp_parity.json`). DeepCenter con pesos exactos y TTA de ocho vistas pasó sobre un frame real en CUDA, con 201 MiB de pico. El control final debe coincidir en los cuatro grafos antes de aceptar las variantes locales.

Descarga: 408 archivos, 1.906 GB, únicamente cuatro videos de test visibles; comprobar `local_visible_complete.json`. El proceso finito local espera la descarga hasta 90 minutos, ejecuta el control y, solo si reproduce los cuatro grafos finales remotos, continúa con los otros cuatro candidatos y un c3 de control. Log: `outputs/e071/local_replay.log`. No es un reminder ni una tarea recurrente. Las conexiones del CDN se reabren con límite de tiempo y los archivos se reemplazan atómicamente tras completar su descarga.

Una ejecución local verificada **no es todavía un notebook Kaggle COMPLETE** ni permite predecir imágenes privadas. Se descarta combinar cinco salidas en un notebook de producción: repetir tres ILP y cinco posprocesados en cada evaluación privada aumentaría el riesgo de timeout en esta última tanda. Se reutiliza la inferencia para las pruebas locales, conservando la ruta de ejecución c3 para producción. No contar cinco READY hasta verificar todas las salidas de Kaggle.

### Control local terminado y registro final (21:15 UTC)

Los cinco candidatos terminaron en la laptop: CSV/grafos válidos, cinco resultados distintos y ninguno idéntico al c3. C3_D04_R094 reproduce exactamente los cuatro grafos de su versión Kaggle. El sexto recorrido, C3_CONTROL, reproduce exactamente los cuatro grafos del c3 público 0.955. Evidencia: `results/E071/local_candidate_comparison.json` y `outputs/e071/local/*/verified.json`. Son comprobaciones de reproducción e integridad, no nuevos scores ni estimaciones de ganancia.

Se comunicó a Arturo que los ensayos usan la laptop y que registrar los cuatro notebooks de producción requiere la pasada final en Kaggle. La cuota leída a las 21:07 UTC deja aproximadamente 10099 s (2 h 48 min). El lote final estima 70–80 minutos y exige conservar al menos una hora tras cada ejecución estimada. Se retiró el bloqueo de registro; no se añadieron entrenamientos ni candidatos nuevos.

`e071_ops.py run-batch` está ejecutándose como proceso finito, log `outputs/e071/final_registration.log`. C3_R094 v1 fue aceptado a las 21:15:36 UTC, kernel 136310403, RUNNING confirmado. Al terminar descarga y verifica manifiesto, CSV, grafo y paridad con su resultado local, y continúa con D04, D08_R094 y D04_R092 bajo el límite compartido de dos sesiones. Se detiene ante error, discrepancia o cuota insuficiente. No hace submissions al leaderboard; no reactivó el reminder. Ver recibos, ledger y log antes de iniciar otro proceso. Al completar los cinco `finalize_e071.py` congela la cola READY.

- Primer lanzamiento aceptado: `jarturo/biohub-e071-c3-d04-r094` v1, kernel 136304955, 20:05:33 UTC. RUNNING confirmado 20:07 UTC.
- Al inicio hay otro notebook GPU de la cuenta en QUEUED: `jarturo/gemma-4-paired-localization-policy-experiment`. No detenerlo ni editarlo. Contarlo en el máximo compartido de dos sesiones; mientras siga activo, lanzar solo un Biohub simultáneamente.
- Consultar `results/E071/ledger.json` y recibos `*_launch.json` antes de cualquier operación. Un recibo REQUEST_PENDING requiere conciliación; nunca repetir a ciegas.
- No reconstruir un notebook lanzado. Los siguientes usan slugs propios, privados, con internet desactivado y T4.
- Cuota diaria de submissions consultada 19:56 UTC: 5 consumidos, 0 disponibles. Reset final previsto 29-09 00:00 UTC / hoy 19:00 Lima.

Comandos, ejecutados desde `C:/dev/biohub`:

    C:/dev/rsna-knee/.venv/Scripts/python.exe kaggle/x138_xr/e071_ops.py check
    C:/dev/rsna-knee/.venv/Scripts/python.exe kaggle/x138_xr/e071_ops.py logs C3_D04_R094 --seconds 15
    C:/dev/rsna-knee/.venv/Scripts/python.exe kaggle/x138_xr/e071_ops.py download C3_D04_R094
    C:/dev/rsna-knee/.venv/Scripts/python.exe kaggle/x138_xr/e071_ops.py launch C3_R094
    C:/dev/biohub/.venv/Scripts/python.exe kaggle/x138_xr/finalize_e071.py

Las pruebas locales ya terminaron y el proceso finito encadena exclusivamente el registro final. `launch` comprueba ocupación, identidad, privacidad, hash, recibo local y cuota antes de subir; consultar el ledger antes de utilizarlo. Esta preparación no utiliza una automatización programada.

## Criterio para declarar READY

1. Los cinco kernels COMPLETE, versión y código congelados.
2. Manifiestos exactos, parámetros resueltos correctos y ausencia de fallos, degradación por plazo o fallback de readmisión.
3. CSV con los cuatro videos, coordenadas válidas, identificadores coherentes y grafo de linaje válido.
4. Conteos del log iguales a los del CSV; una sola escritura final con `config=base`.
5. Comparación de grafos por coordenadas y aristas, ignorando diferencias arbitrarias de numeración. No incluir duplicados ni copias sin efecto de c3.

`finalize_e071.py` guarda `visible_comparison.json` y solo crea `submission_queue.json` si los cinco pasan. Las diferencias de nodos por coordenadas también incluyen desplazamientos por suavizado: no interpretarlas automáticamente como detecciones verdaderas añadidas o perdidas. Distinción técnica no implica mejora de score ni independencia estadística.

Si una variante produce cambios excesivos o se duplica, reconsiderarla antes de congelar la cola. No llamarla lista solo porque se haya escrito el notebook. No reactivar el reminder sin autorización nueva.
