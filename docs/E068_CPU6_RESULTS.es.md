# CPU6 terminado y siguiente comparación — 27-09-2026

CPU6 v1 (`jarturo/biohub-c3-context-cpu6`) terminó correctamente: 199 videos por configuración, cero discrepancias al reproducir las divisiones, 20 comprobaciones de suavizado/redondeo sin diferencias y 7086 segundos de ejecución. Se descargaron resultados, métricas por video, diagnósticos de grafo y log. No consumió GPU ni submissions.

## Resultado frente a c3

| Variante | Score local | Delta local | Decisión |
|---|---:|---:|---|
| C3_public0955 | 0.92751338 | 0 | Control |
| Sin suavizado en madres e hijas inmediatas | 0.92757703 | +0.00006365 | No promover por ahora |
| Intercambios temporales estrictos | 0.92753448 | +0.00002110 | No promover por ahora |
| Intercambios temporales moderados | 0.92751345 | +0.00000006 | Descartar |
| Protección de enlaces largos | 0.92751338 | 0 | No cambió ningún grafo |
| Suavizado 0.3 en divisiones | 0.92749229 | −0.00002109 | Descartar |
| Protección de extremos | 0.92742328 | −0.00009011 | Descartar |
| Protección de extremos y enlaces largos | 0.92742328 | −0.00009011 | Repite protección de extremos |
| Intercambios y protección | 0.92744437 | −0.00006901 | Descartar |
| Protección y suavizado 0.3 | 0.92740218 | −0.00011120 | Descartar |

Los intercambios estrictos sí modificaron 37 grafos (48 pares), pero toda la ganancia evaluada se concentra en un video. La protección de extremos conservó 5433 enlaces y modificó 197 grafos: 14 videos mejoraron y 183 empeoraron. La variante sin suavizado redujo dos divisiones falsas, pero su pequeña ganancia global desaparece al excluir los cinco videos de mayor contribución. La agregación global incluye una fracción de divisiones, por lo que no equivale al promedio simple de scores por video.

Los nueve notebooks derivados de CPU6 siguen preparados localmente; **no se subieron ni se ejecutaron en GPU**. La señal obtenida no justifica gastar envíos en ellos ahora.

## Candidato con evidencia anterior aún no enviado

**C3_fork8:** c3 (público 0.955) más eliminación de la rama corta de bifurcaciones cuando esa rama viene de una reparación sin probabilidad ILP. Tau sigue en 0.6 para todos los grupos. No se relaja la generación de divisiones, a diferencia de S1/S2/S5.

Es la configuración `F_defensive_E` de CPU4. Al comparar sus métricas con el c3 reproducido por CPU6:

- Score local: **0.92839584**, delta **+0.00088245**.
- Divisiones verdaderas/falsas: **24/49**, frente a **24/62** de c3; mismas 127 divisiones omitidas.
- Delta por grupo: +0.00062829 en 44b6 y +0.00092895 en 6bba.
- Al retirar los cinco videos que más aportan: +0.00050511.
- Intervalo bootstrap descriptivo al 90%: [+0.00048266, +0.00136621]. Los videos están reutilizados; no es probabilidad de mejora pública ni de medalla.

Esta comparación cruza dos ejecuciones con las mismas capturas y el mismo replay. CPU7 repetirá ambos controles juntos. No se suma el delta local al score público.

Se lanzó privado **`jarturo/biohub-c3-fork8` v1**, kernelId 136127017, el 27-09 a las 14:53:01 UTC. Usa GPU T4 para inferencia. Recibo: `results/E068/c3_fork8_push_receipt.json`. Código generado por `build_c3.py`, parámetros en `c3_variants.py`, manifiesto dentro del notebook. Queda pendiente completar la ejecución y verificar CSV/log/manifiesto antes de enviar.

Última consulta, **14:57 UTC**: C3_fork8 **QUEUED**, CPU7 **RUNNING**. Lanzamiento aceptado no significa archivo final verificado. Registro: `results/E068/c3_cpu7_execution_status.json`.

## CPU7: nueve configuraciones, sin GPU

Se lanzó **`jarturo/biohub-c3-pruning-cpu7` v1**, kernelId 136127019, a las 14:53:04 UTC:

1. C3 público 0.955, control local.
2. C3_fork8, repetición de la hipótesis principal.
3. Fork8 con protección de divisiones cerca del final del video: no juzgar una rama corta si faltan fotogramas para observar ocho nodos.
4. Fork6, para medir sensibilidad al mínimo de rama.
5. Poda E con mínimo 12 en 6bba.
6. Poda E con mínimo 14 en 6bba.
7. Corte del último enlace con probabilidad <0.6.
8. Corte <0.6 más fork8, para medir si se complementan.
9. Poda uniforme con mínimo 10.

Todos mantienen tau 0.6, mismos pesos y una sola secuencia de procesamiento. Reutilizan capturas existentes, exigen los 199 videos y recalculan suavizado y redondeo. Se comprueba el score exacto del control contra CPU6 y la paridad con la función original en cada configuración. El estado opcional de protección del borde se reinicia entre configuraciones. No se seleccionarán nueve envíos por tener nueve pruebas.

Código local: 13 tests aprobados, incluida la protección temporal de ramas, su estado desactivado, preservación de aristas ILP y caso vacío. Sintaxis de ambos notebooks comprobada. `verify_run.py` comprueba manifiesto/código congelado, estructura y límites físicos del CSV, conteos exactos y finalización sin degradación.

## Próxima acción al terminar

Descargar CPU7 y analizar con `analyze_cpu4.py <carpeta cpu7> --base C3_public0955`. Priorizar la mejora reproducible sobre c3 y evitar configuraciones con salidas o errores equivalentes. Si CPU7 no aporta, C3_fork8 sigue siendo una prueba pública pendiente; A o B pueden ocupar un único envío exploratorio, no ambos automáticamente.

Los cinco envíos del día UTC 27-09 ya están consumidos. La siguiente renovación es **28-09 00:00 UTC = 27-09 19:00 Lima**. No se creó un proceso automático de envío y no se envió ninguna submission nueva en este turno. Conservar el mejor c3 confirmado, submission **56592151, 0.955**.

Artefactos archivados: `results/E068/cpu6_results.json`, `cpu6_rows.csv`, `cpu6_robustness.json`, `cpu6_graph_diagnostics.json`, `c3_fork8_crosslab_robustness.json` y ambos recibos de lanzamiento.
