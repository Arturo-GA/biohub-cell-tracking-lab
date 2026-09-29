# E068 — Traspaso operativo (estado al 26-09-2026 16:10 UTC)

> **Estado posterior, 29-09:** E068–E070 son etapas cerradas. La última tanda E071 terminó sus cinco submissions, con mejor público **0.957** y cuatro alternativas **0.956**. Consultar [E071_EXECUTION.es.md](E071_EXECUTION.es.md). Las prioridades, cuotas, fechas operativas y órdenes de envío de este documento son históricas. No hay lanzamientos pendientes y el reminder permanece apagado.

> **E069 completado, 27-09 a las 20:15 UTC:** ver [E069_RESULTS.es.md](E069_RESULTS.es.md). Se evaluaron seis configuraciones en 199 videos y once en 32 videos. Duplicados y reconexión no actuaron; el veto no mejoró; el linker reprodujo una corrección ya vista en CPU6, sin incremento en confirmación. No se promovió un candidato adicional. C3_fork8 v1 sigue siendo el envío prioritario listo para el reset. Los tres notebooks E069 están COMPLETE y descargados con manifiestos verificados. Se usó GPU de la laptop y Kaggle CPU; no repetir sus lanzamientos ni el intento GPU rechazado por capacidad.

> **Actualización 27-09, 18:57 UTC:** CPU7 y C3_fork8 completaron. Fork8 es el mejor de los nueve controles locales y su CSV v1 pasó todas las verificaciones; todavía no se envió porque hay cinco submissions del día. Próxima ventana: 27-09 a las 19:00 Lima. Continuar desde [E068_CPU7_RESULTS.es.md](E068_CPU7_RESULTS.es.md). No hay un envío automático programado.

> **Estado histórico, reemplazado el 27-09.** Los cinco kernels de la tabla ya se enviaron y completaron; no repetir esos comandos. Mejor público confirmado: c3, 0.955 (`56592151`). `submit5.sh` está desactivado. CPU6 terminó sin una mejora convincente; C3_fork8 y CPU7 se lanzaron después. Continuar desde [E068_CPU6_RESULTS.es.md](E068_CPU6_RESULTS.es.md) y sus recibos. Las afirmaciones siguientes sobre puesto, cuota GPU y requisitos de scoring son del traspaso original y no describen necesariamente el estado actual.

Documento para quien continúe la competencia (Codex u otro operador). Complementa `docs/E068_X138_XR.es.md`
(método y resultados). Todo lo de abajo se verificó a la hora indicada; nada aquí se ejecuta solo.

## 1. Situación

- Mejor público propio: **0.954** (`56539929`, `56540269`), puesto 177. Bloque de 0.954: puestos 171–253 (83 equipos). Plata ≈ puesto 196; bronce ≈ 392.
- Cierre: **29-09-2026 23:59 UTC**. Cada envío tarda ~5 h en puntuar (reejecución sobre ~199 videos ocultos) y **debe terminar antes del cierre**: último envío razonable el 29-09 a las 10:00 UTC. Las dos finales se marcan a mano antes de las 20:00 UTC del 29.
- Cupo: 5 envíos por día UTC (se renueva a las 00:00 UTC). El 26-09 quedaron consumidos. Máximo 2 sesiones GPU simultáneas en Kaggle. GPU semanal: ~30 h; consumidas esta semana ≈ 19 h a las 16:00 UTC del 26-09 (incluye labs 6–8).
- Credenciales: CLI de Kaggle en `C:\Users\Arturo\Diplomado\Scripts\kaggle` (usuario `jarturo`, token en `~/.kaggle`). GitHub vía `gh` (cuenta `Arturo-GA`).

## 2. Los cinco kernels listos para enviar (ninguno enviado todavía)

Todos son x138 en una sola pasada, `OUTPUT_MIN_TRACK_LEN=7`, `REPAIR_DEADLINE_S=34200`, ya ejecutados en Kaggle y verificados con `gate.py` sobre el test visible. El generador es `kaggle/x138_xr/build.py --xr` y las copias exactas de los notebooks generados están en `kaggle/x138_xr/kernels/`.

| Prioridad | Kernel (versión) | Reglas | Qué aísla |
|---|---|---|---|
| 1 | `jarturo/biohub-x138-s1-tau6b-e-cuts-fork` (v1) | tau 1.0 en 6bba + poda E (44b6 <8, 6bba <11, prob <0.7) + relink >8 µm + último enlace <0.5 + fork8-nanonly | todo |
| 2 | `jarturo/biohub-x138-s5-tau6b-p8-cuts` (**v2**) | igual con poda uniforme P8 (<8, prob <0.7) | poda uniforme |
| 3 | `jarturo/biohub-x138-s2-tau6b-e-cuts` (v1) | sin fork8-nanonly | sin bifurcaciones |
| 4 | `jarturo/biohub-x138-c6-tau10` (v1) | solo `SAFE_DIV_SISTER_SYMMETRY_TAU=1.0` global | palanca de divisiones |
| 5 | `jarturo/biohub-x138-c3-e-cuts` (v1) | poda E + cortes, tau 0.6 | reglas XR |

Comando de envío (uno por kernel, a partir de las 00:00 UTC del 27-09):

```bash
K=/c/Users/Arturo/Diplomado/Scripts/kaggle
$K competitions submit biohub-cell-tracking-during-development -k jarturo/biohub-x138-s1-tau6b-e-cuts-fork -v 1 -f submission.csv -m "S1 tau6bba1.0 + E + nan8 + end5last + fork8-nanonly"
$K competitions submit biohub-cell-tracking-during-development -k jarturo/biohub-x138-s5-tau6b-p8-cuts -v 2 -f submission.csv -m "S5 tau6bba1.0 + P8 + nan8 + end5last + fork8-nanonly"
$K competitions submit biohub-cell-tracking-during-development -k jarturo/biohub-x138-s2-tau6b-e-cuts -v 1 -f submission.csv -m "S2 tau6bba1.0 + E + nan8 + end5last"
$K competitions submit biohub-cell-tracking-during-development -k jarturo/biohub-x138-c6-tau10 -v 1 -f submission.csv -m "c6 probe tau1.0 global"
$K competitions submit biohub-cell-tracking-during-development -k jarturo/biohub-x138-c3-e-cuts -v 1 -f submission.csv -m "c3 probe E + nan8 + end5last"
```

`kaggle/x138_xr/submit5.sh` hace exactamente esto (espera a la fecha UTC, re-verifica el log y envía en orden). La instancia que estaba programada en la sesión anterior **se detuvo** al hacer el traspaso: hay que lanzar los envíos manualmente o con ese script.

Otros kernels verificados, no en la lista: `biohub-x138-c4-p8-cuts` (P8 + cortes, sin tau), `biohub-x138-c5-tau10-e-cuts` (tau 1.0 global + E + cortes), `biohub-x138-t4-cuts` (solo cortes), `biohub-x138-t3-fork9` (**no enviar**: regla negativa en 199 videos).

## 3. Experimento en curso

- `jarturo/biohub-safediv-lab-cpu3` **v3** (CPU) estaba RUNNING a las 16:04 UTC: replay de safe-division con la rejilla fina de tau sobre **todos los videos con rastreo** (labs 4–8, ≈199 videos), incluidas variantes por embrión (`6bba1.0|44b6 0.8/0.9/1.2`), `parent12`, `dc0.28` y combos con XR. Leer con:
  `kaggle kernels logs jarturo/biohub-safediv-lab-cpu3` (líneas con `score=`; formato: delta oficial frente a x138, IC 90 %, videos +/−, `forks_added`, `div TP/FP/FN`, desglose 44b6/6bba).
- Decisiones que dependen de ese resultado: si `44b6 0.9` o `1.0` (con 6bba 1.0) supera a `44b6 0.6` con IC positivo y ≥ 60 videos 44b6, aplicar tau también a 44b6 en la tanda del 28; elegir 0.9 vs 1.0 en 6bba por cota inferior; `parent12` solo si su IC es positivo en 199 videos.

## 4. Reglas que no hay que romper

1. **Una sola pasada.** Nunca validador, sweep ni segunda escritura dentro del kernel de envío (0.904 el 24-09). Parámetros fijos en la celda 0; `build.py` actualiza la guardia de la celda 1.
2. `REPAIR_DEADLINE_S=34200`. No añadir tiempo apreciable al kernel.
3. Verificar el log del test visible con `gate.py` antes de enviar (conteos por video, `config=base`, sin `DEADLINE`/`REPAIR FAILED`).
4. Título del kernel debe "slugificarse" al id (`Biohub prune lab5` → `biohub-prune-lab5`), si no el push falla en silencio dentro de bucles: revisar siempre la salida del push.
5. Coordenadas del CSV enteras (el escritor de x138 ya redondea).
6. En Windows, rutas cortas (MAX_PATH) y `git -c core.longpaths=true`.
7. No elegir umbrales al filo de un precipicio medido: DeepCenter 0.30 (0.35 destruye), `fork_min_branch` ≥ 9 sin `nanonly`, tau > 1.2.
8. No tunear con el público (±0.0005 de ruido); decidir con el laboratorio de 199 videos y usar el público solo como comprobación de signo y de que no hubo degradación (`total_bytes` del CSV oculto ≈ 208 MB × proporción de filas del visible).

## 5. Plan sugerido

- **27-09 00:00 UTC:** enviar los 5 de la tabla. Leer scores ~05:00 UTC junto con cpu3 v3.
- **27-09 (día):** construir la tanda del 28 con `build.py --xr` como vecinos de una perilla del mejor combo: `BIOHUB_XR_TAU_SPEC=44b6:0.6,6bba:0.9`; `44b6:0.9,6bba:1.0` si cpu3 v3 lo respalda; `BIOHUB_XR_CUT_END_MODE=both`; `BIOHUB_SAFE_DIV_MAX_UM=12` solo si confirmado. Ejecutar cada kernel (~20 min GPU), pasar `gate.py`, enviar a las 00:00 UTC del 28.
- **28-09:** última tanda de 5 antes de las 10:00 UTC del 29 si hace falta cerrar una comparación; si no, guardar.
- **Selección final:** A = combo con mejor cota inferior en el laboratorio de 199 videos y público ≥ 0.954; B = variante que difiera en una palanca (o L7 `56540269` si ningún combo alcanza 0.955 público). El x138 exacto (`56510963`) solo como último recurso: por fecha queda tarde dentro del bloque de empates.

## 6. Dónde está cada cosa

- Código: `kaggle/x138_xr/` (ver su README). Notebooks generados de los candidatos: `kaggle/x138_xr/kernels/`.
- Resultados: `results/E068/` (logs de todas las evaluaciones oficiales; `cpu_lab2_v6_199v_nanonly.txt` y `cpu_lab2_v5_199v_forktable.txt` son la referencia de 199 videos; `cpu_lab3_v2_tau_grid_72traced.txt` la de tau).
- Capturas en Kaggle (privadas, salidas de kernel): `jarturo/biohub-prune-lab0`…`lab8` (`lab/graphs/<stem>.npz`; labs 4–8 con `sdtrace`). Cabezas V1284 propias: `jarturo/biohub-v1284-capture-train` y `-train2` (descartadas: 0.952).
- Tabla de envíos con `total_bytes`: `results/E068/subs_table_2026-09-26.csv` (regenerar con `python kaggle/x138_xr/subs.py`).
