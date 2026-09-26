# x138 + reglas XR (E068)

Generador de kernels de envío y de laboratorio a partir del notebook público x138 (`upstream/biohub-x138.ipynb`).
Ver `docs/E068_X138_XR.es.md` para el contexto y los resultados.

## Archivos

| Archivo | Uso |
|---|---|
| `build.py` | Genera kernels: `--fixed` (x138 + overrides de la celda 0), `--xr` (x138 + módulo de reglas), `--lab` (capturas GPU, partes 0–8), `--cpulab2` / `--cpulab3` (evaluación CPU con métrica oficial). Sustituye anclas exactas del notebook y aborta si una ancla no aparece exactamente una vez. |
| `extra_rules.py` | Módulo compartido por kernels de envío y laboratorio. Variables `BIOHUB_XR_*`: `PRUNE` (`"44b6:8:0.7,6bba:11:0.7"` o `"*:8:0.7"`), `CUT_NAN_DIST_UM`, `CUT_END_PROB` (número o por prefijo), `CUT_END_MODE` (`first|last|both`), `CUT_END_ITER`, `FORK_MIN_BRANCH`, `FORK_NAN_ONLY`, `TAU_SPEC` / `DC_SPEC` (umbrales de safe-division por embrión). Todo apagado por defecto: la salida es la de x138. |
| `lab_capture.py`, `lab_eval.py`, `trace_safediv.py` | Celdas de los kernels de captura: interceptan el grafo previo al filtro de pistas cortas, el rastreo de candidatos de safe-division, y exportan `lab/graphs/<stem>.npz` más las reglas rápidas. |
| `cpu_lab.py`, `cpu_lab2.py`, `cpu_lab3.py` | Evaluación offline con `src/biohub_official`: etiquetas de enlaces/bifurcaciones, rejillas de reglas, replay de safe-division. |
| `gate.py` | Verificación del log del test visible antes de enviar (conteos por video, `config=base`, sin degradación). |
| `batch.sh`, `submit5.sh` | Orquestación: push con reintentos por el límite de 2 sesiones GPU, verificación y envío en orden al abrirse el cupo diario. |
| `subs.py`, `boot.py` | Tabla de envíos con `total_bytes` del CSV oculto; bootstrap por video de las filas del laboratorio. |
| `done72.txt` | Stems de los 72 primeros videos capturados (particiones de `--lab`). |

## Comandos usados

```bash
# reproducción exacta
python build.py                      # escribe k_repro (y capturas)
# ladder de largo mínimo
python build.py --fixed k_m7d biohub-x138-m7d "Biohub x138 m7d" BIOHUB_OUTPUT_MIN_TRACK_LEN=7 BIOHUB_REPAIR_DEADLINE_S=34200
# combo con reglas XR y tau por embrión
python build.py --xr k_s1 biohub-x138-s1-tau6b-e-cuts-fork "Biohub x138 s1 tau6b e cuts fork" \
  BIOHUB_OUTPUT_MIN_TRACK_LEN=7 BIOHUB_REPAIR_DEADLINE_S=34200 -- \
  BIOHUB_XR_PRUNE="44b6:8:0.7,6bba:11:0.7" BIOHUB_XR_CUT_NAN_DIST_UM=8 BIOHUB_XR_CUT_END_PROB=0.5 \
  BIOHUB_XR_CUT_END_MODE=last BIOHUB_XR_FORK_MIN_BRANCH=8 BIOHUB_XR_FORK_NAN_ONLY=1 BIOHUB_XR_TAU_SPEC=44b6:0.6,6bba:1.0
# laboratorio CPU sobre las capturas
python build.py --cpulab2 jarturo/biohub-prune-lab0,...,jarturo/biohub-prune-lab4
python build.py --cpulab3 jarturo/biohub-prune-lab4,jarturo/biohub-prune-lab5
```

Requisitos en Kaggle: los cuatro datasets públicos del x138 (soporte, segundo seed y DeepCenter de Pilkwang; cabeza V1284 de Anvith Pothula) y la competencia. Los kernels de laboratorio adjuntan además las salidas de las capturas como `kernel_sources`.

Nota de Windows: `build.py` y las rutas de trabajo deben ser cortas (límite MAX_PATH); el clon del repo se hizo con `core.longpaths=true`.
