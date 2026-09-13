# Biohub Cell Tracking Lab

Repositorio privado de Arturo para [Biohub — Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development). Investigación y preparación inicial: **13 de septiembre de 2026**.

La meta es mejorar el tracking real con experimentos reproducibles. **Todavía no hay un score propio confirmado ni evidencia de alcanzar el podio.** El notebook aportado contiene un hack antiguo; el control actual procede de una referencia pública más reciente.

- [Investigación, papers y prioridades](docs/RESEARCH.es.md).
- [Auditoría del notebook y de la métrica](docs/AUDIT.es.md).
- [Experimentos y criterios de decisión](docs/EXPERIMENTS.es.md).
- [Procedencia y licencias](NOTICE.md).
- [Registro de ejecución](results/STATUS.json).

## Notebooks

| Carpeta | Uso | Salida |
|---|---|---|
| `kaggle/control` | Inferencia fija de Harmonic Fusion; sin su antiguo sweep de métrica proxy | `submission.csv` |
| `kaggle/division_guard` | Misma base + menor peso inverso ante posibles divisiones | `submission.csv` |
| `kaggle/diagnostic` | Control y candidato sobre una muestra fija de train | CSVs de ambos, métrica oficial y `run_receipt.json` |
| `kaggle/hoct_diagnostic` | Máscaras 3D + HOCT + matching exacto, reutilizando detecciones | Métrica oficial, morfología y probabilidades por video |
| `kaggle/hoct_test` | Mismo reemplazo completo del enlazador sobre todo el test | `submission.csv` |

El [diagnóstico inicial](https://www.kaggle.com/code/jarturo/biohub-lab-official-metric-ab) terminó: control **0.9666951**, Division Guard **0.9667049**. No recuperó ninguna arista anotada ni división adicional. Se cierra esa línea sin mejora demostrada. Se verificó que **los cuatro videos estuvieron en el entrenamiento del segundo detector**; estas cifras son in-sample, no validación independiente ni leaderboard. [Resultado completo](results/E001_completed.json).

La nueva línea reemplaza todas las asociaciones por HOCT y usa morfología medida en máscaras watershed 3D. No modifica únicamente parámetros del código público. [Implementación y diferencias respecto al paper](docs/HOCT_IMPLEMENTATION.es.md). [Ejecución en Kaggle](https://www.kaggle.com/code/jarturo/biohub-lab-hoct-morphology-diagnostic).

**E003 terminó con resultado negativo:** HOCT **0.9195083**, control **0.9666951**, delta **−0.0471868**. Las divisiones falsas evaluadas aumentaron de 1 a 12. No se promueve esta versión a submission ni se interpreta el score como validación independiente. Se procesaron los cuatro videos en 12,06 minutos, excluyendo instalación/cola. [Registro completo](results/E003_completed.json). El notebook `hoct_test` se conserva como implementación reproducible, **no como candidato recomendado**.

Los notebooks contienen el código necesario y adjuntan tres datasets públicos de Pilkwang: soporte, segundo seed y DeepCenter. La ejecución requiere GPU de Kaggle; internet desactivado. Los hashes de los pesos y del código de soporte se verifican al arrancar. Los datos, los pesos y las credenciales no se guardan en Git.

## Desarrollo

Pruebas: Python 3.12, PyTorch, NumPy, SciPy, scikit-image y nbformat. En PowerShell, desde la raíz:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m unittest discover -s tests -v
python scripts/build_notebooks.py
python scripts/build_hoct_notebooks.py
python -m kaggle kernels push -p kaggle/hoct_diagnostic
```

La inferencia completa instala las dependencias desde los wheels adjuntos en Kaggle. Para evaluar un CSV contra los GEFF de entrenamiento en un entorno con esas dependencias:

```text
python -m biohub_lab.evaluate predictions.csv --data-dir /path/to/train_subset --output metrics.json
```

La carpeta evaluada debe contener exactamente los datasets del CSV, con `.zarr` y `.geff`. La validación rechaza datasets ausentes, nodos fuera del volumen, IDs duplicados, enlaces sin nodos, saltos temporales y grados incompatibles con un linaje binario.

La métrica se conserva en `src/biohub_official`, commit oficial `075fc5f5a52d11077f9dc2b074644618f26939e2`. Se calcula sobre los enteros del **CSV final**, no sobre predicciones intermedias. Esto elimina una discrepancia observada en el notebook original.

Para recuperar el contexto en otra sesión, leer primero los tres documentos de `docs/` y `results/STATUS.json`. Los resultados ajenos son referencias, nunca mediciones propias.
