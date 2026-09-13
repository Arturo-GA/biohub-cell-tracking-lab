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

El diagnóstico está en [Kaggle](https://www.kaggle.com/code/jarturo/biohub-lab-official-metric-ab). El diagnóstico no se presenta como holdout: la pertenencia de esos videos al entrenamiento de los checkpoints públicos no está verificada. No selecciona hiperparámetros ni envía resultados al leaderboard.

Los notebooks contienen el código necesario y adjuntan tres datasets públicos de Pilkwang: soporte, segundo seed y DeepCenter. La ejecución requiere GPU de Kaggle; internet desactivado. Los hashes de los pesos y del código de soporte se verifican al arrancar. Los datos, los pesos y las credenciales no se guardan en Git.

## Desarrollo

Pruebas ligeras: Python 3.12, PyTorch, NumPy y nbformat. En PowerShell, desde la raíz:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m unittest discover -s tests -v
python scripts/build_notebooks.py
python -m kaggle kernels push -p kaggle/diagnostic
```

La inferencia completa instala las dependencias desde los wheels adjuntos en Kaggle. Para evaluar un CSV contra los GEFF de entrenamiento en un entorno con esas dependencias:

```text
python -m biohub_lab.evaluate predictions.csv --data-dir /path/to/train_subset --output metrics.json
```

La carpeta evaluada debe contener exactamente los datasets del CSV, con `.zarr` y `.geff`. La validación rechaza datasets ausentes, nodos fuera del volumen, IDs duplicados, enlaces sin nodos, saltos temporales y grados incompatibles con un linaje binario.

La métrica se conserva en `src/biohub_official`, commit oficial `075fc5f5a52d11077f9dc2b074644618f26939e2`. Se calcula sobre los enteros del **CSV final**, no sobre predicciones intermedias. Esto elimina una discrepancia observada en el notebook original.

Para recuperar el contexto en otra sesión, leer primero los tres documentos de `docs/` y `results/STATUS.json`. Los resultados ajenos son referencias, nunca mediciones propias.
