# Biohub Cell Tracking Lab

Repositorio privado de Arturo para [Biohub — Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development). Investigación y preparación inicial: **13 de septiembre de 2026**.

La meta es mejorar el tracking real con experimentos reproducibles. **Nuestro control obtuvo 0.946 en el leaderboard público**, frente al **0.947 reportado por Arturo para Harmonic Fusion completo**. Todavía no se ha superado esa referencia ni hay evidencia de alcanzar el podio. El notebook inicial de metric hack obtuvo 0.885; el control actual procede de Harmonic Fusion.

La inferencia de los cuatro videos visibles terminó en 26,14 minutos y se envió a Kaggle: **submission 56214656**, [Harmonic Control versión 1](https://www.kaggle.com/code/jarturo/biohub-lab-harmonic-control), script version `349618616`. Kaggle confirmó estado **COMPLETE** y score público **0.946**, verificados mediante su API autenticada. El CSV visible final se validó y su hash se verificó tras descargarlo. Es la referencia pública congelada, no una mejora atribuida a HOCT. [Recibo](results/E000_test_completed.json). La configuración final del notebook aportado por Arturo difiere del control; la comparación se documenta en [Experimentos](docs/EXPERIMENTS.es.md).

- [Investigación, papers y prioridades](docs/RESEARCH.es.md).
- [Auditoría del notebook y de la métrica](docs/AUDIT.es.md).
- [Experimentos y criterios de decisión](docs/EXPERIMENTS.es.md).
- [Procedencia y licencias](NOTICE.md).
- [Registro de ejecución](results/STATUS.json).
- [Modelo temporal propio y evaluación por embrión](docs/TEMPORAL_IMPLEMENTATION.es.md).

## Notebooks

| Carpeta | Uso | Salida |
|---|---|---|
| `kaggle/control` | Inferencia fija de Harmonic Fusion; sin su antiguo sweep de métrica proxy | `submission.csv` |
| `kaggle/division_guard` | Misma base + menor peso inverso ante posibles divisiones | `submission.csv` |
| `kaggle/diagnostic` | Control y candidato sobre una muestra fija de train | CSVs de ambos, métrica oficial y `run_receipt.json` |
| `kaggle/hoct_diagnostic` | Máscaras 3D + HOCT + matching exacto, reutilizando detecciones | Métrica oficial, morfología y probabilidades por video |
| `kaggle/hoct_test` | Mismo reemplazo completo del enlazador sobre todo el test | `submission.csv` |
| `kaggle/temporal_audit` | Inventario de anotaciones y procedencia de splits | Auditoría JSON |
| `kaggle/temporal_prepare` | Cinco frames por célula y distractores DoG sobre los 199 videos | Recortes y ejemplos reutilizables |
| `kaggle/temporal_train` | Dos modelos propios, cada uno excluyendo un embrión; después comparación con Harmonic | Pesos, splits y evaluaciones; sin submission |
| `kaggle/temporal_test` | Ensemble de los dos modelos temporales sobre todo el test | `submission.csv` y recibo de inferencia |
| `kaggle/temporal_dense` | E006: linajes densos sintéticos, preentrenamiento y adaptación por embrión | Pesos, manifiestos y comparación automática con E004 y Harmonic |
| `kaggle/temporal_specialist` | E007: nueva red temporal de mitosis con E006 congelado y reparación aditiva de Harmonic | Pesos, umbrales de desarrollo, CSV diagnóstico y métrica oficial |
| `kaggle/detector_cellect` | E008: detector temporal CELLECT a resolución nativa + Harmonic completo | Propuestas, auditoría de inyección, CSV y métrica oficial |
| `kaggle/detector_gaussian` | E009: separación de núcleos mediante ajuste de dos gaussianas + Harmonic completo | Propuestas, auditoría de inyección, CSV y métrica oficial |

El [diagnóstico inicial](https://www.kaggle.com/code/jarturo/biohub-lab-official-metric-ab) terminó: control **0.9666951**, Division Guard **0.9667049**. No recuperó ninguna arista anotada ni división adicional. Se cierra esa línea sin mejora demostrada. Se verificó que **los cuatro videos estuvieron en el entrenamiento del segundo detector**; estas cifras son in-sample, no validación independiente ni leaderboard. [Resultado completo](results/E001_completed.json).

E003 reemplazó todas las asociaciones por HOCT y usó morfología medida en máscaras watershed 3D. [Implementación y diferencias respecto al paper](docs/HOCT_IMPLEMENTATION.es.md). [Ejecución en Kaggle](https://www.kaggle.com/code/jarturo/biohub-lab-hoct-morphology-diagnostic).

**E003 terminó con resultado negativo:** HOCT **0.9195083**, control **0.9666951**, delta **−0.0471868**. Las divisiones falsas evaluadas aumentaron de 1 a 12. No se promueve esta versión a submission ni se interpreta el score como validación independiente. Se procesaron los cuatro videos en 12,06 minutos, excluyendo instalación/cola. [Registro completo](results/E003_completed.json). El notebook `hoct_test` se conserva como implementación reproducible, **no como candidato recomendado**.

Los notebooks contienen el código necesario y adjuntan tres datasets públicos de Pilkwang: soporte, segundo seed y DeepCenter. La inferencia Harmonic requiere GPU; la auditoría y preparación temporal usan CPU. El entrenamiento temporal admite GPU o CPU y esta ejecución completa se realizó en GPU. Internet está desactivado. Los hashes de los pesos públicos y del código de soporte se verifican al arrancar. Los datos, los pesos y las credenciales no se guardan en Git.

**E004 también se completó:** se implementaron y entrenaron dos redes propias de imágenes de cinco frames, atención entre padres candidatos y una cabeza de divisiones. Cada una excluyó un embrión completo. En la comparación condicional con detecciones Harmonic, obtuvo **0.9412437 frente a 0.9666951**; las divisiones evaluadas pasaron de 2 aciertos a 0. El resultado no es leaderboard y el detector público de esa comparación vio esos videos. En la evaluación separada del modelo temporal sobre los 199 videos tampoco se superó al vecino más cercano al elegir padres. **No se envió este candidato.** [Resultados y límites](docs/TEMPORAL_IMPLEMENTATION.es.md), [recibo completo](results/E004_completed.json), [entrenamiento Kaggle](https://www.kaggle.com/code/jarturo/biohub-lab-temporal-train).

**E005 terminó en el leaderboard con 0.913**, frente a **0.946** del control: submission **56223367**, estado **COMPLETE**, script version `349689941`. El ensemble reutilizó los dos modelos de E004 con los mismos umbrales. El CSV visible pasó la validación y conservó exactamente las detecciones del control, pero sus bifurcaciones predichas pasaron de 98 a 0. **Se cierra esta versión como resultado negativo.** [Implementación, resultado y límites](docs/TEMPORAL_SUBMISSION.es.md), [recibo completo](results/E005_test_completed.json).

## Desarrollo

**E006 completado, sin promoción a submission:** el preentrenamiento con 2.048 películas sintéticas por separación mejoró la AP de divisiones en ambos embriones reservados. Sin embargo, el score oficial condicional fue **0.9436645 frente a 0.9666951** de Harmonic, con divisiones TP/FP/FN **1/9/6 frente a 2/1/5**. Se verificaron el código descargado, los pesos, la procedencia de las texturas y el CSV. Al cerrar E006 no quedó una espera local activa. [Diseño, resultados y límites](docs/DENSE_PRETRAINING.es.md), [recibo completo](results/E006_completed.json).

**E007 completado, sin promoción a submission:** el especialista temporal conservó todos los enlaces y añadió 61, pero obtuvo **0.9558180 frente a 0.9666951** de Harmonic, con divisiones TP/FP/FN **2/6/5 frente a 2/1/5**. Se verificaron pesos, umbrales y reconstrucción del CSV. La auditoría geométrica de mitosis apunta a hijas ausentes o insuficientemente separadas entre las detecciones; la siguiente prioridad será el detector alrededor de las divisiones. No se inició otro notebook durante esta revisión. [Resultados, auditoría y límites](docs/MITOSIS_SPECIALIST.es.md), [recibo completo](results/E007_completed.json).

**E008/E009 implementados:** dos experimentos de detección que ejecutan después el tracking completo, con evaluación oficial y comparación contra el control congelado. El primero incorpora el detector temporal público CELLECT; el segundo resuelve núcleos superpuestos mediante ajuste de intensidades. [Diseño y límites](docs/DETECTOR_EXPERIMENTS.es.md). Los estados observados y recibos de lanzamiento se registran en `results/STATUS.json`.

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
