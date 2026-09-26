# Biohub Cell Tracking Lab

**E068 en curso (24–26 de septiembre):** nueva base pública x138 (0.953) reproducida; **0.954 público** con largo mínimo de pista 7 (submissions 56539929 y 56540269), puesto 177 dentro del bloque de 0.954. Un intento con validador y reescritura dentro del kernel dio 0.904 por el governor de tiempo de x138; desde entonces todo envío es de una sola pasada, con parámetros fijos y verificación previa del test visible. Laboratorio con el pipeline exacto sobre los **199 videos de train** y la métrica oficial: reglas de post-proceso confirmadas (+0.0022 dentro de muestra, positivas en ambos embriones) y relajación de la simetría de safe-division (+0.0035 en 72 videos con rastreo). Cinco envíos programados para el 27-09 a las 00:00 UTC. [Método, resultados y recomendaciones](docs/E068_X138_XR.es.md).

**E067 enviado:** suavizado robusto de trayectorias completas; intensidad media elegida entre tres. Validación reutilizada: **0,922099 frente a 0,919878**, +9 TP y −21 FP; mejora agregada en ambos embriones. Submission **56475507**, versión 1, **PENDING**. CSV validado: 63.593 centros modificados y enlaces intactos. Mejor score público confirmado sigue en **0.946**. [Método, resultados y límites](docs/E067_TRAJECTORY_DENOISE.es.md).

**E065 cerrado:** tres variantes de intercambio de enlaces empatan con el control y no cambian el test visible. Sin GPU ni submission. E066 terminó con +3 TP y sin FP adicionales en validación, pero tampoco cambia el test visible; sin GPU ni envío separado. [Protocolo](docs/E065_APPEARANCE_SWAP.es.md).

**E064 enviado:** nueva corrección de centros con imagen 3D nativa; tres variantes evaluadas en CPU. La mejor nueva obtiene 0,919508 frente a 0,919878 del control, sin mejora demostrada. Submission exploratoria **56466969**, versión 2, **COMPLETE: 0.946**; cambia 2.897 centros del test visible y conserva nodos y enlaces. Cuatro pruebas pasan; 7,22 min de evaluación CPU y 15,99 min de proceso en el notebook GPU. Mejor leaderboard confirmado: **0.946**. [Resultados y comprobantes](docs/E064_NATIVE_CENTERS.es.md).

**E063 completado:** selector aprendido con 758 trayectorias supervisadas (116 útiles), 40 videos de ajuste, 16 de desarrollo y 24 de evaluación reutilizada. Mejora la referencia simplificada, pero falla sobre el ensamble real: 0,918536 global / 0,918708 condicionado por embrión, frente a 0,919878. Auditoría exacta: un enlace nuevo recuperado y tres aciertos perdidos. Se conserva el control; sin nueva submission ni trabajos pendientes. 12 pruebas locales pasan; 18,40 minutos de proceso GPU, resto CPU. [Resultados y límites](docs/E063_TRAJECTORY_SELECTOR.es.md).

**E061–E062 completados:** detector ajustado con 40 videos reales y replay NIS3D; evaluación en 24 videos separados del ajuste, previamente usados en investigación. Recupera 512 centros más a 3 µm que el denso original. Siete variantes de integración no mejoran el score: la asociación neuronal gana dos enlaces TP netos y añade seis FP; 0,919413 frente al control de esta cohorte 0,919878. Entrenamiento e inferencia: 8,83 minutos de proceso GPU, resto CPU. Siete tests pasan. Sin nueva submission ni jobs pendientes; leaderboard **0.946**. [Protocolo, resultados y límites](docs/E061_DETECTOR_TEMPORAL.es.md).

**E058–E060 completados:** dos modelos sintéticos independientes y siete ensambles no superaron el control visual local 0,950283. Los modelos recuperan dos enlaces únicos, pero las mezclas uniformes y selectivas pierden más aciertos de los que recuperan. Auditoría oficial: 83 enlaces fallidos alcanzables y 67 con extremos ausentes. Inferencia GPU 147,7 s de proceso; resto CPU. Sin nueva submission ni jobs pendientes. Leaderboard: **0.946**. [Protocolo y resultados](docs/E058_E059_COMPLEMENTARIEDAD.es.md).

**E055–E057 completados, 21 de septiembre:** el control público E054 terminó en **0.946**. Ocho variantes nuevas no superan la asociación visual: tres filtros selectivos, dos pipelines con un tercer modelo sintético y tres mezclas sobre el grafo original. Los CSV cambian, pero el nuevo modelo y sus mezclas no mejoran los conteos evaluables de ningún video. Notebook GPU: 26,7 min de proceso; evaluación y mezclas CPU. Sin nuevas submissions ni jobs pendientes. **0.947 aún no alcanzado.** [Diagnóstico, fuentes y resultados](docs/E055_E056_ESPECIALISTAS.es.md).

**E051–E054, 21 de septiembre:** se confirmó E050 = 0.946. Tres nuevas fusiones y dieciséis combinaciones con detectores no superaron la asociación visual en los ocho videos reutilizados. Nuestro control `tight55` reproduce byte por byte el CSV público de Harmonic V3. Se envió la ablación limpia **56433260** (**COMPLETE: 0.946**). [Auditoría, resultados y límites](docs/E051_E054_PUBLICOS_Y_ENSAMBLE.es.md).

**E041–E043 completados, sin nueva submission:** la supervisión densa NIS3D
recupera 227 centros a 3 µm frente a 194 del mejor control a esa distancia,
pero pierde cobertura a 7 µm frente al campo estático. La resolución nativa
no mejora la localización: 3,796 frente a 3,767 µm. El seguimiento con
incertidumbre baja a 0,893002/0,891838 local frente a 0,900753 de Harmonic,
con más divisiones falsas. Ninguna variante pasa su criterio completo.
Se conserva la señal de supervisión densa. Siete pruebas locales y tres pesos
verificados; GPU 4,8 minutos de proceso, preparación y métricas CPU. Sin Colab
ni trabajos pendientes de estas etapas.
[Protocolos, resultados y límites](docs/E041_E043_TRES_LINEAS.es.md).

**E038–E040 completados y rechazados:** SpatialDINO no mejora la localización
frente al control de intensidades (5,435 frente a 5,395 µm). El detector temporal
recupera 175 centros a 3 µm frente a 177 del estático, con presupuesto de 256.
El modelo conjunto de madre/hijas, entrenado con 124 divisiones, obtiene AP
0,257 frente a 0,658 de geometría sobre 15 divisiones de desarrollo.
Las tres pruebas se ejecutaron en orden. GPU: 7,6 minutos de proceso; datos y
métricas en CPU. Sin nuevo submission ni trabajos pendientes.
[Protocolo y resultados](docs/E038_E040_TRES_LINEAS.es.md).

**E037 completado y rechazado:** seis cabezas entrenadas con representaciones de los encoders E033. Mejor AP neuronal en desarrollo: 0,325, frente a 0,406 de geometría y 0,443 de apariencia local E035. Ninguna superó la condición fijada; no se lanzó la integración al grafo ni submission. GPU: 7,6 s de proceso; preparación/cabezas en CPU. No quedan ejecuciones pendientes. [Resultados y protocolo](docs/E037_DIVISION_NEURONAL.es.md).

**E036 completado y rechazado:** reasignación de padres y reconstrucción conjunta bajan de 0,900753 a 0,900504 local. La variante conjunta aplica 50 eventos y añade 3 nodos, pero pierde un enlace correcto y añade uno incorrecto; no recupera divisiones anotadas. Todas las optimizaciones alcanzaron su óptimo. Proceso CPU: 212 s, GPU: cero. Sin submission ni ejecuciones pendientes. [Resultados y protocolo](docs/E036_RECONSTRUCCION_CONJUNTA.es.md).

**E035 completado:** entrenados tres clasificadores de divisiones en CPU. AP en desarrollo: geometría 0,406; apariencia inicial 0,443; secuencia 0,413. Ninguno superó la condición fijada en desarrollo; no hubo cambios del grafo y todos conservaron 0,900753 local. La auditoría señala hijas ausentes y conflictos de padre fuera del alcance de esta integración. Sin GPU ni submission; no quedan ejecuciones pendientes. [Resultados y límites](docs/E035_SECUENCIAS_DIVISION.es.md).

**E034 completado:** puentes geométricos 0,901014 local (+4 enlaces correctos, +1 incorrecto); con filtro visual 0,900971 (+3 correctos, +1 incorrecto), frente a 0,900753 del control. Ganancia pequeña, sin ventaja del filtro visual ni nuevo submission. GPU: 17 segundos de proceso; preparación y evaluación CPU. No quedan ejecuciones pendientes. [Resultados y protocolo](docs/E034_PUENTES_TEMPORALES.es.md).

**E033 completado:** encoder temporal 3D entrenado desde imágenes. En 6.725 decisiones obtiene 6.211 aciertos frente a 6.118 de geometría, mejorando en ambos embriones. La integración completa cambia 112 enlaces pero empata con Harmonic en 0,900753 local; las métricas por video son idénticas. Se conserva el encoder como componente prometedor, sin promover la integración ni enviar submission. Ambos procesos GPU sumaron 126 segundos; preparación y métricas en CPU. No quedan ejecuciones pendientes. [Protocolo y resultados](docs/E033_ENCODER_TEMPORAL_3D.es.md).

**E031/E032 completados:** detector denso evaluado en 16 videos completos, con scores locales 0,7560/0,7145 frente a 0,9008 de Harmonic. La condición por lote recupera 152 centros anotados omitidos por Harmonic, pero necesita muchos más candidatos. Cuatro cabezas temporales entrenadas en CPU: apariencia y geometría empatan en 6.021/6.725 aciertos, por debajo de 6.118 del vecino más cercano. No se promueven ni se envían. Proceso GPU: 164 s; preparación, evaluación y cabezas en CPU. No quedan ejecuciones pendientes. [E031](docs/E031_DETECTOR_DENSO.es.md), [E032](docs/E032_ASOCIACION_TEMPORAL_REAL.es.md).

[Revision de papers, discusiones y siete notebooks nuevos posterior a E030](docs/RESEARCH_POST_E030.es.md). Prioridad propuesta: deteccion densa y aprendizaje temporal sobre candidatos reales; sin nuevo entrenamiento ni submission en esta revision.

**E030 completado:** implementados y entrenados el modelo DINO de identidad/progenitores y la cabeza de divisiones, con decodificador conjunto y siete pruebas locales aprobadas. Separación de 40 videos para entrenamiento, ocho para desarrollo y 16 para calibración. En centros anotados perturbados acertó 586/588 enlaces, frente a 585 sin imagen y 588 con vecino más cercano; AP de división 0,219 sobre seis eventos. No demuestra mejora competitiva y no se envía. Extracción GPU: 202 s; entrenamiento/evaluación de las cabezas CPU: 60 s. Pesos y resultados guardados; no hay jobs pendientes. [Resultados y límites de E030](docs/E030_IDENTIDADES_CON_IMAGEN.es.md).

Repositorio privado de Arturo para [Biohub — Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development). Investigación y preparación inicial: **13 de septiembre de 2026**.

**Cierre de E029, 20 de septiembre:** el submission de asociación visual **56370600 terminó con 0,946**, igual al control público. **E027 completó 33 fotogramas de 12 videos seleccionados por errores conocidos:** NucVerse detectó 14 de los 40 centros que Harmonic omitía; ocho tienen un candidato cercano en un fotograma vecino. **E028, Cellpose-DINO, recuperó 2 de 5 omisiones en un piloto de dos imágenes**, donde NucVerse no recuperó ninguna, pero costó aproximadamente 30 veces más por volumen. Son diagnósticos de detección condicionados a errores, no mejoras de tracking ni de leaderboard. La decisión es priorizar integración temporal de NucVerse y reservar Cellpose para casos selectivos. No hubo nuevo submission. **E029 terminó la integración en CPU:** +0,000052 en el diagnóstico del CSV entero, un enlace correcto adicional y ninguna división recuperada. Es insuficiente y no se promueve; se descartan barridos de radios y se prioriza aprender identidades y enlaces con apariencia. No quedan ejecuciones pendientes. [Protocolo de integración](docs/E029_INTEGRACION_DIAGNOSTICA.es.md). [E027 y sus límites](docs/E027_COMPLEMENTARIEDAD_IMAGEN.es.md), [arquitectura independiente E028](docs/E028_CELLPOSE_DINO.es.md), [historial de adaptación E024–E026](docs/E024_NUEVA_LINEA.es.md).

**Cómputo:** laptop y Kaggle CPU para asociación, pruebas y evaluación; Kaggle GPU únicamente para entrenamiento e inferencia neuronal volumétrica. Sin Colab. El optimizador rápido produce exactamente el CSV del piloto; la asociación de sus dos videos tarda 0,88 y 1,58 segundos en la laptop. [Comprobación](results/E023_fast_assignment_equivalence.json). Los experimentos anteriores se conservan como historial.

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
| `kaggle/joint_lineage` | E010: selección conjunta de centros y dos trayectorias de hijas usando propuestas guardadas y evidencia de imagen | Grafo reconstruido, decisiones por etapa y métrica oficial |
| `kaggle/detection_dag` | E011: trayectorias desde detecciones nuevas, sin grafo de Harmonic, en 48 videos de desarrollo | Hipótesis congeladas, cobertura por etapa y criterio previo al entrenamiento |
| `kaggle/harmonic_dag` | E012: centros principales de Harmonic, sin sus asociaciones, más propuestas fijadas de E011 | Comparación de cobertura E011 / Harmonic solo / combinación en los mismos 48 videos |
| `kaggle/event_graph` | E013: nueva red de atención entre detecciones, divisiones explícitas y selección temporal consistente | Pesos, calibración separada, CSV y métrica oficial en los 48 videos de E012 |
| `kaggle/event_graph_control` | E013-control: Harmonic completo sobre los mismos 48 videos | Control emparejado para comparar la nueva selección |
| `kaggle/visual_sequence` | E020: probabilidades del grafo guardado, cobertura limitada por el suavizado | Métricas de calibración; no mejora |
| `kaggle/visual_replay` | E021: Transformer público sobre características antiguas | Métricas de calibración; no se promueve |
| `kaggle/visual_capture` / `kaggle/visual_exact` | E022: captura exacta GPU y comparación CPU separadas | Piloto positivo sobre dos videos |
| `kaggle/visual_validation_capture` / `kaggle/visual_validation` | E023: ocho videos reservados, configuración fija | Validación y criterio de promoción |
| `kaggle/visual_submission` | Inferencia final completada y enviada tras pasar E023 | `submission.csv`; submission 56370600 |

El [diagnóstico inicial](https://www.kaggle.com/code/jarturo/biohub-lab-official-metric-ab) terminó: control **0.9666951**, Division Guard **0.9667049**. No recuperó ninguna arista anotada ni división adicional. Se cierra esa línea sin mejora demostrada. Se verificó que **los cuatro videos estuvieron en el entrenamiento del segundo detector**; estas cifras son in-sample, no validación independiente ni leaderboard. [Resultado completo](results/E001_completed.json).

E003 reemplazó todas las asociaciones por HOCT y usó morfología medida en máscaras watershed 3D. [Implementación y diferencias respecto al paper](docs/HOCT_IMPLEMENTATION.es.md). [Ejecución en Kaggle](https://www.kaggle.com/code/jarturo/biohub-lab-hoct-morphology-diagnostic).

**E003 terminó con resultado negativo:** HOCT **0.9195083**, control **0.9666951**, delta **−0.0471868**. Las divisiones falsas evaluadas aumentaron de 1 a 12. No se promueve esta versión a submission ni se interpreta el score como validación independiente. Se procesaron los cuatro videos en 12,06 minutos, excluyendo instalación/cola. [Registro completo](results/E003_completed.json). El notebook `hoct_test` se conserva como implementación reproducible, **no como candidato recomendado**.

Los notebooks contienen el código necesario y adjuntan tres datasets públicos de Pilkwang: soporte, segundo seed y DeepCenter. La inferencia Harmonic requiere GPU; la auditoría y preparación temporal usan CPU. El entrenamiento temporal admite GPU o CPU y esta ejecución completa se realizó en GPU. Internet está desactivado. Los hashes de los pesos públicos y del código de soporte se verifican al arrancar. Los datos, los pesos y las credenciales no se guardan en Git.

**E004 también se completó:** se implementaron y entrenaron dos redes propias de imágenes de cinco frames, atención entre padres candidatos y una cabeza de divisiones. Cada una excluyó un embrión completo. En la comparación condicional con detecciones Harmonic, obtuvo **0.9412437 frente a 0.9666951**; las divisiones evaluadas pasaron de 2 aciertos a 0. El resultado no es leaderboard y el detector público de esa comparación vio esos videos. En la evaluación separada del modelo temporal sobre los 199 videos tampoco se superó al vecino más cercano al elegir padres. **No se envió este candidato.** [Resultados y límites](docs/TEMPORAL_IMPLEMENTATION.es.md), [recibo completo](results/E004_completed.json), [entrenamiento Kaggle](https://www.kaggle.com/code/jarturo/biohub-lab-temporal-train).

**E005 terminó en el leaderboard con 0.913**, frente a **0.946** del control: submission **56223367**, estado **COMPLETE**, script version `349689941`. El ensemble reutilizó los dos modelos de E004 con los mismos umbrales. El CSV visible pasó la validación y conservó exactamente las detecciones del control, pero sus bifurcaciones predichas pasaron de 98 a 0. **Se cierra esta versión como resultado negativo.** [Implementación, resultado y límites](docs/TEMPORAL_SUBMISSION.es.md), [recibo completo](results/E005_test_completed.json).

## Desarrollo

**E006 completado, sin promoción a submission:** el preentrenamiento con 2.048 películas sintéticas por separación mejoró la AP de divisiones en ambos embriones reservados. Sin embargo, el score oficial condicional fue **0.9436645 frente a 0.9666951** de Harmonic, con divisiones TP/FP/FN **1/9/6 frente a 2/1/5**. Se verificaron el código descargado, los pesos, la procedencia de las texturas y el CSV. Al cerrar E006 no quedó una espera local activa. [Diseño, resultados y límites](docs/DENSE_PRETRAINING.es.md), [recibo completo](results/E006_completed.json).

**E007 completado, sin promoción a submission:** el especialista temporal conservó todos los enlaces y añadió 61, pero obtuvo **0.9558180 frente a 0.9666951** de Harmonic, con divisiones TP/FP/FN **2/6/5 frente a 2/1/5**. Se verificaron pesos, umbrales y reconstrucción del CSV. La auditoría geométrica de mitosis apunta a hijas ausentes o insuficientemente separadas entre las detecciones; la siguiente prioridad será el detector alrededor de las divisiones. No se inició otro notebook durante esta revisión. [Resultados, auditoría y límites](docs/MITOSIS_SPECIALIST.es.md), [recibo completo](results/E007_completed.json).

**E008/E009 completados, sin promoción a submission:** las versiones 2 corrigieron los fallos de integración y terminaron el tracking completo. CELLECT obtuvo **0.9656549** y gaussianas **0.9628131**, frente a **0.9666951** del control; ambos conservaron divisiones TP/FP/FN **2/1/5**. Se verificaron fuentes, CSV, propuestas, inyección y agregación de la métrica. Ambos recuperaron geométricamente una hija antes ausente, pero quedó sin enlace con su madre; otras propuestas cercanas no llegaron al resultado final. La siguiente prioridad es decidir conjuntamente qué centros conservar y cómo reconstruir cada división durante varios fotogramas. Los scores son diagnósticos sobre entrenamiento, no leaderboard. [Resultados, auditoría y límites](docs/DETECTOR_EXPERIMENTS.es.md); recibos `results/E008_completed.json` y `results/E009_completed.json`.

**E010 completado, sin promoción a submission:** la selección conjunta produjo **0.9465246 frente a 0.9666951** del control, con divisiones correctas/falsas/omitidas **2/13/5 frente a 2/1/5**. Seleccionó 305 eventos y sustituyó centros, pero la auditoría posterior encontró que las cinco divisiones pendientes no tenían una ventana candidata cercana en su fotograma anotado: las restricciones sobre trayectorias y contexto de Harmonic las excluían antes de la optimización. Se verificaron entradas, hipótesis, reconstrucción exacta del CSV, objetivo del solver y agregación oficial. Sigue siendo un diagnóstico sobre entrenamiento, no leaderboard. [Resultado, auditoría y siguiente dirección](docs/JOINT_LINEAGE.es.md), [recibo completo](results/E010_completed.json), [cobertura agregada](results/E010_coverage_audit.json). Se cierra esta versión; no se inició otro notebook ni hay monitor activo.

**E011 completado; falló el criterio de cobertura:** el generador desde detecciones CELLECT/gaussianas procesó los **48 videos de desarrollo**. De 33 divisiones anotadas, hay centros compatibles para **18**, parejas representadas para **17** y dos trayectorias para **16**. Solo **11/25** eventos con continuación anotada completa tienen trayectorias compatibles con ambas ramas. El mayor problema inicial es la falta de centros: 15 eventos ya quedan fuera antes de formar parejas. Son techos de cobertura, **no precisión ni score oficial**. Se cierra esta configuración sin entrenar ni enviar una submission. [Resultado, verificación y límites](docs/DETECTION_DAG.es.md), [recibo](results/E011_completed.json), [auditoría agregada](results/E011_bottleneck_audit.json), [Kaggle](https://www.kaggle.com/code/jarturo/biohub-lab-detection-dag-coverage).

**E012 completado; la combinación pasa el criterio de cobertura:** en los mismos 48 videos, **Harmonic más CELLECT/gaussianas representa 33/33 divisiones y 24/25 continuaciones anotadas completas**, frente a 18/33 y 11/25 de E011, y 31/33 y 21/25 de Harmonic solo. La combinación pasa los mínimos de ambos grupos. Conserva los 1.237.567 centros principales y añade 1.462.003 complementarios. Estos son techos de cobertura, **no precisión ni leaderboard**. La siguiente implementación será puntuar y seleccionar los eventos para producir un grafo evaluable con la métrica oficial. No se entrenó un selector ni se envió una submission en E012; el selector siguiente se implementa en E013; no hay monitor activo. [Resultado, verificación y límites](docs/HARMONIC_DAG.es.md), [recibo](results/E012_completed.json), [auditoría de cobertura](results/E012_coverage_audit.json), [Kaggle](https://www.kaggle.com/code/jarturo/biohub-lab-harmonic-detection-dag).

Pruebas: Python 3.12, PyTorch, NumPy, SciPy, scikit-image, Zarr y nbformat. En PowerShell, desde la raíz:

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

**E013 lanzado, versión 1:** nueva red con características visuales de ambos UNet, atención entre detecciones y selección temporal de divisiones. Se separan **48 videos de ajuste, 16 de calibración y 48 de evaluación**. Kaggle aceptó el [candidato](https://www.kaggle.com/code/jarturo/biohub-lab-learned-event-graph) y el [control Harmonic completo](https://www.kaggle.com/code/jarturo/biohub-lab-event-graph-control) sobre los mismos videos, inicialmente en cola. Pasaron **86 pruebas** y una comprobación con pesos públicos reales sobre imágenes sintéticas. Todavía no hay resultados de E013 ni nueva submission. No hay monitor local. [Diseño y límites](docs/EVENT_GRAPH.es.md), [recibo del candidato](results/E013_launch.json), [recibo del control](results/E013-control_launch.json).

### E044: ensamble de componentes (2026-09-20)

Se combinó E041 denso con E039 estático en CPU. Existe complementariedad, pero la mezcla por rango a 256 candidatos obtuvo 211/329 aciertos a 3/7 µm frente a 227/321 y 177/354 de los componentes. La unión de 512 obtuvo 272/394 y tampoco superó a los mejores controles de 512. No se envió al leaderboard. [Protocolo y resultados completos](docs/E044_ENSAMBLE.es.md).

### E045–E048 y revisión pública (2026-09-20)

Terminaron seis combinaciones de detección, dos combinaciones de puentes con encoder, cuatro variantes de seguimiento con mapas y dos filtros DivNet. Ninguna pasó su criterio. El mejor grafo local sube de 0.900753 a 0.901409, con dos enlaces falsos adicionales y regresión en un embrión. DivNet compatible produce exactamente el CSV del control. GPU solo para 68.5 segundos de inferencia congelada; resto CPU. Sin nueva submission. [Experimentos](docs/E045_E048_COMPLEMENTOS.es.md).

Se revisaron listas públicas recientes y diez notebooks. Las adaptaciones anunciadas como 0.948 y 0.949 contienen un cargador DivNet incompatible que no carga ningún tensor; los títulos no se consideran scores verificados. Se documentan MLP V2, DAE, fuentes de pesos compatibles y discusiones recientes. [Auditoría pública](docs/REVISION_KAGGLE_20260920.es.md).

### E049–E050: nueva submission exploratoria

La captura del usuario confirma los scores del listado público, incluidos 0.947 para Harmonic V3 y el notebook de proxy. Se recuperó de ambos la selección real `tight55`. Tres combinaciones ponderadas se compararon en CPU: ganó la mezcla por acuerdo de detectores, 0.9014096 local frente a 0.900753 del control, prácticamente empatada con denso solo. Se integró con `tight55` en E050 y se envió la **submission 56409893**, tras validar el CSV completo. Resultado confirmado: **0.946**, sin mejora de leaderboard. [Diseño, resultados y recibos](docs/E049_E050_SUBMISSION.es.md).
