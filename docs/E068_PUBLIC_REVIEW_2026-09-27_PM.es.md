# Revisión pública adicional — 27-09-2026, 16:03 UTC

Petición: buscar información que explique el estancamiento y líneas distintas para alcanzar plata. Esta revisión descargó código y logs públicos, leyó discusiones actuales en el navegador y contrastó implementaciones; no ejecutó código externo, no entrenó y no consumió nuevas submissions.

## Lo comprobado en el listado público

En la vista accesible ordenada por **Public Score**, Harmonic V3, optimized-biohub-max-score, Biohub Cell Tracking de Kunal y el original de Anvith muestran **0.953**. Nuestro mejor envío confirmado es c3, **0.955**. No encontré en esa lista una solución pública verificada de 0.96+ lista para copiar. Esto no describe notebooks privados o con fuentes inaccesibles.

- [Listado consultado](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/code?competitionId=136605&sortBy=scoreDescending&excludeNonAccessedDatasources=true).
- [Kunal](https://www.kaggle.com/code/kunaldesale2408/biohub-cell-tracking): el árbol sintáctico completo coincide con nuestro x138 upstream. Las diferencias del archivo son espacios y líneas vacías.
- [Harmonic V3](https://www.kaggle.com/code/raunakdey07/biohub-harmonic-fusion-v3): las 81 funciones/clases de nivel superior coinciden tras quitar docstrings. Esta comprobación no equivale a afirmar igualdad de todas las instrucciones de nivel superior. Ya incorpora V1284 y los mecanismos que tenemos.
- [Aman](https://www.kaggle.com/code/amanatar/optimized-biohub-max-score): sin nueva ejecución desde 01:39 UTC, ya revisada. Su texto «0.965+» sigue sin ser el score del listado. Mantener las observaciones de variables de horizonte no definidas y calibración por media/desviación del análisis anterior.

Otros archivos revisados: `binasalama/biohub-learned-unet-transformer-ilp-gap-recovery` repite mecanismos conocidos; `tobimichigan/robust-cell-tracking-during-development-via-chan` muestra 0.1 en el listado y no justifica sustituir c3; `asymortenson/biohub-v5c-final` descomprime predicciones embebidas para cuatro nombres fijos, no ofrece inferencia para videos nuevos. `noisyislands/biohub-transformer-finetune` entrena un Transformer con duplicados y ruido artificial; no encontramos evidencia de ganancia pública que justifique repetir ahora esa línea.

Inventario por tres órdenes de búsqueda: `results/E068/public_list_2026-09-27_1555.json`. Fuentes y hashes: `results/E068/public_source_audit_2026-09-27_pm.json`. Código descargado, sin ejecutar: `outputs/e068_public_27_pm/`.

## Hallazgos que cambian la prioridad

### 1. El score local no debe ordenar por sí solo los próximos envíos

El [hilo de ocho experimentos negativos](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/743222), publicado el 25-09, reporta que desactivar relink subió +0.017 local pero perdió 0.002 público. También reporta pérdidas al añadir Z-flip TTA, variar el umbral de detección y reemplazar el segundo modelo. Son resultados declarados por ese participante, no experimentos nuestros.

Comprobación propia: el `split_manifest.json` del detector secundario que ya tenemos declara **199 videos en train** (`unet_transformer_alltrain_seed314159_v1`, SHA256 `cbe8ace34ffc157172280538441454b60250f0188faa063d1a9eadfb1ac55c0b`). Una separación de esos videos hecha después no convierte al detector en independiente. Esto ya era una limitación conocida: el hilo añade evidencia pública de su efecto práctico. El bootstrap tampoco la corrige.

Consecuencia: terminar CPU7 para detectar errores, sensibilidad y cambios reales, pero no seguir encadenando rejillas solo porque produzcan deltas locales positivos. C3_fork8 continúa siendo una prueba pública razonable pendiente; no es una predicción de 0.956.

### 2. Recurso concreto que no habíamos integrado: linker End2End de hengck23

[Notebook de asociaciones](https://www.kaggle.com/code/hengck23/end2end-cell-linker-raw-edge-ja-0-9-no-ilp), actualizado el 15-09, distinto del detector de puntos que probamos anteriormente. No es una novedad publicada hoy; es una capacidad antes no integrada en nuestro proyecto.

La API confirma el checkpoint `00000008.pth` de **43.076.221 bytes** en el [dataset público](https://www.kaggle.com/datasets/hengck23/hengck23-cell-point-detector-demo). Se descargaron solo `model_v12.py`, el helper y el log, no los pesos. Arquitectura inspeccionada: UNet 64/128/256, rasgos multiescala, atención propia/cruzada y cabeza de pares. Su función `sample_pyr_feature_at_zyx` permite muestrear rasgos en coordenadas suministradas: habilita evaluar las detecciones de c3 sin reemplazarlas por las del donante.

El log publicado confirma carga estricta completa y 11 videos ejecutados, entre **0.20 y 0.93 min por video** en esa ejecución. Media de edge Jaccard bruto 0.895998 y recall de nodos 0.994322. Estos tiempos no están medidos en nuestro entorno y el score no es el de la competencia completa.

Comparación descriptiva con los mismos 11 videos de nuestro c3: el donante gana en edge Jaccard bruto en **3**, empata en **1** y pierde en **7**. Por ejemplo, 44b6_0113de3b: 1.000 frente a 0.868; 44b6_0b24845f: 0.737 frente a 1.000. Las detecciones difieren y el donante se evaluó con coordenadas flotantes; todavía no demuestra complementariedad a nivel de aristas ni mejora del ensamble. El split de entrenamiento del donante no está establecido.

**Siguiente experimento prioritario:** conservar detecciones/divisiones de c3 y utilizar este modelo como segunda opinión sobre asociaciones ambiguas. Cachear sus logits una vez. Evaluar el control, corrección geométrica sola y corrección con rasgos neuronales bajo idénticos candidatos. Aplicar el desacuerdo después del relink final, para que no borre la evidencia. Medir errores corregidos, nuevos errores, aristas sin correspondencia y tiempo; no seleccionar un modelo por el nombre del video ni por su score anotado.

Esta prueba difiere de CPU6, que solo intercambió enlaces por geometría, y de E055–E057, que usaron nuestros especialistas sintéticos. No promediar probabilidades de escalas distintas sin calibración. Empezar por márgenes/consenso bidireccional, proteger bifurcaciones y mantener la opción de no cambiar el enlace.

Las coordenadas de c3 se convierten a la rejilla del donante como `(z, y/4, x/4)`; su voxel isotrópico equivale a 1.625 µm. Omitir esta conversión invalidaría el experimento. Usar características `[d0,d1,e2]`, como en el checkpoint, no cambiar a `[e0,e1,e2]` por similitud de dimensiones.

### 3. Probar duplicados persistentes después del refinamiento

El nuevo linker hace supresión de duplicados **después** de refinar coordenadas. En nuestro código V1284 desplaza centros y después se reconstruyen trayectorias; el filtrado final de c3 elimina componentes cortos, pero no busca pares de trayectorias redundantes cercanas a lo largo del tiempo.

Propuesta propia: auditar con las capturas CPU cuántos pares permanecen muy próximos varios fotogramas, con movimiento similar, y si uno aporta escasa confianza. Solo después comparar una supresión conservadora, protegiendo bifurcaciones y continuaciones únicas. No copiar el radio del otro modelo ni agrupar cualquier célula vecina: podría borrar células distintas. Esta hipótesis usa evidencia espacial y temporal, distinta de subir el mínimo de longitud.

## Papers y límites de la investigación

[ASCENT, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Han_ASCENT_Annotation-free_Self-supervised_Contrastive_Embeddings_for_3D_Neuron_Tracking_in_ICCV_2025_paper.pdf) y su [código oficial](https://github.com/lu-lab/ascent) aportan entrenamiento de identidad mediante vistas deformadas de imágenes sin etiquetas de tracking. Es una dirección para aprender apariencia adaptable, pero su dominio de neuronas no demuestra rendimiento con divisiones de Biohub. Antes del cierre prefiero probar el linker ya entrenado frente a empezar ese entrenamiento.

Se revisitaron [FOCUS-3D](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/738217) y [HOCT](https://arxiv.org/abs/2607.11754), ya conocidos; no se presentan como descubrimientos nuevos. El enfoque directo de FOCUS tiene reportes de timeout. [Luxar](https://zenodo.org/records/22908223) representa/visualiza volúmenes mediante splats; por sí solo no aporta una solución de identidad o score competitivo.

El [hilo de errores de anotación](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/742942) contiene ejemplos y testimonios de enlaces/divisiones dudosos; no demuestra su prevalencia ni cómo está anotado el test. Evitar convertir toda división visualmente plausible en una adición automática. El [hilo dirigido al top 100](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/742169) no revela una receta concreta: un participante top 10 solo señala margen en varias etapas.

## Decisión operativa

1. Completar y verificar C3_fork8 y CPU7, ya lanzados. Reservar envíos para cambios distintos, no toda la rejilla.
2. Dar prioridad al **linker independiente sobre detecciones de c3**, con un ensamble condicionado por confianza y aplicado al grafo final. La mejora aún no está medida.
3. Auditar **trayectorias duplicadas persistentes** en CPU. Descartarlo si no hay candidatos reales; no gastar GPU en una regla inactiva.
4. No iniciar ahora otro detector completo, más TTA global ni otra adaptación sintética del mismo modelo basándose solo en CV.

Consulta 16:02 UTC: **C3_fork8 sigue QUEUED**, sin mensaje de fallo; **CPU7 RUNNING**. No se afirmó que el archivo nuevo esté listo y no se programaron envíos automáticos. Los próximos cupos se abren a las 19:00 Lima.
