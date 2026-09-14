# E008/E009: detectar núcleos ausentes y resolver superposiciones

Fecha: 14 de septiembre de 2026. Se implementan dos mecanismos de detección distintos después de cerrar E007. La auditoría anterior encontró dos hijas sin un centro cercano y dos eventos con ambas hijas asociadas al mismo centro más próximo. Ese análisis tiene tolerancia espacial de 7 µm y fotograma exacto; no es un oráculo de la métrica oficial ni demuestra que el detector explique todos los errores.

## E008 — Detector temporal CELLECT

Se utiliza el UNet público de dos fotogramas de [CELLECT](https://github.com/zzz333za/CELLECT), asociado a [Nature Methods, 2025](https://www.nature.com/articles/s41592-025-02886-x). El código y checkpoint están fijados por commit y SHA256 en `NOTICE.md` y el manifiesto de activos. El estado se carga con `weights_only=True` y coincidencia estricta de todos los parámetros.

La entrada conserva la resolución nativa de Biohub, convierte ZYX a YXZ y aplica `log1p(max(raw, positive_min) + 1900)` a los dos fotogramas. Los parches miden 256×256×32 en YXZ, con solapamiento de 16×16×8; el último fotograma repite su propia imagen como contexto. Se extraen máximos de la clase de localización 4 dentro del vecindario de primer plano. El kernel espacial sigue el original; se normaliza su suma para calcular una probabilidad con escala estable y exigir al menos 0,5. Los bordes del parche se excluyen y los solapamientos se deduplican físicamente.

Es un adaptador del **detector preentrenado**, no una implementación del tracker CELLECT completo. Su MLP de exclusión y sus enlaces se reemplazan por NMS de 3 µm, confirmación en al menos un fotograma adyacente a 3 µm y asociación Harmonic. No se entrenan nuevos pesos en E008.

## E009 — Separación por ajuste de imagen

Implementación propia sobre imágenes a paso ZYX `(1,2,2)`, con escala física explícita. Se normaliza cada imagen con cuantiles 0,01/0,999, se localizan semillas mediante diferencia de suavizados y se seleccionan regiones alargadas. Cada región compara una gaussiana anisotrópica contra dos gaussianas con anchura compartida y centros/amplitudes distintos. Ambas incluyen fondo constante. Se optimizan intensidades mediante mínimos cuadrados con jacobiano analítico.

Se acepta una separación cuando ambas optimizaciones convergen, la segunda reduce el residuo al menos un 20 %, mejora el BIC más de 10 y produce centros separados entre 2,6 y 10 µm, con razón de amplitudes de al menos 0,25. Se evalúan hasta 48 regiones por imagen, ordenadas por elongación y señal. Una gaussiana aislada puede ser alargada: el modelo simple ya contempla esa forma y no necesita partirla para explicarla. Las propuestas requieren señal próxima en un fotograma vecino; las semillas de imagen también pueden corroborarlas cuando las hijas ya se han separado.

Este límite de 48 regiones acota el costo, y el requisito temporal puede perder eventos rápidos. Los ajustes no usan anotaciones, tiempos conocidos de mitosis, modelos entrenados anteriormente ni detecciones almacenadas.

## Comparación y controles

Cada notebook genera propuestas en todos los fotogramas de los mismos cuatro videos diagnósticos y ejecuta después el pipeline completo de Harmonic. Las propuestas entran **antes** del registro de coordenadas, la extracción de características, los enlaces y el ILP. El postprocesamiento puede eliminarlas o cambiar otros enlaces; no se promete conservar el grafo final.

Se mantienen los pesos, TTA, umbrales y postprocesamiento originales. Cada fotograma conserva primero sus detecciones de runtime y añade propuestas a más de 3 µm, hasta `max(16, ceil(0.15*n_base))`, por orden de confianza. Los centros añadidos se redondean en coordenadas nativas y luego se convierten a la grilla del enlazador. Ese límite contiene posibles falsos positivos por transferencia de dominio; las propuestas descartadas siguen contadas en los archivos de auditoría.

La integridad del código y pesos de Harmonic se comprueba antes de instalar el nuevo hook. Se registran hashes antes/después y una fila por video/fotograma con detecciones base, propuestas y adiciones. La ejecución falla si el sitio del parche cambia o faltan/sobran fotogramas de inyección. El CSV final pasa las comprobaciones estructurales y se puntúa con el código oficial fijado en `075fc5f5a52d11077f9dc2b074644618f26939e2`.

El control almacenado se verifica por hash y conjunto de videos. Su score de referencia es **0,9666951095**, con divisiones TP/FP/FN **2/1/5**. Se guardan score, aristas, divisiones, recuentos de nodos y deltas completos. Un aumento de propuestas por sí solo no demuestra una mejora.

**Esta comparación es condicional y sobre entrenamiento**: los cuatro videos estuvieron en el entrenamiento del segundo detector público, y su análisis de errores motivó estos experimentos. No constituye validación independiente ni permite anticipar un puntaje de leaderboard. No se emite `submission.csv` en la raíz ni se hace una submission automática.

## Ejecución y verificación

- `kaggle/detector_cellect`: E008, detector CELLECT y Harmonic completo.
- `kaggle/detector_gaussian`: E009, ajuste de gaussianas y Harmonic completo.
- Dos workers de propuestas por notebook, repartidos entre las GPU disponibles; después se ejecuta Harmonic con su reparto habitual.
- Resultados en `detector_experiment/result.json`; propuestas, recuentos por fotograma, hashes y CSV en subdirectorios.
- Pasaron la suite de 51 pruebas y después las 8 pruebas específicas del detector, incluida una nueva comprobación del hash canónico LF: 52 pruebas distintas. Se comprobaron el jacobiano numéricamente, la separación de núcleos superpuestos con ruido, el rechazo de un núcleo alargado aislado, el orden de ejes CELLECT, distancias físicas, límites temporales y la instalación del hook. Además se verificó una detección completa desde intensidades sintéticas, una carga estricta del checkpoint real y el parche sobre el predictor original.
- Los recibos `results/E008_launch.json` y `results/E009_launch.json` registran las respuestas reales de Kaggle. No hay monitor ni espera de finalización; Arturo avisa cuando terminen.

Ambas versiones 1 fueron aceptadas por Kaggle el 14 de septiembre a las 17:22 UTC (12:22 en Lima), inicialmente en cola. Tras el aviso de Arturo se verificó que ambas terminaron con **ERROR**. Ejecuciones: [E008 CELLECT](https://www.kaggle.com/code/jarturo/biohub-lab-cellect-detector) y [E009 Gaussian Deblending](https://www.kaggle.com/code/jarturo/biohub-lab-gaussian-deblending).

## Corrección de los fallos de la versión 1

E008 falló antes de generar propuestas porque Kaggle quitó el carácter `+` del nombre del archivo: el nombre real es `U-ext-x3rd-149.0-4.6540.pth`. Se descargó ese archivo de Kaggle y se confirmó que conserva el SHA256 fijado. El buscador ahora admite ambos nombres exactos y verifica el contenido antes de iniciar los workers; archivos incorrectos, ausentes o ambiguos siguen produciendo un error explícito.

E009 completó las propuestas en 519,94 segundos: 5.697, 5.263, 3.240 y 3.852 en los cuatro videos, respectivamente. Al instalar el parche de tracking, el recibo intentó usar `hashlib` sin que ese nombre estuviera importado en ese punto del script base. Se movió la instalación y la escritura del recibo a una función con sus propias dependencias. Los archivos de propuestas se descargaron y se verificaron por hash, límites y recuentos. **No hubo CSV final evaluado ni score oficial en ninguna de las dos ejecuciones.**

Pasaron 54 pruebas, incluidas nuevas regresiones para los nombres de archivo de Kaggle y la instalación completa del hook en un contexto sin imports heredados. Además se ejecutó la reparación sobre el predictor real descargado de E009 y se verificó que el predictor resultante conserva exactamente la lógica prevista. Los fallos y sus recibos originales se preservan en `results/E008_v1_failed.json` y `results/E009_v1_failed.json`; la comprobación con archivos reales queda en `results/E008_E009_recovery_verification.json`. Las correcciones no cambian los parámetros de los experimentos.

Las **versiones 2** de ambos notebooks fueron aceptadas el 14 de septiembre a las 17:53 UTC (12:53 en Lima), con estado inicial **QUEUED**. Los recibos actuales `E008_launch.json` y `E009_launch.json` corresponden a estas versiones; los originales están preservados dentro de los registros de fallo de v1. El payload corregido tiene SHA256 `fec5c73b111af40b8e09d402adefc63649ced13db8e4fa705b7a7b01dccef312`. No se espera ni consulta su finalización: Arturo avisa cuando terminen.

## Resultados verificados de la versión 2

Tras el aviso de Arturo, Kaggle confirmó **COMPLETE** para ambos notebooks. Ninguno superó al control:

| Ejecución | Score diagnóstico | Delta frente a Harmonic | Aristas TP/FP/FN | Divisiones TP/FP/FN | Nodos finales |
|---|---:|---:|---|---|---:|
| Harmonic control | 0,9666951095 | — | 2051/70/65 | 2/1/5 | 74.980 |
| E008 CELLECT | 0,9656549406 | −0,0010401688 | 2045/67/71 | 2/1/5 | 74.771 |
| E009 Gaussianas | 0,9628131229 | −0,0038819866 | 2032/61/84 | 2/1/5 | 74.679 |

CELLECT generó 90.345 propuestas corroboradas temporalmente e incorporó 11.134 centros antes del ILP. Gaussianas generó 18.052 e incorporó 5.638. Los nodos finales disminuyeron a pesar de esas incorporaciones: la competencia por enlaces y las etapas posteriores pueden modificar el resultado completo. No se interpreta esa diferencia de recuentos como una medida exacta de cuántas propuestas sobrevivieron. Los forks predichos totales fueron 86 y 77, frente a 73 del control; la métrica oficial siguió encontrando solo dos divisiones correctas. El tiempo registrado del pipeline, sin cola ni bootstrap, fue de 19,74 y 23,96 minutos.

Se verificaron el payload del notebook descargado, el script ejecutado, los hashes de los pesos declarados en el recibo de integridad, los CSV, las propuestas y las 400 filas de inyección por experimento. Se reprodujo localmente la agregación oficial a partir de los recuentos por video; **no se repitió localmente el matching oficial de grafos**. Los recuentos originales proceden de la ejecución oficial dentro de Kaggle. Registros: `results/E008_completed.json`, `results/E009_completed.json`; verificación reproducible: `scripts/verify_detector_results.py`.

## Qué explica la auditoría de divisiones

La inspección de siete eventos anotados compara las propuestas, el grafo final y la existencia de dos candidatos distintos dentro de 7 µm en el fotograma exacto. Es un análisis posterior sobre estos mismos videos; no selecciona parámetros ni cambia predicciones.

- Ambos experimentos recuperaron una hija antes ausente del vecindario de 7 µm. Sin embargo, esa nueva hija quedó sin padre y la madre conservó un único enlace hacia la otra hija. La detección mejoró geométricamente; la división no se reconstruyó.
- En otro evento, ambos métodos propusieron un centro dentro de 7 µm de una hija que siguió sin vecino final dentro de ese radio. Los archivos actuales no permiten separar si la propuesta se descartó por deduplicación, presupuesto de adiciones, ILP o postprocesamiento.
- Compartir el centro más próximo no implica necesariamente ausencia de dos candidatos: la búsqueda de correspondencias distintas encuentra ese caso incluso en el control. Por eso el vecino más cercano, como registró la auditoría anterior, no basta para afirmar que falta una detección.

El resumen agregado se guarda en `results/E008_E009_mitosis_audit.json`. Las coordenadas anotadas y las correspondencias detalladas de los siete eventos quedan localmente en `outputs/E008_E009_mitosis_audit_details.json`, fuera de Git. La correspondencia por distancia y fotograma exacto no sustituye la métrica oficial, que incorpora contexto del linaje y tolerancia temporal.

**Decisión:** cerrar estas dos versiones sin submission y conservar E000 como referencia pública. El siguiente diseño debe evaluar conjuntamente la conservación de centros y la conexión de madre/dos hijas durante varios fotogramas. Debe registrar la selección de cada propuesta por etapa para distinguir rechazo de detección y pérdida de asociación. No se inicia otra ejecución durante esta revisión ni se ajustan umbrales a estos siete eventos.
