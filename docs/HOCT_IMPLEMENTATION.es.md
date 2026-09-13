# E003: reemplazo del enlazador con morfología y HOCT

La hipótesis es que forma, intensidad y relaciones entre enlaces aporten información sobre divisiones y cruces que el transformer público de puntos no está aprovechando. No se afirma una mejora antes de medirla.

## Implementación ejecutable

1. Mantener los centros del detector del control. Para el diagnóstico se reutiliza su CSV ya generado en Kaggle, sin volver a ejecutar el detector. No se usan aristas ni etiquetas como entradas del nuevo enlazador.
2. Reconstruir máscaras por watershed 3D sobre intensidad suavizada, con un marcador por detección. Limitar cada región a 6 µm del marcador más cercano y a una intensidad relativa de 0.25. Son aproximaciones derivadas de imágenes, no máscaras neuronales ni predicciones FOCUS-3D.
3. Extraer diámetro equivalente, intensidad mínima/máxima/media/desviación, tensor de inercia y distancia al borde. Mantener coordenadas originales del detector. Orden y normalización de las 19 características siguen el código oficial HOCT.
4. Crear hasta cinco padres candidatos por célula en el frame anterior, dentro de 14 µm. Las distancias usan la escala física anisotrópica `(1.625, 0.40625, 0.40625)`. El modelo recibe posiciones y morfología en vóxeles, como el adaptador oficial consultado.
5. Inferir con el checkpoint **general_v1** en ventanas de cinco frames y stride tres. Subdividir espacialmente cuando el contexto excede 1.800 enlaces. Cada célula conserva todos sus padres candidatos; las predicciones repetidas se agregan mediante mediana de logits exponenciados y softmax parental con clase huérfana.
6. Resolver asociaciones globales por par de frames mediante matching bipartito de capacidad dos. El primer enlace evita la desaparición; el segundo incurre en coste de división. La celda hija puede aparecer sin padre. No se fuerzan enlaces ni se crean nodos sintéticos.
7. Exportar enteros con el esquema del concurso y comprobar límites, cobertura, grados, referencias y continuidad. Evaluar el CSV final con la métrica oficial fijada en el repositorio.

## Qué cambia respecto a HOCT oficial

Se usa su red preentrenada, no se ha entrenado otra. El solver es una implementación propia sin Gurobi: minimiza exactamente el objetivo aditivo de una pasada con todos los nodos fijos y aristas `dt=1`. No implementa la segunda pasada sobre tracklets ni cierre de huecos de HOCT. Esto permite cumplir directamente el formato de enlaces consecutivos del concurso.

Cada fuente tiene dos columnas en la asignación: costes `edge_bias - p - appearance*(1-orphan)` más `-disappearance` para la primera o `+division` para la segunda. Cada hija tiene una columna ficticia de aparición con coste cero. Al ser la primera columna más barata, nunca resulta óptimo usar la segunda sola. Los costes constantes omitidos no alteran el óptimo. El matching se resuelve con SciPy, sin heurística greedy.

Se usa FP32 porque la GPU T4 no ofrece bfloat16 nativo. El código oficial tiene autocast bfloat16 en CUDA. Las ventanas y el límite de contexto son decisiones de ejecución propias y pueden modificar el comportamiento respecto a inferencia de video completo.

La normalización procede de [HOCT `_api.py`](https://github.com/royerlab/hoct/blob/2ccc5040823bc944ab67790abd1f56eea7cd4f05/src/hoct/_api.py); el orden de características de [`_batching.py`](https://github.com/royerlab/hoct/blob/2ccc5040823bc944ab67790abd1f56eea7cd4f05/src/hoct/data/_batching.py); la agregación de [`_predict.py`](https://github.com/royerlab/hoct/blob/2ccc5040823bc944ab67790abd1f56eea7cd4f05/src/hoct/inference/_predict.py). Checkpoint SHA256: `5bd836dfcb15ad796ea79a9595841a3e73b650a71c4acba3fc66aac65d745b33`.

## Evaluación y límites conocidos

El primer experimento reutiliza cuatro videos completos, 400 frames y 74.980 centros. Es una comparación de arquitectura con detecciones fijas. Los cuatro videos aparecen en `split_manifest.json` de entrenamiento del segundo detector: **in-sample**, nunca una estimación de generalización.

El diagnóstico escribe `hoct_receipt.json`, métricas oficiales, tamaños de máscaras, probabilidades de candidatos y tiempos. Las matrices NPZ permiten analizar fallos sin repetir GPU. No selecciona hiperparámetros ni escribe una submission en la raíz. El notebook separado `hoct_test` implementa inferencia completa en test; ejecutarlo y evaluarlo es un paso posterior.

Riesgos que la ejecución debe medir: máscaras pequeñas o erróneas, representación fuera del dominio de entrenamiento de HOCT, pérdida de contexto al dividir ventanas y divisiones falsas. La detección se conserva para atribuir las diferencias al nuevo enlazador; aún no es un detector independiente.

FOCUS-3D sigue siendo una ruta distinta para mejorar detección/segmentación. El repositorio público está disponible, pero el modelo de Hugging Face pide aceptar acceso y compartir contacto. No se ha aceptado ese trámite ni se han sustituido sus pesos por archivos de procedencia desconocida.

## Reproducción

`python scripts/prepare_hoct_assets.py` descarga únicamente pesos públicos del release oficial y comprueba su SHA256; genera `artifacts/hoct_public_weights` con pesos, licencia y procedencia. Crear el dataset de Kaggle desde esa carpeta (`kaggle datasets create -p . -t`) evita un error del cliente Windows con rutas relativas que contienen `/`. El dataset es privado por defecto.

`python scripts/build_hoct_notebooks.py` empaqueta el código y genera los dos notebooks. El diagnóstico adjunta las salidas ya existentes de `jarturo/biohub-lab-official-metric-ab`; no vuelve a subir sus predicciones. `hoct_test` ejecuta el detector sobre los inputs de esa ejecución, de modo que no depende de IDs o predicciones fijas del test visible y sirve para una eventual reevaluación con test oculto.
