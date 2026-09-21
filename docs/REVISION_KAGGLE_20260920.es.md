# Revisión pública de Kaggle — 20 de septiembre de 2026

**Actualización con evidencia del usuario:** una captura posterior del listado ordenado por Public Score confirma 0.950 para Biohub 0.95 y 0.947 para el notebook de proxy, Harmonic V3 y Harmonic. Esos scores del listado sí están verificados visualmente. La limitación de API descrita abajo no debe interpretarse como que el listado carece de scores reales. Se descargó además la selección pública `tight55` y se incorporó a [E049–E050](E049_E050_SUBMISSION.es.md).

Se consultaron tres listas de 40 notebooks: orden por score, última ejecución y creación. Se descargaron diez notebooks, con hashes, para comparar implementaciones y examinar bloques concretos. La API de listado permite ordenar por score pero no devuelve el valor del score en sus registros. Por ello los números en títulos no se presentan aquí como resultados públicos confirmados. La página dinámica de varios notebooks no pudo leerse con el navegador disponible; se usaron la API oficial, código y registros públicos.

## Hallazgos que cambian decisiones

### Density-Adaptive 0.948 Reproduction

[Notebook de andnyu](https://www.kaggle.com/code/andnyu/biohub-density-adaptive-0-948-reproduction), ejecución pública 20/09 23:10 UTC, COMPLETE. El código declara grupos por promedio de nodos por imagen (<120, <400, resto), modifica peso de velocidad y bonificación del enlace. Sin embargo, `motion_relink_edges` recibe `tight_um` y `relaxed_um` pero nunca lee esos argumentos: los bucles siguen usando los valores globales. La auditoría AST registra ambos argumentos sin uso en `results/public_notebooks_static_audit.json`. No se debe atribuir una ganancia a radios adaptativos que no se aplican.

El problema más importante es DivNet. El log dice que cargó `giorgosi/biohub-divnet-v2/best_overall.pt`. Se descargó ese archivo público y se comparó con las clases que efectivamente instancia el notebook: 46 tensores/buffers esperados, 50 claves en el checkpoint, **cero nombres en común**. `load_state_dict(strict=False)` no modifica ningún tensor del clasificador. El checkpoint espera cinco canales y un UNet con InstanceNorm; el clasificador del notebook tiene un canal y BatchNorm. La prueba local ejecuta solo las dos clases revisadas, sin ejecutar el notebook ni su pipeline. Resultado: `results/public_divnet_compatibility_audit.json`. El mensaje de carga exitosa no demuestra que el filtro esté entrenado.

### SOTA 0.949+ Local Motion-Flow

[Notebook de haideptry](https://www.kaggle.com/code/haideptry/biohub-sota-0-949-local-motion-flow-2xt4-19m), ejecución indicada 18/09. Añade una predicción de movimiento a partir de desplazamientos vecinos, iteraciones de asignación y puertas alrededor de la posición predicha; mantiene el bloque de DivNet con la arquitectura incompatible. No se confirmó el 0.949 del título como score del código descargado. Parte de la idea se relaciona con el movimiento colectivo ya evaluado en E017; su mera presencia en otro notebook no convierte esa línea en una mejora probada.

### DivNet compatible: una fuente aprovechable

[Implementación compatible](https://www.kaggle.com/code/canhtoanle/biohub-div-complete-v33a), [pesos de giorgosi](https://www.kaggle.com/datasets/giorgosi/biohub-divnet-v2). Se implementó el UNet de cinco canales compatible, con carga estricta de los 50 tensores: 1.402.993 parámetros, inferencia finita verificada. No se copió el pipeline entero.

El manifiesto indica cuatro imágenes temporales [-1,0,1,2], un marcador gaussiano, recortes 16x32x32, XY4, percentiles 50/99.5 y clipping [-0.5,6]. La procedencia exacta de entrenamiento de `best_overall` no está publicada; el manifiesto resume 199 películas y resultados de validación del autor. Sus campos `lb_upside_simulation` son simulaciones, no submissions verificadas. También hay ambigüedad entre normalización por imagen o por recorte y entre el uso de XY4 del manifiesto y la entrada nativa de la adaptación pública. E048 declara esas limitaciones y compara dos interpretaciones de normalización con XY4 fijo, sin atribuirles reproducción exacta del entrenamiento.

La construcción propia del recorte se comparó con las funciones públicas sobre la misma entrada XY4, incluidos bordes y centros fraccionarios: máxima diferencia absoluta 2.3842e-7. Esto valida la implementación del recorte, no su identidad con el preprocesamiento original de entrenamiento. Véanse `results/public_divnet_corrected_load.json` y `results/public_divnet_preprocess_audit.json`.

### Linker Association MLP V2

[Notebook de noisyislands](https://www.kaggle.com/code/noisyislands/biohub-linker-association-mlp-v2), ejecutado 20/09 20:19 UTC, COMPLETE. Ocho atributos geométricos/de movimiento y red 8→16→16→1. Valida que enlaces cortos reciban mayor probabilidad que largos y registra AUC 0.98161185. Es AUC de pares construidos con anotaciones y distractores sintéticos, no score de seguimiento. Usa antecedentes de trayectoria anotados durante la construcción de ejemplos y marca pares ausentes como negativos; requiere revisar la correspondencia con predicciones reales y con anotación parcial antes de transferirlo. E033 ya mostró que una mejora del ranking no basta para mejorar el grafo.

### DAE y autoentrenamiento

[Notebook de ghazarosghazaros](https://www.kaggle.com/code/ghazarosghazaros/biohub-dae-self-distill-repeat), actualizado 20/09 22:17 UTC, COMPLETE. Entrena un pequeño eliminador de ruido por video con ruido gaussiano, mezcla su salida con la imagen y realiza dos pasos que convierten máximos en mapas gaussianos para modificar logits. Es una hipótesis real de adaptación, pero el score 0.946 mencionado corresponde al antecedente descrito por el autor; no se verificó una ganancia de esta ejecución. Se conserva como línea posible, sin presentarla como nueva mejora confirmada.

### Los notebooks titulados 0.95

[anvithpothula](https://www.kaggle.com/code/anvithpothula/biohub-0-95) y [codezzzsleep](https://www.kaggle.com/code/codezzzsleep/biohub-095-owned-validation) contienen exactamente el mismo texto de código extraído. Añaden un nodo central y bifurcaciones artificiales con tiempos y coordenadas negativos fuera del volumen. Ese añadido no representa células observadas y no es una fuente de mejora de seguimiento. No se incorporó.

[Proxy-score de evgendvorkin](https://www.kaggle.com/code/evgendvorkin/biohub-0-942-lb-proxy-score-0-9417) calcula y selecciona variantes por un validador local. El proxy impreso no equivale al score público del notebook. Se inspeccionaron también los dos notebooks del autor giorgosi para localizar el origen de modelos; no se ejecutó código público completo.

## Discusiones consultadas y consecuencias concretas

- [Datos sintéticos y comentarios recientes sobre entrenamiento denso](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/732103): se proponen ventanas de cinco fotogramas con pseudoetiquetas densas y revisión de trayectorias cortas, y tratar divisiones como conexiones entre tracklets. Es una dirección de supervisión temporal, no prueba de que un ensamble específico mejore. El autor del dataset reconoce diferencias de contraste/textura y una tasa de divisiones artificialmente elevada. Hay un comentario reciente enlazando datos reales de desarrollo de pez cebra en SSBD.
- [Generación de datos y recursos externos](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/739731): aparecen recursos Zebrahub y datos de ultrack. Un participante también informa dos semanas de síntesis sin mejora clara; no se asume que más datos sintéticos resuelvan el problema automáticamente.
- [Diseño de divisiones en zonas densas](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737438): el autor describe omisiones que incumplen varios filtros a la vez. Ya motivó líneas conjuntas E036–E043; no se presenta como descubrimiento nuevo ni justifica repetirlas sin cambiar su evidencia.
- [CV frente a leaderboard](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/730160): participantes advierten que algunos checkpoints públicos se entrenaron sobre todos los videos. Es una advertencia de procedencia concreta; no se extrapola sin verificar al entrenamiento de todos los modelos existentes.
- [Organizador sobre los cuatro videos visibles de test](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/716062): aclara que son ejemplos para probar el notebook y que la puntuación usa un conjunto privado mayor. Ejecutar un notebook público hasta generar su CSV no valida el número anunciado en su título.

## Acción tomada

E048 utiliza el DivNet compatible como filtro de divisiones existentes de Harmonic, en CPU. Se conservan todos los nodos, se usa el umbral publicado 0.5 y, si se rechaza una bifurcación, se elimina solo el enlace a la hija más distante. Se reportan ambas normalizaciones y se exige mejorar score y FP sin perder TP. E047 también terminó: el consenso con puentes mejora 0.000312 local pero añade cinco enlaces falsos y perjudica un embrión; no supera al donante denso solo. Los resultados se documentan en [E045–E048](E045_E048_COMPLEMENTOS.es.md); ninguna afirmación pública de 0.948/0.949 se usó como prueba de aprobación.

Resultado de E048: las dos normalizaciones terminaron en CPU y el filtro con umbral publicado 0.5 no rechazó ninguna de las 316 bifurcaciones. Sus CSV son exactamente idénticos a Harmonic (score local 0.9007529596). La reparación del cargador fue necesaria para probar la idea, pero no produjo una mejora de esta integración. No hubo submission ni barrido de umbrales.
