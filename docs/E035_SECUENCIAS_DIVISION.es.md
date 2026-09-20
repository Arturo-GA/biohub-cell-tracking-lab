# E035: secuencias madre e hijas

Se investiga si las medidas temporales de imagen distinguen una división de dos células vecinas independientes. E034 solo recuperó unos pocos enlaces de continuación y no corrigió divisiones. Esta prueba cambia el tipo de evento aprendido, en lugar de ajustar sus umbrales.

La separación de E030 se conserva: 40 videos de entrenamiento, 8 de desarrollo y 16 de calibración reutilizada. El recuento previo de E030 era 29 divisiones representadas en entrenamiento y 6 en desarrollo; E035 vuelve a contar las que cuentan con contexto completo. Se requieren tres nodos consecutivos de la madre y tres de cada hija. La preparación usa todos los contextos disponibles, no solo los ocho instantes uniformes de E030.

Se entrenan tres clasificadores ExtraTrees de 256 árboles con los mismos parámetros: geometría temporal; geometría más apariencia inicial; geometría más secuencia de apariencia. La apariencia consta de intensidad local, dos medias gaussianas, desviación local, Laplaciano y contraste entre escalas, medidos en imágenes 3D de resolución isotrópica. Las características son simétricas al intercambiar las hijas. No se entrena una red neuronal y no se usa GPU.

Entrenamiento y desarrollo utilizan trayectorias anotadas con perturbación espacial de hasta 1 µm por eje. Se conservan todos los positivos y hasta 2.000 negativos por video de entrenamiento. Desarrollo conserva todos los negativos conocidos. Este currículo tiene un cambio de distribución respecto a los candidatos reales: por eso se exige la evaluación final sobre los nodos y trayectorias de Harmonic. En calibración, los candidatos se generan sin consultar anotaciones, con hasta cuatro hijas cercanas a 20 µm y continuidad de ambas. Las etiquetas desconocidas no se convierten en negativos.

El umbral se selecciona exclusivamente en desarrollo: al menos dos positivos aceptados y cero negativos conocidos aceptados, maximizando los positivos y eligiendo el umbral más alto en caso de empate. Si ningún umbral cumple, esa variante no modifica el grafo. Es una condición sobre esta muestra pequeña, no una garantía de precisión en datos nuevos.

La primera integración busca divisiones omitidas: añade una segunda hija sin padre, o dos hijas sin padre si la madre termina. Preserva los nodos y todas las divisiones existentes; no puede corregir falsos eventos existentes ni recuperar hijas ausentes. No permite asignar una hija ya conectada a otra madre. Se evalúan los cuatro CSV completos con la métrica oficial congelada y el mismo control local 0,9007529595605399.

La procedencia de entrenamiento de Harmonic sigue sin estar verificada, y los 16 videos son calibración reutilizada. No se declara holdout independiente del detector ni se envía submission automáticamente.

La preparación v1 falló antes de entrenar porque tracksdata incluye el identificador de arista además de sus extremos. Se corrigió la lectura por nombre de columna y se relanzó como v2. Antes de obtener datos o métricas también se eliminó el muestreo de negativos en desarrollo. El notebook y recibo fallidos se conservan para reproducibilidad.

## Datos preparados

| Conjunto | Videos | Positivos representados | Divisiones anotadas | Negativos conocidos | Candidatos sin etiqueta |
|---|---:|---:|---:|---:|---:|
| Entrenamiento | 40 | 20 | 29 | 9.372 | 0 |
| Desarrollo | 8 | 4 | 6 | 1.116 | 0 |
| Calibración real | 16 | 3 | 6 | 13 | 82.336 |

Los candidatos reales de calibración suman 82.352. La evaluación incluye una auditoría de las seis divisiones anotadas para identificar ausencia de nodos, contexto incompleto y conflictos con los padres/hijos existentes. Esta auditoría utiliza anotaciones exclusivamente como diagnóstico y no alimenta candidatos, entrenamiento, umbrales ni decodificación.

## Resultado

| Modelo | AP de desarrollo | Supera condición de desarrollo | Eventos modificados | Puntaje local |
|---|---:|---|---:|---:|
| Geometría temporal | 0,405893 | No | 0 | 0,900752960 |
| Apariencia inicial + geometría | 0,442560 | No | 0 | 0,900752960 |
| Secuencia de apariencia + geometría | 0,413393 | No | 0 | 0,900752960 |

Los tres modelos se entrenaron efectivamente sobre 20 positivos y 9.372 negativos conocidos. Ninguno permite aceptar dos positivos de desarrollo sin aceptar algún negativo conocido, por lo que el protocolo fijado desactivó todas sus ediciones. El empate del grafo no prueba equivalencia de los modelos: es consecuencia de ese rechazo previo. Se verificó que todas las métricas por video son idénticas al control, no solo el promedio. Las divisiones permanecen en 3 correctas, 4 falsas y 3 omitidas.

En los 16 candidatos de calibración con etiqueta verificable, la AP fue 1,000 para geometría y 0,917 para las dos variantes de imagen. Esto no justifica promoverlos: ignora 82.336 candidatos sin etiqueta y la condición de desarrollo ya había fallado. No se ajustaron umbrales con la calibración ni se envió submission.

La auditoría geométrica de correspondencias encuentra tres eventos positivos propuestos: dos ya conectados y uno pendiente. Entre las divisiones que el control omite, en `44b6_aaf8b0ea` falta una hija bajo el radio de correspondencia; en `44b6_c8e2a523` una hija tiene otro padre; en `6bba_9a41d029` existe una propuesta elegible pendiente. La auditoría usa un emparejamiento geométrico uno a uno propio, no el emparejamiento interno de la métrica oficial: no deben equipararse automáticamente sus etiquetas de conectividad con los verdaderos positivos oficiales. Por ejemplo, en `44b6_587a1e22` ambas lecturas difieren.

La conclusión se limita a este clasificador sobre resúmenes locales de imagen: la secuencia no supera a la apariencia inicial en desarrollo. No es una prueba de que una red de secuencias más expresiva no pueda mejorar. Sin embargo, entrenar más esta misma cabeza no corrige las restricciones de candidatos observadas. La siguiente línea propuesta combina reasignación de padres con recuperación de hijas ausentes; aún no está implementada ni lanzada.

La preparación corregida v2 tardó 728,05 segundos de proceso CPU y el entrenamiento/evaluación v1, 137,56 segundos; se añaden el intento inicial fallido y los tiempos de arranque/exportación. Consumo GPU: cero. Modelos guardados en los outputs privados de Kaggle y descargados localmente; código, notebooks y recibos guardados en GitHub. [Resultado estructurado](../results/E035_completed.json).
