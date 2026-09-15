# Estrategia: laptop y Kaggle CPU para Biohub

Decisión de Arturo del **15 de septiembre de 2026**: entrenar con la GPU de su laptop y utilizar Kaggle CPU para preparación y evaluación. No se planifican nuevas ejecuciones en Colab, Kaggle GPU/TPU ni servicios de pago. Este documento define la migración; **todavía no se ha habilitado CUDA local ni se han implementado los nuevos ejecutores**. Los notebooks E013 versión 1 se conservan como antecedentes reproducibles.

El objetivo científico sigue siendo aprender a elegir asociaciones y divisiones a partir de las detecciones combinadas de E012. Conservamos los detectores públicos congelados y entrenamos EventGraphNet. La migración no reduce el experimento a una muestra pequeña ni implica una mejora de score por sí misma.

## Recursos comprobados y presupuesto inicial

| Recurso | Observación | Consecuencia |
|---|---|---|
| GPU local | RTX 3050 Laptop, 4.096 MiB de VRAM; driver 529.04 | Medir memoria antes del entrenamiento completo; procesar un modelo de imagen a la vez |
| CPU local | i5-11320H, 4 núcleos y 8 hilos | Carga de datos y tareas ligeras; empezar con 2 hilos de cálculo |
| RAM local | 15,8 GiB instalados; 3,4 GiB libres en la revisión anterior | Carga diferida, caché de uno o dos videos; iniciar con al menos unos 6 GiB libres |
| Disco local | 28,0 GiB libres al revisar este plan | Mantener los volúmenes completos en Kaggle; reservar al menos 10 GiB libres |
| PyTorch disponible | 2.14.0+cpu, CUDA no disponible en ese entorno | Crear un entorno propio de Biohub; no modificar las dependencias de RSNA |

Medimos **191.255 parámetros** en el selector: unos 0,73 MiB de pesos FP32. Los 48 grafos de E012 disponibles contienen **2.699.570 centros** y ocupan unos **130 MiB comprimidos**. Para esos centros, las 64 características visuales FP16 ocuparían unos **330 MiB**, y los índices de vecindad int32 unos **257 MiB**. Son proyecciones de arrays, no tamaños medidos de un paquete completo: faltan etiquetas, metadatos y estructuras temporales. Las características visuales todavía no están disponibles localmente. [Medición agregada](../results/LOCAL_COMPUTE_SIZING.json).

La ventana más densa encontrada en esos grafos tiene **17.300 centros en diez fotogramas**. Por eso, el tamaño de los pesos no demuestra que el entrenamiento quepa en 4 GB: hay que medir activaciones, atención y memoria reservada. Tampoco se ha medido aún cuánto tardará en esta laptop.

Presupuestos iniciales, sujetos a medición: caché local de entradas de hasta 6 GiB, proceso de entrenamiento de hasta unos 5 GiB de RAM y objetivo de ocupación total de GPU de aproximadamente 3,2 GiB. La instalación de CUDA/PyTorch y los archivos temporales también cuentan contra el disco disponible. Si no caben junto a la reserva, se ajustará el almacenamiento antes de instalar; no se borrarán archivos personales.

## Reparto del trabajo

| Etapa | Equipo | Producto reutilizable |
|---|---|---|
| Conservar los videos completos y preparar lotes de datos | Kaggle CPU | Manifiestos y paquetes por video |
| Reutilizar detecciones; generar gaussianas, vecindades y grafos | Kaggle CPU | Arrays y recibos con hashes |
| Extraer características visuales que falten | GPU de la laptop, un detector congelado a la vez, si supera la prueba de memoria | Características FP16 por centro y video |
| Entrenar el selector de asociaciones y divisiones | GPU de la laptop | Checkpoints reanudables, historial y mejor modelo |
| Puntuar candidatos de calibración y evaluación | Laptop; CPU como alternativa si el tiempo medido lo permite | Tablas compactas de eventos y puntuaciones |
| Elegir umbrales, resolver el grafo temporal y validar el CSV | Kaggle CPU | Calibración, predicciones y recibos del solver |
| Calcular la métrica oficial y comparar con el control | Kaggle CPU | Resultados emparejados en los 48 videos |

**Flujo:** Kaggle CPU → paquetes por video → laptop → características/checkpoints/puntuaciones → Kaggle CPU → grafo final, CSV y evaluación.

Los paquetes de trabajo y modelos se conservarán como salidas guardadas o datasets privados de Kaggle y en carpetas locales ignoradas por Git. El repositorio contendrá código, configuraciones y resultados agregados. No contendrá imágenes, coordenadas anotadas, pesos ni credenciales. Cada paquete tendrá versión, partición, identidad del detector y hashes de arrays, incluida la correspondencia exacta entre características y orden de centros.

## Orden de ejecución

1. **Recuperar el trabajo útil y cerrar el control en CPU.** Ya se recuperó el CSV de E013-control: su inferencia terminó y el fallo se produjo al validar una coordenada Z fuera del volumen. Auditar la causa y corregir la exportación mediante una regla documentada, conservando el CSV original y registrando la nueva versión. Evaluar después con la métrica oficial, sin repetir automáticamente las 2,5 horas de inferencia. No basta con recortar la coordenada silenciosamente. El candidato E013 tenía estado RUNNING en la última consulta del 15 de septiembre a las 17:44 UTC; se recuperarán sus salidas cuando Arturo avise. Esta recuperación no bloquea la adaptación local y no se crea un monitor. [Revisión existente](../results/E013_interruption_review.json).

2. **Habilitar un entorno CUDA independiente para Biohub.** Comprobar espacio, compatibilidad entre driver y distribución de PyTorch y ejecución real de una operación en la RTX 3050. No asumir que el driver 529.04 admite cualquier wheel reciente. Registrar versiones y memoria. La instalación o una eventual actualización del driver son trabajo pendiente.

3. **Separar el ejecutor por etapas y exportar paquetes portables.** Sustituir las rutas obligatorias de Kaggle por argumentos. Separar preparación, características, etiquetas, entrenamiento, puntuación, solver y evaluación. Los recibos de caché distinguirán identidad científica de las rutas locales, para reutilizar archivos entre Windows y Kaggle. Los nuevos notebooks CPU tendrán `enable_gpu=false` y `enable_tpu=false`, y una inicialización que funcione realmente sin CUDA; cambiar solo el metadato del notebook E013 actual no basta.

4. **Completar únicamente las entradas que falten.** Verificar primero qué características, detecciones y pesos pueden recuperarse de ejecuciones previas. Para los faltantes, transferir un video o bloque necesario a la vez. La extracción local cargará un detector congelado por turno y guardará su salida antes de cargar el siguiente. Si una ventana de imagen no cabe, usar CPU para esa extracción o desarrollar una partición espacial cuya equivalencia se compruebe. No cambiar silenciosamente resolución, normalización o contexto del detector. Esta etapa puede ser el mayor coste de tiempo si no se recuperan las características de E013.

5. **Adaptar memoria y reanudación, y medir capacidad.** El entrenamiento actual carga todos los videos de ajuste/calibración en RAM, y la puntuación mueve grafos completos a GPU. Cambiar a lectura diferida por video y arrays que admitan lectura parcial; empezar en Windows con `num_workers=0`. Hacer configurable el bloque de atención, comenzando la prueba con 256 centros por bloque. Conservar las ventanas de diez fotogramas y las cuatro capas. E013 ya usa precisión mixta: se mantiene, no se presenta como una mejora nueva. Si hace falta, recomputar activaciones durante el backward mediante checkpointing, que intercambia memoria por cómputo. [Documentación de AMP](https://docs.pytorch.org/docs/2.14/amp.html), [checkpointing de activaciones](https://docs.pytorch.org/docs/2.14/checkpoint.html).

   Comprobar una secuencia de 100 pasos y el caso denso elegido por número de centros, sin usar sus anotaciones de evaluación. Registrar tiempo por paso, RAM, VRAM asignada/reservada y ocupación del dispositivo. Verificar también la puntuación del video más grande: puntuar solo por ventanas sin conservar los vecinos cambiaría el modelo. Si se divide un grafo, cada bloque debe incluir las dependencias de sus cuatro capas de atención y producir resultados equivalentes al cálculo completo dentro de una tolerancia fijada. No se excluyen videos densos para hacer pasar la prueba. Esta prueba decide capacidad y configuración de ejecución; no es un nuevo experimento de calidad.

6. **Entrenar el experimento completo y evaluar en CPU.** Mantener **48 videos de ajuste, 16 de calibración y 48 de evaluación**, y el presupuesto inicial de 6.000 pasos del experimento. El calendario de muestreo no dependerá de qué videos estén en caché. Medir el tiempo de cada etapa y dividir los trabajos de Kaggle por videos completos, con margen frente al límite vigente de la sesión. Usar una sola tarea pesada a la vez en la laptop, conectada a corriente. Calibrar únicamente con los 16 videos; fijar checkpoint, umbrales y predicciones antes de consultar las anotaciones de evaluación.

## Guardado que permita continuar

Los `best.pt` y `last.pt` actuales guardan pesos y metadatos, pero no todo el estado de entrenamiento. Podrán servir como inicialización si se recuperan; no se llamará a eso una reanudación exacta. La propia guía de PyTorch requiere guardar también el estado del optimizador para retomar entrenamiento. [Guardar y cargar modelos](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).

La migración añadirá un checkpoint con modelo, optimizador, scheduler, scaler de precisión mixta, paso global, generadores aleatorios, estado del muestreador y hashes de configuración/partición. Se guardará al terminar un paso, cada 100 pasos o aproximadamente cinco minutos, mediante escritura temporal y sustitución atómica, conservando una copia válida anterior. Una prueba comparará entrenamiento continuo con entrenamiento interrumpido y reanudado antes de iniciar la ejecución larga.

Cada video terminado tendrá salida y recibo verificable. Las sesiones posteriores omitirán solo resultados completos compatibles. En Kaggle, guardar un archivo en `/kaggle/working` no equivale a confirmar su persistencia externa: hay que guardar/publicar las salidas o versionar el dataset privado y verificar su disponibilidad antes de depender de ellas. La caché local solo se rotará después de comprobar otra copia íntegra.

## Criterios para avanzar

- **Infraestructura lista:** CUDA verificado, memoria dentro del presupuesto, datos portables, cálculo por bloques comprobado y reanudación probada. Hasta entonces no se afirma que el modelo completo quepa en la laptop.
- **Experimento comparable:** mismos candidatos, particiones, pesos de detectores y reglas de supervisión. Se mantiene el orden de congelar grafos/características antes de leer anotaciones. Los centros sin correspondencia conocida no se convierten en negativos. Si una optimización altera las detecciones o el modelo, se registra como otra versión científica.
- **Evaluación interpretable:** CSV válido y métrica oficial sobre el mismo conjunto que el control reparado. Registrar tiempos, estados y alternativas del solver; sus límites temporales pueden cambiar la solución al cambiar de CPU. Los 48 videos siguen siendo desarrollo condicional y no una validación independiente del detector público.
- **Submission viable:** verificar por separado la inferencia final en Kaggle CPU y los requisitos vigentes de la competición. El entrenamiento local no garantiza que la generación de detecciones del test quepa en el tiempo permitido. No se asumirá que un CSV calculado localmente sobre videos visibles sustituye la inferencia que exija Kaggle sobre otros datos.

El primer resultado útil de esta estrategia será una evaluación del control recuperado y un entrenamiento local que pueda continuar tras una interrupción. Después se decidirá la promoción del selector con evidencia del grafo final. El mejor score público propio registrado sigue siendo **0.946**; aún no hay evidencia de mejora de E013.
