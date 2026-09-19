# E017: movimiento del tejido y optimización de trayectorias

19 de septiembre de 2026. Dos ejecuciones privadas separadas: nuevo seguimiento en CPU y reproducción de Harmonic completo con GPU. No hay nueva búsqueda de umbrales ni extensión de E016.

## Por qué cambiar de enfoque

E016 aprendía a clasificar padres con características de dos detecciones. Su mejora de +0.00096 frente al control geométrico, con 12 de 16 videos peores, no justifica repetir ese entrenamiento. Ahora se prueba evidencia de movimiento colectivo y coherencia de trayectorias más largas. Es una hipótesis de arquitectura diferente para este proyecto, no una afirmación de originalidad científica ni una garantía de podio.

El 0.78974 de E016 es una métrica local en 16 videos. No puede compararse directamente con puntuaciones públicas de Kaggle. La referencia pública propia sigue siendo 0.946; la comparación con Harmonic completo en la misma cohorte todavía faltaba.

## Investigación aplicada

- [Ultrack, Nature Methods 2025](https://doi.org/10.1038/s41592-025-02778-0): describe registro temporal para compensar movimiento, incluyendo desplazamientos localmente coherentes en tejidos deformables. Aplicación propia: estimar un campo de movimiento a partir de detecciones y usar desplazamientos residuales para asociar células. No se ejecuta ni reproduce Ultrack completo.
- [Cell Tracking According to Biological Needs, IEEE TMI 2025](https://ieeexplore.ieee.org/document/11051031), [código de autores](https://github.com/TimoK93/BiologicalNeeds): utiliza incertidumbre de movimiento y múltiples hipótesis, con soporte de mitosis, para resolver inconsistencias temporales. Inspira mantener alternativas y revisar decisiones con contexto posterior. E017 no implementa su modelo probabilístico, sus distribuciones de mitosis ni el MHT completo.
- [Trackastra, ECCV 2024](https://arxiv.org/abs/2405.15700): contextualiza asociaciones dentro de ventanas temporales. Refuerza investigar información más amplia que un par aislado. E017 no añade otro transformer ni reutiliza sus pesos: primero mide el valor de la dinámica temporal con las detecciones disponibles.

## Método implementado

1. Estimar desplazamiento del tejido por frame a partir de vecinos mutuamente más cercanos, con radio máximo de 8 µm. Para cada detección, calcular la mediana de hasta 16 vectores próximos dentro de 30 µm; si no hay evidencia local, usar la mediana del frame. Los frames sin semillas mantienen flujo cero. Estas semillas pueden contener errores, por lo que no se tratan como GT.
2. Crear candidatos tanto alrededor de la posición original como de la posición compensada. Conservar hasta 12 vecinos en cada consulta, dentro de 20 µm, más las asociaciones del control inicial.
3. Optimizar un objetivo de trayectoria: recompensa de enlaces, distancia residual tras compensar el flujo y penalizaciones por cambios de velocidad residual entre enlaces consecutivos y separados por un enlace intermedio. Todo se expresa en unidades físicas; no usa nombres de videos ni anotaciones para decidir.
4. Alternar asignación bipartita por frame hacia adelante y hacia atrás, hasta cuatro barridos, admitiendo quedar sin asociación. Aceptar únicamente actualizaciones que no empeoren el objetivo declarado de toda la secuencia.
5. Ejecutar desde dos inicializaciones, geométrica y compensada por flujo. Elegir por el objetivo sin etiquetas. No elegir por score de calibración. Es descenso por bloques con dos inicios; no certifica el óptimo global y puede quedar atrapado en una solución local.

La implementación incorpora continuidad temporal real en el objetivo; no es un ajuste de radio sobre E016. Conserva las detecciones primarias ya guardadas y estudia continuaciones. Aún no incluye divisiones ni recupera células ausentes del detector. Esas limitaciones siguen siendo importantes para alcanzar un sistema completo competitivo.

## Ejecuciones

### E017 CPU

[Biohub Lab Tissue Trajectory CPU](https://www.kaggle.com/code/jarturo/biohub-lab-tissue-trajectory-cpu), versión 1. Reutiliza outputs E016, sin inferencia de imágenes ni entrenamiento. Trabaja sobre los mismos 16 videos de calibración; reconstruye y verifica el CSV del control por hash. Congela el nuevo CSV antes de acceder a anotaciones y ejecuta la métrica oficial fijada. No toca los 48 videos de evaluación.

Salidas: `tissue_trajectory/result.json`, `trajectory_reports.json`, `frozen_predictions.json`, `candidate_metrics.json` y predicciones por video.

### Control Harmonic completo

[Biohub Lab Harmonic Calibration Control](https://www.kaggle.com/code/jarturo/biohub-lab-harmonic-calibration-control), versión 1. Reproduce la fuente congelada completa y sus modelos, TTA, divisiones y postprocesamiento en esos mismos 16 videos. El directorio de entrada de inferencia contiene solo enlaces a imágenes Zarr, sin GEFF. Los detectores completos de este baseline no están representados por el CSV geométrico ni pueden reconstruirse fielmente solo con los features reducidos de E013.

La GPU se usa para la inferencia neuronal volumétrica requerida por este pipeline, junto con su postprocesamiento integrado. No se añade entrenamiento ni evaluación oficial a esta sesión GPU. Guarda el CSV congelado en `calibration_control/harmonic_control/submission.csv`. La validación de exportación y evaluación oficial se harán en CPU cuando esté disponible; si aparece nuevamente redondeo de coordenadas al límite superior, preservar el original y documentar una corrección explícita como en E013. No efectuar recortes generales silenciosos.

## Verificación y criterio de decisión

Cuatro pruebas superadas: flujo de traslación colectivo, preservación de trayectorias coherentes y capacidades, corrección de un intercambio sintético, y equivalencia entre la utilidad local de quitar un enlace y el cambio del objetivo completo. El último test comprueba que las penalizaciones temporales se contabilizan coherentemente. Fuentes y notebooks compilados; payloads y metadatos registrados en `results/E017_preflight.json`.

Las pruebas sintéticas no demuestran rendimiento en Biohub. Al completar ambos notebooks, auditar sus outputs, evaluar el control congelado en CPU y comparar por video los tres sistemas pertinentes: geometría E016, nueva trayectoria E017 y Harmonic completo. Un mejor objetivo interno no equivale a mejor score. No promover E017 si solo supera al control débil; para reemplazar el baseline necesita mejorar la comparación completa, incluidas divisiones y cobertura.

No se configuró monitoreo automático ni envío al leaderboard. Código y recibos en GitHub privado; imágenes, anotaciones, pesos y predicciones detalladas permanecen fuera de Git.
