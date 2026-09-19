# E020/E021: recuperar alternativas visuales antes de decidir trayectorias

Protocolo del 19 de septiembre de 2026. Experimentos CPU, sin entrenamiento nuevo ni submission automático de candidatos inferiores. El objetivo autorizado es continuar hasta disponer de un envío defendible.

E019 perdió contra Harmonic: añadir trayectorias geométricas no aportó suficiente precisión. E020 reutiliza las probabilidades del GEFF de Harmonic; E021 vuelve a ejecutar el Transformer de asociación público sobre las características visuales congeladas de E013. El código de inferencia original filtra candidatos por probabilidad y limita padres/hijos de manera voraz antes del ILP. Por eso el GEFF puede haber perdido alternativas necesarias para resolver conflictos globalmente.

Ambos experimentos conservan las detecciones del CSV completo de Harmonic y fijan sus divisiones y enlaces sin evidencia visual recuperable. Comparan dos decodificaciones: asignación bipartita por probabilidades y asignación con energía de continuidad entre tres fotogramas. La segunda usa descenso por bloques temporales, acepta únicamente mejoras de su energía completa y permite apariciones/desapariciones. No se afirman óptimos globales de esa energía no convexa.

Se fijaron antes de evaluar: costo de aparición equivalente a probabilidad 0,05, peso temporal 0,75, penalización de cambio de velocidad dividida por 3 µm y truncada a 4, seis pasadas máximas. No se realiza un barrido para rescatar resultados negativos. Las dos predicciones se congelan antes de leer la evaluación oficial de los 16 videos de calibración, ya reutilizados y por tanto no independientes.

Las dos primeras ejecuciones E020 abortaron en comprobaciones de correspondencia, antes de generar métricas. La versión 3 usa coincidencia única por fotograma y coordenadas redondeadas; excluye ambigüedades en ambos lados. Terminó sin cambiar score (0,90075295956): la variante visual añadió tres enlaces y la temporal no cambió ninguno. La cobertura de esa correspondencia fue baja: el suavizado final de Harmonic cambia las posiciones. La discrepancia inicial de coordenadas no demostraba que los identificadores fueran distintos; esa fue una interpretación prematura. Se debe leer el GEFF con el mismo lector IndexedRXGraph del exportador para recuperar identidades anteriores al suavizado, conservando los nodos nuevos de reparación aparte. Los notebooks anteriores se conservan. Estos fallos son de integración, no resultados científicos sobre el valor de la evidencia visual completa.

E021 carga exclusivamente el Transformer del checkpoint secundario público verificado mediante SHA256. Une detecciones a características primarias mediante vecinos mutuos a como máximo 1 µm, aplica posición sinusoidal y asociación bidireccional y conserva todas las alternativas dentro de 20 µm para el optimizador. Es una aproximación: E013 interpoló características de ventanas vistas por primera vez; Harmonic original usa muestreo entero y ventanas de cada pareja. Una mejora aquí exigiría validar después la implementación de inferencia completa. No se presenta como reproducción exacta, entrenamiento nuevo ni evidencia independiente del entrenamiento de los modelos públicos.

## Investigación revisada

- [Ultrack, Nature Methods 2025](https://doi.org/10.1038/s41592-025-02778-0): selección conjunta de hipótesis y seguimiento; referencia conceptual para evitar decisiones irreversibles prematuras.
- [Trackastra, ECCV 2024](https://arxiv.org/abs/2405.15700): asociación mediante atención temporal. E021 reutiliza el Transformer público de Harmonic, no implementa Trackastra.
- [FOCUS-3D](https://github.com/yu-lab-vt/FOCUS-3D): segmentación volumétrica con pesos públicos; opción para obtener nueva evidencia de imagen. No se ha ejecutado ni comprobado su precisión/tiempo en estos datos.
- [JunhaoLiXD/Biohub_Cell_Tracking](https://github.com/JunhaoLiXD/Biohub_Cell_Tracking): documenta otra línea basada en modelos públicos similares y mejoras locales que no siempre se trasladan al leaderboard. Sus puntuaciones son declaraciones de ese proyecto, no resultados nuestros.
- [matt-ceran/biohub-cell-tracking](https://github.com/matt-ceran/biohub-cell-tracking): registra dificultades de cobertura y clasificación de divisiones; no proporciona evidencia para sustituir nuestro control por su pipeline.

GPU consumida por E020/E021: ninguna. Código, configuraciones y recibos pueden ir al repositorio privado; características, pesos y predicciones detalladas permanecen fuera de Git.

## E022: evidencia visual exacta

Se lanzó una captura de dos videos elegidos por nombre (primero de cada espécimen) para verificar la inferencia completa. Usa GPU solamente para las redes volumétricas de Harmonic, con sus pesos, TTA y configuración originales. La instrumentación guarda, sin modificar la matriz de probabilidades ni la selección original, la unión de las ocho mejores alternativas por madre y por hija. El resultado original sigue generándose como control.

La fase CPU lee las identidades del GEFF con el mismo lector del exportador; relaciona las coordenadas capturadas con ese grafo antes del suavizado y conserva las coordenadas finales del CSV para comparar únicamente la asociación. Produce las dos variantes con el mismo optimizador y parámetros E020. Se congela todo antes de evaluar. Dos videos son una comprobación funcional, no evidencia suficiente para presentar una mejora general ni autorizar por sí solos un envío competitivo. Los resultados positivos requerirán ampliar la comparación y reproducir la inferencia final.
