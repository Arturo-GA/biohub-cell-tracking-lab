# E061: detector complementario adaptado a imágenes reales

## Hipótesis y diferencia respecto de pruebas anteriores

E058 encontró 67 enlaces anotados fallidos sin al menos un extremo emparejado.
E059–E060 no consiguieron una mejora mezclando asociaciones. Los puentes cortos
de detectores sin adaptar ya se habían probado. E061 ajusta un detector propio
preentrenado con NIS3D, incorpora tres fotogramas y admite trayectorias nuevas
persistentes, además de completar huecos y prolongar extremos.

## Entrenamiento ejecutado

Se reutilizan los 40 videos de ajuste de E039, con 128 recortes por video:
5.120 ejemplos de tres fotogramas y 32³ vóxeles isotrópicos tras XY4. Los 24
videos de evaluación no pertenecen a ese conjunto: los 16 de E031 y los ocho
de E023. Los conjuntos de evaluación ya se usaron en el proyecto; no son un
holdout independiente. La separación actual evita entrenar este detector con
ellos, pero no borra su uso anterior en decisiones de investigación.

La primera convolución se amplía a tres canales conservando los pesos del
fotograma central y poniendo a cero los laterales. La prueba local verifica
que inicialmente produce el mismo mapa que el detector preentrenado.

Objetivo: centros anotados con mapa gaussiano >0,3, mayor peso si el detector
denso congelado los reconoce mal, conservación suave de sus probabilidades
fuera de vecindarios anotados y replay de recortes NIS3D con máscaras válidas.
La conservación es un prior del profesor, no una etiqueta verdadera de fondo.
No se convierte automáticamente una célula no anotada en ejemplo negativo.
La dificultad se mide respecto del detector denso, no de predicciones Harmonic
en los 40 videos de ajuste.

AdamW 1e-4, lote de cuatro ejemplos Biohub y cuatro externos, 2.500 intentos
fijados antes de evaluar. AMP realizó 2.495 actualizaciones efectivas. Cambio L2
de pesos 2,888786 y norma de pesos laterales 0,071226. Entrenamiento: 129,55 s
de proceso. Pesos descargados, hash verificado, carga estricta y valores finitos.
No se seleccionó checkpoint ni duración con los resultados de los 24 videos.

## Inferencia y ensamble

Se procesan todos los fotogramas de los 24 videos con tres condiciones:
detector congelado, adaptado con contexto real y adaptado repitiendo el
fotograma central. El último controla la aportación de pasado/futuro.
Máximos locales 3³, probabilidad >=0,2, máximo 512 candidatos por fotograma.
Descriptores de intensidad local 3³ centrados y normalizados para consistencia.

La integración CPU exige vecinos mutuos a <=6 µm, margen de 0,75 µm frente al
segundo candidato en ambas direcciones, similitud local >=0,75 y persistencia
de al menos cinco fotogramas. Se mapea a nodos originales solo con cercanía
<=3 µm y margen de 0,75 µm. Las correspondencias cercanas ambiguas se rechazan,
no se interpretan como nuevas células. Aceleración máxima 3 µm/fotograma².

Los segmentos con anclas requieren probabilidad mediana >=0,5 y mínima >=0,2;
una prolongación con solo un ancla requiere tres detecciones nuevas. Las
trayectorias sin anclas requieren ocho fotogramas, mediana >=0,8 y mínima >=0,5.
Presupuesto máximo de nodos nuevos: 1% del grafo original de cada video. Se
priorizan segmentos con anclas y mejor confianza/coherencia. Se conservan todos
los nodos, enlaces y divisiones originales, sin ocupar un extremo ya enlazado.

Comparaciones fijadas: control, congelado con anclas, congelado con trayectorias
nuevas, adaptado con anclas, adaptado con trayectorias nuevas y fotograma repetido
con trayectorias nuevas. Métrica oficial sobre los CSV enteros congelados antes
de consultar etiquetas. Controles exigidos: 0,9007529595605399 en los 16 videos
y 0,9502832669356378 en los ocho. Promoción: mejora media >=0,001, enlaces TP
netos positivos, sin aumentar divisiones FP ni perder score en ningún conjunto
o embrión. No se realiza submission automática.

Seis tests locales pasan: inicialización equivalente, recortes y objetivos,
conservación del grafo, rechazo por apariencia incompatible, extremos ocupados
y soporte necesario para trayectorias sin anclas. GPU para entrenamiento e
inferencia neuronal; preparación, asignación y métricas en CPU. Los tiempos
del script no equivalen a cuota GPU facturada.

## Resultado E061

Inferencia completa: 140,15 s de proceso; entrenamiento + inferencia: 269,70 s.
El control combinado de 24 videos es 0,919878413. Su score no es una media
simple de los scores de los conjuntos: la métrica agrupa también divisiones.
Ninguna integración pasa el criterio fijado. Congelado con anclas: 0,919806390;
con trayectorias nuevas: 0,919782510. Adaptado, ambas opciones: 0,919314225;
fotograma repetido: 0,919316641. El adaptado pierde dos TP y añade cinco FP.
No se envía submission.

Diagnóstico geométrico uno a uno sobre 12.417 ocurrencias anotadas (no mide
precisión ni reproduce por sí mismo la métrica del grafo):

| Candidatos completos | Cubiertos a 3 µm | A 7 µm | Omitidos por la base recuperados a 7 µm |
| --- | ---: | ---: | ---: |
| Denso congelado | 7.512 | 11.690 | 190 |
| Adaptado, contexto real | 8.024 | 11.741 | 193 |
| Adaptado, fotograma repetido | 8.032 | 11.738 | 192 |

El ajuste mejora la localización respecto del denso original. El contexto
temporal no demuestra una ventaja relevante frente a repetir el centro.
Con presupuesto por fotograma limitado al número de nodos de la base, la
cobertura adaptada a 7 µm baja a 8.938, frente a 9.315 del congelado: todavía
hay un problema de ranking de candidatos. Después del filtro de persistencia
E061, solo quedan 4 de las 193 ocurrencias complementarias del adaptado.

## E062: asociación neuronal de los candidatos nuevos

Se prepara la unión de nodos originales y candidatos adaptados a más de 3 µm.
Se evalúa el checkpoint principal público de UNet/Transformer, con hash
`12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771` y carga
estricta de 2.076.706 parámetros. Probabilidades forward en FP32 y candidatos
top8 por fila/columna. No se afirma reproducir la fusión completa Harmonic,
que además usa otro modelo, asociaciones inversas y TTA.

Las asociaciones se asignan globalmente por par de fotogramas con utilidad
log(p/0,05) y distancia máxima 14 µm. Se extraen trayectorias de al menos cinco
fotogramas y se aplican los mismos límites de adición de E061. Sustituyen el
filtro duro de vecinos/intensidad; los nodos originales se identifican mediante
un registro exacto, no por IDs supuestos. Se conserva el grafo original y se
comparan integración con anclas y con trayectorias nuevas, sin barrer umbrales.
El criterio de promoción continúa siendo el de E061. Siete tests locales pasan.

## Resultado E062 y cierre del lote

La asociación neuronal termina en los 24 videos: 260,24 s de proceso GPU.
Ambas integraciones obtienen 0,919412668 frente a 0,919878413 del control.
Enlaces: 11.391 TP / 644 FP / 649 FN, frente a 11.389 / 638 / 651. Se recuperan
dos TP netos, pero se añaden seis FP. No se recuperan divisiones adicionales.
Ninguna trayectoria sin anclas pasa los filtros, por lo que las dos opciones
producen el mismo resultado. El grupo de ocho cae de 0,950283267 a 0,949066222;
el de 16 queda prácticamente igual, ligeramente peor. No se promueve.

La mejora de cobertura del detector no se traduce todavía en una selección de
trayectorias útil. La señal conservada es localización más precisa respecto
del denso original; el control de fotograma repetido no respalda atribuirla al
contexto temporal. Estos resultados tampoco demuestran que las 193 ocurrencias
complementarias puedan recuperarse juntas sin errores: ese conteo es diagnóstico.

Dos experimentos completos, siete variantes de integración, siete tests
aprobados. Entrenamiento e inferencia suman 529,94 s (8,83 min) de proceso GPU;
la cuota facturada puede diferir por aceleradores simultáneos y arranque.
Preparación, asignación, diagnóstico y métricas se ejecutaron en CPU. Pesos,
manifiestos, resultados y protocolos guardados. No hay nueva submission ni
ejecuciones pendientes. Leaderboard confirmado del proyecto: 0,946.

Recibo resumido: `results/E061_E062_decision.json`. La siguiente hipótesis sería
aprender a seleccionar trayectorias complementarias con supervisión separada;
no se presenta como implementación ni como mejora ya conseguida.
