# E027: detección en fallos conocidos de Harmonic

La corrección de normalización E026 aumentó la cobertura del nuevo segmentador de 10/39 a 28/39 centros. Sin embargo, Harmonic ya cubría los 39. E027 cambia la pregunta: ¿el segmentador recupera centros anotados que el control omite?

Se congeló una regla de selección antes de la inferencia: entre los 16 videos de calibración, escoger por video un fotograma con centros anotados no emparejados por Harmonic a 7 µm. Priorizar los que incluyen familiares de una división, luego el número de omisiones y después el fotograma más temprano. Evaluar imágenes completas de ese instante y sus vecinos inmediatos, sin recortes alrededor de etiquetas. Esto es un diagnóstico condicionado a errores conocidos, no una estimación imparcial del rendimiento ni una reserva nueva.

La preparación CPU terminó en 17,41 segundos y se verificaron sus 36 archivos. Seleccionó 33 fotogramas de 12 videos; cuatro no tenían omisiones. Los fotogramas centrales reúnen 21 centros omitidos y uno está vinculado directamente a una división. La escasa representación de omisiones en divisiones limita las conclusiones sobre mitosis. CSV del control fijado por SHA256 y misma escala física y correspondencia uno a uno que E024–E026. Recibo: `results/E027_INPUTS_completed.json`.

La inferencia reutiliza el checkpoint final E025 y la normalización E026. No hay optimizador, búsqueda de umbrales ni etiquetas en el predictor. El agrupamiento y la correspondencia de referencia siguen en CPU con el mismo paquete de evaluación que E026. GPU solo ejecuta la red 3D. Los resultados se siguen durante la sesión y la evaluación se encadena sin pedir un aviso del usuario.

Después se examinan en la laptop las distancias de los centros recuperados a las detecciones existentes y a candidatos de fotogramas vecinos. Esto distingue señales de localización y posibles detecciones complementarias, pero la proximidad por sí sola no demuestra identidad celular, aristas correctas ni precisión: las anotaciones son parciales. No se sumarán automáticamente todas las instancias al grafo. La promoción requiere un predictor sin etiquetas, un grafo completo y mejora con la métrica oficial en otros videos reservados.

## Resultado completado

Harmonic cubrió 163 de 203 centros anotados en las ventanas seleccionadas. NucVerse cubrió 132, incluyendo **14 de los 40 que Harmonic omitía**. Son ocurrencias por fotograma, no 14 células distintas ni 14 trayectorias recuperadas. El 35 % corresponde exclusivamente a esta muestra condicionada a errores. Se produjeron 5.208 instancias NucVerse frente a 7.454 nodos del control; los centros sin referencia no se clasifican automáticamente como falsos positivos.

La unión geométrica tiene cobertura 178, un límite diagnóstico sin grafo. No es idéntica a 163+14: el matching conjunto puede reasignar centros en zonas ambiguas y cubrir un punto adicional. No debe presentarse como una puntuación competitiva ni como una fusión validada. NucVerse omite 45 centros que Harmonic sí cubre, por lo que no se promueve como reemplazo.

La laptop descargó solo 66 arrays pequeños, 184.682 bytes, sin imágenes ni pesos. La comparación emparejada reprodujo las 14 recuperaciones: seis están a más de 7 µm de cualquier centro del control; ninguna está a 2 µm o menos; ocho tienen un candidato a menos de 5 µm en un fotograma adyacente. Esa continuidad geométrica es una señal a estudiar, no una verificación de identidad. El frame 62 de la ventana de división no tiene recuperación exclusiva; el frame 63 sí tiene una. No se ha demostrado rescatar la división completa.

Inferencia: 223,91 segundos de proceso GPU. Agrupamiento/evaluación: 755,48 segundos CPU. Se verificaron ocho archivos de inferencia y 25 de evaluación, los pesos congelados, los hashes de imágenes/campos y las predicciones anteriores a la lectura de referencias. Recibos: `results/E027_completed.json` y `results/E027_recovery_diagnostic.json`.

Decisión: conservar NucVerse como fuente complementaria y estudiar una integración que compita entre conservar, corregir o añadir detecciones con evidencia temporal. No ampliar entrenamiento, sustituir el control ni enviar la unión de centros. E028 prueba otra arquitectura sobre dos de estas mismas imágenes antes de elegir el siguiente desarrollo.
