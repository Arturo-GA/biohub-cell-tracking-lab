# E029: primera integración del detector complementario

Después de demostrar recuperación de 14 centros omitidos en E027, se prueba el efecto sobre el grafo, sin más inferencia ni entrenamiento. Este experimento usa la laptop para las comprobaciones y Kaggle CPU para evaluar el CSV con la métrica oficial fijada.

Regla única, sin barrido de parámetros: redondear centroides antes de construir el grafo; excluir candidatos a 7 µm o menos de cualquier nodo de Harmonic en ese fotograma; enlazar únicamente vecinos más cercanos mutuos entre fotogramas consecutivos a no más de 5 µm; conservar secuencias de al menos dos fotogramas. No se inventan enlaces al control ni divisiones. Todos los nodos, coordenadas y aristas originales se conservan. Las secuencias añadidas son propuestas independientes, no identidades verificadas.

Dos comprobaciones locales pasaron: conservación del control con exclusión de candidatos cercanos y aislados; enlaces uno a uno sin saltos a través de fotogramas ausentes. Se ejecutaron directamente las funciones de prueba porque pytest no está instalado. Los 28 archivos del paquete se compilaron al construir el notebook.

Se exporta y valida un CSV entero de los 16 videos de calibración, se fija su hash y recién después se ejecuta la métrica oficial. El problema de selección sigue presente: las predicciones nuevas solo están disponibles en ventanas elegidas por errores de las anotaciones. Por ello **un resultado favorable no sería validación independiente ni habilitaría un submission**. Esta prueba sirve para rechazar una integración que falla incluso en esos casos y para medir cambios en nodos, enlaces y divisiones antes de una ejecución más cara.

Conservar las aristas del control no garantiza conservar su score: nuevos nodos pueden cambiar el emparejamiento con anotaciones y, con ello, las identidades evaluadas. Ese riesgo es precisamente parte de lo que mide E029.

## Resultado y decisión

El experimento terminó en 46,47 segundos CPU. Añadió 217 nodos y 115 aristas. En la métrica oficial del CSV entero, el control obtuvo 0,9007529596 y el candidato 0,9008049190: **+0,0000519595**. Hubo un enlace verdadero adicional, cero falsos positivos medidos adicionales y una omisión de enlace menos. Las divisiones permanecieron en TP/FP/FN = 3/4/3. Las aristas añadidas sin anotación no se consideran automáticamente correctas.

Se verificaron los 28 archivos ejecutados, el hash del CSV descargado, los recuentos de nodos por video y el agregado de la métrica. Recibo: `results/E029_completed.json`. No es una confirmación independiente: solo se añadieron propuestas dentro de ventanas escogidas por errores conocidos del control. La ganancia resulta insuficiente incluso en ese diagnóstico favorable; no se lanza un submission ni una expansión GPU de esta regla.

E027/E028 establecen que hay evidencia de imagen complementaria; E029 muestra que una unión geométrica de trayectorias no la convierte en una mejora competitiva suficiente. La siguiente dirección debe resolver conjuntamente las identidades entre fuentes y los enlaces temporales con apariencia de imagen, en vez de barrer radios de exclusión o sumar más nodos. Una opción concreta por implementar es extraer tokens del encoder DINO en recortes de núcleos y aprender asociaciones/divisiones con los videos de entrenamiento separados. **No usar el vector `styles` de Cellpose-DINO como embedding: el código oficial devuelve allí un vector aleatorio de compatibilidad.** El encoder debe instrumentarse explícitamente. Esa nueva etapa aún no está entrenada ni validada.

Al cerrar E029 no quedan jobs pendientes. Todas las ejecuciones cortas y sus evaluaciones fueron seguidas durante la sesión, sin solicitar avisos del usuario. El leaderboard sigue en 0,946; ninguno de estos diagnósticos se presenta como avance de puntuación pública.
