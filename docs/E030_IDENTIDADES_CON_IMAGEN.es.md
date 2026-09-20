# E030: identidad y linaje con apariencia

Implementación de la propuesta posterior a E029. La unión por cercanía solo añadió un enlace correcto; ahora se aprende la identidad visual del núcleo, la selección del progenitor y la compatibilidad de una división completa.

## Implementación

- `physical_views`: secciones XY, XZ e YZ de 64×64, muestreadas a 0,40625 µm/píxel. Se respeta la separación axial de 1,625 µm y se normaliza cada volumen por percentiles 2/99,8.
- `extract_tokens`: captura la entrada de la capa `out` de Cellpose-DINO ViT-B, después de la normalización del encoder y de retirar CLS/storage. Concatena la media de los cuatro tokens centrales y la media del entorno en cada vista: 4608 características. Reutiliza exactamente los pesos y dependencias fijados en E028; comprueba repetibilidad. **No usa `styles`**, que es aleatorio en ese modelo.
- `ImageIdentityModel`: proyección compartida a 96 dimensiones, clasificación de identidad entre dos detecciones, competencia entre progenitores con opción ausente, y cabeza de división simétrica respecto de las dos hijas. Se usan desplazamientos físicos junto con la apariencia.
- `decode_events`: optimización conjunta entera. Cada madre puede continuar o dividirse en dos; cada hija tiene como máximo un progenitor. Identidades duplicadas de un mismo fotograma compiten por activarse. Se comprueba la factibilidad del resultado.
- `infer_lineage`: acepta detecciones de distintas fuentes y sus descriptores sin etiquetas. Exige proporcionar los puntos de operación de división e identidad; no los inventa a partir de este diagnóstico.

Referencias: [DINOv3](https://arxiv.org/abs/2508.10104) y [implementación oficial Cellpose](https://github.com/MouseLand/cellpose/blob/a54cb48849b7e225a81e8e43dcb042d42427f543/cellpose/vit.py). Esta es una adaptación experimental, no un resultado publicado de esos autores.

## Entrenamiento y cómputo

1. Kaggle **CPU** prepara imágenes reales de los 48 videos fit y los 16 videos de calibración separados. Escoge ocho transiciones repartidas por video; únicamente en fit añade las transiciones de divisiones anotadas para no desperdiciar su escasa supervisión. No selecciona errores del control.
2. Cada centro tiene dos perturbaciones independientes de hasta ±1 µm por eje para aprender consistencia frente a errores de localización. Se generan progenitores cercanos y pares de hijas. Una arista es negativa solo si contradice un progenitor anotado. Una división sin evidencia suficiente queda desconocida; no se convierte en negativa por ausencia de etiqueta.
3. Kaggle **GPU** extrae una sola vez los descriptores del encoder congelado. Límite de 2400 segundos comprobado entre lotes; no hay optimizador ni decodificación de máscaras en GPU.
4. Kaggle **CPU** entrena 600 pasos fijos de las cabezas pequeñas. Entrena también una ablación que recibe solo geometría, con el mismo protocolo. Elimina aleatoriamente algunos progenitores verdaderos solo durante fit para aprender la opción ausente. Congela ambos checkpoints antes de evaluar calibración.
5. Se comparan exactitud de progenitores, disponibilidad del progenitor verdadero, AP de divisiones e identidad, y un control que permuta los descriptores visuales. No hay selección de checkpoints por leaderboard.

## Qué permite concluir

Esta primera ejecución es un **currículo centrado en anotaciones con perturbación**, no una validación del detector. Los negativos son núcleos anotados próximos, no el conjunto completo de candidatos de Harmonic/NucVerse; la cabeza de identidad aún debe enfrentar duplicados y confusores reales. Un buen resultado aquí no prueba que el CSV mejore. Los 16 videos son calibración utilizada en investigación previa; tampoco constituyen un nuevo test ciego.

Antes de promover un submission deben fijarse puntos de operación usando detecciones reales de fit/desarrollo, comprobarse el grafo entero contra Harmonic con la métrica oficial, y confirmarse el resultado en videos reservados. La API de inferencia está implementada; esa validación completa no queda sustituida por la exactitud entre centros anotados.

## Verificación y estado

Siete pruebas locales pasaron (incluyen aprendizaje de identidad con distancias id�nticas y rechazo de saltos temporales): geometría física, extracción repetible independiente de `styles`, simetría de identidad/división, competencia conjunta de duplicados y divisiones, y sensibilidad/gradientes de la apariencia. Paquetes privados con código y configuración; imágenes, etiquetas y pesos quedan fuera de Git.

Preparación CPU lanzada; resultados pendientes. No hay nuevo submission.
