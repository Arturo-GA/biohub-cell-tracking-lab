# E033 — aprender la representación temporal desde imágenes 3D

Prueba solicitada después de E031/E032. Esta vez se entrenan las convoluciones del encoder, desde cero, con imágenes reales. No se reutilizan sus características congeladas como entrada al modelo nuevo.

## Datos y supervisión

Se reutilizan los 16 videos completos de E031 y sus candidatos `batch_bn`: 56.384 recortes únicos, 6.725 decisiones de progenitor y 6.700 decisiones con al menos un negativo permitido durante entrenamiento. El control de vecino más cercano sigue acertando 6.118/6.725; el conjunto de decisiones coincide en tamaño con E032.

Cada entrada tiene tres canales temporales: volumen anterior, actual y siguiente, recortados en la misma posición espacial. Son cubos de 16³ a 1,625 µm por voxel. Los bordes espaciales y temporales se replican. Se usa `log1p` de la imagen normalizada previamente con sus metadatos, sin ajustar estadísticas con las etiquetas evaluadas.

Se conserva el progenitor anotado representado por emparejamiento uno a uno. Las hipótesis cercanas cuya identidad es desconocida se excluyen de los negativos. Un vecino cercano con otra identidad anotada sí es negativo válido; esta es una corrección respecto del enmascarado demasiado general de E032. Los candidatos de evaluación siguen completos, con hasta 12 progenitores a 20 µm.

## Modelo y entrenamiento

Encoder propio: tres convoluciones 3D, canales 16/32/64, GroupNorm, activación SiLU, pooling global y vector normalizado de 64 dimensiones. La semejanza coseno define un objetivo contrastivo entre hijo y progenitores candidatos. La pérdida combina ese objetivo visual con clasificación del progenitor incorporando una preferencia geométrica fija `-distancia_um/5`. Temperatura 0,2, AdamW, tasa 0,0003, 1.000 iteraciones fijadas y lotes de 16 decisiones. Se registran las actualizaciones efectivas del optimizador por separado de las iteraciones, para detectar pasos omitidos por precisión mixta.

Cada encoder se entrena con ocho videos de un embrión y produce representaciones para los ocho videos del otro; luego se invierte. Se verifica y registra que los pesos de la primera convolución cambiaron. No se elige un checkpoint por aciertos de validación. Los pesos se guardan al completar cada dirección.

Preparación de recortes y evaluación: CPU. Entrenamiento e inferencia de representaciones: T4. El script GPU comprueba un presupuesto de 1.500 segundos entre pasos y lotes; carga/exportación y asignación del entorno pueden exceder ese tiempo de proceso. Sin Colab.

## Controles y decisión fijada antes de entrenar

Se comparan sobre los mismos candidatos: geometría, encoder aleatorio sin entrenamiento, encoder entrenado, representaciones entrenadas permutadas entre detecciones y semejanza visual sin geometría. La métrica de esta fase es elegir el progenitor, no el score del leaderboard.

Solo procede la integración y evaluación del grafo completo si el encoder entrenado supera al vecino más cercano por al menos 0,5 puntos porcentuales en total, no pierde en ninguno de los dos embriones y supera también al control de imágenes permutadas por 0,5 puntos. Superar este filtro no autoriza interpretar el ranking como mejora de submission: todavía habrá que medir el CSV completo.

La independencia se aplica al encoder nuevo y a sus etiquetas por embrión. Los candidatos proceden de un detector cuya lista de entrenamiento es desconocida, y estos videos ya se usaron como calibración en experimentos anteriores. Por ello no se presenta como un holdout nuevo ni como una predicción de score público.

## Verificación y estado inicial

Tres pruebas locales aprobadas: recortes y bordes temporales correctos, negativos de identidad conocida separados de hipótesis desconocidas, y gradiente que modifica realmente las convoluciones. Preparación CPU completada en 37,85 s. Entrenamiento GPU lanzado; evaluación CPU preparada. Los recibos `results/E033_*` registran los estados y resultados posteriores.
