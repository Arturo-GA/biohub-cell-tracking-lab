# E064: corrección de centros guiada por imagen nativa

## Hipótesis y método

Tras E063, se prueba una línea distinta: mejorar las coordenadas de células que el rastreador ya encuentra. El suavizado de posiciones de Harmonic no vuelve a consultar la imagen nativa. E064 mide una corrección en la imagen y regulariza esa corrección a lo largo de cada trayectoria; no añade células ni cambia enlaces.

Para cada centro se extrae una ventana nativa de 5×13×13 voxels. Se resta el percentil 20 como fondo local y se pondera el contraste positivo al cuadrado con una gaussiana espacial de sigma 1,2 µm. El primer momento proporciona un desplazamiento limitado a 1,2 µm antes del redondeo. El redondeo al voxel nativo puede aumentar la magnitud final; se registra el máximo efectivo.

La variante temporal combina por igual el desplazamiento observado y su mediana en tres fotogramas de una cadena sin bifurcaciones. No mezcla las correcciones de madre e hijas a través de una división. Las propuestas conservan la pertenencia al centro original más cercano, revierten colisiones y revierten desplazamientos que llevan un enlace originalmente de hasta 14 µm por encima de ese límite. Las coordenadas se mantienen dentro de la imagen.

Se fijaron tres variantes antes de leer sus métricas: corrección de imagen con peso 0,25; corrección temporal con peso 0,5; y corrección temporal con peso 1,0. La intensidad de fluorescencia no tiene por qué coincidir exactamente con el centro anotado; por eso el método requiere evaluación.

## Evaluación y envío

Las tres variantes se comparan en los 24 videos reutilizados E061/E063, sobre el control Harmonic/visual existente. Los CSV se congelan antes de leer anotaciones. No es una evaluación independiente del proyecto. La selección elige la mejor variante **nueva** para una submission exploratoria, conforme al pedido del usuario; esa selección no equivale a una demostración de mejora respecto al control.

El notebook de submission vuelve a ejecutar la inferencia completa de Harmonic sobre todas las imágenes test que Kaggle proporcione, incluyendo la ejecución oculta. Después calcula la corrección nativa en CPU. No lee anotaciones de entrenamiento ni reutiliza predicciones test guardadas. Se validan los CSV descargados, la conservación de nodos y enlaces, los cambios efectivos de coordenadas y los hashes antes de enviar una única submission mediante la API.

La comparación local utiliza Kaggle CPU. La ejecución completa de test requiere GPU para las redes volumétricas de Harmonic; la corrección de centros usa CPU dentro de ese mismo notebook. No se utiliza Colab. Cuatro pruebas locales comprueban imagen plana, desplazamiento hacia un centro brillante, protección frente a cruces entre centros y separación de ramas en el filtro temporal.

## Estado

Evaluación completada en CPU (433,08 s de proceso). Resultados sobre 24 videos reutilizados:

| Variante | Score | TP / FP / FN de enlaces |
| --- | ---: | --- |
| Control | 0,919878413 | 11389 / 638 / 651 |
| Imagen, peso 0,25 | 0,919507907 | 11387 / 641 / 653 |
| Temporal, peso 0,5 | 0,918092705 | 11379 / 652 / 661 |
| Temporal, peso 1,0 | 0,918434333 | 11378 / 646 / 662 |

Se eligió `image_quarter` para el envío exploratorio: delta **−0,000370505**, dos TP menos y tres FP más. Ninguna variante demuestra mejora. La seleccionada modifica 14.369 de 528.064 centros (2,72 %), hasta 0,574525 µm. Las métricas agregadas permanecen iguales en 22 videos y empeoran en `6bba_6479435d` y `6bba_9a41d029`; no se observa mejora agregada por video. La variante elegida no aplica el filtro temporal. El control de evaluación mezcla Harmonic y asociación visual en ocho videos; la submission usa Harmonic puro más la corrección nativa. Esta transferencia y la reutilización de videos limitan la comparación.

La primera versión de inferencia falló al importar `zarr` antes de instalar las dependencias offline de Harmonic. Se movió ese import después de la instalación y se lanzó la versión 2. Se corrigió también el registro del identificador del kernel para conservar el slug real devuelto por Kaggle.

La versión 2 completó la inferencia y corrección en **959,67 s** de proceso (15,99 minutos en el notebook GPU, incluida la corrección CPU). Se descargaron y validaron ambos CSV. El control previo a la corrección reproduce byte por byte el control E054: SHA256 `a69c78229c6556d06d5fe9ff050074254b4a326e83249efbf578b843aec7a848`.

El archivo nuevo modifica **2.897 de 122.794 centros** del test visible (2,36 %), mantiene todos los nodos y enlaces y tiene SHA256 `0852cc78bd131306cf51d20af80542eb2bee9136f2d95dcbd7d54a12516ff394`. No se reutilizaron predicciones test para producirlo: esa comparación se realizó después de la inferencia completa.

Kaggle aceptó la submission **56466969** el 22 de septiembre de 2026 a las 14:24 UTC. Estado confirmado el 22 de septiembre de 2026 a las 22:11 UTC: **COMPLETE, 0.946**, sin error técnico. No mejoró el leaderboard. Sin notebooks ni scoring pendientes de E064.

- [Notebook privado, versión 2](https://www.kaggle.com/code/jarturo/biohub-e064-native-image-centers)
- [Decisión consolidada](../results/E064_decision.json)
- [Validación del CSV](../results/E064_SUBMISSION_completed.json)
- [Comprobante de envío único](../results/E064_SUBMISSION_attempt.json)
- [Último estado de puntuación](../results/E064_SUBMISSION_status.json)

El resultado local no justifica presentar esta idea como una mejora: fue un envío exploratorio solicitado expresamente, y se conserva el control como mejor referencia.
