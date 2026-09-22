# E067: reducción robusta del ruido de trayectoria completa

E065 y E066 no modifican el test visible y se descartan para una nueva submission. E067 prueba una tercera línea: ajustar conjuntamente las coordenadas de cada trayectoria individual mediante una penalización de aceleración, conservando todos los nodos y enlaces.

## Diferencia respecto a lo anterior

Harmonic aplica un ajuste lineal local de cinco fotogramas. E064 buscaba centros brillantes en la imagen y empeoró ligeramente en validación; terminó en 0.946 público. E067 no usa intensidad ni vuelve a buscar células: ajusta una trayectoria completa con penalización de segunda diferencia temporal, en coordenadas físicas. El problema conecta todo el tramo, no solo una ventana local. No se suaviza a través de bifurcaciones.

Se resuelve un sistema pentadiagonal con cuatro iteraciones de mínimos cuadrados reponderados. Los pesos son `1/sqrt(1+(residuo/1.5 µm)^2)`, reduciendo la influencia de observaciones que se separan de la curva. Se preserva el movimiento lineal como punto fijo. Los tramos de menos de cinco nodos permanecen intactos.

Se fijaron tres intensidades de penalización antes de evaluar: **0,5**, **2** y **8**. En todas, la corrección continua se limita a 1,2 µm antes del redondeo nativo. El redondeo puede superar ese límite; se registra el desplazamiento final. Se reutilizan las protecciones de E064: pertenencia al centro original más cercano, rechazo de colisiones y de cruces nuevos del límite de enlace de 14 µm, además de límites de imagen.

## Evaluación y envío

Se comparan las tres variantes en los 24 videos reutilizados de E061, congelando los CSV antes de leer anotaciones y reproduciendo el score del control. Este conjunto no es un holdout independiente y mezcla Harmonic con ocho grafos de asociación visual. La submission usa Harmonic puro como punto de partida.

La comprobación local CPU del test visible tardó unos segundos y encontró cambios en las tres variantes: **26.777**, **63.593** y **92.985** centros respectivamente. Se utilizaron predicciones del control previamente generado únicamente para esta comprobación sin etiquetas; no se exportó una submission desde esa caché.

El notebook de envío recalcula toda la inferencia Harmonic desde las imágenes test que Kaggle suministre y luego aplica el método temporal en CPU. La GPU se usa para las redes volumétricas existentes. El usuario solicitó expresamente enviar el experimento: se elige la mejor variante nueva de evaluación entre las que producen cambios en test visible. Si no supera el control, se informa como exploratorio, sin prometer mejora.

Cuatro pruebas pasan: trayectoria lineal inalterada, reducción de ruido aislado, separación de bifurcaciones y cambio real con límites físicos y nodos conservados. El CSV final se descargará para comprobar hashes, coordenadas válidas, enlaces idénticos y cambios efectivos antes del envío único.

Evaluación CPU completada en 227,01 s de proceso. Se seleccionó la intensidad media (lambda 2), que supera tanto al control como al mejor E066. Inferencia de submission completada y envío aceptado.

## Resultados en 24 videos reutilizados

| Variante | Score | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: |
| control | 0.919878413 | 11389 | 638 | 651 |
| weak | 0.919794727 | 11387 | 637 | 653 |
| medium | 0.922099492 | 11398 | 617 | 642 |
| strong | 0.920330713 | 11390 | 633 | 650 |

La variante media gana **0,002221079** de score, **9 TP netos** y reduce **21 FP**. Mejora la métrica ajustada de enlaces en 10 videos, empeora en 4 y empata en 10. Son conteos agregados, no una auditoría de correspondencia exacta de cada enlace GT. Las divisiones no cambian (3 TP, 6 FP, 8 FN). El node recall baja ligeramente: 0,981003798 a 0,980615612; no todos los indicadores mejoran.

Los agregados por embrión mejoran: 44b6 pasa de 0,923584894 a 0,927223177 y 6bba de 0,923503528 a 0,925214622. Esto respalda probar el método, pero no elimina el sesgo de seleccionar entre experimentos sobre datos reutilizados.

## Submission confirmada

Kaggle aceptó **56475507** el 22 de septiembre de 2026 a las 22:55 UTC, notebook **jarturo/biohub-e067-trajectory-denoise**, versión 1. Último estado leído: **PENDING**, sin score todavía. No quedan notebooks de experimentación ejecutándose; queda la puntuación de la submission. Mejor resultado público previo: **0.946**.

La inferencia y corrección consumieron 1047.88 segundos de proceso en el notebook GPU (no es una lectura de cuota/facturación). Solo esta etapa utilizó GPU. El control previo al ajuste reprodujo exactamente el SHA256 del control E054. El CSV final cambia **63.593 de 122.794 centros**, conserva todos los nodos y enlaces y tiene SHA256 `c6cf1c539c25859c5677698d922eb993993b99dca15a6f87db152622fe05ea58`. La descarga pasó validación completa y se verificó el número de cambios contra los reportes.

- [Notebook privado](https://www.kaggle.com/code/jarturo/biohub-e067-trajectory-denoise)
- [Decisión consolidada](../results/E067_decision.json)
- [Auditoría por video](../results/E067_selected_audit.json)
- [Validación del CSV](../results/E067_SUBMISSION_completed.json)
- [Comprobante de envío único](../results/E067_SUBMISSION_attempt.json)
- [Último estado de Kaggle](../results/E067_SUBMISSION_status.json)
