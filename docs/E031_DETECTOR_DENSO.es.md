# E031 — detector denso sobre videos completos

El usuario autorizó realizar pruebas para mejorar el puntaje. Se implementan dos condiciones de inferencia del detector público de hengck23 y una evaluación completa en CPU. Esta fase prueba detección y un control geométrico de tracking; todavía no entrena el encoder temporal propuesto.

## Protocolo fijado

- Los mismos 16 videos de calibración registrados anteriormente, todos sus fotogramas. No selección por errores ni cambios del conjunto tras observar resultados.
- UNet3D público `00000030.pth`, código `model_v5.py` con hash fijado. Dataset original adjunto directamente al notebook privado; no redistribuimos código ni pesos externos en GitHub.
- Normalización de imagen con cuantiles 0,001 y 0,999 de sus metadatos; reducción XY por cuatro, como la demo. Preparación en Kaggle CPU.
- Dos condiciones GPU: estadísticas BatchNorm congeladas y estadísticas del lote de 20 fotogramas, que reproduce la semántica de normalización de la demo. No dropout, optimizador ni ajuste de pesos. Umbral 0,2 y máximos locales 3×3×3 fijados como en el notebook público.
- Se guardan coordenadas, probabilidades y 64 características neuronales por detección para posteriores pruebas de asociación en CPU.
- Tiempo máximo comprobado al inicio de cada video: 1.800 segundos desde el arranque del script GPU. Puede excederse por el último video, carga y exportación; no equivale a un límite duro del tiempo facturado por Kaggle.

## Evaluación preparada

Comparar cobertura de anotaciones dispersas, omisiones exclusivas y cobertura al limitar detecciones por fotograma al número que conserva Harmonic. Los puntos no anotados no se clasifican automáticamente como falsos positivos.

También se generan CSV completos mediante asignación uno a uno con alternativa de no enlazar, radio físico de 14 µm y mínimo de seis fotogramas por trayectoria. Las dos condiciones se evalúan con la métrica oficial fijada en el repositorio y contra el CSV Harmonic existente. El control geométrico no incluye divisiones y usa otro decodificador: **no atribuir toda diferencia de score exclusivamente al detector**. Su propósito es comprobar si las detecciones permiten una solución temporal coherente antes de invertir en un enlazador aprendido.

Una mejora de cobertura con un gran aumento de puntos no basta para promover. La comparación a igual número de candidatos y el grafo completo deben explicar de dónde vendría la mejora. Un resultado negativo del enlazador geométrico tampoco descarta por sí solo el detector; se separará cobertura de asociación.

## Límites y procedencia

[Notebook público](https://www.kaggle.com/code/hengck23/cell-point-detector). La API del dataset devuelve licencia `unknown`, descripción vacía y ningún manifiesto de entrenamiento. Se realiza un diagnóstico privado; no se afirma independencia respecto de su entrenamiento ni se incorpora el peso a una distribución propia. El ajuste/licencia de un modelo de producción requerirá resolver esa procedencia o entrenar pesos propios con fuentes autorizadas.

Nuestro control también tiene limitaciones de procedencia ya documentadas. Estos 16 videos son calibración reutilizada, no un nuevo holdout. Cualquier score obtenido aquí será local y condicionado; no es el score público de Kaggle.

## Resultado completo

Las tres etapas terminaron. Preparación: 325 s CPU. Proceso GPU: 164 s, incluyendo carga y exportación del script; no es una medición exacta de la cuota facturada. Evaluación: 146 s CPU. Tres pruebas del enlazador aprobadas.

| Condición | Score local completo | Centros anotados cubiertos / 7.432 | Puntos predichos antes del filtro de tracks |
|---|---:|---:|---:|
| Harmonic congelado | 0,900753 | 7.224 | 318.094 |
| Detector + BN congelada + asociación geométrica | 0,756016 | 7.247 | 509.249 |
| Detector + BN por lote + asociación geométrica | 0,714496 | 7.321 | 552.711 |

La condición por lote recupera **152 ocurrencias anotadas** que Harmonic omite y pierde 55 que Harmonic conserva. No son necesariamente 152 células distintas. A igual número máximo de candidatos por fotograma, ordenados por probabilidad del detector, conserva solo 5.329 centros anotados; la condición congelada conserva 6.383. La ganancia de cobertura no se obtiene gratuitamente: mantener los centros recuperados requiere muchos candidatos de baja prioridad según ese score.

**Decisión:** no reemplazar Harmonic ni enviar estos CSV. El modelo aporta candidatos complementarios, pero no una mejora de tracking con este decodificador. Ni estos resultados ni su procedencia justifican promoverlo como solución superior. E032 prueba si las características guardadas permiten aprender una asociación útil; no se repite inferencia GPU.

Recibo completo: `results/E031_completed.json`. Sin submission ni jobs E031 pendientes.
