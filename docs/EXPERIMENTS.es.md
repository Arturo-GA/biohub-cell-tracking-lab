# Experimentos

## E000 — Control fijo y evaluación del CSV

Base: primeras siete celdas de Harmonic Fusion congeladas. Se retira el sweep posterior basado en su métrica proxy. Conserva detector, pesos, TTA, ILP y postproceso de la fase de inferencia. Este control no pretende reproducir un score histórico del notebook completo, que podía seleccionar otra configuración.

Medir: tiempo, nodos, enlaces, divisiones, hash del CSV, adjusted edge Jaccard y división oficial. Validar que todos los datasets estén presentes y el grafo use coordenadas reales. No confundir ejecución sobre cuatro muestras visibles con evaluación del test oculto.

## E001 — Division Guard (implementado)

**Hipótesis:** una fusión simétrica de asociaciones hacia delante y hacia atrás puede debilitar asociaciones correctas alrededor de una división, donde las entradas invertidas difieren del patrón de entrenamiento. El predictor inverso no tiene por qué aportar la misma evidencia en ese caso.

Se conserva la fórmula armónica del control. Solo cambia el peso inverso por célula fuente, desde 0.15 hacia cero, si sus dos mejores candidatos a hija cumplen:

1. La probabilidad forward de ambos como hijos del mismo padre supera 0.55; la protección crece linealmente hasta 0.85.
2. Ambas hijas están a un máximo de 9 µm del padre y a 14 µm entre sí.
3. Todas las decisiones usan predicciones y coordenadas; no se leen etiquetas para la inferencia.

El módulo no crea nodos, enlaces ni divisiones. El ILP existente sigue decidiendo la topología. Las distancias se calculan en coordenadas físicas, tras deshacer el downsample.

**Posible fallo:** proteger una falsa división causada por células cercanas o por exceso de confianza. Aunque no cambie directamente la topología, al cambiar puntuaciones puede alterar muchos enlaces del solver. Vigilar el término de aristas, no solo la división. No se afirma que esta regla sea una novedad científica ni una implementación del paper HOCT.

**Prueba actual:** `kaggle/diagnostic` ejecuta control y candidato con los mismos pesos y muestra determinista de train. La métrica se aplica al CSV final redondeado. La pertenencia de esos videos al train de los pesos públicos está sin verificar; sirve para detectar daños y comprobar el mecanismo, no para certificar generalización.

## E002 — FOCUS-3D como maestro de un detector rápido (diseñado)

Primero segmentar un subconjunto permitido de train; medir cobertura de los centroides anotados y número de detecciones frente a `estimated_number_of_nodes`. Si mejora esa frontera, destilar centros/heatmaps a un UNet3D rápido. Separar embriones/adquisiciones antes de generar targets. Evitar convertir automáticamente cualquier punto sin anotación en negativo.

Comparaciones: mismo linker con detector actual vs detector destilado; después combinación selectiva solo en huecos confirmados. Registrar estabilidad temporal de detecciones y precisión de localización. La inferencia completa de FOCUS-3D puede exceder el límite, según un participante; distilar es una propuesta para reducir ese coste, no una aceleración ya medida.

## E003 — HOCT sobre máscaras reales (diseñado)

Usar un checkpoint público local y máscaras 3D derivadas de imágenes. Medir su linker con detecciones fijas antes de mezclar detectores. Probar CPU y GPU, memoria y tiempo por video. Reconstruir enlaces consecutivos válidos: HOCT admite gaps que no se deben exportar directamente como enlaces que salten frames.

Una vez verificadas las representaciones y las etiquetas utilizables, estudiar una cabeza ligera sobre features congeladas. No iniciar entrenamiento completo sin confirmar la disponibilidad del código y del protocolo. El código de entrenamiento aún no estaba publicado en el anuncio consultado.

## Orden y decisión

1. Completar E000/E001; registrar incluso un resultado negativo.
2. Verificar split de entrenamiento de los artefactos y ampliar la evaluación a videos completos independientes. Agrupar por embrión/adquisición; dividir frames al azar introduce fuga.
3. Priorizar E002 si predominan nodos perdidos; E003 si predominan asociaciones erróneas con nodos detectados.
4. Comparar control/candidato en el leaderboard con presupuesto limitado y luego validar estabilidad antes de elegir finales.

No se transporta el umbral 0.01 de RSNA a esta métrica. No se promueve automáticamente una diferencia pequeña de un diagnóstico sobre train. Para análisis de incertidumbre, re-muestrear videos o adquisiciones, nunca miles de enlaces como si fueran independientes. Con pocos embriones, el intervalo también tiene limitaciones.

## Estado

Ver `results/STATUS.json`. Las rutas E002/E003 son propuestas pendientes; no hay entrenamiento FOCUS-3D/HOCT en este repositorio. E001 y el empaquetado tienen pruebas funcionales locales, además de la ejecución remota iniciada.
