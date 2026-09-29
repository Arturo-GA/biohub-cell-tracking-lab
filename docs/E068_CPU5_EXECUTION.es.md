# CPU5 y submissions preparadas — 26-09-2026

## A y B listas

Ambos notebooks terminaron COMPLETE. Se descargaron CSV, logs y manifiestos. Verificación aprobada: huella de código igual al notebook local, ids continuos, detecciones únicas, coordenadas enteras no negativas, aristas existentes entre frames consecutivos, grado entrante <=1 y saliente <=2, sin aristas duplicadas, conteos CSV iguales a los logs con tolerancia cero y marcador final config=base. No se identificaron errores/fallbacks en los logs.

- A v1: 233056 filas.
- B v1: 232352 filas.
- Recibo con hashes y conteos: `results/E068/ab_verified_outputs.json`.
- Aún no enviados a la competencia. A las 20:48 UTC del 26 faltaban 3 h 12 min para la siguiente renovación prevista del cupo (19:00 Lima).

## Nueva prueba lanzada

`jarturo/biohub-lineage-complement-cpu5`, versión 1, kernelId 136022664, aceptado por Kaggle. Privado, CPU, sin GPU ni internet. Recibo: `results/E068/cpu5_push_receipt.json`.

Siete configuraciones sobre 199 videos: control x138, A, B, A+reparación antes de XR a 4 um, A+reparación después de XR a 4 um, A+reparación antes a 5 um y B+reparación antes a 4 um. Así se distinguen la protección de ramas antes de la poda y la reparación de extremos supervivientes después.

La función `lineage_repair.py` prolonga ramas hijas con longitud menor de ocho mediante detecciones existentes, sin quitar aristas ni reasignar detecciones ocupadas. Exige elección mutua entre extremos disponibles y limita adiciones al 0.2% de enlaces (mínimo uno). No usa etiquetas. En la variante anterior a XR puede cambiar qué nodos sobreviven a la poda, aunque la reparación por sí misma no agrega nodos.

Se inspira en la idea revisada en `amanatar/optimized-biohub-max-score`, con implementación independiente. Seis tests locales aprobados, incluida conservación de detecciones, ausencia de mutación del grafo base y rechazo de continuaciones ocupadas, no mutuas o demasiado lejanas. Sintaxis del notebook comprobada.

Al terminar, comparar cada reparación contra A o B correspondiente, además del control x138. No promover por una ganancia contra x138 que ya explique A/B. Mantener evaluación exacta de suavizado/redondeo y 199 videos completos. No se afirma mejora hasta leer los resultados.
