# Ejecución autorizada — 26-09-2026

Objetivo actualizado por el usuario: mejorar el público con una submission robusta; los diagnósticos del privado no son vetos automáticos.

## Lanzado

- Kaggle privado: https://www.kaggle.com/code/jarturo/biohub-exact-replay-cpu4
- Versión 1; CPU, GPU/TPU e internet desactivados. Último estado consultado: RUNNING, sin failureMessage. El endpoint de logs devolvió vacío; no se afirma todavía que haya completado los controles reales.
- SHA256 del notebook enviado: `7465bc66507865de3fb9a3a648212742f9a6767e3ee08bb8aee9b2b8a0b5bfbd`.
- Snapshot exacto en `kaggle/x138_xr/k_cpu4/notebook.ipynb`. El helper local se limpió posteriormente retirando una función incremental que no se usa; no se volvió a subir ni cambió el cálculo activo.
- Capturas privadas lab4–lab8, 199 videos obligatorios; nueve configuraciones: control, A–E, control XR-E y estabilidad tau 0.85/0.95.
- Recalcula todo el suavizado sobre coordenadas capturadas y redondea igual que el escritor. Las capturas raw son float32: es un replay coherente sobre las capturas, no una promesa de identidad bit a bit con las coordenadas de mayor precisión de la inferencia original.
- Antes de evaluar compara contra el linefit original en los dos videos más pequeños, uno por grupo, para las seis primeras configuraciones. Aborta ante cualquier diferencia entera, aristas base distintas o evaluación incompleta.

## Comprobaciones locales

Cuatro tests pasaron: paridad del suavizado con la función extraída del notebook, cortes/divisiones/gaps/coordenadas float32 y empates de redondeo; gate con marcador ausente/repetido, conteos inconsistentes, fallbacks y tolerancias desactivadas. La agregación del analizador reproduce las 20 configuraciones anteriores de CPU3 con error menor de 1e-12.

Se corrigió `build.py`: ya no intenta leer el archivo ausente `train_heads.py` al construir laboratorios o variantes. Los generadores no ejecutan capturas de entrenamiento como efecto de importarlos.

`submit5.sh` queda deshabilitado porque no vinculaba el log con la versión enviada. El gate ahora exige el marcador final, conteos reales, coincidencia de datasets y ausencia de errores. El flujo antiguo `batch.sh` tampoco pasa este gate con sus objetivos ficticios; no usarlo.

## Preparado, todavía no ejecutado

Notebooks A–D en `kaggle/x138_xr/k_exact_a` … `k_exact_d`, privados por metadata. Tienen L7, deadline 34200, las reglas del plan y un manifiesto al final con la huella del código y los parámetros. Se verificó la sintaxis. Su subida GPU depende de leer CPU4; no se consumió GPU ni se envió ninguna submission nueva.

## Continuación

1. Consultar estado de CPU4. Al finalizar, descargar resultados y `.rows.json` a `outputs/e068_review/cpu4` y revisar el log; si falla, resolver la causa antes de inferencia GPU.
2. Ejecutar `python kaggle/x138_xr/analyze_cpu4.py <carpeta que contiene los .rows.json>`. Ordenar por ganancia corregida y separar pérdidas graves de diagnósticos de mezcla privada.
3. Verificar cuota GPU real; lanzar solo finalistas. Antes del envío verificar versión, manifiesto con hash esperado, CSV completo y gate con conteos reales. Evitar confiar en que `kernels_output` fija versión: esta versión del SDK parsea la versión pero no la aplica a la petición de salidas.
4. La API confirmó cierre 29-09-2026 23:59 UTC y cinco submissions el 26. Próxima ventana prevista: 27-09 00:00 UTC, equivalente a 26-09 19:00 Lima. No hay un proceso automático programado para enviar a esa hora.

No se ha hecho commit ni push de estos cambios locales.
