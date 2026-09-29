# Revisión del traspaso E068 — 26 de septiembre de 2026

## Estado verificado

Se actualizó el checkout limpio por fast-forward a `1b4b4c4` de `origin/main`, incluido el traspaso. El texto adjunto y los planes del repositorio se trataron como contexto, no como órdenes para ejecutar automáticamente los cinco envíos.

La API de Kaggle confirma varios resultados públicos de **0.954**, incluidos 56539929 y 56540269. E067 terminó en 0.946. La posición 177 y el corte de plata son los del documento de traspaso; esta revisión no volvió a descargar el ranking vivo ni garantiza medalla privada.

En la consulta alrededor de las 16:14 UTC, las capturas 6–8 y los cinco candidatos S1, S5, S2, c6 y c3 figuraban COMPLETE; `biohub-safediv-lab-cpu3` seguía RUNNING. Los cinco envíos del 26 de septiembre estaban completos; no aparecían todavía envíos S1/S5/S2/c6/c3. La consulta puntual del log de cpu3 devolvió una cadena vacía, por lo que no se afirman nuevos resultados de su versión 3.

No se lanzaron entrenamientos, kernels, submissions ni automatizaciones durante esta revisión. Los notebooks preparados contienen una llamada directa a `write_test_submission("base")` y el deadline 34200 en la configuración inicial.

## Hallazgos que afectan la siguiente decisión

### 1. El replay no reproduce las coordenadas finales después de cambiar enlaces

`cpu_lab2.py:121` y `cpu_lab3.py:189` utilizan las coordenadas `nodes_final` guardadas para el grafo original. En el notebook S1, las reglas XR se aplican antes de `linefit_smooth_output_graph`. Esa función usa predecesores y sucesores, por lo que cortar enlaces o añadir divisiones cambia sus vecindarios y puede cambiar las coordenadas enteras finales.

Guardar un único suavizado es válido al eliminar componentes completos e independientes; no demuestra equivalencia al cambiar aristas. Reproducir L7 o las aristas safe-division de base no verifica las variantes nuevas.

Se ejecutó un contraejemplo mínimo con la función exacta de x138: al cortar un enlace, tres coordenadas X redondeadas cambiaron entre reutilizar el suavizado anterior y recomputarlo (4→6, 4→3 y 3→4). La evidencia está en `outputs/e068_review/audit.json`. Esto demuestra la diferencia de código; **no cuantifica ni demuestra que desaparezcan las ganancias reales**.

Prioridad: recomputar linefit y el redondeo por candidato, después de XR y el filtro de componentes, y repetir los finalistas sobre las capturas existentes en CPU. No hace falta otra captura GPU para esta comprobación.

### 2. La verificación previa al envío permite falsos positivos

`submit5.sh` usa objetivos de 1 nodo/1 enlace y tolerancia `1e9`: en la práctica desactiva la comparación de conteos prometida por el traspaso. Además, `gate.py` solo rechaza una configuración distinta de base si encuentra el marcador final; si no lo encuentra, puede aprobar igualmente. Un log sintético con cuatro conteos pero sin `Final submission.csv ... config=base` devuelve `GATE PASS`.

Antes de reutilizar el envío automático: exigir el marcador final, conteos esperados reales por candidato y comprobación de la versión que se va a enviar. El script actual consulta el log más reciente pero envía la versión indicada; esas dos identidades deben quedar vinculadas. No se modificaron los notebooks ya terminados.

### 3. Los intervalos actuales no son probabilidades de medalla ni garantías del privado

Las 199 muestras amplían la cobertura, pero incluyen datos vistos por los modelos y reutilizados para elegir reglas. El bootstrap por video no corrige ese sesgo ni la selección de muchos umbrales. Una cota positiva es evidencia útil dentro de esa muestra, no una probabilidad calibrada de mejora privada. Un score mostrado con tres decimales tampoco distingue todos los resultados subyacentes; ±0,0005 por redondeo no es por sí mismo una estimación de ruido estadístico.

También conviene abortar una evaluación incompleta: los laboratorios registran errores por video y después calculan métricas con los casos restantes. La comparación final debe exigir todos los videos previstos y cero discrepancias de reproducción.

## Dirección recomendada

Mantener **x138 + V1284 + L7** como referencia actual. El resultado de 0.954 sí está confirmado y sustituye al antiguo punto de partida de 0.946. No volver a los experimentos anteriores como base competitiva ni añadir un sweep o una segunda inferencia al kernel de envío.

Primero corregir y medir la equivalencia del replay en CPU; luego leer cpu3 v3 y comparar S1/S5/S2 con sus variantes de tau. Priorizar recuperación de divisiones y reglas XR que sobrevivan a esa reproducción y mejoren ambos embriones, conservando un control 0.954. Los umbrales y tiempos del traspaso son hipótesis y límites operativos documentados; cualquier nueva tanda debe basarse en el estado real de Kaggle y en ese replay corregido.

Una cabeza aprendida de divisiones sigue siendo una línea posible, pero no la priorizaría frente a esta auditoría antes del cierre: ya hay candidatos preparados y una fuente concreta de discrepancia que se puede resolver sin GPU.
