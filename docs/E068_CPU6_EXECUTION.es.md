# CPU6: doble revisión, implementación y lanzamiento — 27-09-2026

Petición: revisar otra vez las ideas e implementar pruebas para preparar submissions. Prioridad: superar el público de c3, 0.955, para aspirar a plata.

## Hallazgo adicional que cambia la interpretación

Las versiones antiguas S1/S2/S5 configuran `BIOHUB_XR_TAU_SPEC=44b6:0.6,6bba:1.0` sin comodín. `_xr_scalar` devuelve 0 para prefijos desconocidos; `xr_set_safe_div_params` asignaba ese 0 a tau, desactivando su filtro. Se demostró por inspección y prueba unitaria. No se conocen los prefijos del test oculto: esto es un fallo real y una explicación posible, **no la causa demostrada de los scores 0.949/0.950**. El resultado de c6 con tau global 1.0 sigue siendo evidencia separada de riesgo al relajarlo.

Se corrigió el módulo fuente: toma una instantánea del tau y DeepCenter base y usa ese valor cuando falta un prefijo, evitando heredar el override del video anterior. No cambia las versiones ya ejecutadas en Kaggle. A/B ya tenían comodín explícito `*:0.6`; no padecían esta omisión. Pueden conservarse como una exploración, sin desplazar automáticamente los experimentos sobre c3.

Las capturas CPU contienen geometría y aristas finales, no todas las probabilidades alternativas de los dos modelos. Por eso no se simula un ensamble de logits inexistentes ni se inventan probabilidades cero. Se priorizan hipótesis que pueden probarse fielmente con los datos disponibles.

## Implementación nueva

`context_rules.py`, compartido entre laboratorio y futuros notebooks:

1. **Cruces temporales:** intercambia dos enlaces si mejoran tanto la extrapolación anterior como la posterior, con contextos disjuntos de cuatro fotogramas. Protege bifurcaciones, conserva nodos y grados, y limita cambios. Es distinto del descriptor de apariencia de E065. Tras cambiar enlaces se recalcula linefit; las coordenadas finales pueden cambiar aunque las detecciones crudas sean las mismas.
2. **Cortes contextuales:** parte de los dos cortes de c3 (relink sin probabilidad >8 um; último enlace de probabilidad <0.5), pero permite conservar movimientos respaldados por dos pasos previos y, cuando está disponible, movimiento de vecinos. Para el corte largo exige también continuación posterior. No elimina más nodos a ciegas ni cambia divisiones.
3. **Suavizado según topología:** reduce la intensidad de linefit en madres bifurcadas y sus hijas inmediatas, conservando el suavizado normal en el resto. Busca evitar desplazar centros por mezclar ventanas alrededor de una división. Distinto de generar/prolongar divisiones, que CPU5 no mejoró.

## Laboratorio lanzado

- `jarturo/biohub-c3-context-cpu6`, **v1**, kernelId **136111136**, privado, CPU, sin GPU ni internet. Último estado consultado RUNNING.
- SHA256 notebook: `72807a2750f7777f956bdfc2c7516b944cab2deff96f20fb9e7b96df40fbfae0`.
- Recibo: `results/E068/cpu6_push_receipt.json`.
- Diez configuraciones: C3_public0955; swaps strict/balanced; protección de cortes end/long/both; linefit de bifurcaciones 0.3/0; swaps+contexto; contexto+linefit.
- 199 videos obligatorios y suavizado/redondeo recalculado. Comprobación contra la función original en un video pequeño por grupo para cada configuración. Conservar controles contra c3, no contra una variante tau 0.9.
- Añade `graph_diagnostics.json`: hashes de nodos, enlaces y coordenadas, contadores de operaciones y conteos finales por video. Permitirá distinguir ausencia de cambios de cambios fuera de las anotaciones, algo que CPU5 no documentaba.

## Comprobaciones y preparación de submissions

Doce tests locales aprobados: cruces reparados sin cambiar grados, protección de bifurcaciones, rescate de extremos coherentes, rechazo de movimiento incoherente, equivalencia de cortes desactivados con c3, prefijos desconocidos, paridad de suavizado con referencia y pruebas anteriores de gates/lineage.

`build_context.py` genera nueve candidatos privados de una pasada, a partir de c3, con L7, tau base 0.6, validator apagado y deadline 34200. Se incluye el mismo helper evaluado en CPU, un manifiesto con parámetros y huella de código, y se comprueba sintaxis. Inventario: `results/E068/cpu6_candidates.json`. **Los nueve están preparados localmente, no subidos ni ejecutados en GPU.** No se afirma que sean nueve buenos envíos.

## Continuación al terminar CPU6

1. Descargar resultados y diagnósticos, comprobar paridad, cero errores y 199 videos por configuración.
2. Ejecutar `python kaggle/x138_xr/analyze_cpu4.py <carpeta cpu6> --base C3_public0955`.
3. Comparar ganancias, pérdidas, cambios reales y tiempo. Las combinaciones deben aportar sobre sus componentes; no promover una combinación que solo repita uno de ellos.
4. Elegir hasta tres mecanismos/candidatos diferentes para inferencia GPU, no los nueve. Comprobar cuota y no más de dos sesiones simultáneas. Si un método modifica demasiados enlaces o es lento, no promoverlo sin ajustar y reevaluar.
5. Verificar manifest/versión/CSV y ausencia de fallback antes de enviar. Conservar c3 56592151. Reserva al menos un cupo para correcciones; usar A o B solo como exploración explícita si conviene después de los nuevos resultados.

Ventana siguiente prevista: 28-09 00:00 UTC = **27-09 19:00 Lima**. No hay un nuevo proceso automático de envío. Los CSV públicos finales todavía requieren inferencia y verificación tras seleccionar candidatos.

Las métricas locales son de datos reutilizados, no un pronóstico del público. En esta etapa no se consumieron nuevas submissions ni GPU.
