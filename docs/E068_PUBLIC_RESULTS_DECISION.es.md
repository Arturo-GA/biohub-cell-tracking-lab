# Cambio de prioridad tras el público — 27-09-2026 12:25 UTC

La API confirma los cinco resultados de la tanda del 27 UTC:

| Envío | Configuración | Público |
|---|---|---:|
| 56592151 | c3: L7 + poda E + cortes, tau base | **0.955** |
| 56592042 | S2: tau 1.0 en 6bba + E + cortes | 0.950 |
| 56591883 | S1: S2 + fork8-nanonly | 0.949 |
| 56591970 | S5: tau 1.0 en 6bba + P8 + cortes + fork | 0.949 |
| 56592094 | c6: tau 1.0 global | 0.949 |

El máximo confirmado es 0.955, no 0.945. El usuario informa que se encuentra en bronce y estima el corte de plata alrededor de 0.956; no se verificó independientemente el ranking vivo en esta consulta.

## Decisión

El nuevo control público es c3, 56592151. La ganancia local de aumentar tau NO se transfirió al público en los candidatos de tau 1.0. No atribuir toda la caída a una única causa sin revisar logs ocultos/runtime; c3/S2 ofrece el contraste más directo para sospechar de la relajación de divisiones.

A y B (tau 0.9) siguen técnicamente listas, pero se retira su prioridad automática como próximos dos envíos. No se ha probado todavía su score público y no se afirma que necesariamente fracasen. Tras los nuevos resultados, no consumir dos slots en variantes tan cercanas de la línea debilitada: mantener como máximo una como exploración si queda cupo después de candidatos basados en c3.

Priorizar variantes que mantengan tau base y mejoren la poda/cortes o reparen enlaces sobre c3, con controles de conteo, tiempo y CSV. Las métricas de CPU sirven para descartar errores y comparar mecanismos, no para prometer ganancia pública. CPU5 no aportó ninguna mejora y no debe enviarse.

## Cupos

La API confirma cinco envíos diarios, no un límite total agotado. Siguiente reset: 28-09 00:00 UTC = **27-09 19:00 Lima**. Otro reset: 29-09 00:00 UTC = **28-09 19:00 Lima**. Potencialmente diez slots antes del cierre 29-09 23:59 UTC, sujeto a los límites vigentes y a dejar tiempo para scoring. No se promete un número de resultados completados.

No se enviaron nuevos candidatos ni se crearon automatizaciones en esta revisión.
