# E055–E056: diagnóstico y especialistas selectivos

**Cierre de E055–E057:** ocho variantes nuevas completadas; ninguna supera
al mejor control local anterior. No hay notebooks ni scoring pendientes de
esta ronda, ni nueva submission. El objetivo de 0.947 sigue sin alcanzarse:
último leaderboard confirmado **0.946**.

## Qué demuestra el último envío

E054, submission **56433260**, terminó con **0.946**. La optimización de
tiempo conserva exactamente el CSV visible del control público; reducir el
tiempo de ejecución no implica mejorar las predicciones. Quitar los puentes
del ensamble E050 tampoco cambió el score mostrado. No hemos alcanzado 0.947.

El control local visual mejora a Harmonic en ocho videos, pero su submission
también obtuvo 0.946. Esta ganancia local no se puede extrapolar al test oculto:
los ocho videos se han reutilizado muchas veces y la pertenencia a entrenamiento
de los modelos públicos no está completamente documentada.

## Diferencia entre scores públicos: evidencia y límites

Kaggle anunció una corrección de la métrica y confirmó que terminó de recalcular
los envíos en [COMPLETED: Rescore Underway](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/728324).
Un participante reportó después que algunos notebooks conservaban scores
anteriores en el listado: [Public Notebook Rankings Need a Metric Refresh](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/736937).

Esto ofrece una explicación posible de discrepancias entre listado y resultado
actual. **No demuestra que el 0.947 de Harmonic V3 concreto esté desactualizado**.
La captura de Arturo muestra ese score realmente; no hay razón para negarla.
La igualdad del CSV visible tampoco demuestra igualdad del test oculto ni de la
versión de métrica usada al obtener un score histórico.

Una comparación estática adicional verifica que las 431 instrucciones
principales de nuestra base, quitando imports de presentación y descripciones,
coinciden con el prefijo del código público V3. El público añade validación,
barrido de posprocesado e informes. E054 fija `tight55`; no ejecuta ese barrido.
La API informa versión pública actual 4, pero rechazó la recuperación por
número de versión; no se pudo verificar a qué versión corresponde el score
histórico listado.

## Auditoría de cinco notebooks recientes

Fuentes, hashes y comparación del código: `results/E055_public_audit.json`.

- `beraterolelk/0-947-lb-biohub-deepcenter-ilp-tracker` y
  `mthem77/biohub-b0-harmonic-fusion-safety-re-run`: código ejecutable idéntico
  a V3; solo añaden una descripción inicial.
- `haideptry/biohub-0-951-sota-deepcenter-fast-ilp-19m`: sus clases DivNet son
  idénticas a las que cargaban cero tensores del checkpoint auditado previamente;
  mantiene `strict=False`. El título no verifica el score ni la carga de pesos.
- `andnyu/biohub-947-mutual-rank`: bonificación de rango antes del ILP;
  E051 fue una adaptación posterior, no una reproducción exacta de este cambio.
- `andnyu/biohub-947-synthetic-edge`: tercer Transformer entrenado con datos
  sintéticos, limitado a enlaces ambiguos. Esta es la nueva línea E056.

## E055: ensamble por grupos de enlaces en desacuerdo

Cada intercambio se acepta completo para preservar las restricciones del grafo.
Los nodos y las divisiones originales se conservan. Se evalúan tres criterios
fijados antes de leer etiquetas: ventaja de confianza, coherencia de movimiento
con vecinos originales y ambos criterios juntos. Los CSV se congelan antes de
evaluar la métrica oficial.

| Variante | Score local | TP / FP / FN de enlaces |
|---|---:|---:|
| Harmonic | 0.943682890 | 4699 / 171 / 166 |
| Visual anterior | **0.950283267** | **4715 / 153 / 150** |
| Confianza selectiva | 0.947548243 | 4708 / 160 / 157 |
| Trayectoria selectiva | 0.943682890 | 4699 / 171 / 166 |
| Ambos filtros | 0.943682890 | 4699 / 171 / 166 |

Las tres se rechazan. Sobre 1027 grupos en desacuerdo, confianza acepta 461;
trayectoria, seis; ambos, tres. Los dos últimos cambian enlaces sin alterar
los conteos evaluables. Confianza evita un error de visual en `44b6_12dfb391`,
pero pierde correcciones útiles en otros videos. No se seleccionan métodos por
nombre de video: eso usaría las etiquetas de evaluación como regla de inferencia.
Proceso: **130.391 s CPU**, cero GPU. Tres pruebas unitarias aprobadas.

## E056: especialista sintético para incertidumbre de asociación

Adaptación atribuida del código de [andnyu](https://www.kaggle.com/code/andnyu/biohub-947-synthetic-edge).
Pesos de [bhpepper](https://www.kaggle.com/datasets/bhpepper/biohub-synthetic-5fold-ensemble-v1),
SHA256 `0eacacaf0b43bfd5a063495d6991a4911cd045c37650363d5d8826a0e7ed3dd9`.
Auditoría en laptop: **136 tensores cargados estrictamente**, 2 076 706 parámetros.
No se interpreta el proxy del autor como nuestro score ni como prueba de
independencia respecto a sus datos de entrenamiento.

Conserva la detección original. Interviene en la asociación cuando el margen
entre las dos madres candidatas principales de Harmonic es menor que 0.12 y
el margen sintético lo supera por más de 0.02. Su peso aumenta con la duda del
modelo principal hasta un máximo de 0.25. Se alinean media y escala de logits
por destino antes de combinarlos. Sin etiquetas ni selección por video.

Se infieren los ocho videos reutilizados en GPU y después se evalúan el grafo
ILP y su asociación visual en CPU. La comparación relevante es contra los
controles anteriores, no solo entre los dos candidatos nuevos. La validación
es exploratoria y no permite prometer una mejora en el leaderboard.

Estado y resultados finales se registran en los recibos E056 de `results/`.

E056 terminó correctamente: 792 pares temporales, 1602.276 s de proceso en
el notebook GPU y 128.789 s de evaluación CPU. El tiempo del notebook GPU
incluye instalación y posprocesado; no equivale a una medición de facturación.

| Variante | Score local |
|---|---:|
| Harmonic anterior | 0.943682890 |
| Harmonic + especialista sintético | 0.943677641 |
| Visual anterior | **0.950283267** |
| Sintético + visual | 0.950277921 |

El especialista no cambia los conteos TP/FP/FN de enlaces o divisiones de
ningún video, aunque cambia las predicciones. El filtrado final deja 38 nodos
adicionales netos, sin mejorar el recall de nodos anotados. La diferencia
pequeña del score ajustado procede del cambio de cantidad de nodos.
La mejora de visual sobre el Harmonic de esta nueva ejecución no basta para
promoverlo: la comparación con el mejor control anterior es ligeramente peor.

## E057: aislar la asociación sobre el grafo original

Protocolo fijado antes de observar los scores E056. Reutiliza la misma
inferencia GPU; toda esta etapa es CPU. Conserva los nodos y divisiones del
Harmonic anterior y compara tres asociaciones: probabilidades de E056,
media geométrica con peso 0.5 y peso adaptado a la incertidumbre del destino.
Esta última exige margen original menor que 0.2 y mejora de margen superior
a 0.02; aumenta gradualmente el peso nuevo hasta 0.5.
Las probabilidades de E056 ya mezclan Harmonic con el modelo sintético, con
peso sintético máximo 0.25 en logits; no son la salida del modelo sintético
aislado. E057 combina dos pipelines que comparten los modelos públicos.

La comparación verifica igualdad exacta de coordenadas de detecciones
capturadas antes de mapear probabilidades al grafo original. No supone que
los identificadores GEFF permanezcan iguales después de resolver otro ILP.
Los candidatos ausentes de una caché dispersa no se interpretan como
probabilidades cero. Tres pruebas unitarias cubren esas reglas de mezcla.

E057 terminó en **179.857 s CPU**. Las tres variantes empatan exactamente
con visual: **0.950283267**, 4715 TP / 153 FP / 150 FN de enlaces y
0 TP / 2 FP / 5 FN de divisiones. Los cuatro hashes CSV son diferentes y
las cantidades de enlaces cambiados también: la integración sí tiene efecto.
La variante adaptativa activa la mezcla en 374 de 206200 destinos (0.181 %).
No cambia ningún conteo evaluable por video. Esto no demuestra que cada enlace
cambiado carezca de anotación, pues los conteos pueden compensarse; tampoco
demuestra que no pudiera haber cambios en el test oculto. Sí deja sin evidencia
una mejora frente al control disponible.

Decisión y comparación por video: `results/E056_E057_decision.json`.
Se conservan los pesos públicos y las cachés para investigación, pero no se
gasta otra inferencia GPU de test en estas variantes empatadas o peores.
Seis pruebas locales pasaron y se verificó la carga estricta del checkpoint.

## Nueva lectura sobre confianza

[Paul et al., ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Paul_How_To_Make_Your_Cell_Tracker_Say_I_dunno_ICCV_2025_paper.html)
estudian incertidumbre en asignación de células y describen calibración de
confianza, incluida la de trackers Transformer. Esto sugiere investigar una
calibración aprendida antes de usar diferencias de confianza como árbitro.
Es una hipótesis futura: E055–E057 no implementan sus métodos ni prueban que
la falta de calibración explique por sí sola nuestro score. Las anotaciones
parciales exigen seleccionar enlaces evaluables y separar el ajuste de la prueba.
