# Notebooks y discusiones: revisión del 26-09-2026

Se consultaron los listados API por creación y score, y se descargó el código actual de seis notebooks públicos sin ejecutarlo. No se confunden títulos, orden del listado ni score histórico con un score verificado de la versión descargada. La API de listado no entregó scores numéricos. Huellas y fuentes: `results/E068/public_audit_2026-09-26.json`. Código y diffs locales: `outputs/e068_public_review/`.

## Hallazgo prioritario

[optimized-biohub-max-score, Aman Atar](https://www.kaggle.com/code/amanatar/optimized-biohub-max-score), ejecución actualizada el 26-09 a las 13:18 UTC según listado. Su versión descargada incorpora funciones adicionales respecto de nuestro x138:

1. `complete_division_lineages`: prolonga ramas hijas cortas con asociaciones mutuas entre detecciones existentes. Complemento plausible de nuestra recuperación de divisiones; comprobar TP/FP de divisiones y enlaces, no solo contar bifurcaciones. Antes de la poda de ramas puede rescatar ramas reales, pero también proteger bifurcaciones falsas.
2. `repair_low_margin_parents`: modifica únicamente asociaciones geométricamente ambiguas, conserva nodos, protege bifurcaciones y exige una alternativa libre claramente mejor. Candidato CPU sobre las capturas; contrastarlo con los intercambios anteriores, no presentarlo como un nuevo modelo entrenado.
3. `stitch_track_endpoints`: enlaza finales e inicios consecutivos con dirección y margen. El comentario dice que repara daños de poda, pero el código recorre extremos en general: una adaptación nuestra debería registrar qué extremos produjo XR y limitarse a ellos, evitando deshacer cortes deliberados.
4. `repair_second_daughter`: amplía la búsqueda de segunda hija; requiere consultar DeepCenter. Solapa con CPU4, por lo que queda detrás de los tres anteriores, no se apila sin medir falsos positivos.

**Límite importante:** varias de estas reglas vienen apagadas por defecto y se activan mediante un sweep. Su presencia en un notebook destacado no demuestra que expliquen su score. No importar el notebook completo: mantiene barridos/reexportación y un deadline distinto del nuestro. Tampoco adoptar el presupuesto absoluto de nodos sin verificar disponibilidad legítima y consistencia de esos metadatos en inferencia.

## Ideas ya vistas y comparación

- [andnyu / mutual-rank](https://www.kaggle.com/code/andnyu/biohub-947-mutual-rank): premio de ranking mutuo antes de ILP. Ya revisado en E055 y relacionado con E051. No es un descubrimiento nuevo.
- [andnyu / synthetic-edge](https://www.kaggle.com/code/andnyu/biohub-947-synthetic-edge): mezcla un tercer modelo solo cuando el margen público es pequeño y el sintético mejora el margen. **Ya probado en E056–E057**, sin ganancia relevante en aquella base. No repetirlo solo por encontrar otra vez el enlace. La base actual es distinta, pero necesitaríamos una razón concreta para reabrirlo y gastar GPU.
- [Deconfounded Edge Stack](https://www.kaggle.com/code/seyitkaangunes/biohub-035-deconfounded-edge-stack): combina velocidad 0.25, tight55 y poda de hojas; son cambios de posprocesado, no evidencia de otra arquitectura superior.
- [Lineage Forge](https://www.kaggle.com/code/flexonafft/biohub-lineage-forge-precision-tracking): añade poda de hojas débiles. Se solapa con nuestra línea XR; menor prioridad que corregir enlaces después de la poda.
- [Harmonic Fusion V3](https://www.kaggle.com/code/raunakdey07/biohub-harmonic-fusion-v3), actualizado el 25-09: las funciones de nivel superior extraídas por AST coinciden con las del x138 de referencia. Esto no prueba identidad total de notebook, configuración ni score; no se identificó una función nueva que justifique reemplazar nuestra base.

## Discusiones útiles

- [Estancamiento en 0.947 y etapa de relink](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/742266): un participante reporta divergencia entre CV y público, y señala que el relink puede reemplazar las asociaciones del ILP. Es evidencia anecdótica, no una ley. Implicación práctica: medir cuántos cambios del especialista sobreviven al grafo final y sus efectos en detecciones; no basta mejorar logits antes del relink.
- [Radio de asociación y frecuencia de divisiones](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/733973): reporta 151 eventos y desplazamientos en unidades físicas. Refuerza evaluar divisiones separadamente y mantener distancias en micras; no aporta por sí mismo un nuevo predictor.
- [FOCUS-3D](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/738217): propuestas de etiquetas densas y destilación, con reporte de timeout al usar directamente el modelo. Interesante para desarrollo posterior, menor prioridad frente a CPU4 y los pocos días restantes.

La revisión de discusiones se hizo con páginas indexadas: el listado directo ordenado por nuevos no se pudo recuperar. No se afirma cobertura exhaustiva de cada hilo o comentario publicado hoy.

## Decisión

Mantener CPU4 en curso. Después de leer sus resultados, priorizar una comparación CPU con **continuidad de ramas hijas**, y otra de **reparación localizada de enlaces ambiguos o extremos dañados por XR** sobre el mejor candidato. Mantener nodos fijos cuando sea posible para aislar el efecto. Estas son hipótesis extraídas del código; no se han implementado ni demostrado ganancias en esta revisión. No consumir nuevos envíos hasta obtener una variante concreta y comprobar su ejecución.
